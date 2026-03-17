import argparse
import os
from dataclasses import dataclass
from typing import Any, Dict, Optional

import torch
import torch.distributed as dist

import deep_ep
from utils import init_dist, per_token_cast_back

from sglang.srt.layers import deep_gemm_wrapper
from sglang.srt.layers.moe.moe_runner.base import MoeRunnerConfig
from sglang.srt.layers.moe.moe_runner.deep_gemm import (
    DeepGemmMoeQuantInfo,
    DeepGemmRunnerCore,
    DeepGemmRunnerInput,
    _pack_activation_scale_to_ue8m0,
    post_permute_deep_gemm_to_deepep_normal,
    post_permute_deep_gemm_to_deepep_hybrid_normal,
    pre_permute_deepep_normal_to_deep_gemm,
    pre_permute_deepep_hybrid_normal_to_deep_gemm,
)
from sglang.srt.layers.moe.token_dispatcher.deepep import (
    DeepEPNormalDispatchOutput,
    DeepEPHybridNormalDispatchOutput,
    _DeepEPDispatcherImplHybridNormal,
)
from sglang.srt.layers.quantization.fp8_utils import inverse_transform_scale_ue8m0

from test_bd_with_scatter import (
    build_shared_inputs,
    calc_max_abs_diff,
    canonicalize_rows_per_expert,
    quantize_for_hybrid_dispatch,
)


@dataclass
class FinalPathOutput:
    hidden_states: torch.Tensor
    probs: Optional[torch.Tensor]
    runner_hidden: torch.Tensor
    runner_scale: torch.Tensor
    m_indices: torch.Tensor
    tokens_per_expert: torch.Tensor
    native_hidden_states: Optional[torch.Tensor] = None


def debug_nan_summary(name: str, tensor: Optional[torch.Tensor]) -> None:
    if os.getenv("DEEP_GEMM_EQ_DEBUG_NAN", "0") != "1":
        return
    rank = dist.get_rank() if dist.is_initialized() else -1
    if tensor is None:
        print(f"[rank {rank}] {name}: none", flush=True)
        return
    num_nan = torch.isnan(tensor).sum().item() if tensor.is_floating_point() else 0
    num_inf = torch.isinf(tensor).sum().item() if tensor.is_floating_point() else 0
    print(
        f"[rank {rank}] {name}: shape={tuple(tensor.shape)} dtype={tensor.dtype} "
        f"num_nan={num_nan} num_inf={num_inf}",
        flush=True,
    )


def print_in_order(message: str) -> None:
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    for i in range(world_size):
        if i == rank:
            print(message, flush=True)
        dist.barrier()


def sample_topk_weights_like(topk_idx: torch.Tensor) -> torch.Tensor:
    topk_weights = torch.rand(
        topk_idx.shape,
        device=topk_idx.device,
        dtype=torch.float32,
    )
    return topk_weights / topk_weights.sum(dim=-1, keepdim=True)


def build_runner_config(args: argparse.Namespace, num_experts: int) -> MoeRunnerConfig:
    return MoeRunnerConfig(
        num_experts=num_experts,
        num_local_experts=args.num_local_experts,
        hidden_size=args.hidden_dim,
        intermediate_size_per_partition=args.intermediate_dim,
        top_k=args.num_topk,
        activation="silu",
        is_gated=True,
    )


def build_quant_info(
    num_local_experts: int,
    hidden_dim: int,
    intermediate_dim: int,
    weight_scale: float,
) -> DeepGemmMoeQuantInfo:
    block_size = 128
    gateup_out_dim = intermediate_dim * 2

    w13_weight = (torch.randn(
        (num_local_experts, gateup_out_dim, hidden_dim),
        device="cuda",
        dtype=torch.float32,
    ) * weight_scale).to(torch.float8_e4m3fn)
    w2_weight = (torch.randn(
        (num_local_experts, hidden_dim, intermediate_dim),
        device="cuda",
        dtype=torch.float32,
    ) * weight_scale).to(torch.float8_e4m3fn)

    w13_scale = torch.ones(
        (
            num_local_experts,
            (gateup_out_dim + block_size - 1) // block_size,
            (hidden_dim + block_size - 1) // block_size,
        ),
        device="cuda",
        dtype=torch.float32,
    )
    w2_scale = torch.ones(
        (
            num_local_experts,
            (hidden_dim + block_size - 1) // block_size,
            (intermediate_dim + block_size - 1) // block_size,
        ),
        device="cuda",
        dtype=torch.float32,
    )

    return DeepGemmMoeQuantInfo(
        w13_weight=w13_weight.contiguous(),
        w2_weight=w2_weight.contiguous(),
        use_fp8=True,
        w13_scale=w13_scale.contiguous(),
        w2_scale=w2_scale.contiguous(),
        block_shape=[block_size, block_size],
    )


def build_m_indices(
    tokens_per_expert: torch.Tensor,
    num_local_experts: int,
    device: torch.device,
) -> torch.Tensor:
    expert_ids = torch.arange(num_local_experts, device=device, dtype=torch.long)
    return torch.repeat_interleave(expert_ids, tokens_per_expert.to(torch.long)).to(
        torch.int32
    )


def build_combine_handle_from_permute_handle(handle: tuple) -> tuple:
    return (
        handle[0],
        handle[1],
        handle[2],
        handle[3],
        handle[4],
        handle[6],
        handle[7],
    )


def rebuild_output_index_from_row_id_map(
    recv_topk_idx: torch.Tensor,
    row_id_map: torch.Tensor,
    num_dispatched_tokens: int,
) -> torch.Tensor:
    assert recv_topk_idx.shape[0] == num_dispatched_tokens

    output_index = torch.full_like(recv_topk_idx, -1)
    row_id_map = row_id_map[:num_dispatched_tokens].to(torch.long)

    for topk_slot in range(recv_topk_idx.shape[1]):
        expert_ids = recv_topk_idx[:, topk_slot].to(torch.long)
        valid_mask = (expert_ids >= 0) & (expert_ids < row_id_map.shape[1])
        if not torch.any(valid_mask):
            continue
        token_indices = valid_mask.nonzero(as_tuple=True)[0]
        output_index[token_indices, topk_slot] = (
            row_id_map[token_indices, expert_ids[valid_mask]] - 1
        ).to(output_index.dtype)

    return output_index.contiguous()


def rebuild_hybrid_recv_topk_from_permute_output(
    dispatched_probs: Optional[torch.Tensor],
    local_expert_routing_map: torch.Tensor,
    row_id_map: torch.Tensor,
    router_topk: int,
    num_dispatched_tokens: int,
) -> tuple[torch.Tensor, Optional[torch.Tensor]]:
    local_expert_routing_map = local_expert_routing_map[
        :num_dispatched_tokens
    ].to(torch.bool)
    row_id_map = row_id_map[:num_dispatched_tokens].to(torch.long)

    recv_topk_idx = torch.full(
        (num_dispatched_tokens, router_topk),
        -1,
        device=local_expert_routing_map.device,
        dtype=torch.int64,
    )
    recv_topk_weights = None
    if dispatched_probs is not None:
        recv_topk_weights = torch.zeros(
            (num_dispatched_tokens, router_topk),
            device=dispatched_probs.device,
            dtype=dispatched_probs.dtype,
        )

    local_slot = torch.cumsum(local_expert_routing_map.to(torch.int32), dim=1) - 1
    valid_mask = (
        local_expert_routing_map
        & (row_id_map > 0)
        & (local_slot >= 0)
        & (local_slot < router_topk)
    )
    token_indices, expert_indices = valid_mask.nonzero(as_tuple=True)
    if token_indices.numel() == 0:
        return recv_topk_idx, recv_topk_weights

    slot_indices = local_slot[token_indices, expert_indices].to(torch.long)
    recv_topk_idx[token_indices, slot_indices] = expert_indices.to(torch.int64)

    if recv_topk_weights is not None:
        source_indices = row_id_map[token_indices, expert_indices] - 1
        recv_topk_weights[token_indices, slot_indices] = dispatched_probs[
            source_indices
        ]

    return recv_topk_idx, recv_topk_weights


def maybe_pack_activation_scale(x_scale: torch.Tensor, hidden_dim: int) -> torch.Tensor:
    if deep_gemm_wrapper.DEEPGEMM_SCALE_UE8M0 and x_scale.dtype != torch.int:
        return _pack_activation_scale_to_ue8m0(x_scale, hidden_dim)
    return x_scale


def run_expert_op(
    expert_op: str,
    runner_input: DeepGemmRunnerInput,
    runner: DeepGemmRunnerCore,
    quant_info: DeepGemmMoeQuantInfo,
    running_state: Dict[str, Any],
) -> torch.Tensor:
    if expert_op == "identity":
        hidden_states_scale = runner_input.hidden_states_scale
        if (
            deep_gemm_wrapper.DEEPGEMM_SCALE_UE8M0
            and hidden_states_scale.dtype == torch.int
        ):
            hidden_states_scale = inverse_transform_scale_ue8m0(
                hidden_states_scale,
                mn=runner_input.hidden_states.shape[0],
            )
        return per_token_cast_back(
            runner_input.hidden_states,
            hidden_states_scale,
        )
    if expert_op == "deep_gemm":
        return runner.run(runner_input, quant_info, running_state).hidden_states
    raise ValueError(f"Unsupported expert_op: {expert_op}")


def run_deepep_normal_path(
    buffer: deep_ep.Buffer,
    runner: DeepGemmRunnerCore,
    quant_info: DeepGemmMoeQuantInfo,
    runner_config: MoeRunnerConfig,
    expert_op: str,
    shared_hidden: torch.Tensor,
    topk_idx: torch.Tensor,
    topk_weights: torch.Tensor,
    num_experts: int,
    num_local_experts: int,
) -> FinalPathOutput:
    hidden_fp8, hidden_scale = quantize_for_hybrid_dispatch(shared_hidden)
    dispatch_config = deep_ep.Buffer.get_dispatch_config(buffer.group.size())
    combine_config = deep_ep.Buffer.get_combine_config(buffer.group.size())
    (
        num_tokens_per_rank,
        num_tokens_per_rdma_rank,
        num_tokens_per_expert,
        is_token_in_rank,
        _,
    ) = buffer.get_dispatch_layout(
        topk_idx,
        num_experts,
    )
    recv_x, recv_topk_idx, recv_topk_weights, tokens_per_expert, dispatch_handle, _ = (
        buffer.dispatch(
            x=(hidden_fp8, hidden_scale),
            topk_idx=topk_idx,
            topk_weights=topk_weights,
            num_tokens_per_rank=num_tokens_per_rank,
            num_tokens_per_rdma_rank=num_tokens_per_rdma_rank,
            is_token_in_rank=is_token_in_rank,
            num_tokens_per_expert=num_tokens_per_expert,
            expert_alignment=128,
            config=dispatch_config,
        )
    )
    recv_hidden, recv_scale = recv_x
    if deep_gemm_wrapper.DEEPGEMM_SCALE_UE8M0 and recv_scale.dtype != torch.int32:
        recv_scale = _pack_activation_scale_to_ue8m0(recv_scale, shared_hidden.shape[1])

    dispatch_output = DeepEPNormalDispatchOutput(
        hidden_states=recv_hidden,
        hidden_states_scale=recv_scale,
        topk_ids=recv_topk_idx,
        topk_weights=recv_topk_weights,
        num_recv_tokens_per_expert=tokens_per_expert,
    )
    running_state: Dict[str, Any] = {}
    runner_input = pre_permute_deepep_normal_to_deep_gemm(
        dispatch_output, quant_info, runner_config, running_state
    )
    runner_hidden = runner_input.hidden_states.detach().clone()
    runner_scale = runner_input.hidden_states_scale.detach().clone()
    runner_m_indices = runner_input.m_indices.detach().clone()
    runner_output_hidden = run_expert_op(
        expert_op=expert_op,
        runner_input=runner_input,
        runner=runner,
        quant_info=quant_info,
        running_state=running_state,
    )
    debug_nan_summary("normal.runner_output_hidden", runner_output_hidden)
    runner_output = type("RunnerOutputHolder", (), {"hidden_states": runner_output_hidden})()
    combine_input = post_permute_deep_gemm_to_deepep_normal(
        runner_output, quant_info, runner_config, running_state
    )
    debug_nan_summary("normal.combine_input.hidden_states", combine_input.hidden_states)
    combined_hidden, combined_probs, _ = buffer.combine(
        x=combine_input.hidden_states,
        handle=dispatch_handle,
        config=combine_config,
    )
    debug_nan_summary("normal.combined_hidden", combined_hidden)

    return FinalPathOutput(
        hidden_states=combined_hidden,
        probs=combined_probs,
        runner_hidden=runner_hidden,
        runner_scale=runner_scale,
        m_indices=runner_m_indices,
        tokens_per_expert=torch.bincount(
            runner_m_indices, minlength=num_local_experts
        ).to(torch.int32),
    )


def run_dispatch_with_permute_path(
    buffer: deep_ep.HybridEPBuffer,
    runner: DeepGemmRunnerCore,
    quant_info: DeepGemmMoeQuantInfo,
    expert_op: str,
    shared_hidden: torch.Tensor,
    topk_idx: torch.Tensor,
    topk_weights: torch.Tensor,
    num_experts: int,
    num_local_experts: int,
    pad_multiple: int,
) -> FinalPathOutput:
    from sglang.srt.layers.moe.ep_moe.kernels import ep_gather

    hidden_fp8, hidden_scale = quantize_for_hybrid_dispatch(shared_hidden)
    routing_map, probs = (
        deep_ep_cpp_build_hybrid_routing_map_and_probs(
            topk_idx, topk_weights, num_experts
        )
    )
    (
        dispatched_hidden,
        dispatched_probs,
        dispatched_scale,
        tokens_per_expert,
        handle,
    ) = buffer.dispatch_with_permute(
        hidden=hidden_fp8,
        scaling_factor=hidden_scale,
        routing_map=routing_map,
        probs=probs,
        num_of_experts_per_rank=num_local_experts,
        num_of_experts=num_experts,
        use_fp8=True,
        pad_multiple=pad_multiple,
    )

    tokens_per_expert = tokens_per_expert.to(
        device=dispatched_hidden.device, dtype=torch.int32
    )
    num_dispatched_tokens = int(handle[3].cpu().item())
    local_expert_routing_map = handle[4]
    recv_topk_idx, recv_topk_weights = rebuild_hybrid_recv_topk_from_permute_output(
        dispatched_probs,
        local_expert_routing_map,
        handle[5],
        topk_idx.shape[1],
        num_dispatched_tokens,
    )
    dispatched_scale = maybe_pack_activation_scale(dispatched_scale, shared_hidden.shape[1])
    m_indices = build_m_indices(
        tokens_per_expert,
        num_local_experts,
        dispatched_hidden.device,
    )
    running_state = {
        "all_tokens": dispatched_hidden.shape[0],
        "hidden_states_shape": shared_hidden.shape,
        "hidden_states_device": shared_hidden.device,
        "hidden_states_dtype": shared_hidden.dtype,
    }
    runner_input = DeepGemmRunnerInput(
        hidden_states=dispatched_hidden,
        hidden_states_scale=dispatched_scale,
        use_masked_gemm=False,
        m_indices=m_indices,
    )
    runner_hidden = runner_input.hidden_states.detach().clone()
    runner_scale = runner_input.hidden_states_scale.detach().clone()
    runner_m_indices = runner_input.m_indices.detach().clone()
    runner_output_hidden = run_expert_op(
        expert_op=expert_op,
        runner_input=runner_input,
        runner=runner,
        quant_info=quant_info,
        running_state=running_state,
    )
    debug_nan_summary("permute.runner_output_hidden", runner_output_hidden)
    native_combined_hidden, _ = buffer.combine_with_unpermute(
        hidden=runner_output_hidden,
        probs=dispatched_probs,
        handle=handle,
        pad_multiple=pad_multiple,
        apply_probs_to_hidden=dispatched_probs is not None,
    )
    debug_nan_summary("permute.native_combined_hidden", native_combined_hidden)

    output_index = rebuild_output_index_from_row_id_map(
        recv_topk_idx,
        handle[5],
        num_dispatched_tokens,
    )
    gathered_hidden = torch.empty(
        (num_dispatched_tokens, runner_output_hidden.shape[1]),
        device=runner_output_hidden.device,
        dtype=torch.bfloat16,
    )
    ep_gather(
        runner_output_hidden,
        recv_topk_idx,
        recv_topk_weights,
        output_index,
        gathered_hidden,
    )
    combine_handle = build_combine_handle_from_permute_handle(handle)
    combined_hidden, combined_probs = buffer.combine(
        hidden=gathered_hidden,
        probs=None,
        handle=combine_handle,
    )
    debug_nan_summary("permute.gathered_hidden", gathered_hidden)
    debug_nan_summary("permute.combined_hidden", combined_hidden)

    return FinalPathOutput(
        hidden_states=combined_hidden,
        probs=combined_probs,
        runner_hidden=runner_hidden,
        runner_scale=runner_scale,
        m_indices=runner_m_indices,
        tokens_per_expert=torch.bincount(
            runner_m_indices, minlength=num_local_experts
        ).to(torch.int32),
        native_hidden_states=native_combined_hidden,
    )


def compare_outputs(
    scatter_path: FinalPathOutput,
    with_permute_path: FinalPathOutput,
) -> Dict[str, Any]:
    scatter_runner_hidden_canonical, scatter_runner_scale_canonical, _ = (
        canonicalize_rows_per_expert(
            scatter_path.runner_hidden,
            scatter_path.runner_scale,
            scatter_path.runner_hidden.to(torch.bfloat16),
            scatter_path.tokens_per_expert,
        )
    )
    with_permute_runner_hidden_canonical, with_permute_runner_scale_canonical, _ = (
        canonicalize_rows_per_expert(
            with_permute_path.runner_hidden,
            with_permute_path.runner_scale,
            with_permute_path.runner_hidden.to(torch.bfloat16),
            with_permute_path.tokens_per_expert,
        )
    )

    probs_match = True
    probs_max_abs_diff = 0.0
    if scatter_path.probs is None or with_permute_path.probs is None:
        probs_match = scatter_path.probs is None and with_permute_path.probs is None
    else:
        probs_match = torch.equal(scatter_path.probs, with_permute_path.probs)
        probs_max_abs_diff = calc_max_abs_diff(
            scatter_path.probs,
            with_permute_path.probs,
        )

    return {
        "runner_hidden_raw_exact_match": torch.equal(
            scatter_path.runner_hidden, with_permute_path.runner_hidden
        ),
        "runner_scale_raw_exact_match": torch.equal(
            scatter_path.runner_scale, with_permute_path.runner_scale
        ),
        "runner_hidden_canonical_exact_match": torch.equal(
            scatter_runner_hidden_canonical,
            with_permute_runner_hidden_canonical,
        ),
        "runner_scale_canonical_exact_match": torch.equal(
            scatter_runner_scale_canonical,
            with_permute_runner_scale_canonical,
        ),
        "runner_m_indices_match": torch.equal(
            scatter_path.m_indices, with_permute_path.m_indices
        ),
        "runner_tokens_per_expert_match": torch.equal(
            scatter_path.tokens_per_expert,
            with_permute_path.tokens_per_expert,
        ),
        "final_hidden_exact_match": torch.equal(
            scatter_path.hidden_states, with_permute_path.hidden_states
        ),
        "final_hidden_max_abs_diff": calc_max_abs_diff(
            scatter_path.hidden_states,
            with_permute_path.hidden_states,
        ),
        "native_final_hidden_exact_match": (
            False
            if with_permute_path.native_hidden_states is None
            else torch.equal(
                scatter_path.hidden_states,
                with_permute_path.native_hidden_states,
            )
        ),
        "native_final_hidden_max_abs_diff": (
            float("inf")
            if with_permute_path.native_hidden_states is None
            else calc_max_abs_diff(
                scatter_path.hidden_states,
                with_permute_path.native_hidden_states,
            )
        ),
        "final_probs_exact_match": probs_match,
        "final_probs_max_abs_diff": probs_max_abs_diff,
    }


def deep_ep_cpp_build_hybrid_routing_map_and_probs(
    topk_idx: torch.Tensor,
    topk_weights: torch.Tensor,
    num_experts: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    from sglang.srt.layers.moe.token_dispatcher.deepep import (
        _DeepEPDispatcherImplHybridNormal,
    )

    return _DeepEPDispatcherImplHybridNormal._build_hybrid_routing_map_and_probs(
        topk_idx,
        topk_weights,
        num_experts,
    )


def run_demo(local_rank: int, num_local_ranks: int, args: argparse.Namespace) -> None:
    rank, _, group = init_dist(local_rank, num_local_ranks)

    torch.manual_seed(args.seed + rank)
    torch.cuda.manual_seed(args.seed + rank)

    shared_inputs = build_shared_inputs(
        num_tokens=args.num_tokens,
        hidden_dim=args.hidden_dim,
        num_topk=args.num_topk,
        num_local_experts=args.num_local_experts,
        group=group,
    )
    if args.random_topk_weights:
        shared_inputs.topk_weights.copy_(
            sample_topk_weights_like(shared_inputs.topk_idx)
        )

    runner_config = build_runner_config(args, shared_inputs.num_experts)
    quant_info = build_quant_info(
        num_local_experts=args.num_local_experts,
        hidden_dim=args.hidden_dim,
        intermediate_dim=args.intermediate_dim,
        weight_scale=args.weight_scale,
    )
    runner = DeepGemmRunnerCore(runner_config)

    with_permute_buffer = deep_ep.HybridEPBuffer(
        group=group,
        hidden_dim=args.hidden_dim,
        max_num_of_tokens_per_rank=args.num_tokens,
        num_local_experts=args.num_local_experts,
        use_fp8=True,
    )

    deepep_buffer = deep_ep.Buffer(
        group,
        int(2e9),
        0,
        explicitly_destroy=True,
    )
    normal_path = run_deepep_normal_path(
        deepep_buffer,
        runner,
        quant_info,
        runner_config,
        args.expert_op,
        shared_inputs.hidden,
        shared_inputs.topk_idx,
        shared_inputs.topk_weights,
        shared_inputs.num_experts,
        args.num_local_experts,
    )
    with_permute_path = run_dispatch_with_permute_path(
        with_permute_buffer,
        runner,
        quant_info,
        args.expert_op,
        shared_inputs.hidden,
        shared_inputs.topk_idx,
        shared_inputs.topk_weights,
        shared_inputs.num_experts,
        args.num_local_experts,
        args.pad_multiple,
    )

    compare_payload = compare_outputs(normal_path, with_permute_path)
    print_in_order(
        f"[rank {rank}] compare={compare_payload} "
        f"final_hidden_shape={tuple(normal_path.hidden_states.shape)}"
    )

    if not args.allow_mismatch:
        assert compare_payload["runner_hidden_canonical_exact_match"]
        assert compare_payload["runner_scale_canonical_exact_match"]
        assert compare_payload["runner_m_indices_match"]
        assert compare_payload["runner_tokens_per_expert_match"]
        assert compare_payload["final_hidden_exact_match"]
        assert compare_payload["final_hidden_max_abs_diff"] == 0.0

    dist.barrier()
    deepep_buffer.destroy()
    dist.destroy_process_group()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description=(
            "End-to-end DeepGEMM equivalence check between "
            "dispatch+ep_scatter and dispatch_with_permute."
        )
    )
    parser.add_argument("--num-processes", type=int, default=4)
    parser.add_argument("--num-tokens", type=int, default=4096)
    parser.add_argument("--hidden-dim", type=int, default=7168)
    parser.add_argument("--intermediate-dim", type=int, default=2048)
    parser.add_argument("--num-topk", type=int, default=8)
    parser.add_argument("--num-local-experts", type=int, default=8)
    parser.add_argument("--pad-multiple", type=int, default=128)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--weight-scale", type=float, default=0.01)
    parser.add_argument(
        "--expert-op",
        choices=("deep_gemm", "identity"),
        default="deep_gemm",
    )
    parser.add_argument("--random-topk-weights", action="store_true")
    parser.add_argument("--allow-mismatch", action="store_true")
    args = parser.parse_args()

    torch.multiprocessing.spawn(
        run_demo,
        args=(args.num_processes, args),
        nprocs=args.num_processes,
    )
