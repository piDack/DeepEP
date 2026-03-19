# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved
import argparse
import itertools
import time
import torch
import torch.distributed as dist
import os
import deep_ep
from contextlib import contextmanager

from utils import TorchRef, bench, bench_kineto, init_dist, count_rdma_send_from_routing_map

HIDDEN_DIM = int(os.environ.get("HIDDEN_DIM", 7168))
MAX_NUM_OF_TOKENS_PER_RANK = int(os.environ.get("MAX_NUM_OF_TOKENS_PER_RANK", 4096))
# NUM_TOKENS_PER_RANK should equal or less than MAX_NUM_OF_TOKENS_PER_RANK
NUM_TOKENS_PER_RANK = int(os.environ.get("NUM_TOKENS_PER_RANK", 4096))
NUM_LOCAL_EXPERTS = int(os.environ.get("NUM_LOCAL_EXPERTS", 8))
TOPK = int(os.environ.get("TOPK", 8))
PAD_MULTIPLE = int(os.environ.get("PAD_MULTIPLE", 32))
ITERATIONS = int(os.environ.get("ITERATIONS", 100))
SEED = int(os.environ.get("SEED", 42))
USE_MNNVL = os.environ.get("USE_MNNVL", "0").strip().lower() in {"1", "true", "t", "yes", "y", "on"}
torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
# Will be set after the process group is initialized
NUM_OF_RANKS_PER_NODE = None
NUM_OF_NODES = None
NUM_OF_EXPERTS = None

SWEEP_ENV_KEYS = (
    "NUM_OF_STAGES_DISPATCH_API",
    "NUM_OF_IN_FLIGHT_S2G_DISPATCH_API",
    "NUM_OF_TOKENS_PER_CHUNK_DISPATCH_API",
    "NUM_OF_STAGES_G2S_COMBINE_API",
    "NUM_OF_STAGES_S2G_COMBINE_API",
    "NUM_OF_TOKENS_PER_CHUNK_COMBINE_API",
    "NUM_OF_TOKENS_PER_GROUP_COMBINE_API",
    "NUM_OF_ADDITIONAL_IN_FLIGHT_S2G_COMBINE_API",
)

def print_in_order(msg: str):
    """Print message in order by rank to avoid interleaved output"""
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    for i in range(world_size):
        if i == rank:
            print(msg, flush=True)
        dist.barrier()


def dtype_name(use_fp8: bool) -> str:
    return "FP8" if use_fp8 else "BF16"


def parse_csv_ints(raw: str) -> list[int]:
    return [int(item.strip()) for item in raw.split(",") if item.strip()]


def parse_dtype_list(raw: str) -> list[bool]:
    values = []
    for item in raw.split(","):
        key = item.strip().lower()
        if not key:
            continue
        if key == "bf16":
            values.append(False)
        elif key == "fp8":
            values.append(True)
        else:
            raise ValueError(f"Unsupported dtype '{item}', expected bf16/fp8")
    if not values:
        raise ValueError("At least one dtype must be provided")
    return values


@contextmanager
def temporary_env(overrides: dict[str, int]):
    previous = {}
    try:
        for key, value in overrides.items():
            previous[key] = os.environ.get(key)
            os.environ[key] = str(value)
        yield
    finally:
        for key, value in previous.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value


def build_sweep_cases(args: argparse.Namespace) -> list[dict[str, int]]:
    sweep_axes = {
        "num_sms_dispatch_api": parse_csv_ints(args.sweep_sm_dispatch),
        "num_sms_combine_api": parse_csv_ints(args.sweep_sm_combine),
        "NUM_OF_STAGES_DISPATCH_API": parse_csv_ints(args.sweep_dispatch_stages),
        "NUM_OF_IN_FLIGHT_S2G_DISPATCH_API": parse_csv_ints(args.sweep_dispatch_inflight),
        "NUM_OF_TOKENS_PER_CHUNK_DISPATCH_API": parse_csv_ints(args.sweep_dispatch_chunks),
        "NUM_OF_STAGES_G2S_COMBINE_API": parse_csv_ints(args.sweep_combine_g2s_stages),
        "NUM_OF_STAGES_S2G_COMBINE_API": parse_csv_ints(args.sweep_combine_s2g_stages),
        "NUM_OF_TOKENS_PER_CHUNK_COMBINE_API": parse_csv_ints(args.sweep_combine_chunks),
        "NUM_OF_TOKENS_PER_GROUP_COMBINE_API": parse_csv_ints(args.sweep_combine_groups),
        "NUM_OF_ADDITIONAL_IN_FLIGHT_S2G_COMBINE_API": parse_csv_ints(args.sweep_combine_inflight),
    }
    keys = list(sweep_axes.keys())
    values = [sweep_axes[key] for key in keys]
    return [dict(zip(keys, combo)) for combo in itertools.product(*values)]


def format_sweep_case(case: dict[str, int]) -> str:
    parts = [
        f"sms(d={case['num_sms_dispatch_api']},c={case['num_sms_combine_api']})",
        f"dispatch(stages={case['NUM_OF_STAGES_DISPATCH_API']},inflight={case['NUM_OF_IN_FLIGHT_S2G_DISPATCH_API']},chunk={case['NUM_OF_TOKENS_PER_CHUNK_DISPATCH_API']})",
        f"combine(g2s={case['NUM_OF_STAGES_G2S_COMBINE_API']},s2g={case['NUM_OF_STAGES_S2G_COMBINE_API']},chunk={case['NUM_OF_TOKENS_PER_CHUNK_COMBINE_API']},group={case['NUM_OF_TOKENS_PER_GROUP_COMBINE_API']},inflight={case['NUM_OF_ADDITIONAL_IN_FLIGHT_S2G_COMBINE_API']})",
    ]
    return " | ".join(parts)


def summarize_score(result: dict[str, float], sort_by: str) -> float:
    if sort_by == "dispatch":
        return result["dispatch_torch_nvl_gbps"]
    if sort_by == "combine":
        return result["combine_torch_nvl_gbps"]
    return result["dispatch_torch_nvl_gbps"] + result["combine_torch_nvl_gbps"]

def bitwise_equal(a: torch.Tensor, b: torch.Tensor) -> bool:
    if a.dtype != b.dtype or a.shape != b.shape or a.device != b.device:
        return False
    a_bytes = a.contiguous().view(torch.uint8)
    b_bytes = b.contiguous().view(torch.uint8)
    return torch.equal(a_bytes, b_bytes)

def init_tensor(
    hidden_dim: int,
    seq_len: int,
    topk: int,
    num_of_experts: int,
    use_fp8: bool = False,
):
    if use_fp8:
        hidden = torch.randint(
            low=0,
            high=256,
            size=(seq_len, hidden_dim),
            device="cuda",
            dtype=torch.uint8,
        )
    else:
        hidden = torch.randn(seq_len, hidden_dim, device="cuda", dtype=torch.bfloat16)
    probs = torch.zeros(seq_len, num_of_experts, device="cuda", dtype=torch.float32)
    topk_idx = torch.zeros(seq_len, topk, device="cuda", dtype=torch.int64)
    topk_weights = torch.zeros(seq_len, topk, device="cuda", dtype=torch.float32)
    scaling_factor = torch.randn(
        seq_len, hidden_dim // 128, device="cuda", dtype=torch.float32
    )

    routing_map = torch.zeros(seq_len, num_of_experts, device="cuda", dtype=torch.bool)

    for i in range(seq_len):
        # Force balanced routing for testing
        # selected_experts = torch.tensor([
        #     ((i * topk) % num_of_experts + val) % num_of_experts for val in range(topk)
        # ], device="cuda")
        selected_experts = torch.randperm(num_of_experts, device="cuda")[:topk]
        topk_idx[i, :] = selected_experts.to(torch.int64)
        topk_weights[i, :] = torch.ones(topk, device="cuda", dtype=torch.float32)
        routing_map[i, selected_experts] = True
        probs[i, selected_experts] = topk_weights[i, :]

    return hidden, probs, scaling_factor, routing_map, topk_idx, topk_weights


def sample_normalized_topk_probs(
    topk_idx: torch.Tensor,
    num_of_experts: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    topk_weights = torch.rand(
        topk_idx.shape,
        device=topk_idx.device,
        dtype=torch.float32,
    )
    topk_weights = topk_weights / topk_weights.sum(dim=-1, keepdim=True)
    probs = torch.zeros(
        topk_idx.shape[0],
        num_of_experts,
        device=topk_idx.device,
        dtype=torch.float32,
    )
    probs.scatter_(1, topk_idx, topk_weights)
    return probs, topk_weights


def test_hybrid_ep_correctness(buffer: deep_ep.HybridEPBuffer, ref: TorchRef, use_fp8: bool):
    hidden, probs, scaling_factor, routing_map, topk_idx, topk_weights  = init_tensor(
        hidden_dim=HIDDEN_DIM,
        seq_len=NUM_TOKENS_PER_RANK,
        topk=TOPK,
        num_of_experts=NUM_OF_EXPERTS,
        use_fp8=use_fp8,
    )

    # Dispatch correctness check
    for with_probs in [True, False]:
        # The check for the dispatch
        dispatched_hidden_ref, dispatched_probs_ref, dispatched_scaling_factor_ref = (
            ref.dispatch(
                hidden, routing_map, probs if with_probs else None, scaling_factor
            )
        )
        (
            dispatched_hidden,
            dispatched_probs,
            dispatched_scaling_factor,
            handle,
        ) = buffer.dispatch(
            hidden=hidden, scaling_factor=scaling_factor, topk_idx=topk_idx, topk_weights=topk_weights if with_probs else None, num_of_experts=NUM_OF_EXPERTS,
        )

        assert bitwise_equal(dispatched_hidden_ref, dispatched_hidden)
        if dispatched_probs is not None and dispatched_probs_ref is not None:
            start, end = ref._local_expert_range_per_node()
            masked_probs = torch.zeros_like(dispatched_probs)
            masked_probs[:, start:end] = dispatched_probs[:, start:end]
            assert bitwise_equal(dispatched_probs_ref, dispatched_probs[:, start:end])
            dispatched_probs = masked_probs
        if (
            dispatched_scaling_factor is not None
            and dispatched_scaling_factor_ref is not None
        ):
            assert bitwise_equal(
                dispatched_scaling_factor_ref, dispatched_scaling_factor
            )

        _, _, _, num_dispatched_tokens, local_expert_routing_map, _, _ = handle
        num_dispatched_tokens = num_dispatched_tokens.cpu()
        local_expert_routing_map = local_expert_routing_map[
            : num_dispatched_tokens.item()
        ]
        # Simulate the permute and expert and unpermute. The expert is identity op
        copy_times = local_expert_routing_map.sum(dim=1)
        dispatched_hidden = dispatched_hidden.to(torch.bfloat16)  
        # The combine only support bf16
        hidden_to_combine = dispatched_hidden * copy_times.unsqueeze(1)
        probs_to_combine = dispatched_probs

        # The check for the combine
        combined_hidden, combined_probs = buffer.combine(
            hidden_to_combine, probs_to_combine, handle
        )

        # The reconstucted value should be TOPK times larger than the input hidden
        combined_hidden = combined_hidden / TOPK

        assert torch.allclose(combined_hidden, hidden.to(torch.bfloat16), atol=2e-5, rtol=1e-2)
        if combined_probs is not None and probs is not None:
            assert bitwise_equal(combined_probs, probs)

    # Dispatch with permute correctness check
    for with_probs in [True, False]:
        # The check for the dispatch
        (
            dispatched_hidden,
            dispatched_probs,
            dispatched_scaling_factor,
            tokens_per_expert,
            handle,
        ) = buffer.dispatch_with_permute(
            hidden=hidden,
            routing_map=routing_map,
            probs=probs if with_probs else None,
            scaling_factor=scaling_factor,
            pad_multiple=PAD_MULTIPLE,
        )
        _, _, _, num_dispatched_tokens_tensor, local_expert_routing_map, _, _, _, _ = (
            handle
        )
        num_dispatched_tokens_tensor = num_dispatched_tokens_tensor.cpu()
        local_expert_routing_map = local_expert_routing_map[
            : num_dispatched_tokens_tensor.item()
        ]
        # The out_token_num of permutation is the sum of the tokens_per_expert
        out_token_num = tokens_per_expert.sum().item()
        (
            dispatched_hidden_ref,
            dispatched_probs_ref,
            dispatched_scaling_factor_ref,
        ) = ref.dispatch(
            hidden,
            routing_map,
            probs if with_probs else None,
            scaling_factor,
            local_expert_routing_map=local_expert_routing_map,
            out_token_num=out_token_num,
            pad_multiple=PAD_MULTIPLE,
            enable_permute=True,
        )

        assert bitwise_equal(dispatched_hidden_ref, dispatched_hidden)
        if dispatched_probs is not None and dispatched_probs_ref is not None:
            assert bitwise_equal(dispatched_probs_ref, dispatched_probs)
        if (
            dispatched_scaling_factor is not None
            and dispatched_scaling_factor_ref is not None
        ):
            assert bitwise_equal(
                dispatched_scaling_factor_ref, dispatched_scaling_factor
            )

        # The combine only support bf16
        dispatched_hidden = dispatched_hidden.to(torch.bfloat16)  
        hidden_to_combine = dispatched_hidden
        probs_to_combine = dispatched_probs
 
        # The check for the combine
        combined_hidden, combined_probs = buffer.combine_with_unpermute(
            hidden=hidden_to_combine,
            probs=probs_to_combine,
            handle=handle,
            pad_multiple=PAD_MULTIPLE,
        )

        # The reconstucted value should be TOPK times larger than the input hidden
        combined_hidden = combined_hidden / TOPK

        assert torch.allclose(
            combined_hidden, hidden.to(torch.bfloat16), atol=2e-5, rtol=1e-2
        )
        if combined_probs is not None and probs is not None:
            assert bitwise_equal(combined_probs, probs)

        if with_probs:
            weighted_probs, _ = sample_normalized_topk_probs(topk_idx, NUM_OF_EXPERTS)
            (
                weighted_hidden,
                weighted_dispatched_probs,
                weighted_scaling_factor,
                _,
                weighted_handle,
            ) = buffer.dispatch_with_permute(
                hidden=hidden,
                routing_map=routing_map,
                probs=weighted_probs,
                scaling_factor=scaling_factor,
                pad_multiple=PAD_MULTIPLE,
            )
            weighted_hidden = weighted_hidden.to(torch.bfloat16)
            weighted_combined_hidden, weighted_combined_probs = (
                buffer.combine_with_unpermute(
                    hidden=weighted_hidden,
                    probs=weighted_dispatched_probs,
                    handle=weighted_handle,
                    pad_multiple=PAD_MULTIPLE,
                    apply_probs_to_hidden=True,
                )
            )

            assert torch.allclose(
                weighted_combined_hidden,
                hidden.to(torch.bfloat16),
                atol=2e-5,
                rtol=1e-2,
            )
            assert bitwise_equal(weighted_combined_probs, weighted_probs)

    print_in_order(f'[rank {dist.get_rank()}] Correctness check passed ({"FP8" if hidden.dtype == torch.uint8 else "BF16"})')


def test_hybrid_ep_benchmark(buffer: deep_ep.HybridEPBuffer, group: dist.ProcessGroup, use_fp8: bool, nsys_profile: bool):
    hidden, probs, scaling_factor, routing_map, topk_idx, topk_weights = init_tensor(
        hidden_dim=HIDDEN_DIM,
        seq_len=NUM_TOKENS_PER_RANK,
        topk=TOPK,
        num_of_experts=NUM_OF_EXPERTS,
        use_fp8=use_fp8,
    )

    # warmup
    for _ in range(10):
        dispatched_hidden, dispatched_probs, _, handle = (
            buffer.dispatch(hidden=hidden, scaling_factor=scaling_factor, topk_idx=topk_idx, topk_weights=topk_weights, num_of_experts=NUM_OF_EXPERTS)
        )
        # The combine only support bf16
        dispatched_hidden_bf16 = dispatched_hidden.to(torch.bfloat16)
        dispatched_probs = None
        _, _ = buffer.combine(dispatched_hidden_bf16, dispatched_probs, handle)

    rank = dist.get_rank()
    fp8_factor = (1 + 4 / 128) / 2
    dispatch_bf16_nvl_recv_bytes = dispatched_hidden.numel() * 2
    combine_bf16_nvl_send_bytes = dispatch_bf16_nvl_recv_bytes
    if NUM_OF_NODES > 1:
        local_node_id = rank // NUM_OF_RANKS_PER_NODE
        num_rdma_send = count_rdma_send_from_routing_map(routing_map, local_node_id, NUM_OF_NODES)
        dispatch_bf16_rdma_send_bytes = num_rdma_send * HIDDEN_DIM * 2
        combine_bf16_rdma_recv_bytes = dispatch_bf16_rdma_send_bytes

    '''
    Benchmark of the dispatch and combine torch API without permute
    '''

    dispatched_hidden, dispatched_probs, _, handle= (
        buffer.dispatch(hidden=hidden, scaling_factor=scaling_factor, topk_idx=topk_idx, topk_weights=topk_weights, num_of_experts=NUM_OF_EXPERTS)
    )
    dispatched_hidden_bf16 = dispatched_hidden.to(torch.bfloat16)

    dispatch_args = {'hidden': hidden, 'scaling_factor': scaling_factor, 'topk_idx': topk_idx, 'topk_weights': topk_weights, 'num_of_experts': NUM_OF_EXPERTS, 'handle': handle}
    t = bench(lambda: buffer.dispatch(**dispatch_args))[0]
    nvl_recv_bytes = (dispatch_bf16_nvl_recv_bytes * fp8_factor) if hidden.dtype == torch.uint8 else dispatch_bf16_nvl_recv_bytes
    if NUM_OF_NODES > 1:
        rdma_send_bytes = dispatch_bf16_rdma_send_bytes * fp8_factor if hidden.dtype == torch.uint8 else dispatch_bf16_rdma_send_bytes
    print_in_order(f'[rank {rank}] HybridEP dispatch torch API ({"FP8" if hidden.dtype == torch.uint8 else "BF16"}): '
            f'{nvl_recv_bytes / 1e9 / t:.2f} GB/s (NVL), t: {t * 1e6:.2f} us, nvl_recv_bytes: {nvl_recv_bytes / 1e6:.2f} MB')
    if NUM_OF_NODES > 1:
        print_in_order(f'[rank {rank}] HybridEP dispatch torch API ({"FP8" if hidden.dtype == torch.uint8 else "BF16"}): '
                f'{rdma_send_bytes / 1e9 / t:.2f} GB/s (IB), t: {t * 1e6:.2f} us, rdma_send_bytes: {rdma_send_bytes / 1e6:.2f} MB')

    combine_args = {'hidden': dispatched_hidden_bf16, 'probs': dispatched_probs, 'handle': handle}
    t = bench(lambda: buffer.combine(**combine_args))[0]
    print_in_order(f'[rank {rank}] HybridEP combine torch API: '
            f'{combine_bf16_nvl_send_bytes / 1e9 / t:.2f} GB/s (NVL), t: {t * 1e6:.2f} us, combine_send_bytes: {combine_bf16_nvl_send_bytes / 1e6:.2f} MB')
    if NUM_OF_NODES > 1:
        print_in_order(f'[rank {rank}] HybridEP combine torch API: '
                    f'{combine_bf16_rdma_recv_bytes / 1e9 / t:.2f} GB/s (IB), t: {t * 1e6:.2f} us, rdma_recv_bytes: {combine_bf16_rdma_recv_bytes / 1e6:.2f} MB')

    '''
    Benchmark of the dispatch and combine with permute extension
    '''
    dispatched_hidden_with_permute, dispatched_probs_with_permute, _, tokens_per_expert, handle_with_permute= (
        buffer.dispatch_with_permute(hidden=hidden, scaling_factor=scaling_factor, routing_map=routing_map, probs=probs, pad_multiple=PAD_MULTIPLE)
    )
    num_permuted_tokens = tokens_per_expert.sum().item()
    dispatched_hidden_bf16_with_permute = dispatched_hidden_with_permute.to(torch.bfloat16)

    dispatch_with_permute_args = {'hidden': hidden, 'scaling_factor': scaling_factor, 'routing_map': routing_map, 'probs': probs, 'pad_multiple': PAD_MULTIPLE, 'handle': handle_with_permute, 'num_permuted_tokens': num_permuted_tokens}
    t = bench(lambda: buffer.dispatch_with_permute(**dispatch_with_permute_args))[0]
    nvl_recv_bytes = (dispatch_bf16_nvl_recv_bytes * fp8_factor) if hidden.dtype == torch.uint8 else dispatch_bf16_nvl_recv_bytes
    print_in_order(f'[rank {rank}] HybridEP dispatch+permute torch API ({"FP8" if hidden.dtype == torch.uint8 else "BF16"}): '
            f'{nvl_recv_bytes / 1e9 / t:.2f} GB/s (NVL), t: {t * 1e6:.2f} us, nvl_recv_bytes: {nvl_recv_bytes / 1e6:.2f} MB')
    if NUM_OF_NODES > 1:
        print_in_order(f'[rank {rank}] HybridEP dispatch+permute torch API ({"FP8" if hidden.dtype == torch.uint8 else "BF16"}): '
                f'{rdma_send_bytes / 1e9 / t:.2f} GB/s (IB), t: {t * 1e6:.2f} us, rdma_send_bytes: {rdma_send_bytes / 1e6:.2f} MB')

    combine_with_unpermute_args = {'hidden': dispatched_hidden_bf16_with_permute, 'probs': dispatched_probs_with_permute, 'handle': handle_with_permute, 'pad_multiple': PAD_MULTIPLE}
    t = bench(lambda: buffer.combine_with_unpermute(**combine_with_unpermute_args))[0]
    print_in_order(f'[rank {rank}] HybridEP combine+unpermute torch API: '
            f'{combine_bf16_nvl_send_bytes / 1e9 / t:.2f} GB/s (NVL), t: {t * 1e6:.2f} us, combine_send_bytes: {combine_bf16_nvl_send_bytes / 1e6:.2f} MB')
    if NUM_OF_NODES > 1:
        print_in_order(f'[rank {rank}] HybridEP combine+unpermute torch API: '
                f'{combine_bf16_rdma_recv_bytes / 1e9 / t:.2f} GB/s (IB), t: {t * 1e6:.2f} us, rdma_recv_bytes: {combine_bf16_rdma_recv_bytes / 1e6:.2f} MB')

    if not nsys_profile:
        # noinspection PyShadowingNames
        def test_func():
            dispatched_hidden, dispatched_probs, _, handle = (
                buffer.dispatch(hidden=hidden, scaling_factor=scaling_factor, topk_idx=topk_idx, topk_weights=topk_weights, num_of_experts=NUM_OF_EXPERTS)
            )
            # The combine only support bf16
            dispatched_hidden_bf16 = dispatched_hidden.to(torch.bfloat16)
            dispatched_probs = None
            _, _ = buffer.combine(dispatched_hidden_bf16, dispatched_probs, handle)

        group.barrier()
        dispatch_t, combine_t = bench_kineto(test_func,
                                             kernel_names=('dispatch_kernel', 'combine_kernel'), barrier_comm_profiling=True,
                                             suppress_kineto_output=True)
        print_in_order(f'[rank {rank}] HybridEP dispatch kernel(NVL) ({"FP8" if hidden.dtype == torch.uint8 else "BF16"}): {nvl_recv_bytes / 1e9 / dispatch_t:.2f} GB/s, avg_t={dispatch_t * 1e6:.2f} us | '
              f'HybridEP combine kernel(NVL): {combine_bf16_nvl_send_bytes / 1e9 / combine_t:.2f} GB/s, avg_t={combine_t * 1e6:.2f} us')
        if NUM_OF_NODES > 1:
            print_in_order(f'[rank {rank}] HybridEP dispatch kernel(IB) ({"FP8" if hidden.dtype == torch.uint8 else "BF16"}): {rdma_send_bytes / 1e9 / dispatch_t:.2f} GB/s, avg_t={dispatch_t * 1e6:.2f} us | '
                  f'HybridEP combine kernel(IB): {combine_bf16_rdma_recv_bytes / 1e9 / combine_t:.2f} GB/s, avg_t={combine_t * 1e6:.2f} us')
    else:
        dtype_label = dtype_name(hidden.dtype == torch.uint8)
        if torch.distributed.get_rank() == 0:
            torch.cuda.profiler.start()
        with torch.cuda.nvtx.range(f"hybrid-ep dispatch ({dtype_label})"):
            if rank == 0:
                print(f"profile hybrid-ep dispatch ({dtype_label})", flush=True)
            dispatch_args = {'hidden': hidden, 'scaling_factor': scaling_factor, 'topk_idx': topk_idx, 'topk_weights': topk_weights, 'num_of_experts': NUM_OF_EXPERTS}
            bench(lambda: buffer.dispatch(**dispatch_args))
        with torch.cuda.nvtx.range("hybrid-ep combine"):
            if rank == 0:
                print(f"profile hybrid-ep combine", flush=True)
            combine_args = {'hidden': dispatched_hidden_bf16, 'probs': dispatched_probs, 'handle': handle}
            bench(lambda: buffer.combine(**combine_args))
        with torch.cuda.nvtx.range(f"hybrid-ep dispatch+permute ({dtype_label})"):
            if rank == 0:
                print(f"profile hybrid-ep dispatch+permute ({dtype_label})", flush=True)
            dispatch_with_permute_args = {'hidden': hidden, 'scaling_factor': scaling_factor, 'routing_map': routing_map, 'probs': probs, 'pad_multiple': PAD_MULTIPLE}
            bench(lambda: buffer.dispatch_with_permute(**dispatch_with_permute_args))
        with torch.cuda.nvtx.range("hybrid-ep combine+unpermute"):
            if rank == 0:
                print(f"profile hybrid-ep combine+unpermute", flush=True)
            combine_with_unpermute_args = {'hidden': dispatched_hidden_bf16_with_permute, 'probs': dispatched_probs_with_permute, 'handle': handle_with_permute, 'pad_multiple': PAD_MULTIPLE}
            bench(lambda: buffer.combine_with_unpermute(**combine_with_unpermute_args))
        time.sleep(1)
        if torch.distributed.get_rank() == 0:
            torch.cuda.profiler.stop()


def benchmark_plain_torch_api(
    buffer: deep_ep.HybridEPBuffer,
    use_fp8: bool,
    num_warmups: int,
    num_tests: int,
) -> dict[str, float]:
    hidden, _, scaling_factor, _, topk_idx, topk_weights = init_tensor(
        hidden_dim=HIDDEN_DIM,
        seq_len=NUM_TOKENS_PER_RANK,
        topk=TOPK,
        num_of_experts=NUM_OF_EXPERTS,
        use_fp8=use_fp8,
    )

    for _ in range(10):
        dispatched_hidden, dispatched_probs, _, handle = buffer.dispatch(
            hidden=hidden,
            scaling_factor=scaling_factor,
            topk_idx=topk_idx,
            topk_weights=topk_weights,
            num_of_experts=NUM_OF_EXPERTS,
        )
        dispatched_hidden_bf16 = dispatched_hidden.to(torch.bfloat16)
        buffer.combine(dispatched_hidden_bf16, None, handle)

    dispatched_hidden, dispatched_probs, _, handle = buffer.dispatch(
        hidden=hidden,
        scaling_factor=scaling_factor,
        topk_idx=topk_idx,
        topk_weights=topk_weights,
        num_of_experts=NUM_OF_EXPERTS,
    )
    dispatched_hidden_bf16 = dispatched_hidden.to(torch.bfloat16)

    fp8_factor = (1 + 4 / 128) / 2
    dispatch_bf16_nvl_recv_bytes = dispatched_hidden.numel() * 2
    combine_bf16_nvl_send_bytes = dispatch_bf16_nvl_recv_bytes
    nvl_recv_bytes = (
        dispatch_bf16_nvl_recv_bytes * fp8_factor
        if use_fp8
        else dispatch_bf16_nvl_recv_bytes
    )

    dispatch_args = {
        "hidden": hidden,
        "scaling_factor": scaling_factor,
        "topk_idx": topk_idx,
        "topk_weights": topk_weights,
        "num_of_experts": NUM_OF_EXPERTS,
        "handle": handle,
    }
    combine_args = {
        "hidden": dispatched_hidden_bf16,
        "probs": dispatched_probs,
        "handle": handle,
    }
    dispatch_t = bench(
        lambda: buffer.dispatch(**dispatch_args),
        num_warmups=num_warmups,
        num_tests=num_tests,
    )[0]
    combine_t = bench(
        lambda: buffer.combine(**combine_args),
        num_warmups=num_warmups,
        num_tests=num_tests,
    )[0]
    return {
        "dispatch_torch_nvl_gbps": nvl_recv_bytes / 1e9 / dispatch_t,
        "combine_torch_nvl_gbps": combine_bf16_nvl_send_bytes / 1e9 / combine_t,
        "dispatch_t_us": dispatch_t * 1e6,
        "combine_t_us": combine_t * 1e6,
    }


def run_hybrid_ep_sweep(group: dist.ProcessGroup, args: argparse.Namespace, use_fp8: bool):
    rank = dist.get_rank()
    cases = build_sweep_cases(args)
    if rank == 0:
        print(
            f"[sweep] dtype={dtype_name(use_fp8)} cases={len(cases)} "
            f"warmups={args.sweep_warmups} tests={args.sweep_tests}",
            flush=True,
        )

    results = []
    for idx, case in enumerate(cases, start=1):
        env_overrides = {key: case[key] for key in SWEEP_ENV_KEYS}
        if rank == 0:
            print(
                f"[sweep] case {idx}/{len(cases)} dtype={dtype_name(use_fp8)} "
                f"{format_sweep_case(case)}",
                flush=True,
            )
        dist.barrier()
        with temporary_env(env_overrides):
            try:
                buffer = deep_ep.HybridEPBuffer(
                    group=group,
                    hidden_dim=HIDDEN_DIM,
                    max_num_of_tokens_per_rank=MAX_NUM_OF_TOKENS_PER_RANK,
                    num_local_experts=NUM_LOCAL_EXPERTS,
                    use_fp8=use_fp8,
                    num_sms_dispatch_api=case["num_sms_dispatch_api"],
                    num_sms_combine_api=case["num_sms_combine_api"],
                    load_cached_kernels=True,
                )
                metrics = benchmark_plain_torch_api(
                    buffer,
                    use_fp8=use_fp8,
                    num_warmups=args.sweep_warmups,
                    num_tests=args.sweep_tests,
                )
                result = {**case, **metrics}
                results.append(result)
                if rank == 0:
                    print(
                        f"[sweep-result] dtype={dtype_name(use_fp8)} "
                        f"dispatch={result['dispatch_torch_nvl_gbps']:.2f} GB/s "
                        f"({result['dispatch_t_us']:.2f} us), "
                        f"combine={result['combine_torch_nvl_gbps']:.2f} GB/s "
                        f"({result['combine_t_us']:.2f} us)",
                        flush=True,
                    )
            except Exception as exc:
                if rank == 0:
                    print(
                        f"[sweep-error] dtype={dtype_name(use_fp8)} "
                        f"{format_sweep_case(case)} -> {exc}",
                        flush=True,
                    )
            finally:
                if "buffer" in locals():
                    del buffer
                torch.cuda.empty_cache()
        dist.barrier()

    if rank == 0 and results:
        ranked = sorted(
            results,
            key=lambda item: summarize_score(item, args.sweep_sort_by),
            reverse=True,
        )
        topk = min(args.sweep_topk, len(ranked))
        print(
            f"[sweep-summary] dtype={dtype_name(use_fp8)} sort_by={args.sweep_sort_by} topk={topk}",
            flush=True,
        )
        for i, result in enumerate(ranked[:topk], start=1):
            print(
                f"[sweep-top{i}] dispatch={result['dispatch_torch_nvl_gbps']:.2f} GB/s "
                f"({result['dispatch_t_us']:.2f} us), "
                f"combine={result['combine_torch_nvl_gbps']:.2f} GB/s "
                f"({result['combine_t_us']:.2f} us), "
                f"{format_sweep_case(result)}",
                flush=True,
            )


def test_main(local_rank: int, num_local_ranks: int, args: argparse.Namespace):
    _, _, group = init_dist(local_rank, num_local_ranks)

    # Set missing global vars
    global NUM_OF_RANKS_PER_NODE, NUM_OF_NODES, NUM_OF_EXPERTS
    if USE_MNNVL:
        NUM_OF_RANKS_PER_NODE = group.size()
        NUM_OF_NODES = 1
        NUM_OF_EXPERTS = NUM_LOCAL_EXPERTS * NUM_OF_RANKS_PER_NODE * NUM_OF_NODES
    else:
        NUM_OF_RANKS_PER_NODE = args.num_processes
        NUM_OF_NODES = group.size() // NUM_OF_RANKS_PER_NODE
        NUM_OF_EXPERTS = NUM_LOCAL_EXPERTS * NUM_OF_RANKS_PER_NODE * NUM_OF_NODES

    stream = torch.cuda.Stream()
    with torch.cuda.stream(stream):
        for use_fp8 in args.dtypes:
            if args.sweep:
                run_hybrid_ep_sweep(group, args, use_fp8)
                continue

            buffer = deep_ep.HybridEPBuffer(
                group=group,
                hidden_dim=HIDDEN_DIM,
                max_num_of_tokens_per_rank=MAX_NUM_OF_TOKENS_PER_RANK,
                num_local_experts=NUM_LOCAL_EXPERTS,
                use_fp8=use_fp8
            )
            
            ref = TorchRef(
                ep_group=group,
                num_of_experts=NUM_OF_EXPERTS,
                num_of_ranks_per_node=NUM_OF_RANKS_PER_NODE,
            )

            if not args.skip_correctness:
                test_hybrid_ep_correctness(buffer, ref, use_fp8)
            test_hybrid_ep_benchmark(buffer, group, use_fp8, args.nsys_profile)
    dist.barrier()
    dist.destroy_process_group()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Test intranode EP kernels')
    parser.add_argument('--num-processes', type=int, default=4,
                       help='Number of processes to spawn (default: 4)')
    parser.add_argument('--nsys-profile', action='store_true', default=False,
                       help='benchmark with nsys profile or not (default: False)')
    parser.add_argument('--skip-correctness', action='store_true', default=False,
                       help='skip correctness checks before benchmarking')
    parser.add_argument('--dtypes', type=parse_dtype_list, default=parse_dtype_list('bf16,fp8'),
                       help='comma-separated dtypes to run: bf16,fp8')
    parser.add_argument('--sweep', action='store_true', default=False,
                       help='run parameter sweep for HybridEP torch API benchmarks')
    parser.add_argument('--sweep-warmups', type=int, default=10,
                       help='warmup iterations per sweep case')
    parser.add_argument('--sweep-tests', type=int, default=20,
                       help='timed iterations per sweep case')
    parser.add_argument('--sweep-topk', type=int, default=5,
                       help='number of best sweep cases to summarize')
    parser.add_argument('--sweep-sort-by', choices=('score', 'dispatch', 'combine'), default='score',
                       help='summary ranking metric')
    parser.add_argument('--sweep-sm-dispatch', type=str, default='32',
                       help='comma-separated dispatch SM counts')
    parser.add_argument('--sweep-sm-combine', type=str, default='32',
                       help='comma-separated combine SM counts')
    parser.add_argument('--sweep-dispatch-stages', type=str, default='10',
                       help='comma-separated NUM_OF_STAGES_DISPATCH_API values')
    parser.add_argument('--sweep-dispatch-inflight', type=str, default='8',
                       help='comma-separated NUM_OF_IN_FLIGHT_S2G_DISPATCH_API values')
    parser.add_argument('--sweep-dispatch-chunks', type=str, default='128',
                       help='comma-separated NUM_OF_TOKENS_PER_CHUNK_DISPATCH_API values')
    parser.add_argument('--sweep-combine-g2s-stages', type=str, default='10',
                       help='comma-separated NUM_OF_STAGES_G2S_COMBINE_API values')
    parser.add_argument('--sweep-combine-s2g-stages', type=str, default='2',
                       help='comma-separated NUM_OF_STAGES_S2G_COMBINE_API values')
    parser.add_argument('--sweep-combine-chunks', type=str, default='128',
                       help='comma-separated NUM_OF_TOKENS_PER_CHUNK_COMBINE_API values')
    parser.add_argument('--sweep-combine-groups', type=str, default='4',
                       help='comma-separated NUM_OF_TOKENS_PER_GROUP_COMBINE_API values')
    parser.add_argument('--sweep-combine-inflight', type=str, default='2',
                       help='comma-separated NUM_OF_ADDITIONAL_IN_FLIGHT_S2G_COMBINE_API values')
    args = parser.parse_args()
    torch.multiprocessing.spawn(test_main, args=(args.num_processes, args), nprocs=args.num_processes)
