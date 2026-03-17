from dataclasses import dataclass
from typing import Optional

import torch
import torch.distributed as dist

from utils import per_token_cast_to_fp8


@dataclass
class SharedInputs:
    hidden: torch.Tensor
    topk_idx: torch.Tensor
    topk_weights: torch.Tensor
    num_experts: int


def calc_max_abs_diff(x: torch.Tensor, y: torch.Tensor) -> float:
    if x is None or y is None:
        return float("inf")
    if x.numel() == 0 and y.numel() == 0:
        return 0.0
    x_fp32 = x.to(torch.float32)
    y_fp32 = y.to(torch.float32)

    x_nan = torch.isnan(x_fp32)
    y_nan = torch.isnan(y_fp32)
    if not torch.equal(x_nan, y_nan):
        return float("inf")

    valid_mask = ~x_nan
    if not torch.any(valid_mask):
        return 0.0
    return (x_fp32[valid_mask] - y_fp32[valid_mask]).abs().max().item()


def quantize_for_hybrid_dispatch(hidden: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    hidden_fp8, hidden_scale = per_token_cast_to_fp8(hidden)
    return hidden_fp8.contiguous(), hidden_scale.contiguous()


def _row_sort_indices(proxy_rows: torch.Tensor) -> torch.Tensor:
    if proxy_rows.numel() == 0:
        return torch.empty((0,), dtype=torch.long, device=proxy_rows.device)

    proxy_cpu = proxy_rows.detach().contiguous().cpu()
    keys = [
        proxy_cpu[i].view(torch.uint8).numpy().tobytes()
        for i in range(proxy_cpu.shape[0])
    ]
    order = sorted(range(len(keys)), key=keys.__getitem__)
    return torch.tensor(order, dtype=torch.long, device=proxy_rows.device)


def canonicalize_rows_per_expert(
    hidden: torch.Tensor,
    scale: torch.Tensor,
    proxy_rows: torch.Tensor,
    tokens_per_expert: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    counts = tokens_per_expert.to(torch.int64).cpu().tolist()

    hidden_chunks = []
    scale_chunks = []
    proxy_chunks = []
    start = 0
    for count in counts:
        end = start + count
        hidden_chunk = hidden[start:end]
        scale_chunk = scale[start:end]
        proxy_chunk = proxy_rows[start:end]

        order = _row_sort_indices(proxy_chunk)
        hidden_chunks.append(hidden_chunk.index_select(0, order))
        scale_chunks.append(scale_chunk.index_select(0, order))
        proxy_chunks.append(proxy_chunk.index_select(0, order))
        start = end

    if hidden_chunks:
        return (
            torch.cat(hidden_chunks, dim=0).contiguous(),
            torch.cat(scale_chunks, dim=0).contiguous(),
            torch.cat(proxy_chunks, dim=0).contiguous(),
        )

    return hidden.contiguous(), scale.contiguous(), proxy_rows.contiguous()


def build_shared_inputs(
    num_tokens: int,
    hidden_dim: int,
    num_topk: int,
    num_local_experts: int,
    group: dist.ProcessGroup,
) -> SharedInputs:
    world_size = dist.get_world_size(group)
    num_experts = world_size * num_local_experts
    device = torch.device("cuda")

    hidden = torch.empty((num_tokens, hidden_dim), device=device, dtype=torch.bfloat16)
    topk_idx = torch.empty((num_tokens, num_topk), device=device, dtype=torch.int64)
    topk_weights = torch.empty((num_tokens, num_topk), device=device, dtype=torch.float32)

    if dist.get_rank(group) == 0:
        hidden.copy_(torch.randn_like(hidden))
        scores = torch.randn((num_tokens, num_experts), device=device, dtype=torch.float32).abs() + 1
        topk_idx.copy_(torch.topk(scores, num_topk, dim=-1, largest=True, sorted=False).indices)
        topk_weights.fill_(1.0 / num_topk)

    dist.broadcast(hidden, src=0, group=group)
    dist.broadcast(topk_idx, src=0, group=group)
    dist.broadcast(topk_weights, src=0, group=group)

    return SharedInputs(
        hidden=hidden.contiguous(),
        topk_idx=topk_idx.contiguous(),
        topk_weights=topk_weights.contiguous(),
        num_experts=num_experts,
    )
