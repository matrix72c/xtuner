"""Bounded pinned allocations with ordinary views that retain CUDA event ownership."""

import torch


def allocate_pinned_like(tensors: list[torch.Tensor], *, zero: bool = False) -> list[torch.Tensor]:
    """Allocate compatible dense layouts in bounded pinned slabs without copying inputs.

    Ordinary views retain the host allocator's stream-event context. Full-state
    torch.save deduplicates their storage; DCP compacts individual views. A raw
    torch.save of a single view includes its whole slab unless cloned first.

    Args:
        tensors (list[torch.Tensor]): Plain strided tensors supplying shape, dtype and layout.
        zero (bool): Initialize each slab to zero before exposing its views.

    Returns:
        list[torch.Tensor]: CPU views in input order, with empty_like preserve-format layouts.
    """
    layouts = [torch.empty_like(tensor, device="meta", memory_format=torch.preserve_format) for tensor in tensors]
    groups: dict[tuple[torch.device, torch.dtype], list[int]] = {}
    for index, tensor in enumerate(tensors):
        groups.setdefault((tensor.device, tensor.dtype), []).append(index)
    result: list[torch.Tensor | None] = [None] * len(tensors)
    for (_, dtype), indices in groups.items():
        sizes = [layouts[index].numel() * layouts[index].element_size() for index in indices]
        for members in plan_pinned_slabs(sizes):
            element_size = layouts[indices[members[0]]].element_size()
            slab = torch.empty(
                sum(sizes[i] for i in members) // element_size, dtype=dtype, device="cpu", pin_memory=True
            )
            if zero:
                slab.zero_()
            offset = 0
            for member in members:
                index = indices[member]
                layout = layouts[index]
                result[index] = slab.as_strided(layout.shape, layout.stride(), offset)
                offset += layout.numel()
    assert all(tensor is not None for tensor in result)
    return [tensor for tensor in result if tensor is not None]


def plan_pinned_slabs(sizes: list[int], cap_bytes: int = 1 << 30) -> list[list[int]]:
    """Plan deterministic best-fit-decreasing slabs bounded by a byte cap.

    Args:
        sizes (list[int]): Nonnegative allocation sizes in bytes.
        cap_bytes (int): Maximum slab size; larger individual requests stay standalone.

    Returns:
        list[list[int]]: Input indices grouped into slabs; small slabs allocate only used bytes.
    """
    if cap_bytes <= 0 or any(size < 0 for size in sizes):
        raise ValueError("Slab cap must be positive and request sizes nonnegative")
    bins: list[list[int]] = []
    used: list[int] = []
    for index in sorted(range(len(sizes)), key=lambda index: (-sizes[index], index)):
        size = sizes[index]
        candidates = [i for i, total in enumerate(used) if total <= cap_bytes and total + size <= cap_bytes]
        if candidates:
            best = min(candidates, key=lambda i: (cap_bytes - used[i] - size, i))
            bins[best].append(index)
            used[best] += size
        else:
            bins.append([index])
            used.append(size)
    return bins
