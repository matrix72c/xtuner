from typing import Any
from unittest.mock import patch

import pytest
import torch

from xtuner.v1.utils.pinned_memory import allocate_pinned_like, plan_pinned_slabs


class TestPinnedMemory:
    def test_plan_bounds_and_empty(self) -> None:
        assert plan_pinned_slabs([0, 6, 4, 3, 7, 12, 1], 10) == [[5], [4, 3, 0], [1, 2], [6]]
        assert plan_pinned_slabs([]) == []
        assert plan_pinned_slabs([0, 0]) == [[0, 1]]
        with pytest.raises(ValueError):
            plan_pinned_slabs([-1])
        with pytest.raises(ValueError):
            plan_pinned_slabs([1], 0)

    def test_layout_zeroing_and_nonoverlap(self) -> None:
        tensors = [
            torch.empty(3, 5).t(),
            torch.empty(0, 3),
            torch.empty(6, 8)[:, ::2],
            torch.empty(7, dtype=torch.float64),
        ]
        original = torch.empty
        allocated = []

        def empty(*args: Any, **kwargs: Any) -> torch.Tensor:
            if kwargs.pop("pin_memory", False):
                allocated.append(args[0])
            return original(*args, **kwargs)

        with patch.object(torch, "empty", empty):
            views = allocate_pinned_like(tensors, zero=True)
        assert len(allocated) == 2
        spans = {}
        for source, view in zip(tensors, views):
            assert view.shape == source.shape and view.dtype == source.dtype
            assert view.stride() == torch.empty_like(source).stride()
            assert torch.count_nonzero(view) == 0
            spans.setdefault(view.untyped_storage().data_ptr(), []).append(
                (view.storage_offset(), view.storage_offset() + view.numel())
            )
        for ranges in spans.values():
            ranges.sort()
            assert all(a[1] <= b[0] for a, b in zip(ranges, ranges[1:]))
        assert allocate_pinned_like([]) == []
