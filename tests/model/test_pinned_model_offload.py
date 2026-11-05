"""Model CPU destinations: run GPU cases with torchrun --nproc_per_node=2."""

import os
from pathlib import Path

import pytest
import torch
import torch.distributed as dist
from torch import nn
from torch.distributed.device_mesh import init_device_mesh
from torch.distributed.fsdp import fully_shard
from torch.distributed.tensor import DTensor

from xtuner.v1.model.base import BaseModel
from xtuner.v1.utils.model_offload import offload_to_pinned_cpu
from xtuner.v1.utils.pinned_memory import allocate_pinned_like


def _local(tensor: torch.Tensor) -> torch.Tensor:
    return tensor.to_local() if isinstance(tensor, DTensor) else tensor


class TinyModel(nn.Module):
    fsdp_config = None
    to_device = BaseModel.to_device

    def __init__(self, device: str = "cpu") -> None:
        super().__init__()
        self.first = nn.Linear(7, 5, device=device)
        self.last = nn.Linear(5, 1, device=device, bias=False)
        self.register_buffer("counter", torch.tensor(3, dtype=torch.int64, device=device))

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.last(self.first(inputs))


class TestModelOffloadCPU:
    def test_cpu_noop_keeps_parameters_gradients_and_buffers(self) -> None:
        model = TinyModel()
        parameter = model.first.weight
        gradient = torch.ones_like(parameter)
        parameter.grad = gradient
        buffer = model.counter
        offload_to_pinned_cpu(model)
        assert model.first.weight is parameter
        assert parameter.grad is gradient
        assert model.counter is buffer

    def test_differentiable_buffer_keeps_autograd(self) -> None:
        model = nn.Module()
        model.register_buffer("value", torch.ones(2, requires_grad=True))
        offload_to_pinned_cpu(model)
        model.value.sum().backward()
        torch.testing.assert_close(model.value.grad, torch.ones(2))


@pytest.mark.gpu
@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
class TestModelOffloadGPU:
    def test_dense_alias_gradient_and_noncontiguous_roundtrip(self) -> None:
        torch.cuda.set_device(int(os.environ.get("LOCAL_RANK", 0)))
        model = TinyModel("cuda")
        model.register_parameter("transposed", nn.Parameter(torch.arange(15.0, device="cuda").reshape(3, 5).t()))
        model.register_parameter("tied", model.first.weight)
        model.first.weight.grad = torch.full_like(model.first.weight, 2)
        original = {name: param.detach().cpu().clone() for name, param in model.named_parameters()}
        identities = {name: id(param) for name, param in model.named_parameters()}
        for _ in range(3):
            model.to_device("cpu")
            assert model.first.weight is model.tied
            assert {name: id(param) for name, param in model.named_parameters()} == identities
            for name, param in model.named_parameters():
                assert param.is_pinned()
                torch.testing.assert_close(param, original[name])
            assert model.transposed.stride() == (1, 5)
            assert model.first.weight.grad.is_pinned()
            torch.testing.assert_close(model.first.weight.grad, torch.full_like(model.first.weight, 2))
            assert model.counter.dtype == torch.int64 and model.counter.item() == 3
            model.to_device("cuda")
        model(torch.ones(2, 7, device="cuda")).sum().backward()

    def test_nondefault_stream_copy_lifetime(self) -> None:
        torch.cuda.set_device(int(os.environ.get("LOCAL_RANK", 0)))
        stream = torch.cuda.Stream()
        shapes = [torch.empty(262145, device="meta") for _ in range(4)]
        for _ in range(3):
            with torch.cuda.stream(stream):
                model = nn.Module()
                for index in range(4):
                    model.register_parameter(f"p{index}", nn.Parameter(torch.ones(262145, device="cuda")))
                torch.cuda._sleep(5_000_000)
                offload_to_pinned_cpu(model)
                del model
            # Freed views must retain their host allocator events until D2H finishes.
            pressure = allocate_pinned_like(shapes)
            for tensor in pressure:
                tensor.fill_(-7)
            stream.synchronize()
            assert all(torch.all(tensor == -7) for tensor in pressure)
            del pressure
            with torch.cuda.stream(stream):
                model = nn.Module()
                for index in range(4):
                    model.register_parameter(f"p{index}", nn.Parameter(torch.ones(262145, device="cuda")))
                offload_to_pinned_cpu(model)
            stream.synchronize()
            with torch.cuda.stream(stream):
                torch.cuda._sleep(5_000_000)
                model.to("cuda", non_blocking=True)
                outputs = [param.detach() for param in model.parameters()]
                del model
            # Likewise, CPU slabs must not be reused while H2D still reads them.
            pressure = allocate_pinned_like(shapes)
            for tensor in pressure:
                tensor.fill_(-9)
            stream.synchronize()
            assert all(torch.all(tensor == 1).item() for tensor in outputs)

    def test_sparse_and_differentiable_buffers(self) -> None:
        torch.cuda.set_device(int(os.environ.get("LOCAL_RANK", 0)))
        model = TinyModel("cuda")
        source = torch.ones(3, device="cuda", requires_grad=True)
        model.register_buffer("differentiable", source)
        model.register_buffer("sparse", torch.eye(3, device="cuda").to_sparse())
        model.to_device("cpu")
        assert model.sparse.layout == torch.sparse_coo
        torch.testing.assert_close(model.sparse.to_dense(), torch.eye(3))
        model.differentiable.sum().backward()
        torch.testing.assert_close(source.grad, torch.ones(3, device="cuda"))

    def test_fsdp_padded_backing_identity_and_training(self, tmp_path: Path) -> None:
        torch.cuda.set_device(int(os.environ.get("LOCAL_RANK", 0)))
        if "RANK" in os.environ:
            dist.init_process_group("nccl")
        else:
            dist.init_process_group("nccl", init_method=f"file://{tmp_path / 'pg'}", rank=0, world_size=1)
        try:
            torch.manual_seed(42)
            model = TinyModel("cuda")
            reference = TinyModel("cuda")
            reference.load_state_dict(model.state_dict())
            mesh = init_device_mesh("cuda", (dist.get_world_size(),))
            fully_shard(model.first, mesh=mesh)
            fully_shard(model.last, mesh=mesh)
            fully_shard(model, mesh=mesh)
            optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
            expected = torch.optim.SGD(reference.parameters(), lr=0.01)
            identities = {name: id(param) for name, param in model.named_parameters()}
            inputs = torch.ones(2, 7, device="cuda")
            for _ in range(3):
                output = model(inputs)
                torch.testing.assert_close(output, reference(inputs))
                output.sum().backward()
                reference(inputs).sum().backward()
                optimizer.step()
                expected.step()
                gradients = {
                    name: _local(param.grad).detach().cpu().clone() for name, param in model.named_parameters()
                }
                snapshot = {name: _local(param).detach().cpu().clone() for name, param in model.named_parameters()}
                model.to_device("cpu")
                assert {name: id(param) for name, param in model.named_parameters()} == identities
                assert all(
                    param is current for param, current in zip(optimizer.param_groups[0]["params"], model.parameters())
                )
                for name, param in model.named_parameters():
                    torch.testing.assert_close(_local(param), snapshot[name])
                    torch.testing.assert_close(_local(param.grad), gradients[name])
                    if _local(param).numel():
                        assert _local(param).is_pinned()
                backings = []
                for child in (model.first, model.last):
                    for param in child._get_fsdp_state()._fsdp_param_group.fsdp_params:
                        backing = param._sharded_param_data
                        assert backing.device.type == "cpu" and backing.is_pinned()
                        assert backing.numel() == param.padded_sharded_param_size.numel()
                        assert (
                            backing.untyped_storage().data_ptr()
                            == param.sharded_param._local_tensor.untyped_storage().data_ptr()
                        )
                        backings.append(backing)
                assert len({tensor.untyped_storage().data_ptr() for tensor in backings}) == 1
                checkpoint = tmp_path / f"rank-{dist.get_rank()}.pt"
                torch.save(model.state_dict(), checkpoint)
                restored = torch.load(checkpoint, weights_only=False)
                for name, param in model.named_parameters():
                    torch.testing.assert_close(_local(restored[name]), snapshot[name])
                    assert restored[name].shape == param.shape
                    assert restored[name].placements == param.placements
                del restored, backings
                model.to_device("cuda")
                optimizer.zero_grad(set_to_none=True)
                expected.zero_grad(set_to_none=True)
            torch.testing.assert_close(model(inputs), reference(inputs))
        finally:
            dist.destroy_process_group()
