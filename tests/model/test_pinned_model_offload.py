"""Model CPU destinations: run GPU cases with torchrun --nproc_per_node=2."""

import io
import os
from collections import OrderedDict
from pathlib import Path
from typing import Any

import pytest
import torch
import torch.distributed as dist
import torch.distributed.checkpoint as dcp
from safetensors.torch import load_file, save_file
from torch import nn
from torch.distributed.device_mesh import init_device_mesh
from torch.distributed.fsdp import fully_shard
from torch.distributed.tensor import DTensor, Replicate
from torch.distributed.tensor._dtensor_spec import DTensorSpec, TensorMeta
from torch.testing._internal.distributed.fake_pg import FakeStore

from xtuner.v1.model.base import BaseModel, XTunerBaseModelConfig
from xtuner.v1.utils.model_offload import offload_to_pinned_cpu
from xtuner.v1.utils.pinned_memory import allocate_pinned_like


def _local(tensor: torch.Tensor) -> torch.Tensor:
    return tensor.to_local() if isinstance(tensor, DTensor) else tensor


class TinyModel(BaseModel):
    def __init__(self, device: str = "cpu") -> None:
        super().__init__(XTunerBaseModelConfig())
        self.first = nn.Linear(7, 5, device=device)
        self.last = nn.Linear(5, 1, device=device, bias=False)
        self.register_buffer("counter", torch.tensor(3, dtype=torch.int64, device=device))
        self._init_load_spec()

    def to_hf_key_list(self, key: str) -> list[str]:
        return [key]

    _model_hf_keys = to_hf_key_list

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

    def test_checkpoint_boundary_preserves_hooks_destination_and_live_parameters(self) -> None:
        dist.init_process_group("fake", rank=0, world_size=2, store=FakeStore())
        try:
            model = TinyModel()
            mesh = init_device_mesh("cpu", (2,))
            slab = torch.arange(40.0)
            local = slab[2:37].view(5, 7)
            spec = DTensorSpec(mesh, (Replicate(),), tensor_meta=TensorMeta(local.shape, local.stride(), local.dtype))
            parameter = nn.Parameter(DTensor(local, spec, requires_grad=False))
            model.first.weight = parameter
            calls: list[tuple[str, str, bool]] = []

            def before(module: nn.Module, prefix: str, keep_vars: bool) -> None:
                calls.append(("pre", prefix, keep_vars))

            def after(module: nn.Module, state: dict[str, Any], prefix: str, metadata: dict[str, Any]) -> None:
                calls.append(("post", prefix, False))
                # A hook may create a fresh offset wrapper after the ordinary detach.
                state[prefix + "hook"] = parameter.detach()

            model.register_state_dict_pre_hook(before)
            model.register_state_dict_post_hook(after)
            untouched = parameter.detach()
            destination = OrderedDict({"actor.foreign": untouched, "other.weight": untouched})
            state = model.state_dict(destination=destination, prefix="actor.")
            assert state is destination
            assert state["actor.foreign"] is untouched and state["other.weight"] is untouched
            assert calls == [("pre", "actor.", False), ("post", "actor.", False)]
            for key in ("actor.first.weight", "actor.hook"):
                assert state[key].storage_offset() == 0
                assert state[key]._spec is parameter._spec
                assert state[key]._local_tensor.untyped_storage() is local.untyped_storage()
            assert model.first.weight is parameter and parameter.storage_offset() == 2
            state = model.state_dict()
            stream = io.BytesIO()
            torch.save(state, stream)
            stream.seek(0)
            restored = torch.load(stream, weights_only=False)
            torch.testing.assert_close(restored["first.weight"].to_local(), local)
            assert state._metadata["first"]["version"] == model.first._version
            assert model.state_dict(keep_vars=True)["first.weight"] is parameter
            with pytest.warns(FutureWarning):
                positional = model.state_dict(None, "legacy.", False)
            assert positional["legacy.first.weight"].storage_offset() == 0
            with pytest.warns(FutureWarning):
                live = model.state_dict(None, "legacy.", True)
            assert live["legacy.first.weight"] is parameter
        finally:
            dist.destroy_process_group()

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

    @pytest.mark.parametrize("initial_device", ["meta", "cuda"])
    def test_hf_load_refreshes_parent_owned_shards_before_first_forward(
        self, tmp_path: Path, initial_device: str
    ) -> None:
        torch.cuda.set_device(int(os.environ.get("LOCAL_RANK", 0)))
        if "RANK" in os.environ:
            dist.init_process_group("nccl")
        else:
            dist.init_process_group("nccl", init_method=f"file://{tmp_path / 'pg'}", rank=0, world_size=1)
        try:
            torch.manual_seed(42)
            reference = TinyModel("cuda")
            save_file(
                {name: value.cpu() for name, value in reference.state_dict().items()}, tmp_path / "model.safetensors"
            )
            model = TinyModel(initial_device)
            mesh = init_device_mesh("cuda", (dist.get_world_size(),))
            # The parent owns ordinary leaf parameters. Leaf-only materialization does not
            # run that parent's FSDP _apply reset, exactly as in recursive HF loading.
            fully_shard(model, mesh=mesh)
            if initial_device == "cuda":
                model.first.to_empty(device="cuda", recurse=False)
                model.last.to_empty(device="cuda", recurse=False)
            assert model.from_hf(tmp_path) == (set(reference.state_dict()), set(), set())
            group = model._get_fsdp_state()._fsdp_param_group
            assert group is not None
            assert all(param.sharded_param._local_tensor.device.type == "cuda" for param in group.fsdp_params)
            if initial_device == "meta":
                assert all(param._sharded_param_data.is_meta for param in group.fsdp_params)
            identities = {name: id(param) for name, param in model.named_parameters()}
            snapshot = {name: _local(param).detach().cpu().clone() for name, param in model.named_parameters()}
            for _ in range(3):
                model.to_device("cpu")
                assert {name: id(param) for name, param in model.named_parameters()} == identities
                for name, param in model.named_parameters():
                    torch.testing.assert_close(_local(param), snapshot[name])
                for param in group.fsdp_params:
                    backing = param._sharded_param_data
                    assert backing.is_pinned() and backing.numel() == param.padded_sharded_param_size.numel()
                    assert backing.untyped_storage() is param.sharded_param._local_tensor.untyped_storage()
                stream = io.BytesIO()
                torch.save(model.state_dict(), stream)
                stream.seek(0)
                restored = torch.load(stream, weights_only=False)
                for name in snapshot:
                    torch.testing.assert_close(_local(restored[name]), snapshot[name])
                model.to_device("cuda")
            inputs = torch.ones(2, 7, device="cuda")
            torch.testing.assert_close(model(inputs), reference(inputs))
            model(inputs).sum().backward()
            reference(inputs).sum().backward()
            actual_optim = torch.optim.SGD(model.parameters(), lr=0.01)
            expected_optim = torch.optim.SGD(reference.parameters(), lr=0.01)
            actual_optim.step()
            expected_optim.step()
            actual_optim.zero_grad(set_to_none=True)
            model.to_device("cpu")
            model.to_device("cuda")
            torch.testing.assert_close(model(inputs), reference(inputs))
        finally:
            dist.destroy_process_group()

    def test_fsdp_padded_backing_identity_and_training(self, tmp_path: Path) -> None:
        torch.cuda.set_device(int(os.environ.get("LOCAL_RANK", 0)))
        # CPU HF reconstruction needs Gloo; CUDA training still uses NCCL.
        backend = "cpu:gloo,cuda:nccl"
        if "RANK" in os.environ:
            dist.init_process_group(backend)
        else:
            dist.init_process_group(backend, init_method=f"file://{tmp_path / 'pg'}", rank=0, world_size=1)
        try:
            shared_paths = [str(tmp_path)]
            dist.broadcast_object_list(shared_paths, src=0)
            shared_path = Path(shared_paths[0])
            origin = shared_path / "origin"
            origin.mkdir(exist_ok=True)
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
            model._hf_path = origin
            for step in range(3):
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
                checkpoint_dir = shared_path / f"dcp-{step}"
                dcp.save({"model": model.state_dict()}, checkpoint_id=checkpoint_dir)
                with torch.no_grad():
                    for param in model.parameters():
                        _local(param).zero_()
                dcp.load({"model": model.state_dict()}, checkpoint_id=checkpoint_dir)
                for name, param in model.named_parameters():
                    torch.testing.assert_close(_local(param), snapshot[name])
                hf_dir = shared_path / f"hf-{step}"
                model.save_hf(hf_dir)
                if dist.get_rank() == 0:
                    exported: dict[str, torch.Tensor] = {}
                    for filename in hf_dir.glob("*.safetensors"):
                        exported.update(load_file(filename))
                    assert set(exported) == set(reference.state_dict())
                    for name, tensor in reference.state_dict().items():
                        expected_tensor = tensor.detach().cpu()
                        if expected_tensor.is_floating_point():
                            expected_tensor = expected_tensor.bfloat16()
                        torch.testing.assert_close(exported[name], expected_tensor)
                dist.barrier()
                model.to_device("cuda")
                optimizer.zero_grad(set_to_none=True)
                expected.zero_grad(set_to_none=True)
            torch.testing.assert_close(model(inputs), reference(inputs))
        finally:
            dist.destroy_process_group()
