"""RoleWorker.save_hf 的设备放置回归测试。

HF 分片重建（``_save_hf`` → ``unshard_tensors_for_hf_save``）对参数做
all-gather，集合通信走 NCCL-only 的训练进程组；角色在自己的 fit 结束后可能
被下卡（如 PPO critic 在 actor 训练期间保持在 CPU），``save_hf`` 必须先把
模型搬回训练设备，否则 CPU tensor 进入 NCCL gather 会直接抛出
``RuntimeError: No backend type associated with device type cpu``。
"""

import unittest
from pathlib import Path
from unittest.mock import MagicMock, call

import torch

from xtuner.v1.rl.trainer import role_worker as role_worker_module
from xtuner.v1.rl.trainer.role_worker import RoleWorker


class _SaveRoleStub(RoleWorker):
    """只实现抽象方法、用于验证 save_hf 放置契约的角色桩。"""

    role_name = "stub"

    def fit(self):
        raise NotImplementedError

    def _build_step_input(self):
        raise NotImplementedError


class TestRoleWorkerSaveHfDevice(unittest.TestCase):
    """RoleWorker.save_hf 在保存前的设备放置（public: save_hf）。"""

    def test_save_hf_onloads_model_before_engine_save(self):
        # 行为：save_hf 先把模型搬回训练设备再调用 engine.save_hf，
        # 且不触碰优化器放置。
        role = _SaveRoleStub.__new__(_SaveRoleStub)
        role.engine = MagicMock()
        role.logger = MagicMock()

        RoleWorker.save_hf(role, Path("/tmp/hf-stub"), torch.bfloat16)

        self.assertEqual(
            role.engine.method_calls,
            [
                call.put_model_to_device(role_worker_module.DEVICE),
                call.save_hf(hf_dir=Path("/tmp/hf-stub"), save_dtype=torch.bfloat16),
            ],
        )
