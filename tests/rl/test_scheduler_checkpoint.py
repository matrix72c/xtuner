"""Worker checkpoint wiring with real CPU optimizers and LR schedules."""

from copy import deepcopy
from pathlib import Path
from unittest.mock import Mock

import pytest
import torch

from xtuner.v1.config.optim import LRConfig
from xtuner.v1.rl.trainer.critic_worker import CriticWorker
from xtuner.v1.rl.trainer.schedule import build_lr_scheduler, resume_lr_scheduler, save_lr_scheduler
from xtuner.v1.rl.trainer.worker import TrainingWorker
from xtuner.v1.train.trainer import LoadCheckpointConfig


def scheduled(kind="linear", warmup=2, total=12):
    optimizer = torch.optim.Adam([torch.nn.Parameter(torch.ones(1))], lr=1e-3)
    scheduler = build_lr_scheduler(optimizer, LRConfig(lr_type=kind, warmup_ratio=warmup, lr_min=1e-5), total)
    return optimizer, scheduler


def advance(optimizer, scheduler, count):
    values = []
    for _ in range(count):
        optimizer.step()
        scheduler.step()
        values.append(optimizer.param_groups[0]["lr"])
    return values


@pytest.mark.parametrize("kind", ["constant", "linear", "cosine"])
@pytest.mark.parametrize("checkpoint_step", [1, 5])
def test_resume_matches_uninterrupted_schedule(tmp_path, kind, checkpoint_step):
    opt, sched = scheduled(kind)
    advance(opt, sched, checkpoint_step)
    opt_state = deepcopy(opt.state_dict())
    save_lr_scheduler(sched, tmp_path)
    expected = advance(opt, sched, 4)
    resumed_opt, resumed_sched = scheduled(kind)
    resumed_opt.load_state_dict(opt_state)
    resume_lr_scheduler(resumed_sched, tmp_path)
    assert advance(resumed_opt, resumed_sched, 4) == pytest.approx(expected)


def test_reset_and_missing_or_changed_schedule_are_explicit(tmp_path):
    opt, sched = scheduled()
    with pytest.raises(FileNotFoundError, match="load_scheduler=False"):
        resume_lr_scheduler(sched, tmp_path)
    advance(opt, sched, 5)
    save_lr_scheduler(sched, tmp_path)
    other_opt, other_sched = scheduled(total=24)
    with pytest.raises(ValueError, match="differs"):
        resume_lr_scheduler(other_sched, tmp_path)
    other_opt.load_state_dict(opt.state_dict())
    resume_lr_scheduler(other_sched, tmp_path, load_scheduler=False)
    assert other_opt.param_groups[0]["lr"] == other_sched.get_last_lr()[0] == 0.0005
    with pytest.raises(ValueError, match="disables"):
        resume_lr_scheduler(None, tmp_path)


def test_actor_and_attached_critic_checkpoint_wiring(tmp_path):
    saved_optimizers = {}

    class Engine:
        def __init__(self, optimizer):
            self.optimizer = optimizer

        def save_dcp(self, weights_dir, save_optimizer):
            Path(weights_dir).mkdir(parents=True)
            saved_optimizers[str(weights_dir)] = deepcopy(self.optimizer.state_dict())

        def load_dcp(self, weights_dir, load_states, load_args):
            self.optimizer.load_state_dict(saved_optimizers[str(weights_dir)])

    def worker():
        actor = TrainingWorker.__new__(TrainingWorker)
        critic = CriticWorker.__new__(CriticWorker)
        actor_opt, actor._lr_scheduler = scheduled("linear")
        critic_opt, critic._scheduler = scheduled("cosine")
        actor._engine, critic._engine = Engine(actor_opt), Engine(critic_opt)
        actor._attached, actor._sft_dataloader, actor.logger = {"critic": critic}, None, Mock()
        return actor, critic

    actor, critic = worker()
    advance(actor._engine.optimizer, actor._lr_scheduler, 3)
    advance(critic._engine.optimizer, critic._scheduler, 5)
    actor.save(tmp_path)
    assert (tmp_path / "lr_scheduler.json").exists()
    assert (tmp_path / "critic/lr_scheduler.json").exists()
    expected_actor = advance(actor._engine.optimizer, actor._lr_scheduler, 2)
    expected_critic = advance(critic._engine.optimizer, critic._scheduler, 2)
    restored, restored_critic = worker()
    restored.resume(LoadCheckpointConfig(checkpoint_path=tmp_path, load_scheduler=True))
    assert advance(restored._engine.optimizer, restored._lr_scheduler, 2) == pytest.approx(expected_actor)
    assert advance(restored_critic._engine.optimizer, restored_critic._scheduler, 2) == pytest.approx(expected_critic)
