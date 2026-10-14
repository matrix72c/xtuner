"""Real shared Runner/loop with synthetic model requests and owned environments."""

from __future__ import annotations

import asyncio
from contextlib import asynccontextmanager
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest
import test_sandbox_pool  # noqa: F401 - installs functional optional lagent helpers

from xtuner.v1.data_proto.rl_data import RolloutState, Status
from xtuner.v1.rl.agent_loop import agent_loop
from xtuner.v1.rl.agent_loop.sandbox_agent_loop import (
    AgentInSandboxLoopConfig,
    AgentRolloutItem,
    Judger,
    Runner,
    SandboxInfo,
    SandboxStage,
)
from xtuner.v1.rl.agent_loop.sandbox_agent_loop import agent_in_sandbox_loop as impl


@pytest.mark.asyncio
async def test_one_environment_and_session_survive_pause_and_policy_update(monkeypatch):
    first, next_request, gate = asyncio.Event(), asyncio.Event(), asyncio.Event()
    gate.set()
    response = []
    seen_sessions, owned = [], []

    def segments():
        return [{"messages": [{"role": "assistant", "content": "".join(response)}], "tools": []}]

    async def control(action):
        if action == "pause":
            gate.clear()
        elif action == "resume":
            gate.set()
        return {"active": 0, "completed": len(response)}

    class Infer:
        async def __call__(self, client, item, record):
            seen_sessions.append(item.uid)
            await item.execution.attach(control, lambda expected: read_segments())
            response.append("a")
            first.set()
            await next_request.wait()
            await gate.wait()
            response.append("b")
            item.artifacts.update(messages=segments(), response_message={"content": "ab"})

    async def read_segments():
        return segments()

    class Score:
        async def __call__(self, client, item, record):
            record.score = 1

    class Provider:
        @asynccontextmanager
        async def open(self, item):
            owned.append("open")
            env = SimpleNamespace(
                get=AsyncMock(return_value=object()),
                info=lambda name: SandboxInfo("neutral", "/app", "same-environment"),
                check=AsyncMock(),
            )
            try:
                yield env
            finally:
                owned.append("release")

        async def close(self):
            owned.append("close")

    tokenizer = SimpleNamespace(
        eos_token=None,
        apply_chat_template=lambda messages, **kw: messages[-1]["content"],
        decode=lambda ids: "ab",
    )
    monkeypatch.setattr(agent_loop, "load_tokenizer", lambda *a, **k: tokenizer)
    monkeypatch.setattr(agent_loop, "load_processor", lambda *a, **k: None)

    async def exported(session, key):
        tokens = list(range(10, 10 + len(key)))
        return {
            "input_ids": [1, *tokens],
            "labels": [-100, *tokens],
            "logprobs": [0.0, *[-0.2] * len(tokens)],
            "routed_experts": None,
        }

    store = SimpleNamespace(
        keys=SimpleNamespace(remote=AsyncMock(side_effect=lambda session: ["".join(response)])),
        export_training_trace=SimpleNamespace(remote=exported),
    )
    monkeypatch.setattr(impl, "get_store", lambda: store)
    loop = AgentInSandboxLoopConfig(
        hf_checkpoint="unused", environment_provider={}, max_concurrent_samples=1, partial_rollout=True
    ).build_local()
    loop._environment_provider = Provider()
    sample = RolloutState(
        message=[],
        extra_fields={
            "rollout_item": AgentRolloutItem(
                id="neutral",
                data_source="unit",
                instruction="instruction.md",
                task_root=Path("/tmp"),
                pipeline={
                    "type": Runner,
                    "infer": {"type": SandboxStage, "pre": [Infer()]},
                    "validate": {"type": Judger, "name": "score", "stage": {"type": SandboxStage, "post": [Score()]}},
                },
            )
        },
    )
    first_group = asyncio.create_task(loop.generate_group([sample], enable_partial_rollout=True))
    try:
        await asyncio.wait_for(first.wait(), 5)
        await asyncio.wait_for(loop.prepare_pause(), 5)
        parked = (await first_group)[0]
        assert parked.status == Status.ABORTED
        assert parked.response_ids == [10]
        assert owned == ["open"]
        next_request.set()
        await asyncio.sleep(0)
        assert response == ["a"]
        parked.response_model_steps = [7]  # Producer's existing version stamping.
        await loop.resume()
        final = (await asyncio.wait_for(loop.generate_group([parked], enable_partial_rollout=True), 5))[0]
        assert final.status == Status.COMPLETED and final.reward == {"score": 1.0}
        assert final.response_ids == final.input_ids[len(final.prompt_ids or []):]
        assert final.response_model_steps == [7, 7]
        assert len(seen_sessions) == 1
        assert final.session_id == parked.session_id == seen_sessions[0]
        assert owned == ["open", "release"]
    finally:
        await loop.close()
    assert owned == ["open", "release", "close"]


@pytest.mark.asyncio
@pytest.mark.parametrize("failure", ["create", "health", "cleanup", "cancel", "timeout"])
async def test_failure_never_becomes_reward_and_resources_are_released(monkeypatch, failure):
    from xtuner.v1.rl.agent_loop.sandbox_agent_loop import RolloutStatus

    tokenizer = SimpleNamespace(eos_token=None, apply_chat_template=lambda *a, **k: "", decode=lambda ids: "")
    monkeypatch.setattr(agent_loop, "load_tokenizer", lambda *a, **k: tokenizer)
    monkeypatch.setattr(agent_loop, "load_processor", lambda *a, **k: None)
    entered = asyncio.Event()
    events = []

    class Provider:
        @asynccontextmanager
        async def open(self, item):
            events.append("create")
            if failure == "create":
                raise RuntimeError("create failed")

            async def check():
                if failure == "health":
                    raise RuntimeError("service disappeared")

            try:
                yield SimpleNamespace(check=check)
            finally:
                events.append("release")
                if failure == "cleanup":
                    raise RuntimeError("cleanup failed")

        async def close(self):
            events.append("close")

    async def run(item, environment):
        entered.set()
        if failure in ("cancel", "timeout"):
            async with asyncio.timeout(0.02 if failure == "timeout" else 60):
                await asyncio.Event().wait()
        item.reward, item.status = 1.0, RolloutStatus.COMPLETED
        item.artifacts["response_message"] = {"content": "done"}
        return item

    monkeypatch.setattr(impl, "_resolve_runner", lambda *a: SimpleNamespace(run=run))
    loop = AgentInSandboxLoopConfig(hf_checkpoint="unused", environment_provider={}, mode="eval").build_local()
    loop._environment_provider = Provider()
    state = RolloutState(
        message=[],
        input_ids=[90],
        labels=[90],
        response_ids=[90],
        logprobs=[-1],
        reward={"score": 1},
        extra_fields={"rollout_item": AgentRolloutItem(id="neutral", data_source="unit", instruction="i")},
    )
    task = asyncio.create_task(loop.generate_sample(state))
    if failure == "cancel":
        await entered.wait()
        task.cancel()
    [result] = await task
    await loop.close()
    assert result.status == (Status.ABORTED if failure == "cancel" else Status.FAILED)
    assert result.reward is None and result.input_ids is None and result.response_ids is None
    assert events == (["create", "close"] if failure == "create" else ["create", "release", "close"])
