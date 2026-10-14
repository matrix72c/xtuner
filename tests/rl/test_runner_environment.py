"""Contract tests with real Runner/Judger stages and isolated fake resources."""

from __future__ import annotations

import asyncio
from pathlib import Path

import pytest

# Reuse the functional optional-lagent stand-ins and provider fixture.
from test_sandbox_pool import FakeProvider

from xtuner.v1.rl.agent_loop.sandbox_agent_loop import (
    AgentRolloutItem,
    Judger,
    PoolEnvironmentProvider,
    RolloutStatus,
    Runner,
    SandboxPool,
    SandboxSpec,
    SandboxStage,
)
from xtuner.v1.rl.agent_loop.sandbox_agent_loop import environment as impl


def item(**kwargs):
    return AgentRolloutItem(
        id="neutral", data_source="unit", instruction="instruction.md", task_root=Path("/tmp/neutral"), uid=1, **kwargs
    )


def provider(events):
    return PoolEnvironmentProvider(
        pool={
            "type": SandboxPool,
            "provider": {"type": FakeProvider, "events": events},
            "specs": {"main": SandboxSpec(image="neutral", workspace_path="/app")},
            "creates_per_sec": None,
            "max_attempts": 1,
        }
    )


class Infer:
    async def __call__(self, client, item, record):
        item.artifacts["infer_client"] = client


class Score:
    async def __call__(self, client, item, record):
        assert client is item.artifacts["infer_client"]
        record.score = 1.0


def runner(pre=None):
    return Runner(
        infer=SandboxStage(pre=[Infer()] if pre is None else pre),
        validate=Judger("neutral", SandboxStage(post=[Score()])),
    )


@pytest.mark.asyncio
async def test_real_runner_uses_owned_environment_without_releasing_it():
    source = provider([])
    sample = item()
    async with source.open(sample) as env:
        result = await runner().run(sample, env)
        assert result is sample
        assert result.status == RolloutStatus.COMPLETED
        assert result.reward == 1
        assert result.infer.sandbox_env_id == result.judgers["neutral"].sandbox_env_id
        client = await env.get("main")
        assert not any(event.startswith("delete:") for event in client.events)
        await env.check()
    assert any(event.startswith("delete:") for event in client.events)
    assert any(event.startswith("close:") for event in client.events)


@pytest.mark.asyncio
async def test_concurrent_episodes_never_share_pool_or_client():
    source = provider([])
    async with source.open(item()) as left, source.open(item()) as right:
        a, b = await asyncio.gather(left.get("main"), right.get("main"))
        assert a is not b
        assert left.pool is not right.pool


@pytest.mark.asyncio
async def test_failure_keeps_error_and_cleans_environment():
    class Fail:
        async def __call__(self, client, sample, record):
            sample.artifacts["client"] = client
            raise RuntimeError("neutral inference failed")

    sample = item()
    async with provider([]).open(sample) as env:
        result = await runner([Fail()]).run(sample, env)
        assert result.status == RolloutStatus.FAILED
        assert result.reward is None
        assert "neutral inference failed" in result.error.message
    assert any(event.startswith("delete:") for event in sample.artifacts["client"].events)


@pytest.mark.asyncio
async def test_cancellation_releases_environment_without_a_result():
    started = asyncio.Event()
    clients = []

    class Wait:
        async def __call__(self, client, sample, record):
            clients.append(client)
            started.set()
            await asyncio.Event().wait()

    async def run():
        sample = item()
        async with provider([]).open(sample) as env:
            await runner([Wait()]).run(sample, env)

    task = asyncio.create_task(run())
    await started.wait()
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    assert any(event.startswith("delete:") for event in clients[0].events)


@pytest.mark.asyncio
async def test_repeated_cancellation_joins_cleanup(monkeypatch):
    cleaning, finish = asyncio.Event(), asyncio.Event()
    released = []

    class Pool:
        async def release_all(self):
            cleaning.set()
            await finish.wait()
            released.append(True)

    monkeypatch.setattr(impl, "create_object", lambda config: Pool())

    async def run():
        async with PoolEnvironmentProvider({}).open(item()):
            pass

    task = asyncio.create_task(run())
    await cleaning.wait()
    task.cancel()
    await asyncio.sleep(0)
    task.cancel()
    await asyncio.sleep(0)
    assert not task.done()
    finish.set()
    with pytest.raises(asyncio.CancelledError):
        await task
    assert released == [True]


@pytest.mark.asyncio
async def test_cleanup_error_is_not_silently_lost(monkeypatch):
    class Pool:
        async def release_all(self):
            raise RuntimeError("cleanup failed")

    monkeypatch.setattr(impl, "create_object", lambda config: Pool())
    with pytest.raises(RuntimeError, match="cleanup failed"):
        async with PoolEnvironmentProvider({}).open(item()):
            pass


@pytest.mark.asyncio
async def test_overrides_do_not_mutate_template_or_other_episode():
    source = provider([])
    request = {"specs": {"main": SandboxSpec(image="override")}}
    async with source.open(item(environment=request)) as env:
        assert env.info("main").image == "override"
    async with source.open(item()) as env:
        assert env.info("main").image == "neutral"
        with pytest.raises(KeyError, match="unknown sandbox"):
            env.info("missing")
    assert request["specs"]["main"].image == "override"


@pytest.mark.asyncio
async def test_check_failure_raises_instead_of_returning_reward():
    async with provider([]).open(item()) as env:
        client = await env.get("main")

        async def unhealthy():
            return {"ok": False}

        client.health_check = unhealthy
        with pytest.raises(RuntimeError, match="unhealthy"):
            await env.check()


@pytest.mark.asyncio
async def test_entry_environment_uses_structured_transport_and_keeps_scope():
    from xtuner.v1.rl.agent_loop.sandbox_agent_loop.sandbox import exec_in

    calls = []

    class Client:
        async def execute_with_env(self, command, cwd, timeout, detach, *, env):
            calls.append((command, dict(env)))
            return {"return_code": 0}

    client = Client()
    private = "test value with 'quotes' and $(shell syntax)"
    await exec_in(client, 'printf "%s" "$KEY"', env={"KEY": private})
    await exec_in(client, "true")
    assert calls == [('printf "%s" "$KEY"', {"KEY": private}), ("true", {})]
    assert private not in calls[0][0]


@pytest.mark.asyncio
async def test_eval_infrastructure_failure_is_failed_without_reward():
    from xtuner.v1.data_proto.rl_data import RolloutState, Status
    from xtuner.v1.rl.agent_loop.sandbox_agent_loop import AgentInSandboxLoop

    loop = AgentInSandboxLoop.__new__(AgentInSandboxLoop)
    state = RolloutState(message=[], num_tokens=0)
    sample = item(status=RolloutStatus.FAILED)
    loop._fill_eval_rollout_state(state, sample)
    assert state.status == Status.FAILED
    assert state.reward is None
