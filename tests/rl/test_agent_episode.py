"""Token provenance across branching, compaction and repeated pause cycles."""

import asyncio
from types import SimpleNamespace

import pytest

from xtuner.v1.rl.agent_loop.sandbox_agent_loop.episode import PartialRun
from xtuner.v1.rl.agent_loop.sandbox_agent_loop.training_trace import TraceTokens


def trace(tokens, context=10, logprob=-0.5):
    return TraceTokens.from_trace(
        {
            "input_ids": [context, *tokens],
            "labels": [-100, *tokens],
            "logprobs": [0.0, *[logprob] * len(tokens)],
        }
    )


def test_same_output_in_different_contexts_is_not_shared():
    run = PartialRun(None)
    run.record([trace([2]), trace([2], context=11)])
    assert run.prefix == [2, 2]
    run.versions = [3, 4]
    assert run.segment_versions([trace([2], 11), trace([2, 3])]) == [[4], [3]]


def test_shared_prefix_gets_one_journal_entry_and_repeated_segment_is_safe():
    run = PartialRun(None)
    run.record([trace([2, 3]), trace([2, 4]), trace([2, 3])])
    assert run.prefix == [2, 3, 4]
    run.versions = [0, 0, 0]
    assert run.segment_versions([trace([2, 4, 5]), trace([2, 3])]) == [[0, 0], [0, 0]]


@pytest.mark.parametrize("changed", [[trace([9])], [trace([2], logprob=-0.7)], []])
def test_lost_or_overwritten_history_fails_closed(changed):
    run = PartialRun(None)
    run.record([trace([2])])
    with pytest.raises(ValueError, match="partial trace"):
        run.record(changed)


@pytest.mark.asyncio
async def test_replay_journal_must_match_tokens_and_versions():
    run = PartialRun(None)
    run.record([trace([2])])
    with pytest.raises(ValueError, match="journal changed"):
        await run.resume(SimpleNamespace(response_ids=[3], response_model_steps=[0]))
    with pytest.raises(ValueError, match="lack producer"):
        await run.resume(SimpleNamespace(response_ids=[2], response_model_steps=[]))


def test_inconsistent_shared_branch_is_rejected_before_recording():
    run = PartialRun(None)
    with pytest.raises(ValueError, match="branches disagree"):
        run.record([trace([2]), trace([2], logprob=-0.7)])
    assert run.prefix == []


@pytest.mark.asyncio
async def test_final_export_and_pause_cannot_race():
    run = PartialRun(None)
    run.running.set()
    async with run.completion():
        pause = asyncio.create_task(run.pause())
        await asyncio.sleep(0)
        assert not pause.done()
    await pause
    assert run.finished and not run.parked.is_set()


@pytest.mark.asyncio
async def test_expiry_coalesces_only_the_affected_episode(monkeypatch):
    from unittest.mock import AsyncMock

    from xtuner.v1.data_proto.rl_data import RolloutState, Status
    from xtuner.v1.rl.agent_loop.sandbox_agent_loop.episode import PartialRuns

    def sample(session, status, index=0):
        return RolloutState(
            message=[],
            status=status,
            session_id=session,
            extra_fields={"agent_session_id": session, "agent_trace_segment_index": index},
        )

    release = AsyncMock()
    monkeypatch.setattr("xtuner.v1.rl.agent_loop.sandbox_agent_loop.episode.release_existing_sessions", release)
    pool = PartialRuns(None, None, None, None)
    valid = [sample(2, Status.COMPLETED, i) for i in range(2)]
    new = [RolloutState(message=[]) for _ in range(2)]
    expired = [sample(1, Status.EXPIRED), sample(1, Status.COMPLETED, 1)]
    result = await pool.prepare_group([*valid, *expired, *new])
    assert len(result) == 5
    assert result[:2] == valid and result[3:] == new
    assert result[2].status == Status.EXPIRED
    assert "agent_session_id" not in result[2].extra_fields
    assert expired[0].extra_fields["agent_session_id"] == 1  # Input remains intact.
    release.assert_awaited_once_with(["1"])
    release.reset_mock()
    assert await pool.prepare_group(valid + new) == valid + new
    release.assert_not_awaited()


@pytest.mark.asyncio
async def test_attach_after_pause_closes_gate_before_agent_starts():
    from unittest.mock import AsyncMock

    run = PartialRun(None)
    await run.pause()
    controller = AsyncMock(return_value={"active": 0})
    await run.attach(controller, None)
    controller.assert_awaited_once_with("pause")


@pytest.mark.asyncio
async def test_diagnostic_write_failure_does_not_retain_completed_run(tmp_path):
    from unittest.mock import Mock

    from xtuner.v1.data_proto.rl_data import RolloutState, Status
    from xtuner.v1.rl.agent_loop.sandbox_agent_loop.episode import PartialRuns

    root = tmp_path / "not-a-directory"
    root.write_text("full or read-only artifact location")
    config = SimpleNamespace(artifact_root=root, max_concurrent_samples=1, mode="eval")

    async def execute(state, partial):
        state.status = Status.COMPLETED
        return [state]

    logger = Mock()
    pool = PartialRuns(config, execute, asyncio.Semaphore(1), logger)
    result = await pool.generate(RolloutState(message=[]))
    assert result[0].status == Status.COMPLETED
    assert not pool.runs and not pool.waiters
    logger.exception.assert_called_once()
    await pool.close()
    await pool.close()


@pytest.mark.asyncio
async def test_router_close_keeps_framework_cleanup_error_contract():
    from unittest.mock import AsyncMock

    from xtuner.v1.rl.agent_loop.agent_loop import RouterAgentLoop
    from xtuner.v1.rl.utils.async_utils import CleanupError

    failed = AsyncMock(side_effect=RuntimeError("unreleased environment"))
    other = AsyncMock()
    router = RouterAgentLoop.__new__(RouterAgentLoop)
    router.workers = [SimpleNamespace(close=SimpleNamespace(remote=method)) for method in (failed, other)]
    with pytest.raises(CleanupError, match="unreleased environment"):
        await router.close()
    failed.assert_awaited_once()
    other.assert_awaited_once()
