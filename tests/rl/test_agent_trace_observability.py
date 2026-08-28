import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

from xtuner.v1.data_proto.rl_data import RolloutState
from xtuner.v1.rl.agent_loop.sandbox_agent_loop.agent_in_sandbox_loop import (
    AgentInSandboxLoop,
    _summarize_train_trace_segments,
)
from xtuner.v1.rl.agent_loop.sandbox_agent_loop.schemas import AgentRolloutItem, RolloutStatus


def _messages(*roles: str) -> list[dict[str, str]]:
    return [{"role": role, "content": f"{role}-{index}"} for index, role in enumerate(roles)]


def test_summarize_train_trace_segments_classifies_prefix_growth():
    prefix = _messages("user", "assistant")
    complete = prefix + _messages("tool", "assistant")

    summary = _summarize_train_trace_segments(
        [(prefix, []), (complete, [])],
        artifact_record_count=3,
    )

    assert summary == {
        "artifact_record_count": 3,
        "train_segment_count": 2,
        "record_shape": "linear_prefix_or_duplicate",
        "message_counts": [2, 4],
        "assistant_message_counts": [1, 2],
        "tool_variant_count": 1,
    }


def test_summarize_train_trace_segments_distinguishes_tools_and_rewrites():
    first = _messages("user", "assistant")
    rewritten = _messages("user", "assistant")
    rewritten[-1]["content"] = "rewritten"

    tools_changed = _summarize_train_trace_segments(
        [(first, [{"name": "one"}]), (first, [{"name": "two"}])],
        artifact_record_count=2,
    )
    branch_or_rewrite = _summarize_train_trace_segments(
        [(first, []), (rewritten, [])],
        artifact_record_count=2,
    )

    assert tools_changed["record_shape"] == "tools_changed"
    assert tools_changed["tool_variant_count"] == 2
    assert branch_or_rewrite["record_shape"] == "branch_or_rewrite"


def test_build_rollout_states_logs_each_materialized_prefix_and_summary():
    prefix = _messages("user", "assistant")
    complete = prefix + _messages("tool", "assistant")
    item = AgentRolloutItem(
        id="sample",
        data_source="test",
        instruction="instruction.txt",
        status=RolloutStatus.COMPLETED,
        reward=1.0,
        artifacts={
            "messages": [
                {"messages": prefix, "tools": []},
                {"messages": complete, "tools": []},
            ],
            "response_message": {"role": "assistant", "content": "done"},
        },
    )
    rollout_state = RolloutState(
        group_id=3,
        rollout_id=7,
        session_id=123,
        message=[{"role": "user", "content": "task"}],
        num_tokens=1,
        extra_fields={},
    )
    trace_store = SimpleNamespace(
        export_training_trace=SimpleNamespace(
            remote=AsyncMock(
                side_effect=[
                    {
                        "input_ids": [1, 2],
                        "labels": [-100, 2],
                        "logprobs": [0.0, -0.1],
                        "routed_experts": None,
                        "_trace_metrics": {
                            "action_token_count": 1,
                            "expert_ref_count": 0,
                            "expert_payload_bytes": 0,
                        },
                    },
                    {
                        "input_ids": [1, 2, 3, 4],
                        "labels": [-100, -100, 3, 4],
                        "logprobs": [0.0, 0.0, -0.1, -0.2],
                        "routed_experts": [object(), object()],
                        "_trace_metrics": {
                            "action_token_count": 2,
                            "expert_ref_count": 2,
                            "expert_payload_bytes": 96,
                        },
                    },
                ]
            )
        )
    )
    loop = AgentInSandboxLoop.__new__(AgentInSandboxLoop)
    loop.mode = "train"
    loop.tokenizer = MagicMock()
    loop.tokenizer.apply_chat_template.side_effect = ["prompt-one\n", "prompt-two\n"]
    loop.tokenizer.decode.side_effect = ["first", "second"]

    with (
        patch(
            "xtuner.v1.rl.agent_loop.sandbox_agent_loop.agent_in_sandbox_loop.get_store",
            return_value=trace_store,
        ),
        patch(
            "xtuner.v1.rl.agent_loop.sandbox_agent_loop.agent_in_sandbox_loop._log_agent_trace_metrics"
        ) as log_metrics,
    ):
        states = asyncio.run(loop._build_rollout_states(rollout_state, item))

    assert len(states) == 2
    assert states[0].input_ids == [1, 2]
    assert states[1].input_ids == [1, 2, 3, 4]
    assert trace_store.export_training_trace.remote.await_count == 2

    calls = [(call.args[0], call.args[1]) for call in log_metrics.call_args_list]
    assert calls[0][0] == "AgentTraceRecords"
    assert calls[-1][0] == "AgentTraceExport"
    assert [payload.get("status") for kind, payload in calls if kind == "AgentTraceStateCopy"] == [
        "start",
        "ok",
        "start",
        "ok",
    ]
    assert [payload.get("status") for kind, payload in calls if kind == "AgentTraceExport"] == [
        "start",
        "ok",
        "start",
        "ok",
    ]
    completed_exports = [
        payload for kind, payload in calls if kind == "AgentTraceExport" and payload["status"] == "ok"
    ]
    assert [payload["token_count"] for payload in completed_exports] == [2, 4]
    assert [payload["action_token_count"] for payload in completed_exports] == [1, 2]
    assert [payload["expert_ref_count"] for payload in completed_exports] == [0, 2]
    assert [payload["expert_payload_bytes"] for payload in completed_exports] == [0, 96]
