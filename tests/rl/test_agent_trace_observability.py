from xtuner.v1.rl.agent_loop.sandbox_agent_loop.agent_in_sandbox_loop import _summarize_train_trace_segments


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
