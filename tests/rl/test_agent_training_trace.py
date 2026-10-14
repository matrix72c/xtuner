"""Query rendering must agree with the pinned server's actual trace writes."""

import json
from copy import deepcopy
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from xtuner.v1.rl.agent_loop.sandbox_agent_loop.training_trace import render_trace_key
from xtuner.v1.rl.rollout.session_server import FMT_OPENAI, SessionServer


def test_partial_versions_cover_tool_tokens_without_training_on_them():
    from xtuner.v1.data_proto.rl_data import RolloutState, Status, write_train_meta
    from xtuner.v1.rl.agent_loop.sandbox_agent_loop.training_trace import (
        apply_training_trace,
        response_prefix_versions,
    )

    state = RolloutState(message=[], prompt_ids=[1, 2], status=Status.COMPLETED)
    trace = {
        "input_ids": [1, 2, 10, 30, 31, 11, 32, 12],
        "labels": [-100, -100, 10, -100, -100, 11, -100, 12],
        "logprobs": [0.0, 0.0, -0.2, 0.0, 0.0, -0.3, 0.0, -0.4],
    }
    apply_training_trace(state, trace)
    assert state.response_ids == [10, 30, 31, 11, 32, 12]
    previous = response_prefix_versions(state, [7, 8])
    assert previous == [7, 7, 7, 8, 8]
    state.response_model_steps = previous + [9]
    write_train_meta(state)
    assert [token for token in state.labels if token != -100] == [10, 11, 12]
    assert len(state.response_model_steps) == len(state.response_ids)
    from xtuner.v1.data_proto.rl_data import calculate_group_effective_response_masks

    masks = calculate_group_effective_response_masks([state], current_train_step=10, token_stale_threshold=2)
    assert masks == [[0, 0, 0, 1, 0, 1]]
    assert state.labels == [-100, -100, -100, -100, -100, 11, -100, 12]


class Tokenizer:
    eos_token = "<eos>"

    def apply_chat_template(self, messages, *, tools, tokenize, add_generation_prompt):
        value = json.dumps(tools) + "".join(
            f"<{m['role']}>" + json.dumps(m, sort_keys=True) + self.eos_token for m in messages
        )
        return value + ("<assistant>" if add_generation_prompt else "\n\n")

    def encode(self, text, **kwargs):
        return list(text.encode())


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "content",
    [
        "Invalid tool input echoed <eos> inside an error",
        [{"type": "text", "text": "Invalid <eos> input"}, {"type": "text", "text": "retry <eos>"}],
    ],
)
async def test_query_matches_real_session_server_write_without_mutating_artifacts(content):
    tokenizer = Tokenizer()
    messages = [
        {"role": "user", "content": "solve"},
        {
            "role": "assistant",
            "tool_calls": [
                {
                    "id": "call",
                    "type": "function",
                    "function": {"name": "Bash", "arguments": {"command": "echo '<eos>'"}},
                }
            ],
        },
        {"role": "tool", "tool_call_id": "call", "content": content},
        {"role": "assistant", "content": "retry finished<eos>"},
    ]
    original = deepcopy(messages)
    tools = [{"type": "function", "function": {"name": "Bash"}}]
    server = SessionServer.__new__(SessionServer)
    server.tokenizer, server.stop_word = tokenizer, tokenizer.eos_token
    server.enable_return_routed_experts = False
    insert = AsyncMock()
    server.store = SimpleNamespace(
        search=SimpleNamespace(remote=AsyncMock(return_value=("", []))),
        insert=SimpleNamespace(remote=insert),
    )
    # Synthetic token values are confined to this unit test; verify the real
    # server's rendering path instead of duplicating its scrub in expectations.
    await server.on_response(
        {"choices": [{"message": deepcopy(messages[-1]), "output_ids": [7], "output_token_logprobs": [[-0.5, 7]]}]},
        FMT_OPENAI,
        {"session_id": "s", "messages": deepcopy(messages[:-1]), "tools": tools},
    )
    stored_key = insert.call_args.kwargs["key"]
    assert render_trace_key(tokenizer, messages, tools) == stored_key
    assert messages == original
    assert "echo '<eos>'" in stored_key  # Arguments are not content; preserve them.
    assert "Invalid <eos>" not in stored_key


def test_context_sensitive_template_retains_earlier_response_branches():
    from xtuner.v1.rl.agent_loop.sandbox_agent_loop.training_trace import training_segments

    class ContextualTokenizer:
        # The real Qwen template drops reasoning before the newest user turn.
        def apply_chat_template(self, messages, **kwargs):
            last_user = max(i for i, m in enumerate(messages) if m["role"] == "user")
            return "".join(
                (m.get("reasoning_content", "") if i > last_user else "") + m["content"]
                for i, m in enumerate(messages)
            )

    tokenizer = ContextualTokenizer()
    early = [{"role": "user", "content": "Q"}, {"role": "assistant", "content": "A", "reasoning_content": "think"}]
    late = [*early, {"role": "user", "content": "notification"}, {"role": "assistant", "content": "B"}]
    branches = training_segments(tokenizer, ({"messages": late, "tools": []},))
    assert [key for key, _, _ in branches] == ["QthinkA", "QAnotificationB"]
    # Supplying the original record too does not duplicate the old generation.
    assert (
        training_segments(tokenizer, ({"messages": early, "tools": []}, {"messages": late, "tools": []})) == branches
    )


def test_rendered_prefix_dedup_preserves_only_maximal_branches():
    from xtuner.v1.rl.agent_loop.sandbox_agent_loop.training_trace import training_segments

    class Tokenizer:
        def apply_chat_template(self, messages, **kwargs):
            return "".join(m["content"] for m in messages)

    segments = [
        dict(messages=[dict(role="assistant", content="A"), dict(role="assistant", content="B")], tools=[]),
        dict(messages=[dict(role="assistant", content="A"), dict(role="assistant", content="C")], tools=[]),
    ]
    assert [key for key, _, _ in training_segments(Tokenizer(), segments)] == ["AB", "AC"]
