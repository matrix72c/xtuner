import copy

import pytest
from jinja2.exceptions import TemplateError

from xtuner.v1.rl.rollout.chat_template import apply_chat_template_with_system_fallback


_SYSTEM_MESSAGE_POSITION_ERROR = "System message must be at the beginning."


class _RecordingTokenizer:
    def __init__(self, *, reject_late_system: bool = False, error_message: str | None = None):
        self.reject_late_system = reject_late_system
        self.error_message = error_message
        self.calls: list[tuple[list[dict], dict]] = []

    def apply_chat_template(self, messages: list[dict], **kwargs):
        self.calls.append((copy.deepcopy(messages), copy.deepcopy(kwargs)))
        has_late_system = any(message.get("role") == "system" for message in messages[1:])
        if self.error_message is not None:
            raise TemplateError(self.error_message)
        if self.reject_late_system and has_late_system:
            raise TemplateError(_SYSTEM_MESSAGE_POSITION_ERROR)
        return messages


class TestChatTemplateSystemFallback:
    def test_permissive_template_keeps_original_message_order(self):
        tokenizer = _RecordingTokenizer()
        messages = [
            {"role": "user", "content": "question"},
            {"role": "system", "content": "late instruction"},
        ]

        rendered = apply_chat_template_with_system_fallback(tokenizer, messages, tokenize=False)

        assert rendered == messages
        assert tokenizer.calls == [(messages, {"tokenize": False})]

    def test_known_qwen_error_merges_textual_system_turns_and_retries(self):
        tokenizer = _RecordingTokenizer(reject_late_system=True)
        messages = [
            {"role": "system", "content": "first"},
            {"role": "user", "content": "question"},
            {
                "role": "system",
                "content": [
                    {"type": "text", "text": "second"},
                    {"type": "text", "text": " instruction"},
                ],
            },
            {"role": "assistant", "content": "answer"},
        ]

        rendered = apply_chat_template_with_system_fallback(
            tokenizer, messages, tools=[{"type": "function"}], tokenize=False
        )

        assert rendered == [
            {"role": "system", "content": "first\n\nsecond instruction"},
            {"role": "user", "content": "question"},
            {"role": "assistant", "content": "answer"},
        ]
        assert len(tokenizer.calls) == 2
        assert tokenizer.calls[0][0] == messages
        assert (
            tokenizer.calls[0][1]
            == tokenizer.calls[1][1]
            == {
                "tools": [{"type": "function"}],
                "tokenize": False,
            }
        )

    def test_different_template_error_is_not_swallowed(self):
        tokenizer = _RecordingTokenizer(error_message="Unexpected message role.")
        messages = [
            {"role": "user", "content": "question"},
            {"role": "system", "content": "late instruction"},
        ]

        with pytest.raises(TemplateError, match="Unexpected message role"):
            apply_chat_template_with_system_fallback(tokenizer, messages, tokenize=False)

        assert len(tokenizer.calls) == 1

    def test_known_error_without_late_system_is_not_swallowed(self):
        tokenizer = _RecordingTokenizer(error_message=_SYSTEM_MESSAGE_POSITION_ERROR)
        messages = [{"role": "system", "content": "first"}, {"role": "user", "content": "question"}]

        with pytest.raises(TemplateError, match="System message must be at the beginning"):
            apply_chat_template_with_system_fallback(tokenizer, messages, tokenize=False)

        assert len(tokenizer.calls) == 1

    def test_non_text_system_content_is_rejected_only_during_fallback(self):
        tokenizer = _RecordingTokenizer(reject_late_system=True)
        messages = [
            {"role": "user", "content": "question"},
            {"role": "system", "content": [{"type": "image", "url": "example"}]},
        ]

        with pytest.raises(ValueError, match="only text blocks"):
            apply_chat_template_with_system_fallback(tokenizer, messages, tokenize=False)

    def test_canonicalization_and_fallback_do_not_mutate_input(self):
        tokenizer = _RecordingTokenizer(reject_late_system=True)
        messages = [
            {"role": "user", "content": "question"},
            {"role": "system", "content": "late instruction"},
            {
                "role": "assistant",
                "content": "",
                "tool_calls": [{"function": {"name": "search", "arguments": '{"query": "value"}'}}],
            },
        ]
        original_messages = copy.deepcopy(messages)

        apply_chat_template_with_system_fallback(tokenizer, messages, tokenize=False)

        assert messages == original_messages
        assert tokenizer.calls[0][0][2]["tool_calls"][0]["function"]["arguments"] == {"query": "value"}
