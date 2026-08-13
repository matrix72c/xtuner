import copy

from xtuner.v1.rl.rollout.chat_template import canonicalize_messages_for_chat_template


class TestCanonicalizeMessagesForChatTemplate:
    def test_keeps_messages_without_late_system_unchanged(self):
        messages = [
            {"role": "system", "content": "instruction"},
            {"role": "user", "content": "question"},
        ]

        canonical_messages = canonicalize_messages_for_chat_template(messages)

        assert canonical_messages == messages
        assert canonical_messages is not messages

    def test_merges_system_messages_at_front(self):
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

        canonical_messages = canonicalize_messages_for_chat_template(messages)

        assert canonical_messages == [
            {"role": "system", "content": "first\n\nsecond instruction"},
            {"role": "user", "content": "question"},
            {"role": "assistant", "content": "answer"},
        ]

    def test_canonicalization_does_not_mutate_input(self):
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

        canonical_messages = canonicalize_messages_for_chat_template(messages)

        assert messages == original_messages
        assert canonical_messages[2]["tool_calls"][0]["function"]["arguments"] == {"query": "value"}
