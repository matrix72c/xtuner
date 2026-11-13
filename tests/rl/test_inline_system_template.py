"""Inline system policy belongs to the model template, not trace normalization."""

import copy

import pytest
from tokenizers import Tokenizer
from tokenizers.models import WordLevel

from transformers import PreTrainedTokenizerFast
from xtuner.v1.rl.rollout.chat_template import canonicalize_messages_for_chat_template


class TestInlineSystemTemplate:
    @pytest.mark.parametrize("supports_inline", [False, True])
    def test_template_owns_inline_system_policy(self, supports_inline: bool) -> None:
        template = (
            "{% for message in messages %}"
            "{% if message.role == 'system' and not loop.first and reject_inline %}"
            "{{ raise_exception('System message must be at the beginning.') }}"
            "{% endif %}{{ message.role }}:{{ message.content }};{% endfor %}"
        ).replace("reject_inline", "false" if supports_inline else "true")
        tokenizer = PreTrainedTokenizerFast(
            tokenizer_object=Tokenizer(WordLevel({"<unk>": 0}, unk_token="<unk>")),
            unk_token="<unk>",
            chat_template=template,
        )
        messages = [
            {"role": "system", "content": "initial"},
            {"role": "user", "content": "task"},
            {"role": "assistant", "content": "first answer"},
            {"role": "system", "content": "later instruction"},
            {"role": "user", "content": "continue"},
        ]
        original = copy.deepcopy(messages)
        projected = canonicalize_messages_for_chat_template(messages)
        assert projected == original
        assert projected is not messages
        if supports_inline:
            assert tokenizer.apply_chat_template(projected, tokenize=False) == (
                "system:initial;user:task;assistant:first answer;system:later instruction;user:continue;"
            )
        else:
            with pytest.raises(Exception, match="System message must be at the beginning"):
                tokenizer.apply_chat_template(projected, tokenize=False)
        assert messages == original

    def test_content_blocks_preserve_role_position_and_content(self) -> None:
        messages = [
            {"role": "system", "content": "initial"},
            {"role": "user", "content": "task"},
            {"role": "system", "content": [{"type": "text", "text": "later instruction"}]},
        ]
        original = copy.deepcopy(messages)
        assert canonicalize_messages_for_chat_template(messages) == original
        assert messages == original
