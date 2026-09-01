import unittest
from unittest.mock import patch

from xtuner.v1.rl.rollout.session_server import (
    FMT_ANTHROPIC,
    FMT_OPENAI,
    _is_assistant_response,
    _should_record_response,
)


class TestAssistantResponseDetection(unittest.TestCase):
    def test_openai_requires_chat_assistant_message(self):
        self.assertTrue(
            _is_assistant_response(
                {"choices": [{"message": {"role": "assistant", "content": "ok"}}]},
                FMT_OPENAI,
            )
        )
        self.assertFalse(_is_assistant_response({"input_tokens": 42}, FMT_OPENAI))
        self.assertFalse(_is_assistant_response({"choices": []}, FMT_OPENAI))
        self.assertFalse(_is_assistant_response({"choices": [{"text": "completion"}]}, FMT_OPENAI))

    def test_anthropic_requires_message_envelope(self):
        self.assertTrue(
            _is_assistant_response(
                {"type": "message", "role": "assistant", "content": []},
                FMT_ANTHROPIC,
            )
        )
        self.assertFalse(_is_assistant_response({"input_tokens": 42}, FMT_ANTHROPIC))
        self.assertFalse(
            _is_assistant_response(
                {"type": "message", "role": "user", "content": []},
                FMT_ANTHROPIC,
            )
        )


class TestShouldRecordResponse(unittest.TestCase):
    def test_non_assistant_response_is_passed_through_without_recording(self):
        logger = unittest.mock.Mock()
        with patch("xtuner.v1.rl.rollout.session_server.get_logger", return_value=logger):
            should_record = _should_record_response(
                {"input_tokens": 42},
                FMT_OPENAI,
                "v1/messages/count_tokens",
            )

        self.assertFalse(should_record)
        logger.warning.assert_called_once()
        log_message = logger.warning.call_args.args[0]
        self.assertIn("path=/v1/messages/count_tokens", log_message)
        self.assertIn("response_keys=['input_tokens']", log_message)

    def test_assistant_response_is_recorded_without_warning(self):
        logger = unittest.mock.Mock()
        with patch("xtuner.v1.rl.rollout.session_server.get_logger", return_value=logger):
            should_record = _should_record_response(
                {"choices": [{"message": {"role": "assistant", "content": "ok"}}]},
                FMT_OPENAI,
                "v1/chat/completions",
            )

        self.assertTrue(should_record)
        logger.warning.assert_not_called()


if __name__ == "__main__":
    unittest.main()
