"""ParseJudgerStdout reward-payload contract tests.

Covers the ``(score, criteria, metadata)`` parse contract and the
``record.metadata['reward']`` payload that feeds ``_extract_reward_payload``
on the ``AgentRolloutItem`` -> ``RolloutState`` conversion.
"""

from __future__ import annotations

import asyncio
import json

from xtuner.v1.rl.agent_loop.sandbox_agent_loop.hooks import ParseJudgerStdout
from xtuner.v1.rl.agent_loop.sandbox_agent_loop.schemas import (
    AgentRolloutItem,
    RolloutError,
    RolloutStatus,
    StageRecord,
    StageResult,
)


def _item() -> AgentRolloutItem:
    return AgentRolloutItem(id="t1", data_source="test", instruction="instruction.md")


def _record(stdout: str, return_code: int = 0) -> StageRecord:
    record = StageRecord()
    record.entry_result = StageResult(stdout=stdout, return_code=return_code)
    return record


def _run(hook: ParseJudgerStdout, record: StageRecord, item: AgentRolloutItem) -> None:
    asyncio.run(hook.__call__(client=None, item=item, record=record))


class TestParseJudgerStdout:
    def test_full_payload_metadata_lands_in_reward_dict(self):
        payload = {
            "judger_name": "process_grader",
            "total": 0.28,
            "criteria": {"process": {"score": 0.28, "weight": 1.0}},
            "metadata": {
                "outcome": 0,
                "test_rc": 1,
                "source": "process",
                "parts": {"S1_recon": 0.06, "S2_attempt": 0.1, "S3_effect": 0.08, "S4_proof": 0.04},
            },
        }
        record = _record(json.dumps(payload))
        _run(ParseJudgerStdout("process_grader"), record, _item())

        assert record.score == 0.28
        assert record.metadata["criteria"] == payload["criteria"]
        reward = record.metadata["reward"]
        assert reward["outcome"] == 0
        assert reward["test_rc"] == 1
        assert reward["parts"] == payload["metadata"]["parts"]
        assert reward["criteria"] == payload["criteria"]
        # score is filled by the rollout-item -> RolloutState conversion, not here
        assert "score" not in reward

    def test_payload_without_metadata(self):
        payload = {"total": 1.0, "criteria": {"outcome": {"score": 1.0, "weight": 1.0}}}
        record = _record(json.dumps(payload))
        _run(ParseJudgerStdout("rule_grader"), record, _item())

        assert record.score == 1.0
        assert record.metadata["reward"] == {"criteria": payload["criteria"]}

    def test_noise_before_json_line_is_tolerated(self):
        payload = {"total": 0.5, "metadata": {"test_count": 12}}
        record = _record(f"some stderr-ish noise\n{json.dumps(payload)}\ntrailing line")
        _run(ParseJudgerStdout("rule_grader"), record, _item())

        assert record.score == 0.5
        assert record.metadata["reward"]["test_count"] == 12

    def test_legacy_total_score_shape(self):
        payload = {"total_score": 0.7, "test_a": 1.0, "test_b": 0.0}
        record = _record(json.dumps(payload))
        _run(ParseJudgerStdout("legacy"), record, _item())

        assert record.score == 0.7
        assert record.metadata["reward"] == {
            "criteria": {"test_a": {"score": 1.0}, "test_b": {"score": 0.0}}
        }

    def test_non_dict_metadata_fails_the_stage(self):
        payload = {"total": 0.5, "metadata": [1, 2, 3]}
        record = _record(json.dumps(payload))
        _run(ParseJudgerStdout("bad"), record, _item())

        assert record.score is None
        assert record.status.value == "failed"
        assert isinstance(record.error, RolloutError)
        assert record.error.category == "judger_parse"

    def test_missing_reward_key_keeps_failure_path(self):
        record = _record("not json at all")
        item = _item()
        _run(ParseJudgerStdout("bad"), record, item)

        assert record.score is None
        assert item.status == RolloutStatus.PENDING  # hook does not touch the item
        assert record.error is not None
