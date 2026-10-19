from __future__ import annotations

import asyncio
import copy
import importlib
import json
import math
import os
import traceback
import uuid
from contextlib import nullcontext
from types import SimpleNamespace
from typing import Any, Literal

from lagent.utils import create_object
from pydantic import Field

from xtuner.v1.data_proto.rl_data import RolloutState, SampleParams, Status, write_train_meta
from xtuner.v1.rl.judger import Judger
from xtuner.v1.rl.rollout import RolloutController
from xtuner.v1.rl.trace.rollout_api import trace_rollout_endpoint

from ...rollout.trace_store import get_store
from ..agent_loop import AgentLoop, AgentLoopConfig, maybe_filter_invalid_sample
from .environment import EnvironmentProvider
from .episode import EpisodeContext, PartialRuns, aborted
from .schemas import AgentRolloutItem, RolloutStatus
from .training_trace import (
    TracePending,
    TraceTokens,
    apply_training_trace,
    response_prefix_versions,
    training_segments,
)


_MISSING = object()


def _import_from_path(path: str) -> Any:
    module_name, _, attr = path.rpartition(".")
    if not module_name or not attr:
        raise ValueError(f"Invalid import path: {path!r}. Expected 'module.attr'.")
    module = importlib.import_module(module_name)
    return getattr(module, attr)


def _inject_session_id(runner_cfg: dict[str, Any], session_id: str) -> None:
    for entry in runner_cfg.get("infer", {}).get("entries", []):
        if isinstance(entry, dict) and entry.get("name") == "start_agent_daemon":
            entry.setdefault("env", {})["XTUNER_SESSION_ID"] = session_id


def _resolve_runner(pipeline: Any, session_id: str) -> Any:
    if isinstance(pipeline, str):
        pipeline = _import_from_path(pipeline)
    if isinstance(pipeline, dict):
        runner_cfg = copy.deepcopy(pipeline)
        _inject_session_id(runner_cfg, session_id)
        return create_object(runner_cfg)
    return pipeline


def _drop_failed_train_samples(samples: list[RolloutState], mode: Literal["train", "eval"]) -> list[RolloutState]:
    if mode != "train":
        return samples
    filtered = [sample for sample in samples if sample.status != Status.FAILED]
    return filtered or samples


def _validate_trace_segment(
    segment: dict[str, Any],
    *,
    require_tools: bool = False,
) -> tuple[list[dict[str, Any]], Any]:
    if "messages" not in segment:
        raise ValueError("Agent messages trace segment must contain messages.")
    messages = segment["messages"]
    tools = segment.get("tools", _MISSING)
    if not isinstance(messages, list):
        raise TypeError("Agent messages trace must be a list.")
    if not all(isinstance(message, dict) for message in messages):
        raise TypeError("Agent messages trace must contain only dict messages.")
    if require_tools and tools is _MISSING:
        raise ValueError("Agent messages trace segment must contain tools.")
    return messages, None if tools is _MISSING else tools


def _trim_to_last_assistant_turn(messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
    for index in range(len(messages) - 1, -1, -1):
        if messages[index].get("role") == "assistant":
            return messages[: index + 1]
    return []


def _load_train_trace_segments(artifacts: dict[str, Any]) -> list[tuple[list[dict[str, Any]], Any]]:
    trace = _load_messages_artifact(artifacts)
    if not trace:
        raise ValueError("Agent artifacts must contain at least one messages trace segment.")
    segments: list[tuple[list[dict[str, Any]], Any]] = []
    for segment in trace:
        if not isinstance(segment, dict):
            raise TypeError("Agent messages trace segment must be a dict.")
        messages, tools = _validate_trace_segment(segment, require_tools=True)
        messages = _trim_to_last_assistant_turn(messages)
        if messages:
            segments.append((messages, tools))
    return segments


def _load_eval_trace_segment(artifacts: dict[str, Any]) -> tuple[list[dict[str, Any]], Any]:
    trace = _load_messages_artifact(artifacts, required=False)
    if not trace:
        return [], None
    segment = trace[-1]
    if not isinstance(segment, dict):
        raise TypeError("Agent messages trace segment must be a dict.")
    messages = segment.get("messages") or []
    tools = segment.get("tools")
    if not isinstance(messages, list):
        raise TypeError("Agent messages trace must be a list.")
    if not all(isinstance(message, dict) for message in messages):
        raise TypeError("Agent messages trace must contain only dict messages.")
    return messages, tools


def _load_messages_artifact(artifacts: dict[str, Any], *, required: bool = True) -> list[dict[str, Any]] | None:
    if "messages" not in artifacts:
        if required:
            raise ValueError("Agent artifacts must contain 'messages'.")
        return None
    trace = artifacts["messages"]
    if not isinstance(trace, list):
        raise TypeError("Agent artifact 'messages' must be a list.")
    return trace


def _response_message(artifacts: dict[str, Any], *, required: bool) -> dict[str, Any]:
    if "response_message" not in artifacts:
        if required:
            raise KeyError("Agent artifacts must contain 'response_message'.")
        return {}
    response_message = artifacts["response_message"]
    if not isinstance(response_message, dict):
        raise TypeError("Agent artifact 'response_message' must be a dict.")
    return response_message


def _response_text(response_message: dict[str, Any]) -> str:
    content = response_message.get("content")
    if content is None:
        return ""
    if not isinstance(content, str):
        raise TypeError("Agent response_message['content'] must be a string.")
    return content


def _finish_info(response_message: dict[str, Any]) -> dict[str, Any] | None:
    finish_info = response_message.get("finish_info")
    if finish_info is None:
        return None
    if not isinstance(finish_info, dict):
        raise TypeError("Agent response_message['finish_info'] must be a dict.")
    return finish_info or None


def _to_json_safe(value: Any) -> Any:
    return json.loads(json.dumps(value, ensure_ascii=False, default=str))


def _selected_agent(item: AgentRolloutItem) -> dict[str, Any] | None:
    if item.infer.agent is None:
        return None
    return item.infer.agent.model_dump(mode="json")


def _count_tool_turns(messages: list[dict[str, Any]]) -> int:
    return sum(
        1
        for message in messages
        if isinstance(message, dict) and isinstance(message.get("tool_calls"), list) and message["tool_calls"]
    )


def _extract_reward_payload(item: AgentRolloutItem) -> dict[str, Any] | None:
    for record in item.judgers.values():
        reward = record.metadata.get("reward")
        if isinstance(reward, dict):
            payload = dict(reward)
            if item.reward is not None:
                payload["score"] = item.reward
            return payload
    if item.reward is not None:
        return {"score": item.reward}
    return None


def _task_result_error(item: AgentRolloutItem) -> str | None:
    """Only a completed, finite verifier score is a task outcome.

    A task-budget timeout is scoreable only after the harness confirms the
    agent stopped. Transport timeouts and adapter errors remain unscored.
    """
    if item.error is not None:
        return f"{item.error.stage}/{item.error.category}: {item.error.message}"
    if item.status != RolloutStatus.COMPLETED:
        return f"Agent attempt ended with status {item.status.value}"
    response = _response_message(item.artifacts, required=True)
    extra = response.get("extra_info") or {}
    if response.get("error") or (isinstance(extra, dict) and extra.get("error")):
        return "Agent adapter failed; verifier output is not a valid task result"
    reason = response.get("finish_reason")
    if reason == "timeout":
        info = _finish_info(response) or {}
        if info.get("termination_reason") != "task_budget_exhausted" or info.get("agent_stopped") is not True:
            return "TimeoutError: task-budget exhaustion and agent termination were not confirmed"
    elif reason in {"error", "cancelled", "abort"}:
        return f"Agent attempt ended with {reason}"
    if item.reward is None or not math.isfinite(item.reward):
        return "Verifier did not produce a finite task score"
    return None


class AgentInSandboxLoopConfig(AgentLoopConfig):
    """Run a sandbox agent runner from ``RolloutState.extra_fields``.

    The tb2-rl tokenize function stores an :class:`AgentRolloutItem` in
    ``rollout_state.extra_fields["rollout_item"]``.  This loop executes that
    item's sandbox pipeline, then converts the resulting task reward and agent
    transcript back into the standard ``RolloutState`` fields consumed by the
    replay buffer/trainer.
    """

    environment_provider: dict[str, Any] | str
    max_concurrent_samples: int = Field(default=32, gt=0)
    partial_rollout: bool = False
    partial_pause_timeout: float = Field(default=240, gt=0, lt=280)
    artifact_root: str | None = None
    mode: Literal["train", "eval"] = "train"
    requires_rollout_proxy: bool = True

    def build(self, *args, **kwargs):
        if self.partial_rollout and self.cpu_resources is not None:
            raise ValueError("live partial episodes require a local loop owner; worker migration is unsupported")
        if self.partial_rollout and self.mode != "train":
            raise ValueError("partial rollout is a training option")
        return super().build(*args, **kwargs)

    def build_local(
        self, rollout_controller: RolloutController | None = None, judger: Judger | None = None, logger=None
    ) -> AgentInSandboxLoop:
        return AgentInSandboxLoop(
            rollout_ctl=rollout_controller,
            hf_checkpoint=self.hf_checkpoint,
            sample_params=self.sample_params,
            judger=judger,
            logger=logger,
            max_concurrent_samples=self.max_concurrent_samples,
            mode=self.mode,
            environment_provider=self.environment_provider,
            partial_rollout=self.partial_rollout,
            partial_pause_timeout=self.partial_pause_timeout,
            artifact_root=self.artifact_root,
        )


class AgentInSandboxLoop(AgentLoop):
    def __init__(
        self,
        rollout_ctl: RolloutController | None = None,
        hf_checkpoint: str | None = None,
        sample_params: SampleParams | None = None,
        judger: Judger | None = None,
        logger=None,
        max_concurrent_samples: int | None = None,
        mode: Literal["train", "eval"] = "train",
        *,
        environment_provider: dict[str, Any] | str,
        partial_rollout: bool = False,
        partial_pause_timeout: float = 240,
        artifact_root: str | None = None,
    ):
        if hf_checkpoint is None:
            raise ValueError("hf_checkpoint must be provided for AgentInSandboxLoop.")
        super().__init__(rollout_ctl, sample_params, hf_checkpoint, judger, logger)
        self.mode = mode
        self._environment_config = environment_provider
        self._environment_provider: EnvironmentProvider | None = None
        self._closing = False
        self._close_task: asyncio.Task | None = None
        self.config = SimpleNamespace(
            max_concurrent_samples=max_concurrent_samples or 32,
            mode=mode,
            partial_rollout=partial_rollout,
            partial_pause_timeout=partial_pause_timeout,
            artifact_root=artifact_root,
        )
        self._slots = asyncio.Semaphore(self.config.max_concurrent_samples)
        self._active: set[asyncio.Task] = set()
        self._paused = False
        self._admission = asyncio.Event()
        self._admission.set()
        self._pause_task: asyncio.Task | None = None
        self.partial = PartialRuns(self.config, self._execute, self._slots, self.logger)

    async def generate_group(self, rollout_state: list[RolloutState], **kwargs) -> list[RolloutState]:
        partial = kwargs.get("enable_partial_rollout", False)
        if partial and not self.config.partial_rollout:
            raise ValueError("partial rollout requires an explicitly configured resumable runner")
        if self._paused and self._admission.is_set():
            return [
                state if partial and state.status == Status.COMPLETED else aborted(state) for state in rollout_state
            ]
        if partial:
            rollout_state = await self.partial.prepare_group(rollout_state)
        slots = self._slots
        # XTuner versions supporting teacher scoring install this optional hook
        # through AgentLoopConfig.build.
        teacher = getattr(self, "_teacher_scorer", None)

        async def generate(state: RolloutState) -> list[RolloutState]:
            try:
                if partial and state.status == Status.COMPLETED:
                    return [state]
                await self._admission.wait()
                state.sample_params = state.sample_params or self.sample_params
                if partial:
                    samples = await self.generate_sample(state, **kwargs)
                else:
                    async with slots:
                        samples = await self.generate_sample(state, **kwargs)
                return [await teacher.on_sample_ready(sample) for sample in samples] if teacher else samples
            except asyncio.CancelledError:
                return [aborted(state)]

        jobs = [asyncio.create_task(generate(state)) for state in rollout_state]
        self._active.update(jobs)
        try:
            groups = await asyncio.gather(*jobs)
        finally:
            for job in jobs:
                if not job.done():
                    job.cancel()
            await asyncio.gather(*jobs, return_exceptions=True)
            self._active.difference_update(jobs)
        samples = [sample for group in groups for sample in group]
        samples = maybe_filter_invalid_sample(samples, self.is_valid_sample_fn, self.logger)
        samples = await teacher.on_group_ready(samples) if teacher else samples
        samples = self.canonicalize_train_fields(samples)
        if any(sample.status in (Status.FAILED, Status.FILTERED) for sample in samples):
            # Producer discards the whole group; no retained sibling can be resumed.
            await self.partial.discard(samples)
        return samples

    def canonicalize_train_fields(self, group: list[RolloutState]) -> list[RolloutState]:
        """Validate the canonical training fields of one sandbox agentic
        rollout group.

        Each segment state carries ``input_ids``/``labels``/``logprobs`` exported from
        the trace store as one unshifted full sequence, so there is nothing left to
        assemble. Canonicalization only re-checks the unified convention (``labels``
        present and of equal length, ``logprobs`` either ``None`` or of equal length)
        per segment state and marks a violating sample ``Status.FAILED`` instead of
        raising into the producer. States without ``input_ids`` (eval-mode states) carry
        no training fields and are skipped.

        Args:
            group (list[RolloutState]): One flattened group of segment rollout states
                after generation, judging, and filtering.

        Returns:
            list[RolloutState]: The same group with validated training fields.
        """
        for state in group:
            if state.status != Status.COMPLETED or state.input_ids is None:
                continue
            try:
                labels = state.labels
                if labels is None or len(labels) != len(state.input_ids):
                    expected = 0 if labels is None else len(labels)
                    raise ValueError(f"labels length mismatch: {expected} vs {len(state.input_ids)}")
                if state.logprobs is not None and len(state.logprobs) != len(state.input_ids):
                    raise ValueError(f"logprobs length mismatch: {len(state.logprobs)} vs {len(state.input_ids)}")
            except Exception as exc:
                self.logger.error(f"Canonicalize train fields failed for rollout_id={state.rollout_id}: {exc}")
                state.status = Status.FAILED
                state.reward = None
                state.error_msg = f"canonicalize_train_fields failed: {exc}"
        for state in group:
            write_train_meta(state)
        return group

    @trace_rollout_endpoint("agent_in_sandbox_loop.generate_sample", target_arg="state")
    async def generate_sample(self, state: RolloutState, **kwargs) -> list[RolloutState]:  # type: ignore[override]
        if self.partial.closing:
            raise RuntimeError("Agent episode loop is closed")
        if kwargs.get("enable_partial_rollout", False):
            if not self.config.partial_rollout:
                raise ValueError("partial rollout requires an explicitly configured resumable runner")
            return await self.partial.generate(state)
        if self._paused:
            return [aborted(state)]
        current = asyncio.current_task()
        owns_tracking = current is not None and current not in self._active
        if owns_tracking:
            self._active.add(current)
        try:
            return await self._execute(state)
        finally:
            if owns_tracking:
                self._active.discard(current)

    async def prepare_pause(self) -> None:
        """Close request admission and drain before global inference abort."""
        self._admission.clear()
        await self.pause()

    async def resume(self) -> None:
        self._admission.set()

    async def pause(self) -> None:
        if self._pause_task is None or self._pause_task.done():
            self._pause_task = asyncio.create_task(self._pause())
        await asyncio.shield(self._pause_task)

    async def _pause(self) -> None:
        self._paused = True
        try:
            await self.partial.pause()
            active = tuple(self._active)
            for task in active:
                if task not in self.partial.waiters:
                    task.cancel()
            await asyncio.gather(*active, return_exceptions=True)
        finally:
            self._paused = False

    def _provider(self) -> EnvironmentProvider:
        if self._environment_provider is None:
            config = self._environment_config
            if isinstance(config, str):
                config = _import_from_path(config)
            self._environment_provider = create_object(copy.deepcopy(config))
        return self._environment_provider

    async def close(self) -> None:
        """Cancel live work, drain contexts, then close shared connections."""
        self._closing = True
        if self._close_task is None or (self._close_task.done() and self._close_task.exception() is not None):
            self._close_task = asyncio.create_task(self._close())
        cancelled = False
        while not self._close_task.done():
            try:
                await asyncio.shield(self._close_task)
            except asyncio.CancelledError:
                cancelled = True
        self._close_task.result()
        if cancelled:
            raise asyncio.CancelledError

    async def _close(self) -> None:
        self.partial.closing = True
        self._admission.clear()
        if self._pause_task and not self._pause_task.done():
            self._pause_task.cancel()
            await asyncio.gather(self._pause_task, return_exceptions=True)
        await self.partial.close()
        tasks = list(self._active)
        for task in tasks:
            task.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)
        if self._environment_provider is not None:
            await self._environment_provider.close()
            self._environment_provider = None

    async def _execute(self, state: RolloutState, *, partial=None) -> list[RolloutState]:
        # A reroll is a new attempt, not an extension of stale training tensors.
        for name in (
            "input_ids",
            "labels",
            "response_ids",
            "response_model_steps",
            "logprobs",
            "routed_experts",
            "teacher_targets",
            "response",
            "tool_calls",
            "reward",
            "error_msg",
            "finish_reason",
            "position_ids",
        ):
            setattr(state, name, None)
        for name in tuple(state.extra_fields):
            if name.startswith("agent_") and name != "agent_partial_handle":
                state.extra_fields.pop(name)
        item = None
        trace_diagnostics = None
        try:
            item = state.extra_fields["rollout_item"].model_copy(deep=True)
            # A new execution attempt needs a fresh inference session even when
            # replay supplies a completed/expired state from an earlier attempt.
            state.session_id = uuid.uuid4().int
            item.uid, item.group_id = state.session_id, state.group_id
            item.execution = EpisodeContext(partial, lambda segments: self.partial_trace(state, segments))
            runner = _resolve_runner(item.pipeline, str(item.uid))
            if runner is None:
                raise ValueError("AgentRolloutItem.pipeline is required")
            async with self._provider().open(item) as environment:
                state.extra_fields["agent_session_id"] = state.session_id
                state.extra_fields["agent_environment"] = item.metadata.get("environment_id")
                result = await runner.run(item, environment)
                state.extra_fields["agent_environment"] = item.infer.sandbox_env_id
                await environment.check()
                async with partial.completion() if partial is not None else nullcontext():
                    samples = await self._build_rollout_states(state, result)
                    if partial is not None and result.status == RolloutStatus.COMPLETED:
                        traces = [
                            TraceTokens.from_trace(
                                {
                                    "input_ids": sample.input_ids,
                                    "labels": sample.labels,
                                    "logprobs": sample.logprobs,
                                }
                            )
                            for sample in samples
                        ]
                        for sample, versions in zip(samples, partial.segment_versions(traces), strict=True):
                            sample.response_model_steps = response_prefix_versions(sample, versions)
                            sample.extra_fields["agent_partial_pauses"] = partial.pauses
                await environment.check()
            return samples
        except asyncio.CancelledError:
            if state.extra_fields.get("agent_partial_error"):
                state.status, state.reward = Status.FAILED, None
                state.error_msg = "partial pause failed: " + state.extra_fields["agent_partial_error"]
                return [state]
            return [aborted(state)]
        except Exception as exc:
            if isinstance(exc, TracePending):
                trace_diagnostics = exc.diagnostics
            state.status, state.reward, state.finish_reason = Status.FAILED, None, "error"
            if item is not None:
                state.extra_fields["agent_artifacts"] = _to_json_safe(item.artifacts)
                state.extra_fields["agent_judgers"] = {
                    name: record.model_dump(mode="json") for name, record in item.judgers.items()
                }
            state.error_msg = f"{type(exc).__name__}: {exc}"
            self.logger.error(f"[AgentInSandboxLoop] failed: {exc}\n{traceback.format_exc()}")
            return [state]
        finally:
            if item is not None and self.config.artifact_root:
                from pathlib import Path

                directory = Path(self.config.artifact_root) / str(item.uid)
                try:
                    directory.mkdir(parents=True, exist_ok=True)
                    (directory / "infer-stage.json").write_text(item.infer.model_dump_json(indent=2))
                    (directory / "judgers.json").write_text(
                        json.dumps(
                            {name: record.model_dump(mode="json") for name, record in item.judgers.items()}, indent=2
                        )
                    )
                    if state.status == Status.FAILED:
                        evidence = {
                            "task_id": item.id,
                            "group_id": str(item.group_id) if item.group_id is not None else None,
                            "session_id": str(state.session_id),
                            "environment_id": state.extra_fields.get("agent_environment"),
                            "error": state.error_msg,
                            "finish_reason": state.finish_reason,
                            "trace": trace_diagnostics,
                            "artifacts": {
                                key: _to_json_safe(item.artifacts[key])
                                for key in ("messages", "response_message")
                                if key in item.artifacts
                            },
                        }
                        path = directory / "failure.json"
                        # Includes task/model text. A fresh session owns this
                        # path; do not follow or overwrite an existing file.
                        fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
                        with os.fdopen(fd, "w") as stream:
                            json.dump(evidence, stream, indent=2)
                        state.extra_fields["agent_failure_evidence"] = str(path)
                except OSError:
                    self.logger.exception("Unable to persist episode stage evidence")

    async def _build_rollout_states(self, rollout_state: RolloutState, item: AgentRolloutItem) -> list[RolloutState]:
        if self.mode == "eval":
            self._fill_eval_rollout_state(rollout_state, item)
            return [rollout_state]

        response_message = _response_message(item.artifacts, required=item.status == RolloutStatus.COMPLETED)
        error = _task_result_error(item)
        rollout_state.status = Status.COMPLETED if error is None else Status.FAILED
        rollout_state.finish_reason = str(
            response_message.get("finish_reason") or ("stop" if item.status == RolloutStatus.COMPLETED else "error")
        )
        rollout_state.reward = _extract_reward_payload(item) if error is None else None
        rollout_state.extra_fields["origin_data_source"] = item.data_source
        rollout_state.extra_fields["agent_status"] = item.status.value
        selected_agent = _selected_agent(item)
        if selected_agent is not None:
            rollout_state.extra_fields["agent_name"] = selected_agent.get("name")
            rollout_state.extra_fields["agent_selected"] = _to_json_safe(selected_agent)
        rollout_state.extra_fields["agent_judgers"] = {
            name: record.model_dump(mode="json") for name, record in item.judgers.items()
        }
        finish_info = _finish_info(response_message)
        if finish_info:
            rollout_state.extra_fields["agent_finish_info"] = finish_info
        rollout_state.error_msg = error
        if rollout_state.status == Status.FAILED:
            rollout_state.extra_fields["agent_artifacts"] = _to_json_safe(item.artifacts)
            return [rollout_state]

        segments = _load_messages_artifact(item.artifacts)
        if not segments:
            raise ValueError("Agent artifacts must contain at least one trainable messages trace.")
        branches = await self._trace_branches(rollout_state, segments, require_assistant=True)
        if not branches:
            raise ValueError("Agent artifacts must contain at least one trainable messages trace.")
        rollout_states = []
        for index, (prompt, messages, tools) in enumerate(branches):
            sample = rollout_state.model_copy(deep=True)
            data = await get_store().export_training_trace.remote(str(rollout_state.session_id), prompt)
            apply_training_trace(sample, data)
            sample.response = self.tokenizer.decode(sample.response_ids)
            sample.extra_fields.update(
                agent_messages=messages,
                agent_tools=tools,
                agent_tool_turns=_count_tool_turns(messages),
                agent_trace_segment_index=index,
                agent_trace_segment_count=len(branches),
                agent_session_id=rollout_state.session_id,
            )
            rollout_states.append(sample)

        rollout_states[0].extra_fields["agent_artifacts"] = _to_json_safe(item.artifacts)
        return rollout_states

    def _fill_eval_rollout_state(self, rollout_state: RolloutState, item: AgentRolloutItem) -> None:
        error = _task_result_error(item)
        is_success = error is None
        response_message = _response_message(item.artifacts, required=is_success)
        rollout_state.status = Status.COMPLETED if is_success else Status.FAILED
        rollout_state.finish_reason = str(response_message.get("finish_reason") or ("stop" if is_success else "error"))
        rollout_state.reward = _extract_reward_payload(item) if is_success else None
        rollout_state.input_ids = None
        rollout_state.labels = None
        rollout_state.response_ids = None
        rollout_state.logprobs = None
        rollout_state.routed_experts = None
        rollout_state.routed_experts_owner = None
        rollout_state.response_model_steps = None
        rollout_state.extra_fields["origin_data_source"] = item.data_source
        rollout_state.extra_fields["agent_status"] = item.status.value
        selected_agent = _selected_agent(item)
        if selected_agent is not None:
            rollout_state.extra_fields["agent_name"] = selected_agent.get("name")
            rollout_state.extra_fields["agent_selected"] = _to_json_safe(selected_agent)
        rollout_state.extra_fields["agent_artifacts"] = _to_json_safe(item.artifacts)
        rollout_state.extra_fields["agent_judgers"] = {
            name: record.model_dump(mode="json") for name, record in item.judgers.items()
        }
        rollout_state.error_msg = error

        rollout_state.response = _response_text(response_message)
        finish_info = _finish_info(response_message)
        if finish_info:
            rollout_state.extra_fields["agent_finish_info"] = finish_info
        if not is_success:
            return

        messages, tools = _load_eval_trace_segment(item.artifacts)
        if messages:
            rollout_state.extra_fields["agent_messages"] = messages
            rollout_state.extra_fields["agent_tools"] = tools
            rollout_state.extra_fields["agent_tool_turns"] = _count_tool_turns(messages)

    async def _trace_branches(self, state, segments, *, require_assistant=False):
        store = get_store()
        # A drained gate guarantees the store has committed every response.
        # Agent recording happens later. Check every recorded response, rather
        # than assistant counts (branches can repeat a shared history).
        keys = await store.keys.remote(str(state.session_id))
        branches = training_segments(self.tokenizer, segments, require_assistant=require_assistant, stored_keys=keys)
        prompts = [key for key, _, _ in branches]
        missing = [key for key in keys if not any(prompt.startswith(key) for prompt in prompts)]
        if missing:
            # A failed/cancelled request may leave a prompt-only node. Require
            # coverage of every real response token, not every masked input.
            # Use exact stored provenance; never synthesize tokens/logprobs.
            async def recorded(key):
                trace = TraceTokens.from_trace(await store.export_training_trace.remote(str(state.session_id), key))
                return dict(zip(trace.keys, zip(trace.tokens, trace.logprobs, strict=True), strict=True))

            covered = {}
            for prompt in prompts:
                covered.update(await recorded(prompt))
            for key in missing:
                uncovered = sum(covered.get(token_key) != value for token_key, value in (await recorded(key)).items())
                if uncovered:
                    raise TracePending(
                        "waiting for the agent to record all completed inference branches",
                        diagnostics={
                            "stored_branches": len(keys),
                            "matched_branches": len(prompts),
                            "covered_response_tokens": len(covered),
                            "uncovered_response_tokens": uncovered,
                            "missing_trace_key": key,
                        },
                    )
        return branches

    async def partial_trace(self, state, segments):
        prompts = [key for key, _, _ in await self._trace_branches(state, segments)]
        store = get_store()
        return [
            TraceTokens.from_trace(await store.export_training_trace.remote(str(state.session_id), prompt))
            for prompt in prompts
        ]
