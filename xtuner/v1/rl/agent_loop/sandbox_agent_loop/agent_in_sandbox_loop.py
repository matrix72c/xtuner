from __future__ import annotations

import asyncio
import copy
import importlib
import json
import traceback
import uuid
from contextlib import nullcontext
from types import SimpleNamespace
from typing import Any, Literal

from lagent.utils import create_object
from pydantic import Field

from xtuner.v1.data_proto.rl_data import RolloutState, SampleParams, Status
from xtuner.v1.rl.judger import Judger
from xtuner.v1.rl.rollout import RolloutController
from xtuner.v1.rl.trace.rollout_api import trace_rollout_endpoint

from ...rollout.trace_store import get_store
from ..agent_loop import AgentLoop, AgentLoopConfig, maybe_filter_invalid_sample
from .environment import EnvironmentProvider
from .episode import EpisodeContext, PartialRuns, aborted
from .schemas import AgentRolloutItem, RolloutStatus
from .training_trace import TracePending, TraceTokens, apply_training_trace, training_segments


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
                payload.setdefault("score", item.reward)
            return payload
    if item.reward is not None:
        return {"score": item.reward}
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
        if any(sample.status in (Status.FAILED, Status.FILTERED) for sample in samples):
            # Producer discards the whole group; no retained sibling can be resumed.
            await self.partial.discard(samples)
        return samples

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

    async def close(self) -> None:
        self.partial.closing = True
        self._admission.clear()
        if self._pause_task and not self._pause_task.done():
            self._pause_task.cancel()
            await asyncio.gather(self._pause_task, return_exceptions=True)
        await self.partial.close()
        try:
            await self.pause()
        finally:
            if self._environment_provider is not None:
                await self._environment_provider.close()
                self._environment_provider = None

    def _provider(self) -> EnvironmentProvider:
        if self._environment_provider is None:
            config = self._environment_config
            if isinstance(config, str):
                config = _import_from_path(config)
            self._environment_provider = create_object(copy.deepcopy(config))
        return self._environment_provider

    async def _execute(self, state: RolloutState, *, partial=None) -> list[RolloutState]:
        # A reroll is a new attempt, not an extension of stale training tensors.
        for name in (
            "input_ids",
            "labels",
            "response_ids",
            "response_model_steps",
            "logprobs",
            "routed_experts",
            "response_mask",
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
                            sample.response_model_steps = versions
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
            state.status, state.reward, state.finish_reason = Status.FAILED, None, "error"
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
                except OSError:
                    self.logger.exception("Unable to persist episode stage evidence")

    async def _build_rollout_states(self, rollout_state: RolloutState, item: AgentRolloutItem) -> list[RolloutState]:
        if self.mode == "eval":
            self._fill_eval_rollout_state(rollout_state, item)
            return [rollout_state]

        response_message = _response_message(item.artifacts, required=item.status == RolloutStatus.COMPLETED)
        timed_out = response_message.get("finish_reason") == "timeout"
        rollout_state.status = (
            Status.COMPLETED if item.status == RolloutStatus.COMPLETED and not timed_out else Status.FAILED
        )
        rollout_state.finish_reason = str(
            response_message.get("finish_reason") or ("stop" if item.status == RolloutStatus.COMPLETED else "error")
        )
        rollout_state.reward = _extract_reward_payload(item) if rollout_state.status == Status.COMPLETED else None
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
        if item.error is not None:
            rollout_state.error_msg = f"{item.error.stage}/{item.error.category}: {item.error.message}"
        elif timed_out:
            rollout_state.error_msg = "TimeoutError: agent exceeded the configured timeout"
        if rollout_state.status == Status.FAILED:
            rollout_state.extra_fields["agent_artifacts"] = _to_json_safe(item.artifacts)
            return [rollout_state]

        segments = _load_messages_artifact(item.artifacts)
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
        response_message = _response_message(item.artifacts, required=item.status == RolloutStatus.COMPLETED)
        is_success = item.status == RolloutStatus.COMPLETED and response_message.get("finish_reason") != "timeout"
        rollout_state.status = Status.COMPLETED if is_success else Status.FAILED
        rollout_state.finish_reason = str(response_message.get("finish_reason") or ("stop" if is_success else "error"))
        rollout_state.reward = {"score": item.reward} if is_success and item.reward is not None else None
        rollout_state.input_ids = None
        rollout_state.labels = None
        rollout_state.response_ids = None
        rollout_state.logprobs = None
        rollout_state.routed_experts = None
        rollout_state.routed_experts_owner = None
        rollout_state.response_mask = None
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
        if item.error is not None:
            rollout_state.error_msg = f"{item.error.stage}/{item.error.category}: {item.error.message}"

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
                if any(covered.get(token_key) != value for token_key, value in (await recorded(key)).items()):
                    raise TracePending("waiting for the agent to record all completed inference branches")
        return branches

    async def partial_trace(self, state, segments):
        prompts = [key for key, _, _ in await self._trace_branches(state, segments)]
        store = get_store()
        return [
            TraceTokens.from_trace(await store.export_training_trace.remote(str(state.session_id), prompt))
            for prompt in prompts
        ]
