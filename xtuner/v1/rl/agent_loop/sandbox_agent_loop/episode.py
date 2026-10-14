"""In-process ownership of resumable agent episode runs (not a durable checkpoint)."""

from __future__ import annotations

import asyncio
import json
import time
import uuid
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from xtuner.v1.data_proto.rl_data import Status
from xtuner.v1.rl.rollout.trace_store import release_existing_sessions

from .training_trace import TracePending


@dataclass
class PartialRun:
    state: Any
    task: asyncio.Task | None = None
    running: asyncio.Event = field(default_factory=asyncio.Event)
    parked: asyncio.Event = field(default_factory=asyncio.Event)
    lock: asyncio.Lock = field(default_factory=asyncio.Lock)
    controller: Any = None
    snapshot: Any = None
    pauses: int = 0
    finished: bool = False
    prefix: list[int] = field(default_factory=list)
    versions: list[int] = field(default_factory=list)
    gate_state: dict = field(default_factory=dict)
    # Append-only journal presented to ReplayBuffer solely for version stamping.
    # Segment order/length may change; journal offsets never do.
    token_index: dict[bytes, int] = field(default_factory=dict)
    token_logprobs: list[float] = field(default_factory=list)

    def _check_traces(self, traces):
        seen = {}
        for trace in traces:
            for key, token, logprob in zip(trace.keys, trace.tokens, trace.logprobs, strict=True):
                value = (token, logprob)
                if key in seen and seen[key] != value:
                    raise ValueError("partial trace branches disagree on shared tokens/logprobs")
                seen[key] = value
                index = self.token_index.get(key)
                if index is not None and (self.prefix[index] != token or self.token_logprobs[index] != logprob):
                    raise ValueError("partial trace changed previously recorded tokens/logprobs")
        if not self.token_index.keys() <= seen.keys():
            raise ValueError("partial trace lost previously recorded token provenance")

    def record(self, traces):
        self._check_traces(traces)
        for trace in traces:
            for key, token, logprob in zip(trace.keys, trace.tokens, trace.logprobs, strict=True):
                if key not in self.token_index:
                    self.token_index[key] = len(self.prefix)
                    self.prefix.append(token)
                    self.token_logprobs.append(logprob)

    def segment_versions(self, traces):
        self._check_traces(traces)
        if len(self.versions) != len(self.prefix):
            raise ValueError("partial response tokens lack producer model versions")
        result = []
        for trace in traces:
            versions = []
            new_seen = False
            for key in trace.keys:
                index = self.token_index.get(key)
                if index is None:
                    new_seen = True
                elif new_seen:
                    raise ValueError("old-policy tokens follow new tokens within a trace segment")
                else:
                    versions.append(self.versions[index])
            result.append(versions)
        return result

    @asynccontextmanager
    async def completion(self):
        # Export and version assignment must be atomic with respect to pause:
        # a pause must not append unstamped journal tokens during final export.
        while True:
            await self.running.wait()
            await self.lock.acquire()
            if self.running.is_set():
                break
            self.lock.release()
        try:
            yield
            self.finished = True
        finally:
            self.lock.release()

    async def checkpoint(self):
        await self.running.wait()

    async def attach(self, controller, snapshot):
        async with self.lock:
            self.controller, self.snapshot = controller, snapshot
            await controller("resume" if self.running.is_set() else "pause")

    async def resume(self, state):
        async with self.lock:
            if list(state.response_ids or []) != self.prefix:
                raise ValueError("partial replay token journal changed")
            self.versions = list(state.response_model_steps or [])
            if len(self.versions) != len(self.prefix):
                raise ValueError("partial response tokens lack producer model versions")
            self.parked.clear()
            if self.controller:
                self.gate_state = await self.controller("resume")
            self.running.set()

    async def pause(self):
        async with self.lock:
            if self.finished:
                return
            self.running.clear()
            if self.controller:
                status = await self.controller("pause")
                while status["active"]:
                    await asyncio.sleep(0.2)
                    status = await self.controller("status")
                self.gate_state = status
                if self.snapshot:
                    self.record(await self.snapshot(status["completed"]))
            self.pauses += 1
            self.parked.set()


def aborted(state):
    state.status = Status.ABORTED
    state.reward = None
    state.finish_reason = "abort"
    return state


class PartialRuns:
    """Retain live runs across XTuner pause/resume; replay owns policy versions."""

    def __init__(self, config, execute, slots, logger):
        self.config, self.execute, self.slots, self.logger = config, execute, slots, logger
        self.runs: dict[str, PartialRun] = {}
        self.waiters: set[asyncio.Task] = set()
        self.closing = False

    async def prepare_group(self, states):
        """Reroll an expired episode once, even when it exported many traces."""
        expired = {
            str(state.extra_fields["agent_session_id"])
            for state in states
            if state.status == Status.EXPIRED and "agent_session_id" in state.extra_fields
        }
        if not expired:
            return states
        # A segment is a training representation, not an independently resettable
        # environment. Replace all siblings if any segment of the episode expired.
        replacements = {
            str(state.extra_fields["agent_session_id"]): state
            for state in states
            if state.status == Status.EXPIRED and "agent_session_id" in state.extra_fields
        }
        result, seen = [], set()
        for state in states:
            session = str(state.extra_fields.get("agent_session_id"))
            if session not in expired:
                result.append(state)
            elif session not in seen:
                seen.add(session)
                fresh = replacements[session].model_copy(deep=True)
                for name in (
                    "agent_session_id",
                    "agent_trace_segment_index",
                    "agent_trace_segment_count",
                    "agent_messages",
                    "agent_tools",
                    "agent_artifacts",
                ):
                    fresh.extra_fields.pop(name, None)
                result.append(fresh)
        await release_existing_sessions(sorted(expired))
        return result

    async def generate(self, state):
        if self.closing:
            raise RuntimeError("agent episode loop is closed")
        if state.status == Status.COMPLETED:
            return [state]
        key = state.extra_fields.get("agent_partial_handle")
        run = self.runs.get(key)
        if key and run is None:
            state.status, state.reward = Status.FAILED, None
            state.error_msg = "partial run owner is gone; live sessions cannot migrate or survive trainer restart"
            return [state]
        if run is not None and state.status == Status.EXPIRED:
            run.task.cancel()
            await asyncio.gather(run.task, return_exceptions=True)
            del self.runs[key]
            if self.config.mode == "train" and run.state.session_id is not None:
                await release_existing_sessions([str(run.state.session_id)])
            state.extra_fields.pop("agent_partial_handle", None)
            state.response_ids, state.response_model_steps = [], []
            run = None
        if run is None:
            if len(self.runs) >= self.config.max_concurrent_samples:
                raise RuntimeError("partial run capacity exhausted; bound producer pending samples")
            key = uuid.uuid4().hex
            state.extra_fields["agent_partial_handle"] = key
            run = PartialRun(state.model_copy(deep=True))
            run.running.set()
            self.runs[key] = run

            async def execute():
                async with self.slots:
                    await run.checkpoint()
                    return await self.execute(run.state, partial=run)

            run.task = asyncio.create_task(execute())
        elif not run.task.done():
            try:
                await run.resume(state)
            except Exception as exc:
                run.task.cancel()
                await asyncio.gather(run.task, return_exceptions=True)
                self.runs.pop(key, None)
                state.status, state.reward, state.finish_reason = Status.FAILED, None, "error"
                state.error_msg = f"partial resume failed: {type(exc).__name__}: {exc}"
                state.extra_fields.pop("agent_partial_handle", None)
                self.event("resume_failed", run)
                return [state]
            self.event("resume", run)
        waiter = asyncio.current_task()
        self.waiters.add(waiter)
        parked = asyncio.create_task(run.parked.wait())
        try:
            await asyncio.wait({run.task, parked}, return_when=asyncio.FIRST_COMPLETED)
            if run.task.done():
                samples = await run.task
                self.event("finish", run)
                self.runs.pop(key, None)
                for sample in samples:
                    sample.extra_fields.pop("agent_partial_handle", None)
                return samples
            return [self.snapshot_state(run)]
        except asyncio.CancelledError:
            if run.parked.is_set() and not self.closing:
                return [self.snapshot_state(run)]
            run.task.cancel()
            await asyncio.gather(run.task, return_exceptions=True)
            self.runs.pop(key, None)
            return [aborted(state)]
        finally:
            parked.cancel()
            await asyncio.gather(parked, return_exceptions=True)
            self.waiters.discard(waiter)

    def event(self, event, run):
        root = self.config.artifact_root
        if root:
            path = Path(root) / "partial-events.jsonl"
            path.parent.mkdir(parents=True, exist_ok=True)
            with path.open("a") as file:
                file.write(
                    json.dumps(
                        dict(
                            event=event,
                            time=time.time(),
                            session_id=str(run.state.session_id),
                            environment=run.state.extra_fields.get("agent_environment"),
                            pauses=run.pauses,
                            prefix_tokens=len(run.prefix),
                            model_steps=sorted(set(run.versions)),
                            gate=run.gate_state,
                        )
                    )
                    + "\n"
                )

    def snapshot_state(self, run):
        state = aborted(run.state.model_copy(deep=True))
        state.response_ids = list(run.prefix)
        state.response_model_steps = list(run.versions)
        state.extra_fields["agent_partial_pauses"] = run.pauses
        return state

    async def discard(self, samples):
        handles = {sample.extra_fields.get("agent_partial_handle") for sample in samples}
        runs = [self.runs.pop(key) for key in handles if key in self.runs]
        for run in runs:
            run.task.cancel()
        await asyncio.gather(*(run.task for run in runs), return_exceptions=True)

    async def pause(self):
        async def park(run):
            if run.task.done() or run.parked.is_set():
                return
            try:
                async with asyncio.timeout(self.config.partial_pause_timeout):
                    await run.pause()
                    if run.parked.is_set():
                        self.event("pause", run)
            except Exception as exc:
                self.logger.warning(f"Partial pause failed; reclaiming run: {type(exc).__name__}: {exc}")
                run.state.extra_fields["agent_partial_error"] = type(exc).__name__
                run.task.cancel()
                await asyncio.gather(run.task, return_exceptions=True)

        await asyncio.gather(*(park(run) for run in tuple(self.runs.values())))

    async def close(self):
        self.closing = True
        runs = tuple(self.runs.values())
        for run in runs:
            run.task.cancel()
        await asyncio.gather(*(run.task for run in runs), return_exceptions=True)
        self.runs.clear()


@dataclass
class EpisodeContext:
    """Ephemeral I/O coordination exposed to harness hooks, never serialized.

    A harness supplies request control and transcript reading. The loop supplies
    trace validation and version accounting; neither callback chooses policy age.
    """

    partial: PartialRun | None
    export_trace: Any

    async def checkpoint(self):
        if self.partial is not None:
            await self.partial.checkpoint()

    async def attach(self, controller, read_segments):
        if self.partial is None:
            raise RuntimeError("request boundary requires partial rollout to be enabled")

        async def snapshot(expected):
            async with asyncio.timeout(30):
                while True:
                    try:
                        return await self.export_trace(await read_segments(expected))
                    except TracePending:
                        await asyncio.sleep(0.2)

        await self.partial.attach(controller, snapshot)
