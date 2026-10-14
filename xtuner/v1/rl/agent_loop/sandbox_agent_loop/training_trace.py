"""Render the exact lookup key used by XTuner SessionServer."""

import hashlib
import math
import struct
from copy import deepcopy
from dataclasses import dataclass
from typing import Any

from xtuner.v1.rl.rollout.chat_template import canonicalize_messages_for_chat_template


def render_trace_key(tokenizer, messages, tools):
    # SessionServer scrubs EOS strings from content on both request and response.
    # Tool-error feedback can echo these strings even when the last turn is valid.
    # Preserve original artifacts, tool arguments and reasoning; only the lookup
    # projection changes. Token IDs/logprobs still come exclusively from the store.
    projected = deepcopy(messages)
    stop = getattr(tokenizer, "eos_token", None)
    if stop:
        for message in projected:
            content = message.get("content")
            if isinstance(content, str):
                message["content"] = content.replace(stop, "")
            elif isinstance(content, list):
                for part in content:
                    if isinstance(part, dict) and isinstance(part.get("text"), str):
                        part["text"] = part["text"].replace(stop, "")
    return tokenizer.apply_chat_template(
        canonicalize_messages_for_chat_template(projected),
        tools=tools,
        tokenize=False,
        add_generation_prompt=False,
    ).rstrip()


def training_segments(tokenizer, segments, *, require_assistant=False, stored_keys=None):
    """Recover maximal *rendered* branches, including earlier assistant turns.

    A message prefix need not be a chat-template prefix: Qwen drops historical
    thinking when a later user message arrives (e.g. a subagent notification).
    The proxy may deduplicate those earlier message records. Render each turn
    before deduplicating, so their actual inference tokens remain exportable.
    """
    candidates = {}
    for segment in segments:
        messages, tools = segment["messages"], segment.get("tools", [])
        if not isinstance(messages, list):
            raise ValueError("segment messages must be a list")
        if require_assistant and not any(message.get("role") == "assistant" for message in messages):
            raise ValueError("trace segment has no assistant response")
        for index, message in enumerate(messages):
            if message.get("role") == "assistant":
                prefix = messages[: index + 1]
                key = render_trace_key(tokenizer, prefix, tools)
                candidates[key] = (prefix, tools)
    if stored_keys is not None:
        # Historical prefixes reconstructed from a later transcript need not
        # have been sent to the model. Filter before maximal-prefix selection:
        # an invented longer history must not hide a real recorded response.
        recorded = set(stored_keys)
        candidates = {key: value for key, value in candidates.items() if key in recorded}
    return [
        (key, messages, tools)
        for key, (messages, tools) in candidates.items()
        if not any(other != key and other.startswith(key) for other in candidates)
    ]


class TracePending(Exception):
    """The agent has not yet recorded every response already in the store."""

    def __init__(self, message: str, *, diagnostics: dict[str, Any] | None = None):
        super().__init__(message)
        # May contain task/model text. Persist only in private failure artifacts,
        # never in the exception string (which is sent to shared logs).
        self.diagnostics = diagnostics


@dataclass(frozen=True)
class TraceTokens:
    keys: list[bytes]
    tokens: list[int]
    logprobs: list[float]

    @classmethod
    def from_trace(cls, trace):
        # Identity includes the full actual token/mask prefix, including tool
        # schema and masked context. Equal response text in different contexts
        # is not the same generation. Shared trie prefixes keep the same keys.
        digest = hashlib.sha256()
        keys, tokens, logprobs = [], [], []
        for index, (token, label, logprob) in enumerate(
            zip(trace["input_ids"], trace["labels"], trace["logprobs"], strict=True)
        ):
            digest.update(struct.pack("!qq", token, label))
            if index and label != -100:
                if not math.isfinite(logprob):
                    raise ValueError("non-finite partial trace logprob")
                keys.append(digest.digest())
                tokens.append(token)
                logprobs.append(logprob)
        return cls(keys, tokens, logprobs)


def apply_training_trace(state: Any, trace: dict[str, Any]) -> None:
    """Copy actual inference tokens/logprobs. Never synthesize training data by retokenizing messages."""
    tokens, labels, logprobs = trace["input_ids"], trace["labels"], trace["logprobs"]
    if not tokens or not (len(tokens) == len(labels) == len(logprobs)):
        raise ValueError("training trace token/label/logprob lengths do not match")
    if any(not math.isfinite(value) for value in logprobs):
        raise ValueError("trace logprobs must be finite")
    if not any(label != -100 for label in labels[1:]):
        raise ValueError("trace has no trainable response tokens")
    state.input_ids = tokens
    state.labels = labels
    state.logprobs = logprobs
    state.routed_experts = trace.get("routed_experts")
    if hasattr(state, "routed_experts_owner"):
        state.routed_experts_owner = "trace_store" if state.routed_experts is not None else None
    state.response_ids = list(tokens[len(state.prompt_ids or []) :])


def response_prefix_versions(state, model_versions: list[int]) -> list[int]:
    """Project old model-token versions onto the contiguous response suffix.

    The live journal contains only generated tokens. Upstream replay stamps a
    contiguous suffix including tool/context tokens. Give masked positions the
    adjacent old generation's version; stop before the first new generation so
    replay can stamp the remaining suffix with the current model version.
    """
    if not model_versions:
        return []
    prefix, consumed = [], 0
    for index in range(len(state.prompt_ids or []), len(state.input_ids)):
        if index > 0 and state.labels[index] != -100:
            if consumed == len(model_versions):
                break
            version = model_versions[consumed]
            consumed += 1
        else:
            version = model_versions[max(consumed - 1, 0)]
        prefix.append(version)
    if consumed != len(model_versions):
        raise ValueError("old model versions exceed the generated trace tokens")
    return prefix
