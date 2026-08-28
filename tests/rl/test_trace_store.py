import asyncio
import unittest
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

import ray

from xtuner.v1.rl.rollout import trace_store as trace_store_module
from xtuner.v1.rl.rollout.trace_store import (
    RolloutTraceStore,
    TokenizedSegment,
    Trie,
    _free_ray_refs,
    get_existing_store,
    release_and_discard_rollout_groups,
    release_existing_sessions,
)


class TestRolloutTraceCleanup(unittest.TestCase):
    def test_release_and_discard_detaches_only_trace_owned_refs(self):
        trace_owned_ref = object()
        rollout_owned_ref = object()
        trace_owned = SimpleNamespace(session_id="trace-owned", routed_experts=trace_owned_ref)
        rollout_owned = SimpleNamespace(session_id="rollout-owned", routed_experts=rollout_owned_ref)
        routed_experts_seen_by_discard = {}

        def record_discard(item):
            routed_experts_seen_by_discard[item.session_id] = item.routed_experts

        with (
            patch(
                "xtuner.v1.rl.rollout.trace_store.release_existing_sessions",
                new=AsyncMock(return_value={"trace-owned"}),
            ) as release_sessions,
            patch(
                "xtuner.v1.rl.rollout.trace_store.discard_rollout_state",
                side_effect=record_discard,
            ) as discard,
        ):
            asyncio.run(release_and_discard_rollout_groups([[trace_owned, rollout_owned]]))

        release_sessions.assert_awaited_once_with(["trace-owned", "rollout-owned"])
        self.assertIsNone(routed_experts_seen_by_discard["trace-owned"])
        self.assertIs(routed_experts_seen_by_discard["rollout-owned"], rollout_owned_ref)
        self.assertEqual(discard.call_count, 2)

    def test_get_existing_store_returns_none_when_ray_is_uninitialized(self):
        cached_store = object()
        with (
            patch.object(trace_store_module, "_handle_cache", cached_store),
            patch.object(trace_store_module.ray, "is_initialized", return_value=False),
        ):
            self.assertIsNone(get_existing_store())
            self.assertIsNone(trace_store_module._handle_cache)


class TestTrieObservability(unittest.TestCase):
    def test_stats_count_trace_volume_without_exposing_contents(self):
        trie = Trie()
        trie.insert(
            "prompt",
            TokenizedSegment(
                text="prompt",
                token_ids=[1, 2],
                expert_key=object(),
                expert_nbytes=16,
            ),
        )
        trie.insert(
            "promptresponse",
            TokenizedSegment(
                text="response",
                token_ids=[3, 4, 5],
                labels=[3, -100, 5],
                expert_key=object(),
                expert_nbytes=24,
            ),
        )

        self.assertEqual(
            trie.stats(),
            {
                "tree_node_count": 2,
                "value_node_count": 2,
                "leaf_value_count": 1,
                "branch_node_count": 0,
                "token_segment_count": 2,
                "token_count": 5,
                "action_token_count": 2,
                "expert_ref_count": 2,
                "expert_payload_bytes": 40,
                "text_chars": 14,
            },
        )


class TestRolloutTraceStoreObservability(unittest.TestCase):
    def test_snapshot_tracks_live_and_exported_volume(self):
        store_class = RolloutTraceStore.__ray_metadata__.modified_class
        store = store_class()
        store.insert(
            "observed",
            "prompt",
            TokenizedSegment(
                text="prompt",
                token_ids=[1, 2],
                expert_key=object(),
                expert_nbytes=16,
            ),
        )
        store.insert(
            "observed",
            "promptresponse",
            TokenizedSegment(
                text="response",
                token_ids=[3, 4],
                labels=[3, 4],
                expert_key=object(),
                expert_nbytes=24,
            ),
        )

        exported = store.export_training_trace("observed", "promptresponse")
        snapshot = store.get_observability_snapshot("observed")

        self.assertEqual(exported["input_ids"], [1, 2, 3, 4])
        self.assertEqual(snapshot["live_session_count"], 1)
        self.assertEqual(snapshot["trace_totals"]["token_segment_count"], 2)
        self.assertEqual(snapshot["trace_totals"]["token_count"], 4)
        self.assertEqual(snapshot["trace_totals"]["action_token_count"], 2)
        self.assertEqual(snapshot["export_totals"]["calls"], 1)
        self.assertEqual(snapshot["export_totals"]["tokens"], 4)
        self.assertEqual(snapshot["export_totals"]["expert_refs"], 2)
        self.assertEqual(snapshot["export_totals"]["expert_payload_bytes"], 40)
        self.assertEqual(snapshot["requested_session"]["token_count"], 4)
        self.assertEqual(snapshot["top_sessions"][0]["session_id"], "observed")
        self.assertGreater(snapshot["resource"]["actor_rss_bytes"], 0)

        store.release("observed")
        released_snapshot = store.get_observability_snapshot("observed")
        self.assertEqual(released_snapshot["live_session_count"], 0)
        self.assertIsNone(released_snapshot["requested_session"])


class TestRolloutTraceStore(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.started_ray = False
        try:
            if not ray.is_initialized():
                ray.init(address="local", num_cpus=1, include_dashboard=False, ignore_reinit_error=True)
                cls.started_ray = True
        except Exception as exc:
            raise unittest.SkipTest(f"Ray init failed for trace-store tests: {exc}") from exc

    @classmethod
    def tearDownClass(cls):
        if cls.started_ray and ray.is_initialized():
            ray.shutdown()

    def test_release_sessions_deduplicates_and_skips_missing_ids(self):
        store = RolloutTraceStore.remote()
        try:
            ray.get(store.insert.remote("a", "prompt-a", {"value": 1}))
            ray.get(store.insert.remote("b", "prompt-b", {"value": 2}))

            released = ray.get(store.release_sessions.remote(["a", "missing", "a"]))

            self.assertEqual(released, ["a"])
            self.assertEqual(ray.get(store.list_sessions.remote()), ["b"])
        finally:
            ray.kill(store)

    def test_release_existing_sessions_stably_deduplicates_before_rpc(self):
        release_remote = AsyncMock(return_value=["one"])
        store = SimpleNamespace(release_sessions=SimpleNamespace(remote=release_remote))
        with patch(
            "xtuner.v1.rl.rollout.trace_store.get_existing_store",
            return_value=store,
        ):
            released = asyncio.run(release_existing_sessions(["one", "one", "missing"]))

        self.assertEqual(released, {"one"})
        release_remote.assert_awaited_once_with(["one", "missing"])

    def test_release_existing_sessions_handles_empty_input_and_missing_store(self):
        with patch("xtuner.v1.rl.rollout.trace_store.get_existing_store") as get_store:
            self.assertEqual(asyncio.run(release_existing_sessions([])), set())
            get_store.assert_not_called()

        with patch(
            "xtuner.v1.rl.rollout.trace_store.get_existing_store",
            return_value=None,
        ):
            self.assertEqual(asyncio.run(release_existing_sessions(["missing"])), set())

    def test_free_ray_refs_recurses_into_nested_containers(self):
        object_ref = ray.put({"payload": [1, 2, 3]})
        with patch.object(ray.internal, "free") as free:
            _free_ray_refs({"outer": [({"inner": object_ref},)]})

        free.assert_called_once_with([object_ref], local_only=False)
