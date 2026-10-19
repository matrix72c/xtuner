import asyncio
import unittest
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

import ray

from xtuner.v1.rl.rollout import trace_store as trace_store_module
from xtuner.v1.rl.rollout.trace_store import (
    RolloutTraceStore,
    get_existing_store,
    release_and_discard_rollout_groups,
    release_existing_sessions,
)


class TestRolloutTraceCleanup(unittest.TestCase):
    def test_release_and_discard_detaches_only_trace_owned_refs(self):
        trace_owned_ref = object()
        rollout_owned_ref = object()
        trace_owned = SimpleNamespace(
            session_id="trace-owned", routed_experts=trace_owned_ref, routed_experts_owner="trace_store"
        )
        rollout_owned = SimpleNamespace(
            session_id="rollout-owned", routed_experts=rollout_owned_ref, routed_experts_owner="rollout"
        )
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

    def test_trie_overwrite_drops_handle_without_freeing_borrowed_refs(self):
        """Overwrite must not free replaced refs: sibling RolloutStates keep
        their own handles alive via Ray's reference counting."""
        trie = trace_store_module.Trie()
        old_ref = ray.put({"value": "old"})
        new_ref = ray.put({"value": "new"})
        trie.insert("turn", {"expert_key": old_ref})

        with patch.object(ray.internal, "free") as free:
            trie.insert("turn", {"expert_key": new_ref})

        free.assert_not_called()
        # The borrower keeps its own handle: dropping the trie's handle must
        # not invalidate the object.
        self.assertEqual(ray.get(old_ref), {"value": "old"})
        _, nodes = trie.search("turn", filter_none=True)
        self.assertEqual(ray.get(nodes[-1].value["expert_key"]), {"value": "new"})

    def test_release_keeps_shared_borrowers_readable(self):
        from xtuner.v1.data_proto.rl_data import RolloutState, discard_rollout_state, reset_rollout_response

        trie = trace_store_module.Trie()
        shared_ref = ray.put([1, 2, 3])
        trie.insert("turn-a", {"expert_key": shared_ref})
        trie.insert("turn-b", {"expert_key": shared_ref})
        first = RolloutState(message=[], routed_experts=[shared_ref], routed_experts_owner="trace_store")
        sibling = RolloutState(message=[], routed_experts=[shared_ref], routed_experts_owner="trace_store")
        reset_rollout_response(first)
        self.assertEqual(ray.get(sibling.routed_experts[0]), [1, 2, 3])
        trie.release("turn-a")
        self.assertEqual(ray.get(sibling.routed_experts[0]), [1, 2, 3])
        trie.release()
        trie.release()  # repeated session cleanup
        self.assertEqual(trie.keys(), [])
        self.assertEqual(ray.get(sibling.routed_experts[0]), [1, 2, 3])
        discard_rollout_state(sibling)
        self.assertIsNone(sibling.routed_experts)

    def test_pending_consumer_survives_session_release_after_failure(self):
        @ray.remote(num_cpus=0)
        class Consumer:
            def hold(self, refs):
                self.refs = refs

            def consume(self):
                return ray.get(self.refs[0])

        store = RolloutTraceStore.remote()
        consumer = Consumer.remote()
        try:
            ref = ray.put({"expert": 17})
            ray.get(store.insert.remote("episode", "turn", {"expert_key": ref}))
            ray.get(consumer.hold.remote([ref]))
            # A failed sibling triggers trainer finally cleanup while this
            # consumer still owns its dispatch and has not read the tensor.
            ray.get(store.release_sessions.remote(["episode"]))
            self.assertEqual(ray.get(store.list_sessions.remote()), [])
            self.assertEqual(ray.get(consumer.consume.remote()), {"expert": 17})
        finally:
            ray.kill(consumer)
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
