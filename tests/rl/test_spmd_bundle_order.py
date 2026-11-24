# Copyright (c) OpenMMLab. All rights reserved.
"""Rank 0 of a multi-node placement group must land on the driver's node.

Rank 0 is the rendezvous master, and the driver's node is the address a
launcher already knows. The order must not change on a single node.
"""

from xtuner.v1.rl.utils import AutoAcceleratorWorkers


# {node id: {local accelerator id: bundle index}}, nodes in the order the
# placement group table reports them. Bundle indices do not follow ranks.
TWO_NODES = {
    "worker-node": {1: 0, 0: 3, 3: 1, 2: 2},
    "head-node": {0: 6, 2: 4, 1: 7, 3: 5},
}


class TestSortBundles:
    def test_head_node_comes_first(self):
        order = AutoAcceleratorWorkers.sort_bundles(TWO_NODES, head_node_id="head-node")
        assert order == [6, 7, 4, 5, 3, 0, 2, 1]

    def test_local_accelerator_order_within_each_node(self):
        order = AutoAcceleratorWorkers.sort_bundles(TWO_NODES, head_node_id="worker-node")
        assert order == [3, 0, 2, 1, 6, 7, 4, 5]

    def test_single_node_is_unchanged(self):
        one = {"head-node": {2: 0, 0: 2, 1: 1, 3: 3}}
        assert AutoAcceleratorWorkers.sort_bundles(one, head_node_id="head-node") == [2, 1, 0, 3]
        assert AutoAcceleratorWorkers.sort_bundles(one, head_node_id=None) == [2, 1, 0, 3]

    def test_driver_outside_the_group_keeps_reported_order(self):
        order = AutoAcceleratorWorkers.sort_bundles(TWO_NODES, head_node_id="cpu-only-node")
        assert order == [3, 0, 2, 1, 6, 7, 4, 5]

    def test_other_nodes_keep_reported_order(self):
        three = {"a": {0: 0}, "b": {0: 1}, "head": {0: 2}, "c": {0: 3}}
        assert AutoAcceleratorWorkers.sort_bundles(three, head_node_id="head") == [2, 0, 1, 3]
