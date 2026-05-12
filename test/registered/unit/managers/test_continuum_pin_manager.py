import unittest

from sglang.test.ci.ci_register import register_cpu_ci

from sglang.srt.managers.continuum_pin_manager import ContinuumPinManager

register_cpu_ci(est_time=1, suite="stage-a-test-cpu")


class TestContinuumPinManager(unittest.TestCase):
    def test_queries_do_not_pop_expired_pin(self):
        manager = ContinuumPinManager()
        node = object()

        manager.pin_node(node=node, seconds=0.0, min_protected_len=0, now=1000.0)

        self.assertFalse(manager.is_pinned_node(node, now=1000.1))
        self.assertIsNone(manager.get_pin_info_by_node(node, now=1000.1))

        expired = manager.pop_if_expired(node, now=1000.1)
        self.assertIsNotNone(expired)
        self.assertIs(expired.node, node)

    def test_shared_now_keeps_state_consistent(self):
        manager = ContinuumPinManager()
        node = object()

        manager.pin_node(node=node, seconds=0.1, min_protected_len=0, now=1000.0)

        now = 1000.05
        self.assertIsNone(manager.pop_if_expired(node, now=now))
        self.assertIsNotNone(manager.get_pin_info_by_node(node, now=now))
