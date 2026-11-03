"""Rollout serving 指标抓取测试。

Good Tests:
- 用 pin 版 lmdeploy 实际生成的 /metrics 文本锁定 parse_serving_metrics 的白名单行为。
- aggregate_serving_metrics 的 gauge 均值 / 计数求和，以及失败端点（空 dict）不稀释 gauge。
- ``_serving_step_values`` 的首步零、增量、counter reset clamp、直方图均值派生。
- worker scrape 与 controller collect 的 fail-open 行为（不打断训练）。

Bad Tests:
- 不启动真实推理服务，不用 Ray 集群；远程调用用替身 + ray.get 补丁。
"""

import sys
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

from xtuner.v1.rl.rollout.controller import RolloutController
from xtuner.v1.rl.rollout.worker import (
    SERVING_GAUGE_KEYS,
    RolloutWorker,
    aggregate_serving_metrics,
    parse_serving_metrics,
)
from xtuner.v1.train.rl_trainer import BaseRLTrainer


FIXTURE_PATH = Path(__file__).parent / "fixtures" / "lmdeploy_metrics.txt"


class TestParseServingMetrics(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.metrics = parse_serving_metrics(FIXTURE_PATH.read_text())

    def test_fixture_locks_allowlisted_values(self):
        # 行为：真实 lmdeploy exposition 中只取白名单样本，数值精确锁定
        self.assertEqual(self.metrics["requests_running"], 2.0)
        self.assertEqual(self.metrics["requests_waiting"], 1.0)
        self.assertEqual(self.metrics["gpu_cache_usage"], 0.35)
        self.assertEqual(self.metrics["prefix_cache_hit_rate"], 0.5)
        self.assertEqual(self.metrics["prompt_tokens"], 1024.0)
        self.assertEqual(self.metrics["generation_tokens"], 2048.0)
        self.assertEqual(self.metrics["cached_tokens"], 512.0)
        self.assertEqual(self.metrics["ttft_sum"], 0.25)
        self.assertEqual(self.metrics["ttft_count"], 1.0)
        self.assertEqual(self.metrics["tpot_sum"], 0.02)
        self.assertEqual(self.metrics["tpot_count"], 1.0)
        self.assertEqual(self.metrics["e2e_sum"], 3.5)
        self.assertEqual(self.metrics["e2e_count"], 1.0)

    def test_request_success_split_by_finish_reason(self):
        # 行为：request_success_total 聚合为 finished，并按 reason 拆分（去掉 Enum 前缀）
        self.assertEqual(self.metrics["finished"], 10.0)
        self.assertEqual(self.metrics["finished_finish"], 9.0)
        self.assertEqual(self.metrics["finished_cancel"], 1.0)

    def test_default_collectors_are_ignored(self):
        # 行为：python_gc / process 等默认 collector 不进结果
        for key in self.metrics:
            self.assertNotIn("python_", key)
            self.assertNotIn("process_", key)
        self.assertNotIn("iteration_tokens_sum", self.metrics)

    def test_gauge_keys_cover_all_non_cumulative_outputs(self):
        # 行为：除 gauge 集合外的 key 都是累计值，由消费方做差
        self.assertEqual(
            SERVING_GAUGE_KEYS,
            {"requests_running", "requests_waiting", "gpu_cache_usage", "prefix_cache_hit_rate"},
        )

    def test_empty_and_malformed_text_fail_open(self):
        self.assertEqual(parse_serving_metrics(""), {})
        self.assertEqual(parse_serving_metrics("not prometheus text {{{"), {})

    def test_missing_parser_returns_empty(self):
        # 行为：prometheus_client 不可用时返回空，不抛异常
        with patch.dict(sys.modules, {"prometheus_client.parser": None}):
            self.assertEqual(parse_serving_metrics(FIXTURE_PATH.read_text()), {})


class TestAggregateServingMetrics(unittest.TestCase):
    def test_gauges_average_and_counters_sum(self):
        merged = aggregate_serving_metrics(
            [
                {"requests_running": 2.0, "prompt_tokens": 100.0},
                {"requests_running": 4.0, "prompt_tokens": 50.0},
            ]
        )
        self.assertEqual(merged["requests_running"], 3.0)
        self.assertEqual(merged["prompt_tokens"], 150.0)

    def test_failed_endpoint_does_not_dilute_gauge_average(self):
        # 行为：抓取失败的端点贡献空 dict，gauge 均值只在有值的端点上算
        merged = aggregate_serving_metrics([{"gpu_cache_usage": 0.4}, {}, {"gpu_cache_usage": 0.8}])
        self.assertAlmostEqual(merged["gpu_cache_usage"], 0.6)


class TestServingStepValues(unittest.TestCase):
    def test_first_scrape_yields_zero_deltas(self):
        values = BaseRLTrainer._serving_step_values(None, {"prompt_tokens": 100.0, "requests_running": 3.0})
        self.assertEqual(values["prompt_tokens"], 0.0)
        self.assertEqual(values["requests_running"], 3.0)

    def test_counter_deltas_and_gauge_passthrough(self):
        previous = {"prompt_tokens": 100.0, "requests_running": 3.0, "gpu_cache_usage": 0.5}
        current = {"prompt_tokens": 130.0, "requests_running": 1.0, "gpu_cache_usage": 0.7}
        values = BaseRLTrainer._serving_step_values(previous, current)
        self.assertEqual(values["prompt_tokens"], 30.0)
        self.assertEqual(values["requests_running"], 1.0)
        self.assertEqual(values["gpu_cache_usage"], 0.7)

    def test_counter_reset_is_clamped_to_zero(self):
        # 行为：服务重启导致 counter 归零时，本步记 0 而不是负数
        values = BaseRLTrainer._serving_step_values(
            {"prompt_tokens": 500.0},
            {"prompt_tokens": 20.0},
        )
        self.assertEqual(values["prompt_tokens"], 0.0)

    def test_new_counter_key_after_first_step_is_zero(self):
        # 行为：首个 baseline 缺失的 key（如首次出现的 reason 拆分）记 0，不累计全量
        values = BaseRLTrainer._serving_step_values({"prompt_tokens": 10.0}, {"prompt_tokens": 15.0, "finished": 4.0})
        self.assertEqual(values["finished"], 0.0)
        self.assertEqual(values["prompt_tokens"], 5.0)

    def test_histogram_means_derive_from_deltas(self):
        previous = {"ttft_sum": 10.0, "ttft_count": 20.0, "e2e_sum": 100.0, "e2e_count": 20.0}
        current = {"ttft_sum": 16.0, "ttft_count": 30.0, "e2e_sum": 100.0, "e2e_count": 20.0}
        values = BaseRLTrainer._serving_step_values(previous, current)
        self.assertAlmostEqual(values["ttft_mean"], 6.0 / 10.0)
        self.assertAlmostEqual(values["e2e_mean"], 0.0)
        # 原始 sum/count 不直接写出
        self.assertNotIn("ttft_sum", values)
        self.assertNotIn("ttft_count", values)


class TestTrainerCollectServingMetrics(unittest.TestCase):
    def test_collects_via_ray_remote(self):
        # 行为：经 rollout controller actor 的 .remote() + ray.get 拿聚合指标
        trainer = BaseRLTrainer.__new__(BaseRLTrainer)
        trainer.rollout_controller = SimpleNamespace(collect_server_metrics=SimpleNamespace(remote=lambda: "ref"))
        with patch(
            "xtuner.v1.train.rl_trainer.ray.get",
            side_effect=lambda ref, timeout=None: {"prompt_tokens": 3.0},
        ):
            self.assertEqual(trainer._collect_serving_metrics(), {"prompt_tokens": 3.0})

    def test_controller_without_method_fails_open(self):
        trainer = BaseRLTrainer.__new__(BaseRLTrainer)
        trainer.rollout_controller = SimpleNamespace()
        self.assertEqual(trainer._collect_serving_metrics(), {})

    def test_none_result_fails_open(self):
        trainer = BaseRLTrainer.__new__(BaseRLTrainer)
        trainer.rollout_controller = SimpleNamespace(collect_server_metrics=SimpleNamespace(remote=lambda: "ref"))
        with patch("xtuner.v1.train.rl_trainer.ray.get", side_effect=lambda ref, timeout=None: None):
            self.assertEqual(trainer._collect_serving_metrics(), {})


def _bare_worker(server_url: str = "http://127.0.0.1:12345") -> RolloutWorker:
    worker = RolloutWorker.__new__(RolloutWorker)
    worker.server_url = server_url
    worker.logger = MagicMock()
    worker._last_scrape_warning_s = 0.0
    return worker


class TestWorkerScrapeMetrics(unittest.TestCase):
    def test_successful_scrape_parses_response(self):
        worker = _bare_worker()
        response = SimpleNamespace(text=FIXTURE_PATH.read_text(), raise_for_status=lambda: None)
        with patch("xtuner.v1.rl.rollout.worker.requests.get", return_value=response) as mock_get:
            metrics = worker.scrape_metrics()
        self.assertEqual(metrics["requests_running"], 2.0)
        mock_get.assert_called_once_with("http://127.0.0.1:12345/metrics", timeout=2.0)

    def test_network_failure_returns_empty_and_warns_once(self):
        # 行为：请求失败 fail-open；限频内重复失败只告警一次
        worker = _bare_worker()
        import requests as requests_module

        with patch(
            "xtuner.v1.rl.rollout.worker.requests.get",
            side_effect=requests_module.ConnectionError("down"),
        ):
            self.assertEqual(worker.scrape_metrics(), {})
            self.assertEqual(worker.scrape_metrics(), {})
        self.assertEqual(worker.logger.warning.call_count, 1)


class _FakeEndpoint:
    def __init__(self, metrics: dict[str, float] | Exception):
        self._metrics = metrics
        actor = SimpleNamespace()
        if isinstance(metrics, Exception):

            def raise_remote():
                raise metrics

            actor.scrape_metrics = SimpleNamespace(remote=raise_remote)
        else:
            actor.scrape_metrics = SimpleNamespace(remote=lambda: metrics)
        self.actor = actor


class TestControllerCollectServerMetrics(unittest.TestCase):
    def _controller(self, entrypoints) -> RolloutController:
        controller = RolloutController.__new__(RolloutController)
        controller.registry = SimpleNamespace(active_entrypoints=lambda: entrypoints)
        controller.logger = MagicMock()
        return controller

    def test_aggregates_across_entrypoints(self):
        controller = self._controller(
            [
                _FakeEndpoint({"requests_running": 2.0, "prompt_tokens": 100.0}),
                _FakeEndpoint({"requests_running": 6.0, "prompt_tokens": 50.0}),
            ]
        )
        with patch("xtuner.v1.rl.rollout.controller.ray.get", side_effect=lambda refs, timeout=None: list(refs)):
            metrics = controller.collect_server_metrics()
        self.assertEqual(metrics["requests_running"], 4.0)
        self.assertEqual(metrics["prompt_tokens"], 150.0)

    def test_no_entrypoints_returns_empty(self):
        controller = self._controller([])
        self.assertEqual(controller.collect_server_metrics(), {})

    def test_ray_failure_fails_open(self):
        # 行为：任一端点抓取抛异常时整体 fail-open，绝不向训练传播
        controller = self._controller([_FakeEndpoint(RuntimeError("actor died"))])
        with patch("xtuner.v1.rl.rollout.controller.ray.get", side_effect=RuntimeError("ray.get failed")):
            self.assertEqual(controller.collect_server_metrics(), {})


if __name__ == "__main__":
    unittest.main()
