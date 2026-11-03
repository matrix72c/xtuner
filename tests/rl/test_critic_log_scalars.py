"""Critic 训练指标聚合与 session 价值统计测试。

Good Tests:
- ``_critic_log_scalars``：clip/valid 计数跨 mini-step 求和、clip_frac 从求和派生
  （断言求和而非逐步均值），value_loss/grad_norm 仍取均值。
- ``_session_value_stats``：完美 critic EV=1、常数 critic EV=0、比均值更差时 EV<0、
  零方差与非监督 token 的 guard。

Bad Tests:
- 不构造 Ray worker / GAE estimator，只测纯聚合与纯统计函数。
"""

import unittest

from xtuner.v1.rl.trainer.controller import TrainingController, _session_value_stats


def _critic_log(steps: list[dict]) -> list[dict]:
    return [{"train_metrics": steps}]


class TestCriticLogScalars(unittest.TestCase):
    def test_clip_counters_sum_and_frac_is_count_weighted(self):
        # 行为：两个 mini-step 的计数求和；clip_frac 用和的比，而非逐步 frac 的均值
        scalars = TrainingController._critic_log_scalars(
            _critic_log(
                [
                    {
                        "reduced_llm_loss": 1.0,
                        "grad_norm": 0.5,
                        "reduced_critic_value_clip_count": 1.0,
                        "reduced_critic_value_valid_count": 3.0,
                    },
                    {
                        "reduced_llm_loss": 3.0,
                        "grad_norm": 0.7,
                        "reduced_critic_value_clip_count": 2.0,
                        "reduced_critic_value_valid_count": 10.0,
                    },
                ]
            )
        )
        self.assertEqual(scalars["critic/value_clip_count"], 3.0)
        self.assertEqual(scalars["critic/value_valid_count"], 13.0)
        # 逐步 frac 均值 = (1/3 + 0.2)/2 ≈ 0.2667，计数加权 = 3/13 ≈ 0.2308。
        self.assertAlmostEqual(scalars["critic/value_clip_frac"], 3.0 / 13.0)
        self.assertAlmostEqual(scalars["critic/value_loss"], 2.0)
        self.assertAlmostEqual(scalars["critic/grad_norm"], 0.6)

    def test_empty_logs_return_no_scalars(self):
        self.assertEqual(TrainingController._critic_log_scalars(None), {})
        self.assertEqual(TrainingController._critic_log_scalars([]), {})
        self.assertEqual(TrainingController._critic_log_scalars([{"train_metrics": []}]), {})

    def test_non_numeric_values_are_skipped(self):
        scalars = TrainingController._critic_log_scalars(
            _critic_log([{"reduced_llm_loss": "nan-string", "grad_norm": True}])
        )
        self.assertEqual(scalars, {})


class TestSessionValueStats(unittest.TestCase):
    def test_perfect_critic_explains_all_variance(self):
        # 行为：values == returns 时残差方差为 0，EV = 1
        returns = [[0.0, 1.0, 2.0], [3.0, 0.5]]
        stats = _session_value_stats(returns, returns, [[True] * 3, [True] * 2])
        self.assertAlmostEqual(stats["critic/explained_variance"], 1.0)
        self.assertAlmostEqual(stats["critic/value_mean"], stats["critic/return_mean"])
        self.assertAlmostEqual(stats["critic/value_std"], stats["critic/return_std"])

    def test_constant_critic_scores_zero(self):
        # 行为：常数 critic 的残差方差等于 return 方差，EV = 0
        returns = [[0.0, 1.0, 2.0, 3.0]]
        values = [[0.5, 0.5, 0.5, 0.5]]
        stats = _session_value_stats(values, returns, [[True] * 4])
        self.assertAlmostEqual(stats["critic/explained_variance"], 0.0)
        self.assertAlmostEqual(stats["critic/value_std"], 0.0)

    def test_anticorrelated_critic_scores_negative(self):
        # 行为：反向预测比常数均值更差，EV < 0
        returns = [[0.0, 1.0, 2.0, 3.0]]
        values = [[3.0, 2.0, 1.0, 0.0]]
        stats = _session_value_stats(values, returns, [[True] * 4])
        self.assertLess(stats["critic/explained_variance"], 0.0)

    def test_constant_returns_guard_zero_variance(self):
        returns = [[1.0, 1.0]]
        stats = _session_value_stats(returns, returns, [[True, True]])
        self.assertEqual(stats["critic/explained_variance"], 0.0)

    def test_ignored_positions_are_excluded(self):
        # 行为：mask 为 False 的位置不进统计（野值 9.0 不影响均值/EV）；全被忽略时返回 None
        values = [[9.0, 0.5, 1.5]]
        returns = [[9.0, 0.5, 1.5]]
        stats = _session_value_stats(values, returns, [[False, True, True]])
        self.assertAlmostEqual(stats["critic/value_mean"], 1.0)
        self.assertAlmostEqual(stats["critic/explained_variance"], 1.0)

        self.assertIsNone(_session_value_stats([[1.0]], [[1.0]], [[False]]))
        self.assertIsNone(_session_value_stats([], [], []))


if __name__ == "__main__":
    unittest.main()
