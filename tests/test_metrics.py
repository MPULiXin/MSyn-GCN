from __future__ import annotations

import math
import unittest

import numpy as np

from msyn_gcn.metrics import standard_metrics, syndrome_sparsity


class MetricsTest(unittest.TestCase):
    def test_legacy_ndcg_normalizes_by_retrieved_hits(self) -> None:
        scores = np.asarray([[0.9, 0.8, 0.7, 0.6]], dtype=np.float32)
        metrics = standard_metrics(scores, [[0, 2]], cutoffs=(2,))
        self.assertAlmostEqual(metrics["precision@2"], 0.5)
        self.assertAlmostEqual(metrics["recall@2"], 0.5)
        expected_ndcg = 1.0
        self.assertAlmostEqual(metrics["ndcg@2"], expected_ndcg)

    def test_effective_dimensions(self) -> None:
        eight = np.full((2, 8), 1 / 8, dtype=np.float32)
        zangfu = np.zeros((2, 12), dtype=np.float32)
        zangfu[:, 0] = 1.0
        values = syndrome_sparsity(eight, zangfu)
        self.assertAlmostEqual(values["eight_effective"], 8.0, places=5)
        self.assertAlmostEqual(values["zangfu_effective"], 1.0, places=5)


if __name__ == "__main__":
    unittest.main()
