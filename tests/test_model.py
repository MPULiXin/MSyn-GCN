from __future__ import annotations

import unittest

import numpy as np
import torch

from msyn_gcn.data import MSynDataset
from msyn_gcn.model import sparsemax
from msyn_gcn.spec import PROJECT_ROOT, make_run_config
from msyn_gcn.trainer import build_model, set_reproducible_seed


class ModelTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        set_reproducible_seed(2025)
        cls.config = make_run_config("v4_full", 2025, "cpu")
        cls.data = MSynDataset(PROJECT_ROOT / "data" / "Set2Set", load_train_labels=True)

    def test_sparsemax_is_sparse_simplex(self) -> None:
        logits = torch.tensor([[3.0, 1.0, 0.0], [1.0, 1.0, 1.0]])
        values = sparsemax(logits, dim=1)
        self.assertTrue(torch.allclose(values.sum(dim=1), torch.ones(2)))
        self.assertEqual(int((values[0] == 0).sum()), 2)

    def test_forward_and_paper_losses(self) -> None:
        model = build_model(self.data, self.config).cpu()
        rows = self.data.splits["train"][:4]
        symptoms, herbs = self.data.make_multihot(rows)
        indices = np.arange(4)
        output = model(torch.from_numpy(symptoms))
        self.assertEqual(tuple(output.herb_logits.shape), (4, 753))
        self.assertEqual(tuple(output.eight_distribution.shape), (4, 8))
        self.assertEqual(tuple(output.zangfu_distribution.shape), (4, 12))
        self.assertTrue(torch.allclose(output.eight_distribution.sum(dim=1), torch.ones(4)))
        self.assertTrue(torch.allclose(output.zangfu_distribution.sum(dim=1), torch.ones(4)))

        total, parts, _ = model.compute_loss(
            torch.from_numpy(symptoms),
            torch.from_numpy(herbs),
            torch.from_numpy(self.data.item_weights),
            eight_labels=torch.from_numpy(self.data.train_labels[0][indices]),
            zangfu_labels=torch.from_numpy(self.data.train_labels[1][indices]),
            marginal_targets=torch.from_numpy(
                self.data.marginals[self.data.train_group_ids[indices]]
            ),
        )
        self.assertTrue(torch.isfinite(total))
        self.assertGreater(float(parts["syndrome"]), 0.0)
        self.assertGreater(float(parts["marginal"]), 0.0)
        self.assertGreater(float(parts["alignment"]), 0.0)
        total.backward()
        self.assertIsNotNone(model.symptom_embedding.grad)


if __name__ == "__main__":
    unittest.main()

