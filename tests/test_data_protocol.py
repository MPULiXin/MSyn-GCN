from __future__ import annotations

import unittest
from collections import Counter

import numpy as np

from msyn_gcn.data import MSynDataset
from msyn_gcn.spec import PROJECT_ROOT


class DataProtocolTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.data = MSynDataset(PROJECT_ROOT / "data" / "Set2Set", load_train_labels=True)

    def test_paper_split_counts(self) -> None:
        self.assertEqual({name: len(rows) for name, rows in self.data.splits.items()}, {
            "train": 20625,
            "val": 2292,
            "test": 3443,
        })
        self.assertEqual(self.data.n_symptoms, 360)
        self.assertEqual(self.data.n_herbs, 753)

    def test_graphs_are_train_only_and_use_any_cooccurrence(self) -> None:
        self.assertEqual(self.data.symptom_adj.nnz // 2, 4462)
        self.assertEqual(self.data.herb_adj.nnz // 2, 50419)
        self.assertEqual(float(self.data.symptom_adj.diagonal().sum()), 0.0)
        self.assertEqual(float(self.data.herb_adj.diagonal().sum()), 0.0)

        train_edges = set()
        for row in self.data.splits["train"]:
            ids = sorted(row.symptoms)
            train_edges.update((ids[i], ids[j]) for i in range(len(ids)) for j in range(i + 1, len(ids)))
        validation_only = None
        for row in self.data.splits["val"]:
            ids = sorted(row.symptoms)
            for i in range(len(ids)):
                for j in range(i + 1, len(ids)):
                    if (ids[i], ids[j]) not in train_edges:
                        validation_only = (ids[i], ids[j])
                        break
                if validation_only:
                    break
            if validation_only:
                break
        self.assertIsNotNone(validation_only)
        self.assertEqual(float(self.data.symptom_adj[validation_only]), 0.0)

    def test_weak_labels_align_only_to_train_objective(self) -> None:
        eight, zangfu = self.data.train_labels
        self.assertEqual(eight.shape, (20625, 8))
        self.assertEqual(zangfu.shape, (20625, 12))
        self.assertTrue(np.allclose(eight.sum(axis=1), 1.0))
        self.assertTrue(np.allclose(zangfu.sum(axis=1), 1.0))

    def test_marginal_is_fraction_of_group_prescriptions(self) -> None:
        groups = Counter(tuple(sorted(row.symptoms)) for row in self.data.splits["train"])
        repeated_key = next(key for key, count in groups.items() if count > 1)
        indices = [
            index
            for index, row in enumerate(self.data.splits["train"])
            if tuple(sorted(row.symptoms)) == repeated_key
        ]
        group_id = int(self.data.train_group_ids[indices[0]])
        target = self.data.marginals[group_id]
        herb = self.data.splits["train"][indices[0]].herbs[0]
        expected = sum(
            herb in self.data.splits["train"][index].herbs for index in indices
        ) / len(indices)
        self.assertAlmostEqual(float(target[herb]), expected)
        self.assertGreater(float(target.sum()), 1.0)


if __name__ == "__main__":
    unittest.main()

