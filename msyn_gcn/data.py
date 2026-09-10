from __future__ import annotations

import hashlib
import itertools
import json
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator

import numpy as np
import pandas as pd
from scipy import sparse

from .spec import load_paper_spec


@dataclass(frozen=True)
class Prescription:
    symptoms: tuple[int, ...]
    herbs: tuple[int, ...]


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _read_mapping(path: Path) -> dict[int, str]:
    mapping: dict[int, str] = {}
    for line_no, raw in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if not raw.strip():
            continue
        try:
            name, raw_id = raw.rsplit(maxsplit=1)
            node_id = int(raw_id)
        except Exception as exc:
            raise ValueError(f"Invalid mapping row {path}:{line_no}: {raw!r}") from exc
        if node_id in mapping:
            raise ValueError(f"Duplicate id {node_id} in {path}")
        mapping[node_id] = name
    if sorted(mapping) != list(range(len(mapping))):
        raise ValueError(f"IDs in {path} must be contiguous from zero")
    return mapping


def _read_split(path: Path, n_symptoms: int, n_herbs: int) -> list[Prescription]:
    rows: list[Prescription] = []
    for line_no, raw in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if not raw.strip():
            continue
        if "\t" not in raw:
            raise ValueError(f"Missing tab separator at {path}:{line_no}")
        left, right = raw.split("\t", 1)
        symptom_ids = tuple(int(value) for value in left.split())
        herb_ids = tuple(int(value) for value in right.split())
        if not symptom_ids or not herb_ids:
            raise ValueError(f"Empty symptom/herb set at {path}:{line_no}")
        if len(set(symptom_ids)) != len(symptom_ids) or len(set(herb_ids)) != len(herb_ids):
            raise ValueError(f"Duplicate id inside prescription at {path}:{line_no}")
        if min(symptom_ids) < 0 or max(symptom_ids) >= n_symptoms:
            raise ValueError(f"Symptom id out of range at {path}:{line_no}")
        if min(herb_ids) < 0 or max(herb_ids) >= n_herbs:
            raise ValueError(f"Herb id out of range at {path}:{line_no}")
        rows.append(Prescription(symptom_ids, herb_ids))
    return rows


def _row_normalize(matrix: sparse.csr_matrix) -> sparse.csr_matrix:
    row_sums = np.asarray(matrix.sum(axis=1)).ravel()
    inverse = np.zeros_like(row_sums, dtype=np.float32)
    nonzero = row_sums != 0
    inverse[nonzero] = 1.0 / row_sums[nonzero]
    return sparse.diags(inverse, format="csr") @ matrix


class MSynDataset:
    """Strict three-split loader. All graph and frequency statistics use train only."""

    def __init__(self, dataset_dir: str | Path, *, load_train_labels: bool = True):
        self.root = Path(dataset_dir).resolve()
        if not self.root.is_dir():
            raise FileNotFoundError(f"Dataset directory not found: {self.root}")
        self.spec = load_paper_spec()
        self.symptom_names = _read_mapping(self.root / "symptom_mapping.txt")
        self.herb_names = _read_mapping(self.root / "herb_mapping.txt")
        self.n_symptoms = len(self.symptom_names)
        self.n_herbs = len(self.herb_names)

        self.splits = {
            name: _read_split(self.root / f"{name}.txt", self.n_symptoms, self.n_herbs)
            for name in ("train", "val", "test")
        }
        self._validate_paper_counts()
        self.properties = self._load_properties()
        self.bipartite_adj, self.symptom_adj, self.herb_adj = self._build_train_graphs()
        self.item_weights = self._build_item_weights()
        self._build_marginals()
        self.train_labels = self.load_labels("train") if load_train_labels else None

    def _validate_paper_counts(self) -> None:
        expected = self.spec["dataset"]
        if self.n_symptoms != expected["symptom_vocabulary_size"]:
            raise ValueError(f"Expected 360 symptoms, found {self.n_symptoms}")
        if self.n_herbs != expected["herb_vocabulary_size"]:
            raise ValueError(f"Expected 753 herbs, found {self.n_herbs}")
        for split, count in expected["split_counts"].items():
            actual = len(self.splits[split])
            if actual != count:
                raise ValueError(f"Expected {count} {split} rows, found {actual}")

        locations: dict[tuple[tuple[int, ...], tuple[int, ...]], tuple[str, int]] = {}
        for split, rows in self.splits.items():
            for index, row in enumerate(rows):
                key = (tuple(sorted(row.symptoms)), tuple(sorted(row.herbs)))
                if key in locations:
                    raise ValueError(f"Exact-pair leakage: {locations[key]} and {(split, index)}")
                locations[key] = (split, index)

    def _load_property_file(self, filename: str, columns: list[str]) -> np.ndarray:
        path = self.root / filename
        frame = pd.read_excel(path)
        if list(frame.columns[2:]) != columns:
            raise ValueError(
                f"Unexpected property columns in {filename}: {list(frame.columns[2:])}; "
                f"expected {columns}"
            )
        ids = frame.iloc[:, 1].astype(int).to_numpy()
        names = frame.iloc[:, 0].astype(str).to_numpy()
        if sorted(ids.tolist()) != list(range(self.n_herbs)):
            raise ValueError(f"{filename} must contain every herb id exactly once")
        values = frame.iloc[:, 2:].fillna(0).to_numpy(dtype=np.float32)
        if not np.isin(values, [0.0, 1.0]).all():
            raise ValueError(f"{filename} contains non-binary property cells")
        result = np.zeros((self.n_herbs, len(columns)), dtype=np.float32)
        for herb_id, name, vector in zip(ids, names, values):
            if self.herb_names[int(herb_id)] != name:
                raise ValueError(
                    f"Name mismatch for herb {herb_id}: mapping={self.herb_names[int(herb_id)]!r}, "
                    f"workbook={name!r}"
                )
            result[int(herb_id)] = vector
        return result

    def _load_properties(self) -> dict[str, np.ndarray]:
        categories = self.spec["categories"]
        return {
            "qi": self._load_property_file("herb_property_qi.xlsx", categories["qi"]),
            "flavor": self._load_property_file("herb_property_flavor.xlsx", categories["flavor"]),
            "meridian": self._load_property_file(
                "herb_property_meridian.xlsx", categories["zangfu"]
            ),
        }

    def _build_train_graphs(self) -> tuple[sparse.csr_matrix, sparse.csr_matrix, sparse.csr_matrix]:
        interaction = sparse.lil_matrix((self.n_symptoms, self.n_herbs), dtype=np.float32)
        symptom_edges: set[tuple[int, int]] = set()
        herb_edges: set[tuple[int, int]] = set()
        for row in self.splits["train"]:
            for symptom_id in row.symptoms:
                interaction[symptom_id, list(row.herbs)] = 1.0
            symptom_edges.update(itertools.combinations(sorted(row.symptoms), 2))
            herb_edges.update(itertools.combinations(sorted(row.herbs), 2))

        interaction = interaction.tocsr()
        block = sparse.bmat(
            [[None, interaction], [interaction.transpose(), None]],
            format="csr",
            dtype=np.float32,
        )
        block = _row_normalize(block + sparse.eye(block.shape[0], dtype=np.float32, format="csr"))

        def undirected_adjacency(size: int, edges: set[tuple[int, int]]) -> sparse.csr_matrix:
            if not edges:
                return sparse.csr_matrix((size, size), dtype=np.float32)
            pairs = np.asarray(sorted(edges), dtype=np.int64)
            rows = np.concatenate([pairs[:, 0], pairs[:, 1]])
            cols = np.concatenate([pairs[:, 1], pairs[:, 0]])
            values = np.ones(len(rows), dtype=np.float32)
            return sparse.coo_matrix((values, (rows, cols)), shape=(size, size)).tocsr()

        return (
            block,
            undirected_adjacency(self.n_symptoms, symptom_edges),
            undirected_adjacency(self.n_herbs, herb_edges),
        )

    def _build_item_weights(self) -> np.ndarray:
        frequency = np.zeros(self.n_herbs, dtype=np.float32)
        for row in self.splits["train"]:
            frequency[list(row.herbs)] += 1.0
        safe_frequency = np.maximum(frequency, 1.0)
        return (safe_frequency.max() / safe_frequency).astype(np.float32)

    def _build_marginals(self) -> None:
        grouped: dict[tuple[int, ...], list[Prescription]] = defaultdict(list)
        for row in self.splits["train"]:
            grouped[tuple(sorted(row.symptoms))].append(row)
        keys = sorted(grouped)
        key_to_group = {key: index for index, key in enumerate(keys)}
        marginals = np.zeros((len(keys), self.n_herbs), dtype=np.float32)
        for key, rows in grouped.items():
            group_index = key_to_group[key]
            for row in rows:
                marginals[group_index, list(row.herbs)] += 1.0
            marginals[group_index] /= float(len(rows))
        self.marginals = marginals
        self.train_group_ids = np.asarray(
            [key_to_group[tuple(sorted(row.symptoms))] for row in self.splits["train"]],
            dtype=np.int64,
        )

    def load_labels(self, split: str) -> tuple[np.ndarray, np.ndarray]:
        if split not in self.splits:
            raise ValueError(f"Unknown split: {split}")
        label_dir = self.root / "labels"
        eight_path = label_dir / f"eight_{split}.npy"
        zangfu_path = label_dir / f"zangfu_{split}.npy"
        metadata = json.loads((label_dir / "metadata.json").read_text(encoding="utf-8"))
        split_metadata = metadata["splits"][split]
        if metadata["eight_names"] != self.spec["categories"]["eight_principles"]:
            raise ValueError("Eight-Principles label column order does not match paper_spec.json")
        if metadata["zangfu_names"] != self.spec["categories"]["zangfu"]:
            raise ValueError("Zang-Fu label column order does not match paper_spec.json")
        if split_metadata["data_sha256"] != _sha256(self.root / f"{split}.txt"):
            raise ValueError(f"{split} labels are not keyed to the current {split}.txt")
        if split_metadata["eight_sha256"] != _sha256(eight_path):
            raise ValueError(f"Checksum mismatch for {eight_path}")
        if split_metadata["zangfu_sha256"] != _sha256(zangfu_path):
            raise ValueError(f"Checksum mismatch for {zangfu_path}")
        eight = np.load(eight_path).astype(np.float32, copy=False)
        zangfu = np.load(zangfu_path).astype(np.float32, copy=False)
        if eight.shape != (len(self.splits[split]), 8):
            raise ValueError(f"Incorrect Eight-Principles label shape for {split}: {eight.shape}")
        if zangfu.shape != (len(self.splits[split]), 12):
            raise ValueError(f"Incorrect Zang-Fu label shape for {split}: {zangfu.shape}")
        if not np.allclose(eight.sum(axis=1), 1.0, atol=1e-5):
            raise ValueError(f"Eight-Principles labels for {split} are not normalized")
        if not np.allclose(zangfu.sum(axis=1), 1.0, atol=1e-5):
            raise ValueError(f"Zang-Fu labels for {split} are not normalized")
        return eight, zangfu

    def make_multihot(self, rows: list[Prescription]) -> tuple[np.ndarray, np.ndarray]:
        symptoms = np.zeros((len(rows), self.n_symptoms), dtype=np.float32)
        herbs = np.zeros((len(rows), self.n_herbs), dtype=np.float32)
        for index, row in enumerate(rows):
            symptoms[index, list(row.symptoms)] = 1.0
            herbs[index, list(row.herbs)] = 1.0
        return symptoms, herbs

    def iter_train_batches(
        self,
        batch_size: int,
        rng: np.random.Generator,
        *,
        max_batches: int | None = None,
    ) -> Iterator[dict[str, np.ndarray]]:
        order = rng.permutation(len(self.splits["train"]))
        emitted = 0
        for start in range(0, len(order), batch_size):
            if max_batches is not None and emitted >= max_batches:
                break
            indices = order[start : start + batch_size]
            rows = [self.splits["train"][int(index)] for index in indices]
            symptom_matrix, herb_matrix = self.make_multihot(rows)
            batch = {
                "indices": indices,
                "symptoms": symptom_matrix,
                "herbs": herb_matrix,
                "marginals": self.marginals[self.train_group_ids[indices]],
            }
            if self.train_labels is not None:
                batch["eight_labels"] = self.train_labels[0][indices]
                batch["zangfu_labels"] = self.train_labels[1][indices]
            yield batch
            emitted += 1

    def iter_split_batches(
        self,
        split: str,
        batch_size: int,
        *,
        limit: int | None = None,
    ) -> Iterator[tuple[int, list[Prescription], np.ndarray]]:
        rows = self.splits[split]
        if limit is not None:
            rows = rows[:limit]
        for start in range(0, len(rows), batch_size):
            batch_rows = rows[start : start + batch_size]
            symptom_matrix, _ = self.make_multihot(batch_rows)
            yield start, batch_rows, symptom_matrix

    def audit(self) -> dict[str, object]:
        train_sets = {tuple(sorted(row.symptoms)) for row in self.splits["train"]}
        split_summary = {}
        for split, rows in self.splits.items():
            symptom_sets = [tuple(sorted(row.symptoms)) for row in rows]
            split_summary[split] = {
                "prescriptions": len(rows),
                "used_symptoms": len({value for row in rows for value in row.symptoms}),
                "used_herbs": len({value for row in rows for value in row.herbs}),
                "unique_symptom_sets": len(set(symptom_sets)),
                "seen_in_train_fraction": (
                    None
                    if split == "train"
                    else sum(key in train_sets for key in symptom_sets) / len(rows)
                ),
            }
        return {
            "splits": split_summary,
            "graph_edges": {
                "symptom": int(self.symptom_adj.nnz // 2),
                "herb": int(self.herb_adj.nnz // 2),
            },
            "property_column_sums": {
                name: values.sum(axis=0).astype(int).tolist()
                for name, values in self.properties.items()
            },
            "checksums": {
                path.name: _sha256(path)
                for path in sorted(self.root.iterdir())
                if path.is_file()
            },
        }

    def write_audit(self, path: str | Path) -> None:
        Path(path).write_text(
            json.dumps(self.audit(), ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
