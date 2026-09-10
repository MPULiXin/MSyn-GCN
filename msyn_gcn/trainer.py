from __future__ import annotations

import hashlib
import json
import random
import time
import warnings
from pathlib import Path
from typing import Any

import numpy as np
import torch

from .data import MSynDataset
from .metrics import ambiguity_aware_metrics, standard_metrics, syndrome_sparsity
from .model import MSynGCN
from .spec import load_paper_spec


def set_reproducible_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(True, warn_only=True)
    warnings.filterwarnings(
        "ignore",
        message="cumsum_cuda_kernel does not have a deterministic implementation",
    )


def resolve_device(requested: str) -> torch.device:
    if requested.startswith("cuda") and torch.cuda.is_available():
        return torch.device(requested)
    if requested.startswith("cuda"):
        print("[device] CUDA unavailable; falling back to CPU")
    return torch.device("cpu")


def config_digest(config: dict[str, Any]) -> str:
    payload = json.dumps(config, ensure_ascii=False, sort_keys=True).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def build_model(data: MSynDataset, config: dict[str, Any]) -> MSynGCN:
    return MSynGCN(
        n_symptoms=data.n_symptoms,
        n_herbs=data.n_herbs,
        bipartite_adj=data.bipartite_adj,
        symptom_adj=data.symptom_adj,
        herb_adj=data.herb_adj,
        properties=data.properties,
        model_config=config["model"],
        training_config=config["training"],
        rule_matrices=config["rule_matrices"],
    )


@torch.no_grad()
def predict_split(
    model: MSynGCN,
    data: MSynDataset,
    split: str,
    device: torch.device,
    *,
    batch_size: int,
    limit: int | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, list]:
    model.eval()
    score_rows = []
    eight_rows = []
    zangfu_rows = []
    selected_rows = []
    for _, rows, symptom_matrix in data.iter_split_batches(
        split,
        batch_size,
        limit=limit,
    ):
        symptoms = torch.from_numpy(symptom_matrix).to(device)
        output = model(symptoms)
        score_rows.append(output.herb_logits.detach().cpu().numpy())
        eight_rows.append(output.eight_distribution.detach().cpu().numpy())
        zangfu_rows.append(output.zangfu_distribution.detach().cpu().numpy())
        selected_rows.extend(rows)
    return (
        np.concatenate(score_rows, axis=0),
        np.concatenate(eight_rows, axis=0),
        np.concatenate(zangfu_rows, axis=0),
        selected_rows,
    )


def _write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def train_run(config: dict[str, Any]) -> dict[str, Any]:
    seed = int(config["seed"])
    set_reproducible_seed(seed)
    device = resolve_device(str(config["device"]))
    training = config["training"]
    evaluation = config["evaluation"]
    use_train_labels = float(training["lambda_syndrome"]) > 0
    data = MSynDataset(config["dataset_dir"], load_train_labels=use_train_labels)
    model = build_model(data, config).to(device)
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=float(training["learning_rate"]),
        weight_decay=float(training["weight_decay"]),
    )
    item_weights = torch.from_numpy(data.item_weights).to(device)
    rng = np.random.default_rng(seed)

    run_dir = Path(config["output_dir"]) / config["variant"] / f"seed_{seed}"
    run_dir.mkdir(parents=True, exist_ok=True)
    config_path = run_dir / "config.json"
    history_path = run_dir / "history.jsonl"
    checkpoint_path = run_dir / "best_checkpoint.pt"
    _write_json(config_path, config)
    history_path.write_text("", encoding="utf-8")

    max_epochs = int(training["max_epochs"])
    eval_every = int(training["evaluation_interval"])
    patience_limit = int(training["early_stopping_patience"])
    min_delta = float(training["early_stopping_min_delta"])
    batch_size = int(training["batch_size"])
    max_batches = training.get("max_batches_per_epoch")
    evaluation_limit = training.get("evaluation_limit")
    cutoffs = tuple(int(value) for value in evaluation["cutoffs"])
    selection_metric = str(evaluation["checkpoint_metric"])

    best_value = -float("inf")
    best_epoch = 0
    patience = 0
    validation_evaluations = 0
    started = time.time()

    for epoch in range(1, max_epochs + 1):
        model.train()
        loss_sums = {
            "total": 0.0,
            "reconstruction": 0.0,
            "syndrome": 0.0,
            "marginal": 0.0,
            "alignment": 0.0,
            "regularization": 0.0,
        }
        batches = 0
        for batch in data.iter_train_batches(batch_size, rng, max_batches=max_batches):
            symptoms = torch.from_numpy(batch["symptoms"]).to(device)
            herbs = torch.from_numpy(batch["herbs"]).to(device)
            marginals = torch.from_numpy(batch["marginals"]).to(device)
            eight_labels = (
                torch.from_numpy(batch["eight_labels"]).to(device)
                if "eight_labels" in batch
                else None
            )
            zangfu_labels = (
                torch.from_numpy(batch["zangfu_labels"]).to(device)
                if "zangfu_labels" in batch
                else None
            )
            optimizer.zero_grad(set_to_none=True)
            total, parts, _ = model.compute_loss(
                symptoms,
                herbs,
                item_weights,
                eight_labels=eight_labels,
                zangfu_labels=zangfu_labels,
                marginal_targets=marginals,
            )
            total.backward()
            optimizer.step()
            for name, value in parts.items():
                loss_sums[name] += float(value.cpu())
            batches += 1
        mean_losses = {name: value / max(1, batches) for name, value in loss_sums.items()}

        if epoch % eval_every != 0 and epoch != max_epochs:
            continue
        validation_evaluations += 1
        val_scores, _, _, val_rows = predict_split(
            model,
            data,
            "val",
            device,
            batch_size=batch_size,
            limit=evaluation_limit,
        )
        val_metrics = standard_metrics(
            val_scores,
            [row.herbs for row in val_rows],
            cutoffs,
        )
        record = {
            "epoch": epoch,
            "loss": mean_losses,
            "validation": val_metrics,
            "elapsed_seconds": time.time() - started,
        }
        with history_path.open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(record, ensure_ascii=False) + "\n")
        print(
            f"[epoch {epoch:04d}] val P@5={val_metrics['precision@5']:.5f} "
            f"R@5={val_metrics['recall@5']:.5f} NDCG@5={val_metrics['ndcg@5']:.5f} "
            f"loss={mean_losses['total']:.4f}"
        )

        candidate = float(val_metrics[selection_metric])
        if candidate > best_value + min_delta:
            best_value = candidate
            best_epoch = epoch
            patience = 0
            torch.save(
                {
                    "format_version": 1,
                    "epoch": epoch,
                    "variant": config["variant"],
                    "seed": seed,
                    "validation_metrics": val_metrics,
                    "config_digest": config_digest(config),
                    "model_state": model.state_dict(),
                },
                checkpoint_path,
            )
            print(f"  [checkpoint] selected by validation Precision@5 at epoch {epoch}")
        else:
            patience += 1
            if patience_limit > 0 and patience >= patience_limit:
                print(f"[early stop] {patience_limit} validation evaluations without >= {min_delta:g} gain")
                break

    if not checkpoint_path.exists():
        raise RuntimeError("No validation checkpoint was created")
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model_state"])

    # The test set is predicted exactly once, after all model selection is complete.
    test_scores, test_eight, test_zangfu, test_rows = predict_split(
        model,
        data,
        "test",
        device,
        batch_size=batch_size,
        limit=evaluation_limit,
    )
    test_metrics = standard_metrics(test_scores, [row.herbs for row in test_rows], cutoffs)
    ambiguity = ambiguity_aware_metrics(test_scores, test_rows, cutoffs)
    sparsity = syndrome_sparsity(
        test_eight,
        test_zangfu,
        float(evaluation["active_dimension_threshold"]),
    )
    label_eight, label_zangfu = data.load_labels("test")
    if evaluation_limit is not None:
        label_eight = label_eight[:evaluation_limit]
        label_zangfu = label_zangfu[:evaluation_limit]
    label_sparsity = syndrome_sparsity(
        label_eight,
        label_zangfu,
        float(evaluation["active_dimension_threshold"]),
    )
    paper_reference = load_paper_spec()["paper_reference"]["v4_full_mean"]
    result = {
        "variant": config["variant"],
        "seed": seed,
        "device": str(device),
        "best_epoch": best_epoch,
        "best_validation_precision@5": best_value,
        "validation_evaluations": validation_evaluations,
        "test_evaluations": 1,
        "test": test_metrics,
        "test_ambiguity_aware": ambiguity,
        "test_syndrome_sparsity": sparsity,
        "test_weak_label_reference_sparsity": label_sparsity,
        "paper_v4_mean_reference": paper_reference,
        "elapsed_seconds": time.time() - started,
        "config_digest": config_digest(config),
        "dataset_audit": data.audit(),
    }
    _write_json(run_dir / "result.json", result)
    print(
        f"[test once] P@5={test_metrics['precision@5']:.5f} "
        f"R@5={test_metrics['recall@5']:.5f} NDCG@5={test_metrics['ndcg@5']:.5f}"
    )
    return result


def evaluate_checkpoint(
    checkpoint_path: str | Path,
    config: dict[str, Any],
) -> dict[str, Any]:
    """Explicit post-training evaluation; never used by the training selector."""
    device = resolve_device(str(config["device"]))
    data = MSynDataset(config["dataset_dir"], load_train_labels=False)
    model = build_model(data, config).to(device)
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    if checkpoint.get("config_digest") != config_digest(config):
        raise ValueError("Checkpoint/config digest mismatch")
    model.load_state_dict(checkpoint["model_state"])
    scores, eight, zangfu, rows = predict_split(
        model,
        data,
        "test",
        device,
        batch_size=int(config["training"]["batch_size"]),
    )
    return {
        "standard": standard_metrics(scores, [row.herbs for row in rows], config["evaluation"]["cutoffs"]),
        "ambiguity_aware": ambiguity_aware_metrics(scores, rows, config["evaluation"]["cutoffs"]),
        "sparsity": syndrome_sparsity(eight, zangfu),
    }
