from __future__ import annotations

import copy
import json
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SPEC_PATH = PROJECT_ROOT / "paper" / "paper_spec.json"


def load_paper_spec() -> dict[str, Any]:
    """Load the machine-readable specification transcribed from the manuscript."""
    return json.loads(SPEC_PATH.read_text(encoding="utf-8"))


def make_run_config(
    variant: str = "v4_full",
    seed: int = 2025,
    device: str = "cuda:0",
    *,
    max_epochs: int | None = None,
    evaluation_interval: int | None = None,
    early_stopping_patience: int | None = None,
) -> dict[str, Any]:
    spec = load_paper_spec()
    if variant not in spec["variants"]:
        choices = ", ".join(sorted(spec["variants"]))
        raise ValueError(f"Unknown variant {variant!r}; choose one of: {choices}")
    if seed not in spec["training"]["initialization_seeds"]:
        raise ValueError(
            f"Seed {seed} is outside the paper protocol: "
            f"{spec['training']['initialization_seeds']}"
        )

    config = {
        "variant": variant,
        "seed": int(seed),
        "device": device,
        "dataset_dir": str(PROJECT_ROOT / "data" / "Set2Set"),
        "output_dir": str(PROJECT_ROOT / "results"),
        "model": copy.deepcopy(spec["model"]),
        "training": copy.deepcopy(spec["training"]),
        "evaluation": copy.deepcopy(spec["evaluation"]),
        "rule_matrices": copy.deepcopy(spec["rule_matrices"]),
    }
    config["model"]["syndrome_activation"] = spec["variants"][variant]["syndrome_activation"]
    config["training"]["lambda_syndrome"] = spec["variants"][variant]["lambda_syndrome"]
    config["training"]["lambda_marginal"] = spec["variants"][variant]["lambda_marginal"]

    if max_epochs is not None:
        config["training"]["max_epochs"] = int(max_epochs)
    if evaluation_interval is not None:
        config["training"]["evaluation_interval"] = int(evaluation_interval)
    if early_stopping_patience is not None:
        config["training"]["early_stopping_patience"] = int(early_stopping_patience)
    return config


def make_smoke_config(device: str = "cuda:0") -> dict[str, Any]:
    config = make_run_config(
        variant="v4_full",
        seed=2025,
        device=device,
        max_epochs=1,
        evaluation_interval=1,
        early_stopping_patience=0,
    )
    config["smoke_test"] = True
    config["training"]["max_batches_per_epoch"] = 2
    config["training"]["evaluation_limit"] = 128
    return config


def make_ablation_config(
    ablation: str,
    seed: int = 2025,
    device: str = "cuda:0",
) -> dict[str, Any]:
    """Construct one of the component ablations defined in manuscript Table 9."""
    config = make_run_config("v4_full", seed, device)
    config["variant"] = ablation
    if ablation == "without_diagnostic":
        config["model"]["diagnostic_enabled"] = False
        config["training"]["lambda_syndrome"] = 0.0
        config["training"]["lambda_align"] = 0.0
    elif ablation == "without_properties":
        # Property matrices remain available to L_align; only the herb representation drops H_prop.
        config["model"]["use_property_representation"] = False
    elif ablation == "without_alignment":
        config["training"]["lambda_align"] = 0.0
    elif ablation == "average_pooling":
        config["model"]["symptom_pooling"] = "average"
    elif ablation == "add_graph_fusion":
        config["model"]["graph_fusion"] = "add"
        config["model"]["mlp_sizes"] = [config["model"]["graph_layer_sizes"][-1]]
    elif ablation == "add_property_fusion":
        config["model"]["property_fusion"] = "add"
    elif ablation == "concat_property_fusion":
        config["model"]["property_fusion"] = "concat"
    elif ablation != "v4_full":
        raise ValueError(f"Unknown ablation: {ablation}")
    return config
