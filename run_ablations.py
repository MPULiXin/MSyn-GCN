from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from msyn_gcn.spec import PROJECT_ROOT, load_paper_spec, make_ablation_config
from msyn_gcn.trainer import train_run


ABLATIONS = [
    "without_diagnostic",
    "without_properties",
    "without_alignment",
    "average_pooling",
    "add_graph_fusion",
    "add_property_fusion",
    "concat_property_fusion",
    "v4_full",
]


def main() -> None:
    seeds = load_paper_spec()["training"]["initialization_seeds"]
    output = {}
    for ablation in ABLATIONS:
        rows = [train_run(make_ablation_config(ablation, seed, "cuda:0")) for seed in seeds]
        output[ablation] = {
            metric: float(np.mean([row["test"][metric] for row in rows]))
            for metric in rows[0]["test"]
        }
    path = Path(PROJECT_ROOT) / "results" / "ablations_summary.json"
    path.write_text(json.dumps(output, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(output, indent=2))


if __name__ == "__main__":
    main()

