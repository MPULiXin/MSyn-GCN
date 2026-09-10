from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from msyn_gcn.spec import PROJECT_ROOT, load_paper_spec, make_run_config
from msyn_gcn.trainer import train_run


def main() -> None:
    seeds = load_paper_spec()["training"]["initialization_seeds"]
    results = [train_run(make_run_config("v4_full", seed, "cuda:0")) for seed in seeds]
    metric_names = list(results[0]["test"])
    summary = {
        metric: {
            "mean": float(np.mean([row["test"][metric] for row in results])),
            "std": float(np.std([row["test"][metric] for row in results], ddof=1)),
        }
        for metric in metric_names
    }
    output = Path(PROJECT_ROOT) / "results" / "v4_full" / "five_seed_summary.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()

