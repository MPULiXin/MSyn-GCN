from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from scipy.stats import ttest_rel

from msyn_gcn.spec import PROJECT_ROOT, load_paper_spec, make_run_config
from msyn_gcn.trainer import train_run


def main() -> None:
    spec = load_paper_spec()
    seeds = spec["training"]["initialization_seeds"]
    variants = list(spec["variants"])
    runs = {
        variant: [train_run(make_run_config(variant, seed, "cuda:0")) for seed in seeds]
        for variant in variants
    }
    summary = {}
    for variant, rows in runs.items():
        summary[variant] = {}
        for metric in rows[0]["test"]:
            values = np.asarray([row["test"][metric] for row in rows], dtype=np.float64)
            summary[variant][metric] = {
                "mean": float(values.mean()),
                "std": float(values.std(ddof=1)),
            }
    comparisons = {}
    comparison_pairs = [
        ("v3_syndrome", "v1_base"),
        ("v4_full", "v1_base"),
        ("v4_full", "v3_syndrome"),
    ]
    for left_name, right_name in comparison_pairs:
        comparison_name = f"{left_name}_vs_{right_name}"
        comparisons[comparison_name] = {}
        left = runs[left_name]
        right = runs[right_name]
        for metric in left[0]["test"]:
            left_values = [row["test"][metric] for row in left]
            right_values = [row["test"][metric] for row in right]
            statistic, p_value = ttest_rel(left_values, right_values)
            comparisons[comparison_name][metric] = {
                "mean_difference": float(np.mean(np.asarray(left_values) - np.asarray(right_values))),
                "t": float(statistic),
                "p_uncorrected": float(p_value),
                "p_bonferroni_9": float(min(1.0, p_value * 9)),
            }
    output = {
        "summary": summary,
        "paired_two_sided_tests": comparisons,
        "seeds": seeds,
    }
    path = Path(PROJECT_ROOT) / "results" / "variants_summary.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(output, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(output, indent=2))


if __name__ == "__main__":
    main()
