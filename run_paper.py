from __future__ import annotations

import argparse
import json

from msyn_gcn.spec import make_run_config, make_smoke_config
from msyn_gcn.trainer import train_run


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the paper-aligned MSyn-GCN pipeline")
    parser.add_argument("--mode", choices=["smoke", "train"], default="smoke")
    parser.add_argument(
        "--variant",
        choices=["v1_base", "v2_sparsemax", "v3_syndrome", "v4_full"],
        default="v4_full",
    )
    parser.add_argument("--seed", type=int, choices=range(2025, 2030), default=2025)
    parser.add_argument("--device", default="cuda:0")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.mode == "smoke":
        config = make_smoke_config(device=args.device)
    else:
        config = make_run_config(args.variant, args.seed, args.device)

    result = train_run(config)
    print(json.dumps(result["test"], indent=2))


if __name__ == "__main__":
    main()
