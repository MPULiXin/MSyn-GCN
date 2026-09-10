"""MSyn-GCN reproducibility package."""

import os

# Required by CUDA/cuBLAS for repeatable matrix multiplication on CUDA >= 10.2.
# It must be set before torch is imported by any package submodule.
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

from .data import MSynDataset
from .model import MSynGCN
from .spec import load_paper_spec, make_run_config

__all__ = ["MSynDataset", "MSynGCN", "load_paper_spec", "make_run_config"]
