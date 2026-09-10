# MSyn-GCN

Code accompanying **MSyn-GCN: a knowledge-guided multi-graph network with LLM weak supervision for herb recommendation** by Xin Li and Wuman Luo.

Repository: <https://github.com/MPULiXin/MSyn-GCN>  
Prepared version: **1.1.0** · Intended release tag: **v1.1.0** · Software license: **MIT**

MSyn-GCN combines symptom–symptom, symptom–herb, and herb–herb graphs with herb-property representations and LLM-generated Eight-Principles and Zang-Fu weak supervision. It is research software, not a clinically validated diagnostic or prescribing system. The new GitHub release and its Zenodo DOI have not yet been published.

## Repository contents

```text
msyn_gcn/                    # data loading, model, metrics, training, configuration
data/Set2Set/                # fixed splits, mappings, property matrices, weak labels
data/herb_properties.xlsx    # 753-herb provenance table (Additional file 2)
paper/                      # paper configuration, prompt, generation provenance
tests/                      # unit tests
results/                    # generated automatically when scripts write outputs
run_paper.py                # single-run entry point
run_five_seeds.py            # full model, seeds 2025–2029
run_variants.py              # V1–V4 and paired comparisons
run_ablations.py             # architectural ablations
run_tests.py                # non-training implementation checks
validate_project.py         # fixed-data audit
generate_weak_labels.py      # optional external-API label generation
requirements.txt
.gitignore
LICENSE
CITATION.cff
README.md
```

The fixed dataset contains **20,625 training, 2,292 validation, and 3,443 test prescriptions**, with vocabularies of **360 symptoms and 753 herbs**. The supplied weak labels can be used without an LLM API call.

## Setup and non-training checks

Use Python **3.10**, preferably in an isolated environment. From the repository root:

```bash
python -m pip install -r requirements.txt
python run_tests.py
python validate_project.py
```

The tests check data/graph construction, weak-label alignment, marginal targets, metrics, sparsemax, and a four-record forward/loss/backward pass **without optimizer updates**. The validator writes `results/data_audit.json`. These checks do not retrain the model or reproduce the manuscript's numerical results. No API key is required; the OpenAI-compatible client in the dependency list is used only for optional label generation.

## Training (optional)

The following commands **start new training runs**. A CUDA-capable GPU is recommended; the multi-run scripts select `cuda:0`.

```bash
python run_paper.py --mode train --variant v4_full --seed 2025 --device cuda:0
python run_five_seeds.py
python run_variants.py
python run_ablations.py
```

Settings are recorded in `paper/paper_spec.json`. Validation is performed every 10 epochs; early stopping uses ten checks without a Precision@5 improvement exceeding `1e-4`. The checkpoint is selected by validation Precision@5 and then evaluated once on the test split. Five-seed runs use 2025–2029. V3 vs V1, V4 vs V1, and V4 vs V3 use paired comparisons with Bonferroni correction across nine metrics within each comparison family.

Outputs go to `results/`. The configuration's `paper_reference` values are manuscript reference numbers, not stored run outputs. `msyn_gcn/metrics.py` retains the historical hit-normalized NDCG convention, which differs from standard target-size-normalized NDCG. The optional `--mode smoke --device cpu` run also performs optimizer updates; it is not a non-training check.

## Weak-label provenance

The supplied labels were generated on **24–25 June 2026 (UTC+08:00)** through **Alibaba Cloud Model Studio (Bailian/DashScope), China region**, using `https://dashscope.aliyuncs.com/compatible-mode/v1`. The recorded request model was **`qwen-plus`**. The archived metadata reconstruct its snapshot as **`qwen-plus-2025-12-01`** from the provider's alias mapping; the snapshot identifier was not captured in a local API response log.

The exact prompt, historical generator, and provenance are retained in `paper/llm_prompt_zh.txt`, `paper/llm_syndrome_label_20260624.py`, and `paper/llm_generation_metadata.json`. The original 22,917 training-label rows were partitioned with the same indices as the final training/validation data. Array and split-file hashes are retained in `data/Set2Set/labels/metadata.json`.

Only training weak labels enter the training objective; validation/test labels are reserved for descriptive post-training analyses, not checkpoint selection. These are computational weak labels, not expert-verified clinical annotations.

To generate a **different** label set, privately set `DASHSCOPE_API_KEY` and run:

```bash
python generate_weak_labels.py --splits train val test --model qwen-plus-2025-12-01
```

This optional command transmits symptom/herb inputs to the external provider and may incur charges. Check current model availability before use; new generations need not reproduce the supplied labels. Outputs go to `results/generated_weak_labels`, without overwriting the study arrays. Never upload API credentials.

## Data sources and licensing

**Software.** The original software and its software documentation use the [MIT license](LICENSE), retaining `MPULiXin` as the copyright holder. Dependencies retain their own licenses. The MIT declaration in `CITATION.cff` describes the software, not every bundled data file.

**Set2Set benchmark.** The prescription data and vocabulary mappings derive from Jin Y, Zhang W, He X, Wang X, and Wang X. *Syndrome-aware herb recommendation with multi-graph convolution network.* ICDE 2020, pp. 145–156. The package retains the study's fixed splits but does not supply an independently verified upstream data-license document or redistribution permission. These third-party data are not relicensed under MIT.

**Herb properties.** `data/herb_properties.xlsx` supplies per-herb source attribution, Pharmacopoeia status, encoded properties, and notes for all 753 herbs. The model uses the three `herb_property_*.xlsx` matrices in `data/Set2Set/`. Preserve the row-level source attribution; the software license does not grant new rights over third-party source text or database material.

**Weak-label data.** The earlier [weak-label repository](https://github.com/MPULiXin/msyn-gcn-syndrome-labels) displayed CC0-1.0. Any materials already distributed under CC0 retain those terms. Coverage of every file in the present label directory has not been independently verified, so this package does not assign a blanket new CC0 or MIT license to all data.

Before public archiving, the depositor must confirm benchmark/herb-property redistribution rights and the applicable label-license coverage, and retain the relevant notices. Public availability or a source citation alone does not establish redistribution permission. Reusers should consult the original sources' applicable terms.

## Reproduction scope and citation

The package provides the core MSyn-GCN implementation and fixed study inputs. It does **not** include the original baseline implementations, seed-level logs/checkpoints, sensitivity-sweep scripts, oracle/case-study analysis scripts, or original split-creation script. It is therefore not a complete archive for reconstructing every manuscript table and figure. Unannotated legacy prediction files are excluded because their generating model, seed, and checkpoint were not recorded.

Software citation metadata are in [CITATION.cff](CITATION.cff). After publication, cite the **version-specific Zenodo DOI of the actual v1.1.0 archive**, together with its version. No new DOI is asserted here, and a prior archive's DOI must not be presented as identifying this updated package.
