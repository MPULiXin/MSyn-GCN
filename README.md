# MSyn-GCN

Code for the paper: **MSyn-GCN for Herb Recommendation**

## Overview

This repository contains the implementation of **MSyn-GCN**, a graph-based herb recommendation framework that integrates:

- symptom-symptom relations
- symptom-herb relations
- herb-herb relations
- Eight Principles
- Zang-Fu theory
- herb properties

The model is designed to improve both the accuracy and interpretability of herb recommendation in Traditional Chinese Medicine (TCM).

## Repository Structure

```text
msyn-gcn-pytorch/
├── Data/
│   └── Set2Set/
│       ├── herbPair-40.txt
│       ├── herb_mapping.txt
│       ├── herb_property_flavor.xlsx
│       ├── herb_property_meridian.xlsx
│       ├── herb_property_qi.xlsx
│       ├── symPair-5.txt
│       ├── sym_mapping.txt
│       ├── train.txt
│       └── test.txt
├── torch_impl/
│   ├── msyngcn_torch.py
│   ├── torch_data.py
│   ├── torch_parser.py
│   └── train_torch.py
