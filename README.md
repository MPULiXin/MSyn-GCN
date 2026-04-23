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


Requirements

The code was developed with the following environment:

Python 3.9
PyTorch
NumPy
pandas
openpyxl
Main Training Command

The main experimental setting used in the paper can be reproduced with the following command:

python msyn-gcn-pytorch/torch_impl/train_torch.py \
  --data_path ./Data/ \
  --dataset Set2Set \
  --fusion concat \
  --layer_size "[128,256]" \
  --mlp_layer_size "[512]" \
  --embed_size 64 \
  --adj_type norm \
  --lr 2e-4 \
  --batch_size 1024 \
  --epoch 1500 \
  --verbose 1 \
  --mess_dropout "[0.05,0.05]" \
  --regs "[1e-2]" \
  --device cuda:0 \
  --result_index 1 \
  --seed 2025 \
  --eval_every 10 \
  --early_stop 0 \
  --use_props on \
  --prop_fusion gate \
  --prop_types_fuse concat \
  --prop_flavor herb_property_flavor.xlsx \
  --prop_meridian herb_property_meridian.xlsx \
  --prop_qi herb_property_qi.xlsx \
  --ez_on on \
  --ez_head_dim 128 \
  --lambda_align 0.03 \
  --lambda_qi 1.0 \
  --lambda_flavor 1.0 \
  --lambda_mer 1.0 \
  --loss wmse \
  --attn_pool on \
  --user_mlp_dropout 0.1
