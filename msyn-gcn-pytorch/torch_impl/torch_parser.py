# -*- coding: utf-8 -*-
import argparse

def parse_args():
    p = argparse.ArgumentParser(description="MSYNGCN + Properties + EZ Heads")

    # 基本路径
    p.add_argument('--data_path', type=str, default='./Data/')
    p.add_argument('--dataset', type=str, default='Set2Set')

    # 图与模型结构
    p.add_argument('--fusion', type=str, default='add', choices=['add', 'concat'])
    p.add_argument('--layer_size', type=str, default='[128,256]')      # list[int]
    p.add_argument('--mlp_layer_size', type=str, default='[256]')      # list[int]
    p.add_argument('--embed_size', type=int, default=64)
    p.add_argument('--adj_type', type=str, default='norm')

    # 训练
    p.add_argument('--lr', type=float, default=2e-4)
    p.add_argument('--batch_size', type=int, default=1024)
    p.add_argument('--epoch', type=int, default=1000)
    p.add_argument('--verbose', type=int, default=1)
    p.add_argument('--mess_dropout', type=str, default='[0.0,0.0]')    # drop rate list
    p.add_argument('--regs', type=str, default='[7e-3]')
    p.add_argument('--device', type=str, default='cuda:0')
    p.add_argument('--result_index', type=int, default=1)

    # 新增：随机种子与评测频率
    p.add_argument('--seed', type=int, default=2025)
    p.add_argument('--eval_every', type=int, default=10)
    p.add_argument('--early_stop', type=int, default=0)  # 0=off, 否则耐心值

    # 新增：损失/聚合/属性交互
    p.add_argument('--loss', type=str, default='bce', choices=['bce', 'wmse'])
    p.add_argument('--attn_pool', type=str, default='on', choices=['on', 'off'])
    p.add_argument('--user_mlp_dropout', type=float, default=0.1)
    # 三类属性内部融合：concat+线性 / 平均
    p.add_argument('--prop_types_fuse', type=str, default='concat', choices=['concat', 'avg'])

    # 属性与 EZ 模块
    p.add_argument('--use_props', type=str, default='on', choices=['on', 'off'])
    p.add_argument('--prop_flavor', type=str, default='herb_property_flavor.xlsx')
    p.add_argument('--prop_meridian', type=str, default='herb_property_meridian.xlsx')
    p.add_argument('--prop_qi', type=str, default='herb_property_qi.xlsx')
    # 属性 vs GCN 草药表示融合方式
    p.add_argument('--prop_fusion', type=str, default='gate', choices=['add', 'concat', 'gate'])

    p.add_argument('--ez_on', type=str, default='on', choices=['on', 'off'])
    p.add_argument('--ez_head_dim', type=int, default=128)
    # 对齐损失（KL）权重
    p.add_argument('--lambda_align', type=float, default=0.1)
    p.add_argument('--lambda_qi', type=float, default=1.0)
    p.add_argument('--lambda_flavor', type=float, default=1.0)
    p.add_argument('--lambda_mer', type=float, default=1.0)

    return p.parse_args()
