# -*- coding: utf-8 -*-
"""
msyngcn + 属性感知 + EZ辨证双头 + 软规则 & 对称KL对齐
功能开关与 train_torch.py 完全对齐：
- loss_type {'bce','wmse'}
- attn_pool: bool
- user_mlp_dropout: float
- prop_types_fuse {'concat','avg'}
- prop_fusion {'gate','add','concat'}
"""
from typing import Tuple, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


def _to_sparse_tensor(csr, device):
    """scipy.csr_matrix -> torch.sparse_coo_tensor (float32)"""
    coo = csr.tocoo()
    idx = np.vstack([coo.row, coo.col])
    i = torch.tensor(idx, dtype=torch.long, device=device)
    v = torch.tensor(coo.data, dtype=torch.float32, device=device)
    shape = torch.Size(coo.shape)
    return torch.sparse_coo_tensor(i, v, shape).coalesce()


def row_l2_normalize(x: torch.Tensor, eps=1e-9) -> torch.Tensor:
    return x / (x.norm(p=2, dim=1, keepdim=True) + eps)


class MLP(nn.Module):
    def __init__(self, in_dim, hid_dims, out_dim=None, dropout=0.0, act=nn.ReLU(inplace=True), last_act=False):
        super().__init__()
        dims = [in_dim] + list(hid_dims)
        layers = []
        for i in range(len(dims) - 1):
            layers += [nn.Linear(dims[i], dims[i + 1]), act, nn.Dropout(dropout)]
        if out_dim is not None:
            layers += [nn.Linear(dims[-1], out_dim)]
            if last_act:
                layers += [act]
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


class MSYNGCN_Torch(nn.Module):
    """
    - 双路 GCN 主干（症状路/草药路），逐层参数
    - 配对图融合（add/concat）
    - 症状聚合：注意力池化(可关) -> pre-MLP dropout -> user_mlp => e_sc^GCN
    - 草药属性三类（气/味/归经）先 concat+线性(或平均) => H_types，再与 GCN 草药向量门控/加/concat 融合 => e_H
    - EZ 双头（八纲/脏腑）+ 原型注意力 + 跨知识门控 => e_sc^EZ
    - e_sc = Gate(e_sc^GCN, e_sc^EZ)
    - 预测: y = sigmoid(e_sc @ e_H^T)
    - 损失: 主损失 (BCE/WMSE) + λ_align * 对称KL + L2
    """
    def __init__(
        self,
        n_users: int,
        n_items: int,
        embed_size: int,
        layer_sizes: list,
        mlp_sizes: list,
        fusion: str,
        mess_dropout: list,
        reg_decay: float,
        device: torch.device,
        # 图
        norm_adj_csr,
        sym_pair_adj_csr,
        herb_pair_adj_csr,
        # 新增：损失/聚合/属性交互
        loss_type: str = "bce",                 # 'bce' | 'wmse'
        attn_pool: bool = True,                 # 症状注意力池化
        user_mlp_dropout: float = 0.1,          # 进入 user_mlp 前的 dropout
        prop_types_fuse: str = "concat",        # 'concat' | 'avg'
        # 属性与 EZ 模块
        use_props: bool = True,
        prop_flavor: Optional[np.ndarray] = None,   # (M,5)
        prop_qi: Optional[np.ndarray] = None,       # (M,5)
        prop_meridian: Optional[np.ndarray] = None, # (M,12)
        prop_fusion: str = "gate",                  # 'gate' | 'add' | 'concat' （属性整体 vs GCN）
        ez_on: bool = True,
        ez_head_dim: int = 128,
        lambda_align: float = 0.05,
        lambda_qi: float = 1.0,
        lambda_flavor: float = 1.0,
        lambda_mer: float = 1.0,
    ):
        super().__init__()
        self.device = device
        self.n_users = n_users
        self.n_items = n_items
        self.emb_dim = embed_size
        self.weight_size_list = [embed_size] + list(layer_sizes)  # e.g., [64,128,256]
        self.fusion = fusion.lower()
        self.reg_decay = float(reg_decay)
        self.loss_type = loss_type.lower()
        self.attn_pool = bool(attn_pool)
        self.user_mlp_dropout_p = float(user_mlp_dropout)
        self.prop_types_fuse = prop_types_fuse.lower()
        self.prop_fusion = prop_fusion.lower()

        # --- Register graphs as buffers ---
        self.register_buffer("A", _to_sparse_tensor(norm_adj_csr, device))
        self.register_buffer("S_pair", _to_sparse_tensor(sym_pair_adj_csr, device))
        self.register_buffer("H_pair", _to_sparse_tensor(herb_pair_adj_csr, device))

        # --- base embeddings ---
        self.user_embedding = nn.Parameter(torch.empty(n_users, embed_size))
        self.item_embedding = nn.Parameter(torch.empty(n_items, embed_size))
        nn.init.xavier_uniform_(self.user_embedding)
        nn.init.xavier_uniform_(self.item_embedding)

        # --- GCN parameters: 逐层 ---
        L = len(self.weight_size_list) - 1
        self.Q_user = nn.ModuleList()
        self.Wgc_user = nn.ModuleList()
        self.Q_item = nn.ModuleList()
        self.Wgc_item = nn.ModuleList()
        for k in range(L):
            d_in = self.weight_size_list[k]
            d_out = self.weight_size_list[k + 1]
            self.Q_user.append(nn.Linear(d_in, d_in, bias=False))
            self.Wgc_user.append(nn.Linear(d_in * 2, d_out, bias=True))
            self.Q_item.append(nn.Linear(d_in, d_in, bias=False))
            self.Wgc_item.append(nn.Linear(d_in * 2, d_out, bias=True))

        # --- pair fusion projections ---
        self.M_user = nn.Linear(self.emb_dim, self.weight_size_list[-1], bias=True)
        self.M_item = nn.Linear(self.emb_dim, self.weight_size_list[-1], bias=True)

        # --- dropout list ---
        self.mess_dropout = list(mess_dropout)
        if len(self.mess_dropout) < L:
            self.mess_dropout += [self.mess_dropout[-1]] * (L - len(self.mess_dropout))

        # --- dimensions after pair fusion ---
        self.d_last = self.weight_size_list[-1]
        self.embed_out_dim = self.d_last if self.fusion == "add" else self.d_last * 2

        # --- Symptom attention pooling scorer (if on) ---
        if self.attn_pool:
            self.attn_scorer = nn.Linear(self.embed_out_dim, 1)

        # --- pre-MLP dropout & user-side MLP (to get e_sc^GCN) ---
        self.pre_mlp_dropout = nn.Dropout(p=self.user_mlp_dropout_p)
        assert mlp_sizes[-1] == self.embed_out_dim, \
            f"mlp_sizes[-1] must equal embed_out_dim ({self.embed_out_dim}), got {mlp_sizes[-1]}"
        self.user_mlp = MLP(self.embed_out_dim, mlp_sizes[:-1], mlp_sizes[-1], dropout=0.0, last_act=False)

        # 门控融合（e_sc^GCN 与 e_sc^EZ）
        self.sc_fuse_gate = nn.Sequential(
            nn.Linear(self.embed_out_dim * 2, self.embed_out_dim),
            nn.Sigmoid()
        )

        # --- property-aware herb representation ---
        self.use_props = use_props
        self.lambda_align = float(lambda_align)
        self.lambda_qi = float(lambda_qi)
        self.lambda_flavor = float(lambda_flavor)
        self.lambda_mer = float(lambda_mer)

        if self.use_props:
            assert prop_flavor is not None and prop_qi is not None and prop_meridian is not None, \
                "use_props=True 但未提供属性矩阵"
            self.register_buffer("X_flavor", torch.tensor(prop_flavor, dtype=torch.float32, device=device))  # (M,5)
            self.register_buffer("X_qi", torch.tensor(prop_qi, dtype=torch.float32, device=device))          # (M,5)
            self.register_buffer("X_mer", torch.tensor(prop_meridian, dtype=torch.float32, device=device))   # (M,12)
            # 三类属性先各自投影到 d_last
            self.W_flavor = nn.Linear(self.X_flavor.shape[1], self.d_last, bias=False)
            self.W_qi = nn.Linear(self.X_qi.shape[1], self.d_last, bias=False)
            self.W_mer = nn.Linear(self.X_mer.shape[1], self.d_last, bias=False)
            nn.init.xavier_uniform_(self.W_flavor.weight)
            nn.init.xavier_uniform_(self.W_qi.weight)
            nn.init.xavier_uniform_(self.W_mer.weight)

            # 三类属性内部融合（concat -> d_last 或 avg）
            if self.prop_types_fuse == "concat":
                self.W_types = nn.Linear(self.d_last * 3, self.d_last, bias=True)

            # 将 d_last 的属性向上映射到 embed_out_dim，与 GCN 草药向量同维度后做融合
            self.W_prop_up = nn.Linear(self.d_last, self.embed_out_dim, bias=True)

            # 属性整体 vs GCN 草药的融合
            if self.prop_fusion == "gate":
                self.gate_H = nn.Sequential(
                    nn.Linear(self.embed_out_dim * 2, self.embed_out_dim),
                    nn.ReLU(inplace=True),
                    nn.Linear(self.embed_out_dim, self.embed_out_dim),
                    nn.Sigmoid()
                )
            elif self.prop_fusion == "concat":
                self.W_prop_concat = nn.Linear(self.embed_out_dim * 2, self.embed_out_dim, bias=True)

        # --- EZ heads (Eight Principles / Zang-Fu) ---
        self.ez_on = ez_on
        if self.ez_on:
            d_h = int(ez_head_dim)
            # head classifiers from pooled symptom to distributions
            self.head_eight = nn.Sequential(
                nn.Linear(self.embed_out_dim, d_h), nn.ReLU(inplace=True),
                nn.Linear(d_h, 8)
            )
            self.head_zangfu = nn.Sequential(
                nn.Linear(self.embed_out_dim, d_h), nn.ReLU(inplace=True),
                nn.Linear(d_h, 12)
            )
            # prototype bases (attention with P/Z as weights)
            self.B_E = nn.Parameter(torch.randn(8, d_h))
            self.B_Z = nn.Parameter(torch.randn(12, d_h))
            nn.init.xavier_uniform_(self.B_E)
            nn.init.xavier_uniform_(self.B_Z)

            # cross-knowledge gating to combine e_E and e_Z
            self.cross_gate = nn.Sequential(
                nn.Linear(d_h * 2, d_h),
                nn.ReLU(inplace=True),
                nn.Linear(d_h, 2),
                nn.Softmax(dim=-1)
            )
            # project concatenated gated vector to model dimension
            self.W_ez = nn.Linear(d_h * 2, self.embed_out_dim, bias=True)

            # 软权重规则矩阵（行后续再归一化）
            # Eight Principles: [Yin, Yang, Exterior, Interior, Cold, Hot, Deficiency, Excess]
            # Qi:    [Cold, Hot, Warm, Cool, Neutral]
            # Flavor:[Sour, Bitter, Sweet, Pungent, Salty]
            M_E2QI = np.array([
                [0.30, 0.00, 0.00, 0.40, 0.30],  # Yin -> Cool, Cold, Neutral
                [0.00, 0.60, 0.40, 0.00, 0.00],  # Yang -> Hot, Warm
                [0.00, 0.00, 0.50, 0.50, 0.00],  # Exterior -> Warm, Cool
                [0.00, 0.00, 0.00, 0.00, 1.00],  # Interior -> Neutral
                [0.00, 0.60, 0.40, 0.00, 0.00],  # Cold(证) -> Hot, Warm
                [0.60, 0.00, 0.00, 0.40, 0.00],  # Hot(证) -> Cold, Cool
                [0.00, 0.00, 0.50, 0.00, 0.50],  # Deficiency -> Warm, Neutral
                [0.00, 0.00, 0.00, 0.00, 1.00],  # Excess -> Neutral
            ], dtype=np.float32)

            M_E2FL = np.array([
                [0.50, 0.50, 0.00, 0.00, 0.00],  # Yin -> Sour, Bitter
                [0.00, 0.00, 0.50, 0.50, 0.00],  # Yang -> Sweet, Pungent
                [0.00, 0.00, 0.00, 1.00, 0.00],  # Exterior -> Pungent
                [0.00, 0.60, 0.00, 0.00, 0.40],  # Interior -> Bitter, Salty
                [0.00, 0.00, 0.40, 0.60, 0.00],  # Cold(证) -> Sweet, Pungent
                [0.30, 0.70, 0.00, 0.00, 0.00],  # Hot(证) -> Sour, Bitter
                [0.40, 0.00, 0.60, 0.00, 0.00],  # Deficiency -> Sour, Sweet
                [0.00, 0.50, 0.00, 0.30, 0.20],  # Excess -> Bitter, Pungent, Salty
            ], dtype=np.float32)

            M_Z2MER = np.eye(12, dtype=np.float32)

            self.register_buffer("M_E2QI", torch.tensor(M_E2QI, dtype=torch.float32, device=device))     # (8,5)
            self.register_buffer("M_E2FL", torch.tensor(M_E2FL, dtype=torch.float32, device=device))     # (8,5)
            self.register_buffer("M_Z2MER", torch.tensor(M_Z2MER, dtype=torch.float32, device=device))   # (12,12)

        self.eps = 1e-9

    # --------- Core graph encoder ----------
    def _gcn_pass(self) -> Tuple[torch.Tensor, torch.Tensor]:
        pre = torch.cat([self.user_embedding, self.item_embedding], dim=0)  # (N+M, d0)

        # user route
        x = pre
        for k in range(len(self.weight_size_list) - 1):
            msg = torch.sparse.mm(self.A, x)                 # (N+M, d_k)
            msg = torch.tanh(self.Q_user[k](msg))            # (N+M, d_k)
            x = torch.cat([x, msg], dim=1)                   # (N+M, 2*d_k)
            x = torch.tanh(self.Wgc_user[k](x))              # (N+M, d_{k+1})
            x = F.dropout(x, p=float(self.mess_dropout[k]), training=self.training)
            x = row_l2_normalize(x)
        u_g, _ = x[:self.n_users], x[self.n_users:]          # (N,dL)

        # item route
        x = pre
        for k in range(len(self.weight_size_list) - 1):
            msg = torch.sparse.mm(self.A, x)
            msg = torch.tanh(self.Q_item[k](msg))
            x = torch.cat([x, msg], dim=1)
            x = torch.tanh(self.Wgc_item[k](x))
            x = F.dropout(x, p=float(self.mess_dropout[k]), training=self.training)
            x = row_l2_normalize(x)
        i_g = x[self.n_users:]                                # (M,dL)

        # pair fusion
        u_pair = torch.tanh(self.M_user(torch.sparse.mm(self.S_pair, self.user_embedding)))  # (N,dL)
        i_pair = torch.tanh(self.M_item(torch.sparse.mm(self.H_pair, self.item_embedding)))  # (M,dL)

        if self.fusion == "add":
            u_out = u_g + u_pair                              # (N,dL)
            i_out = i_g + i_pair                              # (M,dL)
        elif self.fusion == "concat":
            u_out = torch.cat([u_g, u_pair], dim=1)           # (N,2*dL)
            i_out = torch.cat([i_g, i_pair], dim=1)           # (M,2*dL)
        else:
            raise ValueError(f"fusion must be add/concat, got {self.fusion}")

        return u_out, i_out  # shapes: (N,embed_out_dim) , (M,embed_out_dim)

    # --------- Property-aware herb vector ----------
    def _herb_rep(self, i_out: torch.Tensor) -> torch.Tensor:
        """
        item-side GCN output i_out (M, embed_out_dim)
        属性：三路各投影到 d_last -> (concat+linear 或 avg) -> d_last -> up 到 embed_out_dim
        与 i_out 用 prop_fusion 融合 => e_H( M, embed_out_dim )
        """
        if not self.use_props:
            return i_out

        # three property projections to d_last
        H_fl = self.W_flavor(self.X_flavor)  # (M,d_last)
        H_qi = self.W_qi(self.X_qi)          # (M,d_last)
        H_me = self.W_mer(self.X_mer)        # (M,d_last)

        if self.prop_types_fuse == "concat":
            H_types = torch.cat([H_fl, H_qi, H_me], dim=1)  # (M,3*d_last)
            H_ch = self.W_types(H_types)                    # (M,d_last)
        elif self.prop_types_fuse == "avg":
            H_ch = (H_fl + H_qi + H_me) / 3.0               # (M,d_last)
        else:
            raise ValueError("prop_types_fuse must be concat/avg")

        # up to embed_out_dim to match i_out dimension
        H_ch = self.W_prop_up(H_ch)                         # (M,embed_out_dim)
        H_gcn = i_out                                       # (M,embed_out_dim)

        if self.prop_fusion == "add":
            e_H = H_gcn + H_ch
        elif self.prop_fusion == "concat":
            e_H = self.W_prop_concat(torch.cat([H_gcn, H_ch], dim=1))
        elif self.prop_fusion == "gate":
            g = self.gate_H(torch.cat([H_gcn, H_ch], dim=1))  # (M,embed_out_dim)
            e_H = g * H_ch + (1.0 - g) * H_gcn
        else:
            raise ValueError("prop_fusion must be add/concat/gate")
        return e_H

    # --------- EZ heads and final syndrome vector ----------
    def _ez_block(self, u_batch_emb: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Args:
            u_batch_emb: (B, embed_out_dim)
        Returns:
            P_sc: (B,8), Z_sc: (B,12), e_sc_ez: (B, embed_out_dim)
        """
        logits_E = self.head_eight(u_batch_emb)
        logits_Z = self.head_zangfu(u_batch_emb)
        P_sc = torch.softmax(logits_E, dim=-1)  # (B,8)
        Z_sc = torch.softmax(logits_Z, dim=-1)  # (B,12)

        # prototype attention
        e_E = torch.matmul(P_sc, self.B_E)  # (B,d_h)
        e_Z = torch.matmul(Z_sc, self.B_Z)  # (B,d_h)

        # cross-knowledge gating
        gate = self.cross_gate(torch.cat([e_E, e_Z], dim=1))  # (B,2)
        e_Eg = gate[:, 0:1] * e_E
        e_Zg = gate[:, 1:2] * e_Z
        e_concat = torch.cat([e_Eg, e_Zg], dim=1)  # (B,2d_h)
        e_sc_ez = self.W_ez(e_concat)              # (B,embed_out_dim)

        return P_sc, Z_sc, e_sc_ez

    # --------- public APIs ----------






    #
    # def forward_user_emb(self, users_multi_hot: torch.Tensor):
    #     """
    #     返回:
    #         e_sc (B,embed_out_dim), P_sc (B,8), Z_sc (B,12), e_H_all (M,embed_out_dim), y_pred (B,M)
    #     """
    #     u_out, i_out = self._gcn_pass()          # (N,D), (M,D)  D=embed_out_dim
    #     e_H_all = self._herb_rep(i_out)          # (M,D)
    #
    #     # ---- symptom aggregation ----
    #     if self.attn_pool:
    #         # all-symptom scores
    #         s_all = self.attn_scorer(u_out).squeeze(1)       # (N,)
    #         # masked scores for each batch
    #         masked = users_multi_hot * s_all.unsqueeze(0) + (1.0 - users_multi_hot) * (-1e9)
    #         alpha = torch.softmax(masked, dim=1)             # (B,N)
    #         u_batch = torch.matmul(alpha, u_out)             # (B,D)
    #     else:
    #         u_batch = torch.matmul(users_multi_hot, u_out)   # sum
    #         denom = users_multi_hot.sum(dim=1, keepdim=True).clamp_min(1.0)
    #         u_batch = u_batch / denom                        # avg
    #
    #     # pre-MLP dropout + e_sc^GCN
    #     u_batch = self.pre_mlp_dropout(u_batch)
    #     e_sc_gcn = self.user_mlp(u_batch)                    # (B,D)
    #
    #     if self.ez_on:
    #         P_sc, Z_sc, e_sc_ez = self._ez_block(u_batch)
    #         g = self.sc_fuse_gate(torch.cat([e_sc_gcn, e_sc_ez], dim=1))  # (B,D)
    #         e_sc = g * e_sc_ez + (1.0 - g) * e_sc_gcn
    #     else:
    #         P_sc = torch.zeros(users_multi_hot.size(0), 8, device=self.device)
    #         Z_sc = torch.zeros(users_multi_hot.size(0), 12, device=self.device)
    #         e_sc = e_sc_gcn
    #
    #     y_pred = torch.sigmoid(torch.matmul(e_sc, e_H_all.t()))  # (B,M)
    #     return e_sc, P_sc, Z_sc, e_H_all, y_pred

    def forward_user_emb(self, users_multi_hot: torch.Tensor, return_debug: bool = False):
        """
        返回:
            标准返回（return_debug=False）:
                e_sc (B,embed_out_dim), P_sc (B,8), Z_sc (B,12), e_H_all (M,embed_out_dim), y_pred (B,M)
            调试返回（return_debug=True）:
                在标准返回后额外返回 debug 字典:
                debug = {
                    "alpha": (B,N) or None,   # 症状注意力权重（attn_pool=on 时）
                    "g_sc":  (B,D) or None,   # sc_fuse_gate 输出（ez_on=on 时）
                }
        """
        u_out, i_out = self._gcn_pass()  # (N,D), (M,D)  D=embed_out_dim
        e_H_all = self._herb_rep(i_out)  # (M,D)

        # ---- symptom aggregation ----
        alpha = None
        if self.attn_pool:
            # all-symptom scores
            s_all = self.attn_scorer(u_out).squeeze(1)  # (N,)
            # masked scores for each batch
            masked = users_multi_hot * s_all.unsqueeze(0) + (1.0 - users_multi_hot) * (-1e9)
            alpha = torch.softmax(masked, dim=1)  # (B,N)
            u_batch = torch.matmul(alpha, u_out)  # (B,D)
        else:
            u_batch = torch.matmul(users_multi_hot, u_out)  # sum
            denom = users_multi_hot.sum(dim=1, keepdim=True).clamp_min(1.0)
            u_batch = u_batch / denom  # avg

        # pre-MLP dropout + e_sc^GCN
        u_batch = self.pre_mlp_dropout(u_batch)
        e_sc_gcn = self.user_mlp(u_batch)  # (B,D)

        g_sc = None
        if self.ez_on:
            P_sc, Z_sc, e_sc_ez = self._ez_block(u_batch)
            g_sc = self.sc_fuse_gate(torch.cat([e_sc_gcn, e_sc_ez], dim=1))  # (B,D)
            e_sc = g_sc * e_sc_ez + (1.0 - g_sc) * e_sc_gcn
        else:
            P_sc = torch.zeros(users_multi_hot.size(0), 8, device=self.device)
            Z_sc = torch.zeros(users_multi_hot.size(0), 12, device=self.device)
            e_sc = e_sc_gcn

        y_pred = torch.sigmoid(torch.matmul(e_sc, e_H_all.t()))  # (B,M)

        if return_debug:
            debug = {"alpha": alpha, "g_sc": g_sc}
            return e_sc, P_sc, Z_sc, e_H_all, y_pred, debug

        return e_sc, P_sc, Z_sc, e_H_all, y_pred




    def forward(self, users_multi_hot: torch.Tensor, return_debug: bool = False):
        return self.forward_user_emb(users_multi_hot, return_debug=return_debug)




    # --------- losses ----------
    @staticmethod
    def _wmse_loss(pred: torch.Tensor, target: torch.Tensor, item_weights: torch.Tensor) -> torch.Tensor:
        """
        加权 MSE：对每个草药列乘以对应权重（列向量）
        pred/target: (B,M), item_weights: (M,1)
        """
        w = item_weights.t()  # (1,M)
        mse = (pred - target) ** 2
        loss = (mse * w).mean()
        return loss

    @staticmethod
    def _wbce_loss(pred: torch.Tensor, target: torch.Tensor, item_weights: torch.Tensor, eps=1e-9) -> torch.Tensor:
        """
        加权 BCE：每个草药列乘以权重（列向量）
        """
        pred = pred.clamp(min=eps, max=1.0 - eps)
        w = item_weights.t()  # (1,M)
        bce = -(target * torch.log(pred) + (1.0 - target) * torch.log(1.0 - pred))
        loss = (bce * w).mean()
        return loss

    def _sym_kl(self, p: torch.Tensor, q: torch.Tensor, eps=1e-9) -> torch.Tensor:
        """
        对称 KL(p||q) + KL(q||p)，对 batch 取均值
        p,q: (B,C) 已做归一化
        """
        p = p.clamp(min=eps); q = q.clamp(min=eps)
        kl1 = (p * (p.log() - q.log())).sum(dim=1).mean()
        kl2 = (q * (q.log() - p.log())).sum(dim=1).mean()
        return kl1 + kl2

    def _align_loss(self, y_true: torch.Tensor, P_sc: torch.Tensor, Z_sc: torch.Tensor) -> torch.Tensor:
        """
        y_true: (B,M) 0/1 多热
        P_sc: (B,8)  Z_sc: (B,12)
        使用软规则矩阵得到属性偏好，与真实处方属性分布做对称KL
        """
        if (not self.use_props) or (not self.ez_on) or self.lambda_align <= 0.0:
            return torch.tensor(0.0, device=self.device)

        def _norm_rows(x):
            s = x.sum(dim=1, keepdim=True)
            s = torch.where(s <= 0, torch.ones_like(s), s)
            return x / s

        # predicted preferences via soft rule matrices
        p_qi = _norm_rows(torch.matmul(P_sc, self.M_E2QI))   # (B,5)
        p_fl = _norm_rows(torch.matmul(P_sc, self.M_E2FL))   # (B,5)
        p_me = _norm_rows(torch.matmul(Z_sc, self.M_Z2MER))  # (B,12)

        # ground-truth from herbs
        gt_qi = _norm_rows(torch.matmul(y_true, self.X_qi))       # (B,5)
        gt_fl = _norm_rows(torch.matmul(y_true, self.X_flavor))   # (B,5)
        gt_me = _norm_rows(torch.matmul(y_true, self.X_mer))      # (B,12)

        l_qi = self._sym_kl(p_qi, gt_qi)
        l_fl = self._sym_kl(p_fl, gt_fl)
        l_me = self._sym_kl(p_me, gt_me)

        return self.lambda_align * (self.lambda_qi * l_qi + self.lambda_flavor * l_fl + self.lambda_mer * l_me)

    def loss_fn(self,
                users_multi_hot: torch.Tensor,
                items_multi_hot: torch.Tensor,
                user_set_idx: torch.Tensor,
                item_set_idx: torch.Tensor,
                item_weights: torch.Tensor):
        """
        返回: total_loss, rec_loss (BCE/WMSE), align_loss, reg_loss
        """
        e_sc, P_sc, Z_sc, e_H_all, y_pred = self.forward_user_emb(users_multi_hot)

        if self.loss_type == "bce":
            rec_loss = self._wbce_loss(y_pred, items_multi_hot, item_weights)
        elif self.loss_type == "wmse":
            rec_loss = self._wmse_loss(y_pred, items_multi_hot, item_weights)
        else:
            raise ValueError("loss_type must be 'bce' or 'wmse'")

        align_loss = self._align_loss(items_multi_hot, P_sc, Z_sc)

        # L2 正则（对本批的 e_sc/e_H_all 适度约束）
        reg = (e_sc.pow(2).mean() + e_H_all.pow(2).mean())
        reg_loss = self.reg_decay * reg

        total = rec_loss + align_loss + reg_loss
        return total, rec_loss.detach(), align_loss.detach(), reg_loss.detach()

    # @torch.no_grad()
    # def predict_scores(self, test_users_multi_hot: torch.Tensor) -> torch.Tensor:
    #     """
    #     返回 (num_test, M) 的概率矩阵
    #     """
    #     self.eval()
    #     _, _, _, _, y_pred = self.forward_user_emb(test_users_multi_hot)
    #     return y_pred

    @torch.no_grad()
    def predict_scores(self, test_users_multi_hot: torch.Tensor) -> torch.Tensor:
        """
        返回 (num_test, M) 的概率矩阵
        """
        self.eval()
        out = self.forward_user_emb(test_users_multi_hot, return_debug=False)
        # out: (e_sc, P_sc, Z_sc, e_H_all, y_pred)
        y_pred = out[4]
        return y_pred

