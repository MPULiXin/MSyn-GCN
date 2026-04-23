
# -*- coding: utf-8 -*-
import os
import time
from ast import literal_eval

import numpy as np
import torch
from torch.optim import Adam

from torch_impl.torch_parser import parse_args
from torch_impl.torch_data import Data
from torch_impl.msyngcn_torch import MSYNGCN_Torch


def set_seed(seed: int):
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def parse_list(s, typ=int):
    v = literal_eval(s)
    assert isinstance(v, (list, tuple)), f"expect list in string, got {s}"
    if typ == int:
        return [int(x) for x in v]
    else:
        return [float(x) for x in v]


# -------- 原基线评测口径（原封不动迁移）--------
def dcg_at_k(r, k, method=1):
    r = np.asfarray(r)[:k]
    if r.size == 0:
        return 0.0
    if method == 0:
        return r[0] + np.sum(r[1:] / np.log2(np.arange(2, r.size + 1)))
    else:
        return np.sum(r / np.log2(np.arange(2, r.size + 2)))

def ndcg_at_k(r, k, method=1):
    dcg_max = dcg_at_k(sorted(r, reverse=True), k, method)
    if dcg_max == 0:
        return 0.0
    return dcg_at_k(r, k, method) / dcg_max



@torch.no_grad()
def evaluate_baseline_ndcg(model: MSYNGCN_Torch, data: Data, Ks=(5, 10, 15, 20), device=torch.device('cpu')):
    """
    完全沿用原基线：
    - 对每个样本排序得到 ranking
    - 先构造 rbin（长度为 maxK 的 0/1 命中序列）
    - precision/recall 按 topK 命中数统计
    - ndcg@K = ndcg_at_k(rbin, K)     ← 注意：使用 rbin 的命中数来计算 IDCG
    """
    model.eval()
    test_users_t = torch.tensor(data.test_users, dtype=torch.float32, device=device)  # (T, N)
    scores = model.predict_scores(test_users_t).detach().cpu().numpy()                # (T, M)

    precision = np.zeros(len(Ks), dtype=np.float64)
    recall    = np.zeros(len(Ks), dtype=np.float64)
    ndcg      = np.zeros(len(Ks), dtype=np.float64)

    maxK = Ks[-1]
    for idx, (_, gt_items) in enumerate(data.test_group_set):
        gt = set(gt_items)
        ranking = list(enumerate(scores[idx].tolist()))
        ranking.sort(key=lambda x: x[1], reverse=True)

        # rbin: 前 maxK 的命中序列（0/1）
        rbin = []
        for i in range(maxK):
            herb = ranking[i][0]
            rbin.append(1 if herb in gt else 0)

        for j, K in enumerate(Ks):
            topK = ranking[:K]
            hit = sum(1 for (it, _) in topK if it in gt)
            precision[j] += hit / K
            recall[j]    += hit / max(1, len(gt))
            ndcg[j]      += ndcg_at_k(rbin, K)

    precision /= len(data.test_group_set)
    recall    /= len(data.test_group_set)
    ndcg      /= len(data.test_group_set)
    return precision, recall, ndcg
# -------------------------------------------



# -------------------------------------------


@torch.no_grad()
def dump_case_debug(model: MSYNGCN_Torch,
                    data: Data,
                    device: torch.device,
                    case_idx: int = 0,
                    topk: int = 20,
                    out_path: str = None):
    """
    导出论文 Case Study 所需的中间量（最小集合）：
    - P_sc: 八纲分布 (8,)
    - Z_sc: 脏腑分布 (12,)
    - alpha: 症状注意力权重 (N,) （仅 attn_pool=on 时存在）
    - g_sc: syndrome 融合门控 (D,) （仅 ez_on=on 时存在）
    - topK herbs: 推荐 TopK 草药 id
    注意：需要 msyngcn_torch.py 已实现 forward(..., return_debug=True) 或 forward_user_emb(..., return_debug=True)
    """
    model.eval()

    if out_path is None:
        out_dir = os.path.join("output", data.dataset)
        os.makedirs(out_dir, exist_ok=True)
        out_path = os.path.join(out_dir, "case_debug.json")

    # 取一个测试样本 (1, N)
    x = torch.tensor(data.test_users[case_idx:case_idx + 1],
                     dtype=torch.float32, device=device)

    # 兼容两种调用：优先 forward，其次 forward_user_emb
    if hasattr(model, "forward"):
        out = model.forward(x, return_debug=True)
    else:
        out = model.forward_user_emb(x, return_debug=True)

    # out: (e_sc, P_sc, Z_sc, e_H_all, y_pred, debug)
    e_sc, P_sc, Z_sc, e_H_all, y_pred, debug = out

    # 取 TopK
    topk_ids = torch.topk(y_pred.squeeze(0), k=topk).indices.detach().cpu().tolist()

    # 处理 debug
    alpha = debug.get("alpha", None)
    g_sc = debug.get("g_sc", None)

    d = {
        "case_idx": int(case_idx),
        "topk": int(topk),
        "P_sc": P_sc.squeeze(0).detach().cpu().tolist(),
        "Z_sc": Z_sc.squeeze(0).detach().cpu().tolist(),
        "alpha": None if alpha is None else alpha.squeeze(0).detach().cpu().tolist(),
        # g_sc 通常是 (1, D)，我们这里导出均值 + 全向量，写论文更灵活
        "g_sc_mean": None if g_sc is None else float(g_sc.mean().detach().cpu().item()),
        "g_sc": None if g_sc is None else g_sc.squeeze(0).detach().cpu().tolist(),
        "topk_herbs": topk_ids
    }

    import json
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(d, f, ensure_ascii=False, indent=2)

    print(f"[CaseDebug] saved to: {out_path}")



def main():
    args = parse_args()
    set_seed(args.seed)

    device = torch.device(args.device if torch.cuda.is_available() and args.device.startswith('cuda') else 'cpu')
    print(f"Device: {device}")

    layer_sizes = parse_list(args.layer_size, int)
    mlp_sizes = parse_list(args.mlp_layer_size, int)
    mess_dropout = parse_list(args.mess_dropout, float)
    regs = parse_list(args.regs, float)
    reg_decay = float(regs[0])

    # ---- Data ----
    data = Data(
        path=args.data_path,
        dataset=args.dataset,
        batch_size=args.batch_size,
        prop_flavor_file=args.prop_flavor,
        prop_mer_file=args.prop_meridian,
        prop_qi_file=args.prop_qi
    )
    print(f"symtom个数n_users={data.n_users}, herb个数n_items={data.n_items}")
    print(f"#用于sample生成batch的train pres {len(data)}")

    _, norm_adj, _, sym_pair_adj, herb_pair_adj = data.get_adj_mat(adj_type=args.adj_type)

    use_props = (args.use_props == 'on') and (data.prop_flavor is not None) and (data.prop_qi is not None) and (data.prop_mer is not None)

    # ---- Model ----
    model = MSYNGCN_Torch(
        n_users=data.n_users,
        n_items=data.n_items,
        embed_size=args.embed_size,
        layer_sizes=layer_sizes,
        mlp_sizes=mlp_sizes,
        fusion=args.fusion,
        mess_dropout=mess_dropout,
        reg_decay=reg_decay,
        device=device,
        norm_adj_csr=norm_adj,
        sym_pair_adj_csr=sym_pair_adj,
        herb_pair_adj_csr=herb_pair_adj,
        # 新增功能开关（与你的 msyngcn_torch.py 对齐）
        loss_type=args.loss,                               # 'bce' | 'wmse'
        attn_pool=(args.attn_pool == 'on'),                # True/False
        user_mlp_dropout=float(args.user_mlp_dropout),     # 0.1
        prop_types_fuse=args.prop_types_fuse,              # 'concat' | 'avg'
        # 属性/EZ
        use_props=use_props,
        prop_flavor=data.prop_flavor,
        prop_qi=data.prop_qi,
        prop_meridian=data.prop_mer,
        prop_fusion=args.prop_fusion,
        ez_on=(args.ez_on == 'on'),
        ez_head_dim=args.ez_head_dim,
        lambda_align=args.lambda_align,
        lambda_qi=args.lambda_qi,
        lambda_flavor=args.lambda_flavor,
        lambda_mer=args.lambda_mer
    ).to(device)

    opt = Adam(model.parameters(), lr=args.lr)

    # cache item_weights tensor on device
    item_weights_t = torch.tensor(data.item_weights, dtype=torch.float32, device=device)

    # ---- Train loop ----
    Ks = parse_list("[5,10,15,20]", int)  # 与基线一致
    best_p5 = -1.0
    best_line = ""
    patience = 0

    t0 = time.time()
    for epoch in range(1, args.epoch + 1):
        model.train()
        n_batch = len(data) // args.batch_size + 1
        for _ in range(n_batch):
            users_np, items_np, user_set_np, item_set_np = data.sample()
            users = torch.tensor(users_np, dtype=torch.float32, device=device)
            items = torch.tensor(items_np, dtype=torch.float32, device=device)
            user_set = torch.tensor(user_set_np, dtype=torch.long, device=device)
            item_set = torch.tensor(item_set_np, dtype=torch.long, device=device)

            opt.zero_grad()
            total, rec, align, reg = model.loss_fn(users, items, user_set, item_set, item_weights_t)
            total.backward()
            opt.step()

        if epoch % args.eval_every == 0 or epoch == 1:
            P, R, N = evaluate_baseline_ndcg(model, data, Ks=Ks, device=device)
            line = f"Best Iter=[{epoch}]\trecall=[{R[0]:.5f}\t{R[1]:.5f}\t{R[2]:.5f}\t{R[3]:.5f}], precision=[{P[0]:.5f}\t{P[1]:.5f}\t{P[2]:.5f}\t{P[3]:.5f}], ndcg=[{N[0]:.5f}\t{N[1]:.5f}\t{N[2]:.5f}\t{N[3]:.5f}]"
            print(line)

            if P[0] > best_p5:
                best_p5 = P[0]
                best_line = line
                out_dir = os.path.join("output", args.dataset)
                os.makedirs(out_dir, exist_ok=True)
                ckpt_path = os.path.join(out_dir, "ckpt_best_orig.pth")
                torch.save(model.state_dict(), ckpt_path)
                print(f"[CKPT] Best model saved at: {ckpt_path}")
                patience = 0
            else:
                patience += 1
                if args.early_stop > 0 and patience >= args.early_stop:
                    print(f"[EarlyStop] patience={patience}, stop at epoch {epoch}")
                    break

    # elapsed = time.time() - t0
    # print(f"Done. Elapsed: {elapsed:.1f}s")
    #
    # out_dir = os.path.join("output", args.dataset)
    # os.makedirs(out_dir, exist_ok=True)
    # result_path = os.path.join(out_dir, f"msyngcn_torch.result-{args.result_index}")
    # with open(result_path, "w", encoding="utf-8") as f:
    #     f.write(best_line + "\n")
    # print(best_line)

    elapsed = time.time() - t0
    print(f"Done. Elapsed: {elapsed:.1f}s")

    out_dir = os.path.join("output", args.dataset)
    os.makedirs(out_dir, exist_ok=True)

    # ---- 写入 best 结果（保持原逻辑）----
    result_path = os.path.join(out_dir, f"msyngcn_torch.result-{args.result_index}")
    with open(result_path, "w", encoding="utf-8") as f:
        f.write(best_line + "\n")
    print(best_line)

    # ---- 导出 Case Study debug（新增）----
    # 选择一个更有解释价值的 test case：症状数 >= 3 的第一个样本；若不存在则回退到症状数最多的样本
    try:
        # data.test_users: (num_test, n_users) 的 multi-hot
        # 计算每个 test 样本的症状数
        sym_counts = data.test_users.sum(axis=1)  # numpy array

        # 优先找第一个 >=3 的样本
        cand = np.where(sym_counts >= 3)[0]
        if len(cand) > 0:
            case_idx = int(cand[0])
        else:
            # 回退：找症状数最多的样本
            case_idx = int(np.argmax(sym_counts))

        print(f"[CaseSelect] case_idx={case_idx}, symptom_count={int(sym_counts[case_idx])}")

        dump_case_debug(
            model=model,
            data=data,
            device=device,
            case_idx=case_idx,
            topk=20,
            out_path=os.path.join(out_dir, "case_debug.json")
        )
    except Exception as e:
        print(f"[Warn] dump_case_debug failed: {e}")




if __name__ == "__main__":
    main()



