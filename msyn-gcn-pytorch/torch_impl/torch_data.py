# -*- coding: utf-8 -*-
"""
数据读入、图构建、属性读取（xlsx）
"""
import os
import numpy as np
from scipy.sparse import dok_matrix, csr_matrix, lil_matrix, eye

try:
    import pandas as pd
except Exception as e:
    pd = None

class Data(object):
    def __init__(self, path: str, dataset: str, batch_size: int = 1024,
                 prop_flavor_file="herb_property_flavor.xlsx",
                 prop_mer_file="herb_property_meridian.xlsx",
                 prop_qi_file="herb_property_qi.xlsx"):
        """
        path: 例如 ./Data/
        dataset: 例如 Set2Set/
        """
        self.path = path
        self.dataset = dataset
        self.batch_size = batch_size

        self.dataset_dir = os.path.join(path, dataset)
        assert os.path.isdir(self.dataset_dir), f"Dataset dir not found: {self.dataset_dir}"

        self.train_file = os.path.join(self.dataset_dir, "train.txt")
        self.test_file = os.path.join(self.dataset_dir, "test.txt")
        self.sym_pair_file = os.path.join(self.dataset_dir, "symPair-5.txt")
        self.herb_pair_file = os.path.join(self.dataset_dir, "herbPair-40.txt")

        # 属性文件路径
        self.prop_flavor_path = os.path.join(self.dataset_dir, prop_flavor_file)
        self.prop_mer_path = os.path.join(self.dataset_dir, prop_mer_file)
        self.prop_qi_path = os.path.join(self.dataset_dir, prop_qi_file)

        # 基本统计
        self.n_users, self.n_items = 0, 0
        self.train_pres = []  # 每行：(sym_ids, herb_ids)
        self.test_pres = []
        self._read_train_test()
        self._build_R_and_weights()
        self._build_pairs()
        self._build_test_users()

        # 属性矩阵（可能为空）
        self.prop_flavor = None
        self.prop_qi = None
        self.prop_mer = None
        self._load_properties()

    # ---------- IO helpers ----------
    @staticmethod
    def _parse_line(line: str):
        line = line.strip()
        if not line:
            return None, None
        if "\t" not in line:
            return None, None
        left, right = line.split("\t", 1)
        users = [int(x) for x in left.strip().split() if x != ""]
        items = [int(x) for x in right.strip().split() if x != ""]
        return users, items

    def _read_train_test(self):
        # 初次遍历取规模
        max_u, max_i = -1, -1
        with open(self.train_file, "r", encoding="utf-8") as f:
            for raw in f:
                u, it = self._parse_line(raw)
                if u is None:  # 跳过空行
                    continue
                max_u = max(max_u, max(u) if len(u) else -1)
                max_i = max(max_i, max(it) if len(it) else -1)
                self.train_pres.append((u, it))

        with open(self.test_file, "r", encoding="utf-8") as f:
            for raw in f:
                u, it = self._parse_line(raw)
                if u is None:
                    continue
                max_u = max(max_u, max(u) if len(u) else -1)
                max_i = max(max_i, max(it) if len(it) else -1)
                self.test_pres.append((u, it))

        self.n_users = max_u + 1
        self.n_items = max_i + 1

    def _build_R_and_weights(self):
        # 交互矩阵R
        R = dok_matrix((self.n_users, self.n_items), dtype=np.float32)
        freq = np.zeros(self.n_items, dtype=np.int64)
        for (us, its) in self.train_pres:
            for it in its:
                if 0 <= it < self.n_items:
                    for u in us:
                        if 0 <= u < self.n_users:
                            R[u, it] = 1.0
                    freq[it] += 1
        self.R = R.tocsr()

        # item 权重：max_freq / freq（freq=0 -> 1）
        freq[freq == 0] = 1
        max_f = float(freq.max())
        weights = max_f / freq.astype(np.float32)
        self.item_weights = weights.reshape(-1, 1).astype(np.float32)

    def _build_pairs(self):
        # 症状配对
        S = lil_matrix((self.n_users, self.n_users), dtype=np.float32)
        with open(self.sym_pair_file, "r", encoding="utf-8") as f:
            for raw in f:
                raw = raw.strip()
                if not raw:
                    continue
                parts = raw.split()
                if len(parts) < 2:
                    continue
                a = int(parts[0]); b = int(parts[1])
                if 0 <= a < self.n_users and 0 <= b < self.n_users:
                    S[a, b] = 1.0
                    S[b, a] = 1.0
        S.setdiag(1.0)
        self.sym_pair = S.tocsr()

        # 草药
        H = lil_matrix((self.n_items, self.n_items), dtype=np.float32)
        with open(self.herb_pair_file, "r", encoding="utf-8") as f:
            for raw in f:
                raw = raw.strip()
                if not raw:
                    continue
                parts = raw.split()
                if len(parts) < 2:
                    continue
                a = int(parts[0]); b = int(parts[1])
                if 0 <= a < self.n_items and 0 <= b < self.n_items:
                    H[a, b] = 1.0
                    H[b, a] = 1.0
        H.setdiag(1.0)
        self.herb_pair = H.tocsr()

    def _build_test_users(self):
        # 构造测试集多热矩阵
        n_test = len(self.test_pres)
        self.test_users = np.zeros((n_test, self.n_users), dtype=np.float32)
        self.test_group_set = []
        for i, (us, its) in enumerate(self.test_pres):
            self.test_users[i, us] = 1.0
            key = "_".join(map(str, sorted(set(us))))
            self.test_group_set.append((key, its))

    def _load_properties(self):
        # 读取三张 Excel 属性表；若 pandas 不存在则报错（避免静默失败）
        if pd is None:
            print("[WARN] pandas/openpyxl 未安装，无法读取属性文件；将跳过属性增强。")
            return
        def read_prop(path, expect_cols):
            if not os.path.isfile(path):
                print(f"[WARN] 属性文件缺失: {path}")
                return None
            df = pd.read_excel(path)
            # 假设第2列为“草药编号”，第3.. 为属性列（0/1）
            if df.shape[1] < 3:
                print(f"[WARN] 属性文件列数不足: {path}")
                return None
            hid = df.iloc[:, 1].astype(int).to_numpy()
            X = df.iloc[:, 2:].fillna(0).astype(int).to_numpy()
            M = np.zeros((self.n_items, expect_cols), dtype=np.float32)
            for i, row in zip(hid, X):
                if 0 <= int(i) < self.n_items:
                    M[int(i), :min(expect_cols, len(row))] = row[:expect_cols]
            return M

        self.prop_flavor = read_prop(self.prop_flavor_path, 5)   # (M,5)
        self.prop_mer = read_prop(self.prop_mer_path, 12)        # (M,12)
        self.prop_qi = read_prop(self.prop_qi_path, 5)           # (M,5)

    # ---------- public ----------
    def get_adj_mat(self, adj_type='norm'):
        n = self.n_users + self.n_items
        # 构二部图邻接
        R = self.R.tolil()
        adj = lil_matrix((n, n), dtype=np.float32)
        adj[:self.n_users, self.n_users:] = R
        adj[self.n_users:, :self.n_users] = R.T
        adj = adj.tocsr()
        I = eye(n, format='csr', dtype=np.float32)

        # 行归一化
        def row_norm(A: csr_matrix):
            rowsum = np.array(A.sum(1)).flatten()
            rowsum[rowsum == 0.0] = 1.0
            inv = 1.0 / rowsum
            Dinv = csr_matrix((inv, (np.arange(n), np.arange(n))), shape=(n, n))
            return Dinv @ A

        plain_adj = adj
        norm_adj = row_norm(adj + I)
        mean_adj = row_norm(adj)

        return plain_adj, norm_adj, mean_adj, self.sym_pair, self.herb_pair

    def sample(self):
        """
        随机采样 batch（处方级），输出多热矩阵与索引集合
        """
        batch_users = np.zeros((self.batch_size, self.n_users), dtype=np.float32)
        batch_items = np.zeros((self.batch_size, self.n_items), dtype=np.float32)
        user_set, item_set = set(), set()

        idx = np.random.randint(low=0, high=len(self.train_pres), size=self.batch_size)
        for i, rid in enumerate(idx):
            us, its = self.train_pres[rid]
            if len(us) == 0 or len(its) == 0:
                continue
            batch_users[i, us] = 1.0
            batch_items[i, its] = 1.0
            user_set.update(us); item_set.update(its)

        return batch_users, batch_items, np.array(sorted(user_set)), np.array(sorted(item_set))

    # properties (numpy arrays or None)
    def get_properties(self):
        return self.prop_flavor, self.prop_qi, self.prop_mer

    def __len__(self):
        return len(self.train_pres)
