# -*- coding: utf-8 -*-
"""
llm_syndrome_label.py —— 用千问(Qwen)API 给处方做八纲/脏腑「弱监督」标注
==============================================================================
目的：在缺少证候标注的情况下，用 LLM 根据每条处方的【症状(+所用中药作为辅助证据)】
推断出八纲(8维)与脏腑(12维)的**软分布**弱标签，作为证候头的训练目标。
LLM 给出的分布天然稀疏且有临床含义 —— 正好补上无监督证候头学不出的稀疏信号。

对齐保证（务必理解）：
  - 标签按 train.txt / test.txt 的**行顺序**对齐（与 torch_data.Data.train_pres 一致）。
  - 八纲 8 维顺序固定为 [阴,阳,表,里,寒,热,虚,实]（与模型 M_E2QI 行序一致）。
  - 脏腑 12 维顺序 = herb_property_meridian.xlsx 第3列起的 12 个归经列顺序
    （与模型 M_Z2MER=单位阵 → Z 的语义一致），脚本自动读取该表头确定顺序。

依赖：pip install openai pandas openpyxl numpy
API：阿里云百炼(DashScope) 的 OpenAI 兼容端点；设置环境变量 DASHSCOPE_API_KEY，
     或用 --api_key 传入。模型默认 qwen-plus。

== 使用步骤 ==
1) 先试跑 20 条，肉眼确认“症状/中药解码正确 + 标签合理”，再花钱跑全量：
   python llm_syndrome_label.py --stage trial --trial 20
2) 全量标注（带断点续跑；中途崩了再跑会跳过已完成的）：
   python llm_syndrome_label.py --stage all --splits train test --workers 8
   产出 output/<ds>/syn_labels/{eight_train.npy, zangfu_train.npy, ...}
"""
import os
import re
import csv
import json
import time
import argparse
import hashlib
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed

import numpy as np

try:
    import pandas as pd
except Exception:
    pd = None

# ===== 固定的八纲顺序（与模型 M_E2QI 行序严格一致，不要改）=====
EIGHT_NAMES = ["阴", "阳", "表", "里", "寒", "热", "虚", "实"]
EIGHT_GLOSS = "阴, 阳, 表(表证), 里(里证), 寒(寒证), 热(热证), 虚(虚证), 实(实证)"


# ----------------------------------------------------------------------------
# 命令行
# ----------------------------------------------------------------------------
def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--data_path", default="./Data/")
    p.add_argument("--dataset", default="Set2Set")
    p.add_argument("--prop_meridian", default="herb_property_meridian.xlsx",
                   help="用于读取脏腑(归经)12维的列顺序")
    p.add_argument("--sym_vocab", default="",
                   help="症状ID->名 词典文件；留空则在数据目录自动探测")
    p.add_argument("--herb_vocab", default="",
                   help="中药ID->名 词典文件；留空则用属性xlsx第1列(名)/第2列(id)")
    p.add_argument("--prop_qi", default="herb_property_qi.xlsx",
                   help="用于从中读取中药名(第1列)与id(第2列)")
    p.add_argument("--splits", nargs="+", default=["train", "test"])
    p.add_argument("--use_herbs", type=int, default=1,
                   help="1=提示词里包含所用中药作为辅助证据(标签更准);0=只用症状")
    # API
    p.add_argument("--api_key", default=os.environ.get("DASHSCOPE_API_KEY", ""))
    p.add_argument("--base_url", default="https://dashscope.aliyuncs.com/compatible-mode/v1")
    p.add_argument("--model", default="qwen-plus")
    p.add_argument("--temperature", type=float, default=0.2)
    p.add_argument("--workers", type=int, default=8)
    p.add_argument("--max_retries", type=int, default=4)
    # 运行控制
    p.add_argument("--stage", choices=["trial", "label", "assemble", "all"], default="all")
    p.add_argument("--trial", type=int, default=20, help="trial 模式标注条数")
    p.add_argument("--limit", type=int, default=0, help=">0 时只标注前 N 个去重任务(调试用)")
    p.add_argument("--out_subdir", default="syn_labels")
    return p.parse_args()


# ----------------------------------------------------------------------------
# 读处方（与 torch_data 完全一致的解析，保证行序对齐）
# ----------------------------------------------------------------------------
def parse_line(line):
    line = line.strip()
    if not line or "\t" not in line:
        return None, None
    left, right = line.split("\t", 1)
    users = [int(x) for x in left.split() if x != ""]
    items = [int(x) for x in right.split() if x != ""]
    return users, items


def read_pres(path):
    pres = []
    with open(path, "r", encoding="utf-8") as f:
        for raw in f:
            u, it = parse_line(raw)
            if u is None:
                continue
            pres.append((u, it))
    return pres


# ----------------------------------------------------------------------------
# 词典加载
# ----------------------------------------------------------------------------
def load_herb_vocab(args, ds_dir):
    if args.herb_vocab:
        return load_id_name_file(os.path.join(ds_dir, args.herb_vocab)
                                 if not os.path.isabs(args.herb_vocab) else args.herb_vocab)
    # 默认：从属性 xlsx 第1列(名)、第2列(id) 读取
    assert pd is not None, "需要 pandas/openpyxl 读取 xlsx"
    path = os.path.join(ds_dir, args.prop_qi)
    df = pd.read_excel(path)
    assert df.shape[1] >= 2, f"{path} 列数不足"
    names = df.iloc[:, 0].astype(str).tolist()
    ids = df.iloc[:, 1].astype(int).tolist()
    return {int(i): str(n) for i, n in zip(ids, names)}


def load_zangfu_names(args, ds_dir):
    assert pd is not None, "需要 pandas/openpyxl 读取 xlsx"
    path = os.path.join(ds_dir, args.prop_meridian)
    df = pd.read_excel(path)
    cols = list(df.columns[2:])  # 第3列起 = 12 个归经
    names = [str(c) for c in cols[:12]]
    if len(names) < 12:
        raise ValueError(f"meridian 列数不足12: 实际 {len(names)} 列，请检查 {path}")
    return names


def load_sym_vocab(args, ds_dir):
    if args.sym_vocab:
        path = args.sym_vocab if os.path.isabs(args.sym_vocab) else os.path.join(ds_dir, args.sym_vocab)
        return load_id_name_file(path)
    # 自动探测常见命名
    cand = ["sym_mapping.txt", "symptom_dict.txt", "sym_dict.txt", "symptom.txt",
            "symptoms.txt", "sym.txt", "symptom_id.txt", "symptom_list.txt", "sym_list.txt"]
    for c in cand:
        p = os.path.join(ds_dir, c)
        if os.path.isfile(p):
            print(f"[vocab] 自动发现症状词典: {p}")
            return load_id_name_file(p)
    raise FileNotFoundError(
        "未找到症状词典文件。请用 --sym_vocab 指定（格式: 每行 'id<空格/制表>名' 或 "
        "'名<空格/制表>id' 或 每行一个名(行号=id)）。")


def load_id_name_file(path):
    """鲁棒解析 id<->name 词典；自动判断列序或按行号。"""
    assert os.path.isfile(path), f"词典文件不存在: {path}"
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for raw in f:
            s = raw.rstrip("\n").rstrip("\r")
            if s.strip() == "":
                rows.append(None)
            else:
                rows.append(s)
    # 探测格式
    def is_int(x):
        try:
            int(x); return True
        except Exception:
            return False
    first_int = last_int = 0
    sample = [r for r in rows if r][:50]
    for r in sample:
        parts = re.split(r"[\t ]+", r.strip())
        if len(parts) >= 2 and is_int(parts[0]):
            first_int += 1
        if len(parts) >= 2 and is_int(parts[-1]):
            last_int += 1
    mapping = {}
    if first_int >= max(1, len(sample) // 2):           # 格式 A: id name
        for r in rows:
            if not r:
                continue
            parts = re.split(r"[\t ]+", r.strip(), maxsplit=1)
            if len(parts) == 2 and is_int(parts[0]):
                mapping[int(parts[0])] = parts[1].strip()
    elif last_int >= max(1, len(sample) // 2):          # 格式 B: name id
        for r in rows:
            if not r:
                continue
            parts = re.split(r"[\t ]+", r.strip())
            if is_int(parts[-1]):
                mapping[int(parts[-1])] = " ".join(parts[:-1]).strip()
    else:                                               # 格式 C: 行号=id
        for i, r in enumerate(rows):
            if r is not None:
                mapping[i] = r.strip()
    return mapping


def decode(ids, vocab):
    return [vocab.get(int(i), f"<UNK_{int(i)}>") for i in ids]


# ----------------------------------------------------------------------------
# 提示词 + 解析
# ----------------------------------------------------------------------------
def build_prompt(symptoms, herbs, zangfu_names, use_herbs):
    zf = "、".join(zangfu_names)
    sys = (
        "你是一位资深中医专家，擅长依据症状与方药进行八纲辨证与脏腑辨证。\n"
        "请对每一张处方，都**同时综合【症状】与所开【中药】的四气五味、归经**进行辨证"
        "（症状反映病机表现，中药药性反映医者的治法取向，二者需联合判断，缺一不可），"
        "给出两组软分布评分：\n"
        f"1) 八纲（8类，固定顺序：{EIGHT_GLOSS}）；\n"
        f"2) 脏腑（12类，固定顺序：{zf}）。\n"
        "硬性规则（务必遵守）：\n"
        "  a. 与该证候无关的类别给 0；只对相关类别给正分，分值表示相对程度（不必归一，我会归一）。\n"
        "  b. **严禁输出近似均匀/平局的分布**（例如把多个类别都给成相同的小值），那是无效回答。\n"
        "  c. 始终以“症状 + 中药药性”联合推断：例如温热药多提示寒证/阳虚需温补，"
        "寒凉药多提示热证/实热，归经集中提示相应脏腑受累；务必给出明确、有区分度的判断。\n"
        "  d. 八纲允许多类共存（如寒热错杂、虚实夹杂），但应有明确主次，不要四对全部等分。\n"
        "只输出严格 JSON，无多余文字：\n"
        '{"eight": {"类别":分值,...}, "zangfu": {"类别":分值,...}, "rationale":"一句话依据"}\n'
        "示例（综合症状与药性辨证）：\n"
        "输入 症状：腹痛；中药：甘草、当归、枳壳、五味子 →\n"
        '{"eight":{"虚":0.5,"里":0.3,"寒":0.2},"zangfu":{"脾":0.5,"肝":0.3,"胃":0.2},'
        '"rationale":"腹痛属里证，当归补血、甘草缓急、枳壳行气，证属脾虚肝郁、里虚为主"}'
    )
    lines = ["症状：" + "、".join(symptoms) if symptoms else "症状：（无）"]
    if use_herbs and herbs:
        lines.append("中药：" + "、".join(herbs))
    lines.append("请综合症状与中药药性输出 JSON，给出有区分度的证候判断。")
    return sys, "\n".join(lines)


def parse_label(text, zangfu_names):
    """从 LLM 文本中抽取 JSON，并映射为固定顺序的 (8,) / (12,) 向量。"""
    if text is None:
        raise ValueError("empty response")
    t = text.strip()
    # 去掉可能的 ```json ``` 围栏
    t = re.sub(r"^```(?:json)?", "", t).strip()
    t = re.sub(r"```$", "", t).strip()
    # 抽取第一个 {...}
    m = re.search(r"\{.*\}", t, flags=re.S)
    if m:
        t = m.group(0)
    obj = json.loads(t)
    e = obj.get("eight", {}) or {}
    z = obj.get("zangfu", {}) or {}

    def to_vec(d, names):
        v = np.array([max(float(d.get(n, 0.0)), 0.0) for n in names], dtype=np.float32)
        s = v.sum()
        if s <= 0:
            return np.ones(len(names), dtype=np.float32) / len(names), True
        return v / s, False

    pe, fbe = to_vec(e, EIGHT_NAMES)
    pz, fbz = to_vec(z, zangfu_names)
    return pe, pz, (fbe or fbz)


# ----------------------------------------------------------------------------
# API 调用（带重试）
# ----------------------------------------------------------------------------
def make_client(args):
    from openai import OpenAI
    key = args.api_key or os.environ.get("DASHSCOPE_API_KEY", "")
    assert key, "未提供 API key：设置环境变量 DASHSCOPE_API_KEY 或用 --api_key"
    return OpenAI(api_key=key, base_url=args.base_url)


def _is_degenerate(pe):
    """八纲分布是否近均匀/平局（最大值过低 = LLM 没给出有区分度的判断）。"""
    import numpy as _np
    return float(_np.max(pe)) < 0.20  # 8维里有区分度的判断主峰通常 >=0.3


def call_one(client, args, symptoms, herbs, zangfu_names):
    sys, usr = build_prompt(symptoms, herbs, zangfu_names, bool(args.use_herbs))
    last_err = None
    last_good = None
    for attempt in range(args.max_retries):
        try:
            # 重试时略升温度，帮助打破“平局”
            temp = args.temperature + (0.3 if attempt > 0 else 0.0)
            kw = dict(model=args.model, temperature=temp,
                      messages=[{"role": "system", "content": sys},
                                {"role": "user", "content": usr}])
            try:
                resp = client.chat.completions.create(
                    response_format={"type": "json_object"}, **kw)
            except Exception:
                resp = client.chat.completions.create(**kw)  # 模型不支持 json mode 时退化
            text = resp.choices[0].message.content
            pe, pz, fb = parse_label(text, zangfu_names)
            last_good = (pe, pz, fb)
            if not _is_degenerate(pe):
                return pe, pz, fb, None          # 有区分度，采纳
            last_err = "degenerate(near-uniform eight), retrying"
        except Exception as e:
            last_err = str(e)
            time.sleep(min(2 ** attempt, 20))     # 指数退避
    # 多次仍均匀：采纳最后一次成功解析的结果（标记 fb=True 以便统计）
    if last_good is not None:
        pe, pz, _ = last_good
        return pe, pz, True, last_err
    return None, None, True, last_err


# ----------------------------------------------------------------------------
# 任务构建（去重）
# ----------------------------------------------------------------------------
def task_key(symptoms, herbs, use_herbs):
    s = "S:" + "|".join(sorted(symptoms))
    if use_herbs:
        s += "##H:" + "|".join(sorted(herbs))
    return hashlib.sha1(s.encode("utf-8")).hexdigest()


def build_records(splits, ds_dir, sym_vocab, herb_vocab, use_herbs):
    """返回 per-split 列表[(idx, sym_names, herb_names, key)] 与 去重任务字典 key->(syms,herbs)。"""
    per_split = {}
    unique = {}
    for sp in splits:
        path = os.path.join(ds_dir, f"{sp}.txt")
        pres = read_pres(path)
        recs = []
        for idx, (us, its) in enumerate(pres):
            syms = decode(us, sym_vocab)
            herbs = decode(its, herb_vocab)
            k = task_key(syms, herbs, use_herbs)
            recs.append((idx, syms, herbs, k))
            if k not in unique:
                unique[k] = (syms, herbs)
        per_split[sp] = recs
        print(f"[{sp}] 处方 {len(recs)} 条")
    print(f"[dedup] 去重后唯一任务 {len(unique)} 个（这就是实际 API 调用数）")
    return per_split, unique


# ----------------------------------------------------------------------------
# 缓存（断点续跑）
# ----------------------------------------------------------------------------
def load_cache(cache_path):
    done = {}
    if os.path.isfile(cache_path):
        with open(cache_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    o = json.loads(line)
                    done[o["key"]] = (np.array(o["eight"], np.float32),
                                      np.array(o["zangfu"], np.float32))
                except Exception:
                    pass
    return done


# ----------------------------------------------------------------------------
# 主流程
# ----------------------------------------------------------------------------
def run_labeling(args, unique, zangfu_names, cache_path, limit=0):
    done = load_cache(cache_path)
    todo = [k for k in unique if k not in done]
    if limit > 0:
        todo = todo[:limit]
    print(f"[label] 待标注 {len(todo)} / 已缓存 {len(done)}")
    if not todo:
        return done

    client = make_client(args)
    lock = threading.Lock()
    fout = open(cache_path, "a", encoding="utf-8")
    n_ok = n_fb = n_err = 0

    def work(k):
        syms, herbs = unique[k]
        pe, pz, fb, err = call_one(client, args, syms, herbs, zangfu_names)
        return k, pe, pz, fb, err

    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        futs = [ex.submit(work, k) for k in todo]
        for i, fut in enumerate(as_completed(futs)):
            k, pe, pz, fb, err = fut.result()
            if pe is None:
                n_err += 1
                print(f"  [ERR] {k[:8]} {err}")
                continue
            with lock:
                done[k] = (pe, pz)
                fout.write(json.dumps({"key": k, "eight": pe.tolist(),
                                       "zangfu": pz.tolist()},
                                      ensure_ascii=False) + "\n")
                fout.flush()
            n_ok += 1
            if fb:
                n_fb += 1
            if (i + 1) % 50 == 0:
                print(f"  进度 {i+1}/{len(todo)}  ok={n_ok} fallback={n_fb} err={n_err}")
    fout.close()
    print(f"[label] 完成 ok={n_ok} fallback(全零->均匀)={n_fb} err={n_err}")
    return done


def assemble(args, per_split, done, zangfu_names, out_dir):
    os.makedirs(out_dir, exist_ok=True)
    stats = {"eight_names": EIGHT_NAMES, "zangfu_names": zangfu_names,
             "model": args.model, "use_herbs": bool(args.use_herbs)}
    preview_path = os.path.join(out_dir, "labels_preview.csv")
    pf = open(preview_path, "w", newline="", encoding="utf-8-sig")
    pw = csv.writer(pf)
    pw.writerow(["split", "idx", "symptoms", "herbs", "top_eight", "top_zangfu",
                 "active_eight", "active_zangfu"])

    for sp, recs in per_split.items():
        n = len(recs)
        E = np.zeros((n, 8), np.float32)
        Z = np.zeros((n, 12), np.float32)
        miss = 0
        actE = actZ = 0.0
        for (idx, syms, herbs, k) in recs:
            if k in done:
                pe, pz = done[k]
            else:
                pe = np.ones(8, np.float32) / 8
                pz = np.ones(12, np.float32) / 12
                miss += 1
            E[idx] = pe
            Z[idx] = pz
            actE += int((pe > 1e-4).sum())
            actZ += int((pz > 1e-4).sum())
            te = EIGHT_NAMES[int(pe.argmax())]
            tz = zangfu_names[int(pz.argmax())]
            pw.writerow([sp, idx, "、".join(syms), "、".join(herbs), te, tz,
                         int((pe > 1e-4).sum()), int((pz > 1e-4).sum())])
        np.save(os.path.join(out_dir, f"eight_{sp}.npy"), E)
        np.save(os.path.join(out_dir, f"zangfu_{sp}.npy"), Z)
        stats[f"{sp}_n"] = n
        stats[f"{sp}_missing"] = miss
        stats[f"{sp}_avg_active_eight"] = round(actE / max(n, 1), 3)
        stats[f"{sp}_avg_active_zangfu"] = round(actZ / max(n, 1), 3)
        print(f"[assemble] {sp}: 保存 eight_{sp}.npy{E.shape} zangfu_{sp}.npy{Z.shape} "
              f"缺失={miss} 平均激活 八纲={stats[f'{sp}_avg_active_eight']} "
              f"脏腑={stats[f'{sp}_avg_active_zangfu']}")
    pf.close()
    with open(os.path.join(out_dir, "meta.json"), "w", encoding="utf-8") as f:
        json.dump(stats, f, ensure_ascii=False, indent=2)
    print(f"[assemble] 预览: {preview_path}")
    print(f"[assemble] meta: {os.path.join(out_dir,'meta.json')}")


def main():
    args = parse_args()
    ds_dir = os.path.join(args.data_path, args.dataset)
    out_dir = os.path.join("output", args.dataset, args.out_subdir)
    cache_path = os.path.join("output", args.dataset, "syn_label_cache.jsonl")
    os.makedirs(os.path.dirname(cache_path), exist_ok=True)

    # 词典
    herb_vocab = load_herb_vocab(args, ds_dir)
    sym_vocab = load_sym_vocab(args, ds_dir)
    zangfu_names = load_zangfu_names(args, ds_dir)
    print(f"[vocab] 症状 {len(sym_vocab)} 个, 中药 {len(herb_vocab)} 个")
    print(f"[vocab] 八纲顺序: {EIGHT_NAMES}")
    print(f"[vocab] 脏腑顺序(取自 {args.prop_meridian}): {zangfu_names}")

    per_split, unique = build_records(args.splits, ds_dir, sym_vocab, herb_vocab,
                                      bool(args.use_herbs))

    # ===== TRIAL：标注少量并打印，供肉眼验证（解码是否正确 + 标签是否合理）=====
    if args.stage == "trial":
        client = make_client(args)
        shown = 0
        for sp, recs in per_split.items():
            for (idx, syms, herbs, k) in recs:
                if shown >= args.trial:
                    break
                pe, pz, fb, err = call_one(client, args, syms, herbs, zangfu_names)
                print("\n" + "=" * 70)
                print(f"[{sp}#{idx}] 症状: {('、'.join(syms)) or '(无)'}")
                print(f"           中药: {('、'.join(herbs)) or '(无)'}")
                if pe is None:
                    print(f"   [ERR] {err}")
                else:
                    topE = sorted(zip(EIGHT_NAMES, pe), key=lambda x: -x[1])[:4]
                    topZ = sorted(zip(zangfu_names, pz), key=lambda x: -x[1])[:4]
                    print("   八纲:", {n: round(float(v), 2) for n, v in topE if v > 1e-4})
                    print("   脏腑:", {n: round(float(v), 2) for n, v in topZ if v > 1e-4})
                shown += 1
            if shown >= args.trial:
                break
        print("\n请确认：症状/中药中文名是否正确？八纲/脏腑标签是否合理？"
              "无误后再跑 --stage all。")
        return

    # ===== LABEL + ASSEMBLE =====
    if args.stage in ("label", "all"):
        done = run_labeling(args, unique, zangfu_names, cache_path, limit=args.limit)
    else:
        done = load_cache(cache_path)

    if args.stage in ("assemble", "all"):
        assemble(args, per_split, done, zangfu_names, out_dir)


if __name__ == "__main__":
    main()