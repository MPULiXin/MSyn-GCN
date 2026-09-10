from __future__ import annotations

import argparse
import hashlib
import json
import os
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import numpy as np
from openai import OpenAI

from msyn_gcn.data import MSynDataset
from msyn_gcn.spec import PROJECT_ROOT, load_paper_spec


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Offline Qwen-Plus weak-label generation")
    parser.add_argument("--splits", nargs="+", choices=["train", "val", "test"], default=["train"])
    parser.add_argument("--trial", type=int, default=0, help="Only label the first N unique inputs")
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--model", default="qwen-plus-2025-12-01")
    parser.add_argument("--temperature", type=float, default=0.2)
    parser.add_argument("--max-retries", type=int, default=4)
    parser.add_argument(
        "--base-url",
        default="https://dashscope.aliyuncs.com/compatible-mode/v1",
    )
    parser.add_argument("--api-key", default=os.environ.get("DASHSCOPE_API_KEY", ""))
    return parser.parse_args()


def task_key(symptoms: list[str], herbs: list[str]) -> str:
    value = "S:" + "|".join(sorted(symptoms)) + "##H:" + "|".join(sorted(herbs))
    return hashlib.sha1(value.encode("utf-8")).hexdigest()


def normalize_scores(raw: dict, names: list[str]) -> tuple[np.ndarray, bool]:
    values = np.asarray([max(0.0, float(raw.get(name, 0.0))) for name in names], dtype=np.float32)
    fallback = float(values.sum()) <= 0
    if fallback:
        values.fill(1.0 / len(values))
    else:
        values /= values.sum()
    return values, fallback


def parse_response(text: str, eight_names: list[str], zangfu_names: list[str]):
    stripped = text.strip()
    if stripped.startswith("```"):
        stripped = stripped.strip("`")
        if stripped.startswith("json"):
            stripped = stripped[4:].lstrip()
    start = stripped.find("{")
    if start < 0:
        raise ValueError("No JSON object in response")
    value, _ = json.JSONDecoder().raw_decode(stripped[start:])
    eight, fallback_eight = normalize_scores(value.get("eight", {}), eight_names)
    zangfu, fallback_zangfu = normalize_scores(value.get("zangfu", {}), zangfu_names)
    return eight, zangfu, bool(fallback_eight or fallback_zangfu)


def build_messages(
    prompt_template: str,
    symptoms: list[str],
    herbs: list[str],
    zangfu_names: list[str],
):
    system = prompt_template.replace("{ZANGFU_NAMES}", "、".join(zangfu_names))
    user = (
        "症状：" + ("、".join(symptoms) if symptoms else "（无）") + "\n"
        "中药：" + "、".join(herbs) + "\n"
        "请综合症状与中药药性输出 JSON，给出有区分度的证候判断。"
    )
    return [{"role": "system", "content": system}, {"role": "user", "content": user}]


def load_cache(path: Path) -> dict[str, dict]:
    cache = {}
    if path.exists():
        for raw in path.read_text(encoding="utf-8").splitlines():
            if raw.strip():
                row = json.loads(raw)
                cache[row["key"]] = row
    return cache


def main() -> None:
    args = parse_args()
    if not args.api_key:
        raise SystemExit("Set DASHSCOPE_API_KEY or pass --api-key. Existing released labels need no API.")
    spec = load_paper_spec()
    eight_names = spec["categories"]["eight_principles"]
    zangfu_names = spec["categories"]["zangfu"]
    data = MSynDataset(PROJECT_ROOT / "data" / "Set2Set", load_train_labels=False)
    prompt = (PROJECT_ROOT / "paper" / "llm_prompt_zh.txt").read_text(encoding="utf-8")
    output_dir = PROJECT_ROOT / "results" / "generated_weak_labels"
    output_dir.mkdir(parents=True, exist_ok=True)
    cache_path = output_dir / "cache.jsonl"
    cache = load_cache(cache_path)

    split_records = {}
    tasks = {}
    for split in args.splits:
        records = []
        for row in data.splits[split]:
            symptoms = [data.symptom_names[value] for value in row.symptoms]
            herbs = [data.herb_names[value] for value in row.herbs]
            key = task_key(symptoms, herbs)
            records.append(key)
            tasks.setdefault(key, (symptoms, herbs))
        split_records[split] = records
    todo = [key for key in tasks if key not in cache]
    if args.trial > 0:
        todo = todo[: args.trial]
    print(f"Unique inputs={len(tasks)}, cached={len(cache)}, to label={len(todo)}")

    client = OpenAI(api_key=args.api_key, base_url=args.base_url)
    lock = threading.Lock()

    def call(key: str) -> dict:
        symptoms, herbs = tasks[key]
        messages = build_messages(prompt, symptoms, herbs, zangfu_names)
        last_error = None
        last_success = None
        for attempt in range(args.max_retries):
            try:
                temperature = args.temperature + (0.3 if attempt else 0.0)
                request = dict(model=args.model, temperature=temperature, messages=messages)
                try:
                    response = client.chat.completions.create(
                        response_format={"type": "json_object"},
                        **request,
                    )
                except Exception:
                    response = client.chat.completions.create(**request)
                eight, zangfu, fallback = parse_response(
                    response.choices[0].message.content,
                    eight_names,
                    zangfu_names,
                )
                last_success = (eight, zangfu, fallback)
                if float(eight.max()) >= 0.20:
                    break
                last_error = "near-uniform Eight-Principles response"
            except Exception as exc:
                last_error = str(exc)
                time.sleep(min(2**attempt, 20))
        if last_success is None:
            eight = np.full(8, 1 / 8, dtype=np.float32)
            zangfu = np.full(12, 1 / 12, dtype=np.float32)
            fallback = True
        else:
            eight, zangfu, fallback = last_success
        return {
            "key": key,
            "eight": eight.tolist(),
            "zangfu": zangfu.tolist(),
            "fallback": bool(fallback),
            "error": last_error,
        }

    with cache_path.open("a", encoding="utf-8") as stream, ThreadPoolExecutor(
        max_workers=args.workers
    ) as executor:
        futures = {executor.submit(call, key): key for key in todo}
        for index, future in enumerate(as_completed(futures), start=1):
            record = future.result()
            with lock:
                stream.write(json.dumps(record, ensure_ascii=False) + "\n")
                stream.flush()
                cache[record["key"]] = record
            if index % 50 == 0:
                print(f"Completed {index}/{len(todo)}")

    if args.trial > 0:
        print("Trial complete. No arrays assembled; inspect cache.jsonl before a full run.")
        return
    for split, keys in split_records.items():
        missing = [key for key in keys if key not in cache]
        if missing:
            raise RuntimeError(f"Cannot assemble {split}; missing {len(missing)} cache rows")
        eight = np.asarray([cache[key]["eight"] for key in keys], dtype=np.float32)
        zangfu = np.asarray([cache[key]["zangfu"] for key in keys], dtype=np.float32)
        np.save(output_dir / f"eight_{split}.npy", eight)
        np.save(output_dir / f"zangfu_{split}.npy", zangfu)
    print(f"Generated arrays saved under {output_dir}; released paper arrays were not overwritten.")


if __name__ == "__main__":
    main()
