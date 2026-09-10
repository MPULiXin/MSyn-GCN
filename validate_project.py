from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from msyn_gcn.data import MSynDataset
from msyn_gcn.spec import PROJECT_ROOT, load_paper_spec


def main() -> None:
    spec = load_paper_spec()
    data = MSynDataset(Path(PROJECT_ROOT) / "data" / "Set2Set", load_train_labels=True)
    audit = data.audit()
    for split, expected in spec["dataset"]["split_counts"].items():
        assert audit["splits"][split]["prescriptions"] == expected
        assert audit["splits"][split]["unique_symptom_sets"] == spec["dataset"]["unique_symptom_sets"][split]
        assert audit["splits"][split]["used_symptoms"] == spec["dataset"]["used_symptoms"][split]
        assert audit["splits"][split]["used_herbs"] == spec["dataset"]["used_herbs"][split]
    assert audit["graph_edges"] == {"symptom": 4462, "herb": 50419}
    assert data.symptom_adj.diagonal().sum() == 0
    assert data.herb_adj.diagonal().sum() == 0
    assert data.train_labels[0].shape == (20625, 8)
    assert data.train_labels[1].shape == (20625, 12)
    assert np.allclose(data.train_labels[0].sum(axis=1), 1.0)
    assert np.allclose(data.train_labels[1].sum(axis=1), 1.0)

    val_seen = audit["splits"]["val"]["seen_in_train_fraction"]
    test_seen = audit["splits"]["test"]["seen_in_train_fraction"]
    print("项目数据校验通过")
    print(f"  处方划分: train=20625, val=2292, test=3443")
    print(f"  词表: 症状=360, 草药=753")
    print(f"  验证/测试症状集见于训练集: {val_seen:.1%} / {test_seen:.1%}")
    print(f"  训练集动态构图边数: symptom=4462, herb=50419")
    print("  训练弱标签: (20625,8) 与 (20625,12)，行序及归一化正确")

    # These are the counts reported in Additional file 1, Table S7(d).
    meridian_sums = audit["property_column_sums"]["meridian"]
    assert meridian_sums == [312, 7, 169, 133, 3, 30, 260, 30, 56, 219, 379, 186]
    print("  药性矩阵与附录 S7(d) 一致: Heart=169, Kidney=186")
    flavor_zero = int((data.properties["flavor"].sum(axis=1) == 0).sum())
    if flavor_zero:
        zero_ids = np.flatnonzero(data.properties["flavor"].sum(axis=1) == 0).tolist()
        names = [data.herb_names[index] for index in zero_ids]
        print(f"  [论文数据提示] 五味表有 {flavor_zero} 个全零行: {list(zip(zero_ids, names))}")

    output = Path(PROJECT_ROOT) / "results" / "data_audit.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(audit, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"  审计记录: {output}")


if __name__ == "__main__":
    main()
