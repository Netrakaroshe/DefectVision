import json, collections
from pathlib import Path

splits = {}
for split in ["train", "val", "test"]:
    p = Path("experiments/defect_classifier_dataset") / (split + ".jsonl")
    splits[split] = [json.loads(l) for l in p.read_text(encoding="utf-8").splitlines() if l.strip()]
all_recs = [r for recs in splits.values() for r in recs]

CLASSES = ["Scratch", "Crack", "Discoloration", "UNKNOWN"]

print("=== CLASS-SOURCE EXCLUSIVITY ===")
by_cls_ds = collections.defaultdict(set)
for r in all_recs:
    by_cls_ds[r["target_label"]].add(r.get("dataset", "?"))
for cls in CLASSES:
    ds_set = sorted(by_cls_ds[cls])
    multi = "MULTI-SOURCE" if len(ds_set) > 1 else "SINGLE-SOURCE"
    print(f"  {cls}: {ds_set} -> {multi}")

print()
print("=== SOURCE FRACTIONS PER CLASS (all splits combined) ===")
for cls in CLASSES:
    sub = [r for r in all_recs if r["target_label"] == cls]
    by_src = collections.Counter(r.get("dataset","?") for r in sub)
    total = sum(by_src.values())
    fracs = ", ".join(f"{k}={v}/{total}={round(v/total*100,1)}%" for k,v in sorted(by_src.items()))
    print(f"  {cls}: {fracs}")

print()
print("=== TRAIN SPLIT: source fractions per class ===")
train = splits["train"]
for cls in CLASSES:
    sub = [r for r in train if r["target_label"] == cls]
    by_src = collections.Counter(r.get("dataset","?") for r in sub)
    total = sum(by_src.values())
    fracs = ", ".join(f"{k}={v}({round(v/total*100,1)}%)" for k,v in sorted(by_src.items()))
    print(f"  {cls} n={total}: {fracs}")

print()
print("=== MVTec distribution across splits ===")
for split in ["train", "val", "test"]:
    mv = [r for r in splits[split] if r.get("dataset") == "MVTecAD"]
    by_lbl = collections.Counter(r["target_label"] for r in mv)
    print(f"  {split} MVTec (n={len(mv)}): " + ", ".join(f"{cls}={by_lbl.get(cls,0)}" for cls in CLASSES))

print()
print("=== MVTec TEST: product x class ===")
test_mv = [r for r in splits["test"] if r.get("dataset") == "MVTecAD"]
by_prod_lbl = collections.defaultdict(lambda: collections.Counter())
for r in test_mv:
    by_prod_lbl[r["product"]][r["target_label"]] += 1
for prod, cnts in sorted(by_prod_lbl.items()):
    print(f"  {prod}: " + ", ".join(f"{c}={cnts.get(c,0)}" for c in CLASSES))

print()
print("=== MVTec TEST diagnostic performance ===")
diag = json.loads(Path("experiments/defect_classifier_diagnostics/source_dataset_metrics.json").read_text())
mv = diag["MVTecAD"]
n = mv["n"]
acc = mv["accuracy"]
mf1 = mv["macro_f1"]
print(f"  n={n}  accuracy={acc}  macro_f1={mf1}")
for cls in CLASSES:
    pc = mv["per_class"][cls]
    print(f"  {cls}: P={pc['precision']}  R={pc['recall']}  F1={pc['f1']}")
print("  confusion matrix (rows=true, cols=pred):")
for i, row in enumerate(mv["confusion_matrix"]):
    print(f"    {CLASSES[i]}: {row}")

print()
print("=== GENERALIZATION FEASIBILITY ===")
print("Can we build a cross-source generalization eval from EXISTING frozen data?")
print()
mv_test = [r for r in splits["test"] if r.get("dataset") == "MVTecAD"]
for cls in CLASSES:
    n_mv_test = sum(1 for r in mv_test if r["target_label"] == cls)
    n_other_sources_train = sum(1 for r in splits["train"] if r["target_label"] == cls and r.get("dataset") != "MVTecAD")
    print(f"  {cls}: MVTec test n={n_mv_test}, non-MVTec train examples={n_other_sources_train}")
