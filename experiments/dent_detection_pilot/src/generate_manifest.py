"""
Generate Leakage-Safe 7-Class Object Detection Dataset Manifest for CR7-DET.
Offline Dent Pilot Experiment.

Guarantees:
1. Reuses exact 600 Dent image split assignments from Step 2 (420 Train / 90 Val / 90 Test).
2. Uses SHA-256 hash grouping so identical images are never split across train/val/test.
3. Preserves deterministic 70/15/15 ratio across all 7 defect classes.
"""

import json
import csv
import hashlib
import random
from pathlib import Path
from collections import defaultdict, Counter

# Fixed random seed for complete reproducibility
SEED = 20261020
random.seed(SEED)

BASE_DIR = Path(r"D:\DefectVision\experiments\dent_detection_pilot")
MANIFEST_DIR = BASE_DIR / "manifests"
SPLITS_DIR = BASE_DIR / "splits"

CR7_IMG_DIR = Path(r"C:\Users\susha\Downloads\CR7-DET\CR7-DET\image")
CR7_LBL_DIR = Path(r"C:\Users\susha\Downloads\CR7-DET\CR7-DET\label")
DENT_MANIFEST_PATH = Path(r"D:\DefectVision\experiments\dent_classification\manifests\dent_pilot_manifest.jsonl")

CLASS_NAMES = {
    0: "Inclusion",
    1: "Dents",
    2: "Oil Spots",
    3: "Pits",
    4: "Punching",
    5: "Linear",
    6: "Macular"
}


def main():
    print("=== Generating CR7-DET 7-Class Detection Manifest ===")

    # 1. Load Step 2 Dent split assignments
    dent_records = [json.loads(line) for line in DENT_MANIFEST_PATH.read_text(encoding="utf-8").splitlines() if line.strip()]
    dent_image_splits = {r["source_image_stem"]: r["split"] for r in dent_records}
    print(f"Loaded existing Dent image splits: {len(dent_image_splits)} images ({Counter(dent_image_splits.values())})")

    # 2. Index all images and labels
    images = sorted(list(CR7_IMG_DIR.glob("*.jpg")))
    print(f"Total raw images in CR7-DET: {len(images)}")

    hash_to_stems = defaultdict(list)
    stem_to_hash = {}
    stem_to_annotations = {}

    for img_p in images:
        stem = img_p.stem
        img_bytes = img_p.read_bytes()
        h = hashlib.sha256(img_bytes).hexdigest()
        hash_to_stems[h].append(stem)
        stem_to_hash[stem] = h

        lbl_p = CR7_LBL_DIR / f"{stem}.txt"
        boxes = []
        if lbl_p.exists():
            for line in lbl_p.read_text(encoding="utf-8").splitlines():
                parts = line.strip().split()
                if len(parts) == 5:
                    cls_id = int(parts[0])
                    cx, cy, w, h_box = map(float, parts[1:5])
                    boxes.append({
                        "class_id": cls_id,
                        "class_name": CLASS_NAMES[cls_id],
                        "bbox_yolo": [round(cx, 6), round(cy, 6), round(w, 6), round(h_box, 6)]
                    })
        stem_to_annotations[stem] = boxes

    # 3. Group images by hash cluster
    all_assigned_splits = {}

    # Assign Dent clusters first (from existing Step 2 split)
    dent_hashes = set()
    for stem, split in dent_image_splits.items():
        h = stem_to_hash[stem]
        dent_hashes.add(h)
        for s in hash_to_stems[h]:
            all_assigned_splits[s] = split

    # Handle cross-class duplicate pair ['inclusions236', 'macular522'] -> assign to train
    cross_cluster_stems = {"inclusions236", "macular522"}
    cross_h = stem_to_hash["inclusions236"]
    for s in hash_to_stems[cross_h]:
        all_assigned_splits[s] = "train"

    # Identify remaining clusters per primary class
    remaining_clusters_by_class = defaultdict(list)
    processed_hashes = set(dent_hashes)
    processed_hashes.add(cross_h)

    for h, stems in hash_to_stems.items():
        if h in processed_hashes:
            continue
        # determine primary class from the first stem's annotations
        classes_in_cluster = [b["class_id"] for s in stems for b in stem_to_annotations[s]]
        primary_class = Counter(classes_in_cluster).most_common(1)[0][0]
        remaining_clusters_by_class[primary_class].append((h, stems))

    # Split remaining clusters with fixed seed
    for cls_id in sorted(remaining_clusters_by_class.keys()):
        clusters = remaining_clusters_by_class[cls_id]
        # Deterministic shuffle
        random.shuffle(clusters)
        n = len(clusters)
        n_train = int(round(n * 0.70))
        n_val = int(round(n * 0.15))
        # remainder goes to test

        train_clusters = clusters[:n_train]
        val_clusters = clusters[n_train:n_train + n_val]
        test_clusters = clusters[n_train + n_val:]

        for h, stems in train_clusters:
            for s in stems:
                all_assigned_splits[s] = "train"
        for h, stems in val_clusters:
            for s in stems:
                all_assigned_splits[s] = "val"
        for h, stems in test_clusters:
            for s in stems:
                all_assigned_splits[s] = "test"

    # 4. Verification
    assert len(all_assigned_splits) == 4140, f"Expected 4140 images, got {len(all_assigned_splits)}"

    # Verify zero duplicate hash leakage across splits
    split_hashes = defaultdict(set)
    for s, split in all_assigned_splits.items():
        split_hashes[split].add(stem_to_hash[s])

    assert len(split_hashes["train"] & split_hashes["val"]) == 0, "Train-Val hash leakage detected!"
    assert len(split_hashes["train"] & split_hashes["test"]) == 0, "Train-Test hash leakage detected!"
    assert len(split_hashes["val"] & split_hashes["test"]) == 0, "Val-Test hash leakage detected!"
    print("Leakage verification passed: 0% duplicate hash overlap across splits!")

    split_counts = Counter(all_assigned_splits.values())
    print(f"Overall image split counts: {split_counts}")

    # Compute class annotation breakdown per split
    split_class_boxes = defaultdict(lambda: Counter())
    for s, split in all_assigned_splits.items():
        for b in stem_to_annotations[s]:
            split_class_boxes[split][b["class_name"]] += 1

    print("\n--- Bounding Boxes per Class per Split ---")
    for cls_name in CLASS_NAMES.values():
        t = split_class_boxes["train"][cls_name]
        v = split_class_boxes["val"][cls_name]
        te = split_class_boxes["test"][cls_name]
        tot = t + v + te
        print(f"  {cls_name:10s} : Train={t:4d} ({t/tot*100:4.1f}%), Val={v:3d} ({v/tot*100:4.1f}%), Test={te:3d} ({te/tot*100:4.1f}%), Total={tot:4d}")

    # 5. Export Manifests
    manifest_rows = []
    for img_p in images:
        stem = img_p.stem
        split = all_assigned_splits[stem]
        boxes = stem_to_annotations[stem]
        manifest_rows.append({
            "image_stem": stem,
            "image_path": str(img_p),
            "label_path": str(CR7_LBL_DIR / f"{stem}.txt"),
            "image_sha256": stem_to_hash[stem],
            "split": split,
            "width": 400,
            "height": 400,
            "channels": 3,
            "num_boxes": len(boxes),
            "classes_present": sorted(list(set(b["class_name"] for b in boxes))),
            "annotations": boxes
        })

    jsonl_path = MANIFEST_DIR / "detection_dataset_manifest.jsonl"
    with open(jsonl_path, "w", encoding="utf-8") as f:
        for row in manifest_rows:
            f.write(json.dumps(row) + "\n")
    print(f"\nWrote JSONL manifest: {jsonl_path} ({len(manifest_rows)} records)")

    # CSV manifest (simplified without nested annotations for easy viewing in spreadsheets)
    csv_path = MANIFEST_DIR / "detection_dataset_manifest.csv"
    with open(csv_path, "w", encoding="utf-8", newline="") as f:
        fieldnames = ["image_stem", "split", "image_sha256", "num_boxes", "classes_present", "image_path", "label_path"]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for r in manifest_rows:
            writer.writerow({
                "image_stem": r["image_stem"],
                "split": r["split"],
                "image_sha256": r["image_sha256"],
                "num_boxes": r["num_boxes"],
                "classes_present": ";".join(r["classes_present"]),
                "image_path": r["image_path"],
                "label_path": r["label_path"]
            })
    print(f"Wrote CSV manifest: {csv_path}")

    # 6. Export split text files (YOLO format: list of image paths)
    for split_name in ["train", "val", "test"]:
        split_file = SPLITS_DIR / f"{split_name}.txt"
        split_images = [r["image_path"] for r in manifest_rows if r["split"] == split_name]
        split_file.write_text("\n".join(split_images) + "\n", encoding="utf-8")
        print(f"Wrote split file: {split_file} ({len(split_images)} images)")

    print("\nDataset preparation complete!")


if __name__ == "__main__":
    main()
