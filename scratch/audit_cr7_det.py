import os
import sys
import glob
from pathlib import Path
from collections import Counter, defaultdict
import hashlib
from PIL import Image
import numpy as np

def run_audit():
    print("=== CR7-DET DATASET AUDIT ===")
    root_dir = Path(r"C:\Users\susha\Downloads\CR7-DET")
    nested_dir = root_dir / "CR7-DET"
    img_dir = nested_dir / "image"
    lbl_dir = nested_dir / "label"
    readme_path = root_dir / "README.md.txt"

    print(f"Dataset root: {root_dir}")
    print(f"Image directory: {img_dir}")
    print(f"Label directory: {lbl_dir}")
    print(f"Readme path: {readme_path}")
    print(f"Readme exists: {readme_path.exists()}")

    # 1. Inspect files
    image_files = sorted(list(img_dir.glob("*.jpg")) + list(img_dir.glob("*.jpeg")) + list(img_dir.glob("*.png")))
    label_files = sorted(list(lbl_dir.glob("*.txt")))

    total_images = len(image_files)
    total_labels = len(label_files)
    print(f"\nTotal image files: {total_images}")
    print(f"Total label files: {total_labels}")

    # Check 1:1 stem pairing
    img_map = {f.stem: f for f in image_files}
    lbl_map = {f.stem: f for f in label_files}

    missing_labels = [stem for stem in img_map if stem not in lbl_map]
    orphaned_labels = [stem for stem in lbl_map if stem not in img_map]
    print(f"Images without label file: {len(missing_labels)}")
    print(f"Labels without image file: {len(orphaned_labels)}")

    # 2. Check duplicate filenames (case insensitive)
    lower_img_names = [f.name.lower() for f in image_files]
    dup_img_names = [name for name, count in Counter(lower_img_names).items() if count > 1]
    print(f"Duplicate image filenames (case-insensitive): {len(dup_img_names)}")

    # 3. Check identical image contents via SHA-256
    print("\nComputing SHA-256 image hashes for content duplication check...")
    hash_to_files = defaultdict(list)
    for img_path in image_files:
        hasher = hashlib.sha256()
        with open(img_path, "rb") as f:
            while chunk := f.read(65536):
                hasher.update(chunk)
        digest = hasher.hexdigest()
        hash_to_files[digest].append(img_path.name)

    duplicate_content_groups = {h: files for h, files in hash_to_files.items() if len(files) > 1}
    total_duplicate_images = sum(len(files) - 1 for files in duplicate_content_groups.values())
    print(f"Unique content hashes: {len(hash_to_files)} out of {total_images} images")
    print(f"Identical image content clusters: {len(duplicate_content_groups)}")
    print(f"Redundant duplicate image copies: {total_duplicate_images}")
    if duplicate_content_groups:
        print("Sample duplicates:")
        for h, files in list(duplicate_content_groups.items())[:5]:
            print(f"  Hash {h[:12]}... : {files}")

    # 4. Check image dimensions, channels, formats
    print("\nInspecting image dimensions, color channels, and color variance...")
    dimensions = Counter()
    modes = Counter()
    formats = Counter()
    is_true_color = 0
    is_grayscale_rgb = 0

    # sample check color vs grayscale
    sample_stride = 10  # check every 10th image for exact channel diff, but check dimensions for all
    for idx, img_path in enumerate(image_files):
        with Image.open(img_path) as im:
            dimensions[im.size] += 1
            modes[im.mode] += 1
            formats[im.format] += 1
            if idx % sample_stride == 0:
                arr = np.array(im)
                if len(arr.shape) == 3 and arr.shape[2] == 3:
                    diff1 = np.max(np.abs(arr[:, :, 0] - arr[:, :, 1]))
                    diff2 = np.max(np.abs(arr[:, :, 1] - arr[:, :, 2]))
                    if diff1 == 0 and diff2 == 0:
                        is_grayscale_rgb += 1
                    else:
                        is_true_color += 1

    print(f"Dimensions distribution: {dict(dimensions)}")
    print(f"Modes distribution: {dict(modes)}")
    print(f"Formats distribution: {dict(formats)}")
    print(f"Sampled color check ({is_true_color + is_grayscale_rgb} images): {is_true_color} true color, {is_grayscale_rgb} grayscale in 3-channel RGB")

    # 5. Annotation parsing
    print("\nParsing annotations...")
    class_names = {
        0: "Inclusion",
        1: "Dents",
        2: "Oil Spots",
        3: "Pits",
        4: "Punching",
        5: "Linear",
        6: "Macular"
    }

    class_instance_counts = Counter()
    images_with_class = defaultdict(set)
    empty_annotation_files = []
    malformed_lines = []
    out_of_bounds_coords = []
    
    # Store per image details
    image_to_annotations = defaultdict(list)
    dent_bbox_areas = []
    dent_bbox_aspect_ratios = []
    dent_bbox_coords = []

    for lbl_path in label_files:
        stem = lbl_path.stem
        with open(lbl_path, "r", encoding="utf-8") as f:
            lines = [l.strip() for l in f if l.strip()]

        if not lines:
            empty_annotation_files.append(lbl_path.name)
            continue

        for line_no, line in enumerate(lines, 1):
            parts = line.split()
            if len(parts) != 5:
                malformed_lines.append((lbl_path.name, line_no, line, "Expected 5 fields"))
                continue
            try:
                cid = int(parts[0])
                xc, yc, w, h = float(parts[1]), float(parts[2]), float(parts[3]), float(parts[4])
            except ValueError as e:
                malformed_lines.append((lbl_path.name, line_no, line, str(e)))
                continue

            if not (0.0 <= xc <= 1.0 and 0.0 <= yc <= 1.0 and 0.0 <= w <= 1.0 and 0.0 <= h <= 1.0):
                out_of_bounds_coords.append((lbl_path.name, line_no, xc, yc, w, h))

            class_instance_counts[cid] += 1
            images_with_class[cid].add(stem)
            image_to_annotations[stem].append((cid, xc, yc, w, h))

            if cid == 1:
                dent_bbox_areas.append(w * h)
                aspect = w / (h + 1e-6)
                dent_bbox_aspect_ratios.append(aspect)
                dent_bbox_coords.append((xc, yc, w, h))

    total_instances = sum(class_instance_counts.values())
    print(f"\nTotal annotations across all classes: {total_instances}")
    print(f"Empty label files: {len(empty_annotation_files)}")
    print(f"Malformed annotation lines: {len(malformed_lines)}")
    print(f"Out of bounds coordinates: {len(out_of_bounds_coords)}")

    print("\nClass Breakdown:")
    for cid in sorted(class_instance_counts.keys()):
        cname = class_names.get(cid, f"Unknown-{cid}")
        img_count = len(images_with_class[cid])
        print(f"  Class {cid} ({cname:10s}): {class_instance_counts[cid]:5d} instances across {img_count:4d} images")

    # 6. Specific Dent counts
    dent_images = images_with_class[1]
    total_dent_images = len(dent_images)
    total_dent_instances = class_instance_counts[1]
    print(f"\n--- DENT SPECIFIC METRICS ---")
    print(f"Unique images containing at least one dent annotation: {total_dent_images}")
    print(f"Total dent annotations/instances: {total_dent_instances}")

    # Instances per image distribution for Dents
    dents_per_image = Counter()
    for stem in dent_images:
        count = sum(1 for cid, *_ in image_to_annotations[stem] if cid == 1)
        dents_per_image[count] += 1
    print(f"Dent instances per image distribution: {dict(sorted(dents_per_image.items()))}")

    # BBox stats for dents
    if dent_bbox_areas:
        areas = np.array(dent_bbox_areas)
        aspects = np.array(dent_bbox_aspect_ratios)
        print(f"Dent BBox Area (fraction of image): min={areas.min():.4f}, max={areas.max():.4f}, mean={areas.mean():.4f}, median={np.median(areas):.4f}")
        print(f"Dent Aspect Ratio (w/h): min={aspects.min():.2f}, max={aspects.max():.2f}, mean={aspects.mean():.2f}, median={np.median(aspects):.2f}")

    # 7. Multi-defect images analysis
    print("\n--- MULTI-DEFECT ANALYSIS ---")
    multi_class_images = []
    multi_instance_images = []
    dent_co_occurrences = Counter()
    class_co_occurrence_matrix = defaultdict(Counter)

    for stem, annos in image_to_annotations.items():
        if len(annos) > 1:
            multi_instance_images.append(stem)
        distinct_classes = set(cid for cid, *_ in annos)
        if len(distinct_classes) > 1:
            multi_class_images.append(stem)
            for c1 in distinct_classes:
                for c2 in distinct_classes:
                    if c1 != c2:
                        class_co_occurrence_matrix[c1][c2] += 1

        if 1 in distinct_classes:
            for other_cid in distinct_classes:
                if other_cid != 1:
                    dent_co_occurrences[other_cid] += 1

    print(f"Images with multiple defect instances (>1 bounding box): {len(multi_instance_images)} ({len(multi_instance_images)/total_images*100:.2f}%)")
    print(f"Images with multiple defect classes (>1 class): {len(multi_class_images)} ({len(multi_class_images)/total_images*100:.2f}%)")
    print(f"Dent images with multiple classes: {sum(1 for s in dent_images if len(set(c for c, *_ in image_to_annotations[s])) > 1)}")
    print(f"Defect classes co-occurring with Dent:")
    for other_cid, cnt in dent_co_occurrences.items():
        print(f"  With Class {other_cid} ({class_names[other_cid]}): in {cnt} images")

    # 8. Filename prefix vs annotated classes
    prefix_to_classes = defaultdict(Counter)
    for stem, annos in image_to_annotations.items():
        prefix = "".join([c for c in stem if not c.isdigit() and c != "_"])
        for cid, *_ in annos:
            prefix_to_classes[prefix][cid] += 1

    print("\nFilename Prefix vs Annotated Classes:")
    for prefix, clscnt in sorted(prefix_to_classes.items()):
        formatted = ", ".join(f"Class {cid} ({class_names[cid]}): {count}" for cid, count in sorted(clscnt.items()))
        print(f"  Prefix '{prefix}': {formatted}")

    # 9. Representative samples of Dents for inspection
    print("\nRepresentative Dent Samples:")
    dent_sample_stems = sorted(list(dent_images))[:20]
    for s in dent_sample_stems[:10]:
        annos = image_to_annotations[s]
        print(f"  {s}: {len(annos)} annotation(s) -> {annos}")

    return {
        "total_images": total_images,
        "total_labels": total_labels,
        "total_dent_images": total_dent_images,
        "total_dent_instances": total_dent_instances,
        "multi_instance_images": len(multi_instance_images),
        "multi_class_images": len(multi_class_images),
        "empty_labels": len(empty_annotation_files),
        "malformed_lines": len(malformed_lines),
        "out_of_bounds": len(out_of_bounds_coords),
        "duplicate_contents": total_duplicate_images
    }

if __name__ == "__main__":
    run_audit()
