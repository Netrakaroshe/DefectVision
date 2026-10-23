import os
import sys
import json
import csv
import hashlib
import random
from pathlib import Path
from collections import defaultdict, Counter
from PIL import Image, ImageDraw, ImageFont
import numpy as np

def prepare_dent_pilot():
    print("=== CR7-DET DENT-ONLY PILOT DATASET PREPARATION ===")
    
    # 1. Paths
    source_root = Path(r"C:\Users\susha\Downloads\CR7-DET\CR7-DET")
    img_dir = source_root / "image"
    lbl_dir = source_root / "label"
    
    output_root = Path(r"D:\DefectVision\experiments\dent_classification")
    crops_root = output_root / "crops"
    manifests_root = output_root / "manifests"
    contact_root = output_root / "contact_sheets"
    
    for split in ["train", "val", "test"]:
        (crops_root / split).mkdir(parents=True, exist_ok=True)
    manifests_root.mkdir(parents=True, exist_ok=True)
    contact_root.mkdir(parents=True, exist_ok=True)
    
    # 2. Collect Dent Source Files
    dent_images = sorted(list(img_dir.glob("dents*.jpg")))
    dent_labels = sorted(list(lbl_dir.glob("dents*.txt")))
    print(f"Source Dent images found: {len(dent_images)}")
    print(f"Source Dent labels found: {len(dent_labels)}")
    
    # Verify pairing
    img_stems = {p.stem: p for p in dent_images}
    lbl_stems = {p.stem: p for p in dent_labels}
    assert set(img_stems.keys()) == set(lbl_stems.keys()), "Mismatch between Dent images and labels!"
    
    # 3. Compute SHA-256 Hashes and Group Duplicates
    print("Computing SHA-256 image hashes...")
    hash_to_images = defaultdict(list)
    image_to_hash = {}
    
    for p in dent_images:
        h = hashlib.sha256(p.read_bytes()).hexdigest()
        hash_to_images[h].append(p)
        image_to_hash[p.stem] = h
        
    unique_hashes = sorted(list(hash_to_images.keys()))
    duplicate_groups = {h: imgs for h, imgs in hash_to_images.items() if len(imgs) > 1}
    print(f"Total source images: {len(dent_images)}")
    print(f"Unique image hashes: {len(unique_hashes)}")
    print(f"Duplicate hash groups: {len(duplicate_groups)} (comprising {sum(len(v) for v in duplicate_groups.values())} images)")
    
    # 4. Deterministic Hash-Grouped Split (approx 70/15/15)
    rng = random.Random(42)
    shuffled_hashes = list(unique_hashes)
    rng.shuffle(shuffled_hashes)
    
    n_total = len(shuffled_hashes)
    n_train = int(round(n_total * 0.70))
    n_val = int(round(n_total * 0.15))
    n_test = n_total - n_train - n_val
    
    train_hashes = set(shuffled_hashes[:n_train])
    val_hashes = set(shuffled_hashes[n_train:n_train + n_val])
    test_hashes = set(shuffled_hashes[n_train + n_val:])
    
    hash_to_split = {}
    for h in train_hashes:
        hash_to_split[h] = "train"
    for h in val_hashes:
        hash_to_split[h] = "val"
    for h in test_hashes:
        hash_to_split[h] = "test"
        
    print(f"\nHash allocation: train={len(train_hashes)} ({len(train_hashes)/n_total*100:.1f}%), val={len(val_hashes)} ({len(val_hashes)/n_total*100:.1f}%), test={len(test_hashes)} ({len(test_hashes)/n_total*100:.1f}%)")
    
    # 5. Extract Crops and Validate Annotations
    manifest_records = []
    crop_counter = 0
    accepted_crops = 0
    rejected_crops = 0
    invalid_annotations = 0
    
    # Track statistics
    crops_per_split = Counter()
    images_per_split = Counter()
    
    # Track diverse crops for contact sheet
    candidate_crops_for_contact = []
    
    padding_ratio = 0.10  # 10% padding
    
    for stem in sorted(img_stems.keys()):
        img_path = img_stems[stem]
        lbl_path = lbl_stems[stem]
        h = image_to_hash[stem]
        split = hash_to_split[h]
        images_per_split[split] += 1
        
        with Image.open(img_path) as im:
            im_w, im_h = im.size
            im_mode = im.mode
            im_arr = np.array(im)
            
        lines = [l.strip() for l in lbl_path.read_text(encoding="utf-8").splitlines() if l.strip()]
        
        for box_idx, line in enumerate(lines):
            crop_counter += 1
            crop_id = f"dent_{stem}_box{box_idx}_{crop_counter:04d}"
            
            parts = line.split()
            if len(parts) != 5:
                invalid_annotations += 1
                rejected_crops += 1
                manifest_records.append({
                    "crop_id": crop_id,
                    "source_image_path": str(img_path).replace("\\", "/"),
                    "source_image_stem": stem,
                    "sha256": h,
                    "original_label_id": -1,
                    "original_label_name": "UNKNOWN",
                    "box_index": box_idx,
                    "raw_annotation": line,
                    "validation_status": "REJECTED_WRONG_TOKEN_COUNT",
                    "split": split,
                    "crop_path": None
                })
                continue
                
            try:
                cid = int(parts[0])
                xc, yc, w, h_box = float(parts[1]), float(parts[2]), float(parts[3]), float(parts[4])
            except ValueError as e:
                invalid_annotations += 1
                rejected_crops += 1
                manifest_records.append({
                    "crop_id": crop_id,
                    "source_image_path": str(img_path).replace("\\", "/"),
                    "source_image_stem": stem,
                    "sha256": h,
                    "original_label_id": -1,
                    "original_label_name": "UNKNOWN",
                    "box_index": box_idx,
                    "raw_annotation": line,
                    "validation_status": f"REJECTED_NON_NUMERIC: {e}",
                    "split": split,
                    "crop_path": None
                })
                continue
                
            if cid != 1:
                # Requirement: Keep other CR7-DET classes out of Dent-only crop set
                invalid_annotations += 1
                rejected_crops += 1
                manifest_records.append({
                    "crop_id": crop_id,
                    "source_image_path": str(img_path).replace("\\", "/"),
                    "source_image_stem": stem,
                    "sha256": h,
                    "original_label_id": cid,
                    "original_label_name": "NON_DENT_CLASS",
                    "box_index": box_idx,
                    "raw_annotation": line,
                    "validation_status": "REJECTED_NON_DENT_CLASS",
                    "split": split,
                    "crop_path": None
                })
                continue
                
            # Validate unpadded bbox coordinates
            # Tolerance for float rounding: 1e-4 fraction of image (~0.04 pixel)
            tol = 1e-4
            x1_norm = xc - w / 2
            y1_norm = yc - h_box / 2
            x2_norm = xc + w / 2
            y2_norm = yc + h_box / 2
            
            if w <= 0 or h_box <= 0:
                invalid_annotations += 1
                rejected_crops += 1
                manifest_records.append({
                    "crop_id": crop_id,
                    "source_image_path": str(img_path).replace("\\", "/"),
                    "source_image_stem": stem,
                    "sha256": h,
                    "original_label_id": 1,
                    "original_label_name": "Dents",
                    "box_index": box_idx,
                    "raw_annotation": line,
                    "validation_status": "REJECTED_NON_POSITIVE_DIMENSIONS",
                    "split": split,
                    "crop_path": None
                })
                continue
                
            if x1_norm < -tol or y1_norm < -tol or x2_norm > 1.0 + tol or y2_norm > 1.0 + tol:
                invalid_annotations += 1
                rejected_crops += 1
                manifest_records.append({
                    "crop_id": crop_id,
                    "source_image_path": str(img_path).replace("\\", "/"),
                    "source_image_stem": stem,
                    "sha256": h,
                    "original_label_id": 1,
                    "original_label_name": "Dents",
                    "box_index": box_idx,
                    "raw_annotation": line,
                    "validation_status": "REJECTED_OUT_OF_BOUNDS",
                    "split": split,
                    "crop_path": None
                })
                continue
                
            # Compute unpadded pixel coordinates (clamped to image exact boundary)
            raw_px_x1 = max(0.0, min(float(im_w), (xc - w/2) * im_w))
            raw_px_y1 = max(0.0, min(float(im_h), (yc - h_box/2) * im_h))
            raw_px_x2 = max(0.0, min(float(im_w), (xc + w/2) * im_w))
            raw_px_y2 = max(0.0, min(float(im_h), (yc + h_box/2) * im_h))
            
            # Apply modest padding (10% of width and height, minimum 2px)
            box_px_w = raw_px_x2 - raw_px_x1
            box_px_h = raw_px_y2 - raw_px_y1
            pad_w = max(2.0, padding_ratio * box_px_w)
            pad_h = max(2.0, padding_ratio * box_px_h)
            
            crop_x1 = int(max(0, np.floor(raw_px_x1 - pad_w)))
            crop_y1 = int(max(0, np.floor(raw_px_y1 - pad_h)))
            crop_x2 = int(min(im_w, np.ceil(raw_px_x2 + pad_w)))
            crop_y2 = int(min(im_h, np.ceil(raw_px_y2 + pad_h)))
            
            crop_w = crop_x2 - crop_x1
            crop_h = crop_y2 - crop_y1
            
            if crop_w < 5 or crop_h < 5:
                invalid_annotations += 1
                rejected_crops += 1
                manifest_records.append({
                    "crop_id": crop_id,
                    "source_image_path": str(img_path).replace("\\", "/"),
                    "source_image_stem": stem,
                    "sha256": h,
                    "original_label_id": 1,
                    "original_label_name": "Dents",
                    "box_index": box_idx,
                    "raw_annotation": line,
                    "validation_status": "REJECTED_CROP_TOO_SMALL",
                    "split": split,
                    "crop_path": None
                })
                continue
                
            # Extract crop
            crop_arr = im_arr[crop_y1:crop_y2, crop_x1:crop_x2]
            crop_im = Image.fromarray(crop_arr)
            
            crop_rel_path = f"crops/{split}/{crop_id}.png"
            crop_abs_path = crops_root / split / f"{crop_id}.png"
            crop_im.save(crop_abs_path, format="PNG")
            
            accepted_crops += 1
            crops_per_split[split] += 1
            
            record = {
                "crop_id": crop_id,
                "source_image_path": str(img_path).replace("\\", "/"),
                "source_image_stem": stem,
                "sha256": h,
                "original_label_id": 1,
                "original_label_name": "Dents",
                "box_index": box_idx,
                "bbox_norm": [round(xc, 6), round(yc, 6), round(w, 6), round(h_box, 6)],
                "bbox_pixels": [round(raw_px_x1, 1), round(raw_px_y1, 1), round(raw_px_x2, 1), round(raw_px_y2, 1)],
                "crop_pixels": [crop_x1, crop_y1, crop_x2, crop_y2],
                "crop_dimensions": [crop_w, crop_h],
                "aspect_ratio": round(crop_w / crop_h, 3),
                "padding_ratio": padding_ratio,
                "split": split,
                "validation_status": "ACCEPTED",
                "crop_path": crop_rel_path.replace("\\", "/"),
                "crop_abs_path": str(crop_abs_path).replace("\\", "/"),
                "provenance": "CR7-DET cold-rolled steel surface defect dataset; class 1 (Dents)"
            }
            manifest_records.append(record)
            
            candidate_crops_for_contact.append(record)

    print(f"\nProcessing Summary:")
    print(f"Total annotations processed: {crop_counter}")
    print(f"Accepted crops: {accepted_crops}")
    print(f"Rejected crops: {rejected_crops}")
    print(f"Invalid annotations: {invalid_annotations}")
    print(f"Crops per split: train={crops_per_split['train']}, val={crops_per_split['val']}, test={crops_per_split['test']}")
    print(f"Images per split: train={images_per_split['train']}, val={images_per_split['val']}, test={images_per_split['test']}")

    # 6. Save Manifests (JSONL and CSV)
    jsonl_path = manifests_root / "dent_pilot_manifest.jsonl"
    csv_path = manifests_root / "dent_pilot_manifest.csv"
    
    with open(jsonl_path, "w", encoding="utf-8") as f:
        for rec in manifest_records:
            f.write(json.dumps(rec) + "\n")
    print(f"\nSaved JSONL manifest: {jsonl_path} ({len(manifest_records)} records)")
    
    # Save CSV for rapid inspection
    csv_fields = [
        "crop_id", "source_image_stem", "split", "validation_status",
        "original_label_id", "original_label_name", "box_index",
        "crop_width", "crop_height", "aspect_ratio",
        "crop_path", "sha256", "source_image_path"
    ]
    with open(csv_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=csv_fields)
        writer.writeheader()
        for rec in manifest_records:
            row = {
                "crop_id": rec["crop_id"],
                "source_image_stem": rec["source_image_stem"],
                "split": rec["split"],
                "validation_status": rec["validation_status"],
                "original_label_id": rec["original_label_id"],
                "original_label_name": rec["original_label_name"],
                "box_index": rec["box_index"],
                "crop_width": rec.get("crop_dimensions", [0, 0])[0],
                "crop_height": rec.get("crop_dimensions", [0, 0])[1],
                "aspect_ratio": rec.get("aspect_ratio", 0.0),
                "crop_path": rec.get("crop_path", ""),
                "sha256": rec["sha256"],
                "source_image_path": rec["source_image_path"]
            }
            writer.writerow(row)
    print(f"Saved CSV manifest: {csv_path}")

    # 7. Generate Contact Sheet
    # Pick 16 diverse samples: 
    # - 2 small area, 2 large area
    # - 2 tall aspect ratio (w/h < 0.6), 2 wide aspect ratio (w/h > 1.8)
    # - 2 from train, 2 from val, 2 from test
    # - 2 from duplicate pairs (e.g. dents199 and dents202)
    contact_candidates = []
    
    # Duplicate pair samples
    dup_recs = [r for r in candidate_crops_for_contact if r["source_image_stem"] in ["dents199", "dents202"]]
    contact_candidates.extend(dup_recs[:2])
    
    # Sort by area
    sorted_by_area = sorted(candidate_crops_for_contact, key=lambda r: r["crop_dimensions"][0] * r["crop_dimensions"][1])
    contact_candidates.extend(sorted_by_area[:2])   # smallest
    contact_candidates.extend(sorted_by_area[-2:])  # largest
    
    # Sort by aspect ratio
    sorted_by_aspect = sorted(candidate_crops_for_contact, key=lambda r: r["aspect_ratio"])
    contact_candidates.extend(sorted_by_aspect[:2])  # tall
    contact_candidates.extend(sorted_by_aspect[-2:]) # wide
    
    # Split diversity
    train_samples = [r for r in candidate_crops_for_contact if r["split"] == "train" and r not in contact_candidates]
    val_samples = [r for r in candidate_crops_for_contact if r["split"] == "val" and r not in contact_candidates]
    test_samples = [r for r in candidate_crops_for_contact if r["split"] == "test" and r not in contact_candidates]
    
    contact_candidates.extend(train_samples[:2])
    contact_candidates.extend(val_samples[:2])
    contact_candidates.extend(test_samples[:2])
    
    # Ensure exactly 16 unique samples
    unique_contact_recs = []
    seen_ids = set()
    for r in contact_candidates:
        if r["crop_id"] not in seen_ids:
            seen_ids.add(r["crop_id"])
            unique_contact_recs.append(r)
            
    while len(unique_contact_recs) < 16:
        for r in candidate_crops_for_contact:
            if r["crop_id"] not in seen_ids:
                seen_ids.add(r["crop_id"])
                unique_contact_recs.append(r)
                if len(unique_contact_recs) == 16:
                    break
                    
    selected_16 = unique_contact_recs[:16]
    print(f"\nSelected {len(selected_16)} diverse candidate crops for review contact sheet.")
    
    # Build 4x4 contact sheet grid
    grid_cols = 4
    grid_rows = 4
    cell_w = 200
    cell_h = 240  # 200 for image, 40 for caption
    padding = 10
    
    sheet_w = grid_cols * cell_w + (grid_cols + 1) * padding
    sheet_h = grid_rows * cell_h + (grid_rows + 1) * padding + 50 # header
    
    sheet = Image.new("RGB", (sheet_w, sheet_h), color=(30, 32, 36))
    draw = ImageDraw.Draw(sheet)
    
    # Draw Header
    draw.text((padding, 15), "DefectVision: CR7-DET Dent Pilot Candidate Crops (Diverse 16-Sample Contact Sheet)", fill=(240, 240, 240))
    draw.text((padding, 32), "Review Only: Visual verification required prior to semantic acceptance. Not for production use.", fill=(180, 180, 180))
    
    for idx, rec in enumerate(selected_16):
        row = idx // grid_cols
        col = idx % grid_cols
        
        x_cell = padding + col * (cell_w + padding)
        y_cell = 50 + padding + row * (cell_h + padding)
        
        # Load and fit crop into (cell_w - 10, cell_w - 10)
        crop_path = output_root / rec["crop_path"]
        with Image.open(crop_path) as c_im:
            c_w, c_h = c_im.size
            scale = min((cell_w - 10) / c_w, (cell_w - 10) / c_h)
            new_w = max(1, int(round(c_w * scale)))
            new_h = max(1, int(round(c_h * scale)))
            c_resized = c_im.resize((new_w, new_h), Image.Resampling.LANCZOS)
            
        # Center in square tile area
        tile_x = x_cell + (cell_w - new_w) // 2
        tile_y = y_cell + (cell_w - new_h) // 2
        
        # Border box around tile
        draw.rectangle([x_cell, y_cell, x_cell + cell_w, y_cell + cell_w], fill=(18, 19, 21), outline=(60, 64, 72), width=1)
        
        if c_resized.mode != "RGB":
            c_resized = c_resized.convert("RGB")
        sheet.paste(c_resized, (tile_x, tile_y))
        
        # Split tag color
        sp_color = (80, 180, 255) if rec["split"] == "train" else (255, 180, 60) if rec["split"] == "val" else (100, 240, 140)
        
        # Caption
        stem_str = rec["source_image_stem"]
        dim_str = f"{rec['crop_dimensions'][0]}x{rec['crop_dimensions'][1]} (ar:{rec['aspect_ratio']})"
        split_str = f"[{rec['split'].upper()}]"
        
        draw.text((x_cell + 4, y_cell + cell_w + 4), f"{stem_str} b{rec['box_index']}", fill=(220, 220, 220))
        draw.text((x_cell + 4, y_cell + cell_w + 18), dim_str, fill=(160, 160, 160))
        draw.text((x_cell + cell_w - 60, y_cell + cell_w + 4), split_str, fill=sp_color)

    contact_path = contact_root / "dent_crops_contact_sheet.png"
    sheet.save(contact_path, format="PNG")
    print(f"Saved contact sheet: {contact_path}")
    
    return {
        "source_images": len(dent_images),
        "unique_hashes": len(unique_hashes),
        "duplicate_groups": len(duplicate_groups),
        "total_crops": crop_counter,
        "accepted_crops": accepted_crops,
        "rejected_crops": rejected_crops,
        "invalid_annotations": invalid_annotations,
        "split_images": dict(images_per_split),
        "split_crops": dict(crops_per_split),
        "jsonl_path": str(jsonl_path),
        "csv_path": str(csv_path),
        "contact_path": str(contact_path)
    }

if __name__ == "__main__":
    prepare_dent_pilot()
