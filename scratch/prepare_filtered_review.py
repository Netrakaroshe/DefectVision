import os
import sys
import json
import csv
import random
import hashlib
from pathlib import Path
from collections import defaultdict, Counter
from PIL import Image, ImageDraw, ImageFont
import numpy as np

sys.stdout.reconfigure(line_buffering=True)

def run_filtered_review_preparation():
    print("=== DENT PILOT STEP 3D: CANDIDATE FILTERING & HUMAN REVIEW PREPARATION ===")
    
    base_dir = Path(r"D:\DefectVision\experiments\dent_classification")
    bg_audit_dir = base_dir / "background_audit"
    filtered_dir = bg_audit_dir / "filtered_review"
    contact_dir = filtered_dir / "contact_sheets"
    manifests_dir = filtered_dir / "manifests"
    
    filtered_dir.mkdir(parents=True, exist_ok=True)
    contact_dir.mkdir(parents=True, exist_ok=True)
    manifests_dir.mkdir(parents=True, exist_ok=True)
    
    # Load original manifest
    orig_manifest_path = bg_audit_dir / "manifests" / "background_candidates_manifest.jsonl"
    records = [json.loads(line) for line in orig_manifest_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    print(f"Total raw background candidates loaded: {len(records)}")
    
    splits_raw = Counter(r["split"] for r in records)
    print(f"Raw counts by split: {dict(splits_raw)}")
    
    # -------------------------------------------------------------
    # 1. Apply Objective Exclusions
    # -------------------------------------------------------------
    # Rule 1: Margin intersection (outer 25px margin)
    # Rule 2: Black strip-edge / unlit conveyor (min pixel < 30)
    # Rule 3: Defect buffer violation (min_defect_dist < 20.0px)
    # Rule 4: Duplicate / near-duplicate candidates
    
    margin = 25
    im_w, im_h = 400, 400
    black_threshold = 30
    
    excluded_margin = []
    excluded_black = []
    excluded_buffer = []
    
    # Pass 1: Coordinate margin check
    after_margin = []
    for r in records:
        x1, y1, x2, y2 = r["patch_pixels"]
        if x1 < margin or y1 < margin or x2 > (im_w - margin) or y2 > (im_h - margin):
            rec_ex = dict(r)
            rec_ex["exclusion_reason"] = "margin_intersection_outer_25px"
            excluded_margin.append(rec_ex)
        else:
            after_margin.append(r)
            
    # Pass 2: Defect buffer check
    after_buffer = []
    for r in after_margin:
        if r["min_defect_dist_px"] < 20.0:
            rec_ex = dict(r)
            rec_ex["exclusion_reason"] = "defect_buffer_violation_lt_20px"
            excluded_buffer.append(rec_ex)
        else:
            after_buffer.append(r)
            
    # Pass 3: Black strip-edge / unlit conveyor check
    after_black = []
    for r in after_buffer:
        cpath = bg_audit_dir / r["crop_path"]
        arr = np.array(Image.open(cpath))
        min_px = int(np.min(arr))
        if min_px < black_threshold:
            rec_ex = dict(r)
            rec_ex["exclusion_reason"] = f"unlit_conveyor_black_edge_min_px_{min_px}"
            excluded_black.append(rec_ex)
        else:
            rec_ex = dict(r)
            rec_ex["min_pixel_intensity"] = min_px
            after_black.append(rec_ex)
            
    # Pass 4: Duplicate / near-duplicate check
    exact_hash_counts = Counter()
    for r in after_black:
        cpath = bg_audit_dir / r["crop_path"]
        h = hashlib.sha256(cpath.read_bytes()).hexdigest()
        exact_hash_counts[h] += 1
        
    exact_duplicates = sum(c - 1 for c in exact_hash_counts.values())
    
    # Near-duplicates check (MSE diff < 1.0)
    by_dims = defaultdict(list)
    for r in after_black:
        by_dims[tuple(r["patch_dimensions"])].append(r)
        
    near_dups_count = 0
    for dims, group in by_dims.items():
        if len(group) > 1:
            arrays = [np.array(Image.open(bg_audit_dir / g["crop_path"])).astype(float) for g in group]
            for i in range(len(group)):
                for j in range(i + 1, len(group)):
                    diff = np.mean(np.abs(arrays[i] - arrays[j]))
                    if diff < 1.0:
                        near_dups_count += 1

    passed_candidates = after_black
    splits_after = Counter(r["split"] for r in passed_candidates)
    
    print("\n--- Objective Filtering Summary ---")
    print(f"Total input candidates: {len(records)}")
    print(f"Excluded by outer 25px margin: {len(excluded_margin)}")
    print(f"Excluded by defect buffer (<20px): {len(excluded_buffer)}")
    print(f"Excluded by unlit conveyor (<30 px): {len(excluded_black)}")
    print(f"Total excluded: {len(excluded_margin) + len(excluded_buffer) + len(excluded_black)}")
    print(f"Exact duplicates found: {exact_duplicates}")
    print(f"Near duplicates found (mean abs diff < 1.0): {near_dups_count}")
    print(f"Total passed objective filters: {len(passed_candidates)}")
    print(f"Passed breakdown by split: {dict(splits_after)}")

    # Save passed candidates manifest
    passed_jsonl = manifests_dir / "filtered_background_candidates.jsonl"
    with open(passed_jsonl, "w", encoding="utf-8") as f:
        for r in passed_candidates:
            f.write(json.dumps(r) + "\n")
    print(f"\nSaved passed candidates JSONL: {passed_jsonl}")

    # -------------------------------------------------------------
    # 2. Sample Review Set: 100 per split (or all if < 100)
    # -------------------------------------------------------------
    seed = 20261020
    rng = random.Random(seed)
    
    candidates_by_split = defaultdict(list)
    for r in passed_candidates:
        candidates_by_split[r["split"]].append(r)
        
    review_set_by_split = {}
    review_set_all = []
    
    for sp in ["train", "val", "test"]:
        pool = list(candidates_by_split[sp])
        pool.sort(key=lambda r: r["candidate_id"])
        if len(pool) <= 100:
            sampled = pool  # all remaining if <= 100
            print(f"Split '{sp}': {len(sampled)} available <= 100; selected ALL {len(sampled)} candidates.")
        else:
            rng.shuffle(pool)
            sampled = pool[:100]
            sampled.sort(key=lambda r: r["candidate_id"])
            print(f"Split '{sp}': {len(pool)} available > 100; randomly sampled 100 candidates (seed={seed}).")
            
        review_set_by_split[sp] = sampled
        review_set_all.extend(sampled)

    print(f"Total review set size: {len(review_set_all)} candidates ({len(review_set_by_split['train'])} train, {len(review_set_by_split['val'])} val, {len(review_set_by_split['test'])} test)")

    # Save review set manifest
    review_manifest_jsonl = manifests_dir / "filtered_review_sample_manifest.jsonl"
    with open(review_manifest_jsonl, "w", encoding="utf-8") as f:
        for r in review_set_all:
            f.write(json.dumps(r) + "\n")

    # -------------------------------------------------------------
    # 3. Create Blank Decision CSV
    # -------------------------------------------------------------
    # Columns: candidate_id, split, reviewer_decision, reviewer_notes
    # decisions left strictly BLANK
    decision_csv_path = filtered_dir / "blank_reviewer_decisions.csv"
    csv_fields = [
        "candidate_id", "split", "reviewer_decision", "reviewer_notes",
        "source_image_stem", "patch_x1", "patch_y1", "patch_x2", "patch_y2",
        "patch_width", "patch_height", "aspect_ratio", "min_defect_dist_px",
        "crop_path"
    ]
    with open(decision_csv_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=csv_fields)
        writer.writeheader()
        for r in review_set_all:
            writer.writerow({
                "candidate_id": r["candidate_id"],
                "split": r["split"],
                "reviewer_decision": "",  # STRICTLY BLANK
                "reviewer_notes": "",     # STRICTLY BLANK
                "source_image_stem": r["source_image_stem"],
                "patch_x1": r["patch_pixels"][0],
                "patch_y1": r["patch_pixels"][1],
                "patch_x2": r["patch_pixels"][2],
                "patch_y2": r["patch_pixels"][3],
                "patch_width": r["patch_dimensions"][0],
                "patch_height": r["patch_dimensions"][1],
                "aspect_ratio": r["aspect_ratio"],
                "min_defect_dist_px": r["min_defect_dist_px"],
                "crop_path": r["crop_path"]
            })
    print(f"Saved blank decision CSV template: {decision_csv_path}")

    # -------------------------------------------------------------
    # 4. Render Contact Sheets for Train, Val, and Test
    # -------------------------------------------------------------
    cell_w, cell_h = 140, 175  # 140x140 image + 35 text
    padding = 8
    header_h = 55
    
    for sp in ["train", "val", "test"]:
        sp_items = review_set_by_split[sp]
        n_items = len(sp_items)
        grid_cols = 10
        grid_rows = int(np.ceil(n_items / grid_cols))
        
        sheet_w = grid_cols * cell_w + (grid_cols + 1) * padding
        sheet_h = grid_rows * cell_h + (grid_rows + 1) * padding + header_h
        
        sheet = Image.new("RGB", (sheet_w, sheet_h), color=(22, 24, 27))
        draw = ImageDraw.Draw(sheet)
        
        # Header
        draw.text((padding, 12), f"CR7-DET Filtered Negative Review Set — [{sp.upper()}] Partition ({n_items} Candidates)", fill=(245, 245, 245))
        draw.text((padding, 32), f"Filtered: Margin >= 25px, Min Pixel >= 30, Buffer >= 20px | Seed={seed} | For Manual Human Triage", fill=(175, 180, 185))
        
        for idx, r in enumerate(sp_items):
            row = idx // grid_cols
            col = idx % grid_cols
            x_cell = padding + col * (cell_w + padding)
            y_cell = header_h + padding + row * (cell_h + padding)
            
            crop_abs = bg_audit_dir / r["crop_path"]
            with Image.open(crop_abs) as c_im:
                cw, ch = c_im.size
                scale = min((cell_w - 6) / cw, (cell_w - 6) / ch)
                nw = max(1, int(round(cw * scale)))
                nh = max(1, int(round(ch * scale)))
                c_res = c_im.resize((nw, nh), Image.Resampling.LANCZOS)
                
            tx = x_cell + (cell_w - nw) // 2
            ty = y_cell + (cell_w - nh) // 2
            
            draw.rectangle([x_cell, y_cell, x_cell + cell_w, y_cell + cell_w], fill=(12, 13, 15), outline=(50, 55, 62), width=1)
            
            if c_res.mode != "RGB":
                c_res = c_res.convert("RGB")
            sheet.paste(c_res, (tx, ty))
            
            # Caption
            px1, py1, px2, py2 = r["patch_pixels"]
            draw.text((x_cell + 2, y_cell + cell_w + 3), f"{r['source_image_stem']}", fill=(225, 225, 225))
            draw.text((x_cell + 2, y_cell + cell_w + 14), f"[{px1},{py1},{px2},{py2}]", fill=(145, 150, 155))
            draw.text((x_cell + 2, y_cell + cell_w + 24), f"{r['patch_dimensions'][0]}x{r['patch_dimensions'][1]}", fill=(170, 175, 180))
            draw.text((x_cell + cell_w - 45, y_cell + cell_w + 24), f"d>={r['min_defect_dist_px']}p", fill=(100, 190, 240))
            
        sheet_path = contact_dir / f"filtered_review_{sp}_{n_items}.png"
        sheet.save(sheet_path, format="PNG")
        print(f"Saved {sp} contact sheet ({n_items} items): {sheet_path}")

    return {
        "raw_counts": dict(splits_raw),
        "after_counts": dict(splits_after),
        "excluded_margin": len(excluded_margin),
        "excluded_buffer": len(excluded_buffer),
        "excluded_black": len(excluded_black),
        "exact_duplicates": exact_duplicates,
        "near_duplicates": near_dups_count,
        "review_counts": {k: len(v) for k, v in review_set_by_split.items()}
    }

if __name__ == "__main__":
    run_filtered_review_preparation()
