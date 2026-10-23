import os
import sys
import json
import csv
import random
from pathlib import Path
from collections import defaultdict, Counter
from PIL import Image, ImageDraw, ImageFont
import numpy as np
import cv2

sys.stdout.reconfigure(line_buffering=True)

def generate_manual_review():
    print("=== DENT PILOT STEP 3C: MANUAL NEGATIVE REVIEW PREPARATION ===")
    
    base_dir = Path(r"D:\DefectVision\experiments\dent_classification")
    bg_audit_dir = base_dir / "background_audit"
    manifest_path = bg_audit_dir / "manifests" / "background_candidates_manifest.jsonl"
    review_dir = bg_audit_dir / "manual_review"
    contact_dir = review_dir / "contact_sheets"
    
    review_dir.mkdir(parents=True, exist_ok=True)
    contact_dir.mkdir(parents=True, exist_ok=True)
    
    records = [json.loads(line) for line in manifest_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    print(f"Total background candidates: {len(records)}")
    
    # Deterministic sampling: 100 per split
    seed = 20261019
    rng = random.Random(seed)
    
    records_by_split = defaultdict(list)
    for r in records:
        records_by_split[r["split"]].append(r)
        
    sampled_records_by_split = {}
    all_sampled = []
    
    for sp in ["train", "val", "test"]:
        sp_recs = list(records_by_split[sp])
        # Sort by candidate_id before shuffling for absolute determinism
        sp_recs.sort(key=lambda r: r["candidate_id"])
        rng.shuffle(sp_recs)
        sampled = sp_recs[:100]
        # Re-sort sampled by candidate_id for neat display
        sampled.sort(key=lambda r: r["candidate_id"])
        sampled_records_by_split[sp] = sampled
        all_sampled.extend(sampled)
        print(f"Sampled {len(sampled)} candidates from '{sp}' (out of {len(sp_recs)} available).")

    # Analyze each sampled patch for morphological suspicion
    flagged_counts = Counter()
    flag_reasons_by_split = defaultdict(Counter)
    
    enriched_sampled = []
    
    for r in all_sampled:
        crop_path = bg_audit_dir / r["crop_path"]
        with Image.open(crop_path) as im:
            arr = np.array(im)
            
        # If 3 channels, check channel 0
        gray = arr[:, :, 0] if len(arr.shape) == 3 else arr
        
        # Statistical & Edge Metrics
        mean_val = float(np.mean(gray))
        std_val = float(np.std(gray))
        min_val = int(np.min(gray))
        max_val = int(np.max(gray))
        contrast = max_val - min_val
        
        # Laplacian edge variance
        laplacian = cv2.Laplacian(gray, cv2.CV_64F)
        lap_var = float(np.var(laplacian))
        
        # Sobel directional gradients
        sobelx = np.abs(cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3))
        sobely = np.abs(cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3))
        mean_gx = float(np.mean(sobelx))
        mean_gy = float(np.mean(sobely))
        dir_ratio = (mean_gx + 1e-4) / (mean_gy + 1e-4)
        
        # Dark speck / pit anomaly check: pixels < mean - 3.5 * std
        dark_threshold = mean_val - 3.5 * max(std_val, 5.0)
        dark_specks = int(np.sum(gray < dark_threshold))
        
        # Flags evaluation
        flags = []
        if dark_specks >= 4:
            flags.append("dark_speck_anomaly")
        if contrast >= 65:
            flags.append("high_local_contrast")
        if lap_var >= 180.0:
            flags.append("elevated_edge_relief")
        if (dir_ratio > 3.0 or dir_ratio < 0.33) and std_val > 9.0:
            flags.append("directional_striation")
        if r["min_defect_dist_px"] <= 22.0 and std_val > 8.0:
            flags.append("border_buffer_proximity")
            
        is_flagged = len(flags) > 0
        primary_reason = "; ".join(flags) if is_flagged else "none"
        
        sp = r["split"]
        if is_flagged:
            flagged_counts[sp] += 1
            for f in flags:
                flag_reasons_by_split[sp][f] += 1
                
        rec_copy = dict(r)
        rec_copy.update({
            "mean_intensity": round(mean_val, 1),
            "std_intensity": round(std_val, 1),
            "contrast": contrast,
            "laplacian_var": round(lap_var, 1),
            "dir_ratio": round(dir_ratio, 2),
            "dark_speck_count": dark_specks,
            "preliminary_flag": "DEFECT_SUSPECTED" if is_flagged else "CLEAN_CANDIDATE",
            "flag_reason": primary_reason,
            "decision": "defect_suspected" if is_flagged else "clean",
            "reviewer_notes": ""
        })
        enriched_sampled.append(rec_copy)

    print("\n--- Review Flags Summary ---")
    for sp in ["train", "val", "test"]:
        cnt = flagged_counts[sp]
        print(f"Split {sp:5s}: {cnt} / 100 flagged ({cnt}%)")
        for reason, c in flag_reasons_by_split[sp].items():
            print(f"    - {reason}: {c}")

    # Save sampled manifest (JSONL and CSV)
    jsonl_path = review_dir / "manual_review_sampled_candidates.jsonl"
    with open(jsonl_path, "w", encoding="utf-8") as f:
        for r in enriched_sampled:
            f.write(json.dumps(r) + "\n")
    print(f"\nSaved sampled JSONL manifest: {jsonl_path}")
    
    csv_path = review_dir / "manual_review_decisions.csv"
    csv_fields = [
        "candidate_id", "split", "source_image_stem",
        "patch_x1", "patch_y1", "patch_x2", "patch_y2",
        "patch_width", "patch_height", "aspect_ratio",
        "min_defect_dist_px", "preliminary_flag", "flag_reason",
        "mean_intensity", "contrast", "laplacian_var",
        "decision", "reviewer_notes", "crop_path"
    ]
    with open(csv_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=csv_fields)
        writer.writeheader()
        for r in enriched_sampled:
            writer.writerow({
                "candidate_id": r["candidate_id"],
                "split": r["split"],
                "source_image_stem": r["source_image_stem"],
                "patch_x1": r["patch_pixels"][0],
                "patch_y1": r["patch_pixels"][1],
                "patch_x2": r["patch_pixels"][2],
                "patch_y2": r["patch_pixels"][3],
                "patch_width": r["patch_dimensions"][0],
                "patch_height": r["patch_dimensions"][1],
                "aspect_ratio": r["aspect_ratio"],
                "min_defect_dist_px": r["min_defect_dist_px"],
                "preliminary_flag": r["preliminary_flag"],
                "flag_reason": r["flag_reason"],
                "mean_intensity": r["mean_intensity"],
                "contrast": r["contrast"],
                "laplacian_var": r["laplacian_var"],
                "decision": r["decision"],
                "reviewer_notes": r["reviewer_notes"],
                "crop_path": r["crop_path"]
            })
    print(f"Saved review decision template CSV: {csv_path}")

    # Render 10x10 Contact Sheets for each split
    grid_cols, grid_rows = 10, 10
    cell_w, cell_h = 140, 175  # 140x140 image + 35 text
    padding = 8
    header_h = 55
    sheet_w = grid_cols * cell_w + (grid_cols + 1) * padding
    sheet_h = grid_rows * cell_h + (grid_rows + 1) * padding + header_h
    
    for sp in ["train", "val", "test"]:
        sp_items = [r for r in enriched_sampled if r["split"] == sp]
        assert len(sp_items) == 100, f"Expected 100 items in {sp}, got {len(sp_items)}"
        
        sheet = Image.new("RGB", (sheet_w, sheet_h), color=(24, 26, 29))
        draw = ImageDraw.Draw(sheet)
        
        # Header text
        draw.text((padding, 12), f"CR7-DET Background Candidate Manual Review: [{sp.upper()}] Partition (100 Sampled Candidates)", fill=(245, 245, 245))
        draw.text((padding, 32), f"Seed={seed} | Yellow Outline = Suspicious Feature Flagged ({flagged_counts[sp]} flagged) | Cyan = Clean Candidate", fill=(180, 180, 180))
        
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
            
            # Border highlight: Yellow for flagged, Dark Cyan/Grey for clean
            is_flg = r["preliminary_flag"] == "DEFECT_SUSPECTED"
            border_color = (255, 200, 40) if is_flg else (45, 55, 65)
            border_width = 2 if is_flg else 1
            
            draw.rectangle([x_cell, y_cell, x_cell + cell_w, y_cell + cell_w], fill=(14, 15, 17), outline=border_color, width=border_width)
            
            if c_res.mode != "RGB":
                c_res = c_res.convert("RGB")
            sheet.paste(c_res, (tx, ty))
            
            # Caption
            # Line 1: stem & patch coords
            px1, py1, px2, py2 = r["patch_pixels"]
            draw.text((x_cell + 2, y_cell + cell_w + 3), f"{r['source_image_stem']}", fill=(220, 220, 220))
            draw.text((x_cell + 2, y_cell + cell_w + 14), f"[{px1},{py1},{px2},{py2}]", fill=(140, 140, 140))
            
            # Line 2: dimensions & flag
            status_text = "FLAG" if is_flg else "OK"
            status_color = (255, 210, 50) if is_flg else (100, 220, 160)
            draw.text((x_cell + 2, y_cell + cell_w + 24), f"{r['patch_dimensions'][0]}x{r['patch_dimensions'][1]}", fill=(160, 160, 160))
            draw.text((x_cell + cell_w - 32, y_cell + cell_w + 24), status_text, fill=status_color)
            
        sheet_path = contact_dir / f"manual_review_{sp}_100.png"
        sheet.save(sheet_path, format="PNG")
        print(f"Saved {sp} 100-sample contact sheet: {sheet_path}")

    return {
        "total_sampled": len(all_sampled),
        "flagged_counts": dict(flagged_counts),
        "flag_reasons": {k: dict(v) for k, v in flag_reasons_by_split.items()}
    }

if __name__ == "__main__":
    generate_manual_review()
