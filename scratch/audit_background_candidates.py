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

sys.stdout.reconfigure(line_buffering=True)

def run_background_audit():
    print("=== DEFECTVISION: DENT PILOT STEP 3A - BACKGROUND CANDIDATE AUDIT ===")
    
    # 1. Directories setup
    bg_audit_dir = Path(r"D:\DefectVision\experiments\dent_classification\background_audit")
    crops_dir = bg_audit_dir / "crops"
    manifests_dir = bg_audit_dir / "manifests"
    contact_dir = bg_audit_dir / "contact_sheets"
    
    for split in ["train", "val", "test"]:
        (crops_dir / split).mkdir(parents=True, exist_ok=True)
    manifests_dir.mkdir(parents=True, exist_ok=True)
    contact_dir.mkdir(parents=True, exist_ok=True)
    
    # 2. Load existing Dent manifest
    dent_manifest_path = Path(r"D:\DefectVision\experiments\dent_classification\manifests\dent_pilot_manifest.jsonl")
    dent_records = [json.loads(line) for line in dent_manifest_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    print(f"Loaded {len(dent_records)} Dent pilot records.")
    
    # Verify split assignments and hash mapping
    image_to_split = {}
    image_to_hash = {}
    hash_to_split = {}
    split_dent_counts = Counter()
    
    for r in dent_records:
        stem = r["source_image_stem"]
        sp = r["split"]
        h = r["sha256"]
        image_to_split[stem] = sp
        image_to_hash[stem] = h
        hash_to_split[h] = sp
        split_dent_counts[sp] += 1
        
    print(f"Dent crop counts per split: {dict(split_dent_counts)}")
    print(f"Total source images: {len(image_to_split)}, unique hashes: {len(hash_to_split)}")
    
    # 3. Source directory and labels
    source_root = Path(r"C:\Users\susha\Downloads\CR7-DET\CR7-DET")
    img_dir = source_root / "image"
    lbl_dir = source_root / "label"
    im_w, im_h = 400, 400
    safety_buffer = 20  # 20 pixels
    
    # Compute defect masks with safety buffer and raw defect boxes
    raw_defect_boxes = {}
    buffered_defect_masks = {}
    
    for stem in image_to_split:
        lbl_file = lbl_dir / f"{stem}.txt"
        lines = [l.strip() for l in lbl_file.read_text(encoding="utf-8").splitlines() if l.strip()]
        
        boxes = []
        mask = np.zeros((im_h, im_w), dtype=int)
        for l in lines:
            cid, xc, yc, w, h = [float(x) for x in l.split()]
            # Raw pixel coords
            rx1 = max(0.0, (xc - w/2) * im_w)
            ry1 = max(0.0, (yc - h/2) * im_h)
            rx2 = min(float(im_w), (xc + w/2) * im_w)
            ry2 = min(float(im_h), (yc + h/2) * im_h)
            boxes.append((rx1, ry1, rx2, ry2))
            
            # Buffered pixel coords
            bx1 = max(0, int(np.floor(rx1 - safety_buffer)))
            by1 = max(0, int(np.floor(ry1 - safety_buffer)))
            bx2 = min(im_w, int(np.ceil(rx2 + safety_buffer)))
            by2 = min(im_h, int(np.ceil(ry2 + safety_buffer)))
            mask[by1:by2, bx1:bx2] = 1
            
        raw_defect_boxes[stem] = boxes
        buffered_defect_masks[stem] = mask

    # 4. Extract Candidate Background Patches
    # Target: 1 background candidate per Dent crop, matching split and dimensions
    rng = random.Random(20261019)
    
    extracted_records = []
    splits = ["train", "val", "test"]
    split_bg_counts = Counter()
    
    # Cache image arrays as needed
    loaded_images = {}
    
    for sp in splits:
        sp_dent_recs = [r for r in dent_records if r["split"] == sp]
        sp_stems = sorted(list(set(r["source_image_stem"] for r in sp_dent_recs)))
        
        # Working occupied masks for this split (initially buffered defects)
        # 1 means occupied/forbidden, 0 means available
        work_masks = {stem: buffered_defect_masks[stem].copy() for stem in sp_stems}
        
        # Integral images for fast O(1) query
        def get_integral(m):
            pad = np.pad(m, ((1, 0), (1, 0)), mode="constant")
            return np.cumsum(np.cumsum(pad, axis=0), axis=1)
            
        work_integrals = {stem: get_integral(work_masks[stem]) for stem in sp_stems}
        
        print(f"\nProcessing Split '{sp}': Target = {len(sp_dent_recs)} candidates across {len(sp_stems)} source images...")
        
        placed_count = 0
        
        # Strategy: For each Dent crop target in this split:
        # 1. Try placing in the same source image
        # 2. If no space, search in another image in the SAME split
        # 3. If exact dimension cannot fit anywhere, scale down moderately preserving aspect ratio
        
        for idx, d_rec in enumerate(sp_dent_recs, 1):
            target_w, target_h = d_rec["crop_dimensions"]
            orig_stem = d_rec["source_image_stem"]
            
            placed = False
            chosen_stem = None
            chosen_x, chosen_y = None, None
            chosen_w, chosen_h = target_w, target_h
            scale_applied = 1.0
            
            # Step 1: Check same image first
            integ = work_integrals[orig_stem]
            if target_w <= im_w and target_h <= im_h:
                w_sum = integ[target_h:, target_w:] - integ[:-target_h, target_w:] - integ[target_h:, :-target_w] + integ[:-target_h, :-target_w]
                valid_pos = np.argwhere(w_sum == 0)
                if len(valid_pos) > 0:
                    # Pick deterministically via RNG
                    py, px = valid_pos[rng.randint(0, len(valid_pos) - 1)]
                    chosen_stem = orig_stem
                    chosen_x, chosen_y = int(px), int(py)
                    placed = True
                    
            # Step 2: Try other images in the SAME split with exact dimensions
            if not placed:
                # Random permutation of stems to spread evenly
                candidate_stems = list(sp_stems)
                rng.shuffle(candidate_stems)
                for c_stem in candidate_stems:
                    integ = work_integrals[c_stem]
                    if target_w <= im_w and target_h <= im_h:
                        w_sum = integ[target_h:, target_w:] - integ[:-target_h, target_w:] - integ[target_h:, :-target_w] + integ[:-target_h, :-target_w]
                        valid_pos = np.argwhere(w_sum == 0)
                        if len(valid_pos) > 0:
                            py, px = valid_pos[rng.randint(0, len(valid_pos) - 1)]
                            chosen_stem = c_stem
                            chosen_x, chosen_y = int(px), int(py)
                            placed = True
                            break
                            
            # Step 3: If still not placed (e.g. very large box), scale down moderately preserving aspect ratio
            if not placed:
                for scale in [0.85, 0.70, 0.55, 0.40]:
                    sw = max(10, int(round(target_w * scale)))
                    sh = max(10, int(round(target_h * scale)))
                    for c_stem in candidate_stems:
                        integ = work_integrals[c_stem]
                        if sw <= im_w and sh <= im_h:
                            w_sum = integ[sh:, sw:] - integ[:-sh, sw:] - integ[sh:, :-sw] + integ[:-sh, :-sw]
                            valid_pos = np.argwhere(w_sum == 0)
                            if len(valid_pos) > 0:
                                py, px = valid_pos[rng.randint(0, len(valid_pos) - 1)]
                                chosen_stem = c_stem
                                chosen_x, chosen_y = int(px), int(py)
                                chosen_w, chosen_h = sw, sh
                                scale_applied = scale
                                placed = True
                                break
                        if placed: break
                    if placed: break
                    
            if not placed:
                print(f"WARNING: Could not place background patch for Dent {d_rec['crop_id']} ({target_w}x{target_h}) in split {sp}!")
                continue
                
            # Update mask and integral image for chosen_stem
            work_masks[chosen_stem][chosen_y:chosen_y+chosen_h, chosen_x:chosen_x+chosen_w] = 1
            work_integrals[chosen_stem] = get_integral(work_masks[chosen_stem])
            
            # Compute minimum distance to nearest defect bounding box
            min_dist = float("inf")
            patch_x1, patch_y1 = chosen_x, chosen_y
            patch_x2, patch_y2 = chosen_x + chosen_w, chosen_y + chosen_h
            
            for rx1, ry1, rx2, ry2 in raw_defect_boxes[chosen_stem]:
                # Distance between rectangle [patch_x1, patch_y1, patch_x2, patch_y2]
                # and [rx1, ry1, rx2, ry2]
                dx = max(0.0, max(rx1 - patch_x2, patch_x1 - rx2))
                dy = max(0.0, max(ry1 - patch_y2, patch_y1 - ry2))
                dist = np.hypot(dx, dy)
                if dist < min_dist:
                    min_dist = dist
                    
            assert min_dist >= safety_buffer, f"Safety buffer violation: min_dist={min_dist} < {safety_buffer}!"
            
            # Load source image and extract crop
            src_path = img_dir / f"{chosen_stem}.jpg"
            if chosen_stem not in loaded_images:
                with Image.open(src_path) as im:
                    loaded_images[chosen_stem] = np.array(im)
            im_arr = loaded_images[chosen_stem]
            
            patch_arr = im_arr[patch_y1:patch_y2, patch_x1:patch_x2]
            patch_im = Image.fromarray(patch_arr)
            
            candidate_id = f"bg_{chosen_stem}_p{idx:04d}"
            crop_rel = f"crops/{sp}/{candidate_id}.png"
            crop_abs = bg_audit_dir / crop_rel
            patch_im.save(crop_abs, format="PNG")
            
            rec = {
                "candidate_id": candidate_id,
                "source_image_stem": chosen_stem,
                "source_image_path": str(src_path).replace("\\", "/"),
                "sha256": image_to_hash[chosen_stem],
                "split": sp,
                "patch_pixels": [patch_x1, patch_y1, patch_x2, patch_y2],
                "patch_dimensions": [chosen_w, chosen_h],
                "aspect_ratio": round(chosen_w / chosen_h, 3),
                "matched_dent_id": d_rec["crop_id"],
                "target_dimensions": [target_w, target_h],
                "scale_factor": round(scale_applied, 2),
                "safety_buffer_px": safety_buffer,
                "min_defect_dist_px": round(min_dist, 1),
                "crop_path": crop_rel.replace("\\", "/"),
                "crop_abs_path": str(crop_abs).replace("\\", "/"),
                "class_label": "Background_Candidate",
                "validation_status": "CANDIDATE_UNVERIFIED",
                "verification_caveat": "YOLO annotations cannot guarantee unannotated regions are defect-free; visual inspection required."
            }
            extracted_records.append(rec)
            split_bg_counts[sp] += 1
            placed_count += 1
            
        print(f"Split '{sp}': Successfully placed {placed_count} / {len(sp_dent_recs)} candidates.")

    # 5. Save Manifests (JSONL and CSV)
    jsonl_path = manifests_dir / "background_candidates_manifest.jsonl"
    csv_path = manifests_dir / "background_candidates_manifest.csv"
    
    with open(jsonl_path, "w", encoding="utf-8") as f:
        for r in extracted_records:
            f.write(json.dumps(r) + "\n")
    print(f"\nSaved background candidate JSONL: {jsonl_path} ({len(extracted_records)} records)")
    
    csv_fields = [
        "candidate_id", "source_image_stem", "split", "class_label", "validation_status",
        "patch_width", "patch_height", "aspect_ratio", "min_defect_dist_px",
        "matched_dent_id", "crop_path", "sha256", "source_image_path"
    ]
    with open(csv_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=csv_fields)
        writer.writeheader()
        for r in extracted_records:
            writer.writerow({
                "candidate_id": r["candidate_id"],
                "source_image_stem": r["source_image_stem"],
                "split": r["split"],
                "class_label": r["class_label"],
                "validation_status": r["validation_status"],
                "patch_width": r["patch_dimensions"][0],
                "patch_height": r["patch_dimensions"][1],
                "aspect_ratio": r["aspect_ratio"],
                "min_defect_dist_px": r["min_defect_dist_px"],
                "matched_dent_id": r["matched_dent_id"],
                "crop_path": r["crop_path"],
                "sha256": r["sha256"],
                "source_image_path": r["source_image_path"]
            })
    print(f"Saved background candidate CSV: {csv_path}")

    # 6. Generate Contact Sheets for Train, Val, and Test
    # Each contact sheet will display a diverse 16-sample grid (4x4)
    for sp in splits:
        sp_recs = [r for r in extracted_records if r["split"] == sp]
        
        # Pick 16 diverse samples:
        # 2 smallest, 2 largest
        # 2 tallest, 2 widest
        # 8 random/spread
        sorted_area = sorted(sp_recs, key=lambda r: r["patch_dimensions"][0] * r["patch_dimensions"][1])
        sorted_ar = sorted(sp_recs, key=lambda r: r["aspect_ratio"])
        
        selected = []
        selected.extend(sorted_area[:2])
        selected.extend(sorted_area[-2:])
        selected.extend(sorted_ar[:2])
        selected.extend(sorted_ar[-2:])
        
        # Fill to 16
        seen = set(r["candidate_id"] for r in selected)
        rem = [r for r in sp_recs if r["candidate_id"] not in seen]
        rng.shuffle(rem)
        selected.extend(rem[:16 - len(selected)])
        selected = selected[:16]
        
        # Build 4x4 contact sheet grid
        grid_cols, grid_rows = 4, 4
        cell_w, cell_h = 200, 240
        padding = 10
        sheet_w = grid_cols * cell_w + (grid_cols + 1) * padding
        sheet_h = grid_rows * cell_h + (grid_rows + 1) * padding + 50
        
        sheet = Image.new("RGB", (sheet_w, sheet_h), color=(28, 30, 34))
        draw = ImageDraw.Draw(sheet)
        
        # Header
        draw.text((padding, 12), f"CR7-DET Background Candidates — [{sp.upper()}] Partition (16 Diverse Samples)", fill=(240, 240, 240))
        draw.text((padding, 30), f"Audit Only: Safety buffer = 20px. YOLO annotations cannot guarantee unannotated regions are defect-free.", fill=(180, 180, 180))
        
        for idx, r in enumerate(selected):
            row = idx // grid_cols
            col = idx % grid_cols
            x_cell = padding + col * (cell_w + padding)
            y_cell = 50 + padding + row * (cell_h + padding)
            
            crop_abs = bg_audit_dir / r["crop_path"]
            with Image.open(crop_abs) as c_im:
                cw, ch = c_im.size
                scale = min((cell_w - 10) / cw, (cell_w - 10) / ch)
                nw = max(1, int(round(cw * scale)))
                nh = max(1, int(round(ch * scale)))
                c_res = c_im.resize((nw, nh), Image.Resampling.LANCZOS)
                
            tx = x_cell + (cell_w - nw) // 2
            ty = y_cell + (cell_w - nh) // 2
            
            draw.rectangle([x_cell, y_cell, x_cell + cell_w, y_cell + cell_w], fill=(16, 17, 19), outline=(55, 60, 68), width=1)
            
            if c_res.mode != "RGB":
                c_res = c_res.convert("RGB")
            sheet.paste(c_res, (tx, ty))
            
            # Caption
            draw.text((x_cell + 4, y_cell + cell_w + 4), f"{r['candidate_id']}", fill=(210, 210, 210))
            draw.text((x_cell + 4, y_cell + cell_w + 18), f"{r['patch_dimensions'][0]}x{r['patch_dimensions'][1]} (ar:{r['aspect_ratio']}) d>={r['min_defect_dist_px']}px", fill=(150, 150, 150))
            draw.text((x_cell + cell_w - 55, y_cell + cell_w + 4), f"[{sp.upper()}]", fill=(100, 200, 255) if sp=="train" else (255, 180, 80) if sp=="val" else (120, 240, 140))
            
        contact_path = contact_dir / f"background_{sp}_contact_sheet.png"
        sheet.save(contact_path, format="PNG")
        print(f"Saved {sp} contact sheet: {contact_path}")

    # 7. Quality & Distribution Summary
    ar_list = [r["aspect_ratio"] for r in extracted_records]
    min_dist_list = [r["min_defect_dist_px"] for r in extracted_records]
    
    print("\n--- SUMMARY METRICS ---")
    print(f"Total background candidates: {len(extracted_records)}")
    print(f"Split breakdown: {dict(split_bg_counts)}")
    print(f"Aspect ratio range: min={min(ar_list):.3f}, max={max(ar_list):.3f}, mean={np.mean(ar_list):.3f}, median={np.median(ar_list):.3f}")
    print(f"Distance to nearest defect (px): min={min(min_dist_list):.1f}, max={max(min_dist_list):.1f}, mean={np.mean(min_dist_list):.1f}, median={np.median(min_dist_list):.1f}")
    
    # Check duplicate group split consistency
    dup_cluster_splits = defaultdict(set)
    for r in extracted_records:
        dup_cluster_splits[r["sha256"]].add(r["split"])
    violations = [h for h, sp_set in dup_cluster_splits.items() if len(sp_set) > 1]
    print(f"Split integrity violations (duplicate hashes crossing splits): {len(violations)}")
    assert len(violations) == 0, "Duplicate hash split violation detected!"

if __name__ == "__main__":
    run_background_audit()
