import json
from pathlib import Path
from collections import defaultdict, Counter
from PIL import Image
import numpy as np

def run_pretraining_validation():
    print("=== DENT PILOT STEP 3B: PRE-TRAINING VALIDATION ===")
    
    base_dir = Path(r"D:\DefectVision\experiments\dent_classification")
    dent_manifest_path = base_dir / "manifests" / "dent_pilot_manifest.jsonl"
    bg_manifest_path = base_dir / "background_audit" / "manifests" / "background_candidates_manifest.jsonl"
    lbl_dir = Path(r"C:\Users\susha\Downloads\CR7-DET\CR7-DET\label")
    img_dir = Path(r"C:\Users\susha\Downloads\CR7-DET\CR7-DET\image")
    
    dent_recs = [json.loads(l) for l in dent_manifest_path.read_text(encoding="utf-8").splitlines() if l.strip()]
    bg_recs = [json.loads(l) for l in bg_manifest_path.read_text(encoding="utf-8").splitlines() if l.strip()]
    
    print(f"Loaded {len(dent_recs)} Dent records and {len(bg_recs)} Background records.")
    
    # 1. Pixel exact match for every candidate crop
    print("\n--- 1. Candidate Extraction Verification ---")
    pixel_mismatches = 0
    loaded_imgs = {}
    
    for r in bg_recs:
        stem = r["source_image_stem"]
        if stem not in loaded_imgs:
            with Image.open(img_dir / f"{stem}.jpg") as im:
                loaded_imgs[stem] = np.array(im)
        src_arr = loaded_imgs[stem]
        
        x1, y1, x2, y2 = r["patch_pixels"]
        expected_slice = src_arr[y1:y2, x1:x2]
        
        crop_path = base_dir / "background_audit" / r["crop_path"]
        with Image.open(crop_path) as c_im:
            actual_arr = np.array(c_im)
            
        if expected_slice.shape != actual_arr.shape or not np.array_equal(expected_slice, actual_arr):
            pixel_mismatches += 1
            
    print(f"Crop pixel mismatches against source image slices: {pixel_mismatches} / {len(bg_recs)}")
    
    # 2. Check intersection with annotated bounding boxes + 20px buffer
    buffer_violations = 0
    min_dist_overall = float("inf")
    im_w, im_h = 400, 400
    
    for r in bg_recs:
        stem = r["source_image_stem"]
        x1, y1, x2, y2 = r["patch_pixels"]
        
        lbl_file = lbl_dir / f"{stem}.txt"
        lines = [l.strip() for l in lbl_file.read_text().splitlines() if l.strip()]
        
        for l in lines:
            cid, xc, yc, w, h = [float(x) for x in l.split()]
            rx1 = max(0.0, (xc - w/2) * im_w)
            ry1 = max(0.0, (yc - h/2) * im_h)
            rx2 = min(float(im_w), (xc + w/2) * im_w)
            ry2 = min(float(im_h), (yc + h/2) * im_h)
            
            # Check Euclidean distance
            dx = max(0.0, max(rx1 - x2, x1 - rx2))
            dy = max(0.0, max(ry1 - y2, y1 - ry2))
            dist = np.hypot(dx, dy)
            if dist < min_dist_overall:
                min_dist_overall = dist
            if dist < 20.0 - 1e-5:
                buffer_violations += 1
                
    print(f"Safety buffer (< 20px) violations: {buffer_violations}")
    print(f"Minimum observed distance to any annotated defect box: {min_dist_overall:.2f} px")
    
    # 3. Hash and split consistency with existing Dent manifest
    dent_img_to_split = {r["source_image_stem"]: r["split"] for r in dent_recs}
    dent_img_to_hash = {r["source_image_stem"]: r["sha256"] for r in dent_recs}
    
    split_inconsistencies = 0
    hash_inconsistencies = 0
    
    for r in bg_recs:
        stem = r["source_image_stem"]
        bg_split = r["split"]
        bg_hash = r["sha256"]
        
        if stem in dent_img_to_split and dent_img_to_split[stem] != bg_split:
            split_inconsistencies += 1
        if stem in dent_img_to_hash and dent_img_to_hash[stem] != bg_hash:
            hash_inconsistencies += 1
            
    print(f"Split inconsistencies vs Dent manifest: {split_inconsistencies}")
    print(f"Hash inconsistencies vs Dent manifest: {hash_inconsistencies}")
    
    # Check duplicate hashes crossing splits in bg manifest
    hash_to_bg_splits = defaultdict(set)
    for r in bg_recs:
        hash_to_bg_splits[r["sha256"]].add(r["split"])
    dup_hash_split_leaks = {h: sps for h, sps in hash_to_bg_splits.items() if len(sps) > 1}
    print(f"Background duplicate hashes crossing splits: {len(dup_hash_split_leaks)}")

    # 4. Check for duplicate crops
    print("\n--- Duplicate Crop Checks ---")
    import hashlib
    bg_crop_hashes = Counter()
    dent_crop_hashes = Counter()
    
    for r in bg_recs:
        p = base_dir / "background_audit" / r["crop_path"]
        ch = hashlib.sha256(p.read_bytes()).hexdigest()
        bg_crop_hashes[ch] += 1
        
    for r in dent_recs:
        p = base_dir / r["crop_path"]
        ch = hashlib.sha256(p.read_bytes()).hexdigest()
        dent_crop_hashes[ch] += 1
        
    bg_internal_dups = {h: cnt for h, cnt in bg_crop_hashes.items() if cnt > 1}
    bg_dent_intersections = set(bg_crop_hashes.keys()) & set(dent_crop_hashes.keys())
    
    print(f"Unique background crop file hashes: {len(bg_crop_hashes)} / {len(bg_recs)}")
    print(f"Background internal identical crop clusters: {len(bg_internal_dups)}")
    if bg_internal_dups:
        print(f"  Sample internal dups: {list(bg_internal_dups.items())[:3]}")
    print(f"Identical crops between Background and Dent-positive: {len(bg_dent_intersections)}")

    # 2. Check label and sampling validity
    print("\n--- 2. Label and Sampling Validity ---")
    matched_dent_map = {r["crop_id"]: r for r in dent_recs}
    same_image_count = 0
    diff_image_same_split = 0
    diff_image_diff_split = 0
    
    for r in bg_recs:
        matched_dent = matched_dent_map[r["matched_dent_id"]]
        if r["source_image_stem"] == matched_dent["source_image_stem"]:
            same_image_count += 1
        elif r["split"] == matched_dent["split"]:
            diff_image_same_split += 1
        else:
            diff_image_diff_split += 1
            
    print(f"Background candidates from EXACT SAME source image as matched Dent: {same_image_count} ({same_image_count/len(bg_recs)*100:.1f}%)")
    print(f"Background candidates from DIFFERENT image in SAME split: {diff_image_same_split} ({diff_image_same_split/len(bg_recs)*100:.1f}%)")
    print(f"Background candidates from DIFFERENT split: {diff_image_diff_split} ({diff_image_diff_split/len(bg_recs)*100:.1f}%)")
    
    # Unique source images per split
    dent_imgs_by_split = defaultdict(set)
    for r in dent_recs:
        dent_imgs_by_split[r["split"]].add(r["source_image_stem"])
        
    bg_imgs_by_split = defaultdict(set)
    for r in bg_recs:
        bg_imgs_by_split[r["split"]].add(r["source_image_stem"])
        
    for sp in ["train", "val", "test"]:
        d_imgs = len(dent_imgs_by_split[sp])
        b_imgs = len(bg_imgs_by_split[sp])
        shared = len(dent_imgs_by_split[sp] & bg_imgs_by_split[sp])
        print(f"Split {sp:5s}: Dent images = {d_imgs}, BG images = {b_imgs}, Shared = {shared}")

    # Comparative Distribution Analysis: Dent vs Background
    # 1. Size / Area
    dent_areas = np.array([r["crop_dimensions"][0] * r["crop_dimensions"][1] for r in dent_recs])
    bg_areas = np.array([r["patch_dimensions"][0] * r["patch_dimensions"][1] for r in bg_recs])
    
    # 2. Aspect Ratio
    dent_ar = np.array([r["aspect_ratio"] for r in dent_recs])
    bg_ar = np.array([r["aspect_ratio"] for r in bg_recs])
    
    # 3. Spatial Location (center x, center y)
    dent_cx = np.array([(r["crop_pixels"][0] + r["crop_pixels"][2]) / 2 for r in dent_recs])
    dent_cy = np.array([(r["crop_pixels"][1] + r["crop_pixels"][3]) / 2 for r in dent_recs])
    
    bg_cx = np.array([(r["patch_pixels"][0] + r["patch_pixels"][2]) / 2 for r in bg_recs])
    bg_cy = np.array([(r["patch_pixels"][1] + r["patch_pixels"][3]) / 2 for r in bg_recs])
    
    # 4. Pixel Intensity (mean and std across crops)
    print("\nComputing pixel intensities across crops...")
    dent_means = []
    dent_stds = []
    for r in dent_recs:
        p = base_dir / r["crop_path"]
        arr = np.array(Image.open(p))
        dent_means.append(np.mean(arr))
        dent_stds.append(np.std(arr))
        
    bg_means = []
    bg_stds = []
    for r in bg_recs:
        p = base_dir / "background_audit" / r["crop_path"]
        arr = np.array(Image.open(p))
        bg_means.append(np.mean(arr))
        bg_stds.append(np.std(arr))
        
    dent_means = np.array(dent_means)
    bg_means = np.array(bg_means)
    dent_stds = np.array(dent_stds)
    bg_stds = np.array(bg_stds)

    print("\n--- COMPARATIVE STATISTICS ---")
    print(f"Crop Area (px^2):")
    print(f"  Dent: mean={dent_areas.mean():.1f}, std={dent_areas.std():.1f}, median={np.median(dent_areas):.1f}, min={dent_areas.min()}, max={dent_areas.max()}")
    print(f"  BG  : mean={bg_areas.mean():.1f}, std={bg_areas.std():.1f}, median={np.median(bg_areas):.1f}, min={bg_areas.min()}, max={bg_areas.max()}")

    print(f"\nAspect Ratio (W/H):")
    print(f"  Dent: mean={dent_ar.mean():.3f}, std={dent_ar.std():.3f}, median={np.median(dent_ar):.3f}, min={dent_ar.min():.3f}, max={dent_ar.max():.3f}")
    print(f"  BG  : mean={bg_ar.mean():.3f}, std={bg_ar.std():.3f}, median={np.median(bg_ar):.3f}, min={bg_ar.min():.3f}, max={bg_ar.max():.3f}")

    print(f"\nCenter X (px out of 400):")
    print(f"  Dent: mean={dent_cx.mean():.1f}, std={dent_cx.std():.1f}, median={np.median(dent_cx):.1f}")
    print(f"  BG  : mean={bg_cx.mean():.1f}, std={bg_cx.std():.1f}, median={np.median(bg_cx):.1f}")

    print(f"\nCenter Y (px out of 400):")
    print(f"  Dent: mean={dent_cy.mean():.1f}, std={dent_cy.std():.1f}, median={np.median(dent_cy):.1f}")
    print(f"  BG  : mean={bg_cy.mean():.1f}, std={bg_cy.std():.1f}, median={np.median(bg_cy):.1f}")

    print(f"\nMean Pixel Intensity (0-255):")
    print(f"  Dent: mean={dent_means.mean():.1f}, std={dent_means.std():.1f}, median={np.median(dent_means):.1f}")
    print(f"  BG  : mean={bg_means.mean():.1f}, std={bg_means.std():.1f}, median={np.median(bg_means):.1f}")

    print(f"\nCrop Pixel Variance / Texture Std (0-255):")
    print(f"  Dent: mean={dent_stds.mean():.1f}, std={dent_stds.std():.1f}, median={np.median(dent_stds):.1f}")
    print(f"  BG  : mean={bg_stds.mean():.1f}, std={bg_stds.std():.1f}, median={np.median(bg_stds):.1f}")

if __name__ == "__main__":
    run_pretraining_validation()
