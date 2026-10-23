import json
from pathlib import Path

manifest_path = Path(r"D:\DefectVision\experiments\dent_classification\manifests\dent_pilot_manifest.jsonl")
records = [json.loads(line) for line in manifest_path.read_text(encoding="utf-8").splitlines() if line.strip()]

print("--- Severely Tall (AR < 0.25) Examples ---")
tall_ex = sorted([r for r in records if r["aspect_ratio"] < 0.25], key=lambda x: x["aspect_ratio"])[:5]
for r in tall_ex:
    print(f"  {r['crop_id']}: stem={r['source_image_stem']} box={r['box_index']} dims={r['crop_dimensions']} AR={r['aspect_ratio']} split={r['split']}")

print("\n--- Severely Wide (AR > 4.0) Examples ---")
wide_ex = sorted([r for r in records if r["aspect_ratio"] > 4.0], key=lambda x: x["aspect_ratio"], reverse=True)[:5]
for r in wide_ex:
    print(f"  {r['crop_id']}: stem={r['source_image_stem']} box={r['box_index']} dims={r['crop_dimensions']} AR={r['aspect_ratio']} split={r['split']}")

print("\n--- Balanced / Near-Square Examples ---")
square_ex = [r for r in records if 0.98 <= r["aspect_ratio"] <= 1.02][:5]
for r in square_ex:
    print(f"  {r['crop_id']}: stem={r['source_image_stem']} box={r['box_index']} dims={r['crop_dimensions']} AR={r['aspect_ratio']} split={r['split']}")
