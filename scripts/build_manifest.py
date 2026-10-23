from collections import Counter
from pathlib import Path
import hashlib
import json
import random
import xml.etree.ElementTree as ET


SOURCES_ROOT = Path(r"D:\datasets\defect_dataset_workspace\sources")
MANIFEST_OUTPUT = Path(r"D:\DefectVision\manifests\dataset_manifest.jsonl")

RANDOM_SEED = 42


# ============================================================
# DATASET ROOTS
# ============================================================

MVTEC_ROOT = SOURCES_ROOT / "MVTecAD"
MT_ROOT = SOURCES_ROOT / "MT"
NEU_ROOT = SOURCES_ROOT / "NEU-CLS"
GC10_ROOT = SOURCES_ROOT / "GC10-DET"


# ============================================================
# APPROVED TARGET TAXONOMY
# ============================================================

TARGET_LABELS = {
    "Scratch",
    "Crack",
    "Discoloration",
}


# ============================================================
# MVTec MAPPINGS
# ============================================================

MVTEC_DIRECT_MAPPINGS = {
    ("capsule", "scratch"): "Scratch",
    ("metal_nut", "scratch"): "Scratch",
    ("pill", "scratch"): "Scratch",
    ("screw", "scratch_head"): "Scratch",
    ("screw", "scratch_neck"): "Scratch",

    ("capsule", "crack"): "Crack",
    ("hazelnut", "crack"): "Crack",
    ("pill", "crack"): "Crack",
    ("tile", "crack"): "Crack",

    ("metal_nut", "color"): "Discoloration",
}

MVTEC_POSSIBLE_REVIEW = {
    ("tile", "gray_stroke"),
}

MVTEC_NOT_EQUIVALENT = {
    ("leather", "cut"),
}


# ============================================================
# NEU MAPPINGS
# ============================================================

NEU_DIRECT_MAPPINGS = {
    "scratches": "Scratch",
}

NEU_POSSIBLE_REVIEW = {
    "crazing",
    "patches",
}


# ============================================================
# MAGNETIC TILE MAPPINGS
# ============================================================

MT_DIRECT_MAPPINGS = {
    "Crack": "Crack",
    "MT_Crack": "Crack",
}

MT_POSSIBLE_REVIEW = {
    "Break",
    "MT_Break",
    "Uneven",
    "MT_Uneven",
}


# ============================================================
# GC10 MAPPINGS
# ============================================================

GC10_DIRECT_MAPPINGS = {
    "water_spot": "Discoloration",
    "oil_spot": "Discoloration",
    "silk_spot": "Discoloration",
}

GC10_POSSIBLE_REVIEW = {
    "rolled_pit",
}

GC10_NOT_EQUIVALENT = {
    "crease",
}

GC10_TAG_TO_SEMANTIC = {
    "1_chongkong": "punching_hole",
    "2_hanfeng": "welding_line",
    "3_yueyawan": "crescent_gap",
    "4_shuiban": "water_spot",
    "5_youban": "oil_spot",
    "6_siban": "silk_spot",
    "7_yiwu": "inclusion",
    "8_yahen": "rolled_pit",
    "9_zhehen": "crease",
    "10_yaozhe": "waist_folding",
    "10_yaozhed": "waist_folding",
}

GC10_FOLDER_TO_SEMANTIC = {
    "1": "punching_hole",
    "2": "welding_line",
    "3": "crescent_gap",
    "4": "water_spot",
    "5": "oil_spot",
    "6": "silk_spot",
    "7": "inclusion",
    "8": "rolled_pit",
    "9": "crease",
    "10": "waist_folding",
}


def get_gc10_semantic_class(image_path: Path, xml_path: Path | None) -> str:
    folder_semantic = GC10_FOLDER_TO_SEMANTIC.get(image_path.parent.name, "unknown")
    if xml_path is None or not xml_path.exists():
        return folder_semantic
    try:
        tree = ET.parse(xml_path)
        objects = tree.findall("object")
        if not objects:
            return folder_semantic

        class_counts = Counter()
        class_areas = Counter()

        for obj in objects:
            name_elem = obj.find("name")
            if name_elem is not None and name_elem.text:
                raw_tag = name_elem.text.strip().lower()
                sem_class = GC10_TAG_TO_SEMANTIC.get(raw_tag)
                if sem_class:
                    class_counts[sem_class] += 1
                    bndbox = obj.find("bndbox")
                    if bndbox is not None:
                        try:
                            w = abs(float(bndbox.find("xmax").text) - float(bndbox.find("xmin").text))
                            h = abs(float(bndbox.find("ymax").text) - float(bndbox.find("ymin").text))
                            class_areas[sem_class] += (w * h)
                        except (TypeError, ValueError, AttributeError):
                            pass

        if not class_counts:
            return folder_semantic

        max_count = max(class_counts.values())
        top_classes = [c for c, cnt in class_counts.items() if cnt == max_count]
        if len(top_classes) == 1:
            return top_classes[0]
        if folder_semantic in top_classes:
            return folder_semantic
        return max(top_classes, key=lambda c: class_areas[c])
    except Exception:
        return folder_semantic


# ============================================================
# UTILITY FUNCTIONS
# ============================================================

def normalize_path(path: Path) -> str:
    return str(path.resolve()).replace("\\", "/")


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:16]


def make_sample_id(
    dataset: str,
    product: str,
    original_label: str,
    source_image_path: Path,
) -> str:

    try:
        relative_path = source_image_path.relative_to(SOURCES_ROOT)
    except ValueError:
        relative_path = source_image_path

    identity = "|".join(
        [
            dataset,
            product,
            original_label,
            normalize_path(relative_path),
        ]
    )

    return sha256_text(identity)


def deterministic_split(items, train_ratio=0.70, val_ratio=0.15):
    """
    Deterministic 70/15/15 split.

    Supports both:
    - Path objects
    - tuples whose first element is a Path
    """

    def sort_key(item):
        if isinstance(item, tuple):
            return normalize_path(item[0])

        return normalize_path(item)

    items = sorted(items, key=sort_key)

    rng = random.Random(RANDOM_SEED)
    items = list(items)
    rng.shuffle(items)

    n = len(items)

    train_end = int(n * train_ratio)
    val_end = train_end + int(n * val_ratio)

    return (
        items[:train_end],
        items[train_end:val_end],
        items[val_end:],
    )


def deterministic_binary_split(items):
    """
    Deterministic 50/50 split.
    """

    def sort_key(item):
        if isinstance(item, tuple):
            return normalize_path(item[0])

        return normalize_path(item)

    items = sorted(items, key=sort_key)

    rng = random.Random(RANDOM_SEED)
    items = list(items)
    rng.shuffle(items)

    midpoint = len(items) // 2

    return items[:midpoint], items[midpoint:]


def create_record(
    *,
    dataset,
    original_label,
    target_label,
    mapping_status,
    source_image_path,
    mask_path=None,
    annotation_path=None,
    product="unknown",
    split="none",
    role="unclassified",
    is_defective=False,
    provenance="",
):

    return {
        "sample_id": make_sample_id(
            dataset,
            product,
            original_label,
            source_image_path,
        ),
        "dataset": dataset,
        "original_label": original_label,
        "target_label": target_label,
        "mapping_status": mapping_status,
        "source_image_path": normalize_path(source_image_path),
        "mask_path": (
            normalize_path(mask_path)
            if mask_path is not None
            else None
        ),
        "annotation_path": (
            normalize_path(annotation_path)
            if annotation_path is not None
            else None
        ),
        "product": product,
        "split": split,
        "role": role,
        "is_defective": is_defective,
        "provenance": provenance,
    }


# ============================================================
# ROLE ASSIGNMENT
# ============================================================

def get_defective_role(target_label, mapping_status):

    if target_label in TARGET_LABELS:
        return "classifier_source"

    if mapping_status in {
        "POSSIBLE_BUT_REQUIRES_REVIEW",
        "NOT_EQUIVALENT",
    }:
        return "classifier_review"

    return "unknown_defect_source"


# ============================================================
# MVTec AD
# ============================================================

def process_mvtec(records):

    if not MVTEC_ROOT.exists():
        print(f"[WARN] MVTec root not found: {MVTEC_ROOT}")
        return

    categories = sorted(
        [
            p
            for p in MVTEC_ROOT.iterdir()
            if p.is_dir()
        ]
    )

    for category_dir in categories:

        category = category_dir.name

        train_good_dir = category_dir / "train" / "good"
        test_good_dir = category_dir / "test" / "good"
        ground_truth_dir = category_dir / "ground_truth"

        # Official training-good images
        if train_good_dir.exists():

            for image_path in sorted(train_good_dir.glob("*")):

                if not image_path.is_file():
                    continue

                records.append(
                    create_record(
                        dataset="MVTecAD",
                        original_label="good",
                        target_label=None,
                        mapping_status="NORMAL_REFERENCE",
                        source_image_path=image_path,
                        product=category,
                        split="train",
                        role="patchcore_calibration",
                        is_defective=False,
                        provenance=(
                            "MVTec AD official train/good split; "
                            "normal reference image."
                        ),
                    )
                )

        # Official test-good images
        if test_good_dir.exists():

            for image_path in sorted(test_good_dir.glob("*")):

                if not image_path.is_file():
                    continue

                records.append(
                    create_record(
                        dataset="MVTecAD",
                        original_label="good",
                        target_label=None,
                        mapping_status="NORMAL_REFERENCE",
                        source_image_path=image_path,
                        product=category,
                        split="test",
                        role="anomaly_validation",
                        is_defective=False,
                        provenance=(
                            "MVTec AD official test/good split; "
                            "normal anomaly-evaluation image."
                        ),
                    )
                )

        # Defective test images
        test_dir = category_dir / "test"

        if not test_dir.exists():
            continue

        defect_dirs = sorted(
            [
                p
                for p in test_dir.iterdir()
                if p.is_dir() and p.name != "good"
            ]
        )

        for defect_dir in defect_dirs:

            original_label = defect_dir.name

            direct_key = (
                category,
                original_label,
            )

            if direct_key in MVTEC_DIRECT_MAPPINGS:

                target_label = MVTEC_DIRECT_MAPPINGS[direct_key]
                mapping_status = "DIRECT"

            elif direct_key in MVTEC_POSSIBLE_REVIEW:

                target_label = None
                mapping_status = "POSSIBLE_BUT_REQUIRES_REVIEW"

            elif direct_key in MVTEC_NOT_EQUIVALENT:

                target_label = None
                mapping_status = "NOT_EQUIVALENT"

            else:

                target_label = None
                mapping_status = "NO_MAPPING"

            role = get_defective_role(
                target_label,
                mapping_status,
            )

            for image_path in sorted(defect_dir.glob("*")):

                if not image_path.is_file():
                    continue

                mask_path = (
                    ground_truth_dir
                    / original_label
                    / f"{image_path.stem}_mask.png"
                )

                if not mask_path.exists():
                    mask_path = None

                records.append(
                    create_record(
                        dataset="MVTecAD",
                        original_label=original_label,
                        target_label=target_label,
                        mapping_status=mapping_status,
                        source_image_path=image_path,
                        mask_path=mask_path,
                        product=category,
                        split="test",
                        role=role,
                        is_defective=True,
                        provenance=(
                            "MVTec AD official test/defective split."
                        ),
                    )
                )


# ============================================================
# MAGNETIC TILE
# ============================================================

def process_mt(records):

    if not MT_ROOT.exists():
        print(f"[WARN] MT root not found: {MT_ROOT}")
        return

    free_dir = MT_ROOT / "MT_Free" / "Imgs"

    if free_dir.exists():

        normal_images = sorted(
            [
                p
                for p in free_dir.glob("*")
                if p.is_file()
                and p.suffix.lower() in {".jpg", ".jpeg"}
            ]
        )

        train, val, test = deterministic_split(normal_images)

        for image_path in train:

            records.append(
                create_record(
                    dataset="MT",
                    original_label="MT_Free",
                    target_label=None,
                    mapping_status="NORMAL_REFERENCE",
                    source_image_path=image_path,
                    product="magnetic_tile",
                    split="train",
                    role="patchcore_calibration",
                    is_defective=False,
                    provenance=(
                        "Magnetic Tile defect dataset; "
                        "normal MT_Free sample."
                    ),
                )
            )

        for image_path in val:

            records.append(
                create_record(
                    dataset="MT",
                    original_label="MT_Free",
                    target_label=None,
                    mapping_status="NORMAL_REFERENCE",
                    source_image_path=image_path,
                    product="magnetic_tile",
                    split="val",
                    role="anomaly_validation",
                    is_defective=False,
                    provenance=(
                        "Magnetic Tile defect dataset; "
                        "normal validation sample."
                    ),
                )
            )

        for image_path in test:

            records.append(
                create_record(
                    dataset="MT",
                    original_label="MT_Free",
                    target_label=None,
                    mapping_status="NORMAL_REFERENCE",
                    source_image_path=image_path,
                    product="magnetic_tile",
                    split="test",
                    role="anomaly_validation",
                    is_defective=False,
                    provenance=(
                        "Magnetic Tile defect dataset; "
                        "normal test sample."
                    ),
                )
            )

    defect_classes = [
        "MT_Blowhole",
        "MT_Break",
        "MT_Crack",
        "MT_Fray",
        "MT_Uneven",
    ]

    for defect_class in defect_classes:

        defect_dir = MT_ROOT / defect_class / "Imgs"

        if not defect_dir.exists():
            continue

        images = sorted(
            [
                p
                for p in defect_dir.glob("*")
                if p.is_file()
                and p.suffix.lower() in {".jpg", ".jpeg"}
            ]
        )

        if defect_class in MT_DIRECT_MAPPINGS:

            target_label = MT_DIRECT_MAPPINGS[defect_class]
            mapping_status = "DIRECT"

        elif defect_class in MT_POSSIBLE_REVIEW:

            target_label = None
            mapping_status = "POSSIBLE_BUT_REQUIRES_REVIEW"

        else:

            target_label = None
            mapping_status = "NO_MAPPING"

        role = get_defective_role(
            target_label,
            mapping_status,
        )

        val, test = deterministic_binary_split(images)

        for split_name, split_images in [
            ("val", val),
            ("test", test),
        ]:

            for image_path in split_images:

                mask_path = image_path.with_suffix(".png")

                if not mask_path.exists():
                    mask_path = None

                records.append(
                    create_record(
                        dataset="MT",
                        original_label=defect_class,
                        target_label=target_label,
                        mapping_status=mapping_status,
                        source_image_path=image_path,
                        mask_path=mask_path,
                        product="magnetic_tile",
                        split=split_name,
                        role=role,
                        is_defective=True,
                        provenance=(
                            "Magnetic Tile defect dataset; "
                            f"{defect_class} class."
                        ),
                    )
                )


# ============================================================
# NEU-CLS
# ============================================================

def process_neu(records):

    if not NEU_ROOT.exists():
        print(f"[WARN] NEU root not found: {NEU_ROOT}")
        return

    class_names = {
        "crazing",
        "inclusion",
        "patches",
        "pitted_surface",
        "rolled-in_scale",
        "scratches",
    }

    images_by_class = {
        class_name: []
        for class_name in class_names
    }

    for image_path in NEU_ROOT.rglob("*"):

        if not image_path.is_file():
            continue

        if image_path.suffix.lower() not in {
            ".jpg",
            ".jpeg",
            ".png",
        }:
            continue

        prefix = image_path.stem.rsplit("_", 1)[0].lower()

        if prefix in class_names:
            images_by_class[prefix].append(image_path)

    for class_name, images in sorted(
        images_by_class.items()
    ):

        images = sorted(
            images,
            key=lambda p: normalize_path(p),
        )

        if class_name in NEU_DIRECT_MAPPINGS:

            target_label = NEU_DIRECT_MAPPINGS[class_name]
            mapping_status = "DIRECT"

        elif class_name in NEU_POSSIBLE_REVIEW:

            target_label = None
            mapping_status = "POSSIBLE_BUT_REQUIRES_REVIEW"

        else:

            target_label = None
            mapping_status = "NO_MAPPING"

        role = get_defective_role(
            target_label,
            mapping_status,
        )

        train, val, test = deterministic_split(images)

        for split_name, split_images in [
            ("train", train),
            ("val", val),
            ("test", test),
        ]:

            for image_path in split_images:

                records.append(
                    create_record(
                        dataset="NEU-CLS",
                        original_label=class_name,
                        target_label=target_label,
                        mapping_status=mapping_status,
                        source_image_path=image_path,
                        product="steel_surface",
                        split=split_name,
                        role=role,
                        is_defective=True,
                        provenance=(
                            "NEU surface defect classification dataset; "
                            f"{class_name} class."
                        ),
                    )
                )


# ============================================================
# GC10-DET
# ============================================================

def process_gc10(records):

    if not GC10_ROOT.exists():
        print(f"[WARN] GC10 root not found: {GC10_ROOT}")
        return

    image_files = sorted(
        [
            p
            for p in GC10_ROOT.rglob("*")
            if p.is_file()
            and p.suffix.lower() in {
                ".jpg",
                ".jpeg",
                ".png",
            }
        ],
        key=lambda p: normalize_path(p),
    )

    xml_files = {
        p.stem: p
        for p in GC10_ROOT.rglob("*.xml")
        if p.is_file()
    }

    images_by_class = {}

    for image_path in image_files:

        annotation_path = xml_files.get(image_path.stem)
        semantic_name = get_gc10_semantic_class(image_path, annotation_path)

        images_by_class.setdefault(
            semantic_name,
            [],
        ).append(
            (
                image_path,
                annotation_path,
            )
        )

    for class_name, items in sorted(
        images_by_class.items()
    ):

        items = sorted(
            items,
            key=lambda x: normalize_path(x[0]),
        )

        semantic_name = class_name

        if semantic_name in GC10_DIRECT_MAPPINGS:

            target_label = GC10_DIRECT_MAPPINGS[semantic_name]
            mapping_status = "DIRECT"

        elif semantic_name in GC10_POSSIBLE_REVIEW:

            target_label = None
            mapping_status = "POSSIBLE_BUT_REQUIRES_REVIEW"

        elif semantic_name in GC10_NOT_EQUIVALENT:

            target_label = None
            mapping_status = "NOT_EQUIVALENT"

        else:

            target_label = None
            mapping_status = "NO_MAPPING"

        role = get_defective_role(
            target_label,
            mapping_status,
        )

        train, val, test = deterministic_split(items)

        for split_name, split_items in [
            ("train", train),
            ("val", val),
            ("test", test),
        ]:

            for image_path, annotation_path in split_items:

                records.append(
                    create_record(
                        dataset="GC10-DET",
                        original_label=semantic_name,
                        target_label=target_label,
                        mapping_status=mapping_status,
                        source_image_path=image_path,
                        annotation_path=annotation_path,
                        product="steel_surface",
                        split=split_name,
                        role=role,
                        is_defective=True,
                        provenance=(
                            "GC10-DET industrial steel defect dataset; "
                            "original dataset label retained."
                        ),
                    )
                )


# ============================================================
# INTERNAL VALIDATION
# ============================================================

def validate_records(records):

    errors = []

    required_fields = {
        "sample_id",
        "dataset",
        "original_label",
        "target_label",
        "mapping_status",
        "source_image_path",
        "mask_path",
        "annotation_path",
        "product",
        "split",
        "role",
        "is_defective",
        "provenance",
    }

    allowed_splits = {
        "train",
        "val",
        "test",
        "none",
    }

    allowed_roles = {
        "patchcore_calibration",
        "anomaly_validation",
        "classifier_source",
        "classifier_review",
        "unknown_defect_source",
        "unclassified",
    }

    allowed_mapping_statuses = {
        "DIRECT",
        "POSSIBLE_BUT_REQUIRES_REVIEW",
        "NOT_EQUIVALENT",
        "NORMAL_REFERENCE",
        "NO_MAPPING",
    }

    sample_ids = set()
    source_paths = set()

    for index, record in enumerate(
        records,
        start=1,
    ):

        missing = (
            required_fields
            - set(record.keys())
        )

        if missing:
            errors.append(
                f"Record {index}: missing fields "
                f"{sorted(missing)}"
            )

        sample_id = record.get("sample_id")
        source_path = record.get(
            "source_image_path"
        )

        if sample_id in sample_ids:

            errors.append(
                f"Record {index}: duplicate sample_id: "
                f"{sample_id}"
            )

        if source_path in source_paths:

            errors.append(
                f"Record {index}: duplicate "
                f"source_image_path: {source_path}"
            )

        sample_ids.add(sample_id)
        source_paths.add(source_path)

        if record.get("split") not in allowed_splits:

            errors.append(
                f"Record {index}: invalid split: "
                f"{record.get('split')}"
            )

        if record.get("role") not in allowed_roles:

            errors.append(
                f"Record {index}: invalid role: "
                f"{record.get('role')}"
            )

        if (
            record.get("mapping_status")
            not in allowed_mapping_statuses
        ):

            errors.append(
                f"Record {index}: invalid mapping_status: "
                f"{record.get('mapping_status')}"
            )

        target_label = record.get(
            "target_label"
        )

        if (
            target_label is not None
            and target_label not in TARGET_LABELS
        ):

            errors.append(
                f"Record {index}: invalid target_label: "
                f"{target_label}"
            )

        if not isinstance(
            record.get("is_defective"),
            bool,
        ):

            errors.append(
                f"Record {index}: "
                f"is_defective must be boolean"
            )

        if (
            record.get("mapping_status")
            == "DIRECT"
            and target_label is None
        ):

            errors.append(
                f"Record {index}: DIRECT mapping "
                f"has no target_label"
            )

        if (
            record.get("mapping_status")
            == "NORMAL_REFERENCE"
        ):

            if target_label is not None:

                errors.append(
                    f"Record {index}: NORMAL_REFERENCE "
                    f"has target_label"
                )

            if record.get("is_defective") is not False:

                errors.append(
                    f"Record {index}: NORMAL_REFERENCE "
                    f"must be non-defective"
                )

    if errors:

        print(
            "\nMANIFEST BUILD VALIDATION FAILED"
        )

        for error in errors:
            print(f"  - {error}")

        raise RuntimeError(
            "Manifest validation failed with "
            f"{len(errors)} error(s)."
        )

    print(
        "Manifest internal validation: PASS"
    )


# ============================================================
# SUMMARY
# ============================================================

def print_summary(records):

    from collections import Counter

    print("\n" + "=" * 70)
    print(
        "DEFECTVISION DATASET MANIFEST SUMMARY"
    )
    print("=" * 70)

    print(
        f"Total records: {len(records)}"
    )

    dataset_counts = Counter(
        r["dataset"]
        for r in records
    )

    print("\nBy dataset:")

    for key, value in sorted(
        dataset_counts.items()
    ):
        print(
            f"  {key}: {value}"
        )

    role_counts = Counter(
        r["role"]
        for r in records
    )

    print("\nBy role:")

    for key, value in sorted(
        role_counts.items()
    ):
        print(
            f"  {key}: {value}"
        )

    split_counts = Counter(
        r["split"]
        for r in records
    )

    print("\nBy split:")

    for key, value in sorted(
        split_counts.items()
    ):
        print(
            f"  {key}: {value}"
        )

    target_counts = Counter(
        str(r["target_label"])
        for r in records
    )

    print("\nBy target label:")

    for key, value in sorted(
        target_counts.items()
    ):
        print(
            f"  {key}: {value}"
        )

    mapping_counts = Counter(
        r["mapping_status"]
        for r in records
    )

    print("\nBy mapping status:")

    for key, value in sorted(
        mapping_counts.items()
    ):
        print(
            f"  {key}: {value}"
        )

    print("=" * 70)


# ============================================================
# MAIN
# ============================================================

def main():

    print(
        "Building DefectVision dataset manifest..."
    )

    print(
        f"Sources: {SOURCES_ROOT}"
    )

    print(
        f"Output:  {MANIFEST_OUTPUT}"
    )

    records = []

    process_mvtec(records)
    process_mt(records)
    process_neu(records)
    process_gc10(records)

    if not records:

        raise RuntimeError(
            "No dataset records were generated. "
            "Check source paths."
        )

    validate_records(records)

    records.sort(
        key=lambda r: (
            r["dataset"],
            r["product"],
            r["split"],
            r["original_label"],
            r["source_image_path"],
        )
    )

    MANIFEST_OUTPUT.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with MANIFEST_OUTPUT.open(
        "w",
        encoding="utf-8",
        newline="\n",
    ) as f:

        for record in records:

            f.write(
                json.dumps(
                    record,
                    ensure_ascii=False,
                    sort_keys=True,
                )
                + "\n"
            )

    print_summary(records)

    print(
        "\nManifest written successfully:"
    )

    print(MANIFEST_OUTPUT)


if __name__ == "__main__":
    main()