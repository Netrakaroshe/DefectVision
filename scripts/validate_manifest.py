from pathlib import Path
import hashlib
import json
from collections import Counter, defaultdict


MANIFEST_PATH = Path(
    r"D:\DefectVision\manifests\dataset_manifest.jsonl"
)

SOURCES_ROOT = Path(
    r"D:\datasets\defect_dataset_workspace\sources"
)


# ============================================================
# APPROVED DATASET ROOTS
# ============================================================

APPROVED_ROOTS = {
    "MVTecAD": SOURCES_ROOT / "MVTecAD",
    "MT": SOURCES_ROOT / "MT",
    "NEU-CLS": SOURCES_ROOT / "NEU-CLS",
    "GC10-DET": SOURCES_ROOT / "GC10-DET",
}


# ============================================================
# FORBIDDEN / CONTAMINATED DATA
# ============================================================

FORBIDDEN_ROOTS = [
    SOURCES_ROOT / "defect-categories",
    SOURCES_ROOT / "Surface",
]


# ============================================================
# VALID VALUES
# ============================================================

VALID_TARGET_LABELS = {
    "Scratch",
    "Crack",
    "Discoloration",
    None,
}

VALID_SPLITS = {
    "train",
    "val",
    "test",
    "none",
}

VALID_ROLES = {
    "patchcore_calibration",
    "anomaly_validation",
    "classifier_source",
    "classifier_review",
    "unknown_defect_source",
    "unclassified",
}

VALID_MAPPING_STATUSES = {
    "DIRECT",
    "POSSIBLE_BUT_REQUIRES_REVIEW",
    "NOT_EQUIVALENT",
    "NORMAL_REFERENCE",
    "NO_MAPPING",
}

REQUIRED_FIELDS = {
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


# ============================================================
# HELPERS
# ============================================================

def normalize_path(path):
    return str(Path(path).resolve()).replace("\\", "/").lower()


def path_is_inside(path, root):
    try:
        Path(path).resolve().relative_to(
            Path(root).resolve()
        )
        return True
    except ValueError:
        return False


def file_sha256(path):
    sha = hashlib.sha256()

    with open(path, "rb") as f:
        for chunk in iter(
            lambda: f.read(1024 * 1024),
            b"",
        ):
            sha.update(chunk)

    return sha.hexdigest()


def load_manifest():

    if not MANIFEST_PATH.exists():
        raise FileNotFoundError(
            f"Manifest not found:\n{MANIFEST_PATH}"
        )

    records = []

    with MANIFEST_PATH.open(
        "r",
        encoding="utf-8",
    ) as f:

        for line_number, line in enumerate(
            f,
            start=1,
        ):

            line = line.strip()

            if not line:
                continue

            try:
                record = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(
                    f"Invalid JSON at line {line_number}: "
                    f"{exc}"
                )

            record["_line_number"] = line_number

            records.append(record)

    return records


# ============================================================
# VALIDATION
# ============================================================

def validate_required_fields(records, errors):

    for record in records:

        missing = (
            REQUIRED_FIELDS
            - set(record.keys())
        )

        if missing:
            errors.append(
                f"Line {record['_line_number']}: "
                f"missing fields {sorted(missing)}"
            )


def validate_unique_ids_and_paths(
    records,
    errors,
):

    sample_ids = {}
    source_paths = {}

    for record in records:

        sample_id = record.get(
            "sample_id"
        )

        source_path = normalize_path(
            record.get(
                "source_image_path",
                "",
            )
        )

        if sample_id in sample_ids:

            errors.append(
                f"Duplicate sample_id: "
                f"{sample_id} "
                f"(lines "
                f"{sample_ids[sample_id]} and "
                f"{record['_line_number']})"
            )

        else:
            sample_ids[sample_id] = (
                record["_line_number"]
            )

        if source_path in source_paths:

            errors.append(
                f"Duplicate source_image_path: "
                f"{source_path} "
                f"(lines "
                f"{source_paths[source_path]} and "
                f"{record['_line_number']})"
            )

        else:
            source_paths[source_path] = (
                record["_line_number"]
            )


def validate_basic_values(
    records,
    errors,
    warnings,
):

    for record in records:

        line = record["_line_number"]

        dataset = record.get("dataset")
        split = record.get("split")
        role = record.get("role")
        mapping_status = record.get(
            "mapping_status"
        )
        target_label = record.get(
            "target_label"
        )
        is_defective = record.get(
            "is_defective"
        )

        if dataset not in APPROVED_ROOTS:

            errors.append(
                f"Line {line}: unapproved dataset: "
                f"{dataset}"
            )

        if split not in VALID_SPLITS:

            errors.append(
                f"Line {line}: invalid split: "
                f"{split}"
            )

        if role not in VALID_ROLES:

            errors.append(
                f"Line {line}: invalid role: "
                f"{role}"
            )

        if (
            mapping_status
            not in VALID_MAPPING_STATUSES
        ):

            errors.append(
                f"Line {line}: invalid mapping_status: "
                f"{mapping_status}"
            )

        if (
            target_label
            not in VALID_TARGET_LABELS
        ):

            errors.append(
                f"Line {line}: invalid target_label: "
                f"{target_label}"
            )

        if not isinstance(
            is_defective,
            bool,
        ):

            errors.append(
                f"Line {line}: "
                f"is_defective must be boolean"
            )

        # ----------------------------------------------------
        # Mapping consistency
        # ----------------------------------------------------

        if mapping_status == "DIRECT":

            if target_label is None:

                errors.append(
                    f"Line {line}: DIRECT mapping "
                    f"must have target_label"
                )

            if not is_defective:

                errors.append(
                    f"Line {line}: DIRECT mapping "
                    f"cannot be normal"
                )

        if mapping_status == "NORMAL_REFERENCE":

            if target_label is not None:

                errors.append(
                    f"Line {line}: NORMAL_REFERENCE "
                    f"must have target_label=None"
                )

            if is_defective:

                errors.append(
                    f"Line {line}: NORMAL_REFERENCE "
                    f"cannot be defective"
                )

        if (
            mapping_status
            in {
                "POSSIBLE_BUT_REQUIRES_REVIEW",
                "NOT_EQUIVALENT",
                "NO_MAPPING",
            }
            and target_label is not None
        ):

            errors.append(
                f"Line {line}: "
                f"{mapping_status} should not "
                f"have target_label"
            )

        # ----------------------------------------------------
        # Role consistency
        # ----------------------------------------------------

        if (
            target_label is not None
            and role != "classifier_source"
        ):

            errors.append(
                f"Line {line}: mapped defect "
                f"must have role=classifier_source"
            )

        if (
            mapping_status
            in {
                "POSSIBLE_BUT_REQUIRES_REVIEW",
                "NOT_EQUIVALENT",
            }
            and role != "classifier_review"
        ):

            errors.append(
                f"Line {line}: review mapping "
                f"must have role=classifier_review"
            )

        if (
            mapping_status == "NO_MAPPING"
            and role != "unknown_defect_source"
        ):

            errors.append(
                f"Line {line}: NO_MAPPING "
                f"must have role=unknown_defect_source"
            )

        if (
            mapping_status == "NORMAL_REFERENCE"
            and role
            not in {
                "patchcore_calibration",
                "anomaly_validation",
            }
        ):

            errors.append(
                f"Line {line}: normal reference "
                f"has invalid role={role}"
            )


def validate_source_paths(
    records,
    errors,
    warnings,
):

    for record in records:

        line = record["_line_number"]

        source_path = Path(
            record.get(
                "source_image_path",
                "",
            )
        )

        if not source_path.exists():

            errors.append(
                f"Line {line}: source image does not exist: "
                f"{source_path}"
            )

            continue

        dataset = record.get(
            "dataset"
        )

        approved_root = APPROVED_ROOTS.get(
            dataset
        )

        if approved_root is not None:

            if not path_is_inside(
                source_path,
                approved_root,
            ):

                errors.append(
                    f"Line {line}: source image is "
                    f"outside approved root: "
                    f"{source_path}"
                )

        for forbidden_root in FORBIDDEN_ROOTS:

            if path_is_inside(
                source_path,
                forbidden_root,
            ):

                errors.append(
                    f"Line {line}: FORBIDDEN source "
                    f"root detected: "
                    f"{source_path}"
                )


def validate_masks_and_annotations(
    records,
    errors,
    warnings,
):

    for record in records:

        line = record["_line_number"]

        source_path = Path(
            record["source_image_path"]
        )

        dataset = record["dataset"]

        mask_path = record.get(
            "mask_path"
        )

        annotation_path = record.get(
            "annotation_path"
        )

        # ----------------------------------------------------
        # MVTec defective images should have masks.
        # ----------------------------------------------------

        if (
            dataset == "MVTecAD"
            and record["is_defective"]
        ):

            if mask_path is None:

                warnings.append(
                    f"Line {line}: MVTec defective "
                    f"image has no mask path: "
                    f"{source_path}"
                )

            elif not Path(mask_path).exists():

                errors.append(
                    f"Line {line}: MVTec mask does "
                    f"not exist: {mask_path}"
                )

        # ----------------------------------------------------
        # MT defective images should have masks.
        # ----------------------------------------------------

        if (
            dataset == "MT"
            and record["is_defective"]
        ):

            if mask_path is None:

                warnings.append(
                    f"Line {line}: MT defective "
                    f"image has no mask path: "
                    f"{source_path}"
                )

            elif not Path(mask_path).exists():

                errors.append(
                    f"Line {line}: MT mask does "
                    f"not exist: {mask_path}"
                )

        # ----------------------------------------------------
        # GC10 defective images should normally have XML.
        # ----------------------------------------------------

        if dataset == "GC10-DET":

            if annotation_path is None:

                warnings.append(
                    f"Line {line}: GC10 image has "
                    f"no XML annotation: "
                    f"{source_path}"
                )

            elif not Path(
                annotation_path
            ).exists():

                errors.append(
                    f"Line {line}: GC10 annotation "
                    f"does not exist: "
                    f"{annotation_path}"
                )


def validate_split_overlap(
    records,
    errors,
):

    paths_by_split = defaultdict(
        dict
    )

    for record in records:

        path = normalize_path(
            record["source_image_path"]
        )

        split = record["split"]

        if path in paths_by_split[split]:
            continue

        paths_by_split[split][path] = (
            record["_line_number"]
        )

    all_paths = {}

    for split, path_map in paths_by_split.items():

        for path, line in path_map.items():

            if path in all_paths:

                previous_split, previous_line = (
                    all_paths[path]
                )

                if previous_split != split:

                    errors.append(
                        f"Source path appears in "
                        f"multiple splits: "
                        f"{path} "
                        f"({previous_split}/line "
                        f"{previous_line}, "
                        f"{split}/line {line})"
                    )

            else:

                all_paths[path] = (
                    split,
                    line,
                )


def validate_mvtec_structure(
    records,
    errors,
):

    for record in records:

        if record["dataset"] != "MVTecAD":
            continue

        source = Path(
            record["source_image_path"]
        )

        parts = source.parts

        # Find the MVTecAD component.
        try:
            mvtec_index = next(
                i
                for i, part in enumerate(parts)
                if part.lower()
                == "mvtecad"
            )
        except StopIteration:
            continue

        remaining = parts[
            mvtec_index + 1:
        ]

        if len(remaining) < 3:
            errors.append(
                f"Line {record['_line_number']}: "
                f"unexpected MVTec path structure: "
                f"{source}"
            )
            continue

        category = remaining[0]
        partition = remaining[1]

        # Official MVTec training normal data.
        if (
            partition == "train"
            and record["is_defective"]
        ):

            errors.append(
                f"Line {record['_line_number']}: "
                f"MVTec train sample marked defective: "
                f"{source}"
            )

        # MVTec train should only be normal calibration.
        if partition == "train":

            if (
                record["role"]
                != "patchcore_calibration"
            ):

                errors.append(
                    f"Line {record['_line_number']}: "
                    f"MVTec train sample must be "
                    f"patchcore_calibration: "
                    f"{source}"
                )

            if record["split"] != "train":

                errors.append(
                    f"Line {record['_line_number']}: "
                    f"MVTec train sample must have "
                    f"split=train: "
                    f"{source}"
                )

        # MVTec test images must have split=test.
        if partition == "test":

            if record["split"] != "test":

                errors.append(
                    f"Line {record['_line_number']}: "
                    f"MVTec test sample must have "
                    f"split=test: "
                    f"{source}"
                )


def validate_normal_reference_roles(
    records,
    errors,
):

    for record in records:

        if (
            record["mapping_status"]
            != "NORMAL_REFERENCE"
        ):
            continue

        if not record["is_defective"]:

            if record["role"] not in {
                "patchcore_calibration",
                "anomaly_validation",
            }:

                errors.append(
                    f"Line {record['_line_number']}: "
                    f"normal sample has invalid role: "
                    f"{record['role']}"
                )


def calculate_content_hashes(
    records,
    warnings,
):

    """
    Detect exact duplicate image content.

    This is intentionally a warning rather than an error.

    Exact duplicate content can exist legitimately inside
    a dataset, but duplicates must never cross modeling
    splits.

    """

    hashes = defaultdict(list)

    for record in records:

        source = Path(
            record["source_image_path"]
        )

        if not source.exists():
            continue

        try:
            digest = file_sha256(source)

            hashes[digest].append(
                (
                    record["dataset"],
                    record["split"],
                    record["_line_number"],
                    str(source),
                )
            )

        except OSError as exc:

            warnings.append(
                f"Could not hash {source}: {exc}"
            )

    for digest, occurrences in hashes.items():

        if len(occurrences) <= 1:
            continue

        splits = {
            occurrence[1]
            for occurrence in occurrences
        }

        if len(splits) > 1:

            warnings.append(
                "Exact duplicate image content "
                "appears across multiple splits:\n"
                + "\n".join(
                    f"    dataset={dataset}, "
                    f"split={split}, "
                    f"line={line}, "
                    f"path={path}"
                    for (
                        dataset,
                        split,
                        line,
                        path,
                    ) in occurrences
                )
            )


# ============================================================
# SUMMARY
# ============================================================

def print_summary(records):

    print("\n" + "=" * 70)
    print(
        "DEFECTVISION MANIFEST VALIDATION SUMMARY"
    )
    print("=" * 70)

    print(
        f"Total records: {len(records)}"
    )

    counters = {
        "dataset": Counter(),
        "role": Counter(),
        "split": Counter(),
        "target": Counter(),
        "mapping": Counter(),
    }

    for record in records:

        counters["dataset"][
            record["dataset"]
        ] += 1

        counters["role"][
            record["role"]
        ] += 1

        counters["split"][
            record["split"]
        ] += 1

        counters["target"][
            str(record["target_label"])
        ] += 1

        counters["mapping"][
            record["mapping_status"]
        ] += 1

    for title, counter in [
        ("By dataset", counters["dataset"]),
        ("By role", counters["role"]),
        ("By split", counters["split"]),
        ("By target label", counters["target"]),
        ("By mapping status", counters["mapping"]),
    ]:

        print(f"\n{title}:")

        for key, value in sorted(
            counter.items()
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
        "Validating DefectVision dataset manifest..."
    )

    print(
        f"Manifest: {MANIFEST_PATH}"
    )

    records = load_manifest()

    if not records:

        raise RuntimeError(
            "Manifest contains no records."
        )

    errors = []
    warnings = []

    validate_required_fields(
        records,
        errors,
    )

    validate_unique_ids_and_paths(
        records,
        errors,
    )

    validate_basic_values(
        records,
        errors,
        warnings,
    )

    validate_source_paths(
        records,
        errors,
        warnings,
    )

    validate_masks_and_annotations(
        records,
        errors,
        warnings,
    )

    validate_split_overlap(
        records,
        errors,
    )

    validate_mvtec_structure(
        records,
        errors,
    )

    validate_normal_reference_roles(
        records,
        errors,
    )

    calculate_content_hashes(
        records,
        warnings,
    )

    print_summary(records)

    print(
        f"\nErrors: {len(errors)}"
    )

    print(
        f"Warnings: {len(warnings)}"
    )

    if warnings:

        print("\nWARNINGS:")

        for warning in warnings:
            print(f"  - {warning}")

    if errors:

        print("\nERRORS:")

        for error in errors:
            print(f"  - {error}")

        print(
            "\nMANIFEST VALIDATION: FAIL"
        )

        raise SystemExit(1)

    print(
        "\nMANIFEST VALIDATION: PASS"
    )


if __name__ == "__main__":
    main()