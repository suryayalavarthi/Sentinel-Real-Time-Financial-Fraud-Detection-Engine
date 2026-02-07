from __future__ import annotations

import os
import re
import shutil
from typing import Iterable, List


ROOT_DIR = os.path.dirname(os.path.abspath(__file__))


def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def move_file(src: str, dst: str) -> bool:
    if not os.path.exists(src):
        return False
    if os.path.exists(dst):
        return False
    ensure_dir(os.path.dirname(dst))
    shutil.move(src, dst)
    return True


def move_dir_contents(src_dir: str, dst_dir: str) -> None:
    if not os.path.isdir(src_dir):
        return
    ensure_dir(dst_dir)
    for name in os.listdir(src_dir):
        src_path = os.path.join(src_dir, name)
        dst_path = os.path.join(dst_dir, name)
        if os.path.isdir(src_path):
            move_dir_contents(src_path, dst_path)
            if os.path.isdir(src_path) and not os.listdir(src_path):
                os.rmdir(src_path)
        else:
            move_file(src_path, dst_path)
    if os.path.isdir(src_dir) and not os.listdir(src_dir):
        os.rmdir(src_dir)


def print_tree(root: str, max_depth: int = 2) -> None:
    print(f"\n{root}")
    for current_root, dirs, files in os.walk(root):
        depth = current_root.replace(root, "").count(os.sep)
        if depth > max_depth:
            continue
        indent = "  " * depth
        folder_name = os.path.basename(current_root) or current_root
        print(f"{indent}{folder_name}/")
        if depth == max_depth:
            continue
        for filename in sorted(files):
            print(f"{indent}  {filename}")


def update_paths_in_file(path: str) -> bool:
    with open(path, "r", encoding="utf-8") as f:
        original = f.read()

    updated = original
    replacements = [
        (r"(['\"])\\./data/ieee-fraud-detection/", r"\\1../data/raw/"),
        (r"(['\"])data/ieee-fraud-detection/", r"\\1../data/raw/"),
        (r"(['\"])\\./data/", r"\\1../data/processed/"),
        (r"(['\"])data/", r"\\1../data/processed/"),
        (r"(['\"])\\./models/", r"\\1../models/"),
        (r"(['\"])models/", r"\\1../models/"),
        (r"(['\"])\\./reports/", r"\\1../reports/"),
        (r"(['\"])reports/", r"\\1../reports/"),
        (r"(['\"])\\./logs/", r"\\1../logs/"),
        (r"(['\"])logs/", r"\\1../logs/"),
    ]
    for pattern, replacement in replacements:
        updated = re.sub(pattern, replacement, updated)

    if updated != original:
        with open(path, "w", encoding="utf-8") as f:
            f.write(updated)
        return True

    return False


def iter_python_files(directories: Iterable[str]) -> List[str]:
    files: List[str] = []
    for directory in directories:
        if not os.path.isdir(directory):
            continue
        for name in os.listdir(directory):
            if name.endswith(".py"):
                files.append(os.path.join(directory, name))
    return files


def main() -> None:
    print("BEFORE REORGANIZATION")
    print_tree(ROOT_DIR, max_depth=2)

    # Target directories
    src_dir = os.path.join(ROOT_DIR, "src")
    data_dir = os.path.join(ROOT_DIR, "data")
    raw_dir = os.path.join(data_dir, "raw")
    processed_dir = os.path.join(data_dir, "processed")
    models_dir = os.path.join(ROOT_DIR, "models")
    scripts_dir = os.path.join(ROOT_DIR, "scripts")
    tests_dir = os.path.join(ROOT_DIR, "tests")
    docs_dir = os.path.join(ROOT_DIR, "docs")
    reports_dir = os.path.join(ROOT_DIR, "reports")
    logs_dir = os.path.join(ROOT_DIR, "logs")

    for path in [
        src_dir,
        raw_dir,
        processed_dir,
        models_dir,
        scripts_dir,
        tests_dir,
        docs_dir,
        reports_dir,
        logs_dir,
    ]:
        ensure_dir(path)

    # Move core pipeline scripts
    src_files = [
        "data_ingestion.py",
        "feature_engineering.py",
        "model_training.py",
        "model_evaluation.py",
        "run_pipeline.py",
        "pipeline_architecture.py",
    ]
    for filename in src_files:
        move_file(os.path.join(ROOT_DIR, filename), os.path.join(src_dir, filename))

    # Move supplemental scripts
    script_files = ["quick_pipeline.py", "interpretability.py", "monitoring.py"]
    for filename in script_files:
        move_file(os.path.join(ROOT_DIR, filename), os.path.join(scripts_dir, filename))

    # Move documentation
    doc_files = [
        "README.md",
        "DEPLOYMENT_NOTES.md",
        "FEATURE_ENGINEERING.md",
        "MLOPS_SUMMARY.md",
        "README_UPDATES.md",
    ]
    for filename in doc_files:
        move_file(os.path.join(ROOT_DIR, filename), os.path.join(docs_dir, filename))

    # Move tests (keep structure)
    move_dir_contents(os.path.join(ROOT_DIR, "tests"), tests_dir)

    # Move reports and logs if they exist at root
    move_dir_contents(os.path.join(ROOT_DIR, "reports"), reports_dir)
    move_dir_contents(os.path.join(ROOT_DIR, "logs"), logs_dir)

    # Data: move raw CSVs to data/raw
    raw_candidates = [
        "train_transaction.csv",
        "train_identity.csv",
        "test_transaction.csv",
        "test_identity.csv",
        "sample_submission.csv",
    ]
    legacy_raw_dir = os.path.join(data_dir, "ieee-fraud-detection")
    legacy_root_dir = os.path.join(ROOT_DIR, "ieee-fraud-detection")
    move_dir_contents(legacy_raw_dir, raw_dir)
    move_dir_contents(os.path.join(legacy_root_dir, "raw"), raw_dir)
    move_dir_contents(os.path.join(legacy_root_dir, "processed"), processed_dir)
    for filename in raw_candidates:
        move_file(os.path.join(data_dir, filename), os.path.join(raw_dir, filename))
        move_file(os.path.join(legacy_root_dir, filename), os.path.join(raw_dir, filename))

    # Processed data: move engineered outputs
    if os.path.isdir(data_dir):
        for name in os.listdir(data_dir):
            src_path = os.path.join(data_dir, name)
            if os.path.isdir(src_path):
                continue
            if name in raw_candidates:
                continue
            if name.endswith(".pkl") or name.endswith(".csv"):
                move_file(src_path, os.path.join(processed_dir, name))

    # Cleanup legacy root data folder if empty
    if os.path.isdir(legacy_root_dir) and not os.listdir(legacy_root_dir):
        os.rmdir(legacy_root_dir)

    # Update paths in src/ and scripts/
    updated_files: List[str] = []
    for py_path in iter_python_files([src_dir, scripts_dir]):
        if update_paths_in_file(py_path):
            updated_files.append(py_path)

    print("\nUPDATED PATHS")
    for path in sorted(updated_files):
        rel_path = os.path.relpath(path, ROOT_DIR)
        print(f"  - {rel_path}")

    print("\nAFTER REORGANIZATION")
    print_tree(ROOT_DIR, max_depth=2)


if __name__ == "__main__":
    main()
