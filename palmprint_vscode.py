from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
import zipfile
from pathlib import Path
from typing import Iterable

try:
    import cv2  # type: ignore
    import matplotlib.pyplot as plt  # type: ignore
except Exception:
    cv2 = None
    plt = None


def run_command(command: list[str], cwd: Path | None = None) -> None:
    """Rulează o comandă și afișează output-ul în terminal."""
    print(f"\n[CMD] {' '.join(command)}")
    subprocess.run(command, cwd=str(cwd) if cwd else None, check=True)


# ---------------------------
# FOLOSIM DOAR FOLDER LOCAL
# ---------------------------
def ensure_repo(base_path: Path) -> Path:
    """Folosește folderul palmprint din același director cu scriptul."""

    possible_dirs = [
        base_path / "palmprint",
        base_path / "Palmprint",
        base_path
    ]

    for repo_path in possible_dirs:
        if repo_path.exists() and any(repo_path.iterdir()):
            print(f"[INFO] Folosesc folderul local: {repo_path}")
            return repo_path

    raise FileNotFoundError(
        "Nu am găsit folderul palmprint lângă script!"
    )


def extract_zip(zip_path: Path, destination: Path) -> None:
    """Dezarhivează dataset-ul într-un director local."""
    if not zip_path.exists():
        raise FileNotFoundError(f"Nu am găsit arhiva: {zip_path}")

    destination.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(zip_path, "r") as zip_ref:
        zip_ref.extractall(destination)
    print(f"[INFO] Arhiva a fost extrasă în: {destination}")


def find_images(base_dir: Path, extensions: Iterable[str] = (".jpg", ".png", ".bmp", ".jpeg")) -> list[Path]:
    """Caută imagini recursiv."""
    image_files: list[Path] = []
    for ext in extensions:
        image_files.extend(base_dir.rglob(f"*{ext}"))
    return sorted(image_files)


def print_structure(base_dir: Path, max_items: int = 50) -> None:
    """Afișează câteva fișiere găsite în director."""
    if not base_dir.exists():
        print(f"[WARN] Directorul nu există: {base_dir}")
        return

    print(f"\n[INFO] Conținut în: {base_dir}")
    items = list(base_dir.rglob("*"))[:max_items]
    for item in items:
        print(item)


def show_sample_images(base_dir: Path, sample_count: int = 4) -> None:
    """Afișează câteva imagini de test."""
    if cv2 is None or plt is None:
        print("[WARN] cv2/matplotlib nu sunt instalate.")
        return

    image_files = find_images(base_dir)[:sample_count]
    if not image_files:
        print(f"[WARN] Nu am găsit imagini în: {base_dir}")
        return

    plt.figure(figsize=(12, 10))
    for i, img_path in enumerate(image_files, start=1):
        img = cv2.imread(str(img_path), cv2.IMREAD_GRAYSCALE)
        if img is None:
            continue
        plt.subplot(2, 2, i)
        plt.imshow(img, cmap="gray")
        plt.title(img_path.name)
        plt.axis("off")

    plt.tight_layout()
    plt.show()


def normalize_dataset_layout(extracted_dir: Path, repo_data_dir: Path) -> None:
    """Normalizează dataset-ul."""
    repo_data_dir.mkdir(parents=True, exist_ok=True)

    palmprint_dir = extracted_dir / "Palmprint"
    training_src = palmprint_dir / "training"
    testing_src = palmprint_dir / "testing"

    if training_src.exists() and testing_src.exists():
        shutil.copytree(training_src, repo_data_dir / "training", dirs_exist_ok=True)
        shutil.copytree(testing_src, repo_data_dir / "testing", dirs_exist_ok=True)
        print("[INFO] Dataset organizat (training/testing)")
        return

    for item in extracted_dir.iterdir():
        destination = repo_data_dir / item.name
        if item.is_dir():
            shutil.copytree(item, destination, dirs_exist_ok=True)
        else:
            shutil.copy2(item, destination)

    print("[INFO] Dataset copiat direct")


def run_repo_script(repo_path: Path, script_name: str) -> None:
    """Rulează scriptul principal."""
    script_path = repo_path / script_name
    if not script_path.exists():
        raise FileNotFoundError(f"Nu există scriptul: {script_path}")

    run_command([sys.executable, str(script_path)], cwd=repo_path)


# ---------------------------
# CLI
# ---------------------------
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()

    parser.add_argument("--zip", type=Path, default=None)
    parser.add_argument("--script", type=str, default="SIFT_DIP.py")
    parser.add_argument("--show-samples", action="store_true")
    parser.add_argument("--skip-run", action="store_true")

    return parser.parse_args()


def main() -> None:
    args = parse_args()

    base_path = Path(__file__).parent
    extracted_dir = base_path / "palmprint_data"

    print(f"[INFO] Folder script: {base_path}")

    repo_path = ensure_repo(base_path)

    if args.zip is not None:
        extract_zip(args.zip, extracted_dir)
        print_structure(extracted_dir)
        normalize_dataset_layout(extracted_dir, repo_path / "data")

    image_dir = extracted_dir if extracted_dir.exists() else repo_path
    images = find_images(image_dir)
    print(f"[INFO] Imagini găsite: {len(images)}")

    if args.show_samples:
        show_sample_images(image_dir)

    if not args.skip_run:
        run_repo_script(repo_path, args.script)
    else:
        print("[INFO] Skip run activat")


if __name__ == "__main__":
    main()