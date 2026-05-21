from pathlib import Path
import shutil
import kagglehub

DATASET_HANDLE = "yimengfuyao/detectiumfire"

PROJECT_ROOT = Path(__file__).resolve().parent
DEST_FIRE = PROJECT_ROOT / "src" / "detectium_fire" / "fire"
DEST_NON_FIRE = PROJECT_ROOT / "src" / "detectium_fire" / "non_fire"


def copy_files_only(src: Path, dst: Path):
    """Copy only files (not nested folders) into dst."""
    dst.mkdir(parents=True, exist_ok=True)

    for file in src.glob("*"):
        if file.is_file():
            shutil.copy2(file, dst / file.name)


def main():
    dataset_root = Path(kagglehub.dataset_download(DATASET_HANDLE))
    print(f"Downloaded to: {dataset_root}")

    src_fire = dataset_root / "real_video" / "real_video" / "fire"
    src_non_fire = dataset_root / "real_video" / "real_video" / "non_fire"

    if not src_fire.exists() or not src_non_fire.exists():
        raise Exception("Dataset structure not found. Check paths.")

    copy_files_only(src_fire, DEST_FIRE)
    copy_files_only(src_non_fire, DEST_NON_FIRE)


if __name__ == "__main__":
    main()