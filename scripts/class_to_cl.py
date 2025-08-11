from pathlib import Path
from PIL import Image
import shutil
import sys

def standardize_to_jpeg(input_root: Path, output_root: Path, quality: int = 90):
    """Copy/convert images from class folders into a flat JPEG dataset.

    Args:
        input_root  (Path): root dir of original class folders
        output_root (Path): where converted JPEGs will be saved
        quality     (int) : JPEG quality (1–95)
    """
    SUPPORTED_EXT = {".jpg", ".jpeg", ".png", ".webp", ".avif"}
    output_root.mkdir(parents=True, exist_ok=True)

    duplicates = 0
    for class_dir in filter(Path.is_dir, input_root.iterdir()):
        class_name = class_dir.name
        for img_path in class_dir.glob("*"):
            if img_path.suffix.lower() not in SUPPORTED_EXT:
                print(f"Skipping unsupported file: {img_path}")
                continue

            # Build new filename: <class>_<original_stem>.jpg
            new_stem = f"{class_name}_{img_path.stem}"
            new_name = f"{new_stem}.jpg"
            dest_path = output_root / new_name

            # Avoid accidental overwrite (rare if stems collide)
            while dest_path.exists():
                duplicates += 1
                dest_path = output_root / f"{new_stem}_{duplicates}.jpg"

            try:
                # Use Pillow to open & save as JPEG
                with Image.open(img_path) as im:
                    # Convert RGBA or P to RGB first
                    if im.mode in ("RGBA", "P"):
                        im = im.convert("RGB")
                    im.save(dest_path, "JPEG", quality=quality, optimize=True)
            except Exception as e:
                print(f"⚠️  Failed on {img_path}: {e}")

    print(f"Done. Images saved in {output_root}")

if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python convert_images.py <input_root> <output_root>")
        sys.exit(1)

    input_root = Path("/Users/colincatlin/Downloads/project_1")  # Path(sys.argv[1])
    output_root = Path("/Users/colincatlin/Downloads/project_1_rename")   # Path(sys.argv[2])

    if not input_root.exists():
        print(f"Input directory {input_root} does not exist.")
        sys.exit(1)

    standardize_to_jpeg(input_root, output_root, quality=90)
