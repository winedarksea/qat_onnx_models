import os
import tarfile
import random
import json
from pathlib import Path
from tqdm import tqdm
from collections import defaultdict

SOURCE_DIR = "filtered_imagenet_2"
DEST_DIR   = "filtered_imagenet2_native"
TRAIN_SPLIT = 0.95  # 90% train, 10% val
SEED = 42

random.seed(SEED)

# Create output folders
for split in ["train", "val"]:
    os.makedirs(os.path.join(DEST_DIR, split), exist_ok=True)

def infer_classname(tar_path: Path) -> str:
    return tar_path.stem

# Class mapping: name → index
class_map = {}
reverse_map = defaultdict(list)
tar_paths = list(Path(SOURCE_DIR).glob("*.tar"))

for i, tar_path in enumerate(tqdm(tar_paths, desc="Extracting and splitting")):
    class_name = infer_classname(tar_path)
    class_map[class_name] = i

    # Create subfolders for this class
    train_dir = Path(DEST_DIR) / "train" / class_name
    val_dir   = Path(DEST_DIR) / "val" / class_name
    train_dir.mkdir(parents=True, exist_ok=True)
    val_dir.mkdir(parents=True, exist_ok=True)

    # Extract .jpg files and assign to train or val
    with tarfile.open(tar_path, 'r') as tar:
        jpg_members = [m for m in tar.getmembers() if m.isfile() and m.name.endswith(".jpg")]
        random.shuffle(jpg_members)

        split_idx = int(len(jpg_members) * TRAIN_SPLIT)
        train_members = jpg_members[:split_idx]
        val_members   = jpg_members[split_idx:]

        def extract_members(members, out_dir):
            for member in members:
                filename = os.path.basename(member.name)
                out_path = out_dir / filename
                with open(out_path, 'wb') as f:
                    f.write(tar.extractfile(member).read())

        extract_members(train_members, train_dir)
        extract_members(val_members, val_dir)

# Save class_mapping.json
mapping_path = Path(DEST_DIR) / "class_mapping.json"
with open(mapping_path, "w") as fp:
    json.dump(class_map, fp, indent=2)
print(f"✅ Saved class_mapping.json with {len(class_map)} entries to {mapping_path}")
