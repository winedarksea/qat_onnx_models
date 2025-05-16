import os
import zipfile
import subprocess

try:
    import pycocotools
except ImportError:
    subprocess.check_call(["pip", "install", "pycocotools"])

import requests

def download_file(url, dest):
    if not os.path.exists(dest):
        print(f"Downloading {url}...")
        r = requests.get(url, stream=True)
        with open(dest, 'wb') as f:
            for chunk in r.iter_content(chunk_size=8192):
                if chunk:
                    f.write(chunk)

base_dir = r"D:\img_data\coco"
os.makedirs(base_dir, exist_ok=True)

# URLs for COCO 2017
urls = {
    "train_images": "http://images.cocodataset.org/zips/train2017.zip",
    "val_images": "http://images.cocodataset.org/zips/val2017.zip",
    "annotations": "http://images.cocodataset.org/annotations/annotations_trainval2017.zip"
}

# Download files
for name, url in urls.items():
    zip_path = os.path.join(base_dir, f"{name}.zip")
    download_file(url, zip_path)

# Unzip
for name in urls:
    zip_path = os.path.join(base_dir, f"{name}.zip")
    print(f"Extracting {zip_path}...")
    with zipfile.ZipFile(zip_path, 'r') as zf:
        zf.extractall(base_dir)
