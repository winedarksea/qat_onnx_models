import csv
import urllib.request
import os
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor

# --- CONFIGURATION ---
MY_BUCKET = "colin-miap-madness"
MY_REGION = "us-central1"
MIAP_BOXES_URL = "https://storage.googleapis.com/openimages/open_images_extended_miap/open_images_extended_miap_boxes_train.csv"
OUTPUT_FILENAME = "vertex_miap_import.csv"
LOCAL_IMAGE_DIR = "miap_images"
MAX_WORKERS = 20  # Number of parallel downloads
MAX_IMAGES = None  # Set to a number (e.g. 10000) to limit the dataset size, or None for all

try:
    from tqdm import tqdm
except ImportError:
    def tqdm(iterable, **kwargs):
        return iterable

def get_project_id():
    """Gets the default GCP project ID and validates it."""
    result = subprocess.run(["gcloud", "config", "get-value", "project"], capture_output=True, text=True)
    project_id = result.stdout.strip()
    if not project_id or project_id == "(unset)":
        return None
    return project_id

def download_image(image_id):
    """Tries to download an image from Open Images splits via HTTPS."""
    splits = ['train', 'validation', 'test']
    local_path = os.path.join(LOCAL_IMAGE_DIR, f"{image_id}.jpg")
    
    if os.path.exists(local_path):
        return True
    
    for split in splits:
        url = f"https://storage.googleapis.com/open-images-dataset/{split}/{image_id}.jpg"
        try:
            with urllib.request.urlopen(url, timeout=10) as response:
                if response.status == 200:
                    with open(local_path, 'wb') as f:
                        f.write(response.read())
                    return True
        except Exception:
            continue
    return False

def main():
    project_id = "vertexai-486503"  # get_project_id()
    if not project_id:
        print("Error: Could not determine GCP project id. Run 'gcloud auth login'?")
        sys.exit(1)

    print(f"Project: {project_id}")
    print(f"Bucket: {MY_BUCKET}")

    # 1. Create bucket if it doesn't exist
    print(f"Ensuring bucket gs://{MY_BUCKET} exists...")
    # We check if bucket exists first to avoid unnecessary 'mb' errors
    check_bucket = subprocess.run(["gsutil", "ls", f"gs://{MY_BUCKET}/"], capture_output=True)
    if check_bucket.returncode != 0:
        print(f"Bucket gs://{MY_BUCKET} not found. Creating it...")
        subprocess.run(["gsutil", "mb", "-p", project_id, "-l", MY_REGION, f"gs://{MY_BUCKET}/"], check=True)

    # 2. Download metadata
    print(f"Downloading metadata from {MIAP_BOXES_URL}...")
    metadata_file = "miap_raw.csv"
    if not os.path.exists(metadata_file):
        urllib.request.urlretrieve(MIAP_BOXES_URL, metadata_file)

    # 3. Process CSV and gather unique ImageIDs
    print("Processing metadata...")
    image_ids = set()
    annotations = []
    
    with open(metadata_file, "r") as f:
        reader = csv.DictReader(f)
        for row in reader:
            image_id = row['ImageID']
            if MAX_IMAGES and len(image_ids) >= MAX_IMAGES and image_id not in image_ids:
                continue
            image_ids.add(image_id)
            annotations.append({
                'image_id': image_id,
                'x_min': row['XMin'],
                'y_min': row['YMin'],
                'x_max': row['XMax'],
                'y_max': row['YMax']
            })

    unique_ids = sorted(list(image_ids))
    print(f"Total annotations to process: {len(annotations)}")
    print(f"Unique images to download: {len(unique_ids)}")

    # 4. Download images locally
    os.makedirs(LOCAL_IMAGE_DIR, exist_ok=True)
    print(f"Downloading images to {LOCAL_IMAGE_DIR}...")
    
    success_count = 0
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        results = list(tqdm(executor.map(download_image, unique_ids), total=len(unique_ids)))
        success_count = sum(results)

    if success_count == 0:
        print("Error: No images were successfully downloaded. Verification failed.")
        sys.exit(1)

    print(f"Successfully downloaded {success_count}/{len(unique_ids)} images.")

    # 5. Generate Vertex AI Import CSV
    print(f"Generating {OUTPUT_FILENAME}...")
    success_ids = {unique_ids[i] for i, success in enumerate(results) if success}
    
    with open(OUTPUT_FILENAME, "w", newline="") as outfile:
        writer = csv.writer(outfile)
        for ann in annotations:
            if ann['image_id'] in success_ids:
                gcs_path = f"gs://{MY_BUCKET}/train/{ann['image_id']}.jpg"
                writer.writerow([
                    "TRAINING",      # ML_USE
                    gcs_path,        # GCS Path
                    "Person",        # Label
                    ann['x_min'], ann['y_min'], "", "", 
                    ann['x_max'], ann['y_max'], "", ""
                ])

    # 6. Upload images to GCS
    print(f"Uploading images to gs://{MY_BUCKET}/train/ ...")
    # Instead of a wildcard which can be fragile, we pass the file list via stdin
    image_paths = [os.path.join(LOCAL_IMAGE_DIR, f"{img_id}.jpg") for img_id in success_ids]
    upload_process = subprocess.Popen(
        ["gsutil", "-m", "-u", project_id, "cp", "-n", "-I", f"gs://{MY_BUCKET}/train/"],
        stdin=subprocess.PIPE,
        text=True
    )
    upload_process.communicate(input="\n".join(image_paths))
    if upload_process.returncode != 0:
        print("Error: Image upload failed.")
        sys.exit(1)

    # 7. Upload import CSV
    print(f"Uploading {OUTPUT_FILENAME} to gs://{MY_BUCKET}/{OUTPUT_FILENAME}...")
    subprocess.run(["gsutil", "-u", project_id, "cp", OUTPUT_FILENAME, f"gs://{MY_BUCKET}/{OUTPUT_FILENAME}"], check=True)

    print("\n" + "="*50)
    print("DONE!")

    print(f"Vertex AI Import file: gs://{MY_BUCKET}/{OUTPUT_FILENAME}")
    print(f"Images count in bucket: {success_count}")
    print("="*50)

if __name__ == "__main__":
    main()

