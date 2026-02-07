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
MIAP_LIST_URLS = [
    "https://storage.googleapis.com/openimages/open_images_extended_miap/open_images_extended_miap_images_train.lst",
    "https://storage.googleapis.com/openimages/open_images_extended_miap/open_images_extended_miap_images_val.lst",
    "https://storage.googleapis.com/openimages/open_images_extended_miap/open_images_extended_miap_images_test.lst"
]
OUTPUT_FILENAME = "vertex_miap_import.csv"
LOCAL_IMAGE_DIR = "miap_images"
MAX_WORKERS = 10  # Increased since we are using S3 and high bandwidth
MAX_IMAGES = None 
BATCH_SIZE = 500 
DELETE_LOCAL_AFTER_UPLOAD = True

# Global dictionary to map image_id -> split
image_split_map = {}

try:
    from tqdm import tqdm
except ImportError:
    def tqdm(iterable, **kwargs):
        return iterable

def get_project_id():
    """Gets the default GCP project ID and validates it."""
    # First try environment variables
    project_id = os.environ.get("GOOGLE_CLOUD_PROJECT") or os.environ.get("DEVSHELL_PROJECT_ID")
    if project_id:
        return project_id

    # Fallback to gcloud configuration
    try:
        result = subprocess.run(
            ["gcloud", "config", "get-value", "project"], 
            capture_output=True, text=True
        )
        project_id = result.stdout.strip()
        if project_id and project_id != "(unset)":
            return project_id
    except (subprocess.SubprocessError, FileNotFoundError):
        pass
    return None

def get_existing_gcs_images(bucket_name, project_id):
    """Returns a set of image IDs already present in the GCS bucket using a fast scan."""
    print(f"Checking existing images in gs://{bucket_name}/train/ (Fast Scan)...")
    
    # We combine stdout and stderr to avoid potential deadlocks with PIPE buffers
    cmd = ["gsutil", "-m", "-u", project_id, "ls", f"gs://{bucket_name}/train/"]
    try:
        # Using stderr=subprocess.STDOUT to avoid buffer deadlocks
        process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1)
        
        existing_ids = set()
        count = 0
        has_started = False
        for line in process.stdout:
            if not has_started:
                print("First result received from GCS. Processing...")
                has_started = True
            
            line = line.strip()
            # gsutil -m ls output can include progress lines, only process gs:// paths
            if line.startswith("gs://") and line.endswith(".jpg"):
                img_id = os.path.basename(line).replace(".jpg", "")
                existing_ids.add(img_id)
                count += 1
                if count % 1000 == 0:  # More frequent updates for visible progress
                    print(f"Parsed {count} images from GCS...")
            elif "not found" in line.lower() or "matched no objects" in line.lower():
                # If we see a 404-like error in the combined stream
                print("No existing images found (directory may not exist yet).")
                process.terminate()
                return set()
        
        process.wait()
        if len(existing_ids) > 0:
            print(f"Successfully found {len(existing_ids)} existing images.")
            return existing_ids
        elif process.returncode == 0:
            print("Finished scan: No images found.")
            return set()
        else:
            print(f"GCS scan finished with exit code {process.returncode}.")
    except Exception as e:
        print(f"Error during GCS scan: {e}")
            
    return set()

def generate_and_upload_manifest(annotations, final_ids, bucket_name, project_id, output_filename):
    """Generates a manifest for confirmed images and uploads it to GCS."""
    if not final_ids:
        print("Warning: No images confirmed. Not generating manifest.")
        return

    # Filter annotations to only include those with images confirmed in GCS
    with open(output_filename, "w", newline="") as outfile:
        writer = csv.writer(outfile)
        for ann in annotations:
            if ann['image_id'] in final_ids:
                gcs_path = f"gs://{bucket_name}/train/{ann['image_id']}.jpg"
                writer.writerow([
                    "TRAINING",      # ML_USE
                    gcs_path,        # GCS Path
                    "Person",        # Label
                    ann['x_min'], ann['y_min'], "", "", 
                    ann['x_max'], ann['y_max'], "", ""
                ])

    subprocess.run(["gsutil", "-u", project_id, "cp", output_filename, f"gs://{bucket_name}/{output_filename}"], 
                   capture_output=True, text=True)

def download_image(image_id):
    """Tries to download an image from Open Images public S3 mirror."""
    split = image_split_map.get(image_id)
    if not split:
        # Fallback to searching if split map is incomplete
        possible_splits = ['train', 'validation', 'test']
    else:
        possible_splits = [split]

    local_path = os.path.join(LOCAL_IMAGE_DIR, f"{image_id}.jpg")
    if os.path.exists(local_path):
        return True
    
    for s in possible_splits:
        # confirmed that GCS is Requester Pays (403), but S3 mirror is Public (200)
        url = f"https://open-images-dataset.s3.amazonaws.com/{s}/{image_id}.jpg"
        try:
            req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
            with urllib.request.urlopen(req, timeout=10) as response:
                if response.status == 200:
                    with open(local_path, 'wb') as f:
                        f.write(response.read())
                    return True
        except Exception:
            continue
    return False

def main():
    # Attempt to detect project ID automatically
    project_id = get_project_id()
    
    if project_id:
        print(f"Project Detected: {project_id}")
    else:
        project_id = "vertexai-486503" 
        print(f"Warning: Could not detect project ID automatically. Using fallback: {project_id}")

    print(f"Bucket: {MY_BUCKET}")

    # 1. Check if bucket exists
    print(f"Ensuring bucket gs://{MY_BUCKET} exists...")
    check_bucket = subprocess.run(["gsutil", "ls", "-d", f"gs://{MY_BUCKET}/"], capture_output=True, text=True)
    
    if check_bucket.returncode != 0:
        stderr = (check_bucket.stderr or "").lower()
        if "404" in stderr or "notfound" in stderr:
            print(f"Bucket gs://{MY_BUCKET} not found. Attempting to create it...")
            try:
                subprocess.run(["gsutil", "mb", "-p", project_id, "-l", MY_REGION, f"gs://{MY_BUCKET}/"], check=True)
                print(f"Bucket gs://{MY_BUCKET} created.")
            except subprocess.CalledProcessError:
                print(f"Warning: Failed to create bucket. Proceeding assuming it exists or will be created.")
        else:
            print(f"Notice: Bucket check returned code {check_bucket.returncode}. Proceeding assuming bucket exists...")
    else:
        print(f"Bucket gs://{MY_BUCKET} confirmed.")

    # 2. Download metadata and split lists
    print(f"Downloading metadata from {MIAP_BOXES_URL}...")
    metadata_file = "miap_raw.csv"
    if not os.path.exists(metadata_file):
        urllib.request.urlretrieve(MIAP_BOXES_URL, metadata_file)

    global image_split_map
    print("Downloading MIAP image lists for split detection...")
    for list_url in MIAP_LIST_URLS:
        list_name = list_url.split('/')[-1]
        if not os.path.exists(list_name):
            urllib.request.urlretrieve(list_url, list_name)
        
        with open(list_name, 'r') as f:
            for line in f:
                parts = line.strip().split('/')
                if len(parts) == 2:
                    split_name, img_id = parts
                    # Standardize 'validation' (in list) vs 'validation' (in S3)
                    # Note: S3 uses 'validation' and 'train' and 'test'
                    image_split_map[img_id] = split_name

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
    print(f"Total annotations found: {len(annotations)}")
    print(f"Unique images in dataset: {len(unique_ids)}")

    # 4. Filter out images already in GCS
    existing_ids = get_existing_gcs_images(MY_BUCKET, project_id)
    ids_to_process = [img_id for img_id in unique_ids if img_id not in existing_ids]
    print(f"Images already in GCS: {len(existing_ids)}")
    print(f"Images remaining to process: {len(ids_to_process)}")

    if not ids_to_process:
        print("All images already present in GCS.")
    else:
        # 5. Process in batches
        os.makedirs(LOCAL_IMAGE_DIR, exist_ok=True)
        # Keep track of what we've confirmed is in GCS to avoid re-scanning
        confirmed_in_gcs = set(existing_ids)
        
        for i in range(0, len(ids_to_process), BATCH_SIZE):
            batch = ids_to_process[i:i + BATCH_SIZE]
            print(f"\nProcessing batch {i//BATCH_SIZE + 1} ({len(batch)} images)...")
            
            # Download batch
            success_batch_ids = []
            with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
                results = list(tqdm(executor.map(download_image, batch), total=len(batch)))
                success_batch_ids = [batch[j] for j, success in enumerate(results) if success]

            if not success_batch_ids:
                print("Warning: No images downloaded in this batch. This likely means all S3 URLs failed.")
                continue

            # Upload batch
            print(f"Uploading {len(success_batch_ids)} images to GCS...")
            image_paths = [os.path.join(LOCAL_IMAGE_DIR, f"{img_id}.jpg") for img_id in success_batch_ids]
            
            # Use -u project_id for upload to ensure billing context is provided if needed
            upload_cmd = ["gsutil", "-m", "-u", project_id, "cp", "-n", "-I", f"gs://{MY_BUCKET}/train/"]
            upload_process = subprocess.Popen(
                upload_cmd,
                stdin=subprocess.PIPE,
                text=True
            )
            upload_process.communicate(input="\n".join(image_paths))
            
            if upload_process.returncode == 0:
                # Update our tracking set instead of re-scanning the whole bucket
                confirmed_in_gcs.update(success_batch_ids)
            else:
                print(f"Warning: Upload failed for batch {i//BATCH_SIZE + 1}. We should rescan before manifest.")
                # If a batch fails, we might still have some successes, but we'll rescan at the very end
                pass
            
            # Cleanup local disk to save space
            if DELETE_LOCAL_AFTER_UPLOAD:
                for img_id in batch:
                    local_path = os.path.join(LOCAL_IMAGE_DIR, f"{img_id}.jpg")
                    if os.path.exists(local_path):
                        os.remove(local_path)
            
            # Fast manifest generation using our tracked set
            print(f"Updating manifest with {len(confirmed_in_gcs)} confirmed images...")
            generate_and_upload_manifest(annotations, confirmed_in_gcs, MY_BUCKET, project_id, OUTPUT_FILENAME)
    
    # Final full rescan and manifest generation to be 100% sure
    print("\nPerforming final GCS verification...")
    final_ids = get_existing_gcs_images(MY_BUCKET, project_id)
    generate_and_upload_manifest(annotations, final_ids, MY_BUCKET, project_id, OUTPUT_FILENAME)

    print("\n" + "="*50)
    print("DONE!")
    print(f"Vertex AI Import file: gs://{MY_BUCKET}/{OUTPUT_FILENAME}")
    print("="*50)




if __name__ == "__main__":
    main()

