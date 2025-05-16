from datasets import load_dataset

# Define the dataset and target directory
dataset_name = "timm/imagenet-1k-wds"
save_path = "D:/imagenet"

# Load and download dataset
dataset = load_dataset(dataset_name, split="train", cache_dir=save_path)

# Verify files are downloaded
print("Dataset downloaded to:", save_path)
