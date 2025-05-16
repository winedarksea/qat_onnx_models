import os
import json
from datasets import load_dataset
import webdataset as wds
from tqdm import tqdm
import tarfile
from PIL import Image
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
import glob
from io import BytesIO


imagenet_classes = {
	"commercial_ship": ["n02965300", "n03095699", "n04474187"],
	"military_ship": ["n04552696", "n04487894", "n03638623", "n03466600"],
	"small_boat": ["n04115456", "n04229480", "n02951358"],
	"sailing_ship": ["n04128499", "n04128837"],
	"hospital_ship": ["n03541269"],
	"houseboat": ["n03545470"],
	"armored_vehicle": ["n02740533", "n02740300", "n02739889"],
	"truck": ["n04490091", "n04467665", "n03173929"],
	"sports_car": ["n04285008", "n03870105"],
	"golfcart": ["n03053976", "n03445924"],
	"helicopter": ["n02965122", "n03512147"],
	"first_aid_kit": ["n03349469"],
	"paddy": ["n07804900"],
	"beach": ["n09217230", "n09428293", "n09433839"],  # n09428628, n09376786
	"city": ["n08524735", "n04233124", "n08539072"],
	"harbor": ["n03492250", "n07900825", "n03934042", "n03216828"],
	"auditorium": ["n02758134"],
	"stadium": ["n03379204", "n04295881"],
	"airport": ["n02692232"],
	"gorilla": ["n02480855",  "n02481103"],
	"orangutan": ["n02480495"],
	"chimpanzee": ["n02481823", "n02482474"],
	"pinniped": ["n02077152", "n02077923", "n02078574", "n02081571"],
	"horse": ["n02374451", "n02377181", "n02377480"],
	"seabird": ["n02058221", "n02058594", "n02021795"],
	"whale": ["n02062744", "n02066245", "n02064816"],
	"sea_turtle": ["n01663401"],
	"solar_panel": ["n04257790", "n04257986"],
	"riverbank": ["n09411189", "n09415584"],
	"mountain": ["n09359803", "n09361517"],
	"sign": ["n06794110", "n06793231"],
	"dolmen": ["n03220237"],
	"utensil": ["n03621049", "n04516672", "n03101986"],
	"cardinal": ["n01541386"],
	"wind_turbine": ["n04591517"],
	"lawn_furniture": ["n03649674", "n03649797"],
	"athlete": ["n09820263", "n10639359", "n10153594"],
	"group": ["n08182379", "n08238463"],
	"professional_man": ["n10467395", "n09941787"],
	"professional_woman": ["n09944430", "n10521853"],
	"old_person": ["n10375402", "n10377021", "n10375314"],
	"child": ["n09918248", "n09917593", "n09918554"],
}


def download_and_filter_imagenet(imagenet_classes, output_dir='filtered_imagenet_2'):
    """
    Downloads the timm/imagenet-w21-wds dataset from Hugging Face, filters it based on
    the provided class mapping, and saves the filtered images and annotations to disk
    in a WebDataset format.

    Args:
        imagenet_classes (dict): A dictionary mapping new class names to a list of
                                 ImageNet class IDs.
        output_dir (str): The directory to save the filtered dataset.
    """

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Create a reverse mapping from ImageNet class ID to new class name
    id_to_class = {}
    for new_class, imagenet_ids in imagenet_classes.items():
        for imagenet_id in imagenet_ids:
            id_to_class[imagenet_id] = new_class

    # create a dictionary to hold the webdataset writers
    tar_writers = {}
    for new_class in imagenet_classes.keys():
       tar_writers[new_class] =  wds.TarWriter(os.path.join(output_dir, f"{new_class}.tar"))

    # Load the dataset
     # dataset = load_dataset("timm/imagenet-w21-wds", split="train", streaming=True)
    dataset = load_dataset("timm/imagenet-22k-wds", split="train", streaming=True)
    
    sample_count = {new_class: 0 for new_class in imagenet_classes.keys()}
    new_class_names = list(imagenet_classes.keys())

    # Iterate over the dataset and filter
    for sample in tqdm(dataset, desc="Filtering and saving images"):
        # imagenet_id = sample['__key__'].split('_')[0]  # Extract ImageNet ID from the key
        imagenet_id = sample["json"]["class_name"]

        if imagenet_id in id_to_class:
            new_class = id_to_class[imagenet_id]
            new_class_index = new_class_names.index(new_class) # Get the numerical index

            image = sample['jpg']

            # Modify the JSON structure
            sample_json = sample['json']
            sample_json['old_class_name'] = sample_json['class_name']  # Rename original
            sample_json['old_label'] = sample_json['label']  # Rename original
            sample_json['class_name'] = new_class  # Store new class name
            sample_json['label'] = new_class_index   # Store the numerical index!

            wds_sample = {
                "__key__": f"sample_{sample_count[new_class]:06d}",
                "jpg": image,
                "json": sample_json,
            }


            tar_writers[new_class].write(wds_sample)
            sample_count[new_class] += 1
            total_down = sum(sample_count.values())
            if total_down % 1000 == 0:
                print(f"Image number {total_down} downloaded of type {new_class}")
                try:
                    image
                except Exception as e:
                    print(repr(e))

    # Close all tar writers
    for writer in tar_writers.values():
        writer.close()

    # Save class mapping for later use
    class_mapping = {name: i for i, name in enumerate(new_class_names)}
    with open(os.path.join(output_dir, "class_mapping.json"), "w") as f:
        # json.dump(imagenet_classes, f, indent=4)
        json.dump(class_mapping, f, indent=4)
        # Correct class mapping: new class name -> index

    print(f"Filtered dataset saved to: {output_dir}")
    print(f"Number of samples per class: {sample_count}")


class CustomImageDataset(Dataset):
    def __init__(self, tar_files, class_mapping, image_size=224, transform=None):
        self.image_data = []  # Stores (image_bytes, label)
        self.class_mapping = class_mapping
        self.transform = transform if transform else transforms.ToTensor()
        
        for tar_path in tar_files:
            class_name = os.path.basename(tar_path).replace(".tar", "")  # Use directory name as label if needed
            class_label = class_mapping.get(class_name, -1)  # Get class index

            with tarfile.open(tar_path, 'r') as tar:
                for member in tar.getmembers():
                    if member.isfile() and member.name.endswith(".jpg"):
                        # Read JSON file to get label
                        json_filename = member.name.replace(".jpg", ".json")
                        json_member = tar.getmember(json_filename) if json_filename in tar.getnames() else None

                        label = class_label  # Default to directory name
                        if json_member:
                            json_data = json.loads(tar.extractfile(json_member).read().decode("utf-8"))
                            label = json_data.get("label", label)  # Prefer JSON label if available

                        # Store image bytes and label
                        img_bytes = tar.extractfile(member).read()
                        self.image_data.append((img_bytes, label))

    def __len__(self):
        return len(self.image_data)

    def __getitem__(self, idx):
        img_bytes, label = self.image_data[idx]
        img = Image.open(BytesIO(img_bytes)).convert("RGB")

        if self.transform:
            img = self.transform(img)

        return img, label

def get_dataset_and_dataloader(output_dir="filtered_imagenet", batch_size=32, image_size=224):
    """
    Creates dataset/dataloader by manually extracting tar files without WebDataset.
    """
    class_mapping_path = os.path.join(output_dir, "class_mapping.json")
    with open(class_mapping_path, "r") as f:
        class_mapping = json.load(f)

    tar_files = glob.glob(os.path.join(output_dir, "*.tar"))
    if not tar_files:
        raise FileNotFoundError(f"No .tar files found in {output_dir}. Check your dataset path.")

    transform = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    dataset = CustomImageDataset(tar_files, class_mapping, image_size=image_size, transform=transform)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    return dataset, dataloader, len(class_mapping)


if __name__ == '__main__':
    download_and_filter_imagenet(imagenet_classes)  # output_dir=r'D:\filtered_imagenet'

    dataset, dataloader, num_classes = get_dataset_and_dataloader()

    # Print some sample labels
    sample_images, sample_labels = next(iter(dataloader))
    print(f"Sample Labels: {sample_labels.tolist()}")  # Ensure labels are correct

    # Display a few images and labels
    import matplotlib.pyplot as plt

    fig, axs = plt.subplots(1, 5, figsize=(15, 5))
    for i in range(5):
        img = sample_images[i].permute(1, 2, 0).numpy()  # Convert to displayable format
        img = (img * 0.229 + 0.485).clip(0, 1)  # Unnormalize
        axs[i].imshow(img)
        axs[i].set_title(f"Label: {sample_labels[i].item()}")
        axs[i].axis("off")
    plt.show()