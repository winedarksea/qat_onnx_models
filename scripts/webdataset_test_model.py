import io
import os
import json
import glob
from PIL import Image
from datasets import load_dataset
import webdataset as wds
from tqdm import tqdm
import torch
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import torchvision.models as models
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import numpy as np
from sklearn.metrics import confusion_matrix


def get_dataset_and_dataloader(output_dir="filtered_imagenet", batch_size=32, image_size=224, for_prediction=False):
    """
    Creates dataset/dataloader.  Handles training and prediction modes.
    """
    class_mapping_path = os.path.join(output_dir, "class_mapping.json")
    with open(class_mapping_path, "r") as f:
        class_mapping = json.load(f)
    num_classes = len(class_mapping)

    transform = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    # Ensure paths are correct
    tar_files = glob.glob(os.path.join(output_dir, "*.tar"))
    if not tar_files:
        raise FileNotFoundError(f"No .tar files found in {output_dir}. Check your dataset path.")

    dataset = wds.WebDataset(tar_files)


    def transformed_dataset(img):
        return transform(img)

    if for_prediction:
        # For prediction, we don't need labels, just the transformed image.
        dataloader = (
            dataset
            .decode("pil")
            .map_dict(jpg=transformed_dataset)
            .to_tuple("jpg", "__key__")  # Keep the key for identifying the image
            .batched(batch_size, partial=False)
        )


    else:  # for training
        dataloader = (
            dataset
            .decode("pil")
            .map_dict(jpg=transformed_dataset)
            .to_tuple("jpg", "json")
            .batched(batch_size, partial=False)
        )

    return dataset, dataloader, num_classes


def train_model(dataloader, num_classes, num_epochs=10):
    """
    Trains a ResNet-18 model (same as before, but with simplified data loading).
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = models.resnet18(weights='IMAGENET1K_V1') 
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    model = model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    for epoch in range(num_epochs):
        running_loss = 0.0
        for i, (inputs, jsons) in enumerate(dataloader):
            # Extract labels from the JSON data
            try:
                labels = torch.tensor([sample['label'] for sample in jsons]).long()
            except Exception:
                labels = jsons

            inputs = inputs.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            if i % 100 == 99:
                print(f"[{epoch + 1}, {i + 1}] loss: {running_loss / 100:.3f}")
                running_loss = 0.0

    print("Finished Training")
    torch.save(model.state_dict(), "trained_model.pth")


def predict_image(model_path, image_path, class_mapping, image_size=224):
    """
    Predicts the class of a single image.

    Args:
        model_path: Path to the saved model (.pth file).
        image_path: Path to the image file.
        class_mapping: Dictionary mapping class names to indices.
        image_size:  Size to resize the image to.

    Returns:
        predicted_class: The predicted class name (string).
        probabilities:  Tensor of probabilities for each class.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load the model
    model = models.resnet18()  # Use the same architecture as during training
    num_classes = len(class_mapping)
    model.fc = nn.Linear(model.fc.in_features, num_classes)  # Set correct output size
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()  # Set the model to evaluation mode

    # Preprocess the image
    transform = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    image = Image.open(image_path).convert("RGB")  # Ensure image is in RGB format
    image = transform(image).unsqueeze(0)  # Add a batch dimension
    image = image.to(device)

    # Make the prediction
    with torch.no_grad():  # Disable gradient calculation
        outputs = model(image)
        probabilities = torch.nn.functional.softmax(outputs, dim=1)
        _, predicted_index = torch.max(probabilities, 1)

    # Convert index back to class name
    predicted_class_index = predicted_index.item()
    # Create reverse mapping for prediction
    reverse_class_mapping = {index: class_name for class_name, index in class_mapping.items()}
    predicted_class = reverse_class_mapping[predicted_class_index]

    return predicted_class, probabilities.cpu().numpy()[0]  # Return probabilities as numpy array


def evaluate_model(dataloader, model_path, class_mapping):
    """
    Evaluates the model on the given dataloader, computing accuracy and the confusion matrix.

    Args:
        dataloader: DataLoader for evaluation.
        model_path: Path to the trained model.
        class_mapping: Dictionary mapping class names to indices.

    Returns:
        accuracy: Float representing the accuracy.
        confusion_df: DataFrame representing the confusion matrix.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load the trained model
    model = models.resnet18()
    num_classes = len(class_mapping)
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()  # Set to evaluation mode

    all_preds = []
    all_labels = []

    with torch.no_grad():
        for inputs, jsons in tqdm(dataloader, desc="Evaluating Model"):
            try:
                labels = torch.tensor([sample['label'] for sample in jsons]).long()
            except Exception:
                labels = jsons

            inputs = inputs.to(device)
            labels = labels.to(device)

            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)  # Get predicted class index

            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    # Compute accuracy
    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)
    accuracy = np.mean(all_preds == all_labels)
    print(f"Model Accuracy: {accuracy:.4f}")

    # Compute confusion matrix
    cm = confusion_matrix(all_labels, all_preds)

    # Convert to DataFrame
    # class_names = list(class_mapping.keys())
    # confusion_df = pd.DataFrame(cm, index=class_names, columns=class_names)

    return accuracy, cm


if __name__ == '__main__':
    # Create dataset and dataloader
    dataset, dataloader, num_classes = get_dataset_and_dataloader()

    # Train
    train_model(dataloader, num_classes)

    # --- Prediction Phase ---
    # 1. Load the class mapping
    with open("filtered_imagenet/class_mapping.json", "r") as f:
        class_mapping = json.load(f)

    # 2.  Get a sample image for prediction.  We'll use the dataloader in
    #     prediction mode to grab one.
    _, predict_dataloader, _ = get_dataset_and_dataloader(for_prediction=True)
    first_batch_images, first_batch_keys = next(iter(predict_dataloader))
    idx = 1
    example_image = first_batch_images[idx]  # Get the first image
    example_key = first_batch_keys[idx]

    # 3. Save the example image to a temporary file
    # Convert the tensor back to a PIL Image
    inv_normalize = transforms.Normalize(
        mean=[-0.485/0.229, -0.456/0.224, -0.406/0.225],  # Inverse of the original normalization
        std=[1/0.229, 1/0.224, 1/0.225]
    )
    example_image = inv_normalize(example_image) # un-normalize
    example_image = transforms.ToPILImage()(example_image)  # Convert to PIL Image
    temp_image_path = "temp_image.jpg"
    example_image.save(temp_image_path)

    # 4. Predict
    predicted_class, probabilities = predict_image("trained_model.pth", temp_image_path, class_mapping)
    print(f"Predicted class for {example_key}: {predicted_class}")
    print(f"Probabilities: {probabilities}")

    # 5. (Optional) Display probabilities nicely
    print("\nClass Probabilities:")
    for class_name, prob in zip(class_mapping.keys(), probabilities):
        print(f"  {class_name}: {prob:.4f}")
    example_image
    
    #Clean up temp file
    os.remove(temp_image_path)
    
    # Load the class mapping
    with open("filtered_imagenet/class_mapping.json", "r") as f:
        class_mapping = json.load(f)

    # Reload dataset for evaluation
    # _, eval_dataloader, _ = get_dataset_and_dataloader()

    # Evaluate model
    accuracy, confusion_df = evaluate_model(dataloader, "trained_model.pth", class_mapping)
