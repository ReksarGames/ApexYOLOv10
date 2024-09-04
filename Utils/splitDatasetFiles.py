import os
import shutil
import random

def create_dirs(base_dir):
    os.makedirs(os.path.join(base_dir, 'train', 'images'), exist_ok=True)
    os.makedirs(os.path.join(base_dir, 'train', 'labels'), exist_ok=True)
    os.makedirs(os.path.join(base_dir, 'val', 'images'), exist_ok=True)
    os.makedirs(os.path.join(base_dir, 'val', 'labels'), exist_ok=True)
    os.makedirs(os.path.join(base_dir, 'test', 'images'), exist_ok=True)
    os.makedirs(os.path.join(base_dir, 'test', 'labels'), exist_ok=True)

def split_dataset(images_dir, labels_dir, output_base_dir, train_ratio=0.7, val_ratio=0.2):
    image_files = [f for f in os.listdir(images_dir) if f.endswith('.jpg')]
    random.shuffle(image_files)

    train_count = int(len(image_files) * train_ratio)
    val_count = int(len(image_files) * val_ratio)

    train_files = image_files[:train_count]
    val_files = image_files[train_count:train_count + val_count]
    test_files = image_files[train_count + val_count:]

    for split, files in zip(['train', 'val', 'test'], [train_files, val_files, test_files]):
        for file in files:
            image_path = os.path.join(images_dir, file)
            label_path = os.path.join(labels_dir, file.replace('.jpg', '.txt'))

            output_image_path = os.path.join(output_base_dir, split, 'images', file)
            output_label_path = os.path.join(output_base_dir, split, 'labels', file.replace('.jpg', '.txt'))

            shutil.copy(image_path, output_image_path)
            shutil.copy(label_path, output_label_path)

    print(f"Dataset split into train ({len(train_files)}), val ({len(val_files)}), and test ({len(test_files)}) sets.")

if __name__ == "__main__":
    images_dir = "D://Projects//Python//DotaDetection//dataSet//Apex//ManualOutput//outputDataset//augmented_images"
    labels_dir = "D://Projects//Python//DotaDetection//dataSet//Apex//ManualOutput//outputDataset//augmented_labels"
    output_base_dir = "D://Projects//Python//DotaDetection//dataSet//Apex//FinalDataset"

    create_dirs(output_base_dir)
    split_dataset(images_dir, labels_dir, output_base_dir)
