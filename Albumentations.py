import os
import cv2
import albumentations as A
import numpy as np

# Чтение изображения
def read_image(image_path):
    return cv2.imread(image_path)

# Чтение меток
def read_label(label_path):
    labels = []
    with open(label_path, "r") as f:
        for line in f.readlines():
            if line.strip():
                class_id, x, y, w, h = map(float, line.strip().split())
                labels.append([class_id, x, y, w, h])
    return np.array(labels).reshape(-1, 5)

# Сохранение изображения
def save_augmented_image(image, save_path):
    cv2.imwrite(save_path, image)

# Сохранение меток
def save_augmented_label(labels, save_path):
    with open(save_path, "w") as f:
        for label in labels:
            f.write(" ".join(map(str, label)) + "\n")

# Аугментация изображения и меток
def augment_image_and_labels(image, labels, image_file):
    if labels.size == 0:
        return image, labels

    transform = A.Compose([
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.5),
        A.RandomBrightnessContrast(p=0.2),
        A.ShiftScaleRotate(shift_limit=0.0625, scale_limit=0, rotate_limit=15, p=0.5, border_mode=cv2.BORDER_REFLECT),
    ], bbox_params=A.BboxParams(format='yolo', min_area=1024, min_visibility=0.2, label_fields=['class_labels']))

    bboxes = labels[:, 1:].tolist()
    class_labels = labels[:, 0].tolist()

    try:
        transformed = transform(image=image, bboxes=bboxes, class_labels=class_labels)
    except ValueError as e:
        print(f"Error processing file {image_file}: {e}")
        return image, labels

    if len(transformed['bboxes']) > 0:
        transformed_bboxes = np.array(transformed['bboxes'])
        transformed_class_labels = np.array(transformed['class_labels']).reshape(-1, 1)
        transformed_labels = np.hstack((transformed_class_labels, transformed_bboxes))
    else:
        transformed_labels = np.empty((0, 5))

    return transformed['image'], transformed_labels

# Удаление файла
def delete_file(file_path):
    if os.path.exists(file_path):
        os.remove(file_path)

# Основная функция обработки
def main(images_folder, labels_folder, output_images_folder, output_labels_folder, num_augmentations=6):
    if not os.path.exists(output_images_folder):
        os.makedirs(output_images_folder)
    if not os.path.exists(output_labels_folder):
        os.makedirs(output_labels_folder)

    total_images = 0
    total_labels = 0
    total_augmentations = 0
    class_counts = {0: 0, 1: 0, 2: 0}  # Счетчики классов

    for image_file in os.listdir(images_folder):
        image_path = os.path.join(images_folder, image_file)
        label_path = os.path.join(labels_folder, image_file.replace(".jpg", ".txt"))

        labels = read_label(label_path)

        if labels.size == 0:
            delete_file(image_path)
            delete_file(label_path)
            continue

        # Проверка, есть ли некорректные классы
        unique_classes = np.unique(labels[:, 0])
        for class_id in unique_classes:
            if class_id not in class_counts:
                print(f"Invalid label {class_id} in {label_path}. Deleting associated files.")
                delete_file(image_path)
                delete_file(label_path)
                break
            else:
                class_counts[class_id] += np.sum(labels[:, 0] == class_id)

        image = read_image(image_path)

        total_images += 1
        total_labels += len(labels)

        for i in range(num_augmentations):
            augmented_image, augmented_labels = augment_image_and_labels(image, labels, image_file)

            if augmented_labels.size == 0:
                continue

            save_image_path = os.path.join(output_images_folder, f"{os.path.splitext(image_file)[0]}_aug_{i}.jpg")
            save_label_path = os.path.join(output_labels_folder, f"{os.path.splitext(image_file)[0]}_aug_{i}.txt")

            save_augmented_image(augmented_image, save_image_path)
            save_augmented_label(augmented_labels, save_label_path)

            total_augmentations += 1

    print(f"Total images: {total_images}")
    print(f"Total labels: {total_labels}")
    print(f"Total augmentations: {total_augmentations}")

    # Вывод количества классов и их названий
    class_names = {0: 'apex-game', 1: 'avatar', 2: 'object'}
    for class_id, count in class_counts.items():
        print(f"Class '{class_names[class_id]}': {count} instances")

if __name__ == "__main__":
    images_folder = "images_folder" # set images_folder
    labels_folder = "labels_folder" # set labels_folder
    output_images_folder = "dataSet/Apex/ManualOutput/augmented_images"
    output_labels_folder = "dataSet/Apex/ManualOutput/augmented_labels"

    main(images_folder, labels_folder, output_images_folder, output_labels_folder)