
# ApexYOLO Project

## Project Description

The ApexYOLO project focuses on developing and integrating various tools for working with the YOLO model, including training, data augmentation, auto-targeting, and a semi-automatic data collection system. Each script in the project serves a specific task, whether it's preparing a dataset, auto-targeting, or training the model.

## Main Project Scripts:

### splitDatasetFiles.py:

A script for evenly distributing photos and labels for training.
Prepares training, validation, and test datasets by splitting images and corresponding labels into parts.

**Usage:**
```bash
python splitDatasetFiles.py
```

### Albumentations.py:

This script performs the task of augmenting images and corresponding object labels (in YOLO format). 
It also removes images and labels with incorrect classes, improving data quality.

**Main Functions:**
- Reading images and labels.
- Applying augmentations such as random rotations, mirror flips, and shifts.
- Removing files with incorrect labels.

**Usage:**
```bash
python Albumentations.py
```

### ApexBot.py:

This script implements an auto-targeting system based on the YOLO model, capturing frames from the screen and using a random cursor movement technique to target objects.

**Main Functions:**
- Screen frame capturing.
- Object detection using YOLO.
- Automatic targeting.

**Usage:**
```bash
python ApexBot.py
```

### labelConfig.py:

A script for interactive editing of labels in YOLO format. 
Allows viewing images and corresponding labels, adding new labels, or removing existing ones.

**Main Functions:**
- Editing object labels on images.
- Adding and deleting labels.
- Removing images and labels with hotkeys.

**Usage:**
```bash
python labelConfig.py
```

### semiauto_dataset_collector.py:

A system for automatically capturing and saving images using the YOLO model, which also displays detection results in real-time.

**Main Functions:**
- Screen video capture.
- YOLO processing for object detection.
- Saving images and annotations in YOLO format.

**Usage:**
```bash
python semiauto_dataset_collector.py
```

### TrainYolov10.py:

A script for training the YOLO model using specified parameters. 
Logs results to TensorBoard and saves training graphs.

**Main Functions:**
- Training the YOLO model on a custom dataset.
- Logging metrics and training graphs to TensorBoard.
- Plotting graphs for analysis.

**Usage:**
```bash
python TrainYolov10.py
```

## Installation

1. Clone the project repository or download the archive:

```bash
git clone https://github.com/your-repository/ApexYOLO.git
```

2. Install the required dependencies, including PyTorch and YOLO:

```bash
pip install -r requirements.txt
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

3. Additionally, install Albumentations for data augmentation:

```bash
pip install albumentations
```

## Usage

To execute each script, follow the description above. For example, to train the YOLO model, run:

```bash
python TrainYolov10.py
```

To start the auto-targeting system:

```bash
python ApexBot.py
```

## Logs and TensorBoard

To monitor the training process, use TensorBoard. To start TensorBoard and view the training metrics, execute:

```bash
tensorboard --logdir=runs/yolo_training10/ApexEsp80
```

## Project Structure

- **ApexYOLO/** — Root project folder.
- **dataSet/** — Dataset used for training and testing.
- **runs/** — Folder for saving logs, metrics, and training results.
- **scripts/** — Folder containing the main project scripts.

## Contact Information

If you have any questions or issues, contact us via email: kirillnosenko304@gmail.com.
