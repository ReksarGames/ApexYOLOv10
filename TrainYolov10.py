import os
import sys
import torch
import logging
import matplotlib.pyplot as plt
from torch.utils.tensorboard import SummaryWriter
from ultralytics import YOLO

def train_model():
    # Настройка логирования
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)

    # Определение устройства для обучения (GPU, если доступен)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f'Using device: {device}')

    model_path = "runs/yolo_training10/ApexEsp400/weights/best.pt"

    # Инициализация модели YOLOv10
    if not os.path.isfile(model_path):
        logger.error(f'Model file not found: {model_path}')
        sys.exit(1)

    model = YOLO(model_path).to(device)

    data_yaml = 'dataSet/Apex/FinalDataset/data.yaml'

    if not os.path.isfile(data_yaml):
        logger.error(f'Data configuration file not found: {data_yaml}')
        sys.exit(1)

    # Настройка TensorBoard
    log_dir = 'runs/yolo_training10/ApexEsp80'
    writer = SummaryWriter(log_dir)

    # Настройка параметров обучения
    train_params = {
        'data': data_yaml,
        'epochs': 200,
        'imgsz': 640,
        'batch': 16,
        'device': device,
        'project': 'runs/yolo_training10',
        'name': 'ApexEsp400',
        'exist_ok': True,
        'patience': 50,
        'cache': False,
        'augment': False,  # Попробуйте отключить аугментацию
        'lr0': 0.01,
        'lrf': 0.0005,
        'weight_decay': 0.0005,
        'resume': True,
        'workers': 4,
    }

    logger.info(f'Training parameters: {train_params}')

    # Запуск обучения
    logger.info("Starting training...")
    try:
        results = model.train(**train_params)

        # Логирование метрик в TensorBoard
        logger.info("Training completed.")
    except Exception as e:
        logger.error(f"An error occurred during training: {e}")
        sys.exit(1)

    # Закрытие TensorBoard
    writer.close()

    # Логирование результатов обучения
    logger.info(f'Training results: {results}')

    # Построение графиков
    try:
        # Пример получения метрик; возможно, вам потребуется скорректировать это в зависимости от структуры results
        metrics = results.results_dict
        plt.figure(figsize=(10, 6))
        plt.plot(metrics.get('train_loss', []), label='Train Loss')
        plt.plot(metrics.get('val_loss', []), label='Validation Loss')
        plt.legend()
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title('Training and Validation Loss')
        plt.grid(True)
        plt.savefig('training_results.png')
        plt.show()
    except Exception as e:
        logger.error(f"An error occurred while plotting the results: {e}")

if __name__ == '__main__':
    train_model()

# tensorboard --logdir=runs/yolo_training10/ApexEsp80
