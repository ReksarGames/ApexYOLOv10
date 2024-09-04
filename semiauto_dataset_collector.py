import threading
import time
import cv2
import keyboard
import numpy as np
from ultralytics import YOLO
import os
import ctypes
from datetime import datetime
import queue

# Глобальные переменные
frame_to_display = None
stop_flag = False
saved_frame_count = 0
save_on_demand = False
frame_queue = queue.Queue(maxsize=5)  # Буферизация кадров

# Определение нужных функций и констант с помощью ctypes
user32 = ctypes.windll.user32


def obs_vc_init(capture_device=0):
    device = cv2.VideoCapture(capture_device)
    if not device.isOpened():
        raise Exception("Could not open video device")
    return device


def set_cap_size(device, w, h):
    device.set(cv2.CAP_PROP_FRAME_WIDTH, w)
    device.set(cv2.CAP_PROP_FRAME_HEIGHT, h)
    device.set(cv2.CAP_PROP_FPS, 60)  # Установите желаемый FPS, например, 60


def grab_process(config):
    global frame_to_display, stop_flag
    device = obs_vc_init(config["grabber"]["obs_vc_device_index"])
    set_cap_size(device, config["grabber"]["width"], config["grabber"]["height"])

    while not stop_flag:
        ret, frame = device.read()
        if not ret:
            continue
        frame_to_display = cv2.resize(frame, (config["grabber"]["width"] // 2, config["grabber"]["height"] // 2))
        try:
            frame_queue.put(frame_to_display, timeout=1)
        except queue.Full:
            continue


def save_process(config, output_folder, classes, num_frames, auto_grab_delay, auto_grab_required_conf, save_delay):
    global frame_to_display, stop_flag, saved_frame_count, save_on_demand
    model = YOLO(config["model_path"]).to('cuda')  # Перенос модели на GPU
    last_capture_time = time.time()

    while saved_frame_count < num_frames and not stop_flag:
        if not frame_queue.empty():
            frame_to_display = frame_queue.get()
            current_time = time.time()
            if current_time - last_capture_time >= auto_grab_delay or save_on_demand:
                img_rgb = cv2.cvtColor(frame_to_display, cv2.COLOR_BGR2RGB)
                results = model(img_rgb)

                should_save = save_on_demand or (
                        len(results[0].boxes) > 0 and max(
                    [box.conf for box in results[0].boxes]) >= auto_grab_required_conf
                )

                if should_save:
                    timestamp = datetime.now().strftime('%Y%m%d%H%M%S%f')
                    img_name = f'{timestamp}.jpg'
                    label_name = f'{timestamp}.txt'

                    # Сохранение изображения
                    img_bgr = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)
                    cv2.imwrite(os.path.join(output_folder, 'images', img_name), img_bgr)

                    # Сохранение аннотаций в формате YOLO
                    with open(os.path.join(output_folder, 'labels', label_name), 'w') as f:
                        for box in results[0].boxes:
                            cls = 0  # Если у вас только один класс, он должен быть 0
                            x_center, y_center, width, height = box.xywh[0]
                            x_center /= frame_to_display.shape[1]
                            y_center /= frame_to_display.shape[0]
                            width /= frame_to_display.shape[1]
                            height /= frame_to_display.shape[0]
                            f.write(f"{cls} {x_center} {y_center} {width} {height}\n")

                    saved_frame_count += 1
                    last_capture_time = current_time
                    save_on_demand = False
                    print(f'[INFO] Screenshot {saved_frame_count}/{num_frames} saved: {img_name}')
                time.sleep(save_delay)  # Используем переменную задержки между сохранениями


def display_process(config, scaling_factors):
    global stop_flag, save_on_demand
    model = YOLO(config["model_path"]).to('cuda')  # Перенос модели на GPU

    while not stop_flag:
        try:
            frame_to_display = frame_queue.get(timeout=1)
        except queue.Empty:
            continue

        start_time = time.time()

        # Преобразование кадра в RGB для модели
        img_rgb = cv2.cvtColor(frame_to_display, cv2.COLOR_BGR2RGB)
        results = model(img_rgb)  # Выполнение детекции

        annotated_img = frame_to_display.copy()

        # Отрисовка хитбоксов
        for box in results[0].boxes:
            # Получаем координаты и размеры хитбокса
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            conf = box.conf.item()  # Уровень уверенности
            label = f"{box.cls} {conf:.2f}"  # Метка класса и уверенности

            # Рисуем прямоугольник (хитбокс) на кадре
            cv2.rectangle(annotated_img, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(annotated_img, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        window_width, window_height = 1200, 600
        annotated_img_resized = cv2.resize(annotated_img, (window_width, window_height))

        # Расчет FPS
        end_time = time.time()
        fps = 1 / (end_time - start_time)
        cv2.putText(annotated_img_resized, f'FPS: {int(fps)}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)

        # Отображение изображения
        cv2.imshow('YOLOv9 Detection', annotated_img_resized)

        # Проверка нажатия кнопок
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            stop_flag = True
            break
        elif keyboard.is_pressed('caps lock'):  # Нажатие на Caps Lock для сохранения кадра
            save_on_demand = True


def main():
    global stop_flag
    config = {
        "grabber": {
            "obs_vc_device_index": 0,  # Индекс устройства захвата OBS
            "width": 1920,
            "height": 1080,
        },
        "model_path": 'runs/yolo_training10/ApexEsp400/weights/best.pt'
        # Путь к модели YOLO
    }

    output_folder = 'dataSet/Apex/ManualOutput'
    classes = ['apex-game', 'avatar', 'object']  # Ваши классы
    num_frames = 10000  # Количество скриншотов

    # Параметры авто-захвата
    AUTO_GRAB_DELAY = 2  # Сколько секунд ждать до следующего сохранения кадра
    AUTO_GRAB_REQUIRED_CONF = 0.7  # Минимальный уровень уверенности для сохранения детекции

    # Задержка между сохранениями
    SAVE_DELAY = 1  # Задержка в секундах между сохранениями изображений

    # Создание необходимых папок
    os.makedirs(os.path.join(output_folder, 'images'), exist_ok=True)
    os.makedirs(os.path.join(output_folder, 'labels'), exist_ok=True)

    # Запуск потока захвата
    grabber_thread = threading.Thread(target=grab_process, args=(config,))
    grabber_thread.start()

    # Запуск потока сохранения данных
    save_thread = threading.Thread(target=save_process, args=(
    config, output_folder, classes, num_frames, AUTO_GRAB_DELAY, AUTO_GRAB_REQUIRED_CONF, SAVE_DELAY))
    save_thread.start()

    # Запуск потока отображения
    scaling_factors = (user32.GetSystemMetrics(0) / 1920, user32.GetSystemMetrics(1) / 1080)
    display_thread = threading.Thread(target=display_process, args=(config, scaling_factors))
    display_thread.start()

    grabber_thread.join()
    save_thread.join()
    display_thread.join()


if __name__ == '__main__':
    main()
