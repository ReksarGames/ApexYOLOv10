import threading
import time
import cv2
import numpy as np
from ultralytics import YOLO
import ctypes
import random
import math
import win32api
import win32con
from pynput import keyboard

# Глобальные переменные
frame_to_display = None
stop_flag = False
last_target_time = 0
is_aiming = threading.Event()  # Флаг для управления наведением
auto_aim_enabled = threading.Event()  # Флаг для управления автонаводкой
auto_aim_enabled.set()  # По умолчанию автонаводка включена

# Определение нужных функций и констант с помощью ctypes
user32 = ctypes.windll.user32

def get_cursor_pos():
    cursor = ctypes.wintypes.POINT()
    user32.GetCursorPos(ctypes.byref(cursor))
    return cursor.x, cursor.y

def move_mouse(x, y):
    win32api.mouse_event(win32con.MOUSEEVENTF_MOVE, x, y)

def obs_vc_init(capture_device=0):
    device = cv2.VideoCapture(capture_device)
    if not device.isOpened():
        raise Exception("Could not open video device")
    return device

def set_cap_size(device, w, h):
    device.set(cv2.CAP_PROP_FRAME_WIDTH, w)
    device.set(cv2.CAP_PROP_FRAME_HEIGHT, h)

def grab_process(config):
    global frame_to_display, stop_flag
    device = obs_vc_init(config["grabber"]["obs_vc_device_index"])
    set_cap_size(device, config["grabber"]["width"], config["grabber"]["height"])

    while not stop_flag:
        ret, frame = device.read()
        if not ret:
            continue
        frame_to_display = frame


def wind_mouse(x_start, y_start, x_end, y_end, settings):
    wind_strength = settings["wind_strength"]
    gravity = settings["gravity"]
    max_velocity = settings["max_velocity"]
    min_velocity = settings["min_velocity"]
    target_area = settings["target_area"]
    delay = settings["initial_delay"]

    x = x_start
    y = y_start
    wind_x = 0
    wind_y = 0
    velocity_x = 0
    velocity_y = 0

    while math.sqrt((x_end - x) ** 2 + (y_end - y) ** 2) > target_area:
        if not is_aiming.is_set():
            print("Aiming interrupted")  # Отладка
            break

        x_diff = x_end - x
        y_diff = y_end - y
        distance = math.sqrt(x_diff ** 2 + y_diff ** 2)

        # Регулировка максимальной скорости в зависимости от расстояния до цели
        current_max_velocity = max(min_velocity, max_velocity * (distance / 100.0))

        wind_x = wind_x / math.sqrt(3) + (random.random() * (wind_strength * 2 + 1) - wind_strength) / math.sqrt(5)
        wind_y = wind_y / math.sqrt(3) + (random.random() * (wind_strength * 2 + 1) - wind_strength) / math.sqrt(5)

        velocity_x = (velocity_x + wind_x) + gravity * (x_diff) / distance
        velocity_y = (velocity_y + wind_y) + gravity * (y_diff) / distance

        # Ограничение скорости на основе текущей максимальной скорости
        if math.sqrt(velocity_x ** 2 + velocity_y ** 2) > current_max_velocity:
            random_dist = current_max_velocity / 2.0 + random.random() * current_max_velocity / 2.0
            velocity_x = (velocity_x / math.sqrt(velocity_x ** 2 + velocity_y ** 2)) * random_dist
            velocity_y = (velocity_y / math.sqrt(velocity_x ** 2 + velocity_y ** 2)) * random_dist

        x += velocity_x
        y += velocity_y

        # Имитация легких корректировок
        small_adjustment_x = random.uniform(-0.5, 0.5)
        small_adjustment_y = random.uniform(-0.5, 0.5)

        move_mouse(int(round(x - x_start + small_adjustment_x)), int(round(y - y_start + small_adjustment_y)))
        x_start = x
        y_start = y
        time.sleep(delay)  # Задержка, регулирующая скорость наведения


def detection_and_display_process(model_path, scaling_factors, settings):
    global frame_to_display, stop_flag, last_target_time
    model = YOLO(model_path)
    hold_time = settings["hold_time"]

    print("Detection and display process started")  # Отладка

    while not stop_flag:
        start_time = time.time()

        if frame_to_display is not None:
            print("Processing frame")  # Отладка

            frame = frame_to_display.copy()

            # Обработка кадра с помощью YOLO
            results = model.predict(frame, conf=0.5)
            annotated_img = results[0].plot()

            if is_aiming.is_set():
                print("Aiming active")  # Отладка
                if len(results[0].boxes) > 0:
                    boxes = results[0].boxes.xyxy
                    centers = [(int((box[0] + box[2]) // 2), int((box[1] + box[3]) // 2) - int((box[3] - box[1]) // 4)) for box in boxes]
                    mouse_position = get_cursor_pos()

                    # Поиск ближайшего центра к текущему положению курсора
                    nearest_center = min(centers, key=lambda c: (c[0] - mouse_position[0]) ** 2 + (c[1] - mouse_position[1]) ** 2)

                    x_offset = int((nearest_center[0] - mouse_position[0]) / scaling_factors[0])
                    y_offset = int((nearest_center[1] - mouse_position[1]) / scaling_factors[1])

                    distance_to_target = np.sqrt(x_offset**2 + y_offset**2)
                    if distance_to_target < 5:  # Порог фиксации
                        if time.time() - last_target_time >= hold_time:
                            last_target_time = time.time()
                    else:
                        wind_mouse(mouse_position[0], mouse_position[1], mouse_position[0] + x_offset, mouse_position[1] + y_offset, settings)
                        last_target_time = time.time()

            end_time = time.time()
            fps = 1 / (end_time - start_time)
            cv2.putText(annotated_img, f'FPS: {int(fps)}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)

            cv2.imshow('YOLOv9 Detection', annotated_img)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                stop_flag = True
                break

    cv2.destroyAllWindows()
    print("Detection and display process stopped")  # Отладка


def on_press(key):
    try:
        if key == keyboard.Key.caps_lock:
            print("Caps Lock pressed")  # Отладка
            is_aiming.set()
    except AttributeError:
        pass

def on_release(key):
    try:
        if key == keyboard.Key.caps_lock:
            print("Caps Lock released")  # Отладка
            is_aiming.clear()
    except AttributeError:
        pass

def main():
    global stop_flag
    config = {
        "grabber": {
            "obs_vc_device_index": 0,
            "width": 1920,
            "height": 1080,
        },
        "model_path": 'runs/yolo_training10/ApexEsp400/weights/best.pt'
    }
    aiming_settings = {
        "wind_strength": 2,  # Сила ветра, влияет на разброс курсора
        "gravity": 12,  # Гравитация, влияет на направленность движения курсора к цели
        "max_velocity": 1,  # Максимальная начальная скорость курсора
        "min_velocity": 12,  # Минимальная скорость при приближении к цели
        "target_area": 5,  # Размер области вокруг цели, в которой курсор считается достигнувшим цели
        "initial_delay": 0.01,  # Задержка между шагами при движении курсора
        "hold_time": 1  # Время удержания курсора на цели
    }
    scaling_factors = (user32.GetSystemMetrics(0) / 1920, user32.GetSystemMetrics(1) / 1080)

    # Запуск потока захвата
    print("Starting grabber thread")  # Отладка
    grabber_thread = threading.Thread(target=grab_process, args=(config,))
    grabber_thread.start()

    # Запуск мониторинга клавиатуры в фоновом режиме
    listener = keyboard.Listener(on_press=on_press, on_release=on_release)
    listener.start()

    # Обработка и отображение в основном потоке
    detection_and_display_process(config["model_path"], scaling_factors, aiming_settings)

    grabber_thread.join()
    listener.stop()

if __name__ == '__main__':
    main()

# Когда зажимаю капс для автонаводки, экран зависает на 5 фпс, скорее всего проблема связано в том что так 1 поток, или что-то такое