import threading
import time
import cv2
import keyboard
import numpy as np
import os
from datetime import datetime
import queue
import win32api
import ctypes

from keyboard._winkeyboard import user32
from ultralytics import YOLO
import utils.dxshot as dxcam

class ScreenCapture:
    def __init__(self, config):
        self.config = config
        self.frame_to_display = None
        self.stop_flag = False
        self.saved_frame_count = 0
        self.save_on_demand = False
        self.frame_queue = queue.Queue(maxsize=5)
        self.region = None
        self.camera = None

    def capture_init(self):
        screen_width = win32api.GetSystemMetrics(0)
        screen_height = win32api.GetSystemMetrics(1)
        crop_height = int(screen_height * self.config['grabber']['crop_size'])
        crop_width = int(crop_height * (screen_width / screen_height))
        x = (screen_width - crop_width) // 2
        y = (screen_height - crop_height) // 2
        self.region = (x, y, x + crop_width, y + crop_height)
        self.camera = dxcam.create(region=self.region)

    def take_shot(self):
        img = None
        while img is None:
            img = self.camera.grab(region=self.region)
        return img

    def grab_process(self):
        self.capture_init()
        while not self.stop_flag:
            frame = self.take_shot()
            self.frame_to_display = cv2.resize(frame, (self.config["grabber"]["width"], self.config["grabber"]["height"]))
            try:
                self.frame_queue.put(self.frame_to_display, timeout=1)
            except queue.Full:
                continue

    def save_process(self, output_folder, classes, num_frames, auto_grab_delay, auto_grab_required_conf, save_delay):
        model = YOLO(self.config["model_path"]).to('cuda')
        last_capture_time = time.time()

        while self.saved_frame_count < num_frames and not self.stop_flag:
            if not self.frame_queue.empty():
                frame_to_save = self.frame_queue.get()
                current_time = time.time()
                if current_time - last_capture_time >= auto_grab_delay or self.save_on_demand:
                    img_rgb = cv2.cvtColor(frame_to_save, cv2.COLOR_BGR2RGB)
                    results = model(img_rgb)

                    should_save = self.save_on_demand or (
                        len(results[0].boxes) > 0 and max([box.conf for box in results[0].boxes]) >= auto_grab_required_conf
                    )

                    if should_save:
                        timestamp = datetime.now().strftime('%Y%m%d%H%M%S%f')
                        img_name = f'{timestamp}.jpg'
                        label_name = f'{timestamp}.txt'

                        img_bgr = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)
                        cv2.imwrite(os.path.join(output_folder, 'images', img_name), img_bgr)

                        with open(os.path.join(output_folder, 'labels', label_name), 'w') as f:
                            for box in results[0].boxes:
                                cls = int(box.cls)
                                x_center, y_center, width, height = box.xywh[0]
                                x_center /= frame_to_save.shape[1]
                                y_center /= frame_to_save.shape[0]
                                width /= frame_to_save.shape[1]
                                height /= frame_to_save.shape[0]
                                f.write(f"{cls} {x_center} {y_center} {width} {height}\n")

                        self.saved_frame_count += 1
                        last_capture_time = current_time
                        self.save_on_demand = False
                        print(f'[INFO] Screenshot {self.saved_frame_count}/{num_frames} saved: {img_name}')
                    time.sleep(save_delay)

    def display_process(self, scaling_factors, classes):
        model = YOLO(self.config["model_path"]).to('cuda')

        while not self.stop_flag:
            try:
                frame_to_display = self.frame_queue.get(timeout=1)
            except queue.Empty:
                continue

            start_time = time.time()
            img_rgb = cv2.cvtColor(frame_to_display, cv2.COLOR_BGR2RGB)
            results = model(img_rgb)

            annotated_img = frame_to_display.copy()

            for box in results[0].boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                conf = box.conf.item()
                class_id = int(box.cls)

                if class_id < len(classes):
                    class_name = classes[class_id]
                else:
                    class_name = "unknown"

                label = f"{class_name} {conf:.2f}"
                cv2.rectangle(annotated_img, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(annotated_img, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

            window_width, window_height = 1200, 600
            annotated_img_resized = cv2.resize(annotated_img, (window_width, window_height))

            end_time = time.time()
            fps = 1 / (end_time - start_time)
            cv2.putText(annotated_img_resized, f'FPS: {int(fps)}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)

            cv2.imshow('YOLOv9 Detection', annotated_img_resized)

            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                self.stop_flag = True
                break
            elif keyboard.is_pressed('caps lock'):
                self.save_on_demand = True

def main():
    config = {
        "grabber": {
            "crop_size": 0.8,
            "width": 1920,
            "height": 1080,
        },
        "model_path": 'model/best_8n.pt'
    }

    output_folder = 'dataSet/Apex/ManualOutput'
    classes = ['ally', 'enemy', 'tag']
    num_frames = 10000
    AUTO_GRAB_DELAY = 2
    AUTO_GRAB_REQUIRED_CONF = 0.7
    SAVE_DELAY = 1

    os.makedirs(os.path.join(output_folder, 'images'), exist_ok=True)
    os.makedirs(os.path.join(output_folder, 'labels'), exist_ok=True)

    screen_capture = ScreenCapture(config)

    grabber_thread = threading.Thread(target=screen_capture.grab_process)
    save_thread = threading.Thread(target=screen_capture.save_process, args=(output_folder, classes, num_frames, AUTO_GRAB_DELAY, AUTO_GRAB_REQUIRED_CONF, SAVE_DELAY))
    display_thread = threading.Thread(target=screen_capture.display_process, args=((user32.GetSystemMetrics(0) / 1920, user32.GetSystemMetrics(1) / 1080), classes))

    grabber_thread.start()
    save_thread.start()
    display_thread.start()

    grabber_thread.join()
    save_thread.join()
    display_thread.join()

if __name__ == '__main__':
    main()
