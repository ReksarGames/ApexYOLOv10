import random
import math
import time
import cv2
import numpy as np
import matplotlib.pyplot as plt

def linear_wind_mouse(x_start, y_start, x_end, y_end, wind_strength=1, gravity=1, max_steps=50, target_area=3):
    x = x_start
    y = y_start
    wind_x = 0
    wind_y = 0
    velocity_x = 0
    velocity_y = 0
    max_velocity = 20  # Увеличение максимальной скорости для ускорения движения
    min_velocity = 1

    points = [(x_start, y_start)]  # Список точек для рисования линии

    while math.sqrt((x_end - x) ** 2 + (y_end - y) ** 2) > target_area:
        x_diff = x_end - x
        y_diff = y_end - y
        distance = math.sqrt(x_diff ** 2 + y_diff ** 2)

        wind_x = wind_x / math.sqrt(3) + (random.random() * (wind_strength * 2 + 1) - wind_strength) / math.sqrt(5)
        wind_y = wind_y / math.sqrt(3) + (random.random() * (wind_strength * 2 + 1) - wind_strength) / math.sqrt(5)

        velocity_x = (velocity_x + wind_x) + gravity * (x_diff) / distance
        velocity_y = (velocity_y + wind_y) + gravity * (y_diff) / distance

        if math.sqrt(velocity_x ** 2 + velocity_y ** 2) > max_velocity:
            velocity_x = (velocity_x / math.sqrt(velocity_x ** 2 + velocity_y ** 2)) * max_velocity
            velocity_y = (velocity_y / math.sqrt(velocity_x ** 2 + velocity_y ** 2)) * max_velocity

        x += velocity_x
        y += velocity_y

        points.append((int(x), int(y)))

    points.append((x_end, y_end))  # Добавляем конечную точку
    return points

def draw_windmouse_path(image, start_pos, end_pos, color=(0, 255, 0), thickness=2):
    # Получаем точки движения с помощью алгоритма WindMouse
    points = linear_wind_mouse(start_pos[0], start_pos[1], end_pos[0], end_pos[1])

    # Рисуем линии между точками
    for i in range(len(points) - 1):
        cv2.line(image, points[i], points[i + 1], color, thickness)

def main():
    # Загрузка фона
    background_img = cv2.imread('/dataSet/Apex/ArrowUnpackedFace/train/images/0.jpg')  # Замените на путь к вашему изображению

    # Размер изображения (если необходимо, можно изменить размер)
    img_height, img_width, _ = background_img.shape

    # Начальная позиция (можно выбрать любые координаты)
    start_pos = (100, 100)

    # Целевые позиции для тестирования (список координат)
    targets = [
        (img_width - 100, 100),
        (img_width - 100, img_height - 100),
        (100, img_height - 100),
        (img_width // 2, img_height // 2)
    ]

    # Создаем копию изображения для рисования
    img_with_lines = background_img.copy()

    # Рисуем линии от начальной позиции к каждой цели
    for target in targets:
        draw_windmouse_path(img_with_lines, start_pos, target)
        start_pos = target  # Обновляем начальную позицию на конец последней линии

    # Отображаем результат
    cv2.imshow('WindMouse Path', img_with_lines)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    # Сохраняем изображение с нарисованными линиями
    cv2.imwrite('windmouse_test_output.jpg', img_with_lines)

if __name__ == '__main__':
    main()
