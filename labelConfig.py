import cv2
import os

# Глобальные переменные для взаимодействия с мышью
ref_point = []
cropping = False
current_labels = []
image_copy = None

def draw_labels_on_image(image_path, label_path):
    global current_labels

    # Открываем изображение
    image = cv2.imread(image_path)
    if image is None:
        print(f"Error: Image not found at {image_path}")
        return None, []

    # Проверяем наличие файла с метками
    if not os.path.exists(label_path):
        print(f"Error: Label file not found at {label_path}")
        return image, []

    # Чтение файла меток
    with open(label_path, 'r') as file:
        lines = file.readlines()

    current_labels = []

    # Проход по всем меткам и нанесение их на изображение
    for line in lines:
        parts = line.strip().split()
        if len(parts) != 5:
            continue

        # Получение данных метки
        class_id, x_center, y_center, width, height = map(float, parts)
        current_labels.append((class_id, x_center, y_center, width, height))

        # Преобразование нормализованных координат в реальные пиксели
        img_h, img_w = image.shape[:2]
        x_center = int(x_center * img_w)
        y_center = int(y_center * img_h)
        width = int(width * img_w)
        height = int(height * img_h)

        # Вычисление координат углов рамки
        x1 = int(x_center - width / 2)
        y1 = int(y_center - height / 2)
        x2 = int(x_center + width / 2)
        y2 = int(y_center + height / 2)

        # Рисование точки в центре объекта и прямоугольника вокруг него
        cv2.circle(image, (x_center, y_center), 5, (0, 0, 255), -1)  # Красная точка
        cv2.rectangle(image, (x1, y1), (x2, y2), (255, 0, 0), 2)  # Синий прямоугольник

    return image, current_labels

def click_and_crop(event, x, y, flags, param):
    global ref_point, cropping, image_copy

    if event == cv2.EVENT_RBUTTONDOWN:
        ref_point = [(x, y)]
        cropping = True

    elif event == cv2.EVENT_MOUSEMOVE:
        if cropping:
            # Восстанавливаем копию изображения для динамического рисования
            temp_image = image_copy.copy()
            cv2.rectangle(temp_image, ref_point[0], (x, y), (0, 255, 0), 2)
            cv2.imshow("Image with Labels", temp_image)

    elif event == cv2.EVENT_RBUTTONUP:
        ref_point.append((x, y))
        cropping = False

        # Рисуем финальный прямоугольник на изображении
        cv2.rectangle(image_copy, ref_point[0], ref_point[1], (0, 255, 0), 2)
        cv2.imshow("Image with Labels", image_copy)

        # Добавление новой метки на основе выделенной области
        add_label_from_selected_area(ref_point)

    elif event == cv2.EVENT_LBUTTONDOWN:
        ref_point = [(x, y)]
        cropping = True

    elif event == cv2.EVENT_LBUTTONUP:
        ref_point.append((x, y))
        cropping = False

        # Удаление меток внутри выбранной области
        remove_labels_within_selected_area(ref_point)

def remove_labels_within_selected_area(ref_point):
    global current_labels, image_copy

    x1, y1 = ref_point[0]
    x2, y2 = ref_point[1]

    # Убедимся, что координаты корректны
    x1, x2 = min(x1, x2), max(x1, x2)
    y1, y2 = min(y1, y2), max(y1, y2)

    img_h, img_w = image_copy.shape[:2]

    new_labels = []
    for label in current_labels:
        class_id, x_center, y_center, width, height = label
        x_center_pixel = int(x_center * img_w)
        y_center_pixel = int(y_center * img_h)

        # Проверяем, находится ли центр метки в выделенной области
        if not (x1 <= x_center_pixel <= x2 and y1 <= y_center_pixel <= y2):
            new_labels.append(label)
        else:
            print(f"Label removed: {label}")

    if len(new_labels) == len(current_labels):
        print("No labels removed.")
    else:
        print(f"Labels after removal: {len(new_labels)}")

    current_labels = new_labels

def add_label_from_selected_area(ref_point):
    global current_labels, image_copy

    img_h, img_w = image_copy.shape[:2]
    x1, y1 = ref_point[0]
    x2, y2 = ref_point[1]

    # Убедимся, что координаты корректны
    x1, x2 = min(x1, x2), max(x1, x2)
    y1, y2 = min(y1, y2), max(y1, y2)

    # Вычисляем центр и размеры прямоугольника в нормализованных координатах
    x_center = (x1 + x2) / 2 / img_w
    y_center = (y1 + y2) / 2 / img_h
    width = abs(x2 - x1) / img_w
    height = abs(y2 - y1) / img_h

    class_id = 0  # Класс 'player', по умолчанию класс 0 для YOLOv9
    new_label = (class_id, x_center, y_center, width, height)
    current_labels.append(new_label)

    print(f"Added label: {new_label}")

def save_labels(label_path):
    global current_labels

    with open(label_path, 'w') as file:
        for label in current_labels:
            # Сохранение метки в формате YOLO: class_id x_center y_center width height
            file.write(f"{int(label[0])} {label[1]:.6f} {label[2]:.6f} {label[3]:.6f} {label[4]:.6f}\n")

def browse_images(image_folder, label_folder):
    global image_copy

    # Получение списка всех изображений в папке
    image_files = [f for f in os.listdir(image_folder) if f.endswith(('.png', '.jpg', '.jpeg'))]
    image_files.sort()  # Сортируем для последовательного просмотра
    current_image_index = 0

    while 0 <= current_image_index < len(image_files):
        image_path = os.path.join(image_folder, image_files[current_image_index])
        label_name = os.path.splitext(image_files[current_image_index])[0] + '.txt'
        label_path = os.path.join(label_folder, label_name)

        # Загрузка изображения и меток
        image, labels = draw_labels_on_image(image_path, label_path)
        image_copy = image.copy()

        if image is not None:
            cv2.imshow('Image with Labels', image_copy)
            cv2.setMouseCallback("Image with Labels", click_and_crop)

        key = cv2.waitKey(0)

        if key == ord('d'):
            # Сохранение изменений и переход к следующему изображению
            save_labels(label_path)
            current_image_index += 1
        elif key == ord('a'):
            # Сохранение изменений и переход к предыдущему изображению
            save_labels(label_path)
            current_image_index -= 1
        elif key == ord('h'):
            # Удаление текущего изображения и файла меток
            os.remove(image_path)
            if os.path.exists(label_path):
                os.remove(label_path)
            print(f"Deleted: {image_path} and {label_path}")
            current_image_index += 1
        elif key == ord('q'):
            # Выход из программы
            break

    cv2.destroyAllWindows()

# Пример использования
image_folder = 'D://Projects//Python//DotaDetection//dataSet//Apex//ArrowUnpackedFace//train//images'  # Путь к папке с изображениями
label_folder = 'D://Projects//Python//DotaDetection//dataSet//Apex//ArrowUnpackedFace//train//labels'  # Путь к папке с метками

browse_images(image_folder, label_folder)
