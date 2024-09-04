import cv2
import os

# Путь к видео
video_path = 'D://Videos//Dota 2//Apex Legends//Desktop.mp4'

# Путь к папке, где будут сохраняться кадры
output_folder = 'output_frames'
if not os.path.exists(output_folder):
    os.makedirs(output_folder)

# Интервал между кадрами (в кадрах)
frame_interval = 60  # например, сохранять каждый 20-й кадр

# Открытие видео
cap = cv2.VideoCapture(video_path)

# Получение общего количества кадров в видео
total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
fps = int(cap.get(cv2.CAP_PROP_FPS))
print(f'Total frames in video: {total_frames}')
print(f'Frames per second (FPS): {fps}')

frame_count = 0
saved_frame_count = 0

while cap.isOpened():
    # Явное перемещение на следующий кадр
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_count)

    ret, frame = cap.read()

    if not ret:
        break

    # Сохранение кадра только если номер кадра делится на интервал без остатка
    if frame_count % frame_interval == 0:
        frame_name = os.path.join(output_folder, f'frame_{saved_frame_count:05d}.jpg')
        cv2.imwrite(frame_name, frame)
        saved_frame_count += 1

    frame_count += frame_interval  # Увеличение счетчика на значение интервала

    # Отображение прогресса
    if frame_count % (frame_interval * 10) == 0:
        print(f'Processed {frame_count}/{total_frames} frames')

cap.release()
cv2.destroyAllWindows()

print(f'Done! Extracted {saved_frame_count} frames with an interval of {frame_interval} frames.')
