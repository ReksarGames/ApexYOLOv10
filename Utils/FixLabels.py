import os


def replace_labels_in_files(labels_dir):
    for root, _, files in os.walk(labels_dir):
        for file_name in files:
            if file_name.endswith('.txt'):
                file_path = os.path.join(root, file_name)
                with open(file_path, 'r') as file:
                    lines = file.readlines()

                # Заменяем метку 1 на 0
                new_lines = [line.replace('1 ', '0 ') for line in lines]

                # Записываем обратно в файл
                with open(file_path, 'w') as file:
                    file.writelines(new_lines)


if __name__ == "__main__":
    # Укажите путь к папке, содержащей ваши файлы меток
    labels_dir = 'D://Projects//Python//DotaDetection//dataSet//Apex//ManualOutput//train//labels'

    # Запускаем процесс замены меток
    replace_labels_in_files(labels_dir)

    print(f"Labels in directory '{labels_dir}' have been updated.")
