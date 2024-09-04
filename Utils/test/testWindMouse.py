import matplotlib.pyplot as plt
import random
import math

def wind_mouse(x_start, y_start, x_end, y_end, wind_strength=3, gravity=19, max_steps=100, target_area=5):
    x = x_start
    y = y_start
    wind_x = 0
    wind_y = 0
    velocity_x = 0
    velocity_y = 0
    max_velocity = 2
    min_velocity = 13

    path = [(x, y)]  # Сохранение пути для построения графика

    while math.sqrt((x_end - x) ** 2 + (y_end - y) ** 2) > target_area:
        x_diff = x_end - x
        y_diff = y_end - y
        distance = math.sqrt(x_diff ** 2 + y_diff ** 2)

        wind_x = wind_x / math.sqrt(3) + (random.random() * (wind_strength * 2 + 1) - wind_strength) / math.sqrt(5)
        wind_y = wind_y / math.sqrt(3) + (random.random() * (wind_strength * 2 + 1) - wind_strength) / math.sqrt(5)

        velocity_x = (velocity_x + wind_x) + gravity * (x_diff) / distance
        velocity_y = (velocity_y + wind_y) + gravity * (y_diff) / distance

        if math.sqrt(velocity_x ** 2 + velocity_y ** 2) > max_velocity:
            random_dist = max_velocity / 2.0 + random.random() * max_velocity / 2.0
            velocity_x = (velocity_x / math.sqrt(velocity_x ** 2 + velocity_y ** 2)) * random_dist
            velocity_y = (velocity_y / math.sqrt(velocity_x ** 2 + velocity_y ** 2)) * random_dist

        if math.sqrt(velocity_x ** 2 + velocity_y ** 2) < min_velocity:
            random_dist = min_velocity / 2.0 + random.random() * min_velocity / 2.0
            velocity_x = (velocity_x / math.sqrt(velocity_x ** 2 + velocity_y ** 2)) * random_dist
            velocity_y = (velocity_y / math.sqrt(velocity_x ** 2 + velocity_y ** 2)) * random_dist

        x += velocity_x
        y += velocity_y
        path.append((x, y))

    return path

# Количество генерируемых линий
num_lines = 20

plt.figure(figsize=(10, 10))
for _ in range(num_lines):
    x_start = 0
    y_start = random.uniform(0, 100)
    x_end = 100
    y_end = y_start + random.uniform(-10, 10)
    path = wind_mouse(x_start, y_start, x_end, y_end)

    x_vals, y_vals = zip(*path)
    plt.plot(x_vals, y_vals)

plt.axis('off')
plt.show()
