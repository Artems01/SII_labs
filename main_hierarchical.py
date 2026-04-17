import math
import time
import tracemalloc
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from kneed import KneeLocator
from sklearn.metrics import silhouette_score

# загрузка данных
data = pd.read_csv('tests/dataset_2.csv', sep=';', encoding='utf-8')

names = data['City'].tolist()
latitudes = data['Latitude'].tolist()
longitudes = data['Longitude'].tolist()
size = len(names)


# расстояние между точками
def distance(x1, y1, x2, y2):
    return math.sqrt((x1 - x2) ** 2 + (y1 - y2) ** 2)


# матрица расстояний
def create_matrix(n, lat, lon):
    matrix = [[0.0] * n for _ in range(n)]

    for i in range(n):
        for j in range(i + 1, n):
            d = distance(lat[i], lon[i], lat[j], lon[j])
            matrix[i][j] = d
            matrix[j][i] = d

    return matrix


# иерархическая кластеризация (complete linkage)
def agglomerative_clustering(k, matrix, n):
    clusters = [[i] for i in range(n)]

    while len(clusters) > k:
        best_dist = float('inf')
        pair = (0, 1)

        for i in range(len(clusters)):
            for j in range(i + 1, len(clusters)):
                current = max(matrix[a][b] for a in clusters[i] for b in clusters[j])

                if current < best_dist:
                    best_dist = current
                    pair = (i, j)

        i, j = pair
        clusters[i].extend(clusters[j])
        clusters.pop(j)

    return clusters


# инерция
def inertia(clusters, lat, lon):
    total = 0.0

    for cluster in clusters:
        if not cluster:
            continue

        center_x = sum(lat[i] for i in cluster) / len(cluster)
        center_y = sum(lon[i] for i in cluster) / len(cluster)

        total += sum(
            distance(lat[i], lon[i], center_x, center_y) ** 2
            for i in cluster
        )

    return total


# --- основной блок ---
dist_matrix = create_matrix(size, latitudes, longitudes)

k_values = list(range(1, min(10, size) + 1))
inertia_values = [
    inertia(agglomerative_clustering(k, dist_matrix, size), latitudes, longitudes)
    for k in k_values
]

kneedle = KneeLocator(k_values, inertia_values, curve='convex', direction='decreasing')
optimal_k = kneedle.elbow if kneedle.elbow else 2

print(f"Оптимальное K: {optimal_k}")

# замеры
tracemalloc.start()
start = time.perf_counter()

clusters = agglomerative_clustering(optimal_k, dist_matrix, size)

elapsed = time.perf_counter() - start
_, peak = tracemalloc.get_traced_memory()
tracemalloc.stop()

# вывод
print(f"\nРаспределение точек на {optimal_k} кластеров:\n")

for i, cluster in enumerate(clusters):
    cluster_names = [names[idx] for idx in cluster]
    print(f"Кластер {i + 1}: {', '.join(cluster_names)} ({len(cluster_names)} элементов)")

print(f"\nВремя выполнения: {elapsed:.6f} сек.")
print(f"Пиковая память: {peak / 1024:.4f} KB")

# силуэт
labels = [0] * size
for i, cluster in enumerate(clusters):
    for idx in cluster:
        labels[idx] = i

# --- График локтя ---
plt.figure(figsize=(8, 5))

plt.plot(k_values, inertia_values, marker='o', linestyle='-', linewidth=2)

plt.axvline(optimal_k, linestyle='--', label=f'K = {optimal_k}')

plt.xlabel('Количество кластеров')
plt.ylabel('Инерция')
plt.title('Определение оптимального числа кластеров')

plt.xticks(k_values)
plt.legend()
plt.grid(True, linestyle=':', alpha=0.6)

plt.tight_layout()
plt.show()

# --- Визуализация кластеров ---
palette = plt.cm.tab10(np.linspace(0, 1, optimal_k))

plt.figure(figsize=(8, 6))

for idx, cluster in enumerate(clusters):
    x_coords = [longitudes[i] for i in cluster]
    y_coords = [latitudes[i] for i in cluster]

    plt.scatter(x_coords, y_coords, s=90, color=palette[idx], label=f'Кластер {idx + 1}')

    # подписи городов
    for i in cluster:
        plt.annotate(
            names[i],
            (longitudes[i], latitudes[i]),
            textcoords="offset points",
            xytext=(4, 4),
            fontsize=7
        )

plt.xlabel('Долгота')
plt.ylabel('Широта')
plt.title(f'Результат кластеризации (K = {optimal_k})')

plt.legend()
plt.grid(True, alpha=0.4)

plt.tight_layout()
plt.show()
points = np.array(list(zip(latitudes, longitudes)))
print(f"Коэффициент силуэта: {silhouette_score(points, labels):.4f}")