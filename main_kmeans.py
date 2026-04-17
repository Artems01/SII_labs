import math
import random
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


# расстояние
def dist(x1, y1, x2, y2):
    return math.sqrt((x1 - x2) ** 2 + (y1 - y2) ** 2)


# KMeans
def run_kmeans(k, lat, lon, n, max_iter=100, eps=1e-4):
    # случайные центры
    indices = random.sample(range(n), k)
    centers_x = [lat[i] for i in indices]
    centers_y = [lon[i] for i in indices]

    clusters = {i: [] for i in range(k)}

    for _ in range(max_iter):
        new_clusters = {i: [] for i in range(k)}

        # распределение по кластерам
        for i in range(n):
            best_cluster = min(
                range(k),
                key=lambda c: dist(lat[i], lon[i], centers_x[c], centers_y[c])
            )
            new_clusters[best_cluster].append(i)

        # пересчет центров
        new_x, new_y = [], []
        for i in range(k):
            if len(new_clusters[i]) == 0:
                new_x.append(centers_x[i])
                new_y.append(centers_y[i])
            else:
                new_x.append(sum(lat[j] for j in new_clusters[i]) / len(new_clusters[i]))
                new_y.append(sum(lon[j] for j in new_clusters[i]) / len(new_clusters[i]))

        # проверка сходимости
        shift = max(
            dist(centers_x[i], centers_y[i], new_x[i], new_y[i])
            for i in range(k)
        )

        centers_x, centers_y = new_x, new_y
        clusters = new_clusters

        if shift < eps:
            break

    return clusters, centers_x, centers_y


# инерция
def calc_inertia(clusters, lat, lon):
    total = 0.0

    for cluster in clusters.values():
        if not cluster:
            continue

        cx = sum(lat[i] for i in cluster) / len(cluster)
        cy = sum(lon[i] for i in cluster) / len(cluster)

        total += sum(dist(lat[i], lon[i], cx, cy) ** 2 for i in cluster)

    return total


# --- метод локтя ---
k_values = list(range(1, min(10, size) + 1))

inertia_values = [
    min(calc_inertia(run_kmeans(k, latitudes, longitudes, size)[0], latitudes, longitudes)
        for _ in range(5))
    for k in k_values
]

kneedle = KneeLocator(k_values, inertia_values, curve='convex', direction='decreasing')
optimal_k = kneedle.elbow if kneedle.elbow else 2

print(f"Оптимальное K: {optimal_k}")


# --- замеры ---
tracemalloc.start()
start = time.perf_counter()

clusters, centers_x, centers_y = run_kmeans(optimal_k, latitudes, longitudes, size)

elapsed = time.perf_counter() - start
_, peak = tracemalloc.get_traced_memory()
tracemalloc.stop()


# --- вывод ---
print(f"\nРаспределение точек на {optimal_k} кластеров:\n")
print("\nКластеры:\n")
for i in range(optimal_k):
    cluster_names = [names[j] for j in clusters[i]]
    print(f"{i + 1}: {', '.join(cluster_names)} ({len(cluster_names)})")

print(f"\nВремя выполнения: {elapsed:.6f} сек.")
print(f"Пиковая память: {peak / 1024:.4f} KB")


# --- силуэт ---
labels = [0] * size
for i, cluster in clusters.items():
    for idx in cluster:

        labels[idx] = i

points = np.array(list(zip(latitudes, longitudes)))
print(f"Коэффициент силуэта: {silhouette_score(points, labels):.4f}")


# --- график локтя ---
plt.figure(figsize=(8, 5))
plt.plot(k_values, inertia_values, marker='o')

plt.axvline(optimal_k, linestyle='--', label=f'K = {optimal_k}')

plt.xlabel('K')
plt.ylabel('Inertia')
plt.title('Метод локтя (KMeans)')

plt.xticks(k_values)
plt.legend()
plt.grid(True, linestyle=':')

plt.tight_layout()
plt.show()


# --- график кластеров ---
colors = plt.cm.tab10(np.linspace(0, 1, optimal_k))

plt.figure(figsize=(8, 6))

for i in range(optimal_k):
    pts = clusters[i]

    plt.scatter(
        [longitudes[j] for j in pts],
        [latitudes[j] for j in pts],
        s=90,
        color=colors[i],
        label=f'Кластер {i + 1}'
    )

    for j in pts:
        plt.annotate(names[j], (longitudes[j], latitudes[j]), fontsize=7, xytext=(4, 4),
                     textcoords="offset points")

# центроиды
plt.scatter(centers_y, centers_x, marker='X', s=200, label='Центры')

plt.xlabel('Долгота')
plt.ylabel('Широта')
plt.title(f'KMeans (K = {optimal_k})')

plt.legend()
plt.grid(True, alpha=0.4)

plt.tight_layout()
plt.show()