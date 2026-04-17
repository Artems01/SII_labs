import time
import tracemalloc
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from kneed import KneeLocator
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

# загрузка данных
data = pd.read_csv('tests/dataset_2.csv', sep=';', encoding='utf-8')

names = data['City'].tolist()
points = data[['Latitude', 'Longitude']].values
size = len(names)

# --- метод локтя ---
k_values = list(range(1, min(10, size) + 1))

inertia_values = []
for k in k_values:
    model = KMeans(n_clusters=k, random_state=42, n_init=10)
    model.fit(points)
    inertia_values.append(model.inertia_)

# поиск оптимального K
kneedle = KneeLocator(k_values, inertia_values, curve='convex', direction='decreasing')
optimal_k = kneedle.elbow if kneedle.elbow else 2

print(f"Оптимальное K: {optimal_k}")

# --- замеры ---
tracemalloc.start()
start = time.perf_counter()

model = KMeans(n_clusters=optimal_k, random_state=42, n_init=10)
labels = model.fit_predict(points)

elapsed = time.perf_counter() - start
_, peak = tracemalloc.get_traced_memory()
tracemalloc.stop()

centers = model.cluster_centers_

print(f"\nРаспределение точек на {optimal_k} кластеров:\n")

for i in range(optimal_k):
    cluster_names = [names[j] for j in range(size) if labels[j] == i]
    count = len(cluster_names)
    word = "элемент" if count == 1 else "элементов"
    print(f"Кластер {i + 1}: {', '.join(cluster_names)} ({count} {word})")

print(f"\nВремя выполнения: {elapsed:.6f} сек.")
print(f"Пиковая память: {peak / 1024:.4f} KB")

print(f"Коэффициент силуэта: {silhouette_score(points, labels):.4f}")

# --- график локтя ---
plt.figure(figsize=(8, 5))
plt.plot(k_values, inertia_values, marker='o')

plt.axvline(optimal_k, linestyle='--', label=f'K = {optimal_k}')

plt.xlabel('K')
plt.ylabel('Inertia')
plt.title('Метод локтя (sklearn KMeans)')

plt.xticks(k_values)
plt.legend()
plt.grid(True, linestyle=':')

plt.tight_layout()
plt.show()

# --- график кластеров ---
colors = plt.cm.tab10(np.linspace(0, 1, optimal_k))

plt.figure(figsize=(8, 6))

for i in range(optimal_k):
    cluster_points = points[labels == i]

    plt.scatter(
        cluster_points[:, 1],
        cluster_points[:, 0],
        s=90,
        color=colors[i],
        label=f'Кластер {i + 1}'
    )

    for j in range(size):
        if labels[j] == i:
            plt.annotate(
                names[j],
                (points[j][1], points[j][0]),
                fontsize=7,
                xytext=(4, 4),
                textcoords="offset points"
            )

# центры
plt.scatter(
    centers[:, 1],
    centers[:, 0],
    marker='X',
    s=200,
    color='black',
    label='Центры'
)

plt.xlabel('Долгота')
plt.ylabel('Широта')
plt.title(f'Sklearn KMeans (K = {optimal_k})')

plt.legend()
plt.grid(True, alpha=0.4)

plt.tight_layout()
plt.show()