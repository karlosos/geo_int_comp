"""
TODO:
    - [ ] get parameters from user
    - [ ] changing circle to rectangle
    - [ ] saving to ASCII XYZ
    - [ ] load in qgis
"""

import pandas as pd
import numpy as np
import tqdm
import matplotlib
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
from scipy.spatial import cKDTree
import time


def load_data():
    df = pd.read_csv("./data/wraki_utm.txt", delimiter=" ", header=None)
    df = df.drop(df.columns[[0, 1, 3]], axis=1)
    df.columns = ["X", "Y", "Z"]
    return df


def grid(df, spacing):
    # Wyznaczenie punktów skrajnych
    min_x = np.min(df[:, 0])
    max_x = np.max(df[:, 0])
    min_y = np.min(df[:, 1])
    max_y = np.max(df[:, 1])

    print("Zakresy zmiennych:")
    print(f"X: od {min_x} do {max_x}, odległość: {max_x - min_x}m")
    print(f"Y: od {min_y} do {max_y}, odległość: {max_y - min_y}m")
    print("")

    # Stworzenie siatki
    x = np.arange(min_x, max_x, spacing)
    y = np.arange(min_y, max_y, spacing)
    xx, yy = np.meshgrid(x, y)
    print("Rozmiar siatki grid:", len(x), len(y))

    return xx, yy, x, y


def structure(data, x, y, window_size):
    print("Tworzenie struktury przechowującej punkty")
    t1 = time.time()
    tree = cKDTree(data[:, 0:2])
    print("Czas na stworzenie struktury:", time.time() - t1)

    return tree


def moving_average(data, tree, xx, yy, min_n_points, window_size):
    zz = np.full_like(xx, np.nan)

    for i in tqdm.tqdm(range(xx.shape[0]), desc="Interpolacja MA"):
        for j in range(xx.shape[1]):
            nearby_points = tree.query_ball_point([xx[i, j], yy[i, j]], window_size)
            if len(nearby_points) > min_n_points:
                zz[i, j] = np.mean(data[nearby_points, 2])

    return zz


def idw(data, tree, xx, yy, min_n_points, window_size):
    zz = np.full_like(xx, np.nan)
    from scipy.spatial.distance import cdist

    for i in tqdm.tqdm(range(xx.shape[0]), desc="Interpolacja IDW"):
        for j in range(xx.shape[1]):
            current_point = [xx[i, j], yy[i, j]]
            nearby_points = tree.query_ball_point(current_point, window_size)
            distances = cdist([current_point], data[nearby_points][:, 0:2])

            if len(nearby_points) > min_n_points:
                denominator = distances[0] ** 2 + 1e-10
                zz[i, j] = (np.sum(data[nearby_points, 2] / denominator)) / (
                    np.sum(np.ones_like(denominator) / denominator)
                )

    return zz


def plot(xx, yy, zz):
    fig = plt.figure()

    ax1 = fig.add_subplot(121, projection="3d")
    ax1.plot_surface(xx, yy, zz, color="b")

    ax2 = fig.add_subplot(122)
    depth_map = ax2.imshow(zz, interpolation="none")
    plt.colorbar(depth_map)

    plt.show()


def main():
    spacing = 0.2
    min_n_points = 1
    window_size = 0.4
    file_path = "./data/wraki_utm.txt"

    t1 = time.time()
    print("Wczytywanie danych...")
    data = np.loadtxt(file_path)
    print("Wczytano dane w:", time.time() - t1, "sekund")

    xx, yy, x, y = grid(data, spacing)

    tree = structure(data, x, y, window_size)

    zz = moving_average(data, tree, xx, yy, min_n_points, window_size)
    plot(xx, yy, zz)

    zz = idw(data, tree, xx, yy, min_n_points, window_size)
    plot(xx, yy, zz)


if __name__ == "__main__":
    main()
