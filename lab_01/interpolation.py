"""
TODO:
    - [x] get parameters from user
    - [x] changing circle to rectangle
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
import argparse


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


def moving_average(data, tree, xx, yy, min_n_points, window_size, window_type):
    zz = np.full_like(xx, np.nan)

    for i in tqdm.tqdm(range(xx.shape[0]), desc="Interpolacja MA"):
        for j in range(xx.shape[1]):
            current_point = [xx[i, j], yy[i, j]]
            if window_type == "rect":
                nearby_points = tree.query_ball_point(current_point, window_size, p=np.inf)
            elif window_type == "circle":
                nearby_points = tree.query_ball_point(current_point, window_size, p=2.0)
            if len(nearby_points) > min_n_points:
                zz[i, j] = np.mean(data[nearby_points, 2])

    return zz


def idw(data, tree, xx, yy, min_n_points, window_size, idw_exponent, window_type):
    zz = np.full_like(xx, np.nan)
    from scipy.spatial.distance import cdist

    for i in tqdm.tqdm(range(xx.shape[0]), desc="Interpolacja IDW"):
        for j in range(xx.shape[1]):
            current_point = [xx[i, j], yy[i, j]]
            if window_type == "rect":
                nearby_points = tree.query_ball_point(current_point, window_size, p=np.inf)
            elif window_type == "circle":
                nearby_points = tree.query_ball_point(current_point, window_size, p=2.0)

            distances = cdist([current_point], data[nearby_points][:, 0:2])

            if len(nearby_points) > min_n_points:
                denominator = distances[0] ** idw_exponent + 1e-10
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
    # Wczytywanie argumentów wywołania
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", help="plik wejściowy z współrzędnymi XYZ")
    parser.add_argument("--plot", help="czy wyświetlać wizualizacje. Domyślnie false.", action="store_true")
    parser.add_argument("--spacing", help="odległość pomiędzy punktami w interpolacji. Domyślnie 1.", type=float)
    parser.add_argument("--min_n_points", help="minimalna liczba punktów do interpolacji komórki. Domyślnie 1.", type=int)
    parser.add_argument("--window_type", help="rodzaj okna do interpolacji. rect lub circle. Domyślnie circle")
    parser.add_argument("--window_size", help="rozmiar okna do wyszukiwania najbliższych punktów. Domyślnie 1.", type=float)
    parser.add_argument("-o", help="plik wynikowy ASCII XYZ. Domyślnie brak zapisu do pliku.")
    parser.add_argument("--method", help="metoda interpolacji. idw, ma lub both. Domyślnie ma (moving_average)")
    parser.add_argument("--idw_exponent", help="wykładnik w metodzie idw. Domyślnie 2")

    args = parser.parse_args()

    if args.i == None:
        print("Podaj nazwę pliku wejściowego: -i <nazwa pliku>")
    else:
        file_path = args.i

    if args.spacing == None:
        print("Ustawiam wartość spacing na 1")
        spacing = 1
    else:
        spacing = args.spacing

    if args.min_n_points == None:
        print("Ustawiam wartość min_n_points na 1")
        min_n_points = 1
    else:
        min_n_points = args.min_n_points

    if args.window_type == None:
        print("Ustawiam rodzaj okna na okrąg")
        window_type = "circle"
    else:
        window_type = args.window_type

    if args.window_size == None:
        print("Ustawiam rozmiar okna na 1")
        window_size = 1
    else:
        window_size = args.window_size

    method = args.method
    if method not in ["ma", "idw", "both"]:
        print("Ustawiam metodę na ma")
        method = "ma"

    idw_exponent = None
    if method == "idw" or method == "both":
        if args.idw_exponent == None:
            print("Ustawiam wykładnik w metodzie idw na 2")
            idw_exponent = 2
        else:
            idw_exponent = args.idw_exponent

    # Wczytywanie danych
    t1 = time.time()
    print("Wczytywanie danych...")
    data = np.loadtxt(file_path)
    print("Wczytano dane w:", time.time() - t1, "sekund")

    # Tworzenie siatki i struktury
    xx, yy, x, y = grid(data, spacing)

    tree = structure(data, x, y, window_size)

    # Przeprowadzanie interpolacji
    if method in ["ma", "both"]:
        zz = moving_average(data, tree, xx, yy, min_n_points, window_size, window_type)

        if plot:
            plot(xx, yy, zz)

    if method in ["idw", "both"]:
        zz = idw(data, tree, xx, yy, min_n_points, window_size, idw_exponent, window_type)

        if plot:
            plot(xx, yy, zz)


if __name__ == "__main__":
    main()

    """
    Przykładowe uruchomienie:
    python .\interpolation.py -i "./data/wraki_utm.txt" --plot --window_size=1 --spacing=0.2 --min_n_points=1 --window_type=rect  --method=idw
    python .\interpolation.py -i "./data/wraki_utm.txt" --plot --window_size=1 --spacing=0.2 --min_n_points=1 --window_type=rect  --method=ma
    """
