import pandas as pd
import numpy as np
import tqdm
import matplotlib
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt


def load_data():
    df = pd.read_csv("./data/wraki_utm.txt", delimiter=" ", header=None)
    df = df.drop(df.columns[[0, 1, 3]], axis=1)
    df.columns = ["X", "Y", "Z"]
    return df


def grid(df, spacing):
    min_x = np.min(df[:, 0])
    max_x = np.max(df[:, 0])
    min_y = np.min(df[:, 1])
    max_y = np.max(df[:, 1])

    print("Zakresy zmiennych:")
    print(f"X: od {min_x} do {max_x}, odległość: {max_x - min_x}m")
    print(f"Y: od {min_y} do {max_y}, odległość: {max_y - min_y}m")
    print("")

    # Tworzenie siatki
    x = np.arange(min_x, max_x, spacing)
    y = np.arange(min_y, max_y, spacing)
    xx, yy = np.meshgrid(x, y)
    print("Rozmiar siatki grid:", len(x), len(y))

    return xx, yy


def moving_average(data, xx, yy, min_n_points, window_size):
    zz = np.full_like(xx, np.nan)

    for i in tqdm.tqdm(range(xx.shape[0]), desc=" interpolation"):
        for j in range(xx.shape[1]):
            # TODO: wylicz wartosc Z - interpolacja ;)
            # Należy znaleźć takie punkty które są w zasięgu xx[i, j], yy[i, j]
            # Odległe maksymalnie o window_size
            # Przygotowac macierz z odległościami?
            #   Dla każdej możliwej kombinacji i, j obliczyć odległości wszystkich punktów
            # Uzyc cdist https://docs.scipy.org/doc/scipy/reference/generated/scipy.spatial.distance.cdist.html
            # Moze przed petla?
            zz[i, j] = 0

    return zz


def plot_surface(X, Y, Z):
    fig = plt.figure()
    ax = fig.gca(projection="3d")

    X, Y = np.meshgrid(X, Y)

    surf = ax.plot_surface(X, Y, Z, cmap=cm.coolwarm, linewidth=0, antialiased=False)

    fig.colorbar(surf, shrink=0.5, aspect=5)
    plt.show()


def main():
    df = load_data()
    print(df.head())
    print("")
    data = df.to_numpy()

    # Spytanie użytkownika o parametry
    spacing = float(input("Rozdzielczość: "))
    min_n_points = int(input("Minimalna liczba punktów na siatkę: "))
    window_size = float(input("Rozmiar okna interpolacji: "))

    # spacing = 0.5
    # min_n_points = 1
    # window_size = 1

    xx, yy = grid(data, spacing)
    print(xx.shape)

    zz = moving_average(data, xx, yy, min_n_points, window_size)


if __name__ == "__main__":
    main()
