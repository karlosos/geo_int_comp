import pandas as pd
import numpy as np


def load_data():
    df = pd.read_csv("./data/wraki_utm.txt", delimiter=" ", header=None)
    df = df.drop(df.columns[[0, 1, 3]], axis=1)
    df.columns = ["X", "Y", "Z"]
    return df


def grid(df, grid_step):
    min_x = np.min(df[:, 0])
    max_x = np.max(df[:, 0])
    min_y = np.min(df[:, 1])
    max_y = np.max(df[:, 1])

    print("Zakresy zmiennych:")
    print(f"X: od {min_x} do {max_x}")
    print(f"Y: od {min_y} do {max_y}")
    print("")

    # Tworzenie siatki
    x = np.arange(min_x, max_x, grid_step)
    y = np.arange(min_y, max_y, grid_step)
    xx, yy = np.meshgrid(x, y)
    print("Rozmiar siatki grid:", len(x), len(y))

    return xx, yy


def main():
    df = load_data()
    print(df.head())
    print("")
    data = df.to_numpy()

    # TODO: get those variables from user
    # either input or create gui
    grid_step = 0.5  # odleglości pomiędzy siatką
    min_n_points = 1  # minimalna liczba punktów na siatkę
    window_size = 2  # określenie rozmiaru okna

    xv, yv = grid(data, grid_step)
    print(xv.shape)


if __name__ == "__main__":
    main()
