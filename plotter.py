"""
Program wyświetlający dane wygenerowane w procesie interpolacji.


Przykładowe uruchomienie programu:

    python .\plotter.py -i ".\data\output\wraki_utm_idw.txt"

Wyświetli wizualizację dla interpolowanych danych z pliku "wraki_utm_idw.txt"
"""

import argparse
from interpolation import plot
import time
import numpy as np


def main():
    # Parametry programu
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", help="plik wejściowy z współrzędnymi XYZ")
    parser.add_argument(
        "--pickle",
        help="czy wczytac plik binarny. Domyślnie false.",
        action="store_true",
    )
    args = parser.parse_args()

    if args.i == None:
        print("Podaj nazwę pliku wejściowego: -i <nazwa pliku>")
        return None
    else:
        file_path = args.i

    # Wczytanie danych
    t1 = time.time()
    print("Wczytywanie danych...")
    if args.pickle:
        from interpolation import load_binary

        Z = load_binary(file_path)
        x = np.arange(Z.shape[1])
        y = np.arange(Z.shape[0])
        X, Y = np.meshgrid(x, y)
    else:
        data = np.loadtxt(file_path, skiprows=1)

        # Tworzenie grida
        x = np.unique(data[:, 0])
        y = np.unique(data[:, 1])
        z = data[:, 2]

        X, Y = np.meshgrid(x, y)
        Z = z.reshape(X.shape)
    print("Wczytano dane w:", time.time() - t1, "sekund")

    # Wyswietlanie danych
    plot(X, Y, Z)


if __name__ == "__main__":
    main()
