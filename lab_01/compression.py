"""
- Jeżeli w bloku jest jeden nan to pomijam cały blok (blok pusty)
- Kompresuje tylko bloki które są zupełnie pełne
- Nie będą ładne brzegi tylko poszarpane w kwadraty
"""

import argparse
from interpolation import plot
import time
import numpy as np
from scipy.fftpack import dct, idct
from scipy.fftpack import dctn, idctn


def load_data(file_path):
    # Wczytanie danych
    t1 = time.time()
    print("Wczytywanie danych...")
    data = np.loadtxt(file_path, skiprows=1)
    print("Wczytano dane w:", time.time() - t1, "sekund")

    # Tworzenie grida
    x = np.unique(data[:, 0])
    y = np.unique(data[:, 1])
    z = data[:, 2]

    X, Y = np.meshgrid(x, y)
    Z = z.reshape(X.shape)
    return X, Y, Z


def blocks(Z):
    heigh, width = Z.shape

    for i in range(0, height, 5):
        for j in range(0, width, 5):
            Z[i : i + 5, j : j + 5]


def to_blocks(img, window_size):
    """
    Slice image into blocks of size window_size and flatten them.
    """

    height, width = img.shape
    print("Before padding:")
    print(height)
    print(width)

    if height % window_size != 0 or width % window_size != 0:
        width_padding = width + (window_size - (width % window_size))
        height_padding = height + (window_size - (height % window_size))
        img_padding = np.empty((height_padding, width_padding))
        img_padding[:] = np.nan
        img_padding[0:height, 0:width] = img
        img = img_padding
        height, width = img.shape

    num_of_vectors = height // window_size * width // window_size
    blocks = []

    index = 0
    ws = window_size
    for i in range(height // ws):
        for j in range(width // ws):
            blocks.append(img[i * ws : i * ws + ws, j * ws : j * ws + ws])
            index += 1

    return blocks


def command_line_arguments():
    # Parametry programu
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", help="plik wejściowy z danymi wejściowymi")
    parser.add_argument(
        "--block_size",
        help="Rozmiar bloku danych. Domyślnie blok 8x8",
        type=int,
    )
    parser.add_argument(
        "--decompression_acc",
        help="Dokładność dekompresji. Liczba w wartości bewzględnej, np. 5cm, oznacza to, że po dekompresji w żadnym punkcie błąd nie przekroczy tej wartości.",
        type=int,
    )
    parser.add_argument(
        "--zip",
        help="Czy na końcu dodatkowo kompresujemy dane metodą ZIP",
        action="store_true",
    )
    args = parser.parse_args()

    if args.i == None:
        print("Podaj nazwę pliku wejściowego: -i <nazwa pliku>")
        return None
    else:
        file_path = args.i

    if args.block_size == None:
        block_size = 8
    else:
        block_size = args.block_size

    if args.decompression_acc == None:
        decompression_acc = 5
    else:
        decompression_acc = args.decompression_acc

    return file_path, block_size, decompression_acc


def zig_zag_index(k, n):
    # upper side of interval
    if k >= n * (n + 1) // 2:
        i, j = zig_zag_index(n * n - 1 - k, n)
        return n - 1 - i, n - 1 - j

    # lower side of interval
    i = int((np.sqrt(1 + 8 * k) - 1) / 2)
    j = k - i * (i + 1) // 2
    return (j, i - j) if i & 1 else (i - j, j)


def zig_zag_value(i, j, n):
    # upper side of interval
    if i + j >= n:
        return n * n - 1 - zig_zag_value(n - 1 - i, n - 1 - j, n)
    # lower side of interval
    k = (i + j) * (i + j + 1) // 2
    return k + i if (i + j) & 1 else k + j


def main():
    # Argumenty linii komend
    file_path, block_size, decompression_acc = command_line_arguments()

    # Dzialanie
    X, Y, Z = load_data(file_path)

    # Podzial na bloki
    blocks = to_blocks(Z, block_size)

    # DCT w blokach
    dct_blocks = []
    for block in blocks:
        has_none = np.any(np.isnan(block))
        if not has_none:
            dct_blocks.append(dct(block))
        else:
            dct_blocks.append(np.empty((block_size, block_size)))

    # Kwantyzacja współczynników transformaty kosinusowej

    # Zamiana tablicy współczynników na wektor

    # Kodowanie entropijne wektora współczynników


def test_block_division():
    # Test block division
    img = np.arange(36).reshape(6, 6)
    block_size = 3
    blocks = to_blocks(img, window_size=block_size)
    dct_blocks = []
    for block in blocks:
        has_none = np.any(np.isnan(block))
        if not has_none:
            dct_blocks.append(dct(block, norm='ortho'))
        else:
            dct_blocks.append(np.full((block_size, block_size), np.nan))

    print("DCT:")
    print(dct_blocks)

def test_dct_idct():
    block_size = 5
    block = np.arange(block_size**2).reshape(block_size, block_size)
    print("Block input:")
    print(block)

    block_dct = dctn(block, norm='ortho')
    print("Block dct:")
    print(block_dct)

    # Ortho, żeby była normalizacja
    block_idct = idctn(block_dct, norm='ortho')
    np.set_printoptions(suppress=True)
    print("Block idct:")
    print(block_idct)

def test_dct_block_reduction():
    block_size = 3
    block = np.arange(block_size**2).reshape(block_size, block_size)
    print("Block input:")
    print(block)

    block_dct = dctn(block, norm='ortho')
    print("Block dct:")
    print(block_dct)
    
    # Reduction with rectangle
    reduction = 2
    block_dct_reduction = block_dct[0:reduction, 0:reduction]
    print("Block dct reduction:")
    # print(block_dct_reduction)
    block_dct_reduction_padded = np.zeros((block_size, block_size))
    block_dct_reduction_padded[0:reduction, 0:reduction] = block_dct_reduction
    print(block_dct_reduction_padded)

    block_idct = idctn(block_dct_reduction_padded, norm='ortho')
    np.set_printoptions(suppress=True)
    print("Block idct:")
    print(block_idct)


if __name__ == "__main__":
    # main()
    # test()
    # test_dct_idct()
    test_dct_block_reduction()
