import argparse
import time
import numpy as np
from scipy.fftpack import dctn, idctn

from zigzag import zigzag, reverse_zigzag


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

    blocks = []

    index = 0
    ws = window_size
    for i in range(height // ws):
        for j in range(width // ws):
            blocks.append(img[i * ws : i * ws + ws, j * ws : j * ws + ws])
            index += 1

    return blocks


def find_shortest_components(block, acceptable_error):
    block_dct = dctn(block, norm="ortho")
    block_dct_zigzag, positions = zigzag(block_dct)
    prev_block_dct_components = None
    for i in range(len(block_dct_zigzag), 0, -1):
        block_dct_components = reverse_zigzag(block_dct_zigzag, positions, i)
        block_idct = idctn(block_dct_components, norm="ortho")
        error = np.max(np.abs(block - block_idct))
        if error > acceptable_error:
            return prev_block_dct_components
        prev_block_dct_components = block_dct_zigzag[:i]
    return prev_block_dct_components


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
        help=(
            "Dokładność dekompresji. Liczba w wartości bewzględnej, np. 5cm, "
            "oznacza to, że po dekompresji w żadnym punkcie błąd nie "
            "przekroczy tej wartości."
        ),
        type=int,
    )
    parser.add_argument(
        "--zip",
        help="Czy na końcu dodatkowo kompresujemy dane metodą ZIP",
        action="store_true",
    )
    args = parser.parse_args()

    if args.i is None:
        print("Podaj nazwę pliku wejściowego: -i <nazwa pliku>")
        return None
    else:
        file_path = args.i

    if args.block_size is None:
        block_size = 8
    else:
        block_size = args.block_size

    if args.decompression_acc is None:
        decompression_acc = 5
    else:
        decompression_acc = args.decompression_acc

    return file_path, block_size, decompression_acc


def main():
    # Argumenty linii komend
    file_path, block_size, decompression_acc = command_line_arguments()

    # Dzialanie
    X, Y, Z = load_data(file_path)

    # Podzial na bloki
    blocks = to_blocks(Z, block_size)

    # DCT w blokach
    dct_components = []
    for block in blocks:
        has_none = np.any(np.isnan(block))
        if not has_none:
            components = find_shortest_components(
                block, acceptable_error=decompression_acc
            )
            dct_components.append(components)
        else:
            dct_components.append(np.nan)


if __name__ == "__main__":
    main()
