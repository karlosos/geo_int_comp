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


def main():
    # Argumenty linii komend
    file_path, block_size, decompression_acc = command_line_arguments()

    # Dzialanie
    X, Y, Z = load_data(file_path)

    # Podzial na bloki
    blocks = to_blocks(Z, block_size)

    # DCT w blokach
    # TODO: find shortest len of components
    print("Blocks:")
    dct_blocks = []
    for block in blocks:
        has_none = np.any(np.isnan(block))
        if not has_none:
            print(block)
            breakpoint()
            dct_blocks.append(dct(block))
        else:
            dct_blocks.append(np.nan)


def test_block_division():
    # Test block division
    img = np.arange(36).reshape(6, 6)
    block_size = 3
    blocks = to_blocks(img, window_size=block_size)
    dct_blocks = []
    for block in blocks:
        has_none = np.any(np.isnan(block))
        if not has_none:
            dct_blocks.append(dct(block, norm="ortho"))
        else:
            dct_blocks.append(np.full((block_size, block_size), np.nan))

    print("DCT:")
    print(dct_blocks)


def test_dct_idct():
    block_size = 5
    block = np.arange(block_size ** 2).reshape(block_size, block_size)
    print("Block input:")
    print(block)

    block_dct = dctn(block, norm="ortho")
    print("Block dct:")
    print(block_dct)

    # Ortho, żeby była normalizacja
    block_idct = idctn(block_dct, norm="ortho")
    np.set_printoptions(suppress=True)
    print("Block idct:")
    print(block_idct)


def test_dct_block_reduction():
    """
    Reduction with rectangle
    """
    block_size = 3
    block = np.arange(block_size ** 2).reshape(block_size, block_size)
    print("Block input:")
    print(block)

    block_dct = dctn(block, norm="ortho")
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

    block_idct = idctn(block_dct_reduction_padded, norm="ortho")
    np.set_printoptions(suppress=True)
    print("Block idct:")
    print(block_idct)

def test_dct_block_reduction_triangle():
    block_size = 3
    block = np.arange(block_size ** 2).reshape(block_size, block_size)
    print("Block input:")
    print(block)

    block_dct = dctn(block, norm="ortho")
    print("Block dct:")
    print(block_dct)

    # Reduction with rectangle
    print("Block dct reduction:")
    # print(block_dct_reduction)
    # Will have to control k parameter
    # k = 0 is half
    # k = block_size - 1 is max
    block_dct_reduction_padded = np.flip(np.triu(np.flip(block_dct, axis=1), k=-(block_size)), axis=1)
    print(block_dct_reduction_padded)

    block_idct = idctn(block_dct_reduction_padded, norm="ortho")
    np.set_printoptions(suppress=True)
    print("Block idct:")
    print(block_idct)

def triangle_matrix(a):
    m, n = a.shape
    crop_positions = np.arange(-m+1, m)

    for k in crop_positions:
        a_out = np.flip(np.triu(np.flip(a, axis=1), k=k), axis=1)
        print(a_out)

def rectangle_matrix(a):
    m, n = a.shape
    crop_positions = np.arange(1, m+1)

    for k in crop_positions:
        a_out = np.zeros(a.shape)
        a_out[:k, :k] = a[:k, :k]
        print(a_out)




def test_single_block():
    acceptable_error = 0.05
    block = np.array([[-5.20027311, -5.24577522, -5.26438249, -5.28278582, -5.28943763,
            -5.28620667, -5.22991674, -5.33      ],
           [-5.22109916, -5.24805949, -5.26489261, -5.28189572, -5.28784972,
            -5.28336978, -5.28205847, -5.32737509],
           [-5.22052606, -5.24764238, -5.26266829, -5.27922573, -5.28511089,
            -5.28144084, -5.29060727, -5.32440359],
           [-5.21491611, -5.24384541, -5.25877567, -5.27448409, -5.28015134,
            -5.26120412, -5.30130115, -5.32732734],
           [-5.20352587, -5.23489846, -5.25087798, -5.26483021, -5.26682295,
            -5.23236983, -5.28127881, -5.32623842],
           [-5.19568956, -5.2264212 , -5.24397157, -5.25615359, -5.2282465 ,
            -5.25661258, -5.30286088, -5.33032184],
           [-5.18668727, -5.21677869, -5.23435284, -5.24922688, -5.25531286,
            -5.25587859, -5.28376752, -5.31631953],
           [-5.1785295 , -5.20611774, -5.2237025 , -5.23685771, -5.2353568 ,
            -5.25431689, -5.28182451, -5.31847462]])

    block_dct_zigzag = find_shortest_components(block, acceptable_error)
    print(block_dct_zigzag)

    _, positions = zigzag(np.zeros(block.shape))
    block_dct_components = reverse_zigzag(block_dct_zigzag, positions)
    block_idct = idctn(block_dct_components, norm="ortho")
    print(block_idct)
    print(np.max(np.abs(block - block_idct)))



if __name__ == "__main__":
    # main()
    # test()
    # test_dct_idct()
    # test_dct_block_reduction()
    # test_dct_block_reduction_triangle()

#     a = np.arange(1, 37).reshape(6, 6)
    # triangle_matrix(a)
    # rectangle_matrix(a)
    test_single_block()
