import argparse
import time
import numpy as np
from scipy.fftpack import dctn, idctn
from tqdm import tqdm
from rich.console import Console
import pickle

from zigzag import zigzag, reverse_zigzag
from interpolation import plot


def load_data(file_path):
    console = Console()

    with console.status("[bold green]Wczytywanie danych...") as status:
        # Wczytanie danych
        t1 = time.time()
        data = np.loadtxt(file_path, skiprows=1)

        # Tworzenie grida
        x = np.unique(data[:, 0])
        y = np.unique(data[:, 1])
        z = data[:, 2]

        X, Y = np.meshgrid(x, y)
        Z = z.reshape(X.shape)
    t = time.time() - t1
    console.print(f"Wczytano dane w [bold green]{np.round(t)}[/bold green] s.")
    return X, Y, Z


def to_blocks(img, window_size):
    """
    Slice image into blocks of size window_size and flatten them.
    """

    height, width = img.shape

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

    return blocks, img


def from_blocks(blocks, block_size, image_shape):
    image = np.zeros(image_shape)
    height, width = image.shape
    ws = block_size
    index = 0
    for i in range(height // ws):
        for j in range(width // ws):
            image[i * ws : i * ws + ws, j * ws : j * ws + ws] = blocks[index]
            index += 1
    return image


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
        decompression_acc = 0.05
    else:
        decompression_acc = args.decompression_acc

    return file_path, block_size, decompression_acc


def decompression_blocks(dct_components, block_size, shape):
    block_shape = (block_size, block_size)
    _, positions = zigzag(np.zeros(block_shape))
    blocks = []
    empty_block = np.full(block_shape, np.nan)
    for component in tqdm(dct_components, desc="Dekompresja"):
        if component is not None:
            block_dct_components = reverse_zigzag(component, positions)
            block_idct = idctn(block_dct_components, norm="ortho")
            blocks.append(block_idct)
        else:
            blocks.append(empty_block)

    out = from_blocks(blocks, block_size, shape)
    return out


def plot_diff(xx, yy, orig_z, compress_z):
    import matplotlib.pyplot as plt

    fig = plt.figure()

    ax2 = fig.add_subplot(121)
    depth_map = ax2.imshow(compress_z, interpolation="none")
    plt.colorbar(depth_map)

    ax2 = fig.add_subplot(122)
    diff = np.abs(orig_z - compress_z)
    depth_map = ax2.imshow(diff, interpolation="none")
    plt.colorbar(depth_map)

    plt.show()


def compression(X, Y, Z, block_size, decompression_acc):
    t1 = time.time()
    # Obliczenie danych
    x_start = X[0, 0]
    y_start = Y[0, 0]
    grid_step = X[0, 1] - X[0, 0]

    # Podzial na bloki
    blocks, image_padding = to_blocks(Z, block_size)

    # DCT w blokach
    dct_components = []
    for block in tqdm(blocks, desc="Kompresja"):
        has_none = np.any(np.isnan(block))
        if not has_none:
            components = find_shortest_components(
                block, acceptable_error=decompression_acc
            )
            dct_components.append(components)
        else:
            dct_components.append(None)

    t = time.time() - t1
    console = Console()
    console.print(f"Skompresowano dane w [bold green]{np.round(t, 3)}[/bold green] s.")

    # Saving to file
    filename = "compressed.pckl"
    outfile = open(filename, "wb")
    data = (
        dct_components,
        block_size,
        image_padding.shape,
        Z.shape,
        x_start,
        y_start,
        grid_step,
    )
    pickle.dump(data, outfile)
    outfile.close()

    return (
        dct_components,
        block_size,
        image_padding.shape,
        Z.shape,
        x_start,
        y_start,
        grid_step,
    )


def decompression(
    dct_components, block_size, padded_shape, orig_shape, x_start, y_start, grid_step
):
    t1 = time.time()
    decompressed = decompression_blocks(dct_components, block_size, padded_shape)
    t = time.time() - t1
    console = Console()
    console.print(f"Odtworzono dane w [bold green]{np.round(t, 3)}[/bold green] s.")

    # Plotting
    x_end = x_start + orig_shape[1] * (grid_step)
    y_end = y_start + orig_shape[0] * (grid_step)
    xx = np.linspace(x_start, x_end, orig_shape[1])
    yy = np.linspace(y_start, y_end, orig_shape[0])

    X, Y = np.meshgrid(xx, yy)
    image_out = decompressed[: orig_shape[0], : orig_shape[1]]
    # plot(X, Y, image_out)
    return image_out


def print_options(file_path, block_size, decompression_acc):
    console = Console()
    console.print(f"Plik: [bold cyan]{file_path}[/bold cyan]")
    console.print(f"Rozmiar bloku: [bold cyan]{block_size}[/bold cyan]")
    console.print(f"Zadana dokładność: [bold cyan]{decompression_acc}[/bold cyan]")


def test_compression_decompression():
    file_path = "./data/output/UTM-obrotnica_idw.txt"
    block_size = 32
    decompression_acc = 0.05

    print_options(file_path, block_size, decompression_acc)

    # Kompresja 
    X, Y, Z = load_data(file_path)

    (
        dct_components,
        block_size,
        padded_shape,
        orig_shape,
        x_start,
        y_start,
        grid_step,
    ) = compression(X, Y, Z, block_size, decompression_acc)

    # Dekompresja 
    Z_out = decompression(
        dct_components,
        block_size,
        padded_shape,
        orig_shape,
        x_start,
        y_start,
        grid_step,
    )

    # Porównanie różnicy
    plot_diff(X, Y, Z, Z_out)


def main():
    # Argumenty linii komend
    file_path, block_size, decompression_acc = command_line_arguments()

    # Dzialanie
    X, Y, Z = load_data(file_path)

    # Podzial na bloki
    print("Compression...")
    blocks, image_padding = to_blocks(Z, block_size)

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
            dct_components.append(None)

    # Decompression
    print("Decompression...")
    decompressed = decompression_blocks(dct_components, block_size, image_padding.shape)

    error = np.nanmax(np.abs(decompressed - image_padding))
    print("Err:", error)

    # Plotting
    # TODO: what about x, y, z?
    # Need to reconsider this :O
    image_out = decompressed[: Z.shape[0], : Z.shape[1]]
    plot(X, Y, image_out)

    # TODO: saving to file


if __name__ == "__main__":
    test_compression_decompression()
