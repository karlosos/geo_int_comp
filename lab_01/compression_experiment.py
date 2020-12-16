import numpy as np
from compression import compression, decompression
from interpolation import load_binary
from tqdm import tqdm


def experiment(file_path, block_size, decompression_acc):
    output_path = file_path + "_compressed.pckl"

    # Kompresja
    Z = load_binary(file_path)
    x = np.arange(Z.shape[1])
    y = np.arange(Z.shape[0])
    X, Y = np.meshgrid(x, y)

    (
        dct_components,
        block_size,
        padded_shape,
        orig_shape,
        x_start,
        y_start,
        grid_step,
        t,
    ) = compression(X, Y, Z, block_size, decompression_acc, output_path)

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

    # Różnica
    mean_error = np.nanmean(np.abs(Z - Z_out))

    # Rozmiary plików
    import os

    input_size = os.stat(file_path).st_size
    output_size = os.stat(output_path).st_size

    import zipfile

    zip_path = file_path + ".zip"
    my_zipfile = zipfile.ZipFile(zip_path, mode="w", compression=zipfile.ZIP_DEFLATED)

    my_zipfile.write(output_path)
    my_zipfile.close()

    zip_size = os.stat(zip_path).st_size

    compression_rate = input_size / output_size
    compression_rate_zip = input_size / zip_size

    return t, compression_rate, compression_rate_zip, mean_error


if __name__ == "__main__":
    # file_name = "UTM-brama_0.5_idw.pckl"
    file_name = "UTM-obrotnica_0.5_idw.pckl"
    # file_name = "UTM-brama_0.5_idw.pckl"
    file_path = f"./data/output/{file_name}"
    data = {"block_size": [], "time": [], "cr": [], "cr_zip": [], "mean_error": []}
    block_sizes = [*range(2, 32, 2), *range(40, 200, 20)]
    for block_size in tqdm(block_sizes):
        t, compression_rate, compression_rate_zip, mean_error = experiment(file_path, block_size, 0.05)
        data["block_size"].append(block_size)
        data["time"].append(t)
        data["cr"].append(compression_rate)
        data["cr_zip"].append(compression_rate_zip)
        data["mean_error"].append(mean_error)

    import pandas as pd

    df = pd.DataFrame.from_dict(data)
    df.to_csv("exp_block_sizes_{file_name}.csv")
    print(df)
