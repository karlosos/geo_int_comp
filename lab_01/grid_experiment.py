import pandas as pd
import numpy as np

from interpolation import plot_surface

def main():
    data = np.array(
        [[2.9, 5.5, 1], [3, 6.5, 2], [4.5, 5, 0], [4.5, 6.5, 1], [4.5, 8, 2], [6, 6, 2], [6, 7.5, 3]]
    )
    x_min = np.min(data[:, 0])
    x_max = np.max(data[:, 0])
    y_min = np.min(data[:, 1])
    y_max = np.max(data[:, 1])
    print(x_min, x_max, y_min, y_max)

    x = np.linspace(x_min, x_max, int(np.round((x_max - x_min)/0.5)) + 1)
    y = np.linspace(y_min, y_max, int(np.round((y_max - y_min)/0.5)) + 1)
    print(x)
    print(y)

    z_values = np.full((len(x), len(y)), None, dtype='object')

    for i in range(data.shape[0]):
        print(data[i, :])

    print(z_values)
    # plot_surface(x, y, z_values)


if __name__ == "__main__":
    main()
