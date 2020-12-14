from scipy.fftpack import dct, idct
from scipy.fftpack import dctn, idctn

import numpy as np

from compression import to_blocks, find_shortest_components
from zigzag import zigzag, reverse_zigzag


def test_block_division():
    """
    This tests block fivision with calculating dct

    WARNING: this test is out of date, because of other dct method
    """
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
    """
    This function tests if dct <--> idct reconstruction works
    """
    block_size = 5
    block = np.arange(block_size ** 2).reshape(block_size, block_size)
    print("Block input:")
    print(block)

    block_dct = dctn(block, norm="ortho")
    print("Block dct:")
    print(block_dct)

    block_idct = idctn(block_dct, norm="ortho")
    np.set_printoptions(suppress=True)
    print("Block idct:")
    print(block_idct)


def test_dct_block_reduction():
    """
    This function tests how rectangle reduction works 

    Values are close to input
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
    """
    This function tests how triangle reduction works 

    Values are close to input
    """
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
    block_dct_reduction_padded = np.flip(
        np.triu(np.flip(block_dct, axis=1), k=-(block_size)), axis=1
    )
    print(block_dct_reduction_padded)

    block_idct = idctn(block_dct_reduction_padded, norm="ortho")
    np.set_printoptions(suppress=True)
    print("Block idct:")
    print(block_idct)


def triangle_matrix(a):
    """
    Crop upper left triangle from matrix a

    Currently unused in project.
    """
    m, n = a.shape
    crop_positions = np.arange(-m + 1, m)

    for k in crop_positions:
        a_out = np.flip(np.triu(np.flip(a, axis=1), k=k), axis=1)
        print(a_out)


def rectangle_matrix(a):
    """
    Crop upper left rectangle from matrix

    Currently unused in project
    """
    m, n = a.shape
    crop_positions = np.arange(1, m + 1)

    for k in crop_positions:
        a_out = np.zeros(a.shape)
        a_out[:k, :k] = a[:k, :k]
        print(a_out)


def test_shortest_components_single_block():
    """
    This function tests finding shortest components for given error
    for single block.

    Block values are real data.
    """
    acceptable_error = 0.05
    block = np.array(
        [
            [
                -5.20027311,
                -5.24577522,
                -5.26438249,
                -5.28278582,
                -5.28943763,
                -5.28620667,
                -5.22991674,
                -5.33,
            ],
            [
                -5.22109916,
                -5.24805949,
                -5.26489261,
                -5.28189572,
                -5.28784972,
                -5.28336978,
                -5.28205847,
                -5.32737509,
            ],
            [
                -5.22052606,
                -5.24764238,
                -5.26266829,
                -5.27922573,
                -5.28511089,
                -5.28144084,
                -5.29060727,
                -5.32440359,
            ],
            [
                -5.21491611,
                -5.24384541,
                -5.25877567,
                -5.27448409,
                -5.28015134,
                -5.26120412,
                -5.30130115,
                -5.32732734,
            ],
            [
                -5.20352587,
                -5.23489846,
                -5.25087798,
                -5.26483021,
                -5.26682295,
                -5.23236983,
                -5.28127881,
                -5.32623842,
            ],
            [
                -5.19568956,
                -5.2264212,
                -5.24397157,
                -5.25615359,
                -5.2282465,
                -5.25661258,
                -5.30286088,
                -5.33032184,
            ],
            [
                -5.18668727,
                -5.21677869,
                -5.23435284,
                -5.24922688,
                -5.25531286,
                -5.25587859,
                -5.28376752,
                -5.31631953,
            ],
            [
                -5.1785295,
                -5.20611774,
                -5.2237025,
                -5.23685771,
                -5.2353568,
                -5.25431689,
                -5.28182451,
                -5.31847462,
            ],
        ]
    )

    block_dct_zigzag = find_shortest_components(block, acceptable_error)
    print(block_dct_zigzag)

    _, positions = zigzag(np.zeros(block.shape))
    block_dct_components = reverse_zigzag(block_dct_zigzag, positions)
    block_idct = idctn(block_dct_components, norm="ortho")
    print(block_idct)
    print(np.max(np.abs(block - block_idct)))
