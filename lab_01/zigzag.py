import numpy as np


def zigzag(a):
    """
    Create zigzag vector from matrix

    :param a: input matrix
    :returns: vector with length m*n with matrix values under zigzag indexing
    """
    m, n = a.shape
    solution = [[] for i in range(m + n - 1)]

    for i in range(m):
        for j in range(n):
            sum = i + j
            value = a[i][j]
            index = (i, j)
            if sum % 2 == 0:
                solution[sum].insert(0, (value, index))
            else:
                solution[sum].append((value, index))

    output_vector = []
    positions = []
    for i in solution:
        for j in i:
            output_vector.append(j[0])
            positions.append(j[1])

    return output_vector, positions


def reverse_zigzag(v, positions, length):
    """
    Create array from zigzag vector

    :param v: zigzag vector
    :param positions: positions of elements
    :param length: length of vector to deconstruction. Can be less than len(v)
    :returns:
    """
    m = int(np.sqrt(len(v)))
    a = np.zeros((m, m))

    v = v[:length]

    for i in range(len(v)):
        j, k = positions[i]
        a[j, k] = v[i]

    return a


if __name__ == "__main__":
    a = np.arange(1, 37).reshape(6, 6)
    print(a)
    v, positions = zigzag(a)
    for i in range(len(v), 0, -1):
        a_out = reverse_zigzag(v, positions, length=i)
        print(a_out)
