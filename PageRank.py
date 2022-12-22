import numpy as np
import Potenzmethode
from scipy.linalg import null_space
import time


def e(index, dim=12):
    # method to create standard base vectors
    arr = np.zeros(dim)
    arr[index - 1] = 1
    return arr


def row_sum(matrix, row):
    result = 0
    for j in range(matrix.shape[1]):
        result += matrix[row, j]
    return result


def tilde(matrix):
    result = np.ones((12, 12))
    for row in range(12):
        for j in range(12):
            if row_sum(matrix, row) != 0:
                result[row, j] = matrix[row, j]
    return result


# Aufgabenteil a
L = np.array([e(2) + e(3), e(5), e(1) + e(2), e(5), e(4) + e(7) + e(9),
              e(5), np.zeros(12), e(5), e(6), e(8), e(8), e(8)])


def diagonal_inverse(matrix):
    result = np.zeros((12, 12))
    for i in range(matrix.shape[1]):
        result[i, i] = 1 / row_sum(matrix, i)
    return result


def random_surfer(alpha, dim=12):
    return (1 - alpha) * np.dot(tilde(L).transpose(), diagonal_inverse(tilde(L))) + alpha / dim * np.ones((dim, dim))


def eigenvector_for_one(matrix):
    start = time.time()
    result = null_space(random_surfer(i) - np.eye(12))
    end = time.time()
    print("Laufzeit Nullspace-Methode:", end - start)
    return result

if __name__ == "__main__":
    print(L)
    epsilon = 0.00000001
    v0 = 1 / 12 * np.ones(12)
    for i in [0.1, 0.3, 0.6]:
        print("Potenzmethode für " + str(i) + ":" + str(Potenzmethode.vector_iteration(random_surfer(i), v0, epsilon)))
        print("Nullspacemethode für " + str(i) + ":" + str(eigenvector_for_one(random_surfer(i))))
