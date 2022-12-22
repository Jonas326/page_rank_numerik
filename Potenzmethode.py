import numpy as np
import time


def vector_iteration(matrix, vector, epsilon):
    start = time.time()
    vector = vector * (1 / np.linalg.norm(vector))
    r = rayleigh(matrix, vector)
    while np.linalg.norm(np.dot(matrix, vector) - r * vector) >= epsilon:
        vector = np.dot(matrix, vector)
        vector = vector * (1 / np.linalg.norm(vector))
        r = rayleigh(matrix, vector)
    end = time.time()
    print("Laufzeit Potenz-Methode:", end - start)
    return vector


def rayleigh(matrix, vector):
    dividend = float(np.dot(vector.transpose(), np.dot(matrix, vector)))
    divisor = float(np.dot(vector.transpose(), vector))
    return dividend/divisor
