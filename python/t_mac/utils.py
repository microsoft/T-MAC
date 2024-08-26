import numpy as np

from .platform import *  # for backward compatibility


def get_bits_alphas(bits: int):
    alphas = [1 / 2, 1, 2, 4]
    return alphas[:bits]


def nmse(a: np.ndarray, b: np.ndarray):
    a, b = a.astype(np.float32), b.astype(np.float32)
    return np.mean(np.square(a - b)) / np.mean(np.square(a))
