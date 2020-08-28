import os
import pretty_midi
import numpy as np


def generate_sparse_matrix_of_genre_colab(genre, phase):
    npy_path = '../dataset/' + genre + f'/{phase}.npz'
    with np.load(npy_path) as f:
        shape = f['shape']
        data = np.zeros(shape, np.float_)
        nonzeros = f['nonzeros']
        for x in nonzeros:
            data[(int(x[0]), int(x[1]), int(x[2]))] = 1.

    np.random.shuffle(data)
    return data
