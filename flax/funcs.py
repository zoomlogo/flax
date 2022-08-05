"""funcs: holds the functions used by atoms"""
import functools
import itertools
import more_itertools

from flax.common import mp, mpc, inf, mpf


def boolify(f):
    """boolify: wrapper around boolean functions to only return 1/0"""
    return lambda *args: int(f(*args))


def depth(x):
    """depth: how deeply x is nested"""
    if type(x) != list:
        return 0
    else:
        if not x:
            return 1
        else:
            return max([depth(i) for i in x]) + 1


def diagonals(x, antidiagonals=False):
    """diagonals: returns all diagonals or antidiagonals of x"""
    diag = [[] for _ in range(len(x) + len(x[0]) - 1)]
    anti = [[] for _ in range(len(diag))]
    min_d = -len(x) + 1

    for i in range(len(x[0])):
        for j in range(len(x)):
            diag[i + j].append(x[j][i])
            anti[i - j - min_d].append(x[j][i])

    return anti if antidiagonals else diag
