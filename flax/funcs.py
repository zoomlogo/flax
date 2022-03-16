# funcs: holds the functions used by atoms
import functools
import itertools
import math
import more_itertools
import operator
import random

from flax.common import mp, mpc, mpf


def depth(x):
    # depth: returns the depth of x
    return 0 if type(x) != list else (1 if not x else max([depth(a) for a in x]) + 1)


def diagonals(x, anti=False):
    # diagonals: returns the diagonals or the anti-diagonals of x
    diag = [[] for _ in range(len(x) + len(x[0]) - 1)]
    anti = [[] for _ in range(len(diag))]
    min_d = -len(x) + 1

    for i in range(len(x[0])):
        for j in range(len(x)):
            diag[i + j].append(x[j][i])
            anti[i - j - min_d].append(x[j][i])


def divisors(x):
    # divisors: returns the factors of x
    return [a for a in range(1, int(x) + 1) if x % a == 0]


@functools.cache
def fibonacci(x):
    # fibonacci: return the x'th fibonacci number
    if x < 2:
        return x
    else:
        return fibonacci(x - 1) + fibonacci(x - 2)


def find(w, x):
    # find: find the occurence of x in w
    try:
        return iterable(w, digits=True).index(x)
    except ValueError:
        return []


def find_all(w, x):
    # find_all: returns all indicies of occurences of x in w
    return [i for i, e in enumerate(w) if e == x]


def find_sublist(w, x):
    # find_sublist: find the occurence of the sublist x in w
    w = iterable(w, digits=True)
    x = iterable(x, digits=True)
    for i in range(len(w)):
        if w[i : i + len(x)] == x:
            return i
    return []


def flatten(x):
    # flatten: flatten x
    return list(more_itertools.collapse(x))


def from_bin(x):
    # from_bin: convert x from binary
    x = iterable(x, digits=True)
    sign = -1 if sum(x) < 0 else 1
    num = 0
    i = 0
    for b in x[::-1]:
        num += abs(b) * 2 ** i
        i += 1
    return num * sign


def from_digits(x):
    # from_digits: convert x from digits:
    x = iterable(x, range_=True)
    sign = -1 if sum(x) < 0 else 1
    num = 0
    i = 0
    for b in x[::-1]:
        num += abs(b) * 10 ** i
        i += 1
    return num * sign


def iterable(x, digits=False, range_=False):
    # iterable: make sure x is a list
    if type(x) != list:
        if range_:
            return list(range(int(x)))
        elif digits:
            return to_digits(x)
        else:
            return [x]
    else:
        return x


def grade_down(x):
    # grade_down: grade x in descending order
    x = iterable(x, digits=True)
    grades = []
    for a in reversed(sorted(x)):
        grades.append(find_all(x, a))
    return flatten(grades)


def grade_up(x):
    # grade_up: grade x in ascending order
    x = iterable(x, digits=True)
    grades = []
    for a in sorted(x):
        grades.append(find_all(x, a))
    return flatten(grades)


def group_indicies(x):
    # group_indicies: groups indicies with equal values
    res = {}
    for i, e in enumerate(x):
        e = str(e)
        if e in res:
            res[e].append(i)
        else:
            res[e] = [i]
    return [res[k] for k in sorted(res, key=eval)]


def group_equal(x):
    # group_equal: group equal adjacent elements
    res = []
    for e in x:
        if res and res[-1][0] == e:
            res[-1].append(e)
        else:
            res.append([e])
    return res


def to_digits(x):
    # to_digits: turn x into a list of digits
    return [
        -int(i) if x < 0 else int(i) for i in str(x)[1 if x < 0 else 0 :] if i != "."
    ]
