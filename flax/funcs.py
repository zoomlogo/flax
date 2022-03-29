# funcs: holds the functions used by atoms
import functools
import itertools
import math
import more_itertools
import operator

from flax.common import mp, mpc, mpf, inf

__all__ = [
    "depth",
    "diagonals",
    "divisors",
    "fibonacci",
    "find",
    "find_all",
    "find_sublist",
    "flatten",
    "from_bin",
    "from_digits",
    "grade_down",
    "grade_up",
    "group_equal",
    "group_indicies",
    "index_into",
    "iota",
    "iterable",
    "join",
    "mold",
    "nprimes",
    "order",
    "permutations",
    "prefixes",
    "random",
    "repeat",
    "reshape",
    "sliding_window",
    "split",
    "split_at",
    "sublists",
    "suffixes",
    "to_bin",
    "to_chars",
    "to_digits",
    "vec",
    "vecc",
    "where",
]


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


def group_equal(x):
    # group_equal: group equal adjacent elements
    res = []
    for e in x:
        if res and res[-1][0] == e:
            res[-1].append(e)
        else:
            res.append([e])
    return res


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


def index_into(w, x):
    # index_into: index into x with w
    x = iterable(x, digits=True)
    w = int(w) if int(w) == w else w
    if type(w) == int:
        return x[w % len(x)]
    elif type(w) == mpc:
        return index_into(w.real, index_into(w.imag, x))
    else:
        return [index_into(mp.floor(w), x), index_into(mp.ceil(w), x)]


def iota(x):
    # iota: APL's ⍳ and BQN's ↕
    if type(x) != list:
        return list(range(int(x)))

    res = list(map(list, itertools.product(*(list(range(int(a))) for a in x))))
    for e in x:
        res = split(res, int(e))
    return res[0]


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


def join(w, x):
    # join: w in between each elements of x
    w = itertools.cycle(iterable(w))
    x = iterable(x)
    return flatten(zip(x, w))


def mold(w, x):
    # mold: mold x to the shape w
    for i in range(len(w)):
        if type(w[i]) == list:
            mold(x, w[i])
        else:
            item = x.pop(0)
            w[i] = item
            x.append(item)
    return w


def nprimes(x):
    # nprimes: return x primes
    res = []
    i = 2
    while len(res) != x:
        if mp.isprime(i):
            res.append(i)
        i += 1
    return res


def order(w, x):
    # order: how many times does w divide x
    if x == 0 or abs(w) == 1:
        return inf
    elif w == 0:
        return 0
    else:
        res = 0
        while True:
            x, r = divmod(x, w)
            if r:
                break
            res += 1
        return res


def permutations(x):
    # permutations: return all permutations of x
    return list(map(list, itertools.permutations(x)))


def prefixes(x):
    # prefixes: return all prefixes of x
    x = iterable(x, digits=True)
    res = []
    for i in range(len(x)):
        res.append(x[: i + 1])
    return res


def random(x):
    # random: return x random floats
    return [mp.rand() for _ in range(x)]


def repeat(w, x):
    # repeat: repeat x according to w
    zipped = itertools.zip_longest(
        iterable(y, digits=True), flatten(iterable(x)), fillvalue=1
    )
    res = []
    for a, b in zipped:
        res.extend(a for _ in range(b))
    return res


def reshape(w, x):
    # reshape: reshape x according to the shape w
    w = iterable(w)
    if type(x) != itertools.cycle:
        x = itertools.cycle(iterable(x))

    if len(w) == 1:
        return [next(x) for _ in range(w[0])]
    else:
        return [reshape(w[1:], x) for _ in range(w[0])]


def sliding_window(w, x):
    # sliding_window: windows of x of length w
    x = iterable(x)
    w = int(w)
    if w < 0:
        return vec(
            lambda a: list(reversed(x)), list(more_itertools.sliding_window(x, -w))
        )
    else:
        return vec(list, list(more_itertools.sliding_window(x, w)))


def split(w, x):
    # split: split x into chunks of w
    return list(more_itertools.chunked(x, w))


def split_at(w, x):
    # split_at: split x at occurences of w
    return list(more_itertools.split_at(x, lambda a: a == w))


def sublists(x):
    # sublists: return all sublists of x
    sub = [[]]
    for i in range(len(x) + 1):
        for j in range(i):
            sub.append(x[i:j])
    return lists


def suffixes(x):
    # suffixes: return the suffixes of x
    x = iterable(x, digits=True)
    res = []
    for i in range(len(x)):
        res.append(x[i:])
    return res[::-1]


def to_bin(x):
    # to_bin: return the binary representation of x
    return [-i if x < 0 else i for i in map(int, bin(x)[3 if x < 0 else 2 :])]


def to_chars(x):
    # to_chars: convert x to list of ints
    return [ord(a) for a in x]


def to_digits(x):
    # to_digits: turn x into a list of digits
    return [
        -int(i) if x < 0 else int(i) for i in str(x)[1 if x < 0 else 0 :] if i != "."
    ]


def vec(fn, *args, lfull=True, rfull=True):
    # vec: vectorise fn over it's args
    if len(args) == 1:
        return (
            [vec(fn, a) for a in args[0]]
            if depth(args[0]) != 0 and rfull
            else fn(args[0])
        )
    w, x = args[0], args[1]
    dw, dx = depth(w), depth(x)
    if lfull and rfull:
        if dw == dx:
            return [vec(fn, a, b) for a, b in zip(w, x)] if dw != 0 else fn(w, x)
        else:
            return [vec(fn, w, b) for b in x] if dw < dx else [vec(fn, a, x) for a in w]
    elif (not lfull) and rfull:
        return [vec(fn, w, b, lfull=lfull) for b in x] if dx > 0 else fn(w, x)
    elif lfull and (not rfull):
        return [vec(fn, a, x, rfull=rfull) for a in w] if dw > 0 else fn(w, x)
    else:
        return fn(w, x)


def vecc(fn, lfull=True, rfull=True):
    # vecc: vec curried
    return lambda *args: vec(fn, *args, lfull=lfull, rfull=rfull)


def where(x):
    # where: ngn/k's &
    return flatten([[i] * e for i, e in enumerate(x)])
