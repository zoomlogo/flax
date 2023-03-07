"""funcs: holds the functions used by atoms"""
import functools
import itertools
import more_itertools
import copy
import re
from random import randrange
import operator as ops

from flax.common import mp, mpc, inf, mpf

__all__ = [
    "base",
    "base_i",
    "binary",
    "binary_i",
    "boolify",
    "cartesian_product",
    "convolve",
    "depth",
    "diagonal_leading",
    "diagonal_trailing",
    "diagonals",
    "digits",
    "digits_i",
    "enumerate_md",
    "ensure_square",
    "fibonacci",
    "find",
    "find_md",
    "find_all",
    "find_sublist",
    "flatten",
    "get_req",
    "grade_down",
    "grade_up",
    "group_equal",
    "group_indicies",
    "index_into",
    "index_into_md",
    "iota",
    "iota1",
    "iterable",
    "join",
    "json_decode",
    "lucas",
    "mapval",
    "maximal_indicies",
    "maximal_indicies_md",
    "mold",
    "multiset_difference",
    "multiset_intersection",
    "multiset_union",
    "nprimes",
    "ones",
    "order",
    "permutations",
    "prefixes",
    "prime_factors",
    "random",
    "repeat",
    "reshape",
    "shuffle",
    "sliding_window",
    "split",
    "split_at",
    "split_into",
    "sublists",
    "suffixes",
    "to_braille",
    "transpose",
    "unrepeat",
    "where",
]


def base(w, x):
    """base: convert x from base w"""
    x = iterable(x, digits_=True)
    sign = -1 if sum(x) < 0 else 1
    num = 0
    i = 0
    for d in x[::-1]:
        num += abs(d) * w**i
        i += 1
    return num * sign


def base_i(w, x):
    """base_i: convert x into base w"""
    if x == 0:
        return [0]
    res = []
    sign = 1
    if x < 0:
        x = abs(x)
        sign = -1
    while x:
        res.append(x % w)
        x = x // w
    return [r * sign for r in res][::-1]


def binary(x):
    """binary: converts x to binary"""
    return [-i if x < 0 else i for i in map(int, bin(x)[3 if x < 0 else 2 :])]


def binary_i(x):
    """binary_i: convert x from binary"""
    x = iterable(x, digits_=True)
    sign = -1 if sum(x) < 0 else 1
    num = 0
    i = 0
    for b in x[::-1]:
        num += abs(b) * 2**i
        i += 1
    return num * sign


def boolify(f):
    """boolify: wrapper around boolean functions to only return 1/0"""
    return lambda *args: int(f(*args))


def cartesian_product(*args):
    """cartesian_product: find the cartesian product"""
    return list(map(list, itertools.product(*args)))


def convolve(w, x):
    """convolve: find the convolution of w and x"""
    conv = []
    i = 0
    j = 1
    for _ in range(len(w) + len(x) - 1):
        conv.append(sum(map(ops.mul, w[i:j], x[i:j][::-1])))
        if j >= min(len(w), len(x)):
            i += 1
        j += 1
    return conv


def depth(x):
    """depth: how deeply x is nested"""
    if type(x) == str:
        return 0 if len(str) < 2 else 1
    elif type(x) == list:
        if not x:
            return 1
        else:
            return max([depth(i) for i in x]) + 1
    else:
        return 0


def diagonal_leading(x):
    """diagonal_leading: the leading diagonal of x"""
    x = iterable(x, range_=True)
    return [iterable(x[i], range_=True)[i] for i in range(len(x))]


def diagonal_trailing(x):
    """diagonal_trailing: the trailing diagonal of x"""
    x = iterable(x, range_=True)
    return [iterable(x[i], range_=True)[len(x[i]) - 1 - i] for i in range(len(x))]


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


def digits(x):
    """digits: turn x into a list of digits"""
    return [
        -int(i) if x < 0 else int(i) for i in str(x)[1 if x < 0 else 0 :] if i != "."
    ]


def digits_i(x):
    """digits_i: convert x from digits"""
    x = iterable(x, range_=True)
    sign = -1 if sum(x) < 0 else 1
    num = 0
    i = 0
    for b in x[::-1]:
        num += abs(b) * 10**i
        i += 1
    return num * sign


def enumerate_md(x, upper_level=[]):
    """enumerate_md: enumerate multidimensionally"""
    for i, e in enumerate(x):
        if type(e) != list:
            yield [upper_level + [i], e]
        else:
            yield from enumerate_md(e, upper_level + [i])


def ensure_square(x):
    """ensure_square: make sure x is a square matrix"""
    x = iterable(x)
    l = max([len(iterable(i)) for i in x] + [len(x)])
    return reshape(l, [reshape(l, i) for i in x])


@functools.cache
def fibonacci(x):
    """fibonacci: return the x'th fibonacci number"""
    if x < 2:
        return x
    else:
        return fibonacci(x - 1) + fibonacci(x - 2)


def find(w, x):
    """find: find the occurence of w in x"""
    try:
        return iterable(x, digits_=True).index(w)
    except ValueError:
        return []


def find_all(w, x):
    """find_all: returns all indicies of occurences of w in x"""
    return [i for i, e in enumerate(x) if e == w]


def find_md(w, x):
    """find_md: find multidimensionally"""
    for i, e in enumerate_md(x):
        if e == w:
            return i
    return []


def find_sublist(w, x):
    """find_sublist: find the occurence of the sublist w in x"""
    x = iterable(x, digits_=True)
    w = iterable(w, digits_=True)
    for i in range(len(x)):
        if x[i : i + len(w)] == w:
            return i
    return []


def flatten(x):
    """flatten: flatten x"""
    return list(more_itertools.collapse(x))


def get_req(x):
    """get_req: GET request for url x"""
    url = "".join(map(chr, x))
    url = (
        re.match(r"[A-Za-z][A-Za-z0-9+.-]*://", url) is None and "http://" or ""
    ) + url
    response = urllib_request.request.urlopen(url).read()
    try:
        return response.decode("utf-8")
    except:
        return response.decode("latin-1")


def grade_down(x):
    """grade_down: grade x in descending order"""
    x = iterable(x, digits_=True)
    grades = []
    for i in reversed(sorted(x)):
        grades.append(find_all(i, x))
    return flatten(grades)


def grade_up(x):
    """grade_up: grade x in ascending order"""
    x = iterable(x, digits_=True)
    grades = []
    for i in sorted(x):
        grades.append(find_all(i, x))
    return flatten(grades)


def group_equal(x):
    """group_equal: group equal adjacent elements"""
    res = []
    for i in x:
        if res and res[-1][0] == i:
            res[-1].append(i)
        else:
            res.append([i])
    return res


def group_indicies(x, md=False):
    """group_indicies: groups indicies with equal values"""
    res = {}
    enum = enumerate_md(x) if md else enumerate(x)
    for i, e in enum:
        e = str(e)
        if e in res:
            res[e].append(i)
        else:
            res[e] = [i]
    return [res[k] for k in sorted(res, key=eval)]


def index_into(w, x):
    """index_into: index into w with x"""
    w = iterable(w, digits_=True)
    x = int(x) if type(x) != mpc and type(x) != list and int(x) == x else x
    if type(x) == int:
        return w[x % len(w)]
    elif type(x) == mpc:
        return index_into(index_into(w, x.real), x.imag)
    else:
        return [index_into(w, mp.floor(x)), index_into(w, mp.ceil(x))]


def index_into_md(w, x):
    """index_into_md: index into w multidimensionally with x"""
    res = w
    for i in x:
        res = index_into(i, res)
    return res


def iota(x):
    """iota: APL's ⍳ and BQN's ↕"""
    if type(x) != list:
        return list(range(int(x)))

    res = cartesian_product(*(list(range(int(a))) for a in x))
    for e in x:
        res = split(int(e), res)
    return res[0]


def iota1(x):
    """iota1: iota but 1 based"""
    if type(x) != list:
        return [i + 1 for i in range(int(x))]

    res = cartesian_product(*([i + 1 for i in range(int(a))] for a in x))
    for e in x:
        res = split(int(e), res)
    return res[0]


def iterable(x, digits_=False, range_=False, copy_=False):
    """iterable: make sure x is a list"""
    if type(x) != list:
        if type(x) == str:
            return list(x)
        else:
            if range_:
                return list(range(int(x)))
            elif digits_:
                return digits(x)
            else:
                return [x]
    else:
        return copy.deepcopy(x) if copy_ else x


def join(w, x):
    """join: w in between each elements of x"""
    w = itertools.cycle(iterable(w))
    x = iterable(x)
    return flatten(zip(x, w))


def json_decode(x):
    """json_decode: convert jsoned x to flax arrays"""
    if type(x) == list or type(x) == tuple:
        return [json_decode(i) for i in x]
    elif type(x) == str:
        return x
    elif type(x) == dict:
        return [json_decode(i) for i in x.items()]
    elif type(x) == bool:
        return int(x)
    elif x is None:
        return inf
    else:
        return mpf(x)


@functools.cache
def lucas(x):
    """lucas: nth lucas number"""
    if x < 2:
        return x + 1
    else:
        return lucas(x - 1) + lucas(x - 2)


def mapval(w, x):
    """mapval: map values from x specified by w"""
    ins = w[0]
    outs = w[1]

    res = []
    for i in iterable(x, range_=True):
        res.append(outs[ins.index(i)])

    return res


def maximal_indicies(x):
    """maximal_indicies: indicies of elements with the maximal value"""
    return [i for i, e in enumerate(x) if e == max(x)]


def maximal_indicies_md(x, m=None, upper_level=[]):
    """maximal_indicies_md: multidimensional indicies of maximal elements"""
    x = iterable(x, digits_=True)
    if m is None:
        m = max(flatten(x) or [0])
    res = []
    for i, e in enumerate(x):
        if type(e) != list:
            if e == m:
                res.append(upper_level + [i])
            else:
                res.extend(maximal_indicies_md(e, m, upper_level + [i]))
    return res


def mold(w, x):
    """mold: mold x to the shape w"""
    for i in range(len(w)):
        if type(w[i]) == list:
            mold(x, w[i])
        else:
            item = x.pop(0)
            w[i] = item
            x.append(item)
    return w


def multiset_difference(w, x):
    """multiset_difference: multiset difference"""
    res = iterable(w)[::-1]
    for i in iterable(x):
        if i in res:
            res.remove(i)
    return res[::-1]


def multiset_intersection(w, x):
    """multiset_intersection: multiset intersection"""
    x = iterable(x, copy_=True)
    res = []
    for i in iterable(w):
        if i in x:
            res.append(i)
            x.remove(i)
    return res


def multiset_union(w, x):
    """multiset_union: multiset union"""
    return iterable(w) + multiset_difference(x, w)


def nprimes(x):
    """nprimes: return x primes"""
    res = []
    i = 2
    while len(res) != x:
        if mp.isprime(i):
            res.append(i)
        i += 1
    return res


def ones(x, shape=None, upper_level=[]):
    """ones: matrix with ones at x"""
    if not shape:
        shape = [max(zipped) for zipped in zip(*x)]
    upper_len = len(upper_level)
    if upper_len < len(shape) - 1:
        return [
            ones(x, shape=shape, upper_level=upper_level + [i])
            for i in range(shape[upper_len])
        ]
    else:
        return [1 if (upper_level + [i] in x) else 0 for i in range(shape[-1])]


def order(w, x):
    """order: how many times does w divide x"""
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
    """permutations: return all permutations of x"""
    return list(map(list, itertools.permutations(iterable(x))))


def prefixes(x):
    """prefixes: return all prefixes of x"""
    x = iterable(x, digits_=True)
    res = []
    for i in range(len(x)):
        res.append(x[: i + 1])
    return res


def prime_factors(x):
    """prime_factors: calculate prime factors of x"""
    p = primes()
    res = []
    while x != 1:
        prime = next(p)
        times = order(prime, x)
        res.append([prime] * times)
        for _ in range(times):
            x = x / prime
    return flatten(res)


def primes():
    """primes: an infinite list of primes"""
    i = 0
    while True:
        if mp.isprime(i):
            yield i
        i += 1


def random(x):
    """random: return x random floats"""
    return [mp.rand() for _ in range(x)]


def repeat(w, x):
    """repeat: repeat x according to w"""
    zipped = itertools.zip_longest(
        iterable(w, digits_=True), flatten(iterable(x)), fillvalue=1
    )
    res = []
    for a, b in zipped:
        res.extend(a for _ in range(b))
    return res


def reshape(w, x):
    """reshape: reshape x according to the shape w"""
    w = iterable(w)
    x = iterable(x)

    if len(w) == 1:
        reshaped = []
        x = x[::-1] if w[0] < 0 else x
        for _ in range(abs(w[0])):
            reshaped.append(x[0])
            x.append(x.pop(0))
        return reshaped[::-1] if w[0] < 0 else reshaped
    else:
        x = x[::-1] if w[0] < 0 else x
        reshaped = [reshape(w[1:], x) for _ in range(abs(w[0]))]
        return reshaped[::-1] if w[0] < 0 else reshaped


def shuffle(x):
    """shuffle: return a random permutation of x"""
    x = iterable(x, copy_=True, digits_=True)
    for i in range(len(x) - 1, 0, -1):
        j = randrange(i + 1)
        x[i], x[j] = x[j], x[i]
    return x


def sliding_window(w, x):
    """sliding_window: windows of x of length w"""
    x = iterable(x)
    w = int(w)
    if w < 0:
        return list(
            map(lambda e: list(reversed(e)), list(more_itertools.sliding_window(x, -w)))
        )
    else:
        return list(map(list, list(more_itertools.sliding_window(x, w))))


def split(w, x):
    """split: split x into chunks of w"""
    return list(more_itertools.chunked(iterable(x), w))


def split_at(w, x):
    """split_at: split x at occurences of w"""
    return list(more_itertools.split_at(x, lambda a: a == w))


def split_into(w, x):
    """split_into: split x into sizes defined by w"""
    return list(more_itertools.split_into(x, w))


def sublists(x):
    """sublists: return all sublists of x"""
    sub = []
    for i in range(len(x) + 1):
        for j in range(i):
            sub.append(x[j:i])
    return sub


def suffixes(x):
    """suffixes: return the suffixes of x"""
    x = iterable(x, digits_=True)
    res = []
    for i in range(len(x)):
        res.append(x[i:])
    return res[::-1]


def to_braille(x):
    """to_braille: compress boolean matrix x to braille"""
    res = []
    a = 0
    for i in x:
        a -= 1
        res += a % 4 // 3 * [-~len(i) // 2 * [10240]]
        b = 0
        for j in i:
            res[-1][b // 2] |= j << (6429374 >> a % 4 * 6 + b % 2 * 3 & 7)
            b += 1
    return join(10, res)


def type2str(x):
    """type2str: [helper] converts a type to string for dict keying"""
    t = type(x)
    if t == str:
        return "str"
    elif t == list:
        return "lst"
    elif t == ilist:
        return "ils"
    else:
        return "num"


def transpose(x, filler=None):
    """transpose: transpose x"""
    return list(
        map(
            lambda x: list(filter(None.__ne__, x)),
            itertools.zip_longest(*map(iterable, x), fillvalue=filler),
        )
    )


def unrepeat(x):
    """unrepeat: find the pattern in x"""
    return list(map(len, group_equal(x)))


def where(x, upper_level=[]):
    """where: ngn/k's &:"""
    x = iterable(x)
    if type(x[0]) != list:
        return flatten([(upper_level + [i]) * e for i, e in enumerate(x)])
    else:
        return [where(e, upper_level + [i]) for i, e in enumerate(x)]
