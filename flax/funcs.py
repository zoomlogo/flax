# funcs: holds the functions used by atoms
import functools
import itertools
import more_itertools
import operator

from flax.common import mp, mpc, inf, mpf

__all__ = [
    "boolify",
    "depth",
    "diagonals",
    "divisors",
    "fibonacci",
    "find",
    "find_all",
    "find_sublist",
    "flatten",
    "from_base",
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
    "json_decode",
    "lucas",
    "mold",
    "nprimes",
    "ones",
    "order",
    "permutations",
    "prefixes",
    "prime_factors",
    "random",
    "repeat",
    "reshape",
    "sliding_window",
    "split",
    "split_at",
    "sublists",
    "suffixes",
    "to_base",
    "to_bin",
    "to_braille",
    "to_chars",
    "to_digits",
    "unrepeat",
    "vec",
    "vecc",
    "where",
]


def boolify(fn):
    # boolify: return a function which returns a int bool (0/1)
    return lambda *args: int(fn(*args))


def depth(x):
    # depth: returns the depth of x
    return 0 if type(x) != list else (1 if not x else max([depth(a) for a in x]) + 1)


def diagonals(x, antidiagonals=False):
    # diagonals: returns the diagonals or the anti-diagonals of x
    diag = [[] for _ in range(len(x) + len(x[0]) - 1)]
    anti = [[] for _ in range(len(diag))]
    min_d = -len(x) + 1

    for i in range(len(x[0])):
        for j in range(len(x)):
            diag[i + j].append(x[j][i])
            anti[i - j - min_d].append(x[j][i])

    return anti if antidiagonals else diag


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
    # find: find the occurence of w in x
    try:
        return iterable(x, digits=True).index(w)
    except ValueError:
        return []


def find_all(w, x):
    # find_all: returns all indicies of occurences of w in x
    return [i for i, e in enumerate(x) if e == w]


def find_sublist(w, x):
    # find_sublist: find the occurence of the sublist w in x
    x = iterable(x, digits=True)
    w = iterable(w, digits=True)
    for i in range(len(x)):
        if x[i : i + len(w)] == w:
            return i
    return []


def flatten(x):
    # flatten: flatten x
    return list(more_itertools.collapse(x))


def from_base(w, x):
    # from_base: convert x from base w
    x = iterable(x, digits=True)
    sign = -1 if sum(x) < 0 else 1
    num = 0
    i = 0
    for d in x[::-1]:
        num += abs(d) * w**i
        i += 1
    return num * sign


def from_bin(x):
    # from_bin: convert x from binary
    x = iterable(x, digits=True)
    sign = -1 if sum(x) < 0 else 1
    num = 0
    i = 0
    for b in x[::-1]:
        num += abs(b) * 2**i
        i += 1
    return num * sign


def from_digits(x):
    # from_digits: convert x from digits:
    x = iterable(x, range_=True)
    sign = -1 if sum(x) < 0 else 1
    num = 0
    i = 0
    for b in x[::-1]:
        num += abs(b) * 10**i
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
    # index_into: index into w with x
    w = iterable(w, digits=True)
    x = int(x) if type(x) != mpc and int(x) == x else x
    if type(x) == int:
        return w[x % len(w)]
    elif type(x) == mpc:
        return index_into(x.real, index_into(x.imag, w))
    else:
        return [index_into(mp.floor(x), w), index_into(mp.ceil(x), w)]


def iota(x):
    # iota: APL's ⍳ and BQN's ↕
    if type(x) != list:
        return list(range(int(x)))

    res = list(map(list, itertools.product(*(list(range(int(a))) for a in x))))
    for e in x:
        res = split(int(e), res)
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


def json_decode(x):
    # json_decode: convert jsoned x to flax arrays
    if type(x) == list or type(x) == tuple:
        return [json_decode(i) for i in x]
    elif type(x) == str:
        return [ord(i) for i in x]
    elif type(x) == dict:
        return [json_decode(i) for i in x.items()]
    elif x is None:
        return inf
    elif type(x) == bool:
        return int(x)
    else:
        return mpf(x)


@functools.cache
def lucas(x):
    # lucas: nth lucas number
    if x < 2:
        return x + 1
    else:
        return lucas(x - 1) + lucas(x - 2)


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


def ones(x, shape=None, upper_level=[]):
    # ones: matrix with ones at x
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
    return list(map(list, itertools.permutations(iterable(x))))


def prefixes(x):
    # prefixes: return all prefixes of x
    x = iterable(x, digits=True)
    res = []
    for i in range(len(x)):
        res.append(x[: i + 1])
    return res


def prime_factors(x):
    # prime_factors: calculate prime factors of x
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
    # primes: an infinite list of primes
    i = 0
    while True:
        if mp.isprime(i):
            yield i
        i += 1


def random(x):
    # random: return x random floats
    return [mp.rand() for _ in range(x)]


def repeat(w, x):
    # repeat: repeat x according to w
    zipped = itertools.zip_longest(
        iterable(w, digits=True), flatten(iterable(x)), fillvalue=1
    )
    res = []
    for a, b in zipped:
        res.extend(a for _ in range(b))
    return res


def reshape(w, x, level=0):
    # reshape: reshape x according to the shape w
    w = flatten(iterable(w))
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


def sliding_window(w, x):
    # sliding_window: windows of x of length w
    x = iterable(x)
    w = int(w)
    if w < 0:
        return vec(
            lambda e: list(reversed(e)), list(more_itertools.sliding_window(x, -w))
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
            sub.append(x[j:i])
    return sub


def suffixes(x):
    # suffixes: return the suffixes of x
    x = iterable(x, digits=True)
    res = []
    for i in range(len(x)):
        res.append(x[i:])
    return res[::-1]


def to_base(w, x):
    # to_base: convert x into base w
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


def to_bin(x):
    # to_bin: return the binary representation of x
    return [-i if x < 0 else i for i in map(int, bin(x)[3 if x < 0 else 2 :])]


def to_braille(x):
    # to_braille: compress boolean matrix x to braille
    c = [[1, 8], [2, 16], [4, 32], [64, 128]]
    c = [len(x[0]) // 2 * e for e in c]
    c = len(x) // 4 * c
    c = [[c[i][j] * x[i][j] for j in range(len(c[0]))] for i in range(len(c))]
    c = [c[i : i + 4] for i in range(0, len(c), 4)]
    c = [functools.reduce(lambda x, y: list(map(operator.add, x, y)), e) for e in c]
    c = [[c[i][j : j + 2] for j in range(0, len(c), 2)] for i in range(len(c))]
    c = [[sum(i) for i in e] for e in c]
    c = [[10240 + c[i][j] for j in range(len(c[0]))] for i in range(len(c))]
    c = join(10, c)
    return c


def to_chars(x):
    # to_chars: convert x to list of ints
    return [ord(a) for a in x]


def to_digits(x):
    # to_digits: turn x into a list of digits
    return [
        -int(i) if x < 0 else int(i) for i in str(x)[1 if x < 0 else 0 :] if i != "."
    ]


def unrepeat(x):
    # unrepeat: find the pattern in x
    return list(map(len, group_equal(x)))


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


def where(x, upper_level=[]):
    # where: ngn/k's &
    x = iterable(x)
    if type(x[0]) != list:
        return flatten([(upper_level + [i]) * e for i, e in enumerate(x)])
    else:
        return [where(e, upper_level + [i]) for i, e in enumerate(x)]
