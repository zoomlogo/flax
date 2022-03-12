# ======== Imports =======
import sys
import random as rand

import itertools as it
import functools as ft
import more_itertools as mit
import operator as op
import math

from flax.error import error
from mpmath import mp

# Flags
DEBUG = False
PRINT_CHARS = False
DISABLE_GRID = False

# Setup mp context
mp.dps = 20
mp.pretty = True

# Attrdict class
class attrdict(dict):
    def __init__(self, *args, **kwargs):
        dict.__init__(self, *args, **kwargs)
        self.__dict__ = self


# ===== Atom functions =====
depth = (
    lambda x: 0
    if not isinstance(x, list)
    else (1 if not x else max([depth(a) for a in x]) + 1)
)


def diagonals(x):
    d = [[] for _ in range(len(x) + len(x[0]) - 1)]
    min_d = -len(x) + 1

    for i in range(len(x[0])):
        for j in range(len(x)):
            d[i - j - min_d].append(x[j][i])
    return d


def divisors(x):
    res = []

    i = 1
    while i <= x:
        if x % i == 0:
            res.append(i)
        i += 1

    return res


def factors(x):
    return [c for c in range(1, int(x) + 1) if x % c == 0]


@ft.cache
def fibonacci(x):
    if x < 2:
        return x
    return fibonacci(x - 1) + fibonacci(x - 2)


def find(x, y):
    try:
        return iterable(x, make_digits=True).index(y)
    except ValueError:
        return -1


def find_all(x, y):
    return [i for i, e in enumerate(x) if e == y]


flatten = lambda x: list(mit.collapse(x))


def flax_indent(x):
    res = ""
    level = 0
    for i in range(len(x)):
        if x[i] == "[":
            if i != 0 and x[i - 1] == " ":
                res += "\n" + " " * level + "["
            else:
                res += "["
            level += 1
        elif x[i] == "]":
            res += "]"
            level -= 1
        else:
            res += x[i]
    return res


def flax_string(x):
    if not isinstance(x, list):
        if isinstance(x, mp.mpc):
            return "j".join([flax_string(x.real), flax_string(x.imag)])
        return (
            (
                str(int(x))
                if isinstance(x, int)
                or (isinstance(x, mp.mpf) and (x != mp.inf and int(x) == x))
                else str(x)
            )
            .replace("-", "¯")
            .replace("inf", "∞")
        )

    res = ""
    for e in x:
        res += flax_string(e) + " "
    res = "[" + res.strip() + "]"

    return res


def flax_print(x):
    if PRINT_CHARS:
        print("".join(chr(c) for c in flatten(x)))
    else:
        if DISABLE_GRID:
            print(flax_string(x))
        else:
            print(flax_indent(flax_string(x)))
    return x


def from_bin(x):
    x = iterable(x)
    sign = -1 if sum(x) < 0 else 1
    num = 0
    i = 0
    for b in x[::-1]:
        num += abs(b) * 2 ** i
        i += 1
    return num * sign


def from_digits(x):
    x = iterable(x)
    sign = -1 if sum(x) < 0 else 1
    num = 0
    i = 0
    for b in x[::-1]:
        num += abs(b) * 10 ** i
        i += 1
    return num * sign


def grade_down(x):
    x = iterable(x, make_digits=True)

    grades = []
    for a in list(sorted(x))[::-1]:
        grades.append(find_all(a, x))
    return flatten(grades)


def grade_up(x):
    x = iterable(x, make_digits=True)

    grades = []
    for a in list(sorted(x)):
        grades.append(find_all(a, x))
    return flatten(grades)


def group(x):
    res = {}
    for i, it in enumerate(x):
        it = repr(it)
        if it in res:
            res[it].append(i + 1)
        else:
            res[it] = [i + 1]
    return [res[k] for k in sorted(res, key=eval)]


def group_equal(x):
    res = []
    for e in x:
        if res and res[-1][0] == e:
            res[-1].append(e)
        else:
            res.append([e])
    return res


def index_generator(x):
    if not isinstance(x, list):
        return list(range(int(x)))

    res = list(map(list, it.product(*(list(range(int(a))) for a in x))))
    for e in x:
        res = split(res, int(e))
    return res[0]


def index_into(x, y):
    x = iterable(x, make_digits=True)
    y = int(y) if int(y) == y else y
    if isinstance(y, int):
        return x[y % len(x)]
    return [index_into(x, mp.floor(y)), index_into(x, mp.ceil(y))]


def indices_multidimensional(x, up_lvl=[]):
    a_in = []
    for i, item in enumerate(x):
        if not isinstance(item, list):
            a_in.append(up_lvl + [i + 1])
        else:
            a_in.extend(indices_multidimensional(item, up_lvl=up_lvl + [i + 1]))
    return a_in


def iterable(x, make_range=False, make_digits=False):
    if not isinstance(x, list):
        if make_range:
            return list(range(int(x)))
        if make_digits:
            return to_digits(x)
        return [x]
    return x


def join(x, y):
    x = iterable(x)
    y = it.cycle(iterable(y))
    return flatten(zip(x, y))


def join_newlines(x):
    return join(x, 10)


def join_spaces(x):
    return join(x, 32)


lzip = lambda *x, fillvalue=0: [
    list(x) for x in it.zip_longest(*x, fillvalue=fillvalue)
]


def mold(x, y):
    for i in range(len(y)):
        if isinstance(y[i], list):
            mold(x, y[i])
        else:
            item = x.pop(0)
            y[i] = item
            x.append(item)
    return y


def nprimes(x):
    res = []
    i = 2
    while len(res) != x:
        if mp.isprime(i):
            res.append(i)
        i += 1
    return res


def order(x, y):
    if x == 0 or abs(y) == 1:
        return mp.inf

    if y == 0:
        return 0

    res = 0
    while True:
        x, r = divmod(x, y)
        if r:
            break
        res += 1
    return res


def prefixes(x):
    res = []
    for i in range(len(x)):
        res.append(x[: i + 1])
    return res


def random(x):
    if x == 0:
        return rand.random()
    else:
        return rand.randrange(int(x))


def repeat(x, y):
    zipped = lzip(iterable(y, make_digits=True), flatten(iterable(x)), fillvalue=1)
    res = []
    for a, b in zipped:
        res.extend(a for _ in range(b))
    return res


def reshape(x, y):
    y = iterable(y)
    if not isinstance(x, it.cycle):
        x = it.cycle(iterable(x))

    if len(y) == 1:
        return [next(x) for _ in range(y[0])]
    else:
        return [reshape(x, y[1:]) for _ in range(y[0])]


def reverse_every_other(x):
    x = iterable(x)

    for i in range(len(x)):
        if i % 2:
            x[i] = list(reversed(iterable(x[i])))
    return x


def sliding_window(x, y):
    x = iterable(x)
    y = int(y)
    if y < 0:
        return vecd(lambda x: list(reversed(x)))(list(mit.sliding_window(x, -y)))
    else:
        return vecd(list)(list(mit.sliding_window(x, y)))


split = lambda x, y: list(mit.chunked(x, y))

split_at = lambda x, y: list(mit.split_at(x, lambda a: a == y))


def sub_lists(l):
    lists = [[]]
    for i in range(len(l) + 1):
        for j in range(i):
            lists.append(l[j:i])
    return lists


def suffixes(x):
    res = []
    for i in range(len(x)):
        res.append(x[i:])
    return res


def to_bin(x):
    return [-i if x < 0 else i for i in map(int, bin(x)[3 if x < 0 else 2 :])]


def to_digits(x):
    return [-int(i) if x < 0 else int(i) for i in str(x)[1 if x < 0 else 0 :]]


def to_chars(x):
    return [ord(v) for v in x]


def vec(fn, x, y=None, lfull=True, rfull=True):
    if y is None:
        return [vec(fn, a) for a in x] if depth(x) != 0 and lfull else fn(x)
    dx, dy = depth(x), depth(y)
    # TODO: Clean this mess
    if lfull and rfull:
        if dx == dy:
            return [vec(fn, a, b) for a, b in zip(x, y)] if dx != 0 else fn(x, y)
        else:
            return [vec(fn, x, b) for b in y] if dx < dy else [vec(fn, a, y) for a in x]
    elif (not lfull) and rfull:
        return [vec(fn, x, b) for b in y] if dy > 0 else fn(x, y)
    elif lfull and (not rfull):
        return [vec(fn, a, y) for a in x] if dx > 0 else fn(x, y)
    else:
        return fn(x, y)


vecd = lambda fn, lfull=True, rfull=True: lambda x, y=None: vec(fn, x, y, lfull, rfull)


# ====== Atoms ========
atoms = {
    # Single byte nilads
    "₀": attrdict(arity=0, call=lambda: 100),
    "₁": attrdict(arity=0, call=lambda: [0, 1]),
    "₂": attrdict(arity=0, call=lambda: 10),
    "₃": attrdict(arity=0, call=lambda: 16),
    "₄": attrdict(arity=0, call=lambda: 32),
    "₅": attrdict(arity=0, call=lambda: 64),
    "₆": attrdict(arity=0, call=lambda: 0),
    "₇": attrdict(arity=0, call=lambda: 26),
    "₈": attrdict(arity=0, call=lambda: ord(sys.stdin.read(1))),
    "₉": attrdict(arity=0, call=lambda: [ord(c) for c in input()]),
    "₎": attrdict(arity=0, call=lambda: []),
    "₍": attrdict(arity=0, call=lambda: []),
    # Single byte monads
    "A": attrdict(arity=1, call=vecd(lambda a: abs(a))),
    "Ạ": attrdict(arity=1, call=lambda x: int(all(iterable(x, make_digits=True)))),
    "Ȧ": attrdict(arity=1, call=lambda x: int(any(iterable(x)))),
    "Ă": attrdict(arity=1, call=lambda x: int(iterable(x) > [] and all(flatten(x)))),
    "Æ": attrdict(arity=1, call=vecd(lambda a: int(mp.isprime(a)))),
    "B": attrdict(arity=1, call=vecd(to_bin)),
    "Ḃ": attrdict(arity=1, call=from_bin),
    "Ḅ": attrdict(arity=1, call=vecd(lambda a: 2 ** a)),
    "Ƀ": attrdict(arity=1, call=vecd(lambda a: a % 2)),
    "C": attrdict(arity=1, call=vecd(lambda a: 1 - a)),
    "Ċ": attrdict(arity=1, call=vecd(lambda a: a * 3)),
    "Ç": attrdict(arity=1, call=lambda x: split(x, 2)),
    "D": attrdict(arity=1, call=vecd(to_digits)),
    "Ḋ": attrdict(arity=1, call=from_digits),
    "Ḍ": attrdict(arity=1, call=vecd(divisors)),
    "Ð": attrdict(arity=1, call=vecd(lambda a: a * 2)),
    "Ď": attrdict(
        arity=1,
        call=lambda x: [ft.reduce(lambda x, y: y - x, a) for a in sliding_window(x, 2)],
    ),
    "E": attrdict(arity=1, call=vecd(lambda a: list(range(1, a + 1)))),
    "F": attrdict(arity=1, call=flatten),
    "Ḟ": attrdict(
        arity=1,
        call=lambda x: [
            i for i, e in enumerate(iterable(x, make_digits=True)) if not e
        ],
    ),
    "G": attrdict(arity=1, call=lambda x: group_equal(iterable(x, make_digits=True))),
    "H": attrdict(arity=1, call=lambda x: iterable(x, make_digits=True)[0]),
    "Ḣ": attrdict(arity=1, call=lambda x: iterable(x, make_digits=True)[-1]),
    "I": attrdict(arity=1, call=index_generator),
    "J": attrdict(arity=1, call=join_spaces),
    "Ĵ": attrdict(arity=1, call=join_newlines),
    "K": attrdict(arity=1, call=lambda x: list(it.accumulate(iterable(x)))),
    "L": attrdict(arity=1, call=len),
    "M": attrdict(arity=1, call=vecd(lambda a: a ** 2)),
    "N": attrdict(arity=1, call=vecd(lambda a: -a)),
    "O": attrdict(arity=1, call=lambda x: x),
    "P": attrdict(arity=1, call=lambda x: list(it.permutations(x))),
    "Ṗ": attrdict(arity=1, call=lambda x: print(end="".join(chr(c) for c in x)) or x),
    "Ƥ": attrdict(arity=1, call=lambda x: flax_print(x)),
    "Q": attrdict(arity=1, call=vecd(lambda a: a / 2)),
    "R": attrdict(
        arity=1,
        call=lambda x: [z[::-1] if isinstance(z, list) else z for z in x]
        if isinstance(x, list)
        else to_digits(x)[::-1],
    ),
    "Ř": attrdict(
        arity=1, call=lambda x: list(range(len(iterable(x, make_digits=True))))
    ),
    "S": attrdict(arity=1, call=lambda x: list(sorted(iterable(x)))),
    "Ṡ": attrdict(arity=1, call=lambda x: list(reversed(sorted(iterable(x))))),
    "T": attrdict(arity=1, call=lambda x: iterable(x, make_digits=True)[1:]),
    "Ṫ": attrdict(arity=1, call=lambda x: iterable(x, make_digits=True)[:-1]),
    "Ṭ": attrdict(
        arity=1,
        call=lambda x: [i for i, e in enumerate(iterable(x, make_digits=True)) if e],
    ),
    "U": attrdict(arity=1, call=lambda x: list(set(iterable(x)))),
    "V": attrdict(arity=1, call=lambda x: group(iterable(x, make_digits=True))),
    "W": attrdict(arity=1, call=lambda x: [x]),
    "Ẇ": attrdict(
        arity=1, call=lambda x: [x] if not isinstance(x, list) or len(x) != 1 else x
    ),
    "X": attrdict(arity=1, call=lambda x: split(x, int(len(x) / 2))),
    "Y": attrdict(arity=1, call=lambda x: [x[i] for i in range(len(x)) if i % 2 == 0]),
    "Ẏ": attrdict(arity=1, call=lambda x: [x[i] for i in range(len(x)) if i % 2]),
    "Z": attrdict(arity=1, call=lambda x: lzip(*x)),
    "Π": attrdict(arity=1, call=lambda x: ft.reduce(op.mul, flatten(x))),
    "Σ": attrdict(arity=1, call=sum),
    "!": attrdict(arity=1, call=vecd(lambda a: mp.factorial(a))),
    "~": attrdict(arity=1, call=vecd(lambda a: ~a)),
    "¬": attrdict(arity=1, call=(vecd(lambda a: int(not a)))),
    "√": attrdict(arity=1, call=vecd(mp.sqrt)),
    "≈": attrdict(
        arity=1,
        call=lambda a: int(mit.all_equal(a)),
    ),
    "∇": attrdict(arity=1, call=lambda x: min(iterable(x, make_digits=True))),
    "∆": attrdict(arity=1, call=lambda x: max(iterable(x, make_digits=True))),
    "±": attrdict(
        arity=1,
        call=vecd(lambda a: -1 if a < 0 else (0 if a == 0 else 1)),
    ),
    "Θ": attrdict(arity=1, call=lambda x: iterable(x).insert(0, 0)),
    "⌽": attrdict(arity=1, call=lambda x: iterable(x, make_range=True)[::-1]),
    "{": attrdict(arity=1, call=vecd(lambda a: a - 1)),
    "}": attrdict(arity=1, call=vecd(lambda a: a + 1)),
    "ε": attrdict(arity=1, call=lambda x: sub_lists(iterable(x, make_range=True))),
    "σ": attrdict(arity=1, call=reverse_every_other),
    "?": attrdict(arity=1, call=vecd(random)),
    "⍋": attrdict(arity=1, call=grade_up),
    "⍒": attrdict(arity=1, call=grade_down),
    "⅟": attrdict(arity=1, call=vecd(lambda a: 1 / a if a else mp.inf)),
    "⌈": attrdict(arity=1, call=vecd(lambda a: int(mp.ceil(a)))),
    "⌊": attrdict(arity=1, call=vecd(lambda a: int(mp.floor(a)))),
    "(": attrdict(arity=1, call=prefixes),
    ")": attrdict(arity=1, call=suffixes),
    "∀": attrdict(arity=1, call=lambda x: list(map(sum, x))),
    # Single byte dyads
    "ċ": attrdict(
        arity=2,
        call=vecd(lambda x, y: iterable(x, make_digits=True).count(y), lfull=False),
    ),
    "d": attrdict(arity=2, call=vecd(lambda a, b: list(divmod(a, b)))),
    "ḍ": attrdict(
        arity=2,
        call=vecd(lambda a, b: a % b == 0 if b else mp.inf),
    ),
    "f": attrdict(
        arity=2,
        call=lambda x, y: [a for a in iterable(x, make_digits=True) if a not in y],
    ),
    "ḟ": attrdict(
        arity=2,
        call=lambda x, y: [a for a in iterable(x, make_digits=True) if a in y],
    ),
    "g": attrdict(arity=2, call=vecd(lambda a, b: int(math.gcd(a, b)))),
    "h": attrdict(
        arity=2,
        call=vecd(lambda x, y: iterable(x, make_digits=True)[:y], lfull=False),
    ),
    "i": attrdict(arity=2, call=vecd(index_into, lfull=False)),
    "l": attrdict(arity=2, call=vecd(lambda a, b: int(math.lcm(a, b)))),
    "m": attrdict(arity=2, call=lambda x, y: mold(iterable(x), iterable(y))),
    "n": attrdict(arity=2, call=vecd(op.floordiv)),
    "o": attrdict(arity=2, call=split_at),
    "q": attrdict(arity=2, call=vecd(order)),
    "r": attrdict(
        arity=2,
        call=vecd(lambda a, b: list(range(a, b + 1))),
    ),
    "s": attrdict(arity=2, call=split),
    "t": attrdict(
        arity=2,
        call=vecd(lambda x, y: iterable(x, make_digits=True)[y - 1 :], lfull=False),
    ),
    "u": attrdict(
        arity=2, call=lambda x, y: [find(y, a) for a in iterable(x, make_digits=True)]
    ),
    "x": attrdict(arity=2, call=vecd(sliding_window, lfull=False)),
    "y": attrdict(arity=2, call=join),
    "z": attrdict(arity=2, call=lzip),
    "+": attrdict(arity=2, call=vecd(op.add)),
    "-": attrdict(arity=2, call=vecd(op.sub)),
    "×": attrdict(arity=2, call=vecd(op.mul)),
    "÷": attrdict(
        arity=2,
        call=vecd(lambda a, b: a / b if b else (mp.inf if a else 0)),
    ),
    "%": attrdict(
        arity=2,
        call=vecd(lambda a, b: a % b if b else (mp.inf if a else 0)),
    ),
    "*": attrdict(arity=2, call=vecd(op.pow)),
    "<": attrdict(arity=2, call=vecd(lambda a, b: int(a < b))),
    ">": attrdict(arity=2, call=vecd(lambda a, b: int(a > b))),
    "=": attrdict(arity=2, call=vecd(lambda a, b: int(a == b))),
    "≠": attrdict(arity=2, call=vecd(lambda a, b: int(a != b))),
    "≥": attrdict(arity=2, call=vecd(lambda a, b: int(a >= b))),
    "≤": attrdict(arity=2, call=vecd(lambda a, b: int(a <= b))),
    "≡": attrdict(arity=2, call=lambda x, y: int(x == y)),
    "≢": attrdict(arity=2, call=lambda x, y: int(x != y)),
    ",": attrdict(arity=2, call=lambda x, y: iterable(x) + iterable(y)),
    "⍪": attrdict(arity=2, call=lambda x, y: iterable(y) + iterable(x)),
    "⋈": attrdict(arity=2, call=lambda x, y: [x, y]),
    "∧": attrdict(arity=2, call=vecd(lambda a, b: a and b)),
    "∨": attrdict(arity=2, call=vecd(lambda a, b: a or b)),
    "∊": attrdict(arity=2, call=vecd(lambda x, y: x in y, rfull=False)),
    "&": attrdict(arity=2, call=vecd(op.and_)),
    "|": attrdict(arity=2, call=vecd(op.or_)),
    "^": attrdict(arity=2, call=vecd(op.xor)),
    "⊂": attrdict(
        arity=2,
        call=vecd(find, lfull=False),
    ),
    "⊆": attrdict(arity=2, call=vecd(find_all, lfull=False)),
    "⊏": attrdict(
        arity=2,
        call=vecd(
            lambda a, b: [
                iterable(a, make_range=True)[i]
                for i in range(len(iterable(a, make_range=True)))
                if i % b == 0
            ],
            lfull=False,
        ),
    ),
    "⊣": attrdict(arity=2, call=lambda x, y: x),
    "⊢": attrdict(arity=2, call=lambda x, y: y),
    "·": attrdict(arity=2, call=lambda x, y: list(it.product(x, y))),
    "/": attrdict(arity=2, call=repeat),
    "#": attrdict(arity=2, call=reshape),
    # Niladic diagraphs
    "_₁": attrdict(arity=0, call=lambda: 128),
    "_₂": attrdict(arity=0, call=lambda: 256),
    "_₃": attrdict(arity=0, call=lambda: 512),
    "_₄": attrdict(arity=0, call=lambda: 1024),
    "_₅": attrdict(arity=0, call=lambda: 2048),
    "_₆": attrdict(arity=0, call=lambda: 4096),
    "_₇": attrdict(arity=0, call=lambda: 8192),
    "_₈": attrdict(arity=0, call=lambda: 16384),
    "_₉": attrdict(arity=0, call=lambda: 32768),
    "_0": attrdict(arity=0, call=lambda: [0, 0]),
    "_1": attrdict(arity=0, call=lambda: [1, 1]),
    "_3": attrdict(arity=0, call=lambda: 1000),
    "_4": attrdict(arity=0, call=lambda: 10000),
    "_5": attrdict(arity=0, call=lambda: 100000),
    "_6": attrdict(arity=0, call=lambda: 1000000),
    "_A": attrdict(arity=0, call=lambda: to_chars("ABCDEFGHIJKLMNOPQRSTUVWXYZ")),
    "_C": attrdict(arity=0, call=lambda: to_chars("BCDFGHJKLMNPQRSTVWXYZ")),
    "_D": attrdict(arity=0, call=lambda: [[0, 1], [1, 0], [0, -1], [-1, 0]]),
    "_H": attrdict(arity=0, call=lambda: to_chars("Hello, World!")),
    "_L": attrdict(
        arity=0,
        call=lambda: to_chars("abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ"),
    ),
    "_P": attrdict(arity=0, call=lambda: mp.phi),
    "_S": attrdict(arity=0, call=lambda: [1, -1]),
    "_V": attrdict(arity=0, call=lambda: to_chars("AEIOU")),
    "_a": attrdict(arity=0, call=lambda: to_chars("abcdefghijklmnopqrstuvwxyz")),
    "_c": attrdict(arity=0, call=lambda: to_chars("bcdfghjklmnpqrstvwxyz")),
    "_e": attrdict(arity=0, call=lambda: mp.e),
    "_h": attrdict(arity=0, call=lambda: to_chars("Hello World")),
    "_p": attrdict(arity=0, call=lambda: mp.pi),
    "_v": attrdict(arity=0, call=lambda: to_chars("aeiou")),
    "_y": attrdict(arity=0, call=lambda: to_chars("aeiouy")),
    "_∞": attrdict(arity=0, call=lambda: mp.inf),
    "_⁰": attrdict(arity=0, call=lambda: 2 ** 20),
    "_¹": attrdict(arity=0, call=lambda: 2 ** 30),
    "_²": attrdict(arity=0, call=lambda: 2 ** 100),
    "_(": attrdict(arity=0, call=lambda: to_chars("()")),
    "_{": attrdict(arity=0, call=lambda: to_chars("{}")),
    "_[": attrdict(arity=0, call=lambda: to_chars("[]")),
    # Monadic diagraphs
    ";C": attrdict(arity=1, call=vecd(mp.cos)),
    ";Ċ": attrdict(arity=1, call=vecd(mp.acos)),
    ";D": attrdict(arity=1, call=diagonals),
    ";Ḋ": attrdict(arity=1, call=depth),
    ";F": attrdict(arity=1, call=vecd(fibonacci)),
    ";S": attrdict(arity=1, call=vecd(mp.sin)),
    ";Ṡ": attrdict(arity=1, call=vecd(mp.asin)),
    ";T": attrdict(arity=1, call=vecd(mp.tan)),
    ";Ṫ": attrdict(arity=1, call=vecd(mp.atan)),
    ";c": attrdict(arity=1, call=vecd(lambda a: 1 / mp.cos(a))),
    ";d": attrdict(arity=1, call=vecd(lambda x: int(48 <= x <= 57))),
    ";h": attrdict(arity=1, call=vecd(mp.tanh)),
    ";i": attrdict(arity=1, call=indices_multidimensional),
    ";n": attrdict(arity=1, call=vecd(mp.sinh)),
    ";o": attrdict(arity=1, call=vecd(mp.cosh)),
    ";s": attrdict(arity=1, call=vecd(lambda a: 1 / mp.sin(a))),
    ";t": attrdict(arity=1, call=vecd(lambda a: 1 / mp.tan(a))),
    ";Æ": attrdict(arity=1, call=vecd(nprimes)),
    ";I": attrdict(arity=1, call=vecd(int)),
    ";f": attrdict(arity=1, call=vecd(factors)),
    ";r": attrdict(arity=1, call=vecd(lambda a: list(range(2, int(a))))),
    ";R": attrdict(arity=1, call=vecd(lambda a: list(range(int(a) + 1)))),
    # Dyadic diagraphs
    ":T": attrdict(arity=2, call=vecd(mp.atan2)),
    ":<": attrdict(arity=2, call=vecd(lambda a, b: a << b)),
    ":>": attrdict(arity=2, call=vecd(lambda a, b: a >> b)),
    ":s": attrdict(
        arity=2,
        call=lambda x, y: [(z := iterable(y, make_digits=True))[0]]
        + iterable(x)
        + [z[-1]],
    ),
    ":*": attrdict(arity=2, call=lambda x, y: list(it.product(x, repeat=y))),
    ":·": attrdict(
        arity=2, call=lambda x, y: sum(x[i][0] * y[i] for i in range(len(y)))
    ),
}

for k in atoms:
    atoms[k].glyph = k

# ===== Chain functions ====
def arities(links):
    return [link.arity for link in links]


def copy_to(atom, value):
    atom.call = lambda: value
    return value


def create_chain(chain, arity=-1, isF=True):
    return attrdict(
        arity=arity,
        chain=chain,
        call=lambda x=None, y=None: variadic_chain(chain, *(isF and (x, y) or (y, x))),
    )


def dyadic_chain(chain, x, y):
    for link in chain:
        if link.arity < 0:
            link.arity = 2

    if chain and arities(chain[0:3]) == [2, 2, 2]:
        accumulator = chain[0].call(x, y)
        chain = chain[1:]
    elif leading_nilad(chain):
        accumulator = chain[0].call()
        chain = chain[1:]
    else:
        accumulator = x

    atoms["₍"].call = lambda: x
    atoms["₎"].call = lambda: y

    try:
        while chain:
            if DEBUG:
                print(
                    f"DEBUG: λ: {flax_string(accumulator)}, chain: {flax_string([x.glyph for x in chain])}"
                )
            if arities(chain[0:3]) == [2, 2, 0] and leading_nilad(chain[2:]):
                accumulator = chain[1].call(
                    chain[0].call(accumulator, y), chain[2].call()
                )
                chain = chain[3:]
            elif arities(chain[0:2]) == [2, 2]:
                accumulator = chain[0].call(accumulator, chain[1].call(x, y))
                chain = chain[2:]
            elif arities(chain[0:2]) == [2, 0]:
                accumulator = chain[0].call(accumulator, chain[1].call())
                chain = chain[2:]
            elif arities(chain[0:2]) == [0, 2]:
                accumulator = chain[1].call(chain[0].call(), accumulator)
                chain = chain[2:]
            elif chain[0].arity == 2:
                accumulator = chain[0].call(accumulator, y)
                chain = chain[1:]
            elif chain[0].arity == 1:
                accumulator = chain[0].call(accumulator)
                chain = chain[1:]
            else:
                flax_print(accumulator)
                accumulator = chain[0].call()
                chain = chain[1:]
    except FileNotFoundError:
        ...

    return accumulator


def last_input():
    if len(sys.argv) > 2:
        return sys.argv[-1]
    else:
        return eval(input())


def leading_nilad(chain):
    return chain and arities(chain) + [1] < [0, 2] * len(chain)


def max_arity(links):
    return (
        max(arities(links))
        if min(arities(links)) > -1
        else (~max(arities(links))) or -1
    )


def monadic_chain(chain, x):
    init = True

    accumulator = x
    atoms["₍"].call = lambda: x

    try:
        while 1:
            if DEBUG:
                print(f"DEBUG: λ: {flax_string(accumulator)}, chain: {chain}")
            if init:
                for link in chain:
                    if link.arity < 0:
                        link.arity = 1

                if leading_nilad(chain):
                    accumulator = chain[0].call()
                    chain = chain[1:]
                init = False
            if not chain:
                break

            if arities(chain[0:2]) == [2, 1]:
                accumulator = chain[0].call(accumulator, chain[1].call(x))
                chain = chain[2:]
            elif arities(chain[0:2]) == [2, 0]:
                accumulator = chain[0].call(accumulator, chain[1].call())
                chain = chain[2:]
            elif arities(chain[0:2]) == [0, 2]:
                accumulator = chain[1].call(chain[0].call(), accumulator)
                chain = chain[2:]
            elif chain[0].arity == 2:
                accumulator = chain[0].call(accumulator, x)
                chain = chain[1:]
            elif chain[0].arity == 1:
                if not chain[1:] and hasattr(chain[0], "chain"):
                    x = accumulator
                    chain = chain[0].chain
                    init = True
                else:
                    accumulator = chain[0].call(accumulator)
                    chain = chain[1:]
            else:
                flax_print(accumulator)
                accumulator = chain[0].call()
                chain = chain[1:]
    except FileNotFoundError:
        ...

    return accumulator


def niladic_chain(chain):
    if not chain or chain[0].arity > 0:
        return monadic_chain(chain, 0)
    return monadic_chain(chain[1:], chain[0].call())


def ntimes(links, args):
    times = int(links[1].call()) if len(links) == 2 else last_input()
    res, y = args
    for _ in range(times):
        x = res
        res = variadic_link(links[0], x, y)
        y = x
    return res


def qfilter(links, outer_links, i):
    res = [attrdict(arity=links[0].arity or 1)]
    if links[0].arity == 0:
        res[0].call = lambda x: list(
            filter(lambda z: z != links[0].call(), iterable(x, make_range=True))
        )
    else:
        res[0].call = lambda x, y=None: list(
            filter(
                lambda z: variadic_link(links[0], z, y), iterable(x, make_range=True)
            )
        )
    return res


def qfold(links, outer_links, i):
    res = [attrdict(arity=1)]
    if len(links) == 1:
        res[0].call = lambda x, y=None: ft.reduce(links[0].call, x)
    else:
        res[0].call = lambda x, y=None: [
            ft.reduce(links[0].call, z) for z in sliding_window(x, links[1].call())
        ]
    return res


def qscan(links, outer_links, i):
    res = [attrdict(arity=1)]
    if len(links) == 1:
        res[0].call = lambda x, y=None: list(it.accumulate(x, links[0].call))
    else:
        res[0].call = lambda x, y=None: [
            list(it.accumulate(z, links[0].call))
            for z in sliding_window(x, links[1].call())
        ]
    return res


def qsort(links, outer_links, i):
    res = [attrdict(arity=links[0].arity or 1)]
    # TODO: Overload on nilad
    res[0].call = lambda x, y=None: list(
        sorted(
            iterable(x, make_digits=True),
            key=lambda z: variadic_link(links[0], z, y),
        )
    )
    return res


def quick_chain(arity, min_length):
    return attrdict(
        condition=(lambda links: len(links) >= min_length and links[0].arity == 0)
        if arity == 0
        else lambda links: len(links)
        - sum([leading_nilad(x) for x in split_suffix(links)[:-1]])
        >= min_length,
        qlink=lambda links, outer_links, i: [
            attrdict(
                arity=arity, call=lambda x=None, y=None: variadic_chain(links, x, y)
            )
        ],
    )


def replace_indicies(x, with_, at, y):
    x = iterable(x, make_digits=True)
    at = flatten(iterable(at))
    for i in at:
        if with_.arity == 0:
            x[i] = with_.call()
        elif with_.arity == 1:
            x[i] = with_.call(x[i])
        else:
            x[i] = with_.call(x[i], y)
    return x


def split_suffix(array):
    array = iterable(array)
    return [array[i:] for i in range(len(array))]


def variadic_chain(chain, *args):
    args = list(filter(None.__ne__, args))
    if len(args) == 0:
        return niladic_chain(chain)
    elif len(args) == 1:
        return monadic_chain(chain, *args)
    else:
        return dyadic_chain(chain, *args)


def variadic_link(link, *args, swap=False):
    if link.arity < 0:
        args = list(filter(None.__ne__, args))
        link.arity = len(args)

    if link.arity == 0:
        return link.call()
    elif link.arity == 1:
        return link.call(args[0])
    elif link.arity == 2:
        if swap:
            return link.call(args[1], args[0])
        else:
            return link.call(args[0], args[1])


def while_loop(link, cond, args):
    res, y = args
    while variadic_link(cond, res, y):
        x = res
        res = variadic_link(link, x, y)
        y = x
    return res


# ========= Quicks ==========
quicks = {
    "⁶": attrdict(
        condition=lambda links: links,
        qlink=lambda links, outer_links, i: [
            attrdict(
                arity=links[0].arity,
                call=lambda x=None, y=None: copy_to(
                    atoms["₆"], variadic_link(links[0], x, y)
                ),
            )
        ],
    ),
    "ᵝ": attrdict(
        condition=lambda links: True,
        qlink=lambda links, outer_links, i: [create_chain(outer_links[i])],
    ),
    "¨": attrdict(
        condition=lambda links: links,
        qlink=lambda links, outer_links, i: [
            attrdict(
                arity=links[0].arity or 1,
                call=lambda x, y=None: [variadic_link(links[0], a, y) for a in x]
                if isinstance(x, list)
                else variadic_link(links[0], x, y),
            )
        ],
    ),
    '"': attrdict(
        condition=lambda links: links,
        qlink=lambda links, outer_links, i: [
            attrdict(
                arity=2,
                call=lambda x, y: [variadic_link(links[0], x, b) for b in y]
                if isinstance(y, list)
                else variadic_link(links[0], x, y),
            )
        ],
    ),
    "⁰": attrdict(
        condition=lambda links: True,
        qlink=lambda links, outer_links, i: [create_chain(outer_links[i], 0)],
    ),
    "¹": attrdict(
        condition=lambda links: True,
        qlink=lambda links, outer_links, i: [create_chain(outer_links[i], 1)],
    ),
    "²": attrdict(
        condition=lambda links: True,
        qlink=lambda links, outer_links, i: [create_chain(outer_links[i], 2)],
    ),
    "˙": quick_chain(0, 2),
    "ᴹ": quick_chain(1, 2),
    "ᵐ": quick_chain(1, 3),
    "ᶲ": quick_chain(1, 4),
    "ᴰ": quick_chain(2, 2),
    "ᵈ": quick_chain(2, 3),
    "ᵠ": quick_chain(2, 4),
    "˜": attrdict(
        condition=lambda links: links,
        qlink=lambda links, outer_links, i: [
            attrdict(
                arity=links[0].arity,
                call=lambda x=None, y=None: variadic_link(links[0], x, y, swap=True),
            )
        ],
    ),
    "˘": attrdict(
        condition=lambda links: links,
        qlink=lambda links, outer_links, i: [
            attrdict(
                arity=links[0].arity and 1,
                call=lambda x=None: variadic_link(links[0], x, x),
            )
        ],
    ),
    "´": attrdict(condition=lambda links: links and links[0].arity, qlink=qfold),
    "`": attrdict(condition=lambda links: links and links[0].arity, qlink=qscan),
    "⌜": attrdict(
        condition=lambda links: links,
        qlink=lambda links, outer_links, i: [
            attrdict(
                arity=2,
                call=lambda x, y: [
                    [links[0].call(a, b) for a in iterable(x, make_range=True)]
                    for b in iterable(y, make_range=True)
                ],
            )
        ],
    ),
    "ⁿ": attrdict(
        condition=lambda links: len(links) == 2,
        qlink=lambda links, outer_links, i: (
            [links.pop(0)] if len(links) == 2 and links[0].arity == 0 else []
        )
        + [
            attrdict(
                arity=max_arity(links),
                call=lambda x=None, y=None: ntimes(links, (x, y)),
            )
        ],
    ),
    "ˀ": attrdict(
        condition=lambda links: len(links) == 3,
        qlink=lambda links, outer_links, i: [
            attrdict(
                arity=max(arities(links)),
                call=lambda x=None, y=None: (
                    variadic_link(links[0], x, y)
                    if variadic_link(links[2], x, y)
                    else variadic_link(links[1], x, y)
                ),
            )
        ],
    ),
    "ᵂ": attrdict(
        condition=lambda links: len(links) == 2,
        qlink=lambda links, outer_links, i: [
            attrdict(
                arity=max(arities(links)),
                call=lambda x=None, y=None: while_loop(links[0], links[1], (x, y)),
            )
        ],
    ),
    "⁽": attrdict(
        condition=lambda links: links,
        qlink=lambda links, outer_links, i: [
            attrdict(arity=2, call=lambda x, y: links[0].call(x))
        ],
    ),
    "⁾": attrdict(
        condition=lambda links: links,
        qlink=lambda links, outer_links, i: [
            attrdict(arity=2, call=lambda x, y: links[0].call(y))
        ],
    ),
    "ᐣ": attrdict(
        condition=lambda links: links and links[0].arity == 2,
        qlink=lambda links, outer_links, i: [
            attrdict(
                arity=2,
                call=lambda x, y: links[0].call(
                    x,
                    ft.reduce(
                        lambda f, g: lambda a: f(g(a)), [a.call for a in links[1:]]
                    )(y),
                ),
            )
        ],
    ),
    "ᶠ": attrdict(condition=lambda links: links, qlink=qfilter),
    "ˢ": attrdict(condition=lambda links: links, qlink=qsort),
    "°": attrdict(
        condition=lambda links: len(links) == 2,
        qlink=lambda links, outer_links, i: [
            attrdict(
                arity=links[0].arity or 1,
                call=lambda x, y=None: replace_indicies(
                    x, links[0], links[1].call(), y
                ),
            )
        ],
    ),
    "⁺": attrdict(
        condition=lambda links: links,
        qlink=lambda links, outer_links, i: [
            attrdict(
                arity=links[0].arity,
                call=lambda x, y=None: max(
                    iterable(x), key=lambda z: variadic_link(links[0], z, y)
                ),
            )
        ],
    ),
    "⁻": attrdict(
        condition=lambda links: links,
        qlink=lambda links, outer_links, i: [
            attrdict(
                arity=links[0].arity,
                call=lambda x, y=None: min(
                    iterable(x), key=lambda z: variadic_link(links[0], z, y)
                ),
            )
        ],
    ),
}

for k in quicks:
    quicks[k].glyph = k

# == Train Separators ==

train_separators = {
    "ø": (0, True),
    "µ": (1, True),
    "г": (1, True),
    "ð": (2, True),
    "ɓ": (2, False),
}
