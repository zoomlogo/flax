# ======== Imports =======
import sys
import random as R

from collections import deque
from math import factorial, floor, ceil

from pyhof import *
import itertools as it
import functools as ft
import more_itertools as mit
import operator as op
import sympy

# Flags
DEBUG = 0

# Attrdict class
class attrdict(dict):
    def __init__(self, *args, **kwargs):
        dict.__init__(self, *args, **kwargs)
        self.__dict__ = self


# ===== Atom functions =====
def contains_false(x):
    x = iterable(x, make_digits=True)
    if not isinstance(x, list):
        return 1 if x else 0

    if len(x) == 0:
        return 1

    return 1 if 1 in [*map(contains_false, x)] else 0


depth = (
    lambda x: 0 if not isinstance(x, list) else (1 if not x else max(map(depth, x)) + 1)
)


def diagonals(x):
    d = [*map(curry(constantly)([]), range(len(x) + len(x[0]) - 1))]
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


def dyadic_vectorise(fn, x, y):
    dx = depth(x)
    dy = depth(y)

    if dx == dy:
        if dx != 0:
            return [dyadic_vectorise(fn, a, b) for a, b in zip(x, y)]
        else:
            return fn(x, y)
    else:
        if dx < dy:
            return [dyadic_vectorise(fn, x, b) for b in y]
        else:
            return [dyadic_vectorise(fn, a, y) for a in x]


def flatten(L):
    def gen(l):
        if not isinstance(l, list):
            yield l
        else:
            for i in l:
                yield from flatten(i)

    return [*gen(L)]


def falsey_indices(x):
    x = iterable(x, make_digits=True)
    if not isinstance(x, list):
        return []

    i = 0
    indices = []
    while i < len(x):
        if not x[i]:
            indices.append(i)
        i += 1
    return indices


def find_all_indices(x, y):
    res = []
    i = 0
    while i < len(y):
        if y[i] == x:
            res.append(i)
        i += 1
    return res if res else [-1]


flax_boolify = lambda func: compose(vectorised(lambda x: 1 if x else 0), func)


def flax_print(x):
    print(
        str(x)
        .replace(", ", " ")
        .replace("oo", "∞")
        .replace(" + ", "+")
        .replace(" - ", "-")
    )
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
        grades.append(find_all_indices(a, x))
    return flatten(grades)


def grade_up(x):
    x = iterable(x, make_digits=True)

    grades = []
    for a in list(sorted(x)):
        grades.append(find_all_indices(a, x))
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


def index_into(x, y):
    x = iterable(x, make_digits=True)
    if isinstance(y, int):
        return x[y % len(x)]
    return [index_into(x, floor(y)), index_into(x, ceil(y))]


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
            return [*range(x)]
        if make_digits:
            return [-int(i) if x < 0 else int(i) for i in str(x)[1 if x < 0 else 0 :]]
        return [x]
    return x


def join(x, y):
    res = [y] * (len(x) * 2 - 1)
    res[0::2] = iterable(x)
    return x


def join_newlines(x):
    return join(x, 10)


def join_spaces(x):
    return join(x, 32)


def mold(x, y):
    for i in range(len(y)):
        if isinstance(y[i], list):
            mold(x, y[i])
        else:
            item = x.pop(0)
            y[i] = item
            x.append(item)
    return y


def prefixes(x):
    res = []
    for i in range(len(x)):
        res.append(x[: i + 1])
    return res


def random(x):
    x = iterable(x)
    return R.choice(x)


def reshape(shape, L):
    if not isinstance(L, list):
        return L

    if not isinstance(L, deque):
        L = deque(L)

    def nxt(dq):
        x = dq.popleft()
        dq.append(x)
        return x

    if len(shape) == 1:
        return [nxt(L) for _ in range(shape[0])]
    else:
        return [reshape(shape[1:], L) for _ in range(shape[0])]


def reverse_every_other(x):
    x = iterable(x)

    for i in range(len(x)):
        if i % 2 == 0:
            x[i] = iterable(x[i])[::-1]
    return x


def split(x, c):
    if not isinstance(x, list):
        return x

    res = []
    tmp = []

    for i in range(len(x)):
        if not (i % c) and tmp:
            res.append(tmp)
            tmp = []
        tmp.append(x[i])

    if tmp:
        res.append(tmp)
    return res


def split_at_occurences(x, y):
    res = []
    tmp = []

    for e in x:
        if e == y:
            res.append(tmp)
            tmp = []
        else:
            tmp.append(e)

    if tmp:
        res.append(tmp)

    return res


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


rationalised = lambda func: compose(
    vectorised(lambda x: sympy.nsimplify(x, rational=True)), func
)


def to_bin(x):
    return [-i if x < 0 else i for i in map(int, bin(x)[3 if x < 0 else 2 :])]


def to_digits(x):
    return [-int(i) if x < 0 else int(i) for i in str(x)[1 if x < 0 else 0 :]]


def truthy_indices(x):
    x = iterable(x, make_digits=True)
    if not isinstance(x, list):
        return []

    i = 0
    indices = []
    while i < len(x):
        if x[i]:
            indices.append(i)
        i += 1
    return indices


def vectorise(fn, x):
    if depth(x) != 0:
        return [vectorise(fn, a) for a in x]
    else:
        return fn(x)


vectorised = lambda func: lambda x: vectorise(func, x)
vectorised_dyadic = lambda func: lambda x, y: dyadic_vectorise(func, x, y)


zip = lambda *x: [[*x] for x in it.zip_longest(*x, fillvalue=0)]

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
    "₈": attrdict(arity=0, call=lambda: ord(sys.stdin.read(1))),
    "₉": attrdict(arity=0, call=lambda: [ord(c) for c in input()]),
    # Single byte monads
    "A": attrdict(arity=1, call=vectorised(lambda a: abs(a))),
    "Ă": attrdict(arity=1, call=contains_false),
    "B": attrdict(arity=1, call=vectorised(to_bin)),
    "Ḃ": attrdict(arity=1, call=from_bin),
    "Ḅ": attrdict(arity=1, call=vectorised(lambda a: 2 ** a)),
    "Ƀ": attrdict(arity=1, call=vectorised(lambda a: a % 2)),
    "C": attrdict(arity=1, call=vectorised(lambda a: 1 - a)),
    "Ċ": attrdict(arity=1, call=vectorised(lambda a: a * 3)),
    "Ç": attrdict(arity=1, call=lambda x: split(x, 2)),
    "D": attrdict(arity=1, call=vectorised(to_digits)),
    "Ḋ": attrdict(arity=1, call=from_digits),
    "Ḍ": attrdict(arity=1, call=vectorised(divisors)),
    "Ð": attrdict(arity=1, call=vectorised(lambda a: a * 2)),
    "Ď": attrdict(arity=1, call=depth),
    "E": attrdict(arity=1, call=vectorised(lambda a: [*range(a)])),
    "F": attrdict(arity=1, call=flatten),
    "G": attrdict(arity=1, call=lambda x: group_equal(iterable(x, make_digits=True))),
    "H": attrdict(arity=1, call=vectorised(lambda a: a / 2)),
    "Ḣ": attrdict(arity=1, call=lambda x: iterable(x, make_digits=True)[1:]),
    "Ḥ": attrdict(arity=1, call=lambda x: iterable(x, make_digits=True)[0]),
    "I": attrdict(arity=1, call=vectorised(lambda x: [*range(x)])),
    "J": attrdict(arity=1, call=join_spaces),
    "Ĵ": attrdict(arity=1, call=join_newlines),
    "K": attrdict(arity=1, call=lambda x: [*scanl1(op.add, iterable(x))]),
    "L": attrdict(arity=1, call=len),
    "N": attrdict(arity=1, call=vectorised(lambda a: -a)),
    "P": attrdict(arity=1, call=lambda x: flax_print(x)),
    "Ṗ": attrdict(arity=1, call=lambda x: print(end="".join(chr(c) for c in x))),
    "Ƥ": attrdict(arity=1, call=lambda x: [*it.permutations(x)]),
    "R": attrdict(arity=1, call=lambda x: iterable(x, make_range=True)[::-1]),
    "Ŕ": attrdict(arity=1, call=random),
    "Ř": attrdict(arity=1, call=lambda x: [*range(len(iterable(x, make_digits=True)))]),
    "S": attrdict(arity=1, call=lambda x: [*sorted(x)]),
    "Ṡ": attrdict(arity=1, call=lambda x: [*sorted(x)][::-1]),
    "T": attrdict(arity=1, call=lambda x: zip(*x)),
    "Ṫ": attrdict(arity=1, call=lambda x: iterable(x, make_digits=True)[:-2]),
    "Ṭ": attrdict(arity=1, call=lambda x: iterable(x, make_digits=True)[-1]),
    "U": attrdict(arity=1, call=lambda x: list(set(iterable(x)))),
    "V": attrdict(arity=1, call=lambda x: group(iterable(x, make_digits=True))),
    "W": attrdict(arity=1, call=lambda x: [x]),
    "X": attrdict(arity=1, call=lambda x: split(x, int(len(x) / 2))),
    "Π": attrdict(arity=1, call=lambda x: foldl1(op.mul, flatten(x))),
    "Σ": attrdict(arity=1, call=lambda x: sum(flatten(x))),
    "⊤": attrdict(arity=1, call=truthy_indices),
    "⊥": attrdict(arity=1, call=falsey_indices),
    "!": attrdict(arity=1, call=vectorised(factorial)),
    "~": attrdict(arity=1, call=vectorised(lambda a: ~a)),
    "¬": attrdict(arity=1, call=vectorised(lambda a: 1 if not a else 0)),
    "¹": attrdict(arity=1, call=lambda x: x),
    "²": attrdict(arity=1, call=vectorised(lambda a: a ** 2)),
    "√": attrdict(arity=1, call=vectorised(lambda a: a ** (1 / 2))),
    "≈": attrdict(
        arity=1,
        call=lambda x: foldl1(
            lambda a, b: 1 if a == b else 0, iterable(x, make_digits=True)
        ),
    ),
    "∇": attrdict(arity=1, call=lambda x: min(iterable(x, make_digits=True))),
    "∆": attrdict(arity=1, call=lambda x: max(iterable(x, make_digits=True))),
    "±": attrdict(
        arity=1,
        call=vectorised(lambda a: -1 if a < 0 else (0 if a == 0 else 1)),
    ),
    "Θ": attrdict(arity=1, call=lambda x: iterable(x, make_range=True).insert(0, 0)),
    "⤒": attrdict(arity=1, call=vectorised(lambda a: a + 1)),
    "⤓": attrdict(arity=1, call=vectorised(lambda a: a - 1)),
    "ε": attrdict(arity=1, call=lambda x: sub_lists(iterable(x, make_range=True))),
    "σ": attrdict(arity=1, call=reverse_every_other),
    "⍋": attrdict(arity=1, call=grade_up),
    "⍒": attrdict(arity=1, call=grade_down),
    "⅟": attrdict(arity=1, call=vectorised(lambda a: 1 / a if a else sympy.oo)),
    "⌈": attrdict(arity=1, call=vectorised(lambda a: ceil(a))),
    "⌊": attrdict(arity=1, call=vectorised(lambda a: floor(a))),
    "⊢": attrdict(arity=1, call=prefixes),
    "⊣": attrdict(arity=1, call=suffixes),
    "∀": attrdict(arity=1, call=lambda x: [*map(sum, x)]),
    # Single byte dyads
    "+": attrdict(arity=2, call=vectorised_dyadic(op.add)),
    "-": attrdict(arity=2, call=vectorised_dyadic(op.sub)),
    "×": attrdict(arity=2, call=vectorised_dyadic(op.mul)),
    "÷": attrdict(
        arity=2,
        call=vectorised_dyadic(lambda a, b: a / b if b else (sympy.oo if a else 0)),
    ),
    "%": attrdict(arity=2, call=vectorised_dyadic(op.mod)),
    "*": attrdict(arity=2, call=vectorised_dyadic(op.pow)),
    '"': attrdict(arity=2, call=lambda x, y: [x, y]),
    ",": attrdict(arity=2, call=lambda x, y: join(x, y)),
    "<": attrdict(arity=2, call=flax_boolify(vectorised_dyadic(op.lt))),
    ">": attrdict(arity=2, call=flax_boolify(vectorised_dyadic(op.gt))),
    "=": attrdict(arity=2, call=flax_boolify(vectorised_dyadic(op.eq))),
    "≠": attrdict(arity=2, call=flax_boolify(vectorised_dyadic(op.ne))),
    "≥": attrdict(arity=2, call=flax_boolify(vectorised_dyadic(op.ge))),
    "≤": attrdict(arity=2, call=flax_boolify(vectorised_dyadic(op.le))),
    "≡": attrdict(arity=2, call=flax_boolify(op.eq)),
    "≢": attrdict(arity=2, call=flax_boolify(op.ne)),
    "∧": attrdict(arity=2, call=flax_boolify(vectorised_dyadic(lambda a, b: a and b))),
    "∨": attrdict(arity=2, call=flax_boolify(vectorised_dyadic(lambda a, b: a or b))),
    "&": attrdict(arity=2, call=vectorised_dyadic(op.and_)),
    "|": attrdict(arity=2, call=vectorised_dyadic(op.or_)),
    "^": attrdict(arity=2, call=vectorised_dyadic(op.xor)),
    "∊": attrdict(arity=2, call=lambda x, y: x in y),
    "f": attrdict(
        arity=2,
        call=lambda x, y: [a for a in iterable(x, make_digits=True) if a not in y],
    ),
    "ḟ": attrdict(
        arity=2, call=lambda x, y: [a for a in iterable(x, make_digits=True) if a in y]
    ),
    "⊂": attrdict(
        arity=2,
        call=lambda x, y: find_all_indices(iterable(x, make_digits=True), y)[0] + 1,
    ),
    "⊆": attrdict(
        arity=2,
        call=lambda x, y: vectorise(lambda a: a + 1, find_all_indices(x, y)),
    ),
    "⊏": attrdict(
        arity=2, call=lambda x, y: [x[i] for i in range(len(x)) if i % y == 0]
    ),
    "·": attrdict(arity=2, call=compose(list, it.product)),
    "r": attrdict(
        arity=2,
        call=vectorised_dyadic(lambda a, b: [*range(a, b + 1)]),
    ),
    "s": attrdict(arity=2, call=split),
    "ṡ": attrdict(arity=2, call=compose(list, mit.sliding_window)),
    "\\": attrdict(arity=2, call=lambda x, y: [iterable(x) for _ in range(y)]),
    "i": attrdict(arity=2, call=index_into),
    "o": attrdict(arity=2, call=split_at_occurences),
    "a": attrdict(arity=2, call=lambda x, y: iterable(x) + iterable(y)),
    "p": attrdict(arity=2, call=lambda x, y: iterable(y) + iterable(x)),
    "c": attrdict(arity=2, call=lambda x, y: iterable(x, make_digits=True).count(y)),
    "m": attrdict(arity=2, call=lambda x, y: mold(iterable(x), iterable(y))),
    "h": attrdict(arity=2, call=lambda x, y: iterable(x, make_digits=True)[:y]),
    "t": attrdict(arity=2, call=lambda x, y: iterable(x, make_digits=True)[y - 1 :]),
    "z": attrdict(arity=2, call=zip),
    "u": attrdict(arity=2, call=lambda x, y: [y.find(v) + 1 for v in x]),
    "#": attrdict(arity=2, call=reshape),
    "ḍ": attrdict(
        arity=2,
        call=vectorised_dyadic(lambda a, b: 1 if a % b == 0 else 0),
    ),
    # Niladic diagraphs
    "Øp": attrdict(arity=0, call=lambda: sympy.pi),
    "Øe": attrdict(arity=0, call=lambda: sympy.E),
    "ØP": attrdict(arity=0, call=lambda: 1.618033988749895),
    "Ø∞": attrdict(arity=0, call=lambda: sympy.oo),
    "ØA": attrdict(arity=0, call=lambda: 26),
    "Ø₁": attrdict(arity=0, call=lambda: 128),
    "Ø₂": attrdict(arity=0, call=lambda: 256),
    "Ø₀": attrdict(arity=0, call=lambda: 1000),
    # Monadic diagraphs
    "ŒD": attrdict(arity=1, call=diagonals),
    "ŒS": attrdict(arity=1, call=vectorised(sympy.sin)),
    "ŒC": attrdict(arity=1, call=vectorised(sympy.cos)),
    "ŒT": attrdict(arity=1, call=vectorised(sympy.tan)),
    "ŒṠ": attrdict(arity=1, call=vectorised(sympy.asin)),
    "ŒĊ": attrdict(arity=1, call=vectorised(sympy.acos)),
    "ŒṪ": attrdict(arity=1, call=vectorised(sympy.atan)),
    "Œc": attrdict(arity=1, call=lambda x: vectorise(lambda a: 1 / sympy.sin(a), x)),
    "Œs": attrdict(arity=1, call=lambda x: vectorise(lambda a: 1 / sympy.cos(a), x)),
    "Œt": attrdict(arity=1, call=lambda x: vectorise(lambda a: 1 / sympy.tan(a), x)),
    "Œn": attrdict(arity=1, call=vectorised(sympy.sinh)),
    "Œo": attrdict(arity=1, call=vectorised(sympy.cosh)),
    "Œh": attrdict(arity=1, call=vectorised(sympy.tanh)),
    "Œi": attrdict(arity=1, call=indices_multidimensional),
    # Dyadic diagraphs
    "œl": attrdict(arity=2, call=vectorised_dyadic(lambda a, b: a << b)),
    "œr": attrdict(arity=2, call=vectorised_dyadic(lambda a, b: a >> b)),
    "œ*": attrdict(arity=2, call=lambda x, y: [*it.product(x, repeat=y)]),
    "œ·": attrdict(
        arity=2, call=lambda x, y: sum(x[i][0] * y[i] for i in range(len(y)))
    ),
    "œṪ": attrdict(arity=2, call=vectorised_dyadic(sympy.atan2)),
}

for k in atoms:
    # This is for better error messages
    atoms[k].glyph = k
    # Rationalise all outputs (aka sympy.nsimplify)
    atoms[k].call = rationalised(atoms[k].call)

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

    try:
        while chain:
            if DEBUG:
                print(f"DEBUG: λ:{accumulator}, chain: {chain}")
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


def leading_nilad(chain):
    return chain and arities(chain) + [1] < [0, 2] * len(chain)


def monadic_chain(chain, x):
    init = False

    accumulator = x

    try:
        while 1:
            if DEBUG:
                print(f"DEBUG: λ:{accumulator}, chain: {chain}")
            if init:
                for link in chain:
                    if link.arity < 0:
                        link.arity = 1

                if leading_nilad(chain):
                    accumulator = chain[0].call()
                    chain = chain[1:]

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


def variadic_chain(chain, *args):
    args = [*filter(None.__ne__, args)]
    if len(args) == 0:
        return niladic_chain(chain)
    elif len(args) == 1:
        return monadic_chain(chain, *args)
    else:
        return dyadic_chain(chain, *args)


def variadic_link(link, *args, reverself=False):
    if link.arity < 0:
        args = [*filter(None.__ne__, args)]
        link.arity = len(args)

    if link.arity == 0:
        return link.call()
    elif link.arity == 1:
        return link.call(args[0])
    elif link.arity == 2:
        if reverself:
            if len(args) == 1:
                return link.call(args[0], args[0])
            else:
                return link.call(args[1], args[0])
        else:
            return link.call(args[0], args[1])


# ========= Quicks ==========
def qreduce(links, outer_links, i, arity=1):
    ret = [attrdict(arity=arity)]
    if len(links) == 1:
        ret[0].call = lambda x, y=None: foldl1(links[0].call, x)
    else:
        ret[0].call = lambda x, y=None: [
            foldl1(links[0].call, t)
            for t in mit.sliding_window(iterable(x), links[1].call())
        ]
    return ret


def qreducer(links, outer_links, i, arity=1):
    ret = [attrdict(arity=arity)]
    if len(links) == 1:
        ret[0].call = lambda x, y=None: foldr1(links[0].call, x)
    else:
        ret[0].call = lambda x, y=None: [
            foldr1(links[0].call, t)
            for t in mit.sliding_window(iterable(x), links[1].call())
        ]
    return ret


quicks = {
    "©": attrdict(
        condition=lambda links: links,
        qlink=lambda links, outer_links, i: [
            attrdict(
                arity=links[0].arity,
                call=lambda x=None, y=None: copy_to(
                    atoms["®"], variadic_link(links[0], x, y)
                ),
            )
        ],
    ),
    "ß": attrdict(
        condition=lambda links: True,
        qlink=lambda links, outer_links, i: [create_chain(outer_links[i])],
    ),
    "¨": attrdict(
        condition=lambda links: links and links[0].arity == 1,
        qlink=lambda links, outer_links, i: [
            attrdict(
                arity=links[0].arity,
                call=lambda x, y=None: [
                    variadic_link(links[0], a, reverself=True) for a in x
                ],
            )
        ],
    ),
    "₀": attrdict(
        condition=lambda links: True,
        qlink=lambda links, outer_links, i: [create_chain(outer_links[i], 0)],
    ),
    "₁": attrdict(
        condition=lambda links: True,
        qlink=lambda links, outer_links, i: [create_chain(outer_links[i], 1)],
    ),
    "₂": attrdict(
        condition=lambda links: True,
        qlink=lambda links, outer_links, i: [create_chain(outer_links[i], 2)],
    ),
    "/": attrdict(condition=lambda links: links and links[0].arity, qlink=qreduce),
    "⌿": attrdict(condition=lambda links: links and links[0].arity, qlink=qreducer),
}

# == Train Separators ==

train_separators = {
    "ø": (0, True),
    "µ": (1, True),
    "г": (1, True),
    "ð": (2, True),
    "ɓ": (2, False),
}
