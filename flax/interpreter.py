import sys
import math as M
import random as R
import itertools

from flax.utils import (
    dyadic_vectorise,
    reshape,
    vectorise,
    flatten,
    iterable,
    reduce,
    pp,
    zip,
    depth
)

class atom:
    def __init__(self, arity, call):
        self.arity = arity
        self.call = call

def to_bin(x):
    return [-i if x < 0 else i
        for i in map(int, bin(x)[
            3 if x < 0 else 2:])]

def to_digits(x):
    return [-int(i) if x < 0 else int(i)
        for i in str(x)[
            1 if x < 0 else 0:]]

def truthy_indices(x):
    if not isinstance(x, list):
        return []

    i = 0
    indices = []
    while i < len(x):
        if x[i]:
            indices.append(i)
        i += 1
    return indices

def falsey_indices(x):
    if not isinstance(x, list):
        return []

    i = 0
    indices = []
    while i < len(x):
        if not x[i]:
            indices.append(i)
        i += 1
    return indices

def random(x):
    x = iterable(x)
    return R.choice(x)

def from_bin(x):
    x = iterable(x)
    sign = -1 if sum(x) < 0 else 1
    num = 0
    i = 0
    for b in x[::-1]:
        num += abs(b) * 2 ** i
        i += 1
    return num * sign

def contains_false(l):
    if not isinstance(l, list):
        return 1 if l else 0

    if len(l) == 0:
        return 1   

    return 1 if 1 in [contains_false(x) for x in l] else 0

def from_digits(x):
    x = iterable(x)
    sign = -1 if sum(x) < 0 else 1
    num = 0
    i = 0
    for b in x[::-1]:
        num += abs(b) * 10 ** i
        i += 1
    return num * sign

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

def sub_lists(l):
    lists = [[]]
    for i in range(len(l) + 1):
        for j in range(i):
            lists.append(l[j:i])
    return lists

def reverse_every_other(x):
    x = iterable(x)

    for i in range(len(x)):
        if i % 2 == 0:
            x[i] = iterable(x[i])[::-1]
    return x

def grade_up(x):
    x = iterable(x, make_digits=True)

    grades = []
    for a in list(sorted(x)):
        grades.append(find_all_indices(a, x))
    return flatten(grades)

def grade_down(x):
    x = iterable(x, make_digits=True)

    grades = []
    for a in list(sorted(x))[::-1]:
        grades.append(find_all_indices(a, x))
    return flatten(grades)

def divisors(x):
    res = []

    i = 1
    while i <= x:
        if x % i == 0:
            res.append(i)
        i += 1

def join_spaces(x):
    return laminate(x, 32)

def join_newlines(x):
    return laminate(x, 10)

def prefixes(x):
    res = []
    for i in range(len(x)):
        res.append(x[:i + 1])
    return res

def suffixes(x):
    res = []
    for i in range(len(x)):
        res.append(x[i:])
    return res

def laminate(x, y):
    res = [y] * (len(x) * 2 - 1)
    res[0::2] = iterable(x)
    return x

def find_all_indices(x, y):
    res = []
    i = 0
    while i < len(y):
        if y[i] == x:
            res.append(i)
        i += 1
    return res

def index(x, y):
    if not isinstance(x, list):
        return x

    if isinstance(y, int):
        return x[(y - 1) % len(x)]
    return [index(x, M.floor(y)), index(x, M.ceil(y))]

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

def mold(x, y):
    for i in range(len(y)):
        if isinstance(y[i], list):
            mold(x, y[i])
        else:
            item = x.pop(0)
            y[i] = item
            x.append(item)
    return y

def diagonals(x):
    d = [[] for _ in range(len(x) + len(x[0]) - 1)]
    min_d = -len(x) + 1

    for i in range(len(x[0])):
        for j in range(len(x)):
            d[i - j - min_d].append(x[j][i])
    return d

commands = {
    # Single byte nilads
    'Ŧ': atom(0, lambda: 10),
    '³': atom(0, lambda: sys.argv[1] if len(sys.argv) > 1 else 16),
    '⁴': atom(0, lambda: sys.argv[2] if len(sys.argv) > 2 else 32),
    '⁵': atom(0, lambda: sys.argv[2] if len(sys.argv) > 3 else 64),
    '⁰': atom(0, lambda: 100),
    'ƀ': atom(0, lambda: [0, 1]),
    '®': atom(0, lambda: 0),
    'я': atom(0, lambda: sys.stdin.read(1)),
    'д': atom(0, lambda: input()),

    # Single byte monads
    '!': atom(1, lambda x: vectorise(M.factorial, x)),
    '¬': atom(1, lambda x: vectorise(lambda a: 1 if not a else 0, x)),
    '~': atom(1, lambda x: vectorise(lambda a: ~a, x)),
    'B': atom(1, lambda x: vectorise(to_bin, x)),
    'D': atom(1, lambda x: vectorise(to_digits, x)),
    'C': atom(1, lambda x: vectorise(lambda a: 1 - a, x)),
    'F': atom(1, flatten),
    'H': atom(1, lambda x: vectorise(lambda a: a / 2, x)),
    'L': atom(1, len),
    'N': atom(1, lambda x: vectorise(lambda a: -a, x)),
    'Ř': atom(1, lambda x: [*range(len(x))]),
    'Π': atom(1, lambda x: reduce(lambda a, b: a * b, flatten(x))),
    'Σ': atom(1, lambda x: sum(flatten(x))),
    '⍳': atom(1, lambda x: vectorise(lambda a: [*range(1, a + 1)], x)),
    '⊤': atom(1, truthy_indices),
    '⊥': atom(1, falsey_indices),
    'R': atom(1, lambda x: iterable(x, make_range=True)[::-1]),
    'W': atom(1, lambda x: [x]),
    'Ŕ': atom(1, random),
    'T': atom(1, lambda x: zip(*x)),
    '¹': atom(1, lambda x: x),
    '²': atom(1, lambda x: vectorise(lambda a: a ** 2, x)),
    '√': atom(1, lambda x: vectorise(lambda a: a ** (1 / 2), x)),
    'Ḃ': atom(1, from_bin),
    'Ă': atom(1, contains_false),
    'Ḋ': atom(1, from_digits),
    'Ð': atom(1, lambda x: vectorise(lambda a: a * 2, x)),
    '₃': atom(1, lambda x: vectorise(lambda a: a * 3, x)),
    'E': atom(1, lambda x: vectorise(lambda a: [*range(a)], x)),
    '∇': atom(1, lambda x: min(iterable(x))),
    '∆': atom(1, lambda x: max(iterable(x))),
    'S': atom(1, lambda x: [*sorted(x)]),
    'Ṡ': atom(1, lambda x: [*sorted(x)][::-1]),
    'ᵇ': atom(1, lambda x: vectorise(lambda a: a % 2, x)),
    'Ḣ': atom(1, lambda x: iterable(x, make_digits=True)[1:]),
    'Ṫ': atom(1, lambda x: iterable(x, make_digits=True)[:-2]),
    'Ḥ': atom(1, lambda x: iterable(x, make_digits=True)[0]),
    'Ṭ': atom(1, lambda x: iterable(x, make_digits=True)[-1]),
    '±': atom(1, lambda x: vectorise(lambda a: -1 if a < 0 else (0 if a == 0 else 1), x)),
    'Θ': atom(1, lambda x: iterable(x, make_range=True).insert(0, 0)),
    'U': atom(1, lambda x: list(set(iterable(x)))),
    '⤒': atom(1, lambda x: vectorise(lambda a: a + 1, x)),
    '⤓': atom(1, lambda x: vectorise(lambda a: a - 1, x)),
    'P': atom(1, lambda x: pp(x)),
    'Ċ': atom(1, lambda x: print(end=''.join(chr(c) for c in x)) or x),
    'Ç': atom(1, lambda x: split(x, 2)),
    'X': atom(1, lambda x: split(x, int(len(x) / 2))),
    'Ƥ': atom(1, lambda x: [*itertools.permutations(x)]),
    'ε': atom(1, lambda x: sub_lists(iterable(x, make_range=True))),
    'σ': atom(1, reverse_every_other),
    'Ḅ': atom(1, lambda x: vectorise(lambda a: 2 ** a, x)),
    'Ď': atom(1, depth),
    '⍋': atom(1, grade_up),
    '⍒': atom(1, grade_down),
    '⅟': atom(1, lambda x: vectorise(lambda a: 1 / a, x)),
    '⌈': atom(1, lambda x: vectorise(lambda a: M.ceil(a), x)),
    '⌊': atom(1, lambda x: vectorise(lambda a: M.floor(a), x)),
    'A': atom(1, lambda x: vectorise(lambda a: abs(a), x)),
    'Ḍ': atom(1, lambda x: vectorise(divisors, x)),
    'J': atom(1, join_spaces),
    'Ĵ': atom(1, join_newlines),
    '⊢': atom(1, prefixes),
    '⊣': atom(1, suffixes),
    '∀': atom(1, lambda x: [sum(r) for r in iterable(x)]),

    # Single byte dyads
    '+': atom(2, lambda x, y: dyadic_vectorise(lambda a, b: a + b, x, y)),
    '-': atom(2, lambda x, y: dyadic_vectorise(lambda a, b: a - b, x, y)),
    '×': atom(2, lambda x, y: dyadic_vectorise(lambda a, b: a * b, x, y)),
    '÷': atom(2, lambda x, y: dyadic_vectorise(lambda a, b: a / b, x, y)),
    '%': atom(2, lambda x, y: dyadic_vectorise(lambda a, b: a % b, x, y)),
    '*': atom(2, lambda x, y: dyadic_vectorise(lambda a, b: a ** b, x, y)),
    '"': atom(2, lambda x, y: [x, y]),
    ',': atom(2, lambda x, y: laminate(x, y)),
    '<': atom(2, lambda x, y: dyadic_vectorise(lambda a, b: 1 if a < b else 0, x, y)),
    '>': atom(2, lambda x, y: dyadic_vectorise(lambda a, b: 1 if a > b else 0, x, y)),
    '=': atom(2, lambda x, y: dyadic_vectorise(lambda a, b: 1 if a == b else 0, x, y)),
    '≠': atom(2, lambda x, y: dyadic_vectorise(lambda a, b: 1 if a != b else 0, x, y)),
    '≥': atom(2, lambda x, y: dyadic_vectorise(lambda a, b: 1 if a >= b else 0, x, y)),
    '≤': atom(2, lambda x, y: dyadic_vectorise(lambda a, b: 1 if a <= b else 0, x, y)),
    '≡': atom(2, lambda x, y: 1 if x == y else 0),
    '≢': atom(2, lambda x, y: 1 if x != y else 0),
    '∧': atom(2, lambda x, y: dyadic_vectorise(lambda a, b: 1 if a and b else 0, x, y)),
    '∨': atom(2, lambda x, y: dyadic_vectorise(lambda a, b: 1 if a or b else 0, x, y)),
    '&': atom(2, lambda x, y: dyadic_vectorise(lambda a, b: a & b, x, y)),
    '|': atom(2, lambda x, y: dyadic_vectorise(lambda a, b: a | b, x, y)),
    '^': atom(2, lambda x, y: dyadic_vectorise(lambda a, b: a ^ b, x, y)),
    '∊': atom(2, lambda x, y: x in y),
    'f': atom(2, lambda x, y: [a for a in x if a not in y]),
    'ḟ': atom(2, lambda x, y: [a for a in x if a in y]),
    '⊂': atom(2, lambda x, y: x.find(y) + 1),
    '⊆': atom(2, lambda x, y: vectorise(lambda a: a + 1, find_all_indices(x, y))),
    '⊏': atom(2, lambda x, y: [x[i] for i in range(len(x)) if i % y == 0]),
    '·': atom(2, lambda x, y: [*itertools.product(x, y)]),
    'r': atom(2, lambda x, y: dyadic_vectorise(lambda a, b: [*range(a, b + 1)], x, y)),
    's': atom(2, split),
    '\\': atom(2, lambda x, y: [iterable(x) for _ in range(y)]),
    'i': atom(2, index),
    'o': atom(2, split_at_occurences),
    'a': atom(2, lambda x, y: iterable(x) + iterable(y)),
    'p': atom(2, lambda x, y: iterable(y) + iterable(x)),
    'c': atom(2, lambda x, y: iterable(x, make_digits=True).count(y)),
    'm': atom(2, lambda x, y: mold(iterable(x), iterable(y))),
    'h': atom(2, lambda x, y: iterable(x, make_digits=True)[:y]),
    't': atom(2, lambda x, y: iterable(x, make_digits=True)[y - 1:]),
    'z': atom(2, zip),
    'u': atom(2, lambda x, y: [y.find(v) + 1 for v in x]),
    '#': atom(2, reshape),
    'ḍ': atom(2, lambda x, y: dyadic_vectorise(lambda a, b: 1 if a % b == 0 else 0, x, y)),

    # Niladic diagraphs
    'Øp': atom(0, lambda: M.pi),
    'Øe': atom(0, lambda: M.e),
    'ØP': atom(0, lambda: 1.618033988749895),
    'Ø∞': atom(0, lambda: float('inf')),
    'ØA': atom(0, lambda: 26),
    'Ø₁': atom(0, lambda: 128),
    'Ø₂': atom(0, lambda: 256),
    'Ø₀': atom(0, lambda: 1000),

    # Monadic diagraphs
    'ŒD': atom(1, diagonals),
    'ŒS': atom(1, lambda x: vectorise(M.sin, x)),
    'ŒC': atom(1, lambda x: vectorise(M.cos, x)),
    'ŒT': atom(1, lambda x: vectorise(M.tan, x)),
    'ŒṠ': atom(1, lambda x: vectorise(M.asin, x)),
    'ŒĊ': atom(1, lambda x: vectorise(M.acos, x)),
    'ŒṪ': atom(1, lambda x: vectorise(M.atan, x)),
    'Œc': atom(1, lambda x: vectorise(lambda a: 1 / M.sin(a), x)),
    'Œs': atom(1, lambda x: vectorise(lambda a: 1 / M.cos(a), x)),
    'Œt': atom(1, lambda x: vectorise(lambda a: 1 / M.tan(a), x)),
    'Œn': atom(1, lambda x: vectorise(M.sinh, x)),
    'Œo': atom(1, lambda x: vectorise(M.cosh, x)),
    'Œh': atom(1, lambda x: vectorise(M.tanh, x)),

    # Dyadic diagraphs
    'œl': atom(2, lambda x, y: dyadic_vectorise(lambda a, b: a << b, x, y)),
    'œr': atom(2, lambda x, y: dyadic_vectorise(lambda a, b: a >> b, x, y)),
    'œ*': atom(2, lambda x, y: [*itertools.product(x, repeat=y)])
    'œ·': atom(2, lambda x, y: sum(x[i][0] * y[i] for i in range(len(y)))),
}
