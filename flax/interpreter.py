import sys
import math as M
import random as R

import flax.utils as U
from flax.utils import vectorise, flatten, iterable

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
    'Π': atom(1, lambda x: U.reduce(lambda a, b: a * b, flatten(x))),
    'Σ': atom(1, lambda x: sum(flatten(x))),
    '⍳': atom(1, lambda x: vectorise(lambda a: [*range(1, a + 1)], x)),
    '⊤': atom(1, truthy_indices),
    '⊥': atom(1, falsey_indices),
    'R': atom(1, lambda x: iterable(x, True)[::-1]),
    'W': atom(1, lambda x: [x]),
    'Ŕ': atom(1, random),
    'T': atom(1, lambda x: U.zip(*x)),
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
    'Ḣ': atom(1, lambda x: iterable(x, True)[1:]),
    'Ṫ': atom(1, lambda x: iterable(x, True)[:-2]),
    'Ḥ': atom(1, lambda x: iterable(x, True)[0]),
    'Ṭ': atom(1, lambda x: iterable(x, True)[-1]),
    '±': atom(1, lambda x: vectorise(lambda a: -1 if a < 0 else (0 if a == 0 else 1), x)),
    'Θ': atom(1, lambda x: iterable(x, True).insert(0, 0)),
    'U': atom(1, lambda x: list(set(iterable(x)))),
    '⤒': atom(1, lambda x: vectorise(lambda a: a + 1, x)),
    '⤓': atom(1, lambda x: vectorise(lambda a: a - 1, x)),
}
