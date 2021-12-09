import sys
import math as M

from flax.utils import *

class state:
    reg = 0
    accumulator = 0

global_state = state()

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

commands = {
    # Single byte nilads
    'Ŧ': atom(0, lambda: 10),
    '³': atom(0, lambda: sys.argv[1] if len(sys.argv) > 1 else 16),
    '⁴': atom(0, lambda: sys.argv[2] if len(sys.argv) > 2 else 32),
    '⁵': atom(0, lambda: sys.argv[2] if len(sys.argv) > 3 else 64),
    '⁰': atom(0, lambda: 100),
    'ƀ': atom(0, lambda: [0, 1]),
    '®': atom(0, lambda: global_state.reg),
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
    '⍳': atom(1, lambda x: [*range(1, x + 1)]),
    '⊤': atom(1, truthy_indices),
    '⊥': atom(1, falsey_indices),
    'R': atom(1, lambda x: x[::-1]),
}
