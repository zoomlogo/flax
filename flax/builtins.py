# builtins: holds the builtins and some constants for the lexer
import itertools
import functools
import json
import sys
import math
import more_itertools as mit
import operator as Op
import random as Random

from flax.common import *
from flax.funcs import *
from flax.chains import *

__all__ = [
    "COMMENT",
    "COMPLEX_DELIMETER",
    "DECIMAL_POINT",
    "DIAGRAPHS",
    "LIST_DELIMETER_L",
    "LIST_DELIMETER_R",
    "NEGATIVE_SIGN",
    "NEWLINE",
    "STRING_DELIMETER",
    "STRING_NEXT_1",
    "STRING_NEXT_2",
    "ZERO",
    "DIGITS",
    "atoms",
    "transpiled_atoms",
    "quicks",
    "train_separators",
]

# constants
COMMENT = "·"
COMPLEX_DELIMETER = "i"
DECIMAL_POINT = "."
DIAGRAPHS = "ØÆŒæœΔ"
LIST_DELIMETER_L = "("
LIST_DELIMETER_R = ")"
NEGATIVE_SIGN = "¯"
NEWLINE = "\n"
STRING_DELIMETER = '"'
ZERO = "0"
DIGITS = ZERO + "123456789" + DECIMAL_POINT + COMPLEX_DELIMETER + NEGATIVE_SIGN
STRING_NEXT_1 = "_"
STRING_NEXT_2 = ":"

# dicts
atoms = {
    "⁰": attrdict(arity=0, call=lambda: 10),
    "¹": attrdict(arity=0, call=lambda: 16),
    "²": attrdict(arity=0, call=lambda: 26),
    "³": attrdict(arity=0, call=lambda: 32),
    "⁴": attrdict(arity=0, call=lambda: 64),
    "⁵": attrdict(arity=0, call=lambda: 100),
    "⁶": attrdict(arity=0, call=lambda: -2),
    "⁷": attrdict(arity=0, call=lambda: 256),
    "⁸": attrdict(arity=0, call=lambda: 0),
    "⁹": attrdict(arity=0, call=lambda: 0),
    "∃": attrdict(arity=0, call=lambda: 0),

    "A": attrdict(arity=1, dx=0, call=abs),
    "Ȧ": attrdict(arity=1,       call=lambda x: int(any(iterable(x)))),
    "B": attrdict(arity=1, dx=0, call=binary),
    "Ḃ": attrdict(arity=1, dx=1, call=binary_i),
    "C": attrdict(arity=1, dx=0, call=lambda x: 1 - x),
    "Ċ": attrdict(arity=1,       call=Random.choice),
    "D": attrdict(arity=1, dx=0, call=digits),
    "Ḋ": attrdict(arity=1, dx=1, call=digits_i),
    "E": attrdict(arity=1,       call=boolify(mit.all_equal)),
    "Ė": attrdict(arity=1,       call=lambda x: iterable(x)[0] if mit.all_equal(iterable(x)) else []),
    "F": attrdict(arity=1,       call=flatten),
    "Ḟ": attrdict(arity=1,       call=lambda x: [i for i, e in enumerate(iterable(x)) if not e]),
    "G": attrdict(arity=1,       call=group_indicies),
    "Ġ": attrdict(arity=1,       call=group_equal),
    "H": attrdict(arity=1, dx=0, call=lambda x: x / 2),
    "Ḣ": attrdict(arity=1, dx=0, call=lambda x: 2 * x),
    "J": attrdict(arity=1,       call=lambda x: iota(len(iterable(x)))),
    "L": attrdict(arity=1,       call=lambda x: len(iterable(x))),
    "M": attrdict(arity=1,       call=lambda x: x ** 2),
    "Ṁ": attrdict(arity=1,       call=lambda x: [i for i, e in enumerate(iterable(x)) if e == max(iterable(x))]),
    "N": attrdict(arity=1, dx=0, call=Op.neg),
    "O": attrdict(arity=1,       call=lambda x: x),
    "Ȯ": attrdict(arity=1, dx=0, call=divisors),
    "P": attrdict(arity=1,       call=permutations),
    "Ṗ": attrdict(arity=1,       call=shuffle),
    "Q": attrdict(arity=1, dx=2, call=lambda x: [e if i % 2 == 0 else e[::-1] for i, e in enumerate(x)]),
    "R": attrdict(arity=1,       call=lambda x: iterable(x, digits=True)[::-1]),
    "S": attrdict(arity=1,       call=sublists),
    "Ṫ": attrdict(arity=1,       call=lambda x: [i for i, e in enumerate(iterable(x)) if e]),
    "U": attrdict(arity=1,       call=lambda x: list(mit.unique_everseen(x))),
    "V": attrdict(arity=1, dx=0, call=lambda x: int(mp.isprime(x))),
    "W": attrdict(arity=1,       call=where),
    "X": attrdict(arity=1,       call=lambda x: split(len(iterable(x, digits=True)) // 2, iterable(x, digits=True))),
    "Ẋ": attrdict(arity=1,       call=lambda x: split(2, x)),
}

transpiled_atoms = {
    "I": [],
    "K": [],
    "Ŀ": [],
    "Ṙ": [],
    "Ṡ": [],
}

quicks = {}

train_separators = {
    "ø": (0, True),
    "µ": (1, True),
    "[": (1, True),
    "]": (1, True),
    "ð": (2, True),
    "ɓ": (2, False),
}
