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
COMMENT = "‟"
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
    "⍬": attrdict(arity=0, call=lambda: []),
    "A": attrdict(arity=1, dx=0, call=abs),
    "Ȧ": attrdict(arity=1, call=lambda x: int(any(iterable(x)))),
    "B": attrdict(arity=1, dx=0, call=binary),
    "Ḃ": attrdict(arity=1, dx=1, call=binary_i),
    "C": attrdict(arity=1, dx=0, call=lambda x: 1 - x),
    "Ċ": attrdict(arity=1, call=Random.choice),
    "D": attrdict(arity=1, dx=0, call=digits),
    "Ḋ": attrdict(arity=1, dx=1, call=digits_i),
    "E": attrdict(arity=1, call=boolify(mit.all_equal)),
    "Ė": attrdict(
        arity=1, call=lambda x: iterable(x)[0] if mit.all_equal(iterable(x)) else []
    ),
    "F": attrdict(arity=1, call=flatten),
    "Ḟ": attrdict(
        arity=1, call=lambda x: [i for i, e in enumerate(iterable(x)) if not e]
    ),
    "G": attrdict(arity=1, call=group_indicies),
    "Ġ": attrdict(arity=1, call=group_equal),
    "H": attrdict(arity=1, dx=0, call=lambda x: x / 2),
    "Ḣ": attrdict(arity=1, dx=0, call=lambda x: 2 * x),
    "J": attrdict(arity=1, call=lambda x: iota(len(iterable(x)))),
    "L": attrdict(arity=1, call=lambda x: len(iterable(x))),
    "M": attrdict(arity=1, dx=0, call=lambda x: x**2),
    "Ṁ": attrdict(
        arity=1,
        call=lambda x: [i for i, e in enumerate(iterable(x)) if e == max(iterable(x))],
    ),
    "N": attrdict(arity=1, dx=0, call=Op.neg),
    "Ṅ": attrdict(arity=1, dx=0, call=mp.sign),
    "O": attrdict(arity=1, call=lambda x: x),
    "Ȯ": attrdict(
        arity=1, dx=0, call=lambda x: [i for i in range(1, int(x) + 1) if x % i == 0]
    ),
    "P": attrdict(arity=1, call=permutations),
    "Ṗ": attrdict(arity=1, call=shuffle),
    "Q": attrdict(
        arity=1,
        dx=2,
        call=lambda x: [e if i % 2 == 0 else e[::-1] for i, e in enumerate(x)],
    ),
    "R": attrdict(arity=1, call=lambda x: iterable(x, digits_=True)[::-1]),
    "S": attrdict(arity=1, call=sublists),
    "Ṫ": attrdict(arity=1, call=lambda x: [i for i, e in enumerate(iterable(x)) if e]),
    "U": attrdict(arity=1, call=lambda x: list(mit.unique_everseen(x))),
    "V": attrdict(arity=1, dx=0, call=lambda x: int(mp.isprime(x))),
    "W": attrdict(arity=1, call=where),
    "X": attrdict(
        arity=1,
        call=lambda x: split(
            len(iterable(x, digits_=True)) // 2, iterable(x, digits_=True)
        ),
    ),
    "Ẋ": attrdict(arity=1, call=lambda x: split(2, x)),
    "Y": attrdict(
        arity=1, call=lambda x: [e for i, e in enumerate(iterable(x)) if i % 2 == 0]
    ),
    "Ẏ": attrdict(
        arity=1, call=lambda x: [e for i, e in enumerate(iterable(x)) if i % 2]
    ),
    "Z": attrdict(arity=1, call=transpose),
    "Ż": attrdict(arity=1, call=lambda x: [0] + iterable(x)),
    "!": attrdict(
        arity=1,
        dx=0,
        call=lambda x: -mp.gamma(abs(x) + 1) if x < 0 else mp.gamma(x + 1),
    ),
    "¬": attrdict(arity=1, dx=0, call=boolify(Op.not_)),
    "√": attrdict(arity=1, dx=0, call=mp.sqrt),
    "⊂": attrdict(arity=1, call=lambda x: [x]),
    "⊆": attrdict(
        arity=1, call=lambda x: [x] if x != iterable(x) or len(x) != 1 else x
    ),
    "⊃": attrdict(arity=1, call=lambda x: iterable(x)[0]),  # err
    "⊇": attrdict(arity=1, call=lambda x: iterable(x)[1:]),
    "⊐": attrdict(arity=1, call=lambda x: iterable(x)[-1]),  # err
    "⊒": attrdict(arity=1, call=lambda x: iterable(x)[:-1]),
    "~": attrdict(arity=1, dx=0, call=Op.inv),
    "γ": attrdict(arity=1, call=flax_print),
    "ε": attrdict(arity=1, call=lambda x: list(enumerate(x))),
    "ι": attrdict(arity=1, call=iota),
    "κ": attrdict(arity=1, call=iota1),
    "ξ": attrdict(arity=1, call=lambda x: transpose(x, filler=0)),
    "χ": attrdict(arity=1, call=lambda x: int(all(iterable(x)))),
    "ψ": attrdict(arity=1, call=lambda x: int(iterable(x) > [] and all(flatten(x)))),
    "ϕ": attrdict(arity=1, dx=2, call=flatten),
    "∵": attrdict(
        arity=1,
        call=lambda x: min(iterable(x, digits_=True))
        if iterable(x, digits_=True)
        else 0,
    ),
    "∴": attrdict(
        arity=1,
        call=lambda x: max(iterable(x, digits_=True))
        if iterable(x, digits_=True)
        else 0,
    ),
    "↑": attrdict(arity=1, call=grade_up),
    "↓": attrdict(arity=1, call=grade_down),
    # "∞": attrdict(arity=1, ),
    "¼": attrdict(arity=1, dx=0, call=lambda x: 1 / x),
    "½": attrdict(arity=1, dx=0, call=lambda x: x % 2),
    "⌈": attrdict(arity=1, dx=0, call=mp.ceil),
    "⌊": attrdict(arity=1, dx=0, call=mp.floor),
    "→": attrdict(arity=1, dx=0, call=lambda x: x + 1),
    "←": attrdict(arity=1, dx=0, call=lambda x: x - 1),
    "∂": attrdict(arity=1, call=lambda x: list(sorted(iterable(x)))),
    "{": attrdict(arity=1, call=prefixes),
    "}": attrdict(arity=1, call=suffixes),
    "○": attrdict(arity=1, call=lambda x: list(map(list, mit.powerset(iterable(x))))),
    "↶": attrdict(arity=1, call=lambda x: transpose(x)[::-1]),
    "a": attrdict(arity=2, dw=0, dx=0, call=lambda w, x: abs(w - x)),
    "ȧ": attrdict(arity=2, call=lambda w, x: Random.choice([w, x])),
    "b": attrdict(arity=2, dw=0, dx=0, call=base_i),
    "ḃ": attrdict(arity=2, dw=0, dx=1, call=base),
    "c": attrdict(arity=2, dw=0, dx=0, call=mp.binomial),
    "ċ": attrdict(arity=2, dw=0, call=split),
    "d": attrdict(arity=2, dw=0, dx=0, call=lambda w, x: list(divmod(w, x))),
    "ḋ": attrdict(arity=2, dw=0, dx=0, call=lambda w, x: int(x % w == 0)),
    "e": attrdict(arity=2, dw=0, dx=0, call=lambda w, x: list(range(w, x))),
    # "ė": attrdict(arity=2, ),
    "f": attrdict(
        arity=2, call=lambda w, x: [i for i in iterable(x) if i in iterable(w)]
    ),
    "ḟ": attrdict(
        arity=2, call=lambda w, x: [i for i in iterable(x) if i not in iterable(w)]
    ),
    "g": attrdict(arity=2, dw=0, dx=0, call=math.gcd),
    "ġ": attrdict(arity=2, dw=1, dx=1, call=mit.dotproduct),
    "h": attrdict(arity=2, dw=0, call=lambda w, x: iterable(x)[:w]),
    "ḣ": attrdict(arity=2, dw=0, dx=0, call=order),
    "j": attrdict(arity=2, dx=0, call=index_into),
    "k": attrdict(arity=2, dw=1, call=split_into),
    "l": attrdict(arity=2, dw=0, dx=0, call=math.lcm),
    "ŀ": attrdict(arity=2, dw=0, dx=0, call=lambda w, x: mp.log(x, w)),
    "m": attrdict(arity=2, dw=0, dx=0, call=lambda w, x: min([w, x])),
    "ṁ": attrdict(arity=2, call=mold),
    "n": attrdict(
        arity=2,
        dw=0,
        call=lambda w, x: [e for i, e in enumerate(iterable(x)) if i % w == 0],
    ),
    "o": attrdict(arity=2, dw=0, call=split_at),
    "ȯ": attrdict(arity=2, dw=0, call=lambda w, x: iterable(x, digits_=True).count(w)),
    "ṗ": attrdict(
        arity=2, dw=0, call=lambda w, x: functools.reduce(cartesian_product, [x] * w)
    ),
    "p": attrdict(
        arity=2, dw=0, dx=0, call=lambda w, x: mp.factorial(w) / mp.factorial(w - x)
    ),
    "q": attrdict(arity=2, call=lambda w, x: exit(0)),
    "r": attrdict(arity=2, dw=0, dx=0, call=lambda w, x: list(range(w, x + 1))),
    "s": attrdict(arity=2, call=find_sublist),
    "ṡ": attrdict(
        arity=2,
        call=lambda w, x: [index_into(x, i) for i, e in enumerate(iterable(w)) if e],
    ),
    "t": attrdict(arity=2, dw=0, call=lambda w, x: iterable(x)[w:]),
    # "ṫ": attrdict(arity=2, call=),
    "u": attrdict(
        arity=2, call=lambda w, x: [find(w, i) for i in iterable(x, range_=True)]
    ),
    "v": attrdict(arity=2, dw=1, call=lambda w, x: [[i] + iterable(x) for i in w]),
    "w": attrdict(arity=2, dw=0, call=sliding_window),
    "ẇ": attrdict(
        arity=2,
        dw=0,
        call=lambda w, x: list(map(list, mit.distinct_combinations(x, w))),
    ),
    "x": attrdict(arity=2, dw=0, dx=0, call=lambda w, x: max([w, x])),
    "ẋ": attrdict(arity=2, call=lambda w, x: [w, x]),
    "y": attrdict(arity=2, dw=0, call=join),
    "z": attrdict(
        arity=2, call=lambda w, x: list(map(list, zip(iterable(w), iterable(x))))
    ),
    "ż": attrdict(arity=2, dw=0, call=lambda w, x: transpose(x, filler=w)),
    "+": attrdict(arity=2, dw=0, dx=0, call=Op.add),
    "-": attrdict(arity=2, dw=0, dx=0, call=Op.sub),
    "±": attrdict(arity=2, dw=0, dx=0, call=lambda w, x: [w + x, w - x]),
    "×": attrdict(arity=2, dw=0, dx=0, call=Op.mul),
    "÷": attrdict(arity=2, dw=0, dx=0, call=Op.truediv),
    "|": attrdict(arity=2, dw=0, dx=0, call=lambda w, x: x % w),
    "*": attrdict(arity=2, dw=0, dx=0, call=Op.pow),
    "&": attrdict(arity=2, dw=0, dx=0, call=Op.and_),
    "%": attrdict(arity=2, dw=0, dx=0, call=Op.or_),
    "^": attrdict(arity=2, dw=0, dx=0, call=Op.xor),
    "∧": attrdict(arity=2, dw=0, dx=0, call=lambda w, x: w and x),
    "∨": attrdict(arity=2, dw=0, dx=0, call=lambda w, x: w or x),
    ">": attrdict(arity=2, dw=0, dx=0, call=boolify(Op.gt)),
    "<": attrdict(arity=2, dw=0, dx=0, call=boolify(Op.lt)),
    "=": attrdict(arity=2, dw=0, dx=0, call=boolify(Op.eq)),
    "≠": attrdict(arity=2, dw=0, dx=0, call=boolify(Op.ne)),
    "≤": attrdict(arity=2, dw=0, dx=0, call=boolify(Op.ge)),
    "≥": attrdict(arity=2, dw=0, dx=0, call=boolify(Op.le)),
    "≡": attrdict(arity=2, call=lambda w, x: int(w == x)),
    "≢": attrdict(arity=2, call=lambda w, x: int(w != x)),
    "≈": attrdict(
        arity=2,
        call=lambda w, x: len(w) == len(x)
        if type(w) == type(x) == list
        else abs(w - x) <= 1,
    ),
    ",": attrdict(arity=2, call=lambda w, x: [w] + [x]),
    ",": attrdict(arity=2, call=lambda w, x: [x] + [w]),
    "∊": attrdict(arity=2, dw=0, call=lambda w, x: w in iterable(x, digits_=True)),
    "⊏": attrdict(arity=2, dw=0, call=find),
    "⊑": attrdict(arity=2, dw=0, call=find_all),
    "∘": attrdict(arity=2, call=cartesian_product),
    "⊣": attrdict(arity=2, call=lambda w, x: w),
    "⊢": attrdict(arity=2, call=lambda w, x: x),
    "#": attrdict(arity=2, dw=1, call=reshape),
    "δ": attrdict(
        arity=2,
        dw=1,
        dx=1,
        call=lambda w, x: mp.sqrt(sum(map(lambda i: i * i, map(Op.sub, w, x)))),
    ),
    "»": attrdict(
        arity=2,
        dw=0,
        call=lambda w, x: iterable(x, digits_=True)[w:] + iterable(x, digits_=True)[:w],
    ),
    "«": attrdict(
        arity=2,
        dw=0,
        call=lambda w, x: iterable(x, digits_=True)[-w:]
        + iterable(x, digits_=True)[:-w],
    ),
}

transpiled_atoms = {
    "I": [],
    "K": [],
    "Ŀ": [],
    "Ṙ": [],
    "Ṡ": [],
    "Σ": [],
    "Π": [],
    "∩": [],
    "∪": [],
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
