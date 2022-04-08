# builtins: holds the builtins and some constants for the lexer
import enum
import operator
import more_itertools
import itertools

from flax.common import mpc, mpf, inf, mp, attrdict
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
    "quicks",
    "train_separators",
]

# constants
COMMENT = "⍝"
COMPLEX_DELIMETER = "j"
DECIMAL_POINT = "."
DIAGRAPHS = "_;:ᵟ"
LIST_DELIMETER_L = "["
LIST_DELIMETER_R = "]"
NEGATIVE_SIGN = "¯"
NEWLINE = "\n"
STRING_DELIMETER = '"'
STRING_NEXT_1 = "₊"
STRING_NEXT_2 = "₋"
ZERO = "0"
DIGITS = ZERO + "123456789" + DECIMAL_POINT + COMPLEX_DELIMETER + NEGATIVE_SIGN

# dicts
atoms = {
    "!": attrdict(arity=1, call=vecc(mp.factorial)),
    "#": attrdict(arity=1, call=len),
    "$": attrdict(arity=1, call=sublists),
    "%": attrdict(arity=2, call=vecc(operator.mod)),
    "&": attrdict(arity=2, call=vecc(operator.and_)),
    "(": attrdict(arity=1, call=prefixes),
    ")": attrdict(arity=1, call=suffixes),
    "*": attrdict(arity=2, call=vecc(operator.pow)),
    "+": attrdict(arity=2, call=vecc(operator.add)),
    ",": attrdict(arity=2, call=lambda w, x: iterable(w) + iterable(x)),
    "-": attrdict(arity=2, call=vecc(operator.sub)),
    "/": attrdict(arity=2, call=repeat),
    "<": attrdict(arity=2, call=vecc(operator.lt)),
    "=": attrdict(arity=2, call=vecc(operator.eq)),
    "<": attrdict(arity=2, call=vecc(operator.gt)),
    "?": attrdict(arity=1, call=vecc(random)),
    "A": attrdict(arity=1, call=vecc(abs)),
    "B": attrdict(arity=1, call=vecc(to_bin)),
    "C": attrdict(arity=1, call=vecc(lambda x: 1 - x)),
    "D": attrdict(arity=1, call=vecc(to_digits)),
    "E": attrdict(arity=1, call=more_itertools.all_equal),
    "F": attrdict(arity=1, call=flatten),
    "G": attrdict(arity=1, call=group_indicies),
    "H": attrdict(arity=1, call=lambda x: iterable(x).pop(0)),
    "I": attrdict(arity=1, call=lambda x: list(more_itertools.difference(x, initial=0))),
    "J": attrdict(arity=1, call=lambda x: list(range(len(x)))),
    "K": attrdict(arity=1, call=lambda x: list(itertools.accumulate(x))),
    "L": attrdict(arity=1, call=lambda x: iterable(x).pop()),
    "M": attrdict(arity=1, call=vecc(lambda x: x*x)),
    "N": attrdict(arity=1, call=lambda x: iterable(x)[:-1]),
    "O": attrdict(arity=1, call=lambda x: x),
    "P": attrdict(arity=1, call=permutations),
    "Q": attrdict(arity=1, call=vecc(lambda x: x / 2)),
    "R": attrdict(arity=1, call=lambda x: [e[::-1] for e in x] if type(x) == list else to_digits(x)[::-1]),
    "S": attrdict(arity=1, call=lambda x: list(sorted(x))),
    "T": attrdict(arity=1, call=lambda x: iterable(x)[1:]),
    "U": attrdict(arity=1, call=lambda x: list(set(iterable(x, digits=True)))),
    "V": attrdict(arity=1, call=vecc(mp.isprime)),
    "Y": attrdict(arity=1, call=lambda x: [e for i,e in enumerate(iterable(x, digits=True)) if i % 2 == 0]),
    "Z": attrdict(arity=1, call=lambda x: list(map(list, zip(*iterable(x))))),
    "\\": attrdict(arity=1, call=unrepeat),
}

quicks = {}

train_separators = {
    "ø": (0, True),
    "µ": (1, True),
    "г": (1, True),
    "ð": (2, True),
    "ɓ": (2, False),
}
