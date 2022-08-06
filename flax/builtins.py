# builtins: holds the builtins and some constants for the lexer
import itertools
import functools
import json
import sys
import math
import more_itertools
import operator
import random as rrandom

from flax.common import flax_print, mpc, mpf, inf, mp, attrdict
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
COMMENT = "·"
COMPLEX_DELIMETER = "j"
DECIMAL_POINT = "."
DIAGRAPHS = "ØÆŒæœΔ"
LIST_DELIMETER_L = "("
LIST_DELIMETER_R = ")"
NEGATIVE_SIGN = "¯"
NEWLINE = "\n"
STRING_DELIMETER = '"'
ZERO = "0"
DIGITS = ZERO + "123456789" + DECIMAL_POINT + COMPLEX_DELIMETER + NEGATIVE_SIGN

# dicts
atoms = {
}

quicks = {
}

train_separators = {
    "ø": (0, True),
    "µ": (1, True),
    "[": (1, True),
    "]": (1, True),  # impl
    "ð": (2, True),
    "ɓ": (2, False),
}
