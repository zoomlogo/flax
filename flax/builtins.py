# builtins: holds the builtins and some constants for the lexer
import operator
from flax.common import mpc, mpf, inf, mp, attrdict
from flax.funcs import *
from flax.chains import *

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
    "+": attrdict(arity=2, call=vecc(operator.add)),
    "×": attrdict(arity=2, call=vecc(operator.mul)),
    "C": attrdict(arity=1, call=vecc(lambda x: 1 - x)),
    "⍴": attrdict(arity=2, call=reshape),
}

quicks = {}

train_separators = {
    "ø": (0, True),
    "µ": (1, True),
    "г": (1, True),
    "ð": (2, True),
    "ɓ": (2, False),
}
