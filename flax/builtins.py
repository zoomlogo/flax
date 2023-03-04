# builtins: holds the builtins and some constants for the lexer
from flax.common import *

# unused characters: İẆṅṙΛ
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
COMMENT = "⊳"
COMPLEX_DELIMETER = "i"
DECIMAL_POINT = "."
DIAGRAPHS = "ØÆŒæœΔ"
LIST_DELIMETER_L = "["
LIST_DELIMETER_R = "]"
NEGATIVE_SIGN = "¯"
NEWLINE = "\n"
STRING_DELIMETER = '"'
ZERO = "0"
DIGITS = ZERO + "123456789" + DECIMAL_POINT + COMPLEX_DELIMETER + NEGATIVE_SIGN
STRING_NEXT_1 = "_"
STRING_NEXT_2 = ":"

# dicts
transpiled_atoms = {
    "I": [
        [TOKEN_TYPE.ATOM, "-"],
        [TOKEN_TYPE.QUICK, "˜"],
        [TOKEN_TYPE.QUICK, "/"],
        [TOKEN_TYPE.QUICK, "υ"],
    ],
    "K": [[TOKEN_TYPE.ATOM, "+"], [TOKEN_TYPE.QUICK, "\\"]],
    "Ŀ": [[TOKEN_TYPE.ATOM, "L"], [TOKEN_TYPE.QUICK, "'"]],
    "Ṙ": [[TOKEN_TYPE.ATOM, "R"], [TOKEN_TYPE.QUICK, "'"]],
    "Ṡ": [[TOKEN_TYPE.ATOM, "+"], [TOKEN_TYPE.QUICK, "/"], [TOKEN_TYPE.QUICK, "'"]],
    "Σ": [[TOKEN_TYPE.ATOM, "+"], [TOKEN_TYPE.QUICK, "/"]],
    "Π": [[TOKEN_TYPE.ATOM, "×"], [TOKEN_TYPE.QUICK, "/"]],
    "ẏ": [[TOKEN_TYPE.ATOM, "ϕ"], [TOKEN_TYPE.ATOM, "z"], [TOKEN_TYPE.QUICK, "¢"]],
    "∩": [[TOKEN_TYPE.ATOM, "U"], [TOKEN_TYPE.ATOM, "f"], [TOKEN_TYPE.QUICK, "¢"]],
    "∪": [[TOKEN_TYPE.ATOM, "U"], [TOKEN_TYPE.ATOM, ","], [TOKEN_TYPE.QUICK, "¢"]],
    "ŒḂ": [[TOKEN_TYPE.ATOM, "ŒB"], [TOKEN_TYPE.QUICK, "'"]],
    "œ∩": [[TOKEN_TYPE.ATOM, "∂"], [TOKEN_TYPE.ATOM, "f"], [TOKEN_TYPE.QUICK, "¢"]],
    "œ-": [[TOKEN_TYPE.ATOM, "∂"], [TOKEN_TYPE.ATOM, "ḟ"], [TOKEN_TYPE.QUICK, "¢"]],
    "œ∪": [[TOKEN_TYPE.ATOM, "∂"], [TOKEN_TYPE.ATOM, ","], [TOKEN_TYPE.QUICK, "¢"]],
}

train_separators = {
    "ø": (0, True),
    "µ": (1, True),
    "(": (1, True),
    ")": (1, True),
    "ð": (2, True),
    "ɓ": (2, False),
}

atoms = __import__("flax.atoms").atoms.atoms
quicks = __import__("flax.quicks").quicks.quicks
