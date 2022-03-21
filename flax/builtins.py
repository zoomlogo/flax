# builtins: holds the builtins and some constants for the lexer

# constants
STRING_DELIMETER = "'"
STRING_NEXT_1 = "₊"
STRING_NEXT_2 = "₋"
ZERO = "0"
DECIMAL_POINT = "."
COMPLEX_DELIMETER = "j"
NEGATIVE_SIGN = "¯"
DIGITS = ZERO + "123456789" + DECIMAL_POINT + COMPLEX_DELIMETER + NEGATIVE_SIGN
NEWLINE = "\n"
COMMENT = "⍝"
DIAGRAPHS = "_;:ᵟ"
LIST_DELIMETER_L = "["
LIST_DELIMETER_R = "]"

# dicts
atoms = {}

quicks = {}

train_separators = {
    "ø": (0, True),
    "µ": (1, True),
    "г": (1, True),
    "ð": (2, True),
    "ɓ": (2, False),
}
