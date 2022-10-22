# builtins: holds the builtins and some constants for the lexer
import functools
import math
import string
import re
import statistics
import more_itertools as mit
import operator as Op
import random as Random
from itertools import zip_longest

from flax.common import *
from flax.funcs import *
from flax.chains import *
from flax.encoding import codepage

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
atoms = { # single byte atoms
    "⁰": attrdict(arity=0, call=lambda: 10),
    "¹": attrdict(arity=0, call=lambda: 16),
    "²": attrdict(arity=0, call=lambda: 26),
    "³": attrdict(arity=0, call=lambda: 32),
    "⁴": attrdict(arity=0, call=lambda: 64),
    "⁵": attrdict(arity=0, call=lambda: 100),
    "⁶": attrdict(arity=0, call=lambda: 256),
    "⁷": attrdict(arity=0, call=lambda: -2),
    "⁸": attrdict(arity=0, call=lambda: 0),
    "⁹": attrdict(arity=0, call=lambda: 0),
    "∃": attrdict(arity=0, call=lambda: 0),
    "⊶": attrdict(arity=0, call=lambda: [0, 1]),
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
    "ϕ": attrdict(arity=1, call=lambda x: sum(map(iterable, iterable(x)), [])),
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
    "ḋ": attrdict(arity=2, dw=0, dx=0, call=lambda w, x: int(x % w == 0)),    "e": attrdict(arity=2, dw=0, dx=0, call=lambda w, x: list(range(w, x))),
    # "ė": attrdict(arity=2, ),
    "f": attrdict(
        arity=2, call=lambda w, x: [i for i in iterable(x) if i in iterable(w)]
    ),
    "ḟ": attrdict(
        arity=2, call=lambda w, x: [i for i in iterable(x) if i not in iterable(w)]
    ),
    "g": attrdict(arity=2, dw=0, dx=0, call=math.gcd),
    "ġ": attrdict(arity=2, dw=1, dx=1, call=lambda w, x: sum(map(lambda i: i[0]*i[1], zip_longest(w, x, fillvalue=0)))),
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
    ",": attrdict(arity=2, call=lambda w, x: iterable(w) + iterable(x)),
    ";": attrdict(arity=2, call=lambda w, x: iterable(x) + iterable(w)),
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

atoms |= { # diagraphs
    "Ø+": attrdict(arity=0, call=lambda: [1, -1]),
    "Ø-": attrdict(arity=0, call=lambda: [-1, 1]),
    "Ø⁰": attrdict(arity=0, call=lambda: [1,2,3]),
    "Ø¹": attrdict(arity=0, call=lambda: [[0,1],[1,0]]),
    "Ø²": attrdict(arity=0, call=lambda: 2**32),
    "Ø³": attrdict(arity=0, call=lambda: [1,2]),
    "Ø⁴": attrdict(arity=0, call=lambda: 2**64),
    "Ø⁵": attrdict(arity=0, call=lambda: 128),
    "Ø⁶": attrdict(arity=0, call=lambda: 512),
    "Ø⁷": attrdict(arity=0, call=lambda: 1024),
    "Ø⁸": attrdict(arity=0, call=lambda: 2048),
    "Ø⁹": attrdict(arity=0, call=lambda: 65536),
    "Ø0": attrdict(arity=0, call=lambda: [0,0]),
    "Ø1": attrdict(arity=0, call=lambda: [1,1]),
    "Ø2": attrdict(arity=0, call=lambda: [2,2]),
    "Ød": attrdict(arity=0, call=lambda: [[0,1],[1,0],[0,-1],[-1,0]]),
    "Øx": attrdict(arity=0, call=lambda: [[1,1],[1,0],[1,-1],[0,1],[0,0],[0,-1],[-1,1],[-1,0],[-1,-1]]),
    "Øτ": attrdict(arity=0, call=lambda: 360),
    "Ø(": attrdict(arity=0, call=lambda: [40,41]),
    "Ø[": attrdict(arity=0, call=lambda: [91,93]),
    "Ø/": attrdict(arity=0, call=lambda: [47,92]),
    "Ø<": attrdict(arity=0, call=lambda: [60,62]),
    "Ø{": attrdict(arity=0, call=lambda: [123,125]),
    "ØA": attrdict(arity=0, call=lambda: to_chars(string.ascii_uppercase)),
    "Øa": attrdict(arity=0, call=lambda: to_chars(string.ascii_lowercase)),
    "ØB": attrdict(arity=0, call=lambda: to_chars(re.sub("[AEIOU]", "", string.ascii_uppercase))),
    "Øb": attrdict(arity=0, call=lambda: to_chars(re.sub("[aeiou]", "", string.ascii_lowercase))),
    "ØV": attrdict(arity=0, call=lambda: to_chars("AEIOU")),
    "Øv": attrdict(arity=0, call=lambda: to_chars("aeiou")),
    "ØY": attrdict(arity=0, call=lambda: to_chars("AEIOUY")),
    "Øy": attrdict(arity=0, call=lambda: to_chars("aeiouy")),
    "ØD": attrdict(arity=0, call=lambda: to_chars(string.digits)),
    "ØX": attrdict(arity=0, call=lambda: to_chars(string.hexdigits)),
    "ØO": attrdict(arity=0, call=lambda: to_chars(string.octdigits)),
    "Øα": attrdict(arity=0, call=lambda: to_chars(string.ascii_letters)),
    "ØW": attrdict(arity=0, call=lambda: to_chars(string.digits + string.ascii_letters + "_")),
    "Øc": attrdict(arity=0, call=lambda: to_chars(codepage)),
    "Øe": attrdict(arity=0, call=lambda: mp.e),
    "Øφ": attrdict(arity=0, call=lambda: mp.phi),
    "Øπ": attrdict(arity=0, call=lambda: mp.pi),
    "Øδ": attrdict(arity=0, call=lambda: mp.sqrt(2) + 1),
    "Øγ": attrdict(arity=0, call=lambda: mp.euler),
    "Ø∞": attrdict(arity=0, call=lambda: inf),
    "Æ√": attrdict(arity=1, dx=0, call=lambda x: int(mp.sqrt(x))),
    "ÆĊ": attrdict(arity=1, dx=0, call=mp.acos),
    "ÆṠ": attrdict(arity=1, dx=0, call=mp.asin),
    "ÆṪ": attrdict(arity=1, dx=0, call=mp.atan),
    "Æċ": attrdict(arity=1, dx=0, call=mp.asec),
    "Æṡ": attrdict(arity=1, dx=0, call=mp.acsc),
    "Æṫ": attrdict(arity=1, dx=0, call=mp.acot),
    "ÆS": attrdict(arity=1, dx=0, call=mp.sin),
    "ÆC": attrdict(arity=1, dx=0, call=mp.cos),
    "ÆT": attrdict(arity=1, dx=0, call=mp.tan),
    "Æs": attrdict(arity=1, dx=0, call=mp.csc),
    "Æc": attrdict(arity=1, dx=0, call=mp.sec),
    "Æt": attrdict(arity=1, dx=0, call=mp.cot),
    "Æp": attrdict(arity=1, dx=0, call=lambda x: len([i for i in range(x + 1) if mp.isprime(i)])),
    "Æn": attrdict(arity=1, dx=0, call=nprimes),
    "ÆF": attrdict(arity=1, dx=0, call=prime_factors),
    "ÆL": attrdict(arity=1, dx=0, call=mp.ln),
    "ÆĿ": attrdict(arity=1, dx=0, call=mp.exp),
    "Æŀ": attrdict(arity=1, dx=0, call=lambda x: mp.binomial(2*x, x) / (x + 1)),
    "Æl": attrdict(arity=1, dx=0, call=lucas),
    "Æf": attrdict(arity=1, dx=0, call=fibonacci),
    "Æτ": attrdict(arity=1, dx=2, call=lambda x: sum(diagonal_leading(x))),
    "Æ²": attrdict(arity=1, dx=0, call=lambda x: int(int(mp.sqrt(x)) == mp.sqrt(x))),
    "ÆA": attrdict(arity=1, dx=2, call=lambda x: diagonals(x, antidiagonals=True)),
    "ÆȦ": attrdict(arity=1, dx=2, call=diagonals),
    # "ÆD": attrdict(arity=1, dx=2, call=),
    "ÆR": attrdict(arity=1, dx=1, call=mp.polyroots),
    "Æd": attrdict(arity=1, dx=0, call=mp.degrees),
    "Æḋ": attrdict(arity=1, dx=0, call=mp.radians),
    "Æ\\": attrdict(arity=1, dx=2, call=diagonal_leading),
    "Æ/": attrdict(arity=1, dx=2, call=diagonal_trailing),
    "Æj": attrdict(arity=1, dx=0, call=mp.conj),
    "Æσ": attrdict(arity=1, dx=1, call=statistics.pstdev),
    "Æm": attrdict(arity=1, dx=1, call=statistics.mean),
    "Æṁ": attrdict(arity=1, dx=1, call=statistics.median),
    "Æg": attrdict(arity=1, dx=1, call=statistics.geometric_mean),
    "Æh": attrdict(arity=1, dx=1, call=statistics.harmonic_mean),
    "æ∘": attrdict(arity=2, dw=1, dx=1, call=lambda w, x: sum(map(lambda i: i[0]*i[1], zip_longest(w, x, fillvalue=1)))),
}

transpiled_atoms = {
    "I": [
        [TOKEN_TYPE.ATOM, "-"],
        [TOKEN_TYPE.QUICK, "˜"],
        [TOKEN_TYPE.QUICK, "/"],
        [TOKEN_TYPE.QUICK, "υ"],
    ],
    "K": [[TOKEN_TYPE.ATOM, "-"], [TOKEN_TYPE.QUICK, "\\"]],
    "Ŀ": [[TOKEN_TYPE.ATOM, "L"], [TOKEN_TYPE.QUICK, "'"]],
    "Ṙ": [[TOKEN_TYPE.ATOM, "R"], [TOKEN_TYPE.QUICK, "'"]],
    "Ṡ": [[TOKEN_TYPE.ATOM, "+"], [TOKEN_TYPE.QUICK, "/"], [TOKEN_TYPE.QUICK, "'"]],
    "Σ": [[TOKEN_TYPE.ATOM, "+"], [TOKEN_TYPE.QUICK, "/"]],
    "Π": [[TOKEN_TYPE.ATOM, "×"], [TOKEN_TYPE.QUICK, "/"]],
    "∩": [[TOKEN_TYPE.ATOM, "U"], [TOKEN_TYPE.ATOM, "f"], [TOKEN_TYPE.QUICK, "¢"]],
    "∪": [[TOKEN_TYPE.ATOM, "U"], [TOKEN_TYPE.ATOM, ","], [TOKEN_TYPE.QUICK, "¢"]],
}

quicks = { # single byte quicks
    "$": quick_chain(1, 2),
    "¢": quick_chain(2, 2),
    "£": quick_chain(1, 3),
    "¥": quick_chain(2, 3),
    "€": quick_chain(1, 4),
    "₹": quick_chain(2, 4),
    "'": attrdict(
        condition=lambda links: links,
        qlink=lambda links, *_: [
            attrdict(
                arity=links[0].arity or 1,
                call=fix_args(
                    lambda w, x: [
                        variadic_link(links[0], (i, j))
                        for i, j in zip(iterable(w), iterable(x))
                    ]
                    if w is not None
                    else [variadic_link(links[0], (i,)) for i in iterable(x)]
                ),
            )
        ],
    ),
    "’": attrdict(
        condition=lambda links: links,
        qlink=lambda links, *_: [
            attrdict(
                arity=2,
                call=lambda w, x: [
                    variadic_link(links[0], (i, x)) for i in iterable(w)
                ],
            )
        ],
    ),
    "‘": attrdict(
        condition=lambda links: links,
        qlink=lambda links, *_: [
            attrdict(
                arity=2,
                call=lambda w, x: [
                    variadic_link(links[0], (w, i)) for i in iterable(x)
                ],
            )
        ],
    ),
    "¨": attrdict(
        condition=lambda links: len(links) == 2,
        qlink=lambda links, *_: [
            attrdict(
                arity=1,
                call=lambda x: [variadic_chain(links, (i,)) for i in x]
                if type(x) == list
                else variadic_chain(links, (x,)),
            )
        ],
    ),
    "ζ": attrdict(
        condition=lambda links: links,
        qlink=lambda links, *_: [
            atoms["ϕ"],
            attrdict(
                arity=links[0].arity or 1,
                call=fix_args(
                    lambda w, x: [
                        variadic_link(links[0], (i, j))
                        for i, j in zip(iterable(w), iterable(x))
                    ]
                    if w is not None
                    else [variadic_link(links[0], (i,)) for i in iterable(x)]
                ),
            ),
        ],
    ),
    "ρ": attrdict(
        condition=lambda links: links,
        qlink=lambda links, *_: [
            attrdict(
                arity=links[0].arity or 1,
                call=fix_args(
                    lambda w, x: [
                        variadic_link(links[0], (i, j))
                        for i, j in zip(
                            permutations(iterable(w)), permutations(iterable(x))
                        )
                    ]
                    if w is not None
                    else [
                        variadic_link(links[0], (i,)) for i in permutations(iterable(x))
                    ]
                ),
            )
        ],
    ),
    "⊸": attrdict(
        condition=lambda links: links,
        qlink=lambda links, *_: [
            attrdict(
                arity=links[0].arity or 1,
                call=fix_args(
                    lambda w, x: [
                        variadic_link(links[0], (i, j))
                        for i, j in zip(prefixes(iterable(w)), prefixes(iterable(x)))
                    ]
                    if w is not None
                    else [variadic_link(links[0], (i,)) for i in prefixes(iterable(x))]
                ),
            )
        ],
    ),
    "?": attrdict(
        condition=lambda links: len(links) == 3,
        qlink=lambda links, *_: [
            attrdict(
                arity=max(arities(links)),
                call=fix_args(
                    lambda w, x: (
                        variadic_link(links[0], (w, x))
                        if variadic_link(links[2], (w, x))
                        else variadic_link(links[1], (w, x))
                    )
                ),
            )
        ],
    ),
    "ω": attrdict(
        condition=lambda links: len(links) == 2,
        qlink=lambda links, *_: [
            attrdict(
                arity=max(arities(links)),
                call=fix_args(lambda w, x: while_loop(links[0], links[1], (w, x))),
            )
        ],
    ),
    "⍤": attrdict(
        condition=lambda links: links and links[0].arity,
        qlink=lambda links, *_: (
            [links.pop(0)] if len(links) == 2 and links[0].arity == 0 else []
        )
        + [
            attrdict(
                arity=max_arity(links),
                call=fix_args(lambda w, x: ntimes(links, (w, x))),
            )
        ],
    ),
    "@": attrdict(
        condition=lambda links: len(links) == 2,
        qlink=lambda links, *_: [
            attrdict(
                arity=links[0].arity or 1,
                call=fix_args(lambda w, x: apply_at(links[0], links[1].call(), w, x)),
            )
        ],
    ),
    "˘": attrdict(
        condition=lambda links: links,
        qlink=lambda links, *_: [
            attrdict(
                arity=1 if links[0].arity == 2 else (2 if links[0].arity == 1 else 0),
                call=fix_args(
                    lambda _, x: variadic_link(links[0], (x, x))
                    if links[0].arity == 2
                    else variadic_link(links[0], (x,)),
                ),
            )
        ],
    ),
    "˜": attrdict(
        condition=lambda links: links,
        qlink=lambda links, *_: [
            attrdict(
                arity=2 if links[0].arity else 0,
                call=fix_args(
                    lambda w, x: variadic_link(links[0], (x, w))
                    if links[0].arity != 1
                    else links[0].call(w),
                ),
            )
        ],
    ),
    "`": attrdict(
        condition=lambda links: links and links[0].arity,
        qlink=lambda links, *_: [
            attrdict(
                arity=links[0].arity, call=fix_args(lambda w, x: sort(links, w, x))
            )
        ],
    ),
    "η": attrdict(
        condition=lambda links: links and links[0].arity,
        qlink=lambda links, *_: [
            attrdict(
                arity=links[0].arity, call=fix_args(lambda w, x: group(links, w, x))
            )
        ],
    ),
    "⌉": attrdict(
        condition=lambda links: links,
        qlink=lambda links, *_: [
            attrdict(
                arity=links[0].arity,
                call=fix_args(
                    lambda w, x: max(
                        iterable(x), key=lambda k: variadic_link(links[0], (w, k))
                    )
                ),
            )
        ],
    ),
    "⌋": attrdict(
        condition=lambda links: links,
        qlink=lambda links, *_: [
            attrdict(
                arity=links[0].arity,
                call=fix_args(
                    lambda w, x: min(
                        iterable(x), key=lambda k: variadic_link(links[0], (w, k))
                    )
                ),
            )
        ],
    ),
    "τ": attrdict(
        condition=lambda links: links,
        qlink=lambda links, *_: [
            attrdict(
                arity=links[0].arity or 1,
                call=fix_args(
                    lambda w, x: [
                        i
                        for i, e in enumerate(
                            [variadic_link(links[0], (w, i)) for i in iterable(x)]
                        )
                        if e
                    ],
                ),
            )
        ],
    ),
    "α": attrdict(
        condition=lambda links: links,
        qlink=lambda links, *_: [
            attrdict(
                arity=links[0].arity or 1,
                call=fix_args(
                    lambda w, x: mit.all_equal(
                        [variadic_link(links[0], (w, i)) for i in sliding_window(2, x)]
                    )
                ),
            ),
        ],
    ),
    "υ": attrdict(
        condition=lambda links: links,
        qlink=lambda links, *_: [
            attrdict(
                arity=links[0].arity or 1,
                call=fix_args(
                    lambda w, x: [
                        variadic_link(links[0], (w, i)) for i in sliding_window(2, x)
                    ]
                ),
            ),
        ],
    ),
    "⊥": attrdict(
        condition=lambda links: links,
        qlink=lambda links, *_: [
            attrdict(
                arity=links[0].arity or 1,
                call=fix_args(lambda w, x: ffilter(links, w, x)),
            )
        ],
    ),
    "⊤": attrdict(
        condition=lambda links: links,
        qlink=lambda links, *_: [
            attrdict(
                arity=links[0].arity or 1,
                call=fix_args(lambda w, x: ffilter(links, w, x, inverse=True)),
            )
        ],
    ),
    "⁼": attrdict(
        condition=lambda links: links,
        qlink=lambda links, *_: [
            attrdict(
                arity=links[0].arity,
                call=fix_args(lambda w, x: int(w == variadic_link(links[0], w, x))),
            )
        ],
    ),
    "β": attrdict(
        condition=lambda _: True,
        qlink=lambda _, outermost_links, i: [create_chain(outermost_links[i])],
    ),
    "θ": attrdict(
        condition=lambda _: True,
        qlink=lambda _, outermost_links, i: [
            create_chain(outermost_links[(i + 1) % len(outermost_links)], 0)
        ],
    ),
    "λ": attrdict(
        condition=lambda _: True,
        qlink=lambda _, outermost_links, i: [
            create_chain(outermost_links[(i + 1) % len(outermost_links)], 1)
        ],
    ),
    "ν": attrdict(
        condition=lambda _: True,
        qlink=lambda _, outermost_links, i: [
            create_chain(outermost_links[(i + 1) % len(outermost_links)], 2)
        ],
    ),
    "σ": attrdict(
        condition=lambda _: True,
        qlink=lambda _, outermost_links, i: [
            create_chain(outermost_links[(i - 1) % len(outermost_links)], 0)
        ],
    ),
    "ς": attrdict(
        condition=lambda _: True,
        qlink=lambda _, outermost_links, i: [
            create_chain(outermost_links[(i - 1) % len(outermost_links)], 1)
        ],
    ),
    "π": attrdict(
        condition=lambda _: True,
        qlink=lambda _, outermost_links, i: [
            create_chain(outermost_links[(i - 1) % len(outermost_links)], 2)
        ],
    ),
    "φ": attrdict(
        condition=lambda links: len(links) == 2,
        qlink=lambda links, outermost_links, _: [
            create_chain(
                outermost_links[links[0].call() % len(outermost_links)], links[1].call()
            )
        ],
    ),
    "/": attrdict(
        condition=lambda links: links and links[0].arity,
        qlink=lambda links, *_: [
            attrdict(arity=1, call=lambda x: fold(links, x, right=True))
        ],
    ),
    "⌿": attrdict(
        condition=lambda links: links and links[0].arity,
        qlink=lambda links, *_: [
            attrdict(
                arity=2, call=lambda w, x: fold(links, w, x, initial=True, right=True)
            )
        ],
    ),
    "\\": attrdict(
        condition=lambda links: links and links[0].arity,
        qlink=lambda links, *_: [
            attrdict(arity=1, call=lambda x: scan(links, x, right=True))
        ],
    ),
    "⍀": attrdict(
        condition=lambda links: links and links[0].arity,
        qlink=lambda links, *_: [
            attrdict(
                arity=2, call=lambda w, x: scan(links, w, x, initial=True, right=True)
            )
        ],
    ),
    "´": attrdict(
        condition=lambda links: links,
        qlink=lambda links, *_: [
            attrdict(
                arity=links[0].arity or 1,
                call=fix_args(lambda w, x: fold_fixedpoint(links, w, x)),
            )
        ],
    ),
    "˝": attrdict(
        condition=lambda links: links,
        qlink=lambda links, outermost_links, i: [
            attrdict(
                arity=links[0].arity or 1,
                call=fix_args(lambda w, x: scan_fixedpoint(links, w, x)),
            )
        ],
    ),
    "∀": attrdict(
        condition=lambda links: links,
        qlink=lambda links, *_: [
            attrdict(
                arity=links[0].arity,
                call=fix_args(
                    lambda w, x: copy_to(atoms["∃"], variadic_link(links[0], w, x)),
                ),
            )
        ],
    ),
    "∝": attrdict(
        condition=lambda links: links,
        qlink=lambda links, *_: [
            attrdict(
                arity=1,
                call=lambda x: variadic_link(
                    links[0].call,
                    (transpose(variadic_link(links[0].call, (x,), force_monad=True)),),
                    force_monad=True,
                ),
            )
        ],
    ),
    "⟜": attrdict(
        condition=lambda links: links
        and (
            links[-1].arity == 0
            and len(links) == links[-1].call() - 1
            or len(links) == 3
        ),
        qlink=lambda links, *_: [
            attrdict(arity=2, call=lambda w, x: composed(links, w, x))
        ],
    ),
    # "˙": attrdict(), TODO: I FORGOR LOL
    "¾": attrdict(
        condition=lambda links: len(links) == 2,
        qlink=lambda links, *_: [
            attrdict(
                arity=2,
                call=lambda w, x: dyadic_link(
                    links[1],
                    variadic_link(links[0], (x, w)),
                    variadic_link(links[0], (w, x)),
                ),
            )
        ],
    ),
}

train_separators = {
    "ø": (0, True),
    "µ": (1, True),
    "(": (1, True),
    ")": (1, True),
    "ð": (2, True),
    "ɓ": (2, False),
}
