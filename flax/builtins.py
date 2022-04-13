# builtins: holds the builtins and some constants for the lexer
import itertools
import math
import more_itertools
import operator
import random as rrandom

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
    ":*": attrdict(
        arity=2, call=vecc(lambda w, x: (mp.matrix(w) ** x).tolist(), lfull=False)
    ),
    ":<": attrdict(arity=2, call=vecc(operator.ilshift)),
    ":>": attrdict(arity=2, call=vecc(operator.irshift)),
    ":j": attrdict(arity=2, call=vecc(lambda w, x: mpf(w, x))),
    ":Ȧ": attrdict(arity=2, call=vecc(mp.atan2)),
    ":•": attrdict(arity=2, call=lambda w, x: (mp.matrix(w) * mp.matrix(x)).tolist()),
    ";$": attrdict(arity=1, call=lambda x: sublists(permutations(x))),
    ";1": attrdict(arity=1, call=ones),
    ";A": attrdict(arity=1, call=vecc(mp.acos)),
    ";B": attrdict(
        arity=1,
        call=lambda x: iterable(x, digits=True) + iterable(x, digits=True)[::-1],
    ),
    ";C": attrdict(arity=1, call=vecc(mp.cos)),
    ";D": attrdict(arity=1, call=lambda x: mp.det(x)),
    ";F": attrdict(arity=1, call=vecc(prime_factors)),
    ";G": attrdict(
        arity=1,
        call=lambda x: ones(iterable(x) + [iterable(e)[::-1] for e in iterable(x)]),
    ),
    ";I": attrdict(
        arity=1, call=vecc(lambda x: [[i == j for i in range(x)] for j in range(x)])
    ),
    ";J": attrdict(arity=1, call=vecc(lambda x: [mpc(x).real, mpc(x).imag])),
    ";L": attrdict(arity=1, call=vecc(mp.ln)),
    ";M": attrdict(
        arity=1,
        call=lambda x: sum(iterable(x, digits=True)) / len(iterable(x, digits=True)),
    ),
    ";P": attrdict(arity=1, call=mp.polyroots),
    ";R": attrdict(arity=1, call=vecc(lambda x: list(range(2, x)))),
    ";S": attrdict(arity=1, call=vecc(mp.sin)),
    ";T": attrdict(arity=1, call=vecc(mp.tan)),
    ";c": attrdict(arity=1, call=vecc(mp.cosh)),
    ";l": attrdict(arity=1, call=vecc(lucas)),
    ";r": attrdict(arity=1, call=vecc(lambda x: list(range(x + 1)))),
    ";s": attrdict(arity=1, call=vecc(mp.sinh)),
    ";t": attrdict(arity=1, call=vecc(mp.tanh)),
    ";°": attrdict(arity=1, call=vecc(mp.radians)),
    ";²": attrdict(arity=1, call=vecc(lambda x: int(mp.sqrt(x)) == mp.sqrt(x))),
    ";Ċ": attrdict(arity=1, call=vecc(mp.sec)),
    ";ċ": attrdict(arity=1, call=vecc(lambda x: 1 / (x + 1) * mp.binomial(2 * x, x))),
    # ";Ġ": attrdict(arity=1, call=vecc(graph_distance)),
    ";Ȧ": attrdict(arity=1, call=vecc(mp.atan)),
    ";Ḃ": attrdict(
        arity=1,
        call=lambda x: [
            iterable(e, digits=True) + iterable(e, digits=True)[::-1]
            for e in iterable(x)
        ],
    ),
    ";Ṗ": attrdict(arity=1, call=vecc(nprimes)),
    ";Ṡ": attrdict(arity=1, call=vecc(mp.csc)),
    ";Ṫ": attrdict(arity=1, call=vecc(mp.cot)),
    ";Ạ": attrdict(arity=1, call=vecc(mp.asin)),
    ";√": attrdict(arity=1, call=vecc(lambda x: int(mp.sqrt(x)))),
    "<": attrdict(arity=2, call=vecc(operator.lt)),
    "=": attrdict(arity=2, call=vecc(operator.eq)),
    ">": attrdict(arity=2, call=vecc(operator.gt)),
    "?": attrdict(arity=1, call=vecc(random)),
    "A": attrdict(arity=1, call=vecc(abs)),
    "B": attrdict(arity=1, call=vecc(to_bin)),
    "C": attrdict(arity=1, call=vecc(lambda x: 1 - x)),
    "D": attrdict(arity=1, call=vecc(to_digits)),
    "E": attrdict(arity=1, call=more_itertools.all_equal),
    "F": attrdict(arity=1, call=flatten),
    "G": attrdict(arity=1, call=group_indicies),
    "H": attrdict(arity=1, call=lambda x: iterable(x).pop(0)),
    "I": attrdict(
        arity=1, call=lambda x: list(more_itertools.difference(x, initial=0))
    ),
    "J": attrdict(arity=1, call=lambda x: list(range(len(x)))),
    "K": attrdict(arity=1, call=lambda x: list(itertools.accumulate(x))),
    "L": attrdict(arity=1, call=lambda x: iterable(x).pop()),
    "M": attrdict(arity=1, call=vecc(lambda x: x * x)),
    "N": attrdict(arity=1, call=lambda x: iterable(x)[:-1]),
    "O": attrdict(arity=1, call=lambda x: x),
    "P": attrdict(arity=1, call=permutations),
    "Q": attrdict(arity=1, call=vecc(lambda x: x / 2)),
    "R": attrdict(
        arity=1,
        call=lambda x: [e[::-1] for e in x] if type(x) == list else to_digits(x)[::-1],
    ),
    "S": attrdict(arity=1, call=lambda x: list(sorted(x))),
    "T": attrdict(arity=1, call=lambda x: iterable(x)[1:]),
    "U": attrdict(arity=1, call=lambda x: list(set(iterable(x, digits=True)))),
    "V": attrdict(arity=1, call=vecc(mp.isprime)),
    "W": attrdict(arity=1, call=where),
    "X": attrdict(
        arity=1,
        call=lambda x: split(
            iterable(x, digits=True), len(iterable(x, digits=True)) // 2
        ),
    ),
    "Y": attrdict(
        arity=1,
        call=lambda x: [
            e for i, e in enumerate(iterable(x, digits=True)) if i % 2 == 0
        ],
    ),
    "Z": attrdict(arity=1, call=lambda x: list(map(list, zip(*iterable(x))))),
    "\\": attrdict(arity=1, call=unrepeat),
    "^": attrdict(arity=1, call=vecc(operator.xor)),
    "_(": attrdict(arity=0, call=lambda: to_chars("()")),
    "_+": attrdict(arity=0, call=lambda: [1, -1]),
    "_-": attrdict(arity=0, call=lambda: [-1, 1]),
    "_/": attrdict(arity=0, call=lambda: to_chars("/\\")),
    "_0": attrdict(arity=0, call=lambda: [0, 0]),
    "_1": attrdict(arity=0, call=lambda: [1, 1]),
    "_2": attrdict(arity=0, call=lambda: [2, 2]),
    "_<": attrdict(arity=0, call=lambda: to_chars("<>")),
    "_A": attrdict(arity=0, call=lambda: to_chars("ABCDEFGHIJKLMNOPQRSTUVWXYZ")),
    "_D": attrdict(arity=0, call=lambda: [[0,1],[1,0],[0,-1],[-1,0]]),
    "_H": attrdict(arity=0, call=lambda: to_chars("Hello, World!")),
    "_P": attrdict(arity=0, call=lambda: mp.phi),
    "_R": attrdict(arity=0, call=lambda: to_chars("0123456789abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ!\"#$%&'()*+,-./:;<=>?@[\\]^_`{|}~")),
    "_S": attrdict(arity=0, call=lambda: to_chars("ඞ")),
    "_V": attrdict(arity=0, call=lambda: to_chars("AEIOU")),
    "_Y": attrdict(arity=0, call=lambda: to_chars("AEIOUY")),
    "_W": attrdict(arity=0, call=lambda: to_chars("ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789_")),
    "_[": attrdict(arity=0, call=lambda: to_chars("[]")),
    "_a": attrdict(arity=0, call=lambda: to_chars("abcdefghijklmnopqrstuvwxyz")),
    "_d": attrdict(arity=0, call=lambda: [[1,1],[1,0],[1,-1],[0,1],[0,0],[0,-1],[-1,1],[-1,0],[-1,-1]]),
    "_e": attrdict(arity=0, call=lambda: mp.e),
    "_h": attrdict(arity=0, call=lambda: to_chars("hello world")),
    "_p": attrdict(arity=0, call=lambda: mp.pi),
    "_v": attrdict(arity=0, call=lambda: to_chars("aeiou")),
    "_y": attrdict(arity=0, call=lambda: to_chars("aeiouy")),
    "_x": attrdict(arity=0, call=lambda: to_chars("0123456789abcdef")),
    "_{": attrdict(arity=0, call=lambda: to_chars("{}")),
    "_Ạ": attrdict(arity=0, call=lambda: to_chars("ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz")),
    "_₀": attrdict(arity=0, call=lambda: 128),
    "_₁": attrdict(arity=0, call=lambda: 256),
    "_₂": attrdict(arity=0, call=lambda: 512),
    "_₃": attrdict(arity=0, call=lambda: 1024),
    "_₄": attrdict(arity=0, call=lambda: 2048),
    "_₅": attrdict(arity=0, call=lambda: 4096),
    "_₆": attrdict(arity=0, call=lambda: 8192),
    "_₇": attrdict(arity=0, call=lambda: 4294967296),
    "_∞": attrdict(arity=0, call=lambda: inf),
    "a": attrdict(arity=2, call=vecc(lambda w, x: w and x)),
    "b": attrdict(arity=2, call=vecc(to_base)),
    "c": attrdict(arity=2, call=vecc(mp.binomial)),
    "d": attrdict(
        arity=2, call=lambda w, x: [iterable(w) + iterable(e) for e in iterable(x)]
    ),
    "f": attrdict(
        arity=2, call=lambda w, x: [e for e in iterable(x) if e not in iterable(w)]
    ),
    "g": attrdict(arity=2, call=vecc(math.gcd)),
    "h": attrdict(arity=2, call=vecc(lambda w, x: iterable(x)[:w], rfull=False)),
    "i": attrdict(arity=2, call=vecc(index_into, rfull=False)),
    "l": attrdict(arity=2, call=vecc(math.lcm)),
    "m": attrdict(arity=2, call=vecc(lambda w, x: min(w, x))),
    "n": attrdict(arity=2, call=vecc(operator.floordiv)),
    "o": attrdict(arity=2, call=split_at),
    "p": attrdict(arity=2, call=lambda w, x: rrandom.choice([w, x])),
    "q": attrdict(arity=2, call=lambda w, x: exit(0)),
    "r": attrdict(arity=2, call=vecc(lambda w, x: list(range(w, x + 1)))),
    "s": attrdict(arity=2, call=vecc(split, rfull=False)),
    "t": attrdict(arity=2, call=vecc(lambda w, x: iterable(x)[w:], rfull=False)),
    "u": attrdict(
        arity=2, call=lambda w, x: [find(e, x) for e in iterable(w, range_=True)]
    ),
    "v": attrdict(arity=2, call=vecc(lambda w, x: w or x)),
    "w": attrdict(arity=2, call=vecc(sliding_window, rfull=False)),
    "x": attrdict(arity=2, call=vecc(lambda w, x: max(w, x))),
    "y": attrdict(arity=2, call=vecc(join, rfull=False)),
    "z": attrdict(arity=2, call=lambda w, x: list(map(list, zip(w, x)))),
    "{": attrdict(arity=1, call=vecc(lambda x: x - 1)),
    "|": attrdict(arity=2, call=vecc(operator.or_)),
    "}": attrdict(arity=1, call=vecc(lambda x: x + 1)),
    "~": attrdict(arity=1, call=vecc(operator.not_)),
}

quicks = {}

train_separators = {
    "ø": (0, True),
    "µ": (1, True),
    "г": (1, True),
    "ð": (2, True),
    "ɓ": (2, False),
}
