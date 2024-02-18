# atoms: holds the atoms
import functools
import math
import string
import re
import statistics
import more_itertools as mit
import operator as ops
import random as Random
from itertools import zip_longest

from flax.common import *
from flax.funcs import *
from flax.chains import *
from flax.encoding import codepage

atoms = {  # single byte atoms
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
    "N": attrdict(arity=1, dx=0, call=ops.neg),
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
    "T": attrdict(arity=1, dx=0, call=lambda x: 2**x),
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
    "¬": attrdict(arity=1, dx=0, call=boolify(ops.not_)),
    "√": attrdict(arity=1, dx=0, call=mp.sqrt),
    "⊂": attrdict(arity=1, call=lambda x: [x]),
    "⊆": attrdict(
        arity=1, call=lambda x: [x] if x != iterable(x) or len(x) != 1 else x
    ),
    "⊃": attrdict(arity=1, call=lambda x: iterable(x)[0]),  # err
    "⊇": attrdict(arity=1, call=lambda x: iterable(x)[1:]),
    "⊐": attrdict(arity=1, call=lambda x: iterable(x)[-1]),  # err
    "⊒": attrdict(arity=1, call=lambda x: iterable(x)[:-1]),
    "~": attrdict(arity=1, dx=0, call=ops.inv),
    "γ": attrdict(arity=1, call=flax_print),
    "ε": attrdict(arity=1, call=lambda x: list(enumerate(x))),
    "ι": attrdict(arity=1, call=iota),
    "κ": attrdict(arity=1, call=iota1),
    "ξ": attrdict(arity=1, call=lambda x: transpose(x, filler=0)),
    "χ": attrdict(arity=1, call=lambda x: int(all(iterable(x)))),
    "ψ": attrdict(arity=1, call=lambda x: int(iterable(x) > [] and all(flatten(x)))),
    "ϕ": attrdict(arity=1, call=lambda x: list(mit.flatten(x))),
    "∵": attrdict(
        arity=1,
        call=lambda x: (
            min(iterable(x, digits_=True)) if iterable(x, digits_=True) else 0
        ),
    ),
    "∴": attrdict(
        arity=1,
        call=lambda x: (
            max(iterable(x, digits_=True)) if iterable(x, digits_=True) else 0
        ),
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
    "ė": attrdict(arity=2, dw=0, dx=0, call=base_decomp),
    "f": attrdict(
        arity=2, call=lambda w, x: [i for i in iterable(x) if i in iterable(w)]
    ),
    "ḟ": attrdict(
        arity=2, call=lambda w, x: [i for i in iterable(x) if i not in iterable(w)]
    ),
    "g": attrdict(arity=2, dw=0, dx=0, call=math.gcd),
    "ġ": attrdict(
        arity=2,
        dw=1,
        dx=1,
        call=lambda w, x: sum(
            map(lambda i: i[0] * i[1], zip_longest(w, x, fillvalue=0))
        ),
    ),
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
    "ṗ": attrdict(arity=2, dw=0, call=lambda w, x: cartesian_product(*([x] * w))),
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
    "+": attrdict(arity=2, dw=0, dx=0, call=ops.add),
    "-": attrdict(arity=2, dw=0, dx=0, call=ops.sub),
    "±": attrdict(arity=2, dw=0, dx=0, call=lambda w, x: [w + x, w - x]),
    "×": attrdict(arity=2, dw=0, dx=0, call=ops.mul),
    "÷": attrdict(arity=2, dw=0, dx=0, call=ops.truediv),
    "|": attrdict(arity=2, dw=0, dx=0, call=lambda w, x: x % w),
    "*": attrdict(arity=2, dw=0, dx=0, call=ops.pow),
    "&": attrdict(arity=2, dw=0, dx=0, call=ops.and_),
    "%": attrdict(arity=2, dw=0, dx=0, call=ops.or_),
    "^": attrdict(arity=2, dw=0, dx=0, call=ops.xor),
    "∧": attrdict(arity=2, dw=0, dx=0, call=lambda w, x: w and x),
    "∨": attrdict(arity=2, dw=0, dx=0, call=lambda w, x: w or x),
    ">": attrdict(arity=2, dw=0, dx=0, call=boolify(ops.gt)),
    "<": attrdict(arity=2, dw=0, dx=0, call=boolify(ops.lt)),
    "=": attrdict(arity=2, dw=0, dx=0, call=boolify(ops.eq)),
    "≠": attrdict(arity=2, dw=0, dx=0, call=boolify(ops.ne)),
    "≤": attrdict(arity=2, dw=0, dx=0, call=boolify(ops.ge)),
    "≥": attrdict(arity=2, dw=0, dx=0, call=boolify(ops.le)),
    "≡": attrdict(arity=2, call=lambda w, x: int(w == x)),
    "≢": attrdict(arity=2, call=lambda w, x: int(w != x)),
    "≈": attrdict(
        arity=2,
        call=lambda w, x: (
            len(w) == len(x) if type(w) == type(x) == list else abs(w - x) <= 1
        ),
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
        call=lambda w, x: mp.sqrt(sum(map(lambda i: i * i, map(ops.sub, w, x)))),
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

atoms |= {  # diagraphs
    "Ø+": attrdict(arity=0, call=lambda: [1, -1]),
    "Ø-": attrdict(arity=0, call=lambda: [-1, 1]),
    "Ø⁰": attrdict(arity=0, call=lambda: [1, 2, 3]),
    "Ø¹": attrdict(arity=0, call=lambda: [[0, 1], [1, 0]]),
    "Ø²": attrdict(arity=0, call=lambda: 2**32),
    "Ø³": attrdict(arity=0, call=lambda: [1, 2]),
    "Ø⁴": attrdict(arity=0, call=lambda: 2**64),
    "Ø⁵": attrdict(arity=0, call=lambda: 128),
    "Ø⁶": attrdict(arity=0, call=lambda: 512),
    "Ø⁷": attrdict(arity=0, call=lambda: 1024),
    "Ø⁸": attrdict(arity=0, call=lambda: 2048),
    "Ø⁹": attrdict(arity=0, call=lambda: 65536),
    "Ø0": attrdict(arity=0, call=lambda: [0, 0]),
    "Ø1": attrdict(arity=0, call=lambda: [1, 1]),
    "Ø2": attrdict(arity=0, call=lambda: [2, 2]),
    "ØO": attrdict(arity=0, call=lambda: [0, 1]),
    "ØZ": attrdict(arity=0, call=lambda: [1, 0]),
    "Ød": attrdict(arity=0, call=lambda: [[0, 1], [1, 0], [0, -1], [-1, 0]]),
    "Øx": attrdict(
        arity=0,
        call=lambda: [
            [1, 1],
            [1, 0],
            [1, -1],
            [0, 1],
            [0, 0],
            [0, -1],
            [-1, 1],
            [-1, 0],
            [-1, -1],
        ],
    ),
    "Øτ": attrdict(arity=0, call=lambda: 360),
    "Ø(": attrdict(arity=0, call=lambda: "()"),
    "Ø[": attrdict(arity=0, call=lambda: "[]"),
    "Ø/": attrdict(arity=0, call=lambda: "/\\"),
    "Ø<": attrdict(arity=0, call=lambda: "<>"),
    "Ø{": attrdict(arity=0, call=lambda: "{}"),
    "ØA": attrdict(arity=0, call=lambda: string.ascii_uppercase),
    "Øa": attrdict(arity=0, call=lambda: string.ascii_lowercase),
    "ØB": attrdict(arity=0, call=lambda: re.sub("[AEIOU]", "", string.ascii_uppercase)),
    "Øb": attrdict(arity=0, call=lambda: re.sub("[aeiou]", "", string.ascii_lowercase)),
    "ØV": attrdict(arity=0, call=lambda: "AEIOU"),
    "Øv": attrdict(arity=0, call=lambda: "aeiou"),
    "ØY": attrdict(arity=0, call=lambda: "AEIOUY"),
    "Øy": attrdict(arity=0, call=lambda: "aeiouy"),
    "ØD": attrdict(arity=0, call=lambda: string.digits),
    "ØX": attrdict(arity=0, call=lambda: string.hexdigits),
    "ØO": attrdict(arity=0, call=lambda: string.octdigits),
    "Øα": attrdict(arity=0, call=lambda: string.ascii_letters),
    "ØW": attrdict(arity=0, call=lambda: string.digits + string.ascii_letters + "_"),
    "Øc": attrdict(arity=0, call=lambda: codepage),
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
    "Æp": attrdict(
        arity=1, dx=0, call=lambda x: len([i for i in range(x + 1) if mp.isprime(i)])
    ),
    "Æn": attrdict(arity=1, dx=0, call=nprimes),
    "ÆF": attrdict(arity=1, dx=0, call=prime_factors),
    "ÆL": attrdict(arity=1, dx=0, call=mp.ln),
    "ÆĿ": attrdict(arity=1, dx=0, call=mp.exp),
    "Æŀ": attrdict(arity=1, dx=0, call=lambda x: mp.binomial(2 * x, x) / (x + 1)),
    "Æl": attrdict(arity=1, dx=0, call=lucas),
    "Æf": attrdict(arity=1, dx=0, call=fibonacci),
    "Æτ": attrdict(arity=1, dx=2, call=lambda x: sum(diagonal_leading(x))),
    "Æ²": attrdict(arity=1, dx=0, call=lambda x: int(int(mp.sqrt(x)) == mp.sqrt(x))),
    "ÆA": attrdict(arity=1, dx=2, call=lambda x: diagonals(x, antidiagonals=True)),
    "ÆȦ": attrdict(arity=1, dx=2, call=diagonals),
    "ÆD": attrdict(arity=1, dx=2, call=lambda x: mp.det(mp.matrix(ensure_square(x)))),
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
    "æ∘": attrdict(
        arity=2,
        dw=1,
        dx=1,
        call=lambda w, x: sum(
            map(lambda i: i[0] * i[1], zip_longest(w, x, fillvalue=1))
        ),
    ),
    "æ×": attrdict(
        arity=2, dw=2, dx=2, call=lambda w, x: (mp.matrix(w) @ mp.matrix(x)).tolist()
    ),
    "æ*": attrdict(arity=2, dw=2, dx=0, call=lambda w, x: (mp.matrix(w) ** x).tolist()),
    "æṫ": attrdict(arity=2, dw=0, dx=0, call=mp.atan2),
    "æc": attrdict(arity=2, dw=1, dx=1, call=convolve),
    "æi": attrdict(arity=2, dw=0, dx=0, call=mpc),
    "æ«": attrdict(arity=2, dw=0, dx=0, call=ops.lshift),
    "æ»": attrdict(arity=2, dw=0, dx=0, call=ops.rshift),
    "ær": attrdict(arity=2, dw=0, dx=0, call=lambda w, x: round(x, w)),
    "æl": attrdict(
        arity=2, dw=1, dx=1, call=lambda w, x: list(statistics.linear_regression(w, x))
    ),
    "ŒB": attrdict(arity=1, call=lambda x: iterable(x) + iterable(x)[::-1]),
    "ŒP": attrdict(arity=1, dx=1, call=lambda x: x == x[::-1]),
    "ŒĠ": attrdict(arity=1, dx=1, call=get_req),
    "ŒE": attrdict(arity=1, call=enumerate_md),
    "ŒG": attrdict(arity=1, call=lambda x: group_indicies(x, md=True)),
    "ŒM": attrdict(arity=1, call=maximal_indicies_md),
    "ŒṪ": attrdict(arity=1, call=lambda x: [i for i, e in enumerate_md(x) if e]),
    "Œ1": attrdict(arity=1, call=ones),
    "Œp": attrdict(
        arity=1, call=lambda x: list(map(list, mit.distinct_combinations(x, 2)))
    ),
    "Œe": attrdict(arity=1, dx=1, call=rle),
    "Œd": attrdict(arity=1, dx=2, call=rld),
    "Œ$": attrdict(
        arity=1, dx=0, call=lambda x: x ^ 32 if chr(x) in string.ascii_letters else x
    ),
    "ŒU": attrdict(
        arity=1, dx=0, call=lambda x: x ^ 32 if chr(x) in string.ascii_lowercase else x
    ),
    "ŒL": attrdict(
        arity=1, dx=0, call=lambda x: x ^ 32 if chr(x) in string.ascii_uppercase else x
    ),
    "ŒD": attrdict(arity=1, call=depth),
    # "ŒṖ": attrdict(arity=1, call=),
    "Œb": attrdict(arity=1, call=to_braille),
    "ŒJ": attrdict(arity=1, call=json_decode),
    "ŒF": attrdict(arity=1, dx=1, call=lambda x: open(str(x), encoding="utf-8").read()),
    "œF": attrdict(
        arity=2,
        dx=1,
        call=lambda w, x: open(str(x), "w+", encoding="utf-8").write(str(w)),
    ),
    "œi": attrdict(arity=2, call=index_into_md),
    # "œs": attrdict(arity=2, call=),
    # "œŀ": attrdict(arity=2, call=),
    # "œl": attrdict(arity=2, call=),
    # "œt": attrdict(arity=2, call=),
    # "œo": attrdict(arity=2, call=),
    "œm": attrdict(arity=2, call=mapval),
    # "œr": attrdict(arity=2, call=),
}
