import sys
import math as M
import random as R
import itertools
from collections import deque

# ---------
#   Atoms
# ---------

class attrdict:
    def __init__(self, *args, **kwargs):
        dict.__init__(self, *args, **kwargs)
        self.__dict__ = self

def depth(x):
    if not isinstance(x, list):
        return 0

    if not x:
        return 1

    return max(map(depth, x)) + 1

def reduce_first(fn, L):
    if not isinstance(L, list):
        return L

    res = L[0]
    for x in L[1:]:
        res = fn(res, x)
    return res

def reduce(fn, L):
    if not isinstance(fn, L):
        return L

    if isinstance(L[0], list):
        return [reduce(fn, x) for x in L]

    return reduce_first(fn, L)

def flatten(L):
    def gen(l):
        if not isinstance(l, list):
            yield l
        else:
            for i in l:
                yield from flatten(i)
    return [*gen(L)]

def reshape(shape, L):
    if not isinstance(L, list):
        return L

    if not isinstance(L, deque):
        L = deque(L)

    def nxt(dq):
        x = dq.popleft()
        dq.append(x)
        return x

    if len(shape) == 1:
        return [nxt(L) for _ in range(shape[0])]
    else:
        return [reshape(shape[1:], L) for _ in range(shape[0])]

def vectorise(fn, x):
    if depth(x) != 0:
        return [vectorise(fn, a) for a in x]
    else:
        return fn(x)

def dyadic_vectorise(fn, x, y):
    dx = depth(x)
    dy = depth(y)

    if dx == dy:
        if dx != 0:
            return [dyadic_vectorise(fn, a, b) for a, b in zip(x, y)]
        else:
            return fn(x, y)
    else:
        if dx < dy:
            return [dyadic_vectorise(fn, x, b) for b in y]
        else:
            return [dyadic_vectorise(fn, a, y) for a in x]

def iterable(x, make_range=False, make_digits=False):
    if not isinstance(x, list):
        if make_range:
            return [*range(1, x+1)]
        if make_digits:
            return [-int(i) if x < 0 else int(i)
                        for i in str(x)[
                            1 if x < 0 else 0:]]
        return [x]
    return x

def pp(x):
    x = repr(x) \
        .replace('[]', '⍬') \
        .replace('-', '¯') \
        .replace('j', 'i')

    i = 0
    indent = 0

    while i < len(x):
        if x[i] == '[':
            if x[i - 1] == ' ':
                print(end='\n' + ' ' * indent + '[')
            else:
                print(end='[')
            indent += 1
        elif x[i] == ']':
            print(end=']')
            indent -= 1
        elif x[i] != ' ':
            print(end=x[i])
        i += 1
    print()

    return x

tpp = lambda x:print(repr(x).replace(', ',' ').replace('[]','⍬').replace('-','¯').replace('j','i'))

zip = lambda *x: [[*x] for x in itertools.zip_longest(*x, fillvalue=0)]

def to_bin(x):
    return [-i if x < 0 else i
        for i in map(int, bin(x)[
            3 if x < 0 else 2:])]

def to_digits(x):
    return [-int(i) if x < 0 else int(i)
        for i in str(x)[
            1 if x < 0 else 0:]]

def truthy_indices(x):
    if not isinstance(x, list):
        return []

    i = 0
    indices = []
    while i < len(x):
        if x[i]:
            indices.append(i)
        i += 1
    return indices

def falsey_indices(x):
    if not isinstance(x, list):
        return []

    i = 0
    indices = []
    while i < len(x):
        if not x[i]:
            indices.append(i)
        i += 1
    return indices

def random(x):
    x = iterable(x)
    return R.choice(x)

def from_bin(x):
    x = iterable(x)
    sign = -1 if sum(x) < 0 else 1
    num = 0
    i = 0
    for b in x[::-1]:
        num += abs(b) * 2 ** i
        i += 1
    return num * sign

def contains_false(l):
    if not isinstance(l, list):
        return 1 if l else 0

    if len(l) == 0:
        return 1   

    return 1 if 1 in [contains_false(x) for x in l] else 0

def from_digits(x):
    x = iterable(x)
    sign = -1 if sum(x) < 0 else 1
    num = 0
    i = 0
    for b in x[::-1]:
        num += abs(b) * 10 ** i
        i += 1
    return num * sign

def split(x, c):
    if not isinstance(x, list):
        return x

    res = []
    tmp = []

    for i in range(len(x)):
        if not (i % c) and tmp:
            res.append(tmp)
            tmp = []
        tmp.append(x[i])

    if tmp:
        res.append(tmp)
    return res

def sub_lists(l):
    lists = [[]]
    for i in range(len(l) + 1):
        for j in range(i):
            lists.append(l[j:i])
    return lists

def reverse_every_other(x):
    x = iterable(x)

    for i in range(len(x)):
        if i % 2 == 0:
            x[i] = iterable(x[i])[::-1]
    return x

def grade_up(x):
    x = iterable(x, make_digits=True)

    grades = []
    for a in list(sorted(x)):
        grades.append(find_all_indices(a, x))
    return flatten(grades)

def grade_down(x):
    x = iterable(x, make_digits=True)

    grades = []
    for a in list(sorted(x))[::-1]:
        grades.append(find_all_indices(a, x))
    return flatten(grades)

def group_equal(x):
    res = []
    for e in x:
        if res and res[-1][0] == e:
            res[-1].append(e)
        else:
            res.append([e])
    return res

def group(x):
    res = {}
    for i, it in enumerate(x):
        it = repr(it)
        if it in res:
            res[it].append(i + 1)
        else:
            res[it] = [i + 1]
    return [res[k] for k in sorted(res, key=eval)]


def divisors(x):
    res = []

    i = 1
    while i <= x:
        if x % i == 0:
            res.append(i)
        i += 1

def join_spaces(x):
    return laminate(x, 32)

def join_newlines(x):
    return laminate(x, 10)

def prefixes(x):
    res = []
    for i in range(len(x)):
        res.append(x[:i + 1])
    return res

def suffixes(x):
    res = []
    for i in range(len(x)):
        res.append(x[i:])
    return res

def laminate(x, y):
    res = [y] * (len(x) * 2 - 1)
    res[0::2] = iterable(x)
    return x

def find_all_indices(x, y):
    res = []
    i = 0
    while i < len(y):
        if y[i] == x:
            res.append(i)
        i += 1
    return res

def index(x, y):
    if not isinstance(x, list):
        return x

    if isinstance(y, int):
        return x[(y - 1) % len(x)]
    return [index(x, M.floor(y)), index(x, M.ceil(y))]

def split_at_occurences(x, y):
    res = []
    tmp = []

    for e in x:
        if e == y:
            res.append(tmp)
            tmp = []
        else:
            tmp.append(e)

    if tmp:
        res.append(tmp)

    return res

def mold(x, y):
    for i in range(len(y)):
        if isinstance(y[i], list):
            mold(x, y[i])
        else:
            item = x.pop(0)
            y[i] = item
            x.append(item)
    return y

def diagonals(x):
    d = [[] for _ in range(len(x) + len(x[0]) - 1)]
    min_d = -len(x) + 1

    for i in range(len(x[0])):
        for j in range(len(x)):
            d[i - j - min_d].append(x[j][i])
    return d

atoms = {
    # Single byte nilads
    'Ŧ': attrdict(arity=0, call=lambda: 10),
    '³': attrdict(arity=0, call=lambda: sys.argv[1] if len(sys.argv) > 1 else 16),
    '⁴': attrdict(arity=0, call=lambda: sys.argv[2] if len(sys.argv) > 2 else 32),
    '⁵': attrdict(arity=0, call=lambda: sys.argv[2] if len(sys.argv) > 3 else 64),
    '⁰': attrdict(arity=0, call=lambda: 100),
    'ƀ': attrdict(arity=0, call=lambda: [0, 1]),
    '®': attrdict(arity=0, call=lambda: 0),
    'я': attrdict(arity=0, call=lambda: sys.stdin.read(1)),
    'д': attrdict(arity=0, call=lambda: input()),
    '⍺': attrdict(arity=0, call=lambda: 0),
    '⍵': attrdict(arity=0, call=lambda: 0),

    # Single byte monads
    '!': attrdict(arity=1, call=lambda x: vectorise(M.factorial, x)),
    '¬': attrdict(arity=1, call=lambda x: vectorise(lambda a: 1 if not a else 0, x)),
    '~': attrdict(arity=1, call=lambda x: vectorise(lambda a: ~a, x)),
    'B': attrdict(arity=1, call=lambda x: vectorise(to_bin, x)),
    'D': attrdict(arity=1, call=lambda x: vectorise(to_digits, x)),
    'C': attrdict(arity=1, call=lambda x: vectorise(lambda a: 1 - a, x)),
    'F': attrdict(arity=1, call=flatten),
    'H': attrdict(arity=1, call=lambda x: vectorise(lambda a: a / 2, x)),
    'L': attrdict(arity=1, call=len),
    'N': attrdict(arity=1, call=lambda x: vectorise(lambda a: -a, x)),
    'Ř': attrdict(arity=1, call=lambda x: [*range(len(x))]),
    'Π': attrdict(arity=1, call=lambda x: reduce(lambda a, b: a * b, flatten(x))),
    'Σ': attrdict(arity=1, call=lambda x: sum(flatten(x))),
    '⍳': attrdict(arity=1, call=lambda x: vectorise(lambda a: [*range(1, a + 1)], x)),
    '⊤': attrdict(arity=1, call=truthy_indices),
    '⊥': attrdict(arity=1, call=falsey_indices),
    'R': attrdict(arity=1, call=lambda x: iterable(x, make_range=True)[::-1]),
    'W': attrdict(arity=1, call=lambda x: [x]),
    'Ŕ': attrdict(arity=1, call=random),
    'T': attrdict(arity=1, call=lambda x: zip(*x)),
    '¹': attrdict(arity=1, call=lambda x: x),
    '²': attrdict(arity=1, call=lambda x: vectorise(lambda a: a ** 2, x)),
    '√': attrdict(arity=1, call=lambda x: vectorise(lambda a: a ** (1 / 2), x)),
    'Ḃ': attrdict(arity=1, call=from_bin),
    'Ă': attrdict(arity=1, call=contains_false),
    'Ḋ': attrdict(arity=1, call=from_digits),
    'Ð': attrdict(arity=1, call=lambda x: vectorise(lambda a: a * 2, x)),
    '₃': attrdict(arity=1, call=lambda x: vectorise(lambda a: a * 3, x)),
    'E': attrdict(arity=1, call=lambda x: vectorise(lambda a: [*range(a)], x)),
    'G': attrdict(arity=1, call=lambda x: group_equal(iterable(x, make_digits=True))),
    '∇': attrdict(arity=1, call=lambda x: min(iterable(x))),
    '∆': attrdict(arity=1, call=lambda x: max(iterable(x))),
    'S': attrdict(arity=1, call=lambda x: [*sorted(x)]),
    'Ṡ': attrdict(arity=1, call=lambda x: [*sorted(x)][::-1]),
    'ᵇ': attrdict(arity=1, call=lambda x: vectorise(lambda a: a % 2, x)),
    'Ḣ': attrdict(arity=1, call=lambda x: iterable(x, make_digits=True)[1:]),
    'Ṫ': attrdict(arity=1, call=lambda x: iterable(x, make_digits=True)[:-2]),
    'Ḥ': attrdict(arity=1, call=lambda x: iterable(x, make_digits=True)[0]),
    'Ṭ': attrdict(arity=1, call=lambda x: iterable(x, make_digits=True)[-1]),
    '±': attrdict(arity=1, call=lambda x: vectorise(lambda a: -1 if a < 0 else (0 if a == 0 else 1), x)),
    'Θ': attrdict(arity=1, call=lambda x: iterable(x, make_range=True).insert(0, 0)),
    'U': attrdict(arity=1, call=lambda x: list(set(iterable(x)))),
    '⤒': attrdict(arity=1, call=lambda x: vectorise(lambda a: a + 1, x)),
    '⤓': attrdict(arity=1, call=lambda x: vectorise(lambda a: a - 1, x)),
    'P': attrdict(arity=1, call=lambda x: pp(x)),
    'Ċ': attrdict(arity=1, call=lambda x: print(end=''.join(chr(c) for c in x)) or x),
    'Ç': attrdict(arity=1, call=lambda x: split(x, 2)),
    'X': attrdict(arity=1, call=lambda x: split(x, int(len(x) / 2))),
    'Ƥ': attrdict(arity=1, call=lambda x: [*itertools.permutations(x)]),
    'ε': attrdict(arity=1, call=lambda x: sub_lists(iterable(x, make_range=True))),
    'σ': attrdict(arity=1, call=reverse_every_other),
    'Ḅ': attrdict(arity=1, call=lambda x: vectorise(lambda a: 2 ** a, x)),
    'Ď': attrdict(arity=1, call=depth),
    '⍋': attrdict(arity=1, call=grade_up),
    '⍒': attrdict(arity=1, call=grade_down),
    '⅟': attrdict(arity=1, call=lambda x: vectorise(lambda a: 1 / a, x)),
    '⌈': attrdict(arity=1, call=lambda x: vectorise(lambda a: M.ceil(a), x)),
    '⌊': attrdict(arity=1, call=lambda x: vectorise(lambda a: M.floor(a), x)),
    'A': attrdict(arity=1, call=lambda x: vectorise(lambda a: abs(a), x)),
    'Ḍ': attrdict(arity=1, call=lambda x: vectorise(divisors, x)),
    'J': attrdict(arity=1, call=join_spaces),
    'Ĵ': attrdict(arity=1, call=join_newlines),
    'V': attrdict(arity=1, call=lambda x: group(iterable(x, make_digits=True))),
    '⊢': attrdict(arity=1, call=prefixes),
    '⊣': attrdict(arity=1, call=suffixes),
    '∀': attrdict(arity=1, call=lambda x: [sum(r) for r in iterable(x)]),

    # Single byte dyads
    '+': attrdict(arity=2, call=lambda x, y: dyadic_vectorise(lambda a, b: a + b, x, y)),
    '-': attrdict(arity=2, call=lambda x, y: dyadic_vectorise(lambda a, b: a - b, x, y)),
    '×': attrdict(arity=2, call=lambda x, y: dyadic_vectorise(lambda a, b: a * b, x, y)),
    '÷': attrdict(arity=2, call=lambda x, y: dyadic_vectorise(lambda a, b: a / b, x, y)),
    '%': attrdict(arity=2, call=lambda x, y: dyadic_vectorise(lambda a, b: a % b, x, y)),
    '*': attrdict(arity=2, call=lambda x, y: dyadic_vectorise(lambda a, b: a ** b, x, y)),
    '"': attrdict(arity=2, call=lambda x, y: [x, y]),
    ',': attrdict(arity=2, call=lambda x, y: laminate(x, y)),
    '<': attrdict(arity=2, call=lambda x, y: dyadic_vectorise(lambda a, b: 1 if a < b else 0, x, y)),
    '>': attrdict(arity=2, call=lambda x, y: dyadic_vectorise(lambda a, b: 1 if a > b else 0, x, y)),
    '=': attrdict(arity=2, call=lambda x, y: dyadic_vectorise(lambda a, b: 1 if a == b else 0, x, y)),
    '≠': attrdict(arity=2, call=lambda x, y: dyadic_vectorise(lambda a, b: 1 if a != b else 0, x, y)),
    '≥': attrdict(arity=2, call=lambda x, y: dyadic_vectorise(lambda a, b: 1 if a >= b else 0, x, y)),
    '≤': attrdict(arity=2, call=lambda x, y: dyadic_vectorise(lambda a, b: 1 if a <= b else 0, x, y)),
    '≡': attrdict(arity=2, call=lambda x, y: 1 if x == y else 0),
    '≢': attrdict(arity=2, call=lambda x, y: 1 if x != y else 0),
    '∧': attrdict(arity=2, call=lambda x, y: dyadic_vectorise(lambda a, b: 1 if a and b else 0, x, y)),
    '∨': attrdict(arity=2, call=lambda x, y: dyadic_vectorise(lambda a, b: 1 if a or b else 0, x, y)),
    '&': attrdict(arity=2, call=lambda x, y: dyadic_vectorise(lambda a, b: a & b, x, y)),
    '|': attrdict(arity=2, call=lambda x, y: dyadic_vectorise(lambda a, b: a | b, x, y)),
    '^': attrdict(arity=2, call=lambda x, y: dyadic_vectorise(lambda a, b: a ^ b, x, y)),
    '∊': attrdict(arity=2, call=lambda x, y: x in y),
    'f': attrdict(arity=2, call=lambda x, y: [a for a in x if a not in y]),
    'ḟ': attrdict(arity=2, call=lambda x, y: [a for a in x if a in y]),
    '⊂': attrdict(arity=2, call=lambda x, y: x.find(y) + 1),
    '⊆': attrdict(arity=2, call=lambda x, y: vectorise(lambda a: a + 1, find_all_indices(x, y))),
    '⊏': attrdict(arity=2, call=lambda x, y: [x[i] for i in range(len(x)) if i % y == 0]),
    '·': attrdict(arity=2, call=lambda x, y: [*itertools.product(x, y)]),
    'r': attrdict(arity=2, call=lambda x, y: dyadic_vectorise(lambda a, b: [*range(a, b + 1)], x, y)),
    's': attrdict(arity=2, call=split),
    '\\': attrdict(arity=2,call= lambda x, y: [iterable(x) for _ in range(y)]),
    'i': attrdict(arity=2, call=index),
    'o': attrdict(arity=2, call=split_at_occurences),
    'a': attrdict(arity=2, call=lambda x, y: iterable(x) + iterable(y)),
    'p': attrdict(arity=2, call=lambda x, y: iterable(y) + iterable(x)),
    'c': attrdict(arity=2, call=lambda x, y: iterable(x, make_digits=True).count(y)),
    'm': attrdict(arity=2, call=lambda x, y: mold(iterable(x), iterable(y))),
    'h': attrdict(arity=2, call=lambda x, y: iterable(x, make_digits=True)[:y]),
    't': attrdict(arity=2, call=lambda x, y: iterable(x, make_digits=True)[y - 1:]),
    'z': attrdict(arity=2, call=zip),
    'u': attrdict(arity=2, call=lambda x, y: [y.find(v) + 1 for v in x]),
    '#': attrdict(arity=2, call=reshape),
    'ḍ': attrdict(arity=2, call=lambda x, y: dyadic_vectorise(lambda a, b: 1 if a % b == 0 else 0, x, y)),

    # Niladic diagraphs
    'Øp': attrdict(arity=0, call=lambda: M.pi),
    'Øe': attrdict(arity=0, call=lambda: M.e),
    'ØP': attrdict(arity=0, call=lambda: 1.618033988749895),
    'Ø∞': attrdict(arity=0, call=lambda: float('inf')),
    'ØA': attrdict(arity=0, call=lambda: 26),
    'Ø₁': attrdict(arity=0, call=lambda: 128),
    'Ø₂': attrdict(arity=0, call=lambda: 256),
    'Ø₀': attrdict(arity=0, call=lambda: 1000),

    # Monadic diagraphs
    'ŒD': attrdict(arity=1, call=diagonals),
    'ŒS': attrdict(arity=1, call=lambda x: vectorise(M.sin, x)),
    'ŒC': attrdict(arity=1, call=lambda x: vectorise(M.cos, x)),
    'ŒT': attrdict(arity=1, call=lambda x: vectorise(M.tan, x)),
    'ŒṠ': attrdict(arity=1, call=lambda x: vectorise(M.asin, x)),
    'ŒĊ': attrdict(arity=1, call=lambda x: vectorise(M.acos, x)),
    'ŒṪ': attrdict(arity=1, call=lambda x: vectorise(M.atan, x)),
    'Œc': attrdict(arity=1, call=lambda x: vectorise(lambda a: 1 / M.sin(a), x)),
    'Œs': attrdict(arity=1, call=lambda x: vectorise(lambda a: 1 / M.cos(a), x)),
    'Œt': attrdict(arity=1, call=lambda x: vectorise(lambda a: 1 / M.tan(a), x)),
    'Œn': attrdict(arity=1, call=lambda x: vectorise(M.sinh, x)),
    'Œo': attrdict(arity=1, call=lambda x: vectorise(M.cosh, x)),
    'Œh': attrdict(arity=1, call=lambda x: vectorise(M.tanh, x)),

    # Dyadic diagraphs
    'œl': attrdict(arity=2, call=lambda x, y: dyadic_vectorise(lambda a, b: a << b, x, y)),
    'œr': attrdict(arity=2, call=lambda x, y: dyadic_vectorise(lambda a, b: a >> b, x, y)),
    'œ*': attrdict(arity=2, call=lambda x, y: [*itertools.product(x, repeat=y)]),
    'œ·': attrdict(arity=2, call=lambda x, y: sum(x[i][0] * y[i] for i in range(len(y)))),
}

# =====================
#     Implementation
#       of Chains
# =====================

def arities(links):
    return [link.arity for link in links]

def leading_nilad(chain):
    return chain and arities(chain) + [1] < [0, 2] * len(chain)

def niladic_chain(chain):
    if not chain or chain[0].arity > 0:
        return monadic_chain(chain, 0)
    return monadic_chain(chain[1:], chain[0].call())

def monadic_chain(chain, x):
    atoms['⍺'].call = lambda: x

    init = False

    accumulator = x

    while 1:
        if init:
            for link in chain:
                if link.arity < 0:
                    link.arity = 1

            if leading_nilad(chain):
                accumulator = chain[0].call()
                chain = chain[1:]

        if not chain:
            break

        if arities(chain[0:2]) == [2, 1]:
            accumulator = chain[0].call(accumulator, chain[1].call(x))
            chain = chain[2:]
        elif arities(chain[0:2]) == [2, 0]:
            accumulator = chain[0].call(accumulator, chain[1].call())
            chain = chain[2:]
        elif arities(chain[0:2]) == [0, 2]:
            accumulator = chain[1].call(chain[0].call(), accumulator)
            chain = chain[2:]
        elif chain[0].arity == 2:
            accumulator = chain[0].call(accumulator, x)
            chain = chain[1:]
        elif chain[0].arity == 1:
            if not chain[1:] and hasattr(chain[0], 'chain'):
                x = accumulator
                chain = chain[0].chain
                init = True
            else:
                accumulator = chain[0].call(accumulator)
                chain = chain[1:]
        else:
            pp(accumulator)
            accumulator = chain[0].call()
            chain = chain[1:]

    return accumulator

def dyadic_chain(chain, x, y):
    atoms['⍺'].call = lambda: x
    atoms['⍵'].call = lambda: y

    for link in chain:
        if link.arity < 0:
            link.arity = 2

    if chain and arities(chain[0:3]) == [2, 2, 2]:
        accumulator = chain[0].call(x, y)
        chain = chain[1:]
    elif leading_nilad(chain):
        accumulator = chain[0].call()
        chain = chain[1:]
    else:
        accumulator = x

    while chain:
        if arities(chain[0:3]) == [2, 2, 0] and leading_nilad(chain[2:]):
            accumulator = chain[1].call(chain[0].call(accumulator, y), chain[2].call())
            chain = chain[3:]
        elif arities(chain[0:2]) == [2, 2]:
            accumulator = chain[0].call(accumulator, chain[1].call(x, y))
            chain = chain[2:]
        elif arities(chain[0:2]) == [2, 0]:
            accumulator = chain[0].call(accumulator, chain[1].call())
            chain = chain[2:]
        elif arities(chain[0:2]) == [0, 2]:
            accumulator = chain[1].call(chain[0].call(), accumulator)
            chain = chain[2:]
        elif chain[0].arity == 2:
            accumulator = chain[0].call(accumulator, y)
            chain = chain[1:]
        elif chain[0].arity == 1:
            accumulator = chain[0].call(accumulator)
            chain = chain[1:]
        else:
            pp(accumulator)
            accumulator = chain[0].call()
            chain = chain[1:]

    return accumulator

def variadic_link(link, *args):
    if link.arity < 0:
        args = [*filter(None.__ne__, args)]
        link.arity = len(args)
    
    if link.arity == 0:
        return link.call()
    else:
        return link.call(*args)

def copy_to(atom, value):
    atom.call = lambda: value
    return value

# =====================
#        Parser
# =====================