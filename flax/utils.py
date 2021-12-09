from itertools import zip_longest
from collections import deque

zip = lambda *x: [[*x] for x in zip_longest(*x, fillvalue=0)]

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

def pp(x):
    x = repr(x)
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
