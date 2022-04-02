# common: Holds the attrdict and the common stuff
import itertools
import more_itertools
from mpmath import mp

# Flags:
DEBUG = False
PRINT_CHARS = False
DISABLE_GRID = False
INF_MORE = False

# Set mp context defaults
mp.dps = 20  # 20 by default, sets to 100 by flag
mp.pretty = True

# attrdict
class attrdict(dict):
    def __init__(self, *args, **kwargs):
        dict.__init__(self, *args, **kwargs)
        self.__dict__ = self


# infinite list
class ilist:
    def __init__(self, l):
        self.list = l

    def __next__(self):
        return next(self.list)

    @staticmethod
    def positive_integers():
        def wrapper():
            i = 1
            while True:
                yield i
                i += 1

        return ilist(wrapper())

    @staticmethod
    def primes():
        def wrapper():
            i = 2
            while True:
                if mp.isprime(i):
                    yield i
                i += 1

        return ilist(wrapper())

    @staticmethod
    def to_ilist(x):
        return ilist(itertools.cycle([x] if not is_list(x) else x))


# helpful
mpf = mp.mpf
mpc = mp.mpc
inf = mp.inf


def is_list(x):
    # is_list: is a list or infinite list?
    return type(x) in [list, ilist]


# flax functions
def flax_indent(x):
    # flax_indent: indent x
    res = ""
    level = 0
    for i in range(len(x)):
        if x[i] == "[":
            if i != 0 and x[i - 1] == ",":
                res += "\n" + " " * level + "["
            else:
                res += "["
            level += 1
        elif x[i] == "]":
            res += "]"
            level -= 1
        else:
            res += x[i]
    return res


def flax_string(x):
    # flax_string: convert x into flax representation
    if not is_list(x):
        if type(x) == mpc:
            return "j".join([flax_string(x.real), flax_string(x.imag)])
        elif type(x) == int or type(x) == mpf and x != inf and int(x) == x:
            return str(int(x)).replace("-", "¯").replace("inf", "∞")
        else:
            return str(x).replace("-", "¯").replace("inf", "∞")
    else:
        if type(x) == ilist:
            return (
                "["
                + ",".join(flax_string(next(x)) for _ in range(32 if INF_MORE else 10))
                + "...]"
            )
        else:
            return "[" + ",".join(flax_string(e) for e in x) + "]"


def flax_print(x):
    # flax_print: print x using formatting
    if PRINT_CHARS:
        print(end="".join(chr(c) for c in more_itertools.collapse(x)))
    else:
        s = flax_string(x)
        print(s if DISABLE_GRID else flax_indent(s))
    return x
