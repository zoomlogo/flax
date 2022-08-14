# common: Holds the attrdict and the common stuff
import more_itertools
from mpmath import mp

__all__ = [
    "attrdict",
    "flax_indent",
    "flax_print",
    "flax_string",
    "TOKEN_TYPE",
    "inf",
    "mp",
    "mpc",
    "mpf",
]

# Flags:
DEBUG = False
PRINT_CHARS = False
DISABLE_GRID = False

# Set mp context defaults
mp.dps = 20  # 20 by default, set to anything via flag
mp.pretty = True

# attrdict
class attrdict(dict):
    def __init__(self, *args, **kwargs):
        dict.__init__(self, *args, **kwargs)
        self.__dict__ = self


# lexer
class TOKEN_TYPE(enum.Enum):
    NUMBER = 1
    STRING = 2
    TRAIN_SEPARATOR = 3
    ATOM = 4
    QUICK = 5
    NEWLINE = 6
    LIST = 7


# helpful
mpf = mp.mpf
mpc = mp.mpc
inf = mp.inf

# flax functions
def flax_indent(x):
    # flax_indent: indent x
    res = ""
    level = 0
    for i in range(len(x)):
        if x[i] == "(":
            if i != 0 and x[i - 1] == " ":
                res += "\n" + " " * level + "("
            else:
                res += "("
            level += 1
        elif x[i] == ")":
            res += ")"
            level -= 1
        else:
            res += x[i]
    return res


def flax_string(x):
    # flax_string: convert x into flax representation
    if type(x) != list:
        if type(x) == mpc:
            return "j".join([flax_string(x.real), flax_string(x.imag)])
        elif type(x) == int or (x != inf and int(x) == x):
            return str(int(x)).replace("-", "¯").replace("inf", "∞")
        else:
            return str(x).replace("-", "¯").replace("inf", "∞")
    else:
        return "(" + " ".join(flax_string(e) for e in x) + ")"


def _flax_print_flatten(x):
    # _flax_print_flatten: flatten x and join by newlines
    try:
        if type(x[0]) == list:
            return "\n".join([_flax_print_flatten(i) for i in x])
        else:
            return "".join([chr(int(i)) for i in x])
    except TypeError:
        return chr(int(x))


def flax_print(x):
    # flax_print: print x using formatting
    if PRINT_CHARS:
        print(end=_flax_print_flatten(x))
    else:
        s = flax_string(x)
        print(s if DISABLE_GRID else flax_indent(s))
    return x
