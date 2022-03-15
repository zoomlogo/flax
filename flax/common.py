# common: Holds the attrdict and the common stuff
import more_itertools
from mpmath import mp

# Flags:
DEBUG = False
PRINT_CHARS = False
DISABLE_GRID = False

# Set mp context defaults
mp.dps = 20  # 20 by default, sets to 100 by flag
mp.pretty = True

# Alias
mpf = mp.mpf
mpc = mp.mpc
inf = mp.inf

# attrdict
class attrdict(dict):
    def __init__(self, *args, **kwargs):
        dict.__init__(self, *args, **kwargs)
        self.__dict__ = self


# flax functions
def flax_indent(x):
    # flax_indent: indent x
    res = ""
    level = 0
    for i in range(len(x)):
        if x[i] == "[":
            if i != 0 and x[i - 1] == " ":
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
    if type(x) != list:
        if type(x) == mpc:
            return "j".join([flax_string(x.real), flax_string(x.imag)])
        elif type(x) == int or type(x) == mpf and x != inf and int(x) == x:
            return str(int(x)).replace("-", "¯").replace("inf", "∞")
        else:
            return str(x).replace("-", "¯").replace("inf", "∞")
    else:
        res = ""
        for e in x:
            res += flax_string(e) + " "
        res = "[" + res.strip() + "]"
        return res


def flax_print(x):
    if PRINT_CHARS:
        print(end="".join(chr(c) for c in more_itertools.collapse(x)))
    else:
        s = flax_string(x)
        print(s if DISABLE_GRID else flax_indent(s))
    return x
