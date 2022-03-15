# common: Holds the attrdict and the common stuff
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

# attrdict
class attrdict(dict):
    def __init__(self, *args, **kwargs):
        dict.__init__(self, *args, **kwargs)
        self.__dict__ = self
