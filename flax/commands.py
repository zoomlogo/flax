import sys
import math as M

from flax.utils import *

class state:
    reg = 0
    accumulator = 0

global_state = state()

class atom:
    def __init__(self, arity, call):
        self.arity = arity
        self.call = call

commands = {
    # Single byte nilads
    'Ŧ': atom(0, lambda: 10),
    '³': atom(0, lambda: sys.argv[1] if len(sys.argv) > 1 else 16),
    '⁴': atom(0, lambda: sys.argv[2] if len(sys.argv) > 2 else 32),
    '⁵': atom(0, lambda: sys.argv[2] if len(sys.argv) > 3 else 64),
    '⁰': atom(0, lambda: 100),
    'ƀ': atom(0, lambda: [0, 1]),
    '®': atom(0, lambda: global_state.reg),
    'я': atom(0, lambda: sys.stdin.read(1)),
    'д': atom(0, lambda: input()),

    # Single byte monads
    '!': atom(1, lambda x: vectorise(M.factorial, x)),
}
