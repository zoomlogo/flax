# __main__: main entry point and gluing
import sys

from flax.chains import variadic_chain
from flax.common import flax_print
from flax.error import error, debug
from flax.funcs import to_chars
from flax.lexer import tokenise
from flax.parser import parse
import flax.common

__all__ = ["flax_run"]

# function for running flax
def flax_run(code, args):
    try:
        tokens = tokenise(code)
        if flax.common.DEBUG:
            debug("tokens: " + tokens)
        parsed = parse(tokens)
        if flax.common.DEBUG:
            debug("parsed: " + parsed)
            debug("main chain: " + parsed[-1])
        flax_print(variadic_chain(parsed[-1] if parsed else "", args))
    except KeyboardInterrupt:
        error("KeyboardInterrupt", 130)


sys.argv = sys.argv[1:]
read_from_file = False

# handle flags
if sys.argv:
    if "d" in sys.argv[0]:
        flax.common.DEBUG = True
    if "f" in sys.argv[0]:
        read_from_file = True
    if "C" in sys.argv[0]:
        flax.common.PRINT_CHARS = True
    if "p" in sys.argv[0]:
        flax.common.mp.dps = 100
    if "P" in sys.argv[0]:
        flax.common.DISABLE_GRID = True
    sys.argv = sys.argv[1:]

# run
if read_from_file:
    try:
        code = open(sys.argv[0], encoding="utf-8").read()
    except FileNotFoundError:
        error(f'File "{sys.argv[0]}" not found.', 66)
        exit(66)

    sys.argv = sys.argv[1:]
    args = [eval(arg) for arg in sys.argv]
    args = [to_chars(arg) if type(arg) == str else arg for arg in sys.argv]
    flax_eval(code, args)
else:
    error("'nyi")
