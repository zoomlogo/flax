# __main__: main entry point and gluing
import sys

from flax.chains import variadic_chain
from flax.encoding import *
from flax.error import error, debug
from flax.funcs import to_chars
from flax.lexer import tokenise
from flax.parser import parse
import flax.common
import flax.builtins

__all__ = ["flax_run"]

# function for running flax
def flax_run(code, args):
    tokens = tokenise(code)
    debug("tokens: " + str(tokens))
    parsed = parse(tokens)
    debug("parsed: " + str(parsed))
    # run
    # niladic chains 11 and 13
    # monadic ones only 13, dyadic nothing
    flax.builtins.atoms["₎"].call = lambda: args[-1] if len(args) > 0 else 11
    flax.builtins.atoms["₍"].call = lambda: args[0] if len(args) > 1 else 13
    flax.common.flax_print(variadic_chain(parsed[-1] if parsed else "", args))


sys.argv = sys.argv[1:]
read_from_file = False
should_encode = False
should_decode = False

# handle flags
if sys.argv:
    if "d" in sys.argv[0]:
        flax.common.DEBUG = True
    if "f" in sys.argv[0]:
        read_from_file = True
    if "C" in sys.argv[0]:
        flax.common.PRINT_CHARS = True
    if "p" in sys.argv[0]:
        flax.common.mp.dps = 64
    if "P" in sys.argv[0]:
        flax.common.DISABLE_GRID = True
    if "e" in sys.argv[0]:
        read_from_file = True
        should_encode = True
    if "D" in sys.argv[0]:
        read_from_file = True
        should_decode = True
        should_encode = False
    sys.argv = sys.argv[1:]

# run
if read_from_file:
    try:
        code = open(sys.argv[0], encoding="utf-8").read()
    except FileNotFoundError:
        error(f'fnf "{sys.argv[0]}"', 66)

    try:
        if should_encode:
            new_file = sys.argv[0] + ".sbcs"
            open(new_file, "w+").write(encode(code))
        elif should_decode:
            new_file = sys.argv[0] + ".utf8"
            open(new_file, "w+").write(decode(code))
        else:
            sys.argv = sys.argv[1:]
            args = [eval(arg) for arg in sys.argv]
            args = [to_chars(arg) if type(arg) == str else arg for arg in sys.argv]
            flax_run(code, args)
    except KeyboardInterrupt:
        error("kbdi", 130)
else:
    # repl
    try:
        while True:
            code = input("      ")
            args = [a.strip() for a in input(">>> ").split("|") if a != ""]
            args = [eval(arg) for arg in args]
            args = [to_chars(arg) if type(arg) == str else arg for arg in args]
            flax_run(code, args)
    except KeyboardInterrupt:
        error("kbdi", 130)
