# main: main entry point and gluing
import sys

from flax.chains import variadic_chain
from flax.encoding import *
from flax.error import error, debug
from flax.funcs import chars
from flax.lexer import tokenise
from flax.parser import parse
import flax.common
import flax.builtins

__all__ = ["main", "flax_run"]

# function for running flax
def flax_run(code, *args):
    tokens = tokenise(code)
    debug("tokens: " + str(tokens))
    parsed = parse(tokens)
    debug("parsed: " + str(parsed))
    # run
    # niladic chains 11 and 13
    # monadic ones only 11, dyadic nothing
    flax.builtins.atoms["⁸"].call = lambda: args[0] if len(args) > 1 else 11
    flax.builtins.atoms["⁹"].call = lambda: args[-1] if len(args) > 0 else 13
    return variadic_chain(parsed[-1] if parsed else "", args)


def main():
    sys.argv = sys.argv[1:]
    read_from_file = False
    should_encode = False
    should_decode = False

    # handle flags
    if sys.argv:
        flags = sys.argv[0]
        if "g" in flags:
            flax.common.DEBUG = True
        if "f" in flags:
            read_from_file = True
        if "c" in flags:
            flax.common.PRINT_CHARS = True
        if "p" in flags:
            sys.argv = sys.argv[1:]
            flax.common.mp.dps = int(sys.argv[0])
        if "P" in flags:
            flax.common.DISABLE_GRID = True
        if "e" in flags:
            read_from_file = True
            should_encode = True
        if "d" in flags:
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
                args = [chars(arg) if type(arg) == str else arg for arg in args]
                flax.common.flax_print(flax_run(code, *args))
        except KeyboardInterrupt:
            error("kbdi", 130)
    else:
        # repl
        try:
            while True:
                code = input("      ")
                args = [a.strip() for a in input(">>> ").split("|") if a != ""]
                args = [eval(arg) for arg in args]
                args = [chars(arg) if type(arg) == str else arg for arg in args]
                flax.common.flax_print(flax_run(code, *args))
        except KeyboardInterrupt:
            error("kbdi", 130)
