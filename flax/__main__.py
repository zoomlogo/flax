# Main entry point
import sys

from flax.error import error
from flax.main import flax_eval, cli_repl
import flax.interpreter

sys.argv = sys.argv[1:]
read_from_file = False

if sys.argv:
    if "d" in sys.argv[0]:
        flax.interpreter.DEBUG = True
    if "f" in sys.argv[0]:
        read_from_file = True
    if "C" in sys.argv[0]:
        flax.interpreter.PRINT_CHARS = True
    if "p" in sys.argv[0]:
        flax.interpreter.mp.dps = 100
    if "P" in sys.argv[0]:
        flax.interpreter.DISABLE_GRID = True
    sys.argv = sys.argv[1:]

if read_from_file:
    try:
        code = open(sys.argv[0], encoding="utf-8").read()
    except FileNotFoundError:
        error(f'ERROR: File "{sys.argv[0]}" not found.', 66)
        exit(66)

    sys.argv = sys.argv[1:]
    flax_eval(code, *map(eval, sys.argv))
else:
    cli_repl()
