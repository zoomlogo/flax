# Main entry point
from prompt_toolkit import print_formatted_text, HTML
import sys

from flax.main import flax_eval, cli_repl
import flax.interpreter

sys.argv = argv[1:]
read_from_file = False

if sys.argv:
    if "d" in sys.argv[0]:
        flax.interpreter.DEBUG = True
    if "f" in sys.argv[0]:
        read_from_file = True
    if "C" in sys.argv[0]:
        flax.interpreter.PRINT_CHARS = True
    sys.argv = argv[1:]

if read_from_file:
    try:
        code = open(sys.argv[0], encoding="utf-8").read()
    except FileNotFoundError:
        print_formatted_text(
            HTML(f"<ansired>ERROR: File {sys.argv[0]} not found.</ansired>"),
            file=sys.stderr,
        )
        exit(66)

    sys.argv = argv[1:]
    flax_eval(code, *map(eval, sys.argv))
else:
    cli_repl()
