# Main entry point
from sys import argv
from prompt_toolkit import print_formatted_text as pft, HTML

from flax.main import flax_eval, cli_repl
import flax.interpreter

argv = argv[1:]
read_from_file = False

if argv:
    if "d" in argv[0]:
        flax.interpreter.DEBUG = True
    if "f" in argv[0]:
        read_from_file = True
    if "C" in argv[0]:
        flax.interpreter.PRINT_CHARS = True
    argv = argv[1:]

if read_from_file:
    try:
        code = open(argv[0], encoding="utf-8").read()
    except FileNotFoundError:
        pft(HTML(f"<ansired>ERROR: File {argv[0]} not found.</ansired>"))
        exit(66)

    argv = argv[1:]
    flax_eval(code, *map(eval, argv))
else:
    cli_repl()
