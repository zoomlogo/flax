# Main entry point
from sys import argv
from prompt_toolkit import print_formatted_text as pft, HTML

from flax.main import flax_eval, cli_repl

argv = argv[1:]

if argv:
    try:
        code = open(argv[0], encoding="utf-8").read()
    except FileNotFoundError:
        pft(HTML(f"<ansired>ERROR: File {argv[0]} not found.</ansired>"))
        exit(66)

    argv = argv[1:]
    flax_eval(code, *map(eval, argv))
else:
    cli_repl()
