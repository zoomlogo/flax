# Main entry point
from sys import argv

from flax.main import flax_eval, cli_repl

argv = argv[1:]

if argv:
    code = open(argv[0], encoding="utf-8").read()
    argv = argv[1:]
    flax_eval(code, *map(eval, argv))
else:
    cli_repl()
