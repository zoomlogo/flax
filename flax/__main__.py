# Main entry point
from sys import argv

from flax.main import run

argv = argv[1:]

if argv:
    code = open(argv[0]).read()
    argv = argv[1:]
    run(code, *map(eval, argv))
else:
    # TODO: CLI REPL
    exit(10)
