from flax.interpreter import variadic_chain, pp
from flax.lexer import tokenise
from flax.parser import parse


def run(code, *args):
    try:
        pp(variadic_chain(parse(tokenise(code))[-1] if code else "", *args))
    except KeyboardInterrupt:
        print()
        exit(130)
