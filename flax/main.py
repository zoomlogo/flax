from flax.interpreter import variadic_chain, pp
from flax.lexer import tokenise
from flax.parser import parse


def flax_eval(code, *args):
    try:
        pp(variadic_chain(parse(tokenise(code))[-1] if code else "", *args))
    except KeyboardInterrupt:
        print()
        exit(130)


def cli_repl(prompt="      ", inp_prompt="> "):
    try:
        print("flax REPL version 0.1.0")
        while True:
            chain = parse(tokenise(input(prompt)))[-1]
            chain = chain if chain else ""
            args = [input(inp_prompt), input(inp_prompt)]
            args = filter("".__ne__, args)
            pp(variadic_chain(chain, *args))
    except KeyboardInterrupt:
        print()
        exit(130)
