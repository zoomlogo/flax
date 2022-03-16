# main: glues components of the interpreter together
from flax.error import error
from flax.interpreter import variadic_chain, flax_print
from flax.lexer import tokenise
from flax.parser import parse

from prompt_toolkit import PromptSession


def flax_eval(code, *args):
    # flax_eval: run code with args
    try:
        flax_print(variadic_chain(parse(tokenise(code))[-1] if code else "", *args))
    except KeyboardInterrupt:
        error("ERROR: KeyboardInterrupt Recieved.", 130)


def cli_repl(prompt="      ", inp_prompt="> "):
    # cli_repl: start a repl
    try:
        session = PromptSession()
        print("flax REPL version 0.1.0")
        while True:
            inp = session.prompt(prompt)
            if inp == "":
                continue
            elif inp == "exit":
                break
            chain = parse(tokenise(inp))[-1]
            chain = chain if chain else ""
            args = [input(inp_prompt), input(inp_prompt)]
            args = [
                [ord(x) for x in a] if isinstance(a, str) else a
                for a in map(eval, filter("".__ne__, args))
            ]
            flax_print(variadic_chain(chain, *args))
    except KeyboardInterrupt:
        error("ERROR: KeyboardInterrupt Recieved.", 130)
