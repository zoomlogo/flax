from flax.interpreter import variadic_chain, flax_print
from flax.lexer import tokenise
from flax.parser import parse

from prompt_toolkit import PromptSession
from prompt_toolkit import print_formatted_text as pft, HTML


def flax_eval(code, *args):
    try:
        pp(variadic_chain(parse(tokenise(code))[-1] if code else "", *args))
    except KeyboardInterrupt:
        pft(HTML("<ansired>ERROR: KeyboardInterrupt Recieved.</ansired>"))
        exit(130)


def cli_repl(prompt="      ", inp_prompt="> "):
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
            args = map(eval, filter("".__ne__, args))
            flax_print(variadic_chain(chain, *args))
    except KeyboardInterrupt:
        pft(HTML("<ansired>ERROR: KeyboardInterrupt Recieved.</ansired>"))
        exit(130)
