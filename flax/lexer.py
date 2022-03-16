# lexer: holds the flax lexer
import collections
import enum
import string

from flax.interpreter import atoms
from flax.interpreter import quicks

__all__ = ["TOKEN_TYPE", "tokenise"]


class TOKEN_TYPE(enum.Enum):
    NUMBER = 1
    STRING = 2
    TRAIN_SEPARATOR = 3
    ATOM = 4
    QUICK = 5
    NEWLINE = 6
    LIST = 7


def tokenise(program):
    # tokenise: convert program into tokens
    tokens = []
    program = collections.deque(program)

    while program:
        head = program.popleft()
        if head == "'":
            string_contents = ""
            while program:
                top = program.popleft()
                if top == "\\" and program:
                    string_contents += "\\" + program.popleft()
                elif top == "'":
                    break
                else:
                    string_contents += top
            tokens.append([TOKEN_TYPE.STRING, string_contents])
        elif head == "₊":
            string_contents = program.popleft()
            tokens.append([TOKEN_TYPE.STRING, string_contents])
        elif head == "₋":
            string_contents = program.popleft()
            string_contents += program.popleft()
            tokens.append([TOKEN_TYPE.STRING, string_contents])
        elif head in string.digits + "¯.j":
            contextual_token_value = head
            if head == "0" and not (program and program[0] in "¯.j"):
                # handle the special case of 0
                tokens.append([TOKEN_TYPE.NUMBER, contextual_token_value])
            else:
                while (
                    program
                    and program[0] in string.digits + "¯.j"
                    and (contextual_token_value + program[0]).count("j") < 2
                    and all(
                        (x.count(".") < 2 and x.count("¯") < 2)
                        for x in (contextual_token_value + program[0]).split("j")
                    )
                ):
                    contextual_token_value += program.popleft()
                tokens.append([TOKEN_TYPE.NUMBER, contextual_token_value])
        elif head == "⍝":  # ideally these string constants will be in builtins
            # just ignore comments
            while program and program.popleft() != "\n":
                pass
        elif head == "\n":
            tokens.append([TOKEN_TYPE.NEWLINE, "\n"])
        elif head in "øµðɓг":
            tokens.append([TOKEN_TYPE.TRAIN_SEPARATOR, head])
        elif head in atoms:
            tokens.append([TOKEN_TYPE.ATOM, head])
        elif head in quicks:
            tokens.append([TOKEN_TYPE.QUICK, head])
        elif head in "_;:ᵟ" and program:
            digraph = head + program.popleft()
            if digraph in atoms:
                tokens.append([TOKEN_TYPE.ATOM, digraph])
            elif digraph in quicks:
                tokens.append([TOKEN_TYPE.QUICK, digraph])
            else:
                raise NameError("Digraph not defined.")
        elif head == "[":
            contents = ""
            k = 1
            while True:
                head = program.popleft()
                if head == "]":
                    k -= 1
                    if k == 0:
                        tokens.append([TOKEN_TYPE.LIST, tokenise(contents)])
                        break
                    else:
                        contents += head
                else:
                    if head == "[":
                        k += 1
                    contents += head
    return tokens
