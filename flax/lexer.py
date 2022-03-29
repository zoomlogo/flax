# lexer: holds the flax lexer
import collections
import enum

from flax.builtins import *

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
        if head == STRING_DELIMETER:
            string_contents = ""
            while program:
                top = program.popleft()
                if top == "\\" and program:
                    string_contents += "\\" + program.popleft()
                elif top == STRING_DELIMETER:
                    break
                else:
                    string_contents += top
            tokens.append([TOKEN_TYPE.STRING, string_contents])
        elif head == STRING_NEXT_1:
            string_contents = program.popleft()
            tokens.append([TOKEN_TYPE.STRING, string_contents])
        elif head == STRING_NEXT_2:
            string_contents = program.popleft()
            string_contents += program.popleft()
            tokens.append([TOKEN_TYPE.STRING, string_contents])
        elif head in DIGITS:
            contextual_token_value = head
            if head == ZERO and not (
                program
                and program[0] in DECIMAL_POINT + COMPLEX_DELIMETER + NEGATIVE_SIGN
            ):
                # handle the special case of 0
                tokens.append([TOKEN_TYPE.NUMBER, contextual_token_value])
            else:
                while (
                    program
                    and program[0] in DIGITS
                    and (contextual_token_value + program[0]).count(COMPLEX_DELIMETER)
                    < 2
                    and all(
                        (x.count(DECIMAL_POINT) < 2 and x.count(NEGATIVE_SIGN) < 2)
                        for x in (contextual_token_value + program[0]).split(
                            COMPLEX_DELIMETER
                        )
                    )
                ):
                    contextual_token_value += program.popleft()
                tokens.append([TOKEN_TYPE.NUMBER, contextual_token_value])
        elif head == COMMENT:
            # just ignore comments
            while program and program.popleft() != NEWLINE:
                pass
        elif head == NEWLINE:
            tokens.append([TOKEN_TYPE.NEWLINE, NEWLINE])
        elif head in train_separators:
            tokens.append([TOKEN_TYPE.TRAIN_SEPARATOR, head])
        elif head in atoms:
            tokens.append([TOKEN_TYPE.ATOM, head])
        elif head in quicks:
            tokens.append([TOKEN_TYPE.QUICK, head])
        elif head in DIAGRAPHS and program:
            digraph = head + program.popleft()
            if digraph in atoms:
                tokens.append([TOKEN_TYPE.ATOM, digraph])
            elif digraph in quicks:
                tokens.append([TOKEN_TYPE.QUICK, digraph])
            else:
                raise NameError("Digraph not defined.")
        elif head == LIST_DELIMETER_L:
            contents = ""
            k = 1
            while True:
                head = program.popleft()
                if head == LIST_DELIMETER_R:
                    k -= 1
                    if k == 0:
                        tokens.append([TOKEN_TYPE.LIST, tokenise(contents)])
                        break
                    else:
                        contents += head
                else:
                    if head == LIST_DELIMETER_L:
                        k += 1
                    contents += head
    return tokens
