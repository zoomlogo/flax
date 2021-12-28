import collections
import enum
import string

from interpreter import atoms
from interpreter import quicks


class TOKEN_TYPE(enum.Enum):
    NUMBER = 1
    STRING = 2
    TRAIN_SEPARATOR = 3
    ATOM = 4
    QUICK = 5
    NEWLINE = 6
    LIST = 7


def tokenise(program):

    tokens = []
    program = collections.deque(program)

    while program:
        head_character = program.popleft()
        if head_character == "'":
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
        elif head_character in string.digits + ".j":
            contextual_token_value = head_character
            if head_character == "0" and not (program and program[0] in ".j"):
                # Handle the special case of 0.
                tokens.append([TOKEN_TYPE.NUMBER, contextual_token_value])
            else:
                while (
                    program
                    and program[0] in string.digits + ".j"
                    and (contextual_token_value + program[0]).count("j") < 2
                    and all(
                        x.count(".") < 2
                        for x in (contextual_token_value + program[0]).split(
                            "j"
                        )
                    )
                ):
                    contextual_token_value += program.popleft()
                tokens.append([TOKEN_TYPE.NUMBER, contextual_token_value])
        elif head_character == "⍝":
            while program.popleft() != "\n":
                pass
            # Just ignore comments
        elif head_character == "\n":
            tokens.append([TOKEN_TYPE.NEWLINE, "\n"])
        elif head_character in "øµðɓг":
            tokens.append([TOKEN_TYPE.TRAIN_SEPARATOR, head_character])
        elif head_character in atoms:
            tokens.append([TOKEN_TYPE.ATOM, head_character])
        elif head_character in quicks:
            tokens.append([TOKEN_TYPE.QUICK, head_character])
        elif head_character == "[":
            contents = ""
            while head_character != "]":
                head_character = program.popleft()
                if head_character == "]":
                    tokens.append([TOKEN_TYPE.LIST, contents])
                else:
                    contents += head_character
    return tokens
