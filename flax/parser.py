from flax.interpreter import attrdict
from flax.interpreter import atoms
from flax.interpreter import quicks

from flax.lexer import *


def numberify(x):
    number = x.replace("¯", "-")
    if "j" in number:
        if len(number) == 1:
            return complex(0, 1)
        else:
            parts = number.split("j")
            if parts[0] == "":
                parts[0] = "0"
            if parts[1] == "":
                parts[1] = "1"
            return complex(numberify(parts[0]), numberify(parts[1]))
    elif "." in number:
        if len(number) == 1:
            return 0.5
        else:
            parts = number.split(".")
            if parts[0] == "":
                parts[0] = "0"
            if parts[1] == "":
                parts[1] = "5"
            return float(".".join(parts))
    else:
        if "-" in number:
            if len(number) == 1:
                return -1
            else:
                return int(number[1:]) * -1
        else:
            return int(number)


def parse(tokens):
    trains = []
    stack = []

    while tokens:
        token = tokens[0]
        if token[0] == TOKEN_TYPE.NEWLINE:
            trains.append(stack[::])
            stack = []
        elif token[0] == TOKEN_TYPE.NUMBER:
            stack.append(attrdict(arity=0, call=lambda: numberify(token[1])))
        elif token[0] == TOKEN_TYPE.STRING:
            stack.append(
                attrdict(
                    arity=0,
                    call=lambda: [
                        ord(x)
                        for x in token[1]
                        .replace("\\n", "\n")
                        .replace("\\'", "'")
                    ],
                )
            )
        elif token[0] == TOKEN_TYPE.ATOM:
            stack.append(atoms[token[1]])
        elif token[0] == TOKEN_TYPE.QUICK:
            popped = []
            while not quicks[token[1]].condition(popped) and (stack or trains):
                popped.insert(0, (stack or trains).pop())
            stack += quicks[token[1]].qlink(popped, trains, index)
        tokens = tokens[1:]
    if stack:
        trains.append(stack[::])
    return trains


x = tokenise("1+1ð2+2\n3+3")
y = parse(x)
print(y)
