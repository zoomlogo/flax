from flax.interpreter import attrdict
from flax.interpreter import atoms
from flax.interpreter import quicks
from flax.interpreter import train_separators
from flax.interpreter import create_chain

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

    tokens = split_on_newlines(tokens)
    trains = [[] for _ in tokens]

    for index, train in enumerate(tokens):
        chains = trains[index]
        subtrains = split_on_separators(train)
        for subtrain in subtrains:
            stack = []
            arity, is_forward = train_separators.get(subtrain[0][1], (-1, True))
            if arity != -1:
                subtrain = subtrain[1:]
            for token in subtrain:
                if token[0] == TOKEN_TYPE.NUMBER:
                    stack.append(
                        attrdict(arity=0, call=lambda: numberify(token[1]))
                    )
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
                    while not quicks[token[1]].condition(popped) and (
                        stack or trains
                    ):
                        popped.insert(0, (stack or chains).pop())
                    stack += quicks[token[1]].qlink(popped, trains, index)
            chains.append(create_chain(stack, arity, is_forward))
    return trains


def split_on_newlines(tokens):
    lines = []
    current = []
    for token in tokens:
        if token[0] == TOKEN_TYPE.NEWLINE:
            lines.append(current)
            current = []
        else:
            current.append(token)
    lines.append(current)
    return lines


def split_on_separators(tokens):
    separators = []
    current = []
    for token in tokens:
        if token[0] == TOKEN_TYPE.TRAIN_SEPARATOR:
            separators.append(current)
            current = [token]
        else:
            current.append(token)
    separators.append(current)
    return separators
