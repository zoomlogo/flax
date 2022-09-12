# parser: holds the main components of the flax parser
from flax.error import error
from flax.common import attrdict, mpc, mpf
from flax.builtins import atoms, quicks, train_separators
from flax.chains import create_chain

from flax.lexer import *


def arrayify(arr_list):
    """arrayify: convert to python list"""
    array = []
    for x in arr_list:
        if x[0] == TOKEN_TYPE.NUMBER:
            array.append(numberify(x[1]))
        elif x[0] == TOKEN_TYPE.LIST:
            array.append(arrayify(x[1]))
        elif x[0] == TOKEN_TYPE.STRING:
            array.append(escapes(x[1]))
    return array


def escapes(x):
    """escapes: convert x to a list of numbers and parse escape sequences"""
    x = x.replace("\\\\", "\\")
    x = x.replace("\\n", "\n")
    x = x.replace('\\"', '"')
    x = x.replace("\\t", "\t")
    x = x.replace("\\v", "\v")
    x = x.replace("\\e", "\x1b")
    x = x.replace("\\a", "\a")
    x = x.replace("\\r", "\r")
    x = x.replace("\\b", "\b")
    x = x.replace("\\f", "\f")
    return [ord(i) for i in x]


def numberify(x):
    """numberify: convert to a mp number / python number"""
    number = x.replace("¯", "-")
    if "j" in number:
        if len(number) == 1:
            return mpc(0, 1)
        else:
            parts = number.split("j")
            if parts[0] == "":
                parts[0] = "0"
            if parts[1] == "":
                parts[1] = "1"
            return mpc(numberify(parts[0]), numberify(parts[1]))
    elif "." in number:
        if len(number) == 1:
            return mpf("0.5")
        else:
            parts = number.split(".")
            if parts[0] == "":
                parts[0] = "0"
            if parts[1] == "":
                parts[1] = "5"
            return mpf(".".join(parts))
    else:
        if "-" in number:
            if len(number) == 1:
                return -1
            else:
                return int(number)
        else:
            return int(number)


def parse(tokens):
    """parse: parse and group tokens"""
    tokens = split_on_newlines(tokens)
    tokens = [*filter([].__ne__, tokens)]
    trains = [[] for _ in tokens]

    for index, train in enumerate(tokens):
        chains = trains[index]
        subtrains = split_on_separators(train)
        for subtrain in subtrains:
            stack = []
            arity, is_forward = train_separators.get(str(subtrain[0][1]), (-1, True))
            if arity == 1 and subtrain[0][1] == "[":
                subtrain.insert(1, [TOKEN_TYPE.QUICK, "'"])
            if arity == 1 and subtrain[0][1] == "]":
                subtrain.insert(1, [TOKEN_TYPE.QUICK, "⊥"])
            if arity != -1:
                subtrain = subtrain[1:]
            for token in subtrain:
                if token[0] == TOKEN_TYPE.NUMBER:
                    stack.append(
                        attrdict(
                            arity=0,
                            call=lambda x=token[1]: numberify(x),
                            glyph=token[1],
                        )
                    )
                elif token[0] == TOKEN_TYPE.STRING:
                    stack.append(
                        attrdict(
                            arity=0,
                            call=lambda x=token[1]: escapes(x),
                            glyph=token[1],
                        )
                    )
                elif token[0] == TOKEN_TYPE.LIST:
                    stack.append(
                        attrdict(
                            arity=0, call=lambda x=token[1]: arrayify(x), glyph=token[1]
                        )
                    )
                elif token[0] == TOKEN_TYPE.ATOM:
                    stack.append(atoms[token[1]])
                elif token[0] == TOKEN_TYPE.QUICK:
                    # handle quicks
                    popped = []
                    while not quicks[token[1]].condition(popped) and (stack or trains):
                        if stack == [] and chains == []:
                            if token[1] in "ⁿ":
                                break
                            error(f'not enough links to pop for "{token[1]}"')
                        popped.insert(0, (stack or chains).pop())
                    stack += quicks[token[1]].qlink(popped, trains, index)
            chains.append(create_chain(stack, arity, is_forward))
    return trains


def split_on_newlines(tokens):
    """split_on_newlines: split tokens on newlines"""
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
    """split_on_separators: split tokens on chain separators"""
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
