# chains: holds the functions used by quicks and the chains
from flax.common import attrdict, flax_print


def arities(links):
    # arities: return the arities of the links
    return [link.arity for link in links]


def copy_to(atom, value):
    # copy_to: copy value into the atom
    atom.call = lambda: value
    return value


def create_chain(chain, arity=-1, isForward=True):
    # create_chain: make a new chain
    return attrdict(
        arity=arity,
        chain=chain,
        call=lambda w=None, x=None: variadic_chain(
            chain, *(isForward and (w, x) or (x, w))
        ),
    )


def dyadic_chain(chain, w, x):
    # dyadic_chain: evaluate a dyadic chain
    for link in chain:
        if link.arity < 0:
            link.arity = 2

    if chain and arities(chain[-3:]) == [2, 2, 2]:
        λ = chain[-1].call(w, x)
        chain = chain[:-1]
    elif trailing_nilad(chain):
        λ = chain[-1].call()
        chain = chain[:-1]
    else:
        λ = x

    while chain:
        if arities(chain[-3:]) == [0, 2, 2] and trailing_nilad(chain[:-2]):
            λ = chain[-2].call(chain[-3].call(), chain[-1].call(w + λ))
            chain = chain[:-3]
        elif arities(chain[-2:]) == [2, 2]:
            λ = chain[-1].call(chain[-2].call(x, w), λ)
            chain = chain[:-2]
        elif arities(chain[-2:]) == [0, 2]:
            λ = chain[-1].call(chain[-2].call(), λ)
            chain = chain[:-2]
        elif arities(chain[-2:]) == [0, 2]:
            λ = chain[-2].call(λ, chain[-1].call())
            chain = chain[:-2]
        elif chain[-1].arity == 2:
            λ = chain[-1].call(w, λ)
            chain = chain[:-1]
        elif chain[-1].arity == 1:
            λ = chain[-1].call(λ)
            chain = chain[:-1]
        else:
            flax_print(λ)
            λ = chain[-1].call()
            chain = chain[:-1]
    return λ


def trailing_nilad(chain):
    # trailing_nilad: if the chain is a trailing constant chain
    return chain and arities(chain[::-1]) + [1] < [0, 2] * len(chain)


def max_arity(links):
    # max_arity: return the maximum arity of the links
    return (
        max(arities(links))
        if min(arities(links)) > -1
        else (~max(arities(links))) or -1
    )


def monadic_chain(chain, x):
    # monadic_chain: evaluate a monadic chain
    init = True

    λ = x
    while True:
        if init:
            for link in chain:
                if link.arity < 0:
                    link.arity = 1

            if trailing_nilad(chain):
                λ = chain[-1].call()
                chain = chain[:-1]

            init = False

        if not chain:
            break

        if arities(chain[-2:]) == [1, 2]:
            λ = chain[-1].call(chain[-2].call(x), λ)
            chain = chain[:-2]
        elif arities(chain[-2:]) == [0, 2]:
            λ = chain[-1].call(chain[-2].call(), λ)
            chain = chain[:-2]
        elif arities(chain[-2:]) == [0, 2]:
            λ = chain[-2].call(λ, chain[-1].call())
            chain = chain[:-2]
        elif chain[-1].arity == 2:
            λ = chain[-1].call(x, λ)
            chain = chain[:-1]
        elif chain[-1].arity == 1:
            if not chain[:-1] and hasattr(chain[-1], "chain"):
                x = λ
                chain = chain[-1].chain
                init = True
            else:
                λ = chain[-1].call(λ)
                chain = chain[:-1]
        else:
            flax_print(λ)
            λ = chain[-1].call()
            chain = chain[:-1]

    return λ


def niladic_chain(chain):
    # niladic_chain: evaluate a niladic chain
    if not chain or chain[-1].arity > 0:
        return monadic_chain(chain, 0)
    return monadic_chain(chain[:-1], chain[-1].call())
