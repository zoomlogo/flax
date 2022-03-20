# chains: holds the functions used by quicks and the chains
from flax.common import attrdict


def arities(links):
    # arities: return the arities of the links
    return [links.arity for link in links]


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
