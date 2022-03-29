# chains: holds the functions used by quicks and the chains
import functools
import itertools

import flax.common
from flax.common import attrdict, flax_print, flax_string
from flax.error import debug
from flax.funcs import permutations, iterable, sliding_window, split, flatten, suffixes

__all__ = [
    "apply_at",
    "arities",
    "copy_to",
    "create_chain",
    "dyadic_chain",
    "ffilter",
    "fold",
    "max_arity",
    "monadic_chain",
    "niladic_chain",
    "ntimes",
    "quick_chain",
    "scan",
    "sort",
    "trailing_nilad",
    "variadic_chain",
    "variadic_link",
    "while_loop",
    "while_not_unique",
]


def apply_at(link, indicies, *args):
    # apply_at: apply link at indicies
    x = iterable(args[-1])
    if len(args) == 2:
        w = args[0]
    else:
        w = None

    indicies = flatten(iterable(indicies))
    for i in indicies:
        i = int(i)
        x[i % len(x)] = variadic_link(link, (w, x[i % len(x)]))
    return x


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
            chain, (isForward and (w, x) or (x, w))
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

    if flax.common.DEBUG:
        debug("in dyadic chain | w, x ← " + flax_string(w) + ", " + flax_string(x))

    while chain:
        if flax.common.DEBUG:
            debug("λ: " + flax_string(λ))

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


def ffilter(links, *args, inverse=False, permutation=False):
    # ffilter: filter with optional inverse or permuting the argument
    x = iterable(args[-1], range_=True)
    if len(args) == 2:
        w = args[0]
    else:
        w = None

    if permutation:
        x = permutations(x)

    if links[0].arity == 0:
        return list(
            filter(
                lambda a: a != links[0].call() if inverse else a == links[0].call(), x
            )
        )
    else:
        return list(
            filter(
                lambda a: not variadic_link(links[0], (a, w))
                if inverse
                else variadic_link(links[0], (a, w)),
                x,
            )
        )


def fold(links, *args, right=False, initial=False):
    # fold: fold over args
    x = iterable(args[-1])
    if len(args) == 2:
        w = args[0]
    else:
        w = None

    if right:
        x = x[::-1]
        call = lambda w, x: variadic_link(links[0], (x, w), force_dyad=True)
    else:
        call = lambda w, x: variadic_link(links[0], (w, x), force_dyad=True)

    if len(links) == 1:
        if initial:
            return functools.reduce(call, x, w)
        else:
            return functools.reduce(call, x)
    else:
        if initial:
            return [
                functools.reduce(call, z, w) for z in sliding_window(links[1].call(), x)
            ]
        else:
            return [
                functools.reduce(call, z) for z in sliding_window(links[1].call(), x)
            ]


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

            if flax.common.DEBUG:
                debug("in monadic chain | x ← " + flax_string(x))

        if not chain:
            break

        if flax.common.DEBUG:
            debug("λ: " + flax_string(λ))

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
    if flax.common.DEBUG:
        debug("in niladic chain")
    if not chain or chain[-1].arity > 0:
        return monadic_chain(chain, 0)
    return monadic_chain(chain[:-1], chain[-1].call())


def ntimes(links, args, cumulative=False):
    # ntimes: repeat link n times
    times = int(links[1].call()) if len(links) == 2 else int(input())
    res, x = args
    if cumulative:
        c_res = [res]
    for _ in range(times):
        w = res
        res = variadic_link(links[0], (w, x))
        if cumulative:
            c_res.append(res)
        x = w
    return c_res if cumulative else res


def quick_chain(arity, min_length):
    return attrdict(
        condition=(lambda links: len(links) >= min_length and links[0].arity == 0)
        if arity == 0
        else lambda links: len(links) - sum([leading_nilad(x) for x in suffixes(links)])
        >= min_length,
        qlink=lambda links, outer_links, i: [
            attrdict(
                arity=arity, call=lambda w=None, x=None: variadic_chain(links, (w, x))
            )
        ],
    )


def scan(links, *args, right=False, initial=False):
    # scan: scan over args
    x = iterable(args[-1])
    if len(args) == 2:
        w = args[0]
    else:
        w = None

    if right:
        x = x[::-1]
        call = lambda w, x: variadic_link(links[0], (x, w), force_dyad=True)
    else:
        call = lambda w, x: variadic_link(links[0], (w, x), force_dyad=True)

    if len(links) == 1:
        if initial:
            return itertools.accumulate(call, x, initial=w)
        else:
            return itertools.accumulate(call, x)
    else:
        if initial:
            return [
                itertools.accumulate(call, z, initial=w)
                for z in sliding_window(links[1].call(), x)
            ]
        else:
            return [
                itertools.accumulate(call, z)
                for z in sliding_window(links[1].call(), x)
            ]


def sort(links, *args, i):
    # sort: sort args according to links
    x = iterable(args[-1], digits=True)
    if len(args) == 2:
        w = args[0]
    else:
        w = None

    if len(links) == 2:
        # special nilad case
        x = split(links[1].call(), x)

    res = list(sorted(x, key=lambda a: variadic_link(links[0], (w, a))))
    return sum(res, []) if len(links) == 2 else res


def trailing_nilad(chain):
    # trailing_nilad: if the chain is a trailing constant chain
    return chain and arities(chain[::-1]) + [1] < [0, 2] * len(chain)


def variadic_chain(chain, args):
    args = list(filter(None.__ne__, args))
    if len(args) == 0:
        return niladic_chain(chain)
    elif len(args) == 1:
        return monadic_chain(chain, *args)
    else:
        return dyadic_chain(chain, *args)


def variadic_link(link, args, force_dyad=False):
    # call link with args
    args = list(filter(None.__ne__, args))
    if link.arity == -1:
        link.arity = len(args)

    if link.arity == 0:
        return link.call()
    elif link.arity == 1:
        if force_dyad:
            return [args[0], link.call(args[1])]
        else:
            return link.call(args[0])
    elif link.arity == 2:
        return link.call(args[0], args[1])


def while_loop(link, cond, args, cumulative=False):
    # while_loop: while condition is true apply link
    res, x = args
    if cumulative:
        c_res = [res]
    while variadic_link(cond, (res, x)):
        w = res
        res = variadic_link(link, (w, x))
        if cumulative:
            c_res.append(res)
        x = w
    return c_res if cumulative else res


def while_not_unique(link, x, cumulative=False):
    # while_not_unique: run link while the result is not equal to the previous result
    res = link.call(x)
    before = x
    if cumulative:
        c_res = [res]
    while res != before:
        before = res
        res = link.call(res)
        if cumulative:
            c_res.append(res)
    return c_res if cumulative else res
