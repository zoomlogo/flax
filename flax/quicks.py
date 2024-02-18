# quicks: holds the quicks
import more_itertools as mit

from flax.common import *
from flax.funcs import *
from flax.chains import *

quicks = {  # single byte quicks
    "$": quick_chain(1, 2),
    "¢": quick_chain(2, 2),
    "£": quick_chain(1, 3),
    "¥": quick_chain(2, 3),
    "€": quick_chain(1, 4),
    "₹": quick_chain(2, 4),
    "'": attrdict(
        condition=lambda links: links,
        qlink=lambda links, *_: [
            attrdict(
                arity=links[0].arity or 1,
                call=fix_args(
                    lambda w, x: (
                        [
                            variadic_link(links[0], (i, j))
                            for i, j in zip(iterable(w), iterable(x))
                        ]
                        if w is not None
                        else [variadic_link(links[0], (i,)) for i in iterable(x)]
                    )
                ),
            )
        ],
    ),
    "’": attrdict(
        condition=lambda links: links,
        qlink=lambda links, *_: [
            attrdict(
                arity=2,
                call=lambda w, x: [
                    variadic_link(links[0], (i, x)) for i in iterable(w)
                ],
            )
        ],
    ),
    "‘": attrdict(
        condition=lambda links: links,
        qlink=lambda links, *_: [
            attrdict(
                arity=2,
                call=lambda w, x: [
                    variadic_link(links[0], (w, i)) for i in iterable(x)
                ],
            )
        ],
    ),
    "¨": attrdict(
        condition=lambda links: len(links) == 2,
        qlink=lambda links, *_: [
            attrdict(
                arity=1,
                call=lambda x: (
                    [variadic_chain(links, (i,)) for i in x]
                    if type(x) == list
                    else variadic_chain(links, (x,))
                ),
            )
        ],
    ),
    "ζ": attrdict(
        condition=lambda links: links,
        qlink=lambda links, *_: [
            atoms["ϕ"],
            attrdict(
                arity=links[0].arity or 1,
                call=fix_args(
                    lambda w, x: (
                        [
                            variadic_link(links[0], (i, j))
                            for i, j in zip(iterable(w), iterable(x))
                        ]
                        if w is not None
                        else [variadic_link(links[0], (i,)) for i in iterable(x)]
                    )
                ),
            ),
        ],
    ),
    "ρ": attrdict(
        condition=lambda links: links,
        qlink=lambda links, *_: [
            attrdict(
                arity=links[0].arity or 1,
                call=fix_args(
                    lambda w, x: (
                        [
                            variadic_link(links[0], (i, j))
                            for i, j in zip(
                                permutations(iterable(w)), permutations(iterable(x))
                            )
                        ]
                        if w is not None
                        else [
                            variadic_link(links[0], (i,))
                            for i in permutations(iterable(x))
                        ]
                    )
                ),
            )
        ],
    ),
    "⊸": attrdict(
        condition=lambda links: links,
        qlink=lambda links, *_: [
            attrdict(
                arity=links[0].arity or 1,
                call=fix_args(
                    lambda w, x: (
                        [
                            variadic_link(links[0], (i, j))
                            for i, j in zip(
                                prefixes(iterable(w)), prefixes(iterable(x))
                            )
                        ]
                        if w is not None
                        else [
                            variadic_link(links[0], (i,)) for i in prefixes(iterable(x))
                        ]
                    )
                ),
            )
        ],
    ),
    "?": attrdict(
        condition=lambda links: len(links) == 3,
        qlink=lambda links, *_: [
            attrdict(
                arity=max(arities(links)),
                call=fix_args(
                    lambda w, x: (
                        variadic_link(links[0], (w, x))
                        if variadic_link(links[2], (w, x))
                        else variadic_link(links[1], (w, x))
                    )
                ),
            )
        ],
    ),
    "ω": attrdict(
        condition=lambda links: len(links) == 2,
        qlink=lambda links, *_: [
            attrdict(
                arity=max(arities(links)),
                call=fix_args(lambda w, x: while_loop(links[0], links[1], (w, x))),
            )
        ],
    ),
    "⍤": attrdict(
        condition=lambda links: links and links[0].arity,
        qlink=lambda links, *_: (
            [links.pop(0)] if len(links) == 2 and links[0].arity == 0 else []
        )
        + [
            attrdict(
                arity=max_arity(links),
                call=fix_args(lambda w, x: ntimes(links, (w, x))),
            )
        ],
    ),
    "@": attrdict(
        condition=lambda links: len(links) == 2,
        qlink=lambda links, *_: [
            attrdict(
                arity=links[0].arity or 1,
                call=fix_args(lambda w, x: apply_at(links[0], links[1].call(), w, x)),
            )
        ],
    ),
    "˘": attrdict(
        condition=lambda links: links,
        qlink=lambda links, *_: [
            attrdict(
                arity=1 if links[0].arity == 2 else (2 if links[0].arity == 1 else 0),
                call=fix_args(
                    lambda _, x: (
                        variadic_link(links[0], (x, x))
                        if links[0].arity == 2
                        else variadic_link(links[0], (x,))
                    ),
                ),
            )
        ],
    ),
    "˜": attrdict(
        condition=lambda links: links,
        qlink=lambda links, *_: [
            attrdict(
                arity=2 if links[0].arity else 0,
                call=fix_args(
                    lambda w, x: (
                        variadic_link(links[0], (x, w))
                        if links[0].arity != 1
                        else links[0].call(w)
                    ),
                ),
            )
        ],
    ),
    "`": attrdict(
        condition=lambda links: links and links[0].arity,
        qlink=lambda links, *_: [
            attrdict(
                arity=links[0].arity, call=fix_args(lambda w, x: sort(links, w, x))
            )
        ],
    ),
    "η": attrdict(
        condition=lambda links: links and links[0].arity,
        qlink=lambda links, *_: [
            attrdict(
                arity=links[0].arity, call=fix_args(lambda w, x: group(links, w, x))
            )
        ],
    ),
    "⌉": attrdict(
        condition=lambda links: links,
        qlink=lambda links, *_: [
            attrdict(
                arity=links[0].arity,
                call=fix_args(
                    lambda w, x: max(
                        iterable(x), key=lambda k: variadic_link(links[0], (w, k))
                    )
                ),
            )
        ],
    ),
    "⌋": attrdict(
        condition=lambda links: links,
        qlink=lambda links, *_: [
            attrdict(
                arity=links[0].arity,
                call=fix_args(
                    lambda w, x: min(
                        iterable(x), key=lambda k: variadic_link(links[0], (w, k))
                    )
                ),
            )
        ],
    ),
    "τ": attrdict(
        condition=lambda links: links,
        qlink=lambda links, *_: [
            attrdict(
                arity=links[0].arity or 1,
                call=fix_args(
                    lambda w, x: [
                        i
                        for i, e in enumerate(
                            [variadic_link(links[0], (w, i)) for i in iterable(x)]
                        )
                        if e
                    ],
                ),
            )
        ],
    ),
    "α": attrdict(
        condition=lambda links: links,
        qlink=lambda links, *_: [
            attrdict(
                arity=links[0].arity or 1,
                call=fix_args(
                    lambda w, x: mit.all_equal(
                        [variadic_link(links[0], (w, i)) for i in sliding_window(2, x)]
                    )
                ),
            ),
        ],
    ),
    "υ": attrdict(
        condition=lambda links: links,
        qlink=lambda links, *_: [
            attrdict(
                arity=links[0].arity or 1,
                call=fix_args(
                    lambda w, x: [
                        variadic_link(links[0], (w, i)) for i in sliding_window(2, x)
                    ]
                ),
            ),
        ],
    ),
    "⊥": attrdict(
        condition=lambda links: links,
        qlink=lambda links, *_: [
            attrdict(
                arity=links[0].arity or 1,
                call=fix_args(lambda w, x: ffilter(links, w, x)),
            )
        ],
    ),
    "⊤": attrdict(
        condition=lambda links: links,
        qlink=lambda links, *_: [
            attrdict(
                arity=links[0].arity or 1,
                call=fix_args(lambda w, x: ffilter(links, w, x, inverse=True)),
            )
        ],
    ),
    "⁼": attrdict(
        condition=lambda links: links,
        qlink=lambda links, *_: [
            attrdict(
                arity=links[0].arity,
                call=fix_args(lambda w, x: int(w == variadic_link(links[0], w, x))),
            )
        ],
    ),
    "β": attrdict(
        condition=lambda _: True,
        qlink=lambda _, outermost_links, i: [create_chain(outermost_links[i])],
    ),
    "θ": attrdict(
        condition=lambda _: True,
        qlink=lambda _, outermost_links, i: [
            create_chain(outermost_links[(i + 1) % len(outermost_links)], 0)
        ],
    ),
    "λ": attrdict(
        condition=lambda _: True,
        qlink=lambda _, outermost_links, i: [
            create_chain(outermost_links[(i + 1) % len(outermost_links)], 1)
        ],
    ),
    "ν": attrdict(
        condition=lambda _: True,
        qlink=lambda _, outermost_links, i: [
            create_chain(outermost_links[(i + 1) % len(outermost_links)], 2)
        ],
    ),
    "σ": attrdict(
        condition=lambda _: True,
        qlink=lambda _, outermost_links, i: [
            create_chain(outermost_links[(i - 1) % len(outermost_links)], 0)
        ],
    ),
    "ς": attrdict(
        condition=lambda _: True,
        qlink=lambda _, outermost_links, i: [
            create_chain(outermost_links[(i - 1) % len(outermost_links)], 1)
        ],
    ),
    "π": attrdict(
        condition=lambda _: True,
        qlink=lambda _, outermost_links, i: [
            create_chain(outermost_links[(i - 1) % len(outermost_links)], 2)
        ],
    ),
    "φ": attrdict(
        condition=lambda links: len(links) == 2,
        qlink=lambda links, outermost_links, _: [
            create_chain(
                outermost_links[links[0].call() % len(outermost_links)], links[1].call()
            )
        ],
    ),
    "/": attrdict(
        condition=lambda links: links and links[0].arity,
        qlink=lambda links, *_: [
            attrdict(arity=1, call=lambda x: fold(links, x, right=True))
        ],
    ),
    "⌿": attrdict(
        condition=lambda links: links and links[0].arity,
        qlink=lambda links, *_: [
            attrdict(
                arity=2, call=lambda w, x: fold(links, w, x, initial=True, right=True)
            )
        ],
    ),
    "\\": attrdict(
        condition=lambda links: links and links[0].arity,
        qlink=lambda links, *_: [
            attrdict(arity=1, call=lambda x: scan(links, x, right=True))
        ],
    ),
    "⍀": attrdict(
        condition=lambda links: links and links[0].arity,
        qlink=lambda links, *_: [
            attrdict(
                arity=2, call=lambda w, x: scan(links, w, x, initial=True, right=True)
            )
        ],
    ),
    "´": attrdict(
        condition=lambda links: links,
        qlink=lambda links, *_: [
            attrdict(
                arity=links[0].arity or 1,
                call=fix_args(lambda w, x: fold_fixedpoint(links, w, x)),
            )
        ],
    ),
    "˝": attrdict(
        condition=lambda links: links,
        qlink=lambda links, outermost_links, i: [
            attrdict(
                arity=links[0].arity or 1,
                call=fix_args(lambda w, x: scan_fixedpoint(links, w, x)),
            )
        ],
    ),
    "∀": attrdict(
        condition=lambda links: links,
        qlink=lambda links, *_: [
            attrdict(
                arity=links[0].arity,
                call=fix_args(
                    lambda w, x: copy_to(atoms["∃"], variadic_link(links[0], w, x)),
                ),
            )
        ],
    ),
    "∝": attrdict(
        condition=lambda links: links,
        qlink=lambda links, *_: [
            attrdict(
                arity=1,
                call=lambda x: variadic_link(
                    links[0],
                    (transpose(variadic_link(links[0], (x,), force_monad=True)),),
                    force_monad=True,
                ),
            )
        ],
    ),
    "⟜": attrdict(
        condition=lambda links: links
        and (
            links[-1].arity == 0
            and len(links) == links[-1].call() - 1
            or len(links) == 3
        ),
        qlink=lambda links, *_: [
            attrdict(arity=2, call=lambda w, x: composed(links, w, x))
        ],
    ),
    "˙": attrdict(
        condition=lambda links: links,
        qlink=lambda links, *_: [
            attrdict(
                arity=links[0].arity,
                call=fix_args(lambda w, x: variadic_link(links[0], (w, x), flat=True)),
            ),
        ],
    ),
    "¾": attrdict(
        condition=lambda links: len(links) == 2,
        qlink=lambda links, *_: [
            attrdict(
                arity=2,
                call=lambda w, x: dyadic_link(
                    links[1],
                    variadic_link(links[0], (x, w)),
                    variadic_link(links[0], (w, x)),
                ),
            )
        ],
    ),
}

quicks |= {  # quicky diagraphs
    "Δ/": attrdict(
        condition=lambda links: links and links[0].arity,
        qlink=lambda links, *_: [attrdict(arity=1, call=lambda x: fold(links, x))],
    ),
    "Δ⌿": attrdict(
        condition=lambda links: links and links[0].arity,
        qlink=lambda links, *_: [
            attrdict(arity=2, call=lambda w, x: fold(links, w, x, initial=True))
        ],
    ),
    "Δ\\": attrdict(
        condition=lambda links: links and links[0].arity,
        qlink=lambda links, *_: [attrdict(arity=1, call=lambda x: scan(links, x))],
    ),
    "Δ⍀": attrdict(
        condition=lambda links: links and links[0].arity,
        qlink=lambda links, *_: [
            attrdict(arity=2, call=lambda w, x: scan(links, w, x, initial=True))
        ],
    ),
    "Δω": attrdict(
        condition=lambda links: len(links) == 2,
        qlink=lambda links, *_: [
            attrdict(
                arity=max(arities(links)),
                call=fix_args(
                    lambda w, x: while_loop(links[0], links[1], (w, x), cumulative=True)
                ),
            )
        ],
    ),
    "Δ⍤": attrdict(
        condition=lambda links: links and links[0].arity,
        qlink=lambda links, *_: (
            [links.pop(0)] if len(links) == 2 and links[0].arity == 0 else []
        )
        + [
            attrdict(
                arity=max_arity(links),
                call=fix_args(lambda w, x: ntimes(links, (w, x), cumulative=True)),
            )
        ],
    ),
    "Δe": attrdict(
        condition=lambda links: links,
        qlink=lambda links, *_: [
            attrdict(
                arity=links[0].arity or 1,
                call=fix_args(
                    lambda w, x: apply_at(links[0], list(range(2, len(x) + 2, 2), w, x))
                ),
            )
        ],
    ),
    "Δo": attrdict(
        condition=lambda links: links,
        qlink=lambda links, *_: [
            attrdict(
                arity=links[0].arity or 1,
                call=fix_args(
                    lambda w, x: apply_at(links[0], list(range(1, len(x) + 1, 2), w, x))
                ),
            )
        ],
    ),
}
