# Introduction

## Installation
Make sure you have Python 3.9 or above.

Optionally you can use `rlwrap` to get a better REPL.
```sh
git clone https://github.com/PyGamer0/flax
pip install poetry
cd flax
poetry install
poetry run python -m flax <flags, etc>
```

## Flags
Here is a list of flags, used while running flax.

Flag|Description
----|-----------
`C`|Prints as characters instead of numbers.
`d`|Debug flag.
`f`|Read and run a program from a file.
`p`|Higher precision for floats (specified by the next argument).
`P`|Disable pretty printing as grids.
`e`|encode as sbcs
`D`|decode from sbcs

## Terminology

- Arity: The number of arguments a builtin takes.
- Niladic: Takes no arguments.
- Monadic: Takes a single argument, that is the left argument.
- Dyadic: Takes two arguments, that is the left and right arguments.

### Atoms
Atoms are builtins which perform a specific function.

For example: `+` is a dyadic atom which adds its arguments.

### Quicks
Quicks are sort of like Vyxal's modifiers.
Quicks pop links from the left during parse time.

For example: `'` is the quick which maps a link over its argument.

### Links
Links are atoms, or group of atoms processed by quicks.

### Chains
Chains are a sequence of links.
Chains can be called niladically, monadically or dyadically, depending on the number of arguments a chain was called with.
Chains are evaluated from right to left.

Certain rules are followed when chains are processed.

Here are those rules:
```
$ -> accumulator
w/x -> right / left argument of chain

UNIVERSAL RULES:
monads are applied normally:
FGH -> F(G(H($)))
FHH -> F(H(H($)))

dyads take the right (or left) argument of the chain:
f -> f(w,$)
g -> g(x,$)

nilad-dyads take nilads as the right argument of the dyad
4g -> g(4,$)

dyad-nilads take nilads as the left argument of the dyad
g4 -> g($,4)

unparseable nilads are pusedo-stranded when they are next to dyads, otherwise they trigger a hidden dyad component associated with monads, these are preparsed after the lexer:
0 4 5g -> [0,4,5]g

the ending most nilad becomes the accumulator:
1 -> $ = 1

MONADIC:
monad-dyads modify the left argument
Ff -> f(F(x),$)

DYADIC:
dyad-dyad pairs have the first dyad to be supplied with the arguments, its result gets passed to the left argument of the other dyad
fg -> g(f(w,x),$)

nilad-dyad-dyad triplet, i can't explain so just look:
4gh -> g(4,h(w,$)))
```

## Datatypes
There are 2 datatypes:
- Lists
- Scalars
  - Integers
  - Floats
  - Complex numbers

**NOTE:** A string is just a list of integers.

## Syntax

Here are the syntatic sugar included with flax.
Syntax|Description|Example
------|-----------|-------
`"`|Start / End a string|`"Hello!"`
`012456789`|A number|`12`
`i`|Defines a complex number. By default it is `0i1`.|`2i`
`.`|Defines a decimal number. By default it is `0.5`|`.2`
`()`|Start / End a list|`(1 2 (3 4))`
`_`|Next character's value|`_f`
`:`|Next 2 characters' value|`:()`

## Chain separators
Chain separators separate chains within the same line.
Chain Separator|Description
---------------|-----------
`ø`|Start a niladic chain.
`µ`|Start a monadic chain.
`[`|Start a monadic chain which maps over its argument.
`]`|Start a monadic chain which filters over its argument.
`ð`|Start a dyadic chain.
`ɓ`|Start a dyadic chain with reversed arguments.
