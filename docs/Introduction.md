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
`p`|Higher precision for floats (100 digits).
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

For example: `¨` is the quick which maps a link over its argument.

### Links
Links are atoms, or group of atoms processed by quicks.

### Chains
Chains are a sequence of links.
Chains can be called niladically, monadically or dyadically, depending on the number of arguments a chain was called with.
Chains are evaluated from right to left.

Certain rules are followed when chains are processed.

#### Trailing constant chains
A trailing constant chain (TCC) is a special type of chain that ends with a nilad and is preceded by monads, dyad-nilad pairs, and nilad-dyad pairs.
Essentially, the nilad must not be paired with a dyad immediately before it.
You can think of it as a chain that takes no arguments (i.e. is niladic) because it begins with a nilad itself and then precedes it with a chain of monadic-like operations.
Certain chaining rules depend on this.

#### Niladic chains
- If the chain ends with a nilad say `1`, the accumulator (we will call it `λ`) is that nilad.
`λ = 1`.
- Otherwise `λ = 0`.

Now the rest of the chain is evaluated monadically.

#### Monadic chains
- If the chain ends with an TCC, `... 1` then `λ = 1` and the rest of the chain is evaluated.
- Otherwise `λ = x` (`x` is the right argument).

Now one by one the atoms are considered from right to left, and are applied to the accumulator `λ` according to this table:
Code|New λ|Arities
----|-----|-------
`F +`|`F(x) + λ`| 1, 2
`1 +`|`1 + λ`| 0, 2
`+ 1`|`λ + 1`| 2, 0
`+`|`x + λ`| 2
`F`|`F(λ)`| 1

#### Dyadic chains
- If the chain ends with an TCC `... 1` then `λ = 1`  and the rest of the chain is evaluated.
- If the chain ends with 3 dyads `+ × ÷` then `λ = w ÷ x` (`w` is the left argument) and the rest of the chain is evaluated.
- Otherwise `λ = x`.

Now one by one the atoms are considered from right to left, and are applied to the accumulator `λ` according to this table:
Code|New λ|Arities
----|-----|-------
`1 × +`| `1 × (w + λ)`| 0₁, 2, 2 
`× +`| `(x × w) + λ`| 2, 2
`1 +`|`1 + λ`| 0, 2
`+ 1`|`λ + 1`| 2, 0
`+`|`w + λ`| 2
`F`|`F(λ)`| 1

₁ The rule only applies if the nilad is part of a TCC.

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
`'`|Start / End a string|`'Hello!'`
`012456789`|A number|`12`
`j`|Defines a complex number. By default it is `0j1`.|`2j`
`.`|Defines a decimal number. By default it is `0.5`|`.2`
`[]`|Start / End a list|`[1 2 [3 4]]`
`₊`|Next character's value|`₊f`
`₋`|Next 2 characters' value|`₋(]`

## Chain separators
Chain separators separate chains within the same line.
Chain Separator|Description
---------------|-----------
`ø`|Start a niladic chain.
`µ`|Start a monadic chain.
`г`|Start a monadic chain which maps over its argument.
`ð`|Start a dyadic chain.
`ɓ`|Start a dyadic chain with reversed arguments.
