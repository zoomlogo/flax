# Introduction

## Installation
Make sure you have Python 3.9 or above.
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
Chains are a sequence of atoms.
Chains can be called niladically, monadically or dyadically.

Certain rules are followed when chains are processed.

#### Leading constant chains
A leading constant chain (LCC) is a special type of chain that starts with a nilad and is followed by monads, dyad-nilad pairs, and nilad-dyad pairs.
Essentially, the nilad must not be paired with a dyad immediately after it.
You can think of it as a chain that takes no arguments (i.e. is niladic) because it begins with a nilad itself and then follows it with a chain of monadic-like operations.
Certain chaining rules depend on this.

#### Niladic chains
- If the chain starts with a nilad say `1`, the accumulator (we will call it `λ`) is that nilad.
`λ = 1`.
- Otherwise `λ = 0`.

Now the rest of the chain is evaluated monadically.

#### Monadic chains
- If the chain starts with an LCC, `1 ...` then `λ = 1` and the rest of the chain is evaluated. 
- Otherwise `λ = α` (`α` is the left argument).

Now one by one the atoms are considered, and are applied to the accumulator `λ` according to this table:
Code|New λ|Arities
----|-----|-------
`+ F`|`λ + F(⍺)`| 2, 1
`+ 1`|`λ + 1`| 2, 0
`1 +`|`1 + λ`| 0, 2
`+`|`λ + ⍺`| 2
`F`|`F(λ)`| 1

#### Dyadic chains
- If the chain starts with an LCC `1 ...` then `λ = 1`  and the rest of the chain is evaluated.
- If the chain starts with 3 dyads `+ × ÷` then `λ = ⍺ + ⍵` (`⍵` is the right argument) and the rest of the chain is evaluated.
- Otherwise `λ = ⍺`.

Now one by one the atoms are considered, and are applied to the accumulator `λ` according to this table:
Code|New λ|Arities
----|-----|-------
`+ × 1`| `(λ + ⍵) × 1`| 2, 2, 0₁
`+ ×`| `λ + (⍺ × ⍵)`| 2, 2
`+ 1`|`λ + 1`| 2, 0
`1 +`|`1 + λ`| 0, 2
`+`|`λ + ⍵`| 2
`F`|`F(λ)`| 1

₁ The rule only applies if the nilad is part of an LCC.

## Datatypes
There are 2 datatypes:
- Lists
- Scalars

**NOTE:** A string is just a list of integers.

## Syntax

Here are the syntatic sugar included with flax.
Syntax|Description|Example
------|-----------|-------
`'`|Start / End a string|`'Hello!'`
`012456789`|A number|`12`
`j`|Defines a complex number. By default it is `0j1`.|`2j`
`.`|Defines a decimal number. By default it is `0.5`|`.2`
