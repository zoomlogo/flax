# encoding.py: stores stuff related to encoding of flax

__all__ = ["encode", "decode"]

codepage = "⁰¹²³⁴⁵⁶⁷⁸⁹¶±«»≡≢"
codepage += "øµðɓ‘’¢£¥€₹⌿≥≠≤⍀"
codepage += " !\"#$%&'()*+,-./"
codepage += "0123456789:;<=>?"
codepage += "@ABCDEFGHIJKLMNO"
codepage += "PQRSTUVWXYZ[\\]^_"
codepage += "`abcdefghijklmno"
codepage += "pqrstuvwxyz{|}~˙"
codepage += "ȦḂĊḊĖḞĠḢİĿṀṄȮṖṘṠ"
codepage += "ṪẆẊẎŻȧḃċḋėḟġḣŀṁṅ"
codepage += "ȯṗṙṡṫẇẋẏż→←↑↓∥∦¬"
codepage += "´˝˘¨˜¯¼½¾×⁼∞∝∀∃÷"
codepage += "∘○⟜⊸∊∴∵∂⊂⊃⊆⊇⊏⊐⊑⊒"
codepage += "αβγδεζηθικλνξπρ"
codepage += "ςστυφχψωϕΔ⌈⌊⌉⌋ØÆŒ"
codepage += "√ΣΠ⊢⊣⊤⊥∧∨∪∩⍋⍒⍤æœ"

assert len(codepage) == 256


def encode(program):
    # encode: encode a flax program in unicode to sbcs
    encoded = ""
    for ch in program:
        encoded += chr(codepage.index(ch))
    return encoded


def decode(program):
    # decode: flax program in sbcs to unicode
    decoded = ""
    for ch in program:
        decoded += codepage[ord(ch)]
    return decoded
