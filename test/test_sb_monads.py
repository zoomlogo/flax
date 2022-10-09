# single byte monads
import os, sys
from mpmath import *

THIS_FOLDER = os.path.dirname(os.path.abspath(__file__)) + "/.."
sys.path.insert(1, THIS_FOLDER)

from flax.main import flax_run as run


def test_abs():  # A
    assert run("A") == 0
    assert run("A", -3) == 3
    assert run("A", 3) == 3


def test_any():  # Ȧ
    assert run("Ȧ") == 0
    assert run("Ȧ", [0] * 10) == 0
    assert run("Ȧ", [0, 0, 1, 0, 0]) == 1
    assert run("Ȧ", [1, 2]) == 1


def test_i2b():  # B
    assert run("B") == [0]
    assert run("B", 12) == [1, 1, 0, 0]
    assert run("B", -6) == [-1, -1, 0]
    assert run("B", [5, -3, 10]) == [[1, 0, 1], [-1, -1], [1, 0, 1, 0]]


def test_b2i():  # Ḃ
    assert run("Ḃ") == 0
    assert run("Ḃ", [1, 1, 0]) == 6
    assert run("Ḃ", [-1, -1, -1]) == -7
    assert run("Ḃ", [[1, 0, 1, 1], [1, 0, 1]]) == [11, 5]


def test_complement():  # C
    assert run("C") == 1
    assert run("C", 1) == 0
    assert run("C", 10) == -9
    assert run("C", [10, -10, [4]]) == [-9, 11, [-3]]


def test_randomly_choose():  # Ċ
    assert run("Ċ", [1, 2, 3]) in [1, 2, 3]


def test_i2d():  # D
    assert run("D") == [0]
    assert run("D", 2) == [2]
    assert run("D", 234) == [2, 3, 4]
    assert run("D", -23) == [-2, -3]
    assert run("D", [-45, 45]) == [[-4, -5], [4, 5]]


def test_d2i():  # Ḋ
    assert run("Ḋ") == 0
    assert run("Ḋ", [1, 2]) == 12
    assert run("Ḋ", [-6, -9]) == -69
    assert run("Ḋ", [[2, 3, 1], [-5, -3]]) == [231, -53]


def test_alleq():  # E
    assert run("E") == 1
    assert run("E", [1, 2, 3]) == 0
    assert run("E", [2, 2, 2]) == 1
    assert run("E", [[1, 2], [1, 2]]) == 1


def test_aleq_r():  # Ė
    assert run("E") == 0
    assert run("E", [1, 2, 3]) == []
    assert run("E", [2, 2, 2]) == 2
    assert run("E", [[1, 2], [1, 2]]) == [1, 2]
