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
    assert run("Ḃ", [-1, -1, -1]) == -6
    assert run("Ḃ", [[1, 0, 1, 1], [1, 0, 1]]) == [11, 5]
