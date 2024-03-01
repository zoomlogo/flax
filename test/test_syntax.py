# test syntax things and chain separators
# and niladic atoms
import os, sys
from mpmath import *

THIS_FOLDER = os.path.dirname(os.path.abspath(__file__)) + "/.."
sys.path.insert(1, THIS_FOLDER)

from flax.main import flax_run as run


def test_numbers_egde_cases():
    assert run("¯") == -1
    assert run(".") == 0.5
    assert run("i") == mpc("0", "1")
    assert run(".i¯.") == mpc("0.5", "-0.5")


def test_numbers():
    assert run("0") == 0
    assert run("69") == 69
    assert run("420") == 420
    assert run("69i420") == mpc("69", "420")
    assert run("69.420") == mpf("69.420")


def test_string():
    assert run('"Hello, World!"') == "Hello, World!"

def test_next_string():
    assert run("_E") == "E"
    assert run(":Hi") == "Hi"


def test_lists():
    assert run("[1]") == [1]
    assert run("[1, 2]") == [1, 2]
    assert run("[1+2]") == [1, 2]  # any delimeter
    assert run("[[1 2],[3 4]]") == [[1, 2], [3, 4]]
    assert run('["Hi" 1]') == ["Hi", 1]


def test_chain_separators():
    pass  # TODO


def test_nilads_0_to_7():
    assert run("⁰") == 10
    assert run("¹") == 16
    assert run("²") == 26
    assert run("³") == 32
    assert run("⁴") == 64
    assert run("⁵") == 100
    assert run("⁶") == 256
    assert run("⁷") == -2


def test_nilads_8_9():
    assert run("⁸") == 11
    assert run("⁹") == 13
    assert run("⁸", 3) == 11
    assert run("⁹", 3) == 3
    assert run("⁸", 3, 4) == 3
    assert run("⁹", 3, 4) == 4


def test_nilads_other():
    assert run("∃") == 0
    assert run("⍬") == []
    assert run("⊶") == [0, 1]
