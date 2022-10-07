# test syntax things and chain separators
# and niladic atoms
from flax.main import flax_run as run


def test_numbers():
    assert run("¯") == -1
