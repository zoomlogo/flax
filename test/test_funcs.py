import os, sys
from mpmath import *

THIS_FOLDER = os.path.dirname(os.path.abspath(__file__)) + "/.."
sys.path.insert(1, THIS_FOLDER)

from flax.funcs import *


def test_base():
    assert base(1, [1, 0, 0]) == 1
    assert base(2, [1, 0, 0]) == 4
    assert base(2, [4, 4, 5]) == 29
    assert base(3, [1, 1, 2]) == 14

    assert base(2, 1101) == 13
    assert base(2, [-1, -1, -0, -1]) == -13


def test_base_decomp():
    assert base_decomp(2, 9) == [[1, 0], [0, 1]]


def test_base_i():
    assert base_i(2, 4) == [1, 0, 0]
    assert base_i(3, 14) == [1, 1, 2]
    assert base_i(3, -14) == [-1, -1, -2]


def test_binary():
    assert binary(15) == [1, 1, 1, 1]
    assert binary(0) == [0]
    assert binary(-1) == [-1]
    assert binary(-5) == [-1, 0, -1]


def test_binary_i():
    assert binary_i([1, 1]) == 3
    assert binary_i([0]) == 0
    assert binary_i([]) == 0
    assert binary_i(1) == 1
    assert binary_i([-1, 0, -1]) == -5


def test_cartesian_product():
    assert cartesian_product([1, 2], [3, 4]) == [[1, 3], [1, 4], [2, 3], [2, 4]]
    assert cartesian_product([], [3, 4]) == []
    assert cartesian_product([0], [3, 4]) == [[0, 3], [0, 4]]


def test_convolve():
    assert convolve([1, 2, 3, 4], [1, 2, 3, 4]) == [1, 4, 10, 20, 25, 24, 16]
    assert convolve([1, 2, 3, 4], []) == [0, 0, 0]
    assert convolve([], []) == []
    assert convolve([1, 1, 1], [1, 1, 1]) == [1, 2, 3, 2, 1]


def test_depth():
    assert depth(0) == 0
    assert depth("string") == 1
    assert depth([]) == 0
    assert depth([1]) == 1
    assert depth([[[1]], 1, 3]) == 3


"test_diagonal_leading"
"test_diagonal_trailing"
"test_diagonals"
"test_digits"
"test_digits_i"
"test_enumerate_md"
"test_ensure_square"
"test_fibonacci"
"test_find"
"test_find_md"
"test_find_all"
"test_find_sublist"
"test_flatten"
"test_get_req"
"test_grade_down"
"test_grade_up"
"test_group_equal"
"test_group_indicies"
"test_index_into"
"test_index_into_md"
"test_iota"
"test_iota1"
"test_iterable"
"test_join"
"test_json_decode"
"test_lucas"
"test_mapval"
"test_maximal_indicies"
"test_maximal_indicies_md"
"test_mold"
"test_multiset_difference"
"test_multiset_intersection"
"test_multiset_union"
"test_nprimes"
"test_ones"
"test_order"
"test_permutations"
"test_prefixes"
"test_prime_factors"
"test_random"
"test_repeat"
"test_reshape"
"test_rld"
"test_rle"
"test_shuffle"
"test_sliding_window"
"test_split"
"test_split_at"
"test_split_into"
"test_sublists"
"test_suffixes"
"test_to_braille"
"test_transpose"
"test_trim"
"test_unrepeat"
"test_where"
