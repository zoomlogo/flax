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
    assert depth([]) == 1
    assert depth([1]) == 1
    assert depth([[[1]], 1, 3]) == 3


def test_diagonal_leading():
    assert diagonal_leading([[1, 0, 2], [2, 3, 4], [5, 6, 7]]) == [1, 3, 7]


def test_diagonal_trailing():
    assert diagonal_trailing([[1, 0, 2], [2, 3, 4], [5, 6, 7]]) == [2, 3, 5]


def test_diagonals():
    assert diagonals([[1, 2, 3], [4, 5, 6], [7, 8, 9]]) == [
        [1],
        [4, 2],
        [7, 5, 3],
        [8, 6],
        [9],
    ]
    assert diagonals([[1, 2, 3], [4, 5, 6], [7, 8, 9]], antidiagonals=True) == [
        [7],
        [4, 8],
        [1, 5, 9],
        [2, 6],
        [3],
    ]


def test_digits():
    assert digits(123) == [1, 2, 3]
    assert digits(3.1415) == [3, 1, 4, 1, 5]
    assert digits(mpc(123, 456)) == [mpc(1, 4), mpc(2, 5), mpc(3, 6)]


def test_digits_i():
    assert digits_i([3, 1, 4, 1, 5]) == 31415
    assert digits_i([mpc(1, 4), mpc(2, 5), mpc(3, 6)]) == mpc(123, 456)


def test_enumerate_md():
    assert list(enumerate_md([])) == []
    assert list(enumerate_md([1, 2, 3, 4, 5])) == [
        [[0], 1],
        [[1], 2],
        [[2], 3],
        [[3], 4],
        [[4], 5],
    ]
    assert list(enumerate_md([1, 2, 3, [4, 5]])) == [
        [[0], 1],
        [[1], 2],
        [[2], 3],
        [[3, 0], 4],
        [[3, 1], 5],
    ]


def test_ensure_square():
    assert ensure_square([]) == []
    assert ensure_square(1) == [[1]]
    assert ensure_square([1, 2]) == [[1, 1], [2, 2]]
    assert ensure_square([1, 2, [3, 4], [5, 6, 7]]) == [
        [1, 1, 1, 1],
        [2, 2, 2, 2],
        [3, 4, 3, 4],
        [5, 6, 7, 5],
    ]


def test_fibonacci():
    assert [fibonacci(i) for i in range(50)] == [
        0,
        1,
        1,
        2,
        3,
        5,
        8,
        13,
        21,
        34,
        55,
        89,
        144,
        233,
        377,
        610,
        987,
        1597,
        2584,
        4181,
        6765,
        10946,
        17711,
        28657,
        46368,
        75025,
        121393,
        196418,
        317811,
        514229,
        832040,
        1346269,
        2178309,
        3524578,
        5702887,
        9227465,
        14930352,
        24157817,
        39088169,
        63245986,
        102334155,
        165580141,
        267914296,
        433494437,
        701408733,
        1134903170,
        1836311903,
        2971215073,
        4807526976,
        7778742049,
    ]


def test_find():
    assert find(3, [1,2])==[]
    assert find(3, [1,2,3]) == 2
    assert find(3, [1,2,3,4,5,3]) == 2

def test_find_md():
    assert find_md(4, [[1,2,3,4],[3,4,5],4,6]) == [0, 3]
    assert find_md(3, [1,2]) == []
    assert find_md(3, [1,3]) == [1]

def test_find_all():
    assert find_all(3, [1,2])==[]
    assert find_all(3, [1,2,3]) == [2]
    assert find_all(3, [1,2,3,4,5,3]) == [2,5]

def test_find_sublist():
    assert find_sublist([1,2,3, 4,5],[1,2,2,1,2,3,4,2,1]) == []
    assert find_sublist([1,2,3],[1,2,2,1,2,3,4,2,1]) == 3
    assert find_sublist([1,2,3,4],[1,2,2,1,2,3,4,2,1]) == 3

def test_flatten():
    assert flatten([1,2,[3,4,[5,3],5],[],3]) == [1,2,3,4,5,3,5,3]
    assert flatten([]) == []
    assert flatten(1) == [1]

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
