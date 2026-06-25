from mpmath import *

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
    assert find(3, [1, 2]) == []
    assert find(3, [1, 2, 3]) == 2
    assert find(3, [1, 2, 3, 4, 5, 3]) == 2


def test_find_md():
    assert find_md(4, [[1, 2, 3, 4], [3, 4, 5], 4, 6]) == [0, 3]
    assert find_md(3, [1, 2]) == []
    assert find_md(3, [1, 3]) == [1]


def test_find_all():
    assert find_all(3, [1, 2]) == []
    assert find_all(3, [1, 2, 3]) == [2]
    assert find_all(3, [1, 2, 3, 4, 5, 3]) == [2, 5]


def test_find_sublist():
    assert find_sublist([1, 2, 3, 4, 5], [1, 2, 2, 1, 2, 3, 4, 2, 1]) == []
    assert find_sublist([1, 2, 3], [1, 2, 2, 1, 2, 3, 4, 2, 1]) == 3
    assert find_sublist([1, 2, 3, 4], [1, 2, 2, 1, 2, 3, 4, 2, 1]) == 3


def test_flatten():
    assert flatten([1, 2, [3, 4, [5, 3], 5], [], 3]) == [1, 2, 3, 4, 5, 3, 5, 3]
    assert flatten([]) == []
    assert flatten(1) == [1]


def test_get_req():
    assert "google" in get_req("google.com")


def test_grade_down():
    assert grade_down([]) == []
    assert grade_down([2, 3, 1]) == [1, 0, 2]
    assert grade_down([7, 0, 9, 3, 8, 0, 9, 5, 4, 1]) == [2, 6, 4, 0, 7, 8, 3, 9, 1, 5]
    assert grade_down(grade_down([7, 0, 9, 3, 8, 0, 9, 5, 4, 1])) == [
        7,
        5,
        4,
        1,
        9,
        2,
        6,
        0,
        8,
        3,
    ]


def test_grade_up():
    assert grade_up([]) == []
    assert grade_up([2, 3, 1]) == [2, 0, 1]
    assert grade_up([7, 0, 9, 3, 8, 0, 9, 5, 4, 1]) == [1, 5, 9, 3, 8, 7, 0, 4, 2, 6]
    assert grade_up(grade_up([7, 0, 9, 3, 8, 0, 9, 5, 4, 1])) == [
        6,
        0,
        8,
        3,
        7,
        1,
        9,
        5,
        4,
        2,
    ]


def test_group_equal():
    assert group_equal([]) == []
    assert group_equal([1, 2, 3]) == [[1], [2], [3]]
    assert group_equal([2, 3, 1]) == [[2], [3], [1]]
    assert group_equal([1, 2, 2, 3, 4, 4, 3]) == [[1], [2, 2], [3], [4, 4], [3]]


def test_group_indicies():
    assert group_indicies([]) == []
    assert group_indicies([1, 2, 3]) == [[0], [1], [2]]
    assert group_indicies([3, 2, 1]) == [[2], [1], [0]]
    assert group_indicies([1, 2, 2, 3, 4, 4, 3]) == [[0], [1, 2], [3, 6], [4, 5]]


def test_index_into():
    assert index_into([], 0) == []
    assert index_into([1, 2, [3, 4]], 0) == 1
    assert index_into([1, 2, [3, 4]], 1) == 2
    assert index_into([1, 2, [3, 4]], 2) == [3, 4]
    assert index_into([1, 2, [3, 4]], 3) == 1
    assert index_into([1, 2, [3, 4]], 0.5) == [1, 2]
    assert index_into([1, 2, [3, 4]], mpc(2, 1)) == 4


def test_index_into_md():
    assert index_into_md([[1, 2, [3, 4]], [2]], [0, 2, 1]) == 4


def test_iota():
    assert iota(5) == [0, 1, 2, 3, 4]
    assert iota([5]) == [[0], [1], [2], [3], [4]]
    assert iota([2, 3]) == [[[0, 0], [0, 1], [0, 2]], [[1, 0], [1, 1], [1, 2]]]
    assert iota(2.3) == [0, 1]
    assert iota(mpc(2, 3)) == [
        [(0.0 + 0.0j), (0.0 + 1.0j), (0.0 + 2.0j)],
        [(1.0 + 0.0j), (1.0 + 1.0j), (1.0 + 2.0j)],
    ]


def test_iota1():
    assert iota1(5) == [1, 2, 3, 4, 5]
    assert iota1([5]) == [[1], [2], [3], [4], [5]]
    assert iota1([2, 3]) == [[[1, 1], [1, 2], [1, 3]], [[2, 1], [2, 2], [2, 3]]]
    assert iota1(2.3) == [1, 2]
    assert iota1(mpc(2, 3)) == [
        [(1.0 + 1.0j), (1.0 + 2.0j), (1.0 + 3.0j)],
        [(2.0 + 1.0j), (2.0 + 2.0j), (2.0 + 3.0j)],
    ]


def test_iterable():
    assert iterable([]) == []
    assert iterable([1, 2]) == [1, 2]
    assert iterable(1) == [1]
    assert iterable("abc") == ["a", "b", "c"]
    assert iterable(3, range_=True) == [0, 1, 2]
    assert iterable(31415, digits_=True) == [3, 1, 4, 1, 5]


def test_join():
    assert join(3, iota(3)) == [0, 3, 1, 3, 2, 3]
    assert join([3, 4], iota(5)) == [0, 3, 1, 4, 2, 3, 3, 4, 4, 3]


def test_json_decode():
    assert json_decode({"a": 1, "b": "hello", "x": True}) == [
        ["a", 1],
        ["b", "hello"],
        ["x", 1],
    ]


def test_lucas():
    assert [lucas(i) for i in range(50)] == [
        2,
        1,
        3,
        4,
        7,
        11,
        18,
        29,
        47,
        76,
        123,
        199,
        322,
        521,
        843,
        1364,
        2207,
        3571,
        5778,
        9349,
        15127,
        24476,
        39603,
        64079,
        103682,
        167761,
        271443,
        439204,
        710647,
        1149851,
        1860498,
        3010349,
        4870847,
        7881196,
        12752043,
        20633239,
        33385282,
        54018521,
        87403803,
        141422324,
        228826127,
        370248451,
        599074578,
        969323029,
        1568397607,
        2537720636,
        4106118243,
        6643838879,
        10749957122,
        17393796001,
    ]


def test_mapval():
    assert mapval([[], []], []) == []
    assert mapval([[2, 1, 4], [1, 2, 3]], [2, 1, 4]) == [1, 2, 3]
    assert mapval([[2, 1, 4], [1, 2, 3]], [1, 1, 4]) == [2, 2, 3]


def test_maximal_indicies():
    assert maximal_indicies([]) == []
    assert maximal_indicies([1, 2, 3, 3, 2, 1]) == [2, 3]


def test_maximal_indicies_md():
    assert maximal_indicies_md([]) == []
    assert maximal_indicies_md([1, 2, 3]) == [[2]]
    assert maximal_indicies_md([1, 2, 3, [3]]) == [[2], [3, 0]]


def test_mold():
    assert mold([], []) == []
    assert mold([2], []) == []
    assert mold([2], [1]) == [1]
    assert mold([2, 1], [1]) == [1, 1]
    assert mold([2, 3, 1], [1, [2]]) == [1, 2, 1]
    assert mold([2, [3, 1]], [1, [2]]) == [1, [2, 1]]
    assert mold([2, [3, 1], [[[1]]]], [1, [2], [[3]]]) == [1, [2, 3], [[[1]]]]


def test_multiset_difference():
    assert multiset_difference([], []) == []
    assert multiset_difference([1, 2, 3], []) == [1, 2, 3]
    assert multiset_difference([], [1, 2, 3]) == []
    assert multiset_difference([1, 2, 3], [2]) == [1, 3]
    assert multiset_difference([1, 2, 3], [4, 5]) == [1, 2, 3]
    assert multiset_difference([1, 1, 2, 3], [1]) == [1, 2, 3]
    assert multiset_difference([1, 1, 2, 3], [1, 1]) == [2, 3]
    assert multiset_difference([1, 2, 3], [1, 1, 1]) == [2, 3]
    assert multiset_difference([1, 2, 2, 3], [1, 2, 2, 3]) == []
    assert multiset_difference([3, 1, 2, 1], [1]) == [3, 2, 1]


def test_multiset_intersection():
    assert multiset_intersection([], []) == []
    assert multiset_intersection([1, 2, 3], []) == []
    assert multiset_intersection([], [1, 2, 3]) == []
    assert multiset_intersection([1, 2, 3], [4, 5, 6]) == []
    assert multiset_intersection([1, 2, 3], [2, 3, 4]) == [2, 3]
    assert multiset_intersection([1, 1, 1, 2], [1, 1, 3]) == [1, 1]
    assert multiset_intersection([1, 2, 2], [2, 2, 2, 2]) == [2, 2]
    assert multiset_intersection([1, 2, 2, 3], [1, 2, 2, 3]) == [1, 2, 2, 3]
    assert multiset_intersection([3, 1, 2, 1], [1, 1, 3]) == [3, 1, 1]

def test_multiset_union():
    assert multiset_union([], []) == []
    assert multiset_union([1, 2], []) == [1, 2]
    assert multiset_union([], [3, 4]) == [3, 4]
    assert multiset_union([1, 2], [3, 4]) == [1, 2, 3, 4]
    assert multiset_union([1, 1, 2], [1, 3]) == [1, 1, 2, 3]
    assert multiset_union([1, 2], [1, 1, 1, 3]) == [1, 2, 1, 1, 3]
    assert multiset_union([2, 2], [2, 2]) == [2, 2]

def test_nprimes():
    assert nprimes(0) == []
    assert nprimes(1) == [2]
    assert nprimes(2) == [2, 3]
    assert nprimes(3) == [2, 3, 5]
    assert nprimes(5) == [2, 3, 5, 7, 11]
    assert nprimes(10) == [2, 3, 5, 7, 11, 13, 17, 19, 23, 29]
    assert len(nprimes(50)) == 50

def test_ones():
    assert ones([[0], [2]], shape=[3]) == [1, 0, 1]
    coords_2d = [[0, 1], [1, 2]]
    expected_2d = [
        [0, 1, 0],  # row 0
        [0, 0, 1]   # row 1
    ]
    assert ones(coords_2d, shape=[2, 3]) == expected_2d
    coords_3d = [[0, 0, 0], [1, 1, 1]]
    expected_3d = [
        [[1, 0], [0, 0]],
        [[0, 0], [0, 1]]
    ]
    assert ones(coords_3d, shape=[2, 2, 2]) == expected_3d
    coords_auto = [[0, 1], [1, 2]]
    expected_auto = [
        [0, 1, 0],
        [0, 0, 1]
    ]
    assert ones(coords_auto) == expected_auto

def test_order():
    assert order(2, 8) == 3
    assert order(3, 9) == 2
    assert order(5, 100) == 2
    assert order(3, 10) == 0
    assert order(0, 5) == 0
    assert order(2, 0) == inf
    assert order(0, 0) == inf
    assert order(1, 5) == inf
    assert order(-1, 5) == inf
    assert order(2, -8) == 3
    assert order(-2, 8) == 3
    assert order(-3, -27) == 3

def test_permutations():
    assert permutations([]) == [[]]
    assert permutations([1]) == [[1]]
    expected_3 = [
        [1, 2, 3], [1, 3, 2],
        [2, 1, 3], [2, 3, 1],
        [3, 1, 2], [3, 2, 1]
    ]
    assert sorted(permutations([1, 2, 3])) == sorted(expected_3)
    expected_str = [['a', 'b'], ['b', 'a']]
    assert sorted(permutations("ab")) == sorted(expected_str)
    expected_dupes = [
        [1, 1, 2], [1, 2, 1],
        [1, 1, 2], [1, 2, 1],
        [2, 1, 1], [2, 1, 1]
    ]
    assert sorted(permutations([1, 1, 2])) == sorted(expected_dupes)

def test_prefixes():
    assert prefixes([]) == []
    assert prefixes([1]) == [[1]]
    assert prefixes([1, 2, 3]) == [[1], [1, 2], [1, 2, 3]]
    assert prefixes("abc") == [['a'], ['a', 'b'], ['a', 'b', 'c']]

def test_prime_factors():
    assert prime_factors(0) == []
    assert prime_factors(1) == []
    assert prime_factors(2) == [2]
    assert prime_factors(13) == [13]
    assert prime_factors(6) == [2, 3]
    assert prime_factors(30) == [2, 3, 5]
    assert prime_factors(8) == [2, 2, 2]
    assert prime_factors(12) == [2, 2, 3]
    assert prime_factors(100) == [2, 2, 5, 5]

def test_random():
    assert random(0) == []
    assert len(random(5)) == 5
    assert len(random(100)) == 100
    result = random(50)
    for value in result:
        assert 0.0 <= float(value) < 1.0
    large_sample = random(1000)
    assert len(set(large_sample)) == 1000

def test_repeat():
    assert repeat([2, 3], ['a', 'b']) == ['a', 'a', 'b', 'b', 'b']
    assert repeat([2], ['a', 'b']) == ['a', 'a', 'b']
    assert repeat([2, 3], ['a']) == ['a', 'a', 1, 1, 1]
    assert repeat([0, 2], ['a', 'b']) == ['b', 'b']

def test_reshape():
    assert reshape(5, 1) == [1, 1, 1, 1, 1]
    assert reshape(5, [1, 2, 3]) == [1, 2, 3, 1, 2]
    assert reshape([3], [1, 2, 3]) == [1, 2, 3]
    assert reshape([-3], [1, 2, 3]) == [3, 2, 1]
    assert reshape([2, 3], [1, 2, 3, 4, 5, 6]) == [
        [1, 2, 3],
        [4, 5, 6]
    ]
    assert reshape([-2, 3], [1, 2, 3, 4, 5, 6]) == [
        [4, 5, 6],
        [1, 2, 3]
    ]
    assert reshape([2, -3], [1, 2, 3, 4, 5, 6]) == [
        [3, 2, 1],
        [6, 5, 4]
    ]
    assert reshape([2, 2], [1, 2]) == [
        [1, 2],
        [1, 2]
    ]

def test_rld():
    assert rld([['a', 3], ['b', 2], ['a', 1]]) == ['a', 'a', 'a', 'b', 'b', 'a']
    assert rld([[5, 4]]) == [5, 5, 5, 5]
    assert rld([]) == []

def test_rle():
    assert rle(['a', 'a', 'a', 'b', 'b', 'a']) == [['a', 3], ['b', 2], ['a', 1]]
    assert rle([5, 5, 5, 5]) == [[5, 4]]
    assert rle([]) == []

def test_shuffle():
    assert shuffle([]) == []
    assert shuffle([42]) == [42]
    original = [1, 2, 3, 4, 5]
    shuffled = shuffle(original)
    assert len(shuffled) == len(original)
    assert sorted(shuffled) == sorted(original)
    assert original == [1, 2, 3, 4, 5]

def test_sliding_window():
    assert sliding_window(2, [1, 2, 3, 4]) == [[1, 2], [2, 3], [3, 4]]
    assert sliding_window(1, [1, 2, 3]) == [[1], [2], [3]]
    assert sliding_window(-2, [1, 2, 3, 4]) == [[2, 1], [3, 2], [4, 3]]
    assert sliding_window(3, [1, 2, 3]) == [[1, 2, 3]]
    assert sliding_window(5, [1, 2, 3]) == []

def test_split():
    assert split(2, [1, 2, 3, 4, 5]) == [[1, 2], [3, 4], [5]]
    assert split(3, "abcdef") == [['a', 'b', 'c'], ['d', 'e', 'f']]

def test_split_at():
    assert split_at(0, [1, 2, 0, 3, 4, 0, 5]) == [[1, 2], [3, 4], [5]]
    assert split_at(9, [1, 2, 3]) == [[1, 2, 3]]
    assert split_at('-', ['-', 'a', 'b', '-']) == [[], ['a', 'b'], []]

def test_split_into():
    assert split_into([1, 2, 3], [1, 2, 3, 4, 5, 6, 7]) == [[1], [2, 3], [4, 5, 6]]
    assert split_into([2, 4], [1, 2, 3]) == [[1, 2], [3]]

def test_sublists():
    assert sublists([]) == []
    assert sorted(sublists([1])) == [[1]]
    result_2 = sublists([1, 2])
    assert len(result_2) == 3
    assert [1] in result_2
    assert [2] in result_2
    assert [1, 2] in result_2
    assert len(sublists([1, 2, 3])) == 6

def test_suffixes():
    assert suffixes([]) == []
    assert suffixes([1]) == [[1]]
    assert suffixes([1, 2, 3]) == [
        [3],
        [2, 3],
        [1, 2, 3]
    ]
    assert suffixes("abc") == [
        ['c'],
        ['b', 'c'],
        ['a', 'b', 'c']
    ]

def test_to_braille():
    blank_matrix = [[0, 0], [0, 0], [0, 0], [0, 0]]
    assert to_braille(blank_matrix) == chr(10240) + '\n'
    full_block = [
        [1, 1],
        [1, 1],
        [1, 1],
        [1, 1]
    ]
    assert to_braille(full_block) == '⣿\n'
    top_left = [
        [1, 0],
        [0, 0],
        [0, 0],
        [0, 0]
    ]
    assert to_braille(top_left) == '⠁\n'


def test_transpose():
    assert transpose([[1, 2], [3, 4]]) == [[1, 3], [2, 4]]
    assert transpose([[1, 2], [3]]) == [[1, 3], [2]]
    matrix = [[1, 2, 3], [4], [5, 6]]
    assert transpose(matrix) == [
        [1, 4, 5],
        [2, 6],
        [3]
    ]


def test_trim():
    assert trim([0], [0, 0, 1, 2, 3, 0]) == [1, 2, 3]
    assert trim(['x', 'y'], ['x', 'y', 'a', 'b', 'y', 'x']) == ['a', 'b']
    assert trim([9], [1, 2, 3]) == [1, 2, 3]
    assert trim([1, 2], [1, 2, 1, 2, 2, 1]) == []
    assert trim([], [1, 2, 3]) == [1, 2, 3]
    assert trim([1], []) == []


def test_unrepeat():
    assert unrepeat([1, 2, 1, 2, 1, 2]) == [1, 2]
    assert unrepeat([1, 2, 3, 4]) == [1, 2, 3, 4]
    assert unrepeat([5, 5, 5, 5]) == [5]
    assert unrepeat("abcabc") == ['a', 'b', 'c']
    assert unrepeat([]) == []


def test_where():
    assert where(0) == []
    assert where(1) == [0]
    assert where([1, 2, 3, 4, 5]) == [0, 1, 1, 2, 2, 2, 3, 3, 3, 3, 4, 4, 4, 4, 4]
    assert where([[1, 2], [3, 4]]) == [
        [0, 0],
        [0, 1],
        [0, 1],
        [1, 0],
        [1, 0],
        [1, 0],
        [1, 1],
        [1, 1],
        [1, 1],
        [1, 1],
    ]
