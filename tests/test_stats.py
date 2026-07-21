from src.stats import bootstrap_ci, permutation_test


def test_bootstrap_ci_all_ones():
    result = bootstrap_ci([1.0] * 10, n_resamples=500, seed=0)
    assert result.mean == 1.0
    assert result.ci_low == 1.0
    assert result.ci_high == 1.0
    assert result.n == 10


def test_bootstrap_ci_mixed_bounds_contain_mean():
    values = [1, 0, 1, 1, 0, 1, 0, 1, 1, 0]
    result = bootstrap_ci(values, n_resamples=1000, seed=0)
    assert result.mean == sum(values) / len(values)
    assert result.ci_low <= result.mean <= result.ci_high


def test_bootstrap_ci_empty_is_nan():
    result = bootstrap_ci([], seed=0)
    assert result.mean != result.mean  # NaN != NaN
    assert result.n == 0


def test_bootstrap_ci_is_deterministic_given_seed():
    values = [1, 0, 1, 1, 0, 0, 1, 0]
    a = bootstrap_ci(values, n_resamples=500, seed=7)
    b = bootstrap_ci(values, n_resamples=500, seed=7)
    assert a == b


def test_permutation_test_identical_distributions_high_p():
    a = [1, 0, 1, 0, 1, 0, 1, 0]
    b = [0, 1, 0, 1, 0, 1, 0, 1]
    p = permutation_test(a, b, n_permutations=1000, seed=0)
    assert p > 0.5


def test_permutation_test_clearly_different_groups_low_p():
    a = [1] * 12
    b = [0] * 12
    p = permutation_test(a, b, n_permutations=1000, seed=0)
    assert p < 0.01


def test_permutation_test_is_deterministic_given_seed():
    a = [1, 0, 1, 1, 0]
    b = [0, 0, 1, 0, 1]
    p1 = permutation_test(a, b, n_permutations=500, seed=3)
    p2 = permutation_test(a, b, n_permutations=500, seed=3)
    assert p1 == p2


def test_permutation_test_empty_input_is_nan():
    p = permutation_test([], [1, 2], seed=0)
    assert p != p
