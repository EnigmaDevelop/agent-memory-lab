import pytest

from src.stats import (
    bootstrap_ci,
    clustered_bootstrap_ci,
    count_discordant,
    paired_permutation_test,
    permutation_test,
    wilson_ci,
)


def test_bootstrap_ci_all_ones():
    result = bootstrap_ci([1.0] * 10, n_resamples=500, seed=0)
    assert result.mean == 1.0
    assert result.ci_low == 1.0
    assert result.ci_high == 1.0
    assert result.n == 10
    # The whole reason wilson_ci exists: this interval is a point, and says
    # nothing about sampling uncertainty. It must announce that about itself.
    assert result.degenerate is True


def test_bootstrap_ci_mixed_is_not_degenerate():
    result = bootstrap_ci([1, 0, 1, 1, 0], n_resamples=500, seed=0)
    assert result.degenerate is False


def test_wilson_ci_stays_informative_at_p_equals_one():
    """90/90 must not produce [100%, 100%] — that was the reported bug."""
    result = wilson_ci([1.0] * 90)
    assert result.mean == 1.0
    assert result.ci_high == pytest.approx(1.0)  # exact in math, 1-ulp short in float
    assert 0.9 < result.ci_low < 1.0


def test_wilson_ci_stays_informative_at_p_equals_zero():
    result = wilson_ci([0.0] * 24)
    assert result.mean == 0.0
    assert result.ci_low == 0.0
    assert 0.0 < result.ci_high < 0.3


def test_wilson_ci_narrows_as_n_grows():
    small = wilson_ci([1.0] * 10)
    large = wilson_ci([1.0] * 200)
    assert large.ci_low > small.ci_low


def test_wilson_ci_empty_is_nan():
    result = wilson_ci([])
    assert result.mean != result.mean
    assert result.n == 0


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


def test_clustered_bootstrap_reports_cluster_count_as_n():
    # 6 rows, 3 clusters of 2 -> n should be 3, not 6.
    values = [1, 0, 1, 0, 1, 0]
    clusters = ["a", "a", "b", "b", "c", "c"]
    result = clustered_bootstrap_ci(values, clusters, n_resamples=500, seed=0)
    assert result.n == 3
    assert result.mean == 0.5


def test_clustered_bootstrap_is_wider_than_naive_when_clustered():
    """Perfectly correlated clusters must widen the CI vs treating rows as iid.

    Ten task-clusters, each seen 3 times with identical outcome inside the
    cluster. The naive bootstrap sees 30 'independent' rows; the cluster
    bootstrap sees 10 units and must produce a wider interval.
    """
    per_cluster = [1, 1, 1, 1, 1, 0, 0, 0, 0, 0]  # 10 clusters, 50% accuracy
    values, clusters = [], []
    for i, v in enumerate(per_cluster):
        values.extend([v, v, v])
        clusters.extend([f"t{i}", f"t{i}", f"t{i}"])
    naive = bootstrap_ci(values, n_resamples=2000, seed=0)
    clustered = clustered_bootstrap_ci(values, clusters, n_resamples=2000, seed=0)
    naive_width = naive.ci_high - naive.ci_low
    clustered_width = clustered.ci_high - clustered.ci_low
    assert clustered_width > naive_width


def test_clustered_bootstrap_flags_degenerate():
    values = [1.0] * 9
    clusters = ["a", "a", "a", "b", "b", "b", "c", "c", "c"]
    result = clustered_bootstrap_ci(values, clusters, n_resamples=200, seed=0)
    assert result.degenerate is True
    assert result.n == 3


def test_clustered_bootstrap_empty_is_nan():
    result = clustered_bootstrap_ci([], [], seed=0)
    assert result.mean != result.mean
    assert result.n == 0


def test_paired_permutation_test_identical_pairs_is_p_one():
    a = {f"t{i}": 1.0 for i in range(10)}
    b = dict(a)
    assert paired_permutation_test(a, b, n_permutations=500, seed=0) == 1.0


def test_paired_permutation_test_all_pairs_one_direction_is_significant():
    a = {f"t{i}": 1.0 for i in range(30)}
    b = {f"t{i}": 0.0 for i in range(30)}
    assert paired_permutation_test(a, b, n_permutations=2000, seed=0) < 0.001


def test_paired_permutation_test_respects_sign_test_floor():
    """4 discordant pairs cannot reach p<0.05 two-sided, however lopsided.

    This is the property that keeps the between-strategy comparisons honest:
    full-history beat rolling-summary on 4 pairs and lost on none, and that
    still is not significant. An unpaired test over the pooled rows does not
    have this floor, which is why it overstated the evidence.
    """
    a = {f"t{i}": 1.0 for i in range(90)}
    b = dict(a)
    for i in range(4):
        b[f"t{i}"] = 0.0
    p = paired_permutation_test(a, b, n_permutations=20000, seed=0)
    assert p > 0.05
    assert abs(p - 0.125) < 0.02  # exact two-sided sign-test value is 2^-3


def test_paired_permutation_test_ignores_unmatched_keys():
    a = {"t1": 1.0, "t2": 1.0, "orphan": 1.0}
    b = {"t1": 1.0, "t2": 1.0}
    assert paired_permutation_test(a, b, n_permutations=200, seed=0) == 1.0


def test_paired_permutation_test_no_overlap_is_nan():
    p = paired_permutation_test({"a": 1.0}, {"b": 1.0}, seed=0)
    assert p != p


def test_paired_permutation_test_is_deterministic_given_seed():
    a = {"t1": 1.0, "t2": 0.0, "t3": 1.0, "t4": 1.0}
    b = {"t1": 0.0, "t2": 0.0, "t3": 1.0, "t4": 0.0}
    p1 = paired_permutation_test(a, b, n_permutations=500, seed=3)
    p2 = paired_permutation_test(a, b, n_permutations=500, seed=3)
    assert p1 == p2


def test_count_discordant_reports_both_directions():
    a = {"t1": 1.0, "t2": 0.0, "t3": 1.0, "t4": 1.0}
    b = {"t1": 0.0, "t2": 1.0, "t3": 1.0, "t4": 0.0}
    assert count_discordant(a, b) == (2, 1)
