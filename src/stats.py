"""Bootstrap CI + permutation tests — the statistical-power fixes two
independent external methodology reviews converged on: n=3 frontier seeds
needs a confidence interval and a permutation test to support any "strategy
X beats strategy Y" claim; a bare accuracy point isn't enough.

Two corrections from a later review round (2026-07-22), both of which change
reported numbers rather than just wording:

1. The unpaired `permutation_test` below shuffles the condition label freely
   across two pooled 90-row vectors, which assumes the 90 attempts are
   exchangeable. They are not: the same 30 task items are re-run under every
   condition and across all 3 order-seeds, so each row in `a` has a natural
   partner in `b` (same task, same seed). Empirically the dependence is not
   hypothetical — all 6 wrong answers the three memory conditions produced
   land in the single `ew_*` task family, which is only 18 of the 90 attempts.
   `paired_permutation_test` is the correct test for this design: it permutes
   the condition label *within* each matched (task, seed) pair.
   `permutation_test` is kept for the unpaired case and for regression tests.
2. A percentile bootstrap on a zero-variance sample (e.g. full-history's 90/90)
   resamples to the same value every time and returns a degenerate
   [100%, 100%] interval that carries no information. `BootstrapResult` now
   flags that case, and `wilson_ci` provides the interval to report instead.

Pure stdlib (`random.Random`, seeded) — no numpy — so results are exactly
reproducible across machines without worrying about a different RNG
implementation.
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass


@dataclass
class BootstrapResult:
    mean: float
    ci_low: float
    ci_high: float
    n: int
    degenerate: bool = False


@dataclass
class WilsonResult:
    mean: float
    ci_low: float
    ci_high: float
    n: int


def bootstrap_ci(values: list[float], n_resamples: int = 2000, ci: float = 0.95, seed: int = 0) -> BootstrapResult:
    """Percentile bootstrap CI on the mean of `values` (e.g. 0/1 correctness).

    Sets `degenerate=True` when every value is identical: the interval is then
    a point, and describes only "this sample had no variance", not sampling
    uncertainty. Report `wilson_ci` for those cases instead.
    """
    n = len(values)
    if n == 0:
        return BootstrapResult(mean=float("nan"), ci_low=float("nan"), ci_high=float("nan"), n=0)

    rng = random.Random(seed)
    observed_mean = sum(values) / n
    degenerate = all(v == values[0] for v in values)
    resample_means = []
    for _ in range(n_resamples):
        resample_means.append(sum(values[rng.randrange(n)] for _ in range(n)) / n)
    resample_means.sort()

    alpha = 1 - ci
    lo_idx = max(0, int((alpha / 2) * n_resamples))
    hi_idx = min(n_resamples - 1, int((1 - alpha / 2) * n_resamples) - 1)
    return BootstrapResult(
        mean=observed_mean,
        ci_low=resample_means[lo_idx],
        ci_high=resample_means[hi_idx],
        n=n,
        degenerate=degenerate,
    )


def clustered_bootstrap_ci(
    values: list[float],
    clusters: list[str],
    n_resamples: int = 2000,
    ci: float = 0.95,
    seed: int = 0,
) -> BootstrapResult:
    """Cluster (block) bootstrap: resample whole task clusters, not single rows.

    The plain `bootstrap_ci` resamples the 90 rows independently, which assumes
    90 independent observations. They are not — the 90 come from 30 task items
    seen under 3 orderings, so a row's partners in the other two orderings carry
    almost the same information. Resampling at the *cluster* level (here, one
    cluster per `task_id`) is the standard fix: it draws 30 task-clusters with
    replacement and pools their member rows, so the interval reflects the ~30
    independent units the design actually has rather than an inflated 90. The
    result is a wider, honest CI — the same correction the paired permutation
    test applies to the significance side.

    `clusters` is parallel to `values`: `clusters[i]` is the cluster label of
    `values[i]`. `n` on the result is the cluster count, not the row count.
    """
    if not values:
        return BootstrapResult(mean=float("nan"), ci_low=float("nan"), ci_high=float("nan"), n=0)

    grouped: dict[str, list[float]] = {}
    for v, c in zip(values, clusters):
        grouped.setdefault(c, []).append(v)
    keys = sorted(grouped)
    n_clusters = len(keys)

    rng = random.Random(seed)
    observed_mean = sum(values) / len(values)
    degenerate = all(v == values[0] for v in values)
    resample_means = []
    for _ in range(n_resamples):
        pool: list[float] = []
        for _ in range(n_clusters):
            pool.extend(grouped[keys[rng.randrange(n_clusters)]])
        resample_means.append(sum(pool) / len(pool))
    resample_means.sort()

    alpha = 1 - ci
    lo_idx = max(0, int((alpha / 2) * n_resamples))
    hi_idx = min(n_resamples - 1, int((1 - alpha / 2) * n_resamples) - 1)
    return BootstrapResult(
        mean=observed_mean,
        ci_low=resample_means[lo_idx],
        ci_high=resample_means[hi_idx],
        n=n_clusters,
        degenerate=degenerate,
    )


def wilson_ci(values: list[float], ci: float = 0.95) -> WilsonResult:
    """Wilson score interval for a binomial proportion.

    Unlike the percentile bootstrap, this stays informative at p=0 and p=1 —
    which is the whole reason it's here: full-history scored 90/90, and
    [100%, 100%] is not a defensible interval to publish. Feed it one row per
    *independent unit* (i.e. cluster count, not the 90 rows) so it doesn't
    inherit the same pseudoreplication the cluster bootstrap exists to avoid.
    """
    n = len(values)
    if n == 0:
        return WilsonResult(mean=float("nan"), ci_low=float("nan"), ci_high=float("nan"), n=0)

    # 95% -> 1.96. Inverse normal CDF via the standard rational approximation
    # would be overkill here; the two intervals this project reports are 95%
    # and 90%, so a small table keeps it stdlib-only and exact enough.
    z = {0.90: 1.6449, 0.95: 1.9600, 0.99: 2.5758}.get(round(ci, 2), 1.9600)
    p = sum(values) / n
    denom = 1 + z * z / n
    center = (p + z * z / (2 * n)) / denom
    half = (z / denom) * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n))
    return WilsonResult(mean=p, ci_low=max(0.0, center - half), ci_high=min(1.0, center + half), n=n)


def permutation_test(a: list[float], b: list[float], n_permutations: int = 2000, seed: int = 0) -> float:
    """Two-sided p-value: P(|permuted mean gap| >= |observed mean gap|).

    Null hypothesis: the group label (strategy A vs strategy B) carries no
    information — `a` and `b` are exchangeable. Used to check whether an
    observed accuracy gap between two strategies is bigger than what
    reshuffling the labels would produce by chance alone.
    """
    if not a or not b:
        return float("nan")

    rng = random.Random(seed)
    observed = abs(sum(a) / len(a) - sum(b) / len(b))
    combined = list(a) + list(b)
    n_a = len(a)

    hits = 0
    for _ in range(n_permutations):
        rng.shuffle(combined)
        perm_a = combined[:n_a]
        perm_b = combined[n_a:]
        stat = abs(sum(perm_a) / len(perm_a) - sum(perm_b) / len(perm_b))
        if stat >= observed - 1e-12:
            hits += 1
    return hits / n_permutations


def paired_permutation_test(
    a_by_key: dict[str, float],
    b_by_key: dict[str, float],
    n_permutations: int = 20000,
    seed: int = 0,
) -> float:
    """Two-sided paired permutation test over matched observations.

    `a_by_key` and `b_by_key` map a matching key — here `(task_id, seed)` — to
    that observation's 0/1 correctness under each condition. Only keys present
    in both are used.

    Null hypothesis: within a matched pair, the condition label is arbitrary.
    So the permutation flips the *sign* of each pair's difference rather than
    reshuffling rows across the whole pooled sample. This respects the fact
    that the same 30 task items recur under every condition and in every
    ordering — the structure the unpaired `permutation_test` above destroys.

    Note that only discordant pairs (where the two conditions disagree) carry
    any signal, exactly as in a sign test. With few discordant pairs the
    achievable p-value has a floor: 4 discordant pairs cannot go below 0.125
    two-sided, no matter how lopsided they are. That floor is a real property
    of the design, not a defect of the test, and is why the between-strategy
    comparisons here cannot reach significance.
    """
    keys = sorted(set(a_by_key) & set(b_by_key))
    if not keys:
        return float("nan")

    diffs = [a_by_key[k] - b_by_key[k] for k in keys]
    n = len(diffs)
    observed = abs(sum(diffs) / n)

    rng = random.Random(seed)
    hits = 0
    for _ in range(n_permutations):
        total = 0.0
        for d in diffs:
            total += d if rng.random() < 0.5 else -d
        if abs(total / n) >= observed - 1e-12:
            hits += 1
    return hits / n_permutations


def count_discordant(a_by_key: dict[str, float], b_by_key: dict[str, float]) -> tuple[int, int]:
    """(pairs where a>b, pairs where b>a) — the pairs a paired test actually uses.

    Reported alongside every paired p-value so a reader can see immediately how
    thin the evidence is: "p=0.125 on 4 discordant pairs" is a very different
    statement from "p=0.125 on 400".
    """
    keys = sorted(set(a_by_key) & set(b_by_key))
    a_wins = sum(1 for k in keys if a_by_key[k] > b_by_key[k])
    b_wins = sum(1 for k in keys if b_by_key[k] > a_by_key[k])
    return a_wins, b_wins
