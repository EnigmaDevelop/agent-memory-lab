"""Bootstrap CI + permutation test — the statistical-power fixes two
independent external methodology reviews converged on: n=3 frontier seeds
needs a confidence interval and a permutation test to support any "strategy
X beats strategy Y" claim; a bare accuracy point isn't enough.

Pure stdlib (`random.Random`, seeded) — no numpy — so results are exactly
reproducible across machines without worrying about a different RNG
implementation.
"""

from __future__ import annotations

import random
from dataclasses import dataclass


@dataclass
class BootstrapResult:
    mean: float
    ci_low: float
    ci_high: float
    n: int


def bootstrap_ci(values: list[float], n_resamples: int = 2000, ci: float = 0.95, seed: int = 0) -> BootstrapResult:
    """Percentile bootstrap CI on the mean of `values` (e.g. 0/1 correctness)."""
    n = len(values)
    if n == 0:
        return BootstrapResult(mean=float("nan"), ci_low=float("nan"), ci_high=float("nan"), n=0)

    rng = random.Random(seed)
    observed_mean = sum(values) / n
    resample_means = []
    for _ in range(n_resamples):
        resample_means.append(sum(values[rng.randrange(n)] for _ in range(n)) / n)
    resample_means.sort()

    alpha = 1 - ci
    lo_idx = max(0, int((alpha / 2) * n_resamples))
    hi_idx = min(n_resamples - 1, int((1 - alpha / 2) * n_resamples) - 1)
    return BootstrapResult(mean=observed_mean, ci_low=resample_means[lo_idx], ci_high=resample_means[hi_idx], n=n)


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
