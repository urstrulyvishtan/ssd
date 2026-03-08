"""
Geometric fan-out for async speculative decoding.

Theorem 12 in the SSD paper derives an optimal geometric fan-out that allocates
more bonus token guesses at positions where acceptance is more likely, instead
of uniform F at every position. This module computes fan_out_list from
acceptance rate α and power-law exponent r (both estimable from
METRICS["accepted_suffix_lens_on_hit"]).
"""
import math
from typing import Sequence as Seq


def compute_geometric_fan_out_list(
    K: int,
    F_0: int,
    alpha: float,
    r: float = 0.5,
) -> list[int]:
    """
    Compute per-position fan-out list of length K+1 using geometric allocation.

    F_k = F_0 * alpha^(k/(1+r))  for k < K
    F_K = F_0 * alpha^(K/(1+r)) * (1-alpha)^(-1/(1+r))

    Ensures sum(fan_out_list) is approximately conserved and allocates more
    guesses at earlier positions where acceptance is higher.

    Args:
        K: Speculation depth (number of draft steps).
        F_0: Base fan-out (e.g. config.async_fan_out).
        alpha: Empirical acceptance rate per position (0 < alpha <= 1).
        r: Power-law exponent for decay (>= 0). Higher r => steeper decay.

    Returns:
        List of K+1 integers (fan-out per position). Sum may differ slightly
        from (K+1)*F_0 due to rounding; callers can renormalize if needed.
    """
    if alpha <= 0 or alpha >= 1:
        return [F_0] * (K + 1)
    inv_r = 1.0 / (1.0 + r)
    out: list[float] = []
    for k in range(K):
        f_k = F_0 * (alpha ** (k * inv_r))
        out.append(max(1.0, f_k))
    f_K = F_0 * (alpha ** (K * inv_r)) * ((1 - alpha) ** (-inv_r))
    out.append(max(1.0, f_K))
    total = sum(out)
    target_sum = (K + 1) * F_0
    if total <= 0:
        return [F_0] * (K + 1)
    # Normalize so sum equals (K+1)*F_0 for drop-in replacement
    scaled = [max(1, round(x * target_sum / total)) for x in out]
    # Fix rounding: ensure sum exactly target_sum
    diff = target_sum - sum(scaled)
    if diff != 0 and scaled:
        idx = K if diff > 0 else 0
        scaled[idx] = max(1, scaled[idx] + diff)
    return scaled


def estimate_alpha_from_metrics(
    accepted_suffix_lens_on_hit: Seq[int],
    K: int,
) -> float | None:
    """
    Estimate acceptance rate α from empirical accepted_suffix_lens_on_hit.

    accepted_suffix_lens_on_hit records the number of accepted tokens (including
    recovery) on cache hits. So (suffix_len - 1) is the number of speculated
    tokens accepted. Mean of (suffix_len - 1) / K gives a simple α estimate.

    Returns:
        Estimated α in (0, 1], or None if no data / invalid.
    """
    if not accepted_suffix_lens_on_hit or K <= 0:
        return None
    # suffix_len includes recovery token; accepted speculated = suffix_len - 1
    accepted_spec = [max(0, min(L - 1, K)) for L in accepted_suffix_lens_on_hit]
    mean_accepted = sum(accepted_spec) / len(accepted_spec)
    alpha = mean_accepted / K
    if alpha <= 0 or alpha > 1:
        return None
    return alpha


def suggest_geometric_fan_out_list(
    accepted_suffix_lens_on_hit: Seq[int],
    K: int,
    F_0: int,
    r: float = 0.5,
) -> list[int] | None:
    """
    Suggest fan_out_list for the next run from empirical cache-hit metrics.

    Fits α from accepted_suffix_lens_on_hit and returns the geometric fan-out
    list. Use this after a warmup or previous run to improve cache hit rate at
    higher temperatures.

    Args:
        accepted_suffix_lens_on_hit: List of accepted suffix lengths on cache hit
            (e.g. METRICS["accepted_suffix_lens_on_hit"]).
        K: config.speculate_k.
        F_0: config.async_fan_out.
        r: Power-law exponent (default 0.5).

    Returns:
        List of K+1 integers, or None if estimation failed (e.g. not enough data).
    """
    alpha = estimate_alpha_from_metrics(accepted_suffix_lens_on_hit, K)
    if alpha is None:
        return None
    return compute_geometric_fan_out_list(K, F_0, alpha, r)
