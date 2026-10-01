"""Shared helpers for the displacement analysis scripts."""

import json


def wilson_ci(successes, total, z=1.96):
    # Wilson score confidence interval for a proportion; returns (rate, lo, hi) in percent
    if total == 0:
        return 0.0, 0.0, 0.0
    p_hat = successes / total
    denom = 1 + z**2 / total
    center = (p_hat + z**2 / (2 * total)) / denom
    margin = z * ((p_hat * (1 - p_hat) / total + z**2 / (4 * total**2)) ** 0.5) / denom
    return p_hat * 100, max(0, center - margin) * 100, min(1, center + margin) * 100


def format_rate(successes, total):
    # Format a rate with Wilson CI
    rate, lo, hi = wilson_ci(successes, total)
    return f"{rate:.1f}% ({successes}/{total}) [CI: {lo:.1f}-{hi:.1f}%]"


def load_json(path):
    # Load JSON, return None on failure
    try:
        with open(path) as f:
            return json.load(f)
    except Exception:
        return None


def classify_behavior(cos_to_src, cos_to_dst, threshold=0.05):
    # Classify whether the robot performed the source task, destination task, or neither.
    # Returns 'source', 'destination', or 'ambiguous'.
    if cos_to_src is None or cos_to_dst is None:
        return 'ambiguous'
    diff = cos_to_src - cos_to_dst
    if diff > threshold:
        return 'source'
    elif diff < -threshold:
        return 'destination'
    else:
        return 'ambiguous'
