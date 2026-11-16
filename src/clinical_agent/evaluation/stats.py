"""
Two small statistics helpers.

Why we need them: with ~1,200 questions, a 2-point accuracy difference can easily be luck.
  - The Wilson interval shows how uncertain one accuracy number is.
  - McNemar's test compares two methods on the SAME questions, which is much more sensitive
    than comparing two accuracy numbers, because it only looks at where they disagree.
"""

import math


def wilson_interval(correct, total, z=1.96):
    """The 95% Wilson interval for an accuracy. Unlike the simple "p +- 1.96 * SE" formula,
    it stays sensible near 0% and 100% and for small numbers of questions."""
    if total == 0:
        return (0.0, 0.0)
    p = correct / total
    denominator = 1 + z * z / total
    centre = (p + z * z / (2 * total)) / denominator
    half_width = z * math.sqrt(p * (1 - p) / total + z * z / (4 * total * total)) / denominator
    return (max(0.0, centre - half_width), min(1.0, centre + half_width))


def mcnemar_exact(a_correct, b_correct):
    """Exact McNemar test for two methods answering the same questions.

    Only the questions where they disagree matter:
        b01 = A wrong but B right      (B "fixed" these)
        b10 = A right but B wrong      (B "broke" these)
    If the two methods were equally good, each disagreement would be a fair coin flip, so we
    ask how likely a split at least this uneven would be by chance (a two-sided binomial test).
    """
    b01 = 0
    b10 = 0
    for a, b in zip(a_correct, b_correct):
        if not a and b:
            b01 += 1
        if a and not b:
            b10 += 1

    n = b01 + b10
    if n == 0:
        return {"b01": 0, "b10": 0, "p_value": 1.0}

    smaller = min(b01, b10)
    one_tail = 0
    for i in range(smaller + 1):
        one_tail += math.comb(n, i)
    one_tail = one_tail / 2 ** n
    return {"b01": b01, "b10": b10, "p_value": min(1.0, 2 * one_tail)}
