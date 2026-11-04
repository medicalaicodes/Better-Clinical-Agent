"""
Small statistics helpers for comparing methods on the same questions.

Why these: with ~1,200 questions a 2-point accuracy gap can be noise. Wilson intervals show
the uncertainty of each number; McNemar's exact test compares two methods *on the same
questions* (paired), which is far more sensitive than comparing two independent intervals.
"""

import math
from typing import Dict, Sequence, Tuple


def wilson_interval(correct: int, total: int, z: float = 1.96) -> Tuple[float, float]:
    """95% Wilson score interval for a proportion (better than +-1.96*SE near 0 or 1)."""
    if total == 0:
        return (0.0, 0.0)
    p = correct / total
    denom = 1 + z * z / total
    centre = (p + z * z / (2 * total)) / denom
    half = z * math.sqrt(p * (1 - p) / total + z * z / (4 * total * total)) / denom
    return (max(0.0, centre - half), min(1.0, centre + half))


def mcnemar_exact(a_correct: Sequence[bool], b_correct: Sequence[bool]) -> Dict[str, float]:
    """Exact (binomial) McNemar test on paired outcomes.

    Only discordant questions matter: b01 = A wrong & B right, b10 = A right & B wrong.
    Under H0 (no difference) each discordant question is a fair coin flip.
    """
    b01 = sum(1 for a, b in zip(a_correct, b_correct) if not a and b)
    b10 = sum(1 for a, b in zip(a_correct, b_correct) if a and not b)
    n = b01 + b10
    if n == 0:
        return {"b01": 0, "b10": 0, "p_value": 1.0}
    k = min(b01, b10)
    tail = sum(math.comb(n, i) for i in range(k + 1)) / 2 ** n
    return {"b01": b01, "b10": b10, "p_value": min(1.0, 2 * tail)}


def help_harm(baseline_correct: Sequence[bool], method_correct: Sequence[bool]) -> Dict[str, int]:
    """How often a method fixes vs breaks the baseline's answer (e.g. RAG vs closed-book)."""
    stats = mcnemar_exact(baseline_correct, method_correct)
    return {"fixed": stats["b01"], "broke": stats["b10"]}
