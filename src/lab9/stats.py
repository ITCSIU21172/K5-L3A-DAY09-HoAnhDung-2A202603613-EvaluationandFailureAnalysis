"""Statistics students implement (M2 judge calibration, M4 paired bootstrap).

Rules: a judge is a classifier -- measure it like one. A higher mean is not
"better" until the paired CI of the difference excludes 0.
Reference values to self-check are in tests/test_lab9_stats.py.
"""

from __future__ import annotations

import math
from collections.abc import Sequence

import numpy as np
from scipy.stats import bootstrap

POS = "FAIL"   # the judge's job is to FIND failures, so FAIL is the positive class


def confusion(human: Sequence[str], judge: Sequence[str]) -> dict[str, int]:
    """tp/fp/fn/tn with FAIL as positive."""
    tp = fp = fn = tn = 0
    for h, j in zip(human, judge):
        if h == POS and j == POS:
            tp += 1
        elif h != POS and j == POS:
            fp += 1
        elif h == POS and j != POS:
            fn += 1
        else:
            tn += 1
    return {"tp": tp, "fp": fp, "fn": fn, "tn": tn}


def precision_recall(human: Sequence[str], judge: Sequence[str]) -> tuple[float, float]:
    cm = confusion(human, judge)
    tp, fp, fn = cm["tp"], cm["fp"], cm["fn"]
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    return precision, recall


def cohen_kappa(human: Sequence[str], judge: Sequence[str]) -> float:
    """kappa = (p_o - p_e) / (1 - p_e). Raise ValueError when undefined (p_e == 1)."""
    cm = confusion(human, judge)
    n = len(human)
    if n == 0:
        return 0.0
        
    p_o = (cm["tp"] + cm["tn"]) / n
    p_human_pos = (cm["tp"] + cm["fn"]) / n
    p_judge_pos = (cm["tp"] + cm["fp"]) / n
    p_human_neg = (cm["tn"] + cm["fp"]) / n
    p_judge_neg = (cm["tn"] + cm["fn"]) / n
    
    p_e = (p_human_pos * p_judge_pos) + (p_human_neg * p_judge_neg)
    
    if p_e == 1.0:
        raise ValueError("Undefined Kappa (p_e == 1)")
        
    return (p_o - p_e) / (1.0 - p_e)


def rogan_gladen(observed_rate: float, sensitivity: float, specificity: float) -> float:
    """Correct an observed failure rate for judge error; clip to [0, 1]."""
    denom = sensitivity + specificity - 1.0
    if denom == 0.0:
        return 0.0
    corrected = (observed_rate + specificity - 1.0) / denom
    return max(0.0, min(1.0, corrected))


def paired_bootstrap(a: Sequence[float], b: Sequence[float], *, n_resamples: int = 1000,
                     rng: int = 2027, confidence: float = 0.95) -> tuple[float, float, float]:
    """a[i], b[i] = per-item mean over seeds, SAME items, SAME order.
    Return (mean(a)-mean(b), lo, hi) with scipy.stats.bootstrap"""
    a_arr, b_arr = np.array(a), np.array(b)
    if np.array_equal(a_arr, b_arr):
        return 0.0, 0.0, 0.0
        
    def diff_mean(x, y):
        return np.mean(x) - np.mean(y)
        
    res = bootstrap((a_arr, b_arr), statistic=diff_mean, paired=True, 
                    n_resamples=n_resamples, confidence_level=confidence, 
                    method="percentile", random_state=np.random.default_rng(rng))
                    
    mean_diff = float(diff_mean(a_arr, b_arr))
    return mean_diff, float(res.confidence_interval.low), float(res.confidence_interval.high)


def pass_at_k(n: int, c: int, k: int) -> float:
    """Probability that AT LEAST ONE of k trials drawn from n (c of them passing) passes."""
    if k > n:
        raise ValueError(f"k ({k}) cannot be greater than n ({n})")
    if n - c < k:
        return 1.0
    return 1.0 - (math.comb(n - c, k) / math.comb(n, k))


def pass_hat_k(n: int, c: int, k: int) -> float:
    """pass^k (tau-bench, Yao et al. 2024): probability that ALL k trials drawn from n pass."""
    if k > n:
        raise ValueError(f"k ({k}) cannot be greater than n ({n})")
    if c < k:
        return 0.0
    return math.comb(c, k) / math.comb(n, k)