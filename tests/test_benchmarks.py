"""
test_benchmarks.py
 
Unit tests for benchmarks.py.
 
Tests are self-contained, they build synthetic data that mirrors
merge_results_new_eccm.csv so no real pkl files or CSVs are needed.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd
import pytest

from scripts.benchmarks import (
    score_random,
    score_auc_max,
    precision_at_k,
    evaluate,
    build_pair_df,
    build_summary,
    build_auc_map,
    PRECISION_AT_K,
)

# ===========================================================================
# Shared helpers
# ===========================================================================

def _make_simple_merge_df():
    """
    Three model pairs, each tested at two blend ratios.
    Pair (A,B) and (B,C) improve; pair (C,D) fails.
    Written out explicitly so tests are easy to reason about.
    """
    return pd.DataFrame([
        # pair (A, B) - best improvement is +0.005 at ratio 0.4
        {"model_a": "A", "model_b": "B", "blend_ratio": 0.3,
         "auc_a": 0.90, "auc_b": 0.88, "auc_merged": 0.902,
         "improvement": 0.002, "success": 1,
         "psc": 0.85, "fsc": 0.80, "rsc": 0.75, "eccm": 0.82},
        {"model_a": "A", "model_b": "B", "blend_ratio": 0.4,
         "auc_a": 0.90, "auc_b": 0.88, "auc_merged": 0.905,
         "improvement": 0.005, "success": 1,
         "psc": 0.85, "fsc": 0.80, "rsc": 0.75, "eccm": 0.82},

        # pair (B, C) - best improvement is +0.003 at ratio 0.5
        {"model_a": "B", "model_b": "C", "blend_ratio": 0.4,
         "auc_a": 0.88, "auc_b": 0.91, "auc_merged": 0.882,
         "improvement": 0.002, "success": 1,
         "psc": 0.70, "fsc": 0.65, "rsc": 0.60, "eccm": 0.68},
        {"model_a": "B", "model_b": "C", "blend_ratio": 0.5,
         "auc_a": 0.88, "auc_b": 0.91, "auc_merged": 0.913,
         "improvement": 0.003, "success": 1,
         "psc": 0.70, "fsc": 0.65, "rsc": 0.60, "eccm": 0.68},

        # pair (C, D) - best improvement is -0.001 (merge failed)
        {"model_a": "C", "model_b": "D", "blend_ratio": 0.3,
         "auc_a": 0.91, "auc_b": 0.87, "auc_merged": 0.908,
         "improvement": -0.002, "success": 0,
         "psc": 0.55, "fsc": 0.50, "rsc": 0.45, "eccm": 0.52},
        {"model_a": "C", "model_b": "D", "blend_ratio": 0.4,
         "auc_a": 0.91, "auc_b": 0.87, "auc_merged": 0.909,
         "improvement": -0.001, "success": 0,
         "psc": 0.55, "fsc": 0.50, "rsc": 0.45, "eccm": 0.52},
    ])


def _make_mock_models(merge_df):
    """Fake RF models - only need feature_importances_ for PSC scoring."""
    from unittest.mock import MagicMock
    rng = np.random.default_rng(0)
    model_ids = set(merge_df["model_a"]) | set(merge_df["model_b"])
    return {
        mid: type("M", (), {"feature_importances_": rng.dirichlet(np.ones(10))})()
        for mid in model_ids
    }


# ===========================================================================
# score_random - basic sanity checks
# ===========================================================================

def test_score_random_returns_correct_length():
    assert len(score_random(50)) == 50

def test_score_random_same_seed_gives_same_output():
    # Reproducibility is critical for fair benchmark comparisons
    np.testing.assert_array_equal(score_random(20, seed=42), score_random(20, seed=42))


# ===========================================================================
# score_auc_max - scores a pair by whichever model has higher AUC
# ===========================================================================

def test_score_auc_max_picks_the_stronger_model():
    # Pair (A, B): max(0.90, 0.85) = 0.90
    # Pair (B, C): max(0.85, 0.95) = 0.95
    auc_map = {"A": 0.90, "B": 0.85, "C": 0.95}
    scores = score_auc_max([("A", "B"), ("B", "C")], auc_map)
    assert scores[0] == pytest.approx(0.90)
    assert scores[1] == pytest.approx(0.95)


# ===========================================================================
# build_auc_map - builds a {model_id: auc} lookup from a pairs DataFrame
# ===========================================================================

def test_build_auc_map_stores_correct_values():
    df = pd.DataFrame([
        {"model_a": "X", "model_b": "Y", "auc_a": 0.90, "auc_b": 0.85},
        {"model_a": "Y", "model_b": "Z", "auc_a": 0.87, "auc_b": 0.91},
    ])
    auc_map = build_auc_map(df)
    assert auc_map["X"] == pytest.approx(0.90)
    assert auc_map["Z"] == pytest.approx(0.91)
    assert "Y" in auc_map


# ===========================================================================
# precision_at_k - fraction of top-K pairs that actually improved
# ===========================================================================

def test_precision_at_k_perfect_predictor():
    # All 3 top-scored pairs succeed → P@3 = 1.0
    scores = np.array([0.9, 0.8, 0.7, 0.3, 0.2, 0.1])
    labels = np.array([1,   1,   1,   0,   0,   0])
    assert precision_at_k(scores, labels, k=3) == pytest.approx(1.0)

def test_precision_at_k_worst_predictor():
    # All 3 top-scored pairs fail → P@3 = 0.0
    scores = np.array([0.9, 0.8, 0.7, 0.2, 0.1])
    labels = np.array([0,   0,   0,   1,   1])
    assert precision_at_k(scores, labels, k=3) == pytest.approx(0.0)


# ===========================================================================
# evaluate - computes Spearman r, AUC-ROC, and Precision@K for one method
# ===========================================================================

def test_evaluate_perfect_ranker_gives_spearman_1():
    # Scores = improvements → perfect rank correlation
    improvement = np.array([0.001, 0.002, 0.003, 0.004, 0.005])
    success     = np.ones(5, dtype=int)
    result = evaluate(improvement.copy(), improvement, success)
    assert abs(result["spearman_r"] - 1.0) < 1e-6

def test_evaluate_perfect_separator_gives_auc_roc_1():
    # Scores perfectly split positives from negatives → AUC-ROC = 1.0
    improvement = np.array([-0.002, -0.001,  0.001,  0.002])
    success     = np.array([0,       0,       1,       1])
    result = evaluate(improvement.copy(), improvement, success)
    assert result["auc_roc"] == pytest.approx(1.0)

def test_evaluate_returns_all_expected_keys():
    rng = np.random.default_rng(5)
    improvement = rng.uniform(-0.005, 0.01, 20)
    result = evaluate(rng.uniform(0, 1, 20), improvement, (improvement > 0).astype(int))
    for key in ["spearman_r", "spearman_p", "auc_roc"] + [f"prec@{k}" for k in PRECISION_AT_K]:
        assert key in result, f"Missing key: {key}"


# ===========================================================================
# build_pair_df - collapses to one row per pair, keeps best blend ratio
# ===========================================================================

def test_build_pair_df_one_row_per_pair():
    df = _make_simple_merge_df()
    result = build_pair_df(df, _make_mock_models(df), "test")
    assert len(result) == 3  # 3 unique pairs

def test_build_pair_df_keeps_best_improvement():
    df = _make_simple_merge_df()
    result = build_pair_df(df, _make_mock_models(df), "test")
    ab = result[(result["model_a"] == "A") & (result["model_b"] == "B")]
    assert ab["gt_improvement"].iloc[0] == pytest.approx(0.005)

def test_build_pair_df_success_matches_positive_improvement():
    df = _make_simple_merge_df()
    result = build_pair_df(df, _make_mock_models(df), "test")
    for _, row in result.iterrows():
        expected = 1 if row["gt_improvement"] > 0 else 0
        assert row["gt_success"] == expected

def test_build_pair_df_has_required_columns():
    df = _make_simple_merge_df()
    result = build_pair_df(df, _make_mock_models(df), "test")
    for col in ["score_random", "score_psc", "score_fsc",
                "score_eccm", "score_auc_max", "gt_improvement", "gt_success"]:
        assert col in result.columns, f"Missing column: {col}"


# ===========================================================================
# build_summary - runs all methods, returns one row per method
# ===========================================================================

def test_build_summary_has_all_five_methods():
    df = _make_simple_merge_df()
    pair_df = build_pair_df(df, _make_mock_models(df), "test")
    summary = build_summary(pair_df, "test")
    for method in ["Random", "PSC-only", "FSC-only", "AUC-Max", "ECCM (full)"]:
        assert method in summary["method"].values, f"Missing method: {method}"

def test_build_summary_eccm_beats_random_when_scores_are_perfect():
    # When ECCM scores ≈ actual improvements, its Spearman r must be
    # much higher than random (at least 0.5 higher)
    rng = np.random.default_rng(7)
    n = 20
    improvements = rng.uniform(-0.005, 0.01, n)
    pair_df = pd.DataFrame({
        "gt_improvement": improvements,
        "gt_success":     (improvements > 0).astype(int),
        "score_random":   rng.uniform(0, 1, n),
        "score_psc":      rng.uniform(0.6, 1.0, n),
        "score_fsc":      rng.uniform(0.5, 1.0, n),
        "score_eccm":     improvements + rng.normal(0, 0.0001, n),
        "score_auc_max":  rng.uniform(0.8, 1.0, n),
    })
    summary = build_summary(pair_df, "test")
    eccm_r = summary[summary["method"] == "ECCM (full)"]["spearman_r"].iloc[0]
    rand_r = summary[summary["method"] == "Random"]["spearman_r"].iloc[0]
    assert eccm_r > rand_r + 0.5