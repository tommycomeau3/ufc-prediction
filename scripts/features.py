"""
scripts/features.py

Feature-engineering utilities for UFC fight outcome prediction.

Current functionality
---------------------
• engineer_features(df, drop_original=False)
  – Adds **differential features** for every numeric column that exists
    in both Red- and Blue-prefixed form:
        <col>_diff = Red<col> – Blue<col>
  – Example: ``SigStrLanded_diff = RedAvgSigStrLanded - BlueAvgSigStrLanded``

• Adds `Age_diff = RedAge - BlueAge` when both ages present.

Future tasks (see docs/architecture.md):
• Rolling win-loss ratios, time-decay weighting, “short-notice” flag, etc.
"""

from __future__ import annotations

from typing import List, Tuple

import pandas as pd
import numpy as np


# --------------------------------------------------------------------------- #
# Additional feature helpers
# --------------------------------------------------------------------------- #
def _add_physical_diffs(df: pd.DataFrame, drop_original: bool = False) -> pd.DataFrame:
    """
    Create explicit physical mismatch columns (height, reach, weight, age).

    Returns
    -------
    pd.DataFrame
        DataFrame with new *_diff columns added.
    """
    mapping = {
        "Height": "height_diff",
        "Reach": "reach_diff",
        "Weight": "weight_diff",
        "Age": "age_diff",
    }
    for base, diff_name in mapping.items():
        red_col, blue_col = f"Red{base}", f"Blue{base}"
        if {red_col, blue_col} <= set(df.columns):
            df[diff_name] = df[red_col] - df[blue_col]
            if drop_original:
                df.drop(columns=[red_col, blue_col], inplace=True)
    return df


def build_momentum_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Adds simple momentum-based features:
        • Red_win_streak / Blue_win_streak
        • Red_last3_win_rate / Blue_last3_win_rate
        • Corresponding differential versions.
    The implementation walks chronologically to avoid leakage.
    """
    if "Date" not in df.columns or "Winner" not in df.columns:
        # Not enough information to compute momentum features.
        return df

    df = df.sort_values("Date").copy()

    # Initialise columns
    df["Red_win_streak"] = 0
    df["Blue_win_streak"] = 0
    df["Red_last3_win_rate"] = 0.5
    df["Blue_last3_win_rate"] = 0.5

    streaks: dict[str, int] = {}
    last3: dict[str, list[int]] = {}

    for idx, row in df.iterrows():
        for corner in ["Red", "Blue"]:
            fighter_col = f"{corner}Fighter"
            fighter = row.get(fighter_col)
            if fighter is None:
                continue

            win_hist = last3.get(fighter, [])
            streak_len = streaks.get(fighter, 0)

            if corner == "Red":
                df.at[idx, "Red_win_streak"] = streak_len
                df.at[idx, "Red_last3_win_rate"] = np.mean(win_hist) if win_hist else 0.5
            else:
                df.at[idx, "Blue_win_streak"] = streak_len
                df.at[idx, "Blue_last3_win_rate"] = np.mean(win_hist) if win_hist else 0.5

        # Update history AFTER computing features (no leakage)
        red_f, blue_f = row.get("RedFighter"), row.get("BlueFighter")
        winner = row["Winner"]

        for fighter, is_win in [(red_f, winner == "Red"), (blue_f, winner == "Blue")]:
            if fighter is None:
                continue
            # streak
            streaks[fighter] = streaks.get(fighter, 0) + 1 if is_win else 0
            # last3
            hist = last3.get(fighter, [])
            hist.append(1 if is_win else 0)
            if len(hist) > 3:
                hist.pop(0)
            last3[fighter] = hist

    # differential
    df["win_streak_diff"] = df["Red_win_streak"] - df["Blue_win_streak"]
    df["last3_win_rate_diff"] = df["Red_last3_win_rate"] - df["Blue_last3_win_rate"]
    return df


def build_recent_form_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Placeholder for rolling-performance features (significant strikes, etc.).
    Currently returns the DataFrame unchanged to keep implementation lightweight.
    """
    # TODO: Implement in next sprint
    return df


# Will be populated at runtime inside engineer_features
FEATURE_COLS: List[str] = []


def _find_shared_numeric_columns(df: pd.DataFrame) -> List[str]:
    """
    Return base names that have both Red<base> and Blue<base> numeric columns.
    """
    red_cols = [c for c in df.columns if c.startswith("Red") and df[c].dtype != "object"]
    blue_cols = [c for c in df.columns if c.startswith("Blue") and df[c].dtype != "object"]

    # Strip prefixes
    red_base = {c.removeprefix("Red") for c in red_cols}
    blue_base = {c.removeprefix("Blue") for c in blue_cols}

    shared = sorted(red_base & blue_base)
    return shared


def engineer_features(df: pd.DataFrame, drop_original: bool = False) -> pd.DataFrame:
    """
    Augment `df` with engineered numeric differential features.

    Parameters
    ----------
    df : pd.DataFrame
        Cleaned DataFrame from ``clean_ufc_data``.
    drop_original : bool, default False
        If True, remove the raw Red*/Blue* numeric columns after computing
        the differential features. Non-numeric and categorical columns
        remain untouched.

    Returns
    -------
    pd.DataFrame
        DataFrame with new *_diff columns (and optionally without originals).
    """
    df = df.copy()

    # -------------------------------- Differential stats -------------------------------- #
    for base in _find_shared_numeric_columns(df):
        red_col = f"Red{base}"
        blue_col = f"Blue{base}"
        diff_col = f"{base}_diff"

        df[diff_col] = df[red_col] - df[blue_col]

        if drop_original:
            df.drop(columns=[red_col, blue_col], inplace=True)

    # -------------------------------- Physical mismatches ------------------------------- #
    df = _add_physical_diffs(df, drop_original=drop_original)

    # -------------------------------- Momentum & recent form ---------------------------- #
    df = build_momentum_features(df)
    df = build_recent_form_features(df)

    # Capture engineered feature names for downstream use
    global FEATURE_COLS
    FEATURE_COLS = sorted(
        [c for c in df.columns if c.endswith("_diff") or c.endswith("_streak") or c.endswith("_win_rate")]
    )

    return df