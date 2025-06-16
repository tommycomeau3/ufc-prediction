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


def build_recent_form_features(
    df: pd.DataFrame, *, window: int = 5, ewma_span: int = 10
) -> pd.DataFrame:
    """
    Rolling-performance trends and time-decay (EWMA) features.

    For every numeric stat that exists for both corners (e.g. ``AvgSigStrLanded``),
    compute fighter-level rolling means over the **previous** ``window`` bouts as
    well as an exponentially-weighted moving average (EWMA) with span
    ``ewma_span``.  The current bout itself is excluded via ``shift(1)`` to
    prevent target leakage.

    Added columns (for base *SigStrLanded* as an example):

        RedSigStrLanded_roll5 , BlueSigStrLanded_roll5 , SigStrLanded_roll5_diff
        RedSigStrLanded_ewm10 , BlueSigStrLanded_ewm10 , SigStrLanded_ewm10_diff

    Parameters
    ----------
    df : pd.DataFrame
        Input fight-level dataframe sorted in any order (function will sort).
    window : int, default 5
        Rolling window size (number of bouts).
    ewma_span : int, default 10
        Span parameter for the EWMA (≈ half-life 0.7·span).

    Returns
    -------
    pd.DataFrame
        DataFrame with new *_roll{window}_* and *_ewm{span}_* columns.
    """
    if "Date" not in df.columns:
        # No temporal context – skip.
        return df

    shared_bases = _find_shared_numeric_columns(df)
    if not shared_bases:
        return df

    df = df.sort_values("Date").copy()

    for base in shared_bases:
        for corner in ["Red", "Blue"]:
            stat_col = f"{corner}{base}"
            if stat_col not in df.columns:
                continue

            fighter_col = f"{corner}Fighter"

            # Rolling mean of the previous *window* fights
            roll_col = f"{stat_col}_roll{window}"
            df[roll_col] = (
                df.groupby(fighter_col)[stat_col]
                .shift()  # exclude current bout
                .rolling(window=window, min_periods=1)
                .mean()
            )

            # EWMA (time-decay weighting of all past fights)
            ewm_col = f"{stat_col}_ewm{ewma_span}"
            df[ewm_col] = (
                df.groupby(fighter_col)[stat_col]
                .shift()
                .ewm(span=ewma_span, adjust=False)
                .mean()
            )

        # -------------------------------- Diff versions ----------------------------------- #
        red_roll = f"Red{base}_roll{window}"
        blue_roll = f"Blue{base}_roll{window}"
        diff_roll = f"{base}_roll{window}_diff"
        if {red_roll, blue_roll} <= set(df.columns):
            df[diff_roll] = df[red_roll] - df[blue_roll]

        red_ewm = f"Red{base}_ewm{ewma_span}"
        blue_ewm = f"Blue{base}_ewm{ewma_span}"
        diff_ewm = f"{base}_ewm{ewma_span}_diff"
        if {red_ewm, blue_ewm} <= set(df.columns):
            df[diff_ewm] = df[red_ewm] - df[blue_ewm]

    return df


# --------------------------------------------------------------------------- #
# Context-based feature helpers (short notice, altitude)
# --------------------------------------------------------------------------- #
def _add_short_notice_flag(df: pd.DataFrame, threshold: int = 14) -> pd.DataFrame:
    """
    Add ``short_notice_flag`` (1 = at least one fighter accepted on <threshold
    days’ notice).

    Supported schemas
    -----------------
    • RedNoticeDays / BlueNoticeDays  – numeric days
    • NoticeDays                      – same notice for both fighters
    """
    df = df.copy()

    if {"RedNoticeDays", "BlueNoticeDays"} <= set(df.columns):
        df["short_notice_flag"] = (
            (df["RedNoticeDays"] < threshold) | (df["BlueNoticeDays"] < threshold)
        ).astype(int)
    elif "NoticeDays" in df.columns:
        df["short_notice_flag"] = (df["NoticeDays"] < threshold).astype(int)

    return df


def _add_cage_altitude_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add fight-altitude features when ``EventAltitude`` (metres) exists:

        • altitude_diff      – EventAltitude (continuous, same for corners)
        • high_altitude_flag – 1 if altitude ≥1500 m (Denver, Mexico City, etc.)
    """
    if "EventAltitude" not in df.columns:
        return df

    df = df.copy()
    df["altitude_diff"] = df["EventAltitude"]
    df["high_altitude_flag"] = (df["EventAltitude"] >= 1500).astype(int)
    return df

def _add_ratio_features(df: pd.DataFrame) -> pd.DataFrame:
    """Compute ratio-style performance comparisons between Red and Blue fighters.

    Adds three columns when the requisite statistics are available:
        • strikes_lpm_ratio  – RedAvgSigStrLanded / BlueAvgSigStrLanded
        • td_success_ratio   – RedAvgTDPct         / BlueAvgTDPct
        • striking_def_ratio – (100 − RedAvgSigStrPct) / (100 − BlueAvgSigStrPct)

    Division-by-zero is handled by converting zeros to NaN so pandas propagates
    missing values; callers can impute afterwards.
    """
    df = df.copy()

    def _safe_div(num: pd.Series, den: pd.Series) -> pd.Series:
        den = den.replace(0, np.nan)
        return num / den

    # strikes landed per minute ratio
    if {"RedAvgSigStrLanded", "BlueAvgSigStrLanded"} <= set(df.columns):
        df["strikes_lpm_ratio"] = _safe_div(
            df["RedAvgSigStrLanded"], df["BlueAvgSigStrLanded"]
        )

    # takedown success ratio
    if {"RedAvgTDPct", "BlueAvgTDPct"} <= set(df.columns):
        df["td_success_ratio"] = _safe_div(df["RedAvgTDPct"], df["BlueAvgTDPct"])

    # striking defence ratio (higher → better defence for red relative to blue)
    if {"RedAvgSigStrPct", "BlueAvgSigStrPct"} <= set(df.columns):
        red_def = 100 - df["RedAvgSigStrPct"]
        blue_def = 100 - df["BlueAvgSigStrPct"]
        df["striking_def_ratio"] = _safe_div(red_def, blue_def)

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

    # -------------------------------- Ratio comparisons --------------------------------- #
    df = _add_ratio_features(df)
    
    # -------------------------------- Momentum & recent form ---------------------------- #
    df = build_momentum_features(df)
    df = build_recent_form_features(df)
    df = _add_short_notice_flag(df)
    df = _add_cage_altitude_features(df)

    # Capture engineered feature names for downstream use
    global FEATURE_COLS
    FEATURE_COLS = sorted(
        [
            c
            for c in df.columns
            if c.endswith("_diff")
            or c.endswith("_streak")
            or c.endswith("_win_rate")
            or c.endswith("_ratio")
            or c.endswith("_flag")
        ]
    )

    return df