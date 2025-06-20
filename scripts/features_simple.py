"""
scripts/features_simple.py

Simplified feature engineering without rolling/temporal features.
This ensures compatibility between training and prediction data.
"""

from __future__ import annotations

from typing import List
import pandas as pd
import numpy as np


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


def _add_physical_diffs(df: pd.DataFrame, drop_original: bool = False) -> pd.DataFrame:
    """
    Create explicit physical mismatch columns (height, reach, weight, age).
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


def _add_ratio_features(df: pd.DataFrame) -> pd.DataFrame:
    """Compute ratio-style performance comparisons between Red and Blue fighters."""
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

    # striking defence ratio
    if {"RedAvgSigStrPct", "BlueAvgSigStrPct"} <= set(df.columns):
        red_def = 100 - df["RedAvgSigStrPct"]
        blue_def = 100 - df["BlueAvgSigStrPct"]
        df["striking_def_ratio"] = _safe_div(red_def, blue_def)

    return df


def _add_short_notice_flag(df: pd.DataFrame, threshold: int = 14) -> pd.DataFrame:
    """
    Add short_notice_flag (1 = at least one fighter accepted on <threshold days' notice).
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
    Add fight-altitude features when EventAltitude (metres) exists.
    """
    if "EventAltitude" not in df.columns:
        return df

    df = df.copy()
    df["altitude_diff"] = df["EventAltitude"]
    df["high_altitude_flag"] = (df["EventAltitude"] >= 1500).astype(int)
    return df


def build_simple_momentum_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Adds simple momentum-based features WITHOUT rolling windows:
        • Red_win_streak / Blue_win_streak (current streak only)
        • Red_last3_win_rate / Blue_last3_win_rate (simple last 3 fights)
        • Corresponding differential versions.
    """
    if "Date" not in df.columns or "Winner" not in df.columns:
        # Not enough information to compute momentum features.
        return df

    df = df.sort_values("Date").copy()

    # Initialize columns
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


def engineer_features_simple(df: pd.DataFrame, drop_original: bool = False) -> pd.DataFrame:
    """
    Simplified feature engineering WITHOUT rolling/temporal features.
    
    This ensures compatibility between training data and upcoming fight predictions
    by only using features that can be computed for both scenarios.
    
    Parameters
    ----------
    df : pd.DataFrame
        Cleaned DataFrame from clean_ufc_data.
    drop_original : bool, default False
        If True, remove the raw Red*/Blue* numeric columns after computing
        the differential features.

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
    
    # -------------------------------- Simple momentum features -------------------------- #
    df = build_simple_momentum_features(df)
    df = _add_short_notice_flag(df)
    df = _add_cage_altitude_features(df)

    return df