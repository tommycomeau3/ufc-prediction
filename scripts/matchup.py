#!/usr/bin/env python3
"""
scripts/matchup.py

Command-line utility to predict the winner between two UFC fighters using the
trained Stage-1 logistic-regression pipeline.

Usage
-----
python scripts/matchup.py "Conor McGregor" "Khabib Nurmagomedov"
"""
from __future__ import annotations

import argparse
from datetime import date
from pathlib import Path
import sys
import warnings

import joblib
import numpy as np
import pandas as pd

# Ensure project root on sys.path when executed via `python scripts/matchup.py`
sys.path.append(str(Path(__file__).resolve().parents[1]))

from scripts.features import engineer_features  # noqa: E402

# --------------------------------------------------------------------------- #
# Paths
# --------------------------------------------------------------------------- #
RAW_CSV = Path("data/ufc-master.csv")
# Use CSV to avoid optional pyarrow/fastparquet dependency
STATS_CACHE = Path("data/fighter_stats.csv")
MODEL_PATH = Path("models/stage1_logreg_pipeline.pkl")


# --------------------------------------------------------------------------- #
# 1. Fighter-level statistics cache
# --------------------------------------------------------------------------- #
def _build_stats_cache(csv_path: Path) -> pd.DataFrame:
    """
    Aggregate historical fight table into one row per fighter, containing the
    mean of every numeric statistic that appears with both Red*/Blue* prefixes.
    """
    if not csv_path.exists():
        raise FileNotFoundError(f"Raw dataset missing: {csv_path}")

    df = pd.read_csv(csv_path)

    # Identify numeric Red*/Blue* columns
    red_numeric = [
        c for c in df.select_dtypes(include="number").columns if c.startswith("Red")
    ]
    blue_numeric = [
        c for c in df.select_dtypes(include="number").columns if c.startswith("Blue")
    ]
    bases = sorted(
        {c.removeprefix("Red") for c in red_numeric}
        | {c.removeprefix("Blue") for c in blue_numeric}
    )

    stats: dict[str, dict[str, list[float]]] = {}

    for _, row in df.iterrows():
        red_f = row.get("RedFighter")
        blue_f = row.get("BlueFighter")

        if pd.notna(red_f):
            red_dict = stats.setdefault(red_f, {})
            for base in bases:
                val = row.get(f"Red{base}")
                if pd.notna(val):
                    red_dict.setdefault(base, []).append(val)

        if pd.notna(blue_f):
            blue_dict = stats.setdefault(blue_f, {})
            for base in bases:
                val = row.get(f"Blue{base}")
                if pd.notna(val):
                    blue_dict.setdefault(base, []).append(val)

    # Convert lists → means
    records = []
    for fighter, base_map in stats.items():
        record = {"fighter": fighter}
        for base, vals in base_map.items():
            record[base] = float(np.mean(vals)) if vals else np.nan
        records.append(record)

    cache_df = pd.DataFrame.from_records(records)
    cache_df.set_index("fighter", inplace=True)
    return cache_df


def load_stats_cache(force_rebuild: bool = False) -> pd.DataFrame:
    """
    Load fighter-stats cache (CSV) building it on first run or when forced.
    Sticking to CSV sidesteps extra parquet dependencies (pyarrow / fastparquet).
    """
    if force_rebuild or not STATS_CACHE.exists():
        stats_df = _build_stats_cache(RAW_CSV)
        STATS_CACHE.parent.mkdir(parents=True, exist_ok=True)
        stats_df.to_csv(STATS_CACHE)
        return stats_df

    try:
        return pd.read_csv(STATS_CACHE, index_col="fighter")
    except Exception as err:  # corrupted? rebuild once
        warnings.warn(f"Failed to read stats cache ({err}); rebuilding …")
        stats_df = _build_stats_cache(RAW_CSV)
        stats_df.to_csv(STATS_CACHE)
        return stats_df


# --------------------------------------------------------------------------- #
# 2. Matchup-row construction
# --------------------------------------------------------------------------- #
def build_matchup_row(f1: str, f2: str, stats_df: pd.DataFrame) -> pd.DataFrame:
    """Return single-row DataFrame with Red*/Blue* columns populated."""
    # Case-insensitive lookup
    lookup = {name.lower(): name for name in stats_df.index}
    if f1.lower() not in lookup:
        raise KeyError(f"Fighter not found in stats cache: '{f1}'")
    if f2.lower() not in lookup:
        raise KeyError(f"Fighter not found in stats cache: '{f2}'")

    f1_key = lookup[f1.lower()]
    f2_key = lookup[f2.lower()]

    f1_stats = stats_df.loc[f1_key]
    f2_stats = stats_df.loc[f2_key]

    row_data: dict[str, object] = {
        "RedFighter": f1_key,
        "BlueFighter": f2_key,
        # Supply today's date so any date-dependent features are well-defined
        "Date": pd.Timestamp(date.today()),
    }

    for col in stats_df.columns:
        row_data[f"Red{col}"] = f1_stats[col]
        row_data[f"Blue{col}"] = f2_stats[col]

    return pd.DataFrame([row_data])


# --------------------------------------------------------------------------- #
# 3. Prediction routine
# --------------------------------------------------------------------------- #
def predict_matchup(f1: str, f2: str) -> tuple[float, float]:
    """Return (prob_f1_win, prob_f2_win)."""
    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"Trained model not found: {MODEL_PATH}")

    stats_df = load_stats_cache()
    matchup_df = build_matchup_row(f1, f2, stats_df)

    # Engineer features (diffs, ratios, etc.)
    matchup_df = engineer_features(matchup_df)

    # Load persisted pipeline
    pipeline = joblib.load(MODEL_PATH)

    # Align columns with training schema
    if hasattr(pipeline, "feature_names_in_"):
        required_cols = list(pipeline.feature_names_in_)
    else:
        required_cols = list(matchup_df.columns)

    # Add missing columns as NaN
    missing = set(required_cols) - set(matchup_df.columns)
    for col in missing:
        matchup_df[col] = np.nan

    matchup_df = matchup_df[required_cols]

    prob_f1_win = float(pipeline.predict_proba(matchup_df)[:, 1][0])
    return prob_f1_win, 1.0 - prob_f1_win


# --------------------------------------------------------------------------- #
# 4. CLI
# --------------------------------------------------------------------------- #
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Predict win probabilities between two UFC fighters."
    )
    parser.add_argument("fighter1", type=str, help="Name of first fighter (red corner)")
    parser.add_argument("fighter2", type=str, help="Name of second fighter (blue corner)")
    parser.add_argument(
        "--rebuild-cache",
        action="store_true",
        help="Force rebuild of fighter statistics cache.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    try:
        if args.rebuild_cache:
            load_stats_cache(force_rebuild=True)

        p1, p2 = predict_matchup(args.fighter1, args.fighter2)
        winner = args.fighter1 if p1 >= p2 else args.fighter2

        print("🥊  Matchup")
        print(f"    {args.fighter1} vs {args.fighter2}")
        print(f"    P({args.fighter1} wins): {p1:0.2%}")
        print(f"    P({args.fighter2} wins): {p2:0.2%}")
        print(f"\n🏆  Predicted winner: {winner}")

    except KeyError as e:
        sys.exit(str(e))
    except Exception as e:
        sys.exit(f"Error: {e}")


if __name__ == "__main__":
    main()