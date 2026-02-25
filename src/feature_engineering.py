"""
feature_engineering.py
-----------------------
Transforms raw game data into ML-ready features.

Feature groups:
    1. Temporal     — day of week, month, weekend, holiday proximity
    2. Team quality — rolling win%, Elo ratings, standings position
    3. Matchup      — rivalry flag, combined quality, head-to-head history
    4. Logistics    — back-to-back, arena capacity, home/away fatigue
    5. Derived      — attendance % of capacity (target), log-attendance
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder


# ── Elo Rating Engine ──────────────────────────────────────────────────────────

INITIAL_ELO = 1500
K_FACTOR = 20


def compute_elo_ratings(df: pd.DataFrame) -> pd.DataFrame:
    """
    Computes pre-game Elo ratings for home and away teams.
    Elo updates after each game using standard formula.
    No data leakage — ratings reflect state BEFORE each game.
    """
    df = df.sort_values("date").reset_index(drop=True)
    elo = {}

    home_elo_pre = []
    away_elo_pre = []

    for _, row in df.iterrows():
        ht = row["home_team"]
        at = row["away_team"]

        he = elo.get(ht, INITIAL_ELO)
        ae = elo.get(at, INITIAL_ELO)

        home_elo_pre.append(he)
        away_elo_pre.append(ae)

        # Expected win probability
        exp_home = 1 / (1 + 10 ** ((ae - he) / 400))
        actual_home = 1.0 if row["home_score"] > row["away_score"] else 0.0

        # Update
        elo[ht] = he + K_FACTOR * (actual_home - exp_home)
        elo[at] = ae + K_FACTOR * ((1 - actual_home) - (1 - exp_home))

    df["home_elo"] = home_elo_pre
    df["away_elo"] = away_elo_pre
    df["elo_diff"] = df["home_elo"] - df["away_elo"]
    df["elo_sum"] = df["home_elo"] + df["away_elo"]
    return df


# ── Temporal Features ──────────────────────────────────────────────────────────

HOLIDAY_DATES = [
    "12-24", "12-25", "12-26", "12-31", "01-01",
    "02-14", "03-17", "11-11",
]

def add_temporal_features(df: pd.DataFrame) -> pd.DataFrame:
    df["date"] = pd.to_datetime(df["date"])
    df["day_of_week"] = df["date"].dt.dayofweek
    df["month"] = df["date"].dt.month
    df["day_of_month"] = df["date"].dt.day
    df["week_of_season"] = ((df["date"].dt.dayofyear - 270) % 365 // 7).clip(0, 30)

    # Weekend: Friday(4), Saturday(5), Sunday(6)
    df["is_weekend"] = (df["day_of_week"] >= 4).astype(int)
    df["is_saturday"] = (df["day_of_week"] == 5).astype(int)

    # Holiday proximity (within 2 days of major holiday)
    df["month_day"] = df["date"].dt.strftime("%m-%d")
    df["near_holiday"] = df["month_day"].isin(HOLIDAY_DATES).astype(int)

    # Season phase
    df["is_early_season"] = (df["month"].isin([10, 11])).astype(int)
    df["is_late_season"] = (df["month"].isin([2, 3])).astype(int)
    df["is_playoff_push"] = (df["month"] == 4).astype(int)

    return df


# ── Matchup Features ───────────────────────────────────────────────────────────

ORIGINAL_SIX = {"BOS", "BUF", "CHI", "DET", "MTL", "NYR", "TOR"}

HISTORIC_RIVALRIES = {
    frozenset({"TOR", "MTL"}), frozenset({"BOS", "NYR"}),
    frozenset({"CHI", "DET"}), frozenset({"PIT", "PHI"}),
    frozenset({"WSH", "PIT"}), frozenset({"EDM", "CGY"}),
    frozenset({"VAN", "CGY"}), frozenset({"TOR", "BOS"}),
}

def add_matchup_features(df: pd.DataFrame) -> pd.DataFrame:
    df["is_original_six"] = (
        df["home_team"].isin(ORIGINAL_SIX) & df["away_team"].isin(ORIGINAL_SIX)
    ).astype(int)

    df["is_historic_rivalry"] = df.apply(
        lambda r: int(frozenset({r["home_team"], r["away_team"]}) in HISTORIC_RIVALRIES), axis=1
    )

    df["combined_win_pct"] = (
        df.get("home_win_pct", 0.5) + df.get("away_win_pct", 0.5)
    ) / 2

    df["win_pct_diff"] = df.get("home_win_pct", 0.5) - df.get("away_win_pct", 0.5)

    return df


# ── Capacity & Derived Target ──────────────────────────────────────────────────

def add_capacity_features(df: pd.DataFrame) -> pd.DataFrame:
    df["attendance_pct"] = (df["attendance"] / df["arena_capacity"]).clip(0.3, 1.05)
    df["log_attendance"] = np.log1p(df["attendance"])
    return df


# ── Team Encoding ──────────────────────────────────────────────────────────────

def encode_teams(df: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    encoders = {}
    for col in ["home_team", "away_team"]:
        le = LabelEncoder()
        df[f"{col}_enc"] = le.fit_transform(df[col].astype(str))
        encoders[col] = le
    return df, encoders


# ── Master Pipeline ────────────────────────────────────────────────────────────

FEATURE_COLS = [
    "day_of_week", "month", "is_weekend", "is_saturday", "near_holiday",
    "is_early_season", "is_late_season", "is_playoff_push",
    "home_win_pct", "away_win_pct", "combined_win_pct", "win_pct_diff",
    "home_elo", "away_elo", "elo_diff", "elo_sum",
    "is_original_six", "is_historic_rivalry",
    "home_back_to_back", "away_back_to_back",
    "arena_capacity",
    "home_team_enc", "away_team_enc",
]

TARGET_COL = "attendance"


def build_features(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    """
    Full feature pipeline. Returns:
        X       — feature matrix
        df      — enriched DataFrame with all engineered columns
        meta    — dict with encoders and feature list
    """
    df = add_temporal_features(df.copy())
    df = compute_elo_ratings(df)
    df = add_matchup_features(df)
    df = add_capacity_features(df)
    df, encoders = encode_teams(df)

    # Drop any rows still missing key features
    df = df.dropna(subset=FEATURE_COLS + [TARGET_COL])

    X = df[FEATURE_COLS]
    meta = {"encoders": encoders, "feature_cols": FEATURE_COLS}
    return X, df, meta


if __name__ == "__main__":
    import sys
    sys.path.append("..")
    from src.data_collection import collect_all_data

    raw = collect_all_data(use_synthetic=True)
    X, enriched, meta = build_features(raw)

    print(f"Feature matrix shape: {X.shape}")
    print(f"Features: {meta['feature_cols']}")
    print(f"\nSample:\n{X.head()}")
    print(f"\nTarget stats:\n{enriched[TARGET_COL].describe()}")
