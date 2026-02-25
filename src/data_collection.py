"""
data_collection.py
------------------
Fetches NHL game-level data from the NHL Stats API (public, no key required).
Collects seasons 2018-19 through 2023-24, excluding 2020-21 (COVID bubble).

Data collected per game:
    - Date, home/away teams, arena, attendance
    - Game type (regular season only)
    - Day of week, month
    - Back-to-back indicator (home and away)
    - Team records at time of game (home/away win%)
    - Rivalry flag (Original Six matchups)
    - Weekend flag
"""

import requests
import pandas as pd
import numpy as np
import time
import json
import os
from datetime import datetime, timedelta

NHL_API = "https://api-web.nhle.com/v1"
LEGACY_API = "https://statsapi.web.nhl.com/api/v1"

SEASONS = ["20182019", "20192020", "20212022", "20222023", "20232024"]

ORIGINAL_SIX = {"BOS", "BUF", "CHI", "DET", "MTL", "NYR", "TOR"}

ARENA_CAPACITY = {
    "TOR": 18800, "MTL": 21105, "BOS": 17850, "NYR": 18006,
    "CHI": 19717, "DET": 19515, "PHI": 19543, "PIT": 18387,
    "WSH": 18573, "NYI": 17255, "NJD": 16514, "CBJ": 19500,
    "CAR": 18680, "FLA": 19250, "TBL": 19092, "OTT": 18652,
    "BUF": 19070, "MIN": 17954, "STL": 18096, "NSH": 17159,
    "WPG": 15321, "DAL": 18532, "COL": 18007, "ARI": 17125,
    "VGK": 17367, "SEA": 17100, "EDM": 18347, "CGY": 19289,
    "VAN": 18910, "SJS": 17435, "ANA": 17174, "LAK": 18230,
}


def fetch_schedule(season: str) -> list[dict]:
    """Pull full regular-season schedule for a given season string e.g. '20232024'."""
    print(f"  Fetching schedule for {season}...")
    url = f"https://statsapi.web.nhl.com/api/v1/schedule?season={season}&gameType=R&expand=schedule.game.seriesSummary"
    resp = requests.get(url, timeout=15)
    resp.raise_for_status()
    data = resp.json()

    games = []
    for date_block in data.get("dates", []):
        for game in date_block.get("games", []):
            games.append(game)
    print(f"    Found {len(games)} games")
    return games


def fetch_game_boxscore(game_id: int) -> dict | None:
    """Fetch boxscore for a specific game to get attendance."""
    url = f"https://statsapi.web.nhl.com/api/v1/game/{game_id}/boxscore"
    try:
        resp = requests.get(url, timeout=10)
        resp.raise_for_status()
        return resp.json()
    except Exception:
        return None


def parse_game(game: dict, season: str) -> dict | None:
    """Extract relevant fields from a raw game dict."""
    try:
        game_id = game["gamePk"]
        game_date = datetime.strptime(game["gameDate"], "%Y-%m-%d")

        home_team = game["teams"]["home"]["team"]["abbreviation"]
        away_team = game["teams"]["away"]["team"]["abbreviation"]
        home_score = game["teams"]["home"].get("score", 0)
        away_score = game["teams"]["away"].get("score", 0)

        # Venue
        venue = game.get("venue", {}).get("name", "Unknown")

        # Attendance from linescore if available
        attendance = game.get("teams", {}).get("home", {}).get("team", {}).get("attendance")

        return {
            "game_id": game_id,
            "season": season,
            "date": game_date,
            "day_of_week": game_date.weekday(),          # 0=Mon, 6=Sun
            "month": game_date.month,
            "is_weekend": int(game_date.weekday() >= 4), # Fri/Sat/Sun
            "home_team": home_team,
            "away_team": away_team,
            "venue": venue,
            "arena_capacity": ARENA_CAPACITY.get(home_team, 18000),
            "home_score": home_score,
            "away_score": away_score,
            "attendance": attendance,
            "is_rivalry": int(
                home_team in ORIGINAL_SIX and away_team in ORIGINAL_SIX
            ),
        }
    except (KeyError, ValueError):
        return None


def compute_team_records(df: pd.DataFrame) -> pd.DataFrame:
    """
    For each game, compute rolling win% for home and away teams
    using only games PRIOR to the current game (no data leakage).
    """
    df = df.sort_values("date").reset_index(drop=True)

    home_wins = {}
    home_games = {}
    away_wins = {}
    away_games = {}

    home_win_pct = []
    away_win_pct = []

    for _, row in df.iterrows():
        ht = row["home_team"]
        at = row["away_team"]

        # Record BEFORE this game
        hw = home_wins.get(ht, 0) + away_wins.get(ht, 0)
        hg = home_games.get(ht, 0) + away_games.get(ht, 0)
        aw = home_wins.get(at, 0) + away_wins.get(at, 0)
        ag = home_games.get(at, 0) + away_games.get(at, 0)

        home_win_pct.append(hw / hg if hg > 0 else 0.5)
        away_win_pct.append(aw / ag if ag > 0 else 0.5)

        # Update after
        home_won = int(row["home_score"] > row["away_score"])
        home_wins[ht] = home_wins.get(ht, 0) + home_won
        home_games[ht] = home_games.get(ht, 0) + 1
        away_wins[at] = away_wins.get(at, 0) + (1 - home_won)
        away_games[at] = away_games.get(at, 0) + 1

    df["home_win_pct"] = home_win_pct
    df["away_win_pct"] = away_win_pct
    df["combined_win_pct"] = (df["home_win_pct"] + df["away_win_pct"]) / 2
    return df


def compute_back_to_back(df: pd.DataFrame) -> pd.DataFrame:
    """Flag games where home or away team played the previous day."""
    df = df.sort_values("date").reset_index(drop=True)
    last_game_date = {}

    home_b2b = []
    away_b2b = []

    for _, row in df.iterrows():
        ht = row["home_team"]
        at = row["away_team"]
        d = row["date"]

        h_last = last_game_date.get(ht)
        a_last = last_game_date.get(at)

        home_b2b.append(int(h_last is not None and (d - h_last).days == 1))
        away_b2b.append(int(a_last is not None and (d - a_last).days == 1))

        last_game_date[ht] = d
        last_game_date[at] = d

    df["home_back_to_back"] = home_b2b
    df["away_back_to_back"] = away_b2b
    return df


def generate_synthetic_data(n_games: int = 3000) -> pd.DataFrame:
    """
    Generate realistic synthetic NHL game data for development/demo purposes.
    Used when the live API is unavailable. Mirrors real NHL distributions.
    """
    np.random.seed(42)
    teams = list(ARENA_CAPACITY.keys())

    records = []
    base_date = datetime(2018, 10, 3)

    for i in range(n_games):
        home = np.random.choice(teams)
        away = np.random.choice([t for t in teams if t != home])
        days_offset = np.random.randint(0, 365 * 5)
        game_date = base_date + timedelta(days=days_offset)

        # Skip COVID year roughly
        if datetime(2020, 3, 12) <= game_date <= datetime(2021, 7, 1):
            continue

        capacity = ARENA_CAPACITY.get(home, 18000)
        is_weekend = int(game_date.weekday() >= 4)
        is_rivalry = int(home in ORIGINAL_SIX and away in ORIGINAL_SIX)
        month = game_date.month
        is_playoffs_month = int(month in [4, 5, 6])

        # Realistic attendance model with noise
        base_fill = 0.88
        base_fill += 0.06 * is_weekend
        base_fill += 0.04 * is_rivalry
        base_fill += 0.03 * is_playoffs_month
        base_fill -= 0.02 * int(game_date.weekday() == 0)  # Monday dip
        base_fill = np.clip(base_fill + np.random.normal(0, 0.05), 0.60, 1.02)
        attendance = int(capacity * base_fill)

        home_win_pct = np.clip(np.random.normal(0.5, 0.12), 0.25, 0.75)
        away_win_pct = np.clip(np.random.normal(0.5, 0.12), 0.25, 0.75)

        records.append({
            "game_id": 2018000000 + i,
            "season": "synthetic",
            "date": game_date,
            "day_of_week": game_date.weekday(),
            "month": month,
            "is_weekend": is_weekend,
            "home_team": home,
            "away_team": away,
            "venue": f"{home} Arena",
            "arena_capacity": capacity,
            "attendance": attendance,
            "is_rivalry": is_rivalry,
            "home_win_pct": home_win_pct,
            "away_win_pct": away_win_pct,
            "combined_win_pct": (home_win_pct + away_win_pct) / 2,
            "home_back_to_back": int(np.random.random() < 0.12),
            "away_back_to_back": int(np.random.random() < 0.12),
            "home_score": np.random.randint(0, 7),
            "away_score": np.random.randint(0, 7),
        })

    return pd.DataFrame(records)


def collect_all_data(use_synthetic: bool = False) -> pd.DataFrame:
    """Main entry point. Returns cleaned DataFrame ready for feature engineering."""

    if use_synthetic:
        print("Using synthetic data for development...")
        df = generate_synthetic_data(3500)
        return df

    all_games = []
    for season in SEASONS:
        try:
            games = fetch_schedule(season)
            for game in games:
                parsed = parse_game(game, season)
                if parsed:
                    all_games.append(parsed)
            time.sleep(0.5)
        except Exception as e:
            print(f"  Warning: could not fetch season {season}: {e}")

    df = pd.DataFrame(all_games)

    # Fetch attendance from boxscores for games missing it
    missing = df[df["attendance"].isna()]["game_id"].tolist()
    print(f"\nFetching attendance for {len(missing)} games missing it...")
    for gid in missing[:200]:  # Cap API calls in demo
        box = fetch_game_boxscore(gid)
        if box:
            att = box.get("gameData", {}).get("game", {}).get("attendance")
            if att:
                df.loc[df["game_id"] == gid, "attendance"] = att
        time.sleep(0.2)

    df = compute_team_records(df)
    df = compute_back_to_back(df)
    df = df.dropna(subset=["attendance"])
    df = df[df["attendance"] > 5000]  # Remove anomalies

    return df


if __name__ == "__main__":
    os.makedirs("../data", exist_ok=True)
    df = collect_all_data(use_synthetic=True)
    df.to_csv("../data/nhl_games_raw.csv", index=False)
    print(f"\nSaved {len(df)} games to data/nhl_games_raw.csv")
    print(df.head())
