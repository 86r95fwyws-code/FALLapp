from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations
from math import exp, factorial
from typing import Iterable, Optional

import numpy as np
import pandas as pd
import re
import unicodedata
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
from concurrent.futures import ThreadPoolExecutor, as_completed
from bs4 import BeautifulSoup


COMPETITIONS = {
    "Eredivisie": {"code": "N1", "country": "Nederland"},
    "Premier League": {"code": "E0", "country": "Engeland"},
    "La Liga": {"code": "SP1", "country": "Spanje"},
    "Bundesliga": {"code": "D1", "country": "Duitsland"},
    "Serie A": {"code": "I1", "country": "Italië"},
    "Ligue 1": {"code": "F1", "country": "Frankrijk"},
}

SEASON_SLUGS = {
    "2023/24": "2324",
    "2024/25": "2425",
    "2025/26": "2526",
    "2026/27": "2627",
}

DEFAULT_HISTORICAL_SEASONS = ["2023/24", "2024/25", "2025/26"]


def dataset_url(competition: str, season: str) -> str:
    code = COMPETITIONS[competition]["code"]
    slug = SEASON_SLUGS[season]
    return f"https://www.football-data.co.uk/mmz4281/{slug}/{code}.csv"


FIXTURES_URL = "https://www.football-data.co.uk/fixtures.csv"
DIVISION_TO_COMPETITION = {v["code"]: k for k, v in COMPETITIONS.items()}

TOTAL_LINES = [0.5, 1.5, 2.5, 3.5, 4.5, 5.5]
TEAM_LINES = TOTAL_LINES


# Football-Data kolommen die we in de Historical Data Explorer herkennen.
MATCH_STAT_FIELDS = [
    "HTHG", "HTAG", "HS", "AS", "HST", "AST", "HHW", "AHW",
    "HC", "AC", "HF", "AF", "HFKC", "AFKC", "HO", "AO",
    "HY", "AY", "HR", "AR", "HBP", "ABP", "Attendance",
]

ODDS_FIELDS = [
    # 1X2 opening
    "B365H", "B365D", "B365A", "PSH", "PSD", "PSA",
    "MaxH", "MaxD", "MaxA", "AvgH", "AvgD", "AvgA",
    # 1X2 closing
    "B365CH", "B365CD", "B365CA", "PSCH", "PSCD", "PSCA",
    "MaxCH", "MaxCD", "MaxCA", "AvgCH", "AvgCD", "AvgCA",
    # totals opening/closing
    "B365>2.5", "B365<2.5", "P>2.5", "P<2.5", "Max>2.5", "Max<2.5", "Avg>2.5", "Avg<2.5",
    "B365C>2.5", "B365C<2.5", "PC>2.5", "PC<2.5", "MaxC>2.5", "MaxC<2.5", "AvgC>2.5", "AvgC<2.5",
    # Asian handicap opening/closing
    "AHh", "B365AHH", "B365AHA", "PAHH", "PAHA", "MaxAHH", "MaxAHA", "AvgAHH", "AvgAHA",
    "AHCh", "B365CAHH", "B365CAHA", "PCAHH", "PCAHA", "MaxCAHH", "MaxCAHA", "AvgCAHH", "AvgCAHA",
]

DERIVED_RAW_FIELDS = [
    "TotalGoals", "FirstHalfGoals", "SecondHalfGoals", "SecondHalfHomeGoals", "SecondHalfAwayGoals",
    "TotalShots", "TotalShotsOnTarget", "TotalCorners", "TotalFouls", "TotalYellow", "TotalRed",
    "HomeShotAccuracy", "AwayShotAccuracy", "Opening1X2Margin", "Closing1X2Margin",
    "OpeningOU25Margin", "ClosingOU25Margin",
]


@dataclass(frozen=True)
class BacktestConfig:
    pseudo_matches: int = 2
    warmup_matches: int = 45  # ongeveer vijf volledige speelrondes
    max_goals: int = 8
    odds_source: str = "closing_avg"


def poisson_pmf(k: int, lam: float) -> float:
    lam = max(float(lam), 1e-12)
    return exp(-lam) * (lam ** k) / factorial(k)


def score_matrix(lam_home: float, lam_away: float, max_goals: int = 8) -> np.ndarray:
    h = np.array([poisson_pmf(i, lam_home) for i in range(max_goals + 1)], dtype=float)
    a = np.array([poisson_pmf(i, lam_away) for i in range(max_goals + 1)], dtype=float)
    matrix = np.outer(h, a)
    total = matrix.sum()
    return matrix / total if total > 0 else matrix


def load_data(
    competitions: Iterable[str],
    seasons: Iterable[str],
    uploaded_files: Optional[dict[tuple[str, str], object]] = None,
) -> pd.DataFrame:
    """Laad Football-Data CSV's voor meerdere competities en seizoenen en verrijk ruwe velden."""
    frames = []
    uploaded_files = uploaded_files or {}

    for competition in competitions:
        if competition not in COMPETITIONS:
            raise ValueError(f"Onbekende competitie: {competition}")
        for season in seasons:
            key = (competition, season)
            if key in uploaded_files and uploaded_files[key] is not None:
                df = pd.read_csv(uploaded_files[key])
            else:
                df = pd.read_csv(dataset_url(competition, season))

            # Sommige CSV's bevatten een BOM in Div; normaliseer alle kolomnamen.
            df = df.copy()
            df.columns = [str(c).replace("\\ufeff", "").strip() for c in df.columns]
            df["Competition"] = competition
            df["Season"] = season
            frames.append(df)

    if not frames:
        return pd.DataFrame()

    data = pd.concat(frames, ignore_index=True, sort=False)

    if "Date" in data.columns:
        data["Date"] = pd.to_datetime(data["Date"], dayfirst=True, errors="coerce")
    else:
        data["Date"] = pd.NaT

    # Alle bekende numerieke bronvelden plus alle odds-/statkolommen die daadwerkelijk aanwezig zijn.
    numeric_candidates = set(["FTHG", "FTAG"] + MATCH_STAT_FIELDS + ODDS_FIELDS)
    # Neem ook overige bookmakers / marktvelden mee zonder vooraf elke bookmakernaam te hardcoden.
    non_numeric = {"Div", "Date", "Time", "HomeTeam", "AwayTeam", "FTR", "HTR", "Referee", "Competition", "Season"}
    for col in data.columns:
        if col not in non_numeric and col not in {"Match"}:
            if any(token in col for token in ["365", "Avg", "Max", "PS", "P>", "P<", "AH", "BFE", "BW", "IW", "WH", "VC", "LB", "CL", "BFD", "BMGM", "BV"]):
                numeric_candidates.add(col)

    for col in numeric_candidates:
        if col in data.columns:
            data[col] = pd.to_numeric(data[col], errors="coerce")

    # Alleen gespeelde wedstrijden; toekomstige fixtures worden apart geladen.
    data = data[data["FTHG"].notna() & data["FTAG"].notna()].copy()
    data["TotalGoals"] = data["FTHG"] + data["FTAG"]
    data["BTTS"] = (data["FTHG"] > 0) & (data["FTAG"] > 0)
    data["Over0_5"] = data["TotalGoals"] > 0.5
    data["Over1_5"] = data["TotalGoals"] > 1.5
    data["Over2_5"] = data["TotalGoals"] > 2.5
    data["Over3_5"] = data["TotalGoals"] > 3.5
    data["Over4_5"] = data["TotalGoals"] > 4.5
    data["Over5_5"] = data["TotalGoals"] > 5.5
    data["Over6_5"] = data["TotalGoals"] > 6.5
    data["HomeResult"] = np.select(
        [data["FTHG"] > data["FTAG"], data["FTHG"] == data["FTAG"]],
        ["Thuiswinst", "Gelijkspel"],
        default="Uitwinst",
    )
    data["Match"] = data["HomeTeam"].astype(str) + " - " + data["AwayTeam"].astype(str)

    if {"HTHG", "HTAG"}.issubset(data.columns):
        data["FirstHalfGoals"] = data["HTHG"] + data["HTAG"]
        data["SecondHalfHomeGoals"] = data["FTHG"] - data["HTHG"]
        data["SecondHalfAwayGoals"] = data["FTAG"] - data["HTAG"]
        data["SecondHalfGoals"] = data["SecondHalfHomeGoals"] + data["SecondHalfAwayGoals"]
        data["HalfTimeResultNL"] = np.select(
            [data["HTHG"] > data["HTAG"], data["HTHG"] == data["HTAG"]],
            ["Thuis voor", "Gelijk"], default="Uit voor"
        )

    paired_totals = {
        "TotalShots": ("HS", "AS"),
        "TotalShotsOnTarget": ("HST", "AST"),
        "TotalCorners": ("HC", "AC"),
        "TotalFouls": ("HF", "AF"),
        "TotalYellow": ("HY", "AY"),
        "TotalRed": ("HR", "AR"),
    }
    for target, (hcol, acol) in paired_totals.items():
        if hcol in data.columns and acol in data.columns:
            data[target] = data[hcol] + data[acol]

    if {"HS", "HST"}.issubset(data.columns):
        data["HomeShotAccuracy"] = np.where(data["HS"] > 0, data["HST"] / data["HS"], np.nan)
    if {"AS", "AST"}.issubset(data.columns):
        data["AwayShotAccuracy"] = np.where(data["AS"] > 0, data["AST"] / data["AS"], np.nan)

    def _margin(cols, target):
        if all(c in data.columns for c in cols):
            valid = data[list(cols)].gt(1).all(axis=1)
            data[target] = np.nan
            data.loc[valid, target] = sum(1 / data.loc[valid, c] for c in cols) - 1

    _margin(("AvgH", "AvgD", "AvgA"), "Opening1X2Margin")
    _margin(("AvgCH", "AvgCD", "AvgCA"), "Closing1X2Margin")
    _margin(("Avg>2.5", "Avg<2.5"), "OpeningOU25Margin")
    _margin(("AvgC>2.5", "AvgC<2.5"), "ClosingOU25Margin")

    sort_cols = [c for c in ["Competition", "Season", "Date", "Time"] if c in data.columns]
    return data.sort_values(sort_cols, kind="stable").reset_index(drop=True)

def competition_goal_summary(data: pd.DataFrame) -> pd.DataFrame:
    if data.empty:
        return pd.DataFrame()
    out = data.groupby(["Competition", "Season"], as_index=False).agg(
        Wedstrijden=("Match", "count"),
        Thuisgoals=("FTHG", "sum"),
        Uitgoals=("FTAG", "sum"),
        Goals_totaal=("TotalGoals", "sum"),
        Gem_thuisgoals=("FTHG", "mean"),
        Gem_uitgoals=("FTAG", "mean"),
        Gem_goals=("TotalGoals", "mean"),
        Over_2_5=("Over2_5", "mean"),
        BTTS=("BTTS", "mean"),
    )
    return out


def team_goal_summary(data: pd.DataFrame) -> pd.DataFrame:
    """Teamtabel met doelpunten voor/tegen, thuis, uit en gecombineerd."""
    if data.empty:
        return pd.DataFrame()

    rows = []
    for (competition, season), g in data.groupby(["Competition", "Season"], sort=False):
        teams = sorted(set(g["HomeTeam"].dropna()) | set(g["AwayTeam"].dropna()))
        for team in teams:
            home = g[g["HomeTeam"] == team]
            away = g[g["AwayTeam"] == team]
            hm, am = len(home), len(away)
            gf_home = float(home["FTHG"].sum())
            gf_away = float(away["FTAG"].sum())
            ga_home = float(home["FTAG"].sum())
            ga_away = float(away["FTHG"].sum())
            rows.append({
                "Competition": competition,
                "Season": season,
                "Team": team,
                "Thuiswedstrijden": hm,
                "Uitwedstrijden": am,
                "Wedstrijden_totaal": hm + am,
                "Goals_voor_thuis": gf_home,
                "Goals_voor_uit": gf_away,
                "Goals_voor_totaal": gf_home + gf_away,
                "Gem_goals_voor_thuis": gf_home / hm if hm else np.nan,
                "Gem_goals_voor_uit": gf_away / am if am else np.nan,
                "Gem_goals_voor_totaal": (gf_home + gf_away) / (hm + am) if hm + am else np.nan,
                "Goals_tegen_thuis": ga_home,
                "Goals_tegen_uit": ga_away,
                "Goals_tegen_totaal": ga_home + ga_away,
                "Gem_goals_tegen_thuis": ga_home / hm if hm else np.nan,
                "Gem_goals_tegen_uit": ga_away / am if am else np.nan,
                "Gem_goals_tegen_totaal": (ga_home + ga_away) / (hm + am) if hm + am else np.nan,
                "Doelsaldo": (gf_home + gf_away) - (ga_home + ga_away),
            })
    return pd.DataFrame(rows)


def add_team_perspective(data: pd.DataFrame, team: Optional[str]) -> pd.DataFrame:
    """Voeg team-voor/tegen velden toe, onafhankelijk van of het gekozen team thuis of uit speelde."""
    x = data.copy()
    if not team or team == "Alle teams" or x.empty:
        return x

    is_home = x["HomeTeam"] == team
    x["TeamVenue"] = np.where(is_home, "Thuis", "Uit")

    pairs = {
        "TeamGoalsFor": ("FTHG", "FTAG"),
        "TeamGoalsAgainst": ("FTAG", "FTHG"),
        "TeamHTGoalsFor": ("HTHG", "HTAG"),
        "TeamHTGoalsAgainst": ("HTAG", "HTHG"),
        "TeamShots": ("HS", "AS"),
        "OpponentShots": ("AS", "HS"),
        "TeamShotsOnTarget": ("HST", "AST"),
        "OpponentShotsOnTarget": ("AST", "HST"),
        "TeamCorners": ("HC", "AC"),
        "OpponentCorners": ("AC", "HC"),
        "TeamFouls": ("HF", "AF"),
        "OpponentFouls": ("AF", "HF"),
        "TeamYellow": ("HY", "AY"),
        "OpponentYellow": ("AY", "HY"),
        "TeamRed": ("HR", "AR"),
        "OpponentRed": ("AR", "HR"),
    }
    for target, (home_col, away_col) in pairs.items():
        if home_col in x.columns and away_col in x.columns:
            x[target] = np.where(is_home, x[home_col], x[away_col])

    if {"TeamShots", "TeamShotsOnTarget"}.issubset(x.columns):
        x["TeamShotAccuracy"] = np.where(x["TeamShots"] > 0, x["TeamShotsOnTarget"] / x["TeamShots"], np.nan)
    if {"OpponentShots", "OpponentShotsOnTarget"}.issubset(x.columns):
        x["OpponentShotAccuracy"] = np.where(x["OpponentShots"] > 0, x["OpponentShotsOnTarget"] / x["OpponentShots"], np.nan)
    return x


def filter_history(
    data: pd.DataFrame,
    competitions: Optional[Iterable[str]] = None,
    seasons: Optional[Iterable[str]] = None,
    team: Optional[str] = None,
    venue: str = "Alles",
    min_total_goals: Optional[int] = None,
    max_total_goals: Optional[int] = None,
    result: str = "Alles",
    btts: str = "Alles",
    goal_market: str = "Alles",
    half_time_result: str = "Alles",
    numeric_filters: Optional[dict[str, tuple[Optional[float], Optional[float]]]] = None,
) -> pd.DataFrame:
    x = data.copy()
    if competitions:
        x = x[x["Competition"].isin(list(competitions))]
    if seasons:
        x = x[x["Season"].isin(list(seasons))]
    if team and team != "Alle teams":
        if venue == "Thuis":
            x = x[x["HomeTeam"] == team]
        elif venue == "Uit":
            x = x[x["AwayTeam"] == team]
        else:
            x = x[(x["HomeTeam"] == team) | (x["AwayTeam"] == team)]
        x = add_team_perspective(x, team)
    if min_total_goals is not None:
        x = x[x["TotalGoals"] >= min_total_goals]
    if max_total_goals is not None:
        x = x[x["TotalGoals"] <= max_total_goals]
    if result != "Alles":
        x = x[x["HomeResult"] == result]
    if btts == "Ja":
        x = x[x["BTTS"]]
    elif btts == "Nee":
        x = x[~x["BTTS"]]

    if goal_market != "Alles":
        try:
            direction, raw_line = goal_market.split()
            line = float(raw_line)
            if direction == "Over":
                x = x[x["TotalGoals"] > line]
            elif direction == "Under":
                x = x[x["TotalGoals"] < line]
        except Exception:
            pass

    if half_time_result != "Alles" and "HalfTimeResultNL" in x.columns:
        x = x[x["HalfTimeResultNL"] == half_time_result]

    for col, bounds in (numeric_filters or {}).items():
        if col not in x.columns:
            continue
        low, high = bounds
        values = pd.to_numeric(x[col], errors="coerce")
        mask = values.notna()
        if low is not None:
            mask &= values >= float(low)
        if high is not None:
            mask &= values <= float(high)
        x = x[mask]
    return x

def load_fixtures(source=None) -> pd.DataFrame:
    """Laad het actuele Football-Data fixtures-bestand en beperk tot onze zes competities."""
    df = pd.read_csv(source if source is not None else FIXTURES_URL)
    df = df.copy()
    if "Date" in df.columns:
        df["Date"] = pd.to_datetime(df["Date"], dayfirst=True, errors="coerce")
    if "Div" in df.columns:
        df["Competition"] = df["Div"].map(DIVISION_TO_COMPETITION)
        df = df[df["Competition"].notna()].copy()
    else:
        df["Competition"] = np.nan
    df["ISOWeek"] = df["Date"].dt.isocalendar().week.astype("Int64")
    df["Match"] = df["HomeTeam"].astype(str) + " - " + df["AwayTeam"].astype(str)
    return df.sort_values(["Date", "Time"] if "Time" in df.columns else ["Date"], kind="stable").reset_index(drop=True)


def _last_n_team_matches(data: pd.DataFrame, team: str, n: int) -> pd.DataFrame:
    x = data[(data["HomeTeam"] == team) | (data["AwayTeam"] == team)].sort_values("Date")
    return x.tail(max(int(n), 1))


def select_prediction_history(
    data: pd.DataFrame,
    competition: str,
    fixture_date,
    home_team: str,
    away_team: str,
    mode: str = "Afgelopen 3 volledige seizoenen",
    n_matches: int = 10,
):
    """Maak baseline + teamspecifieke historie voor een komende wedstrijd."""
    x = data[data["Competition"] == competition].copy()
    if pd.notna(fixture_date):
        x = x[x["Date"] < pd.Timestamp(fixture_date)]

    # Baseline is de league-context waarmee aanval/verdediging wordt genormaliseerd.
    if mode == "Afgelopen 3 volledige seizoenen":
        base = x[x["Season"].isin(["2023/24", "2024/25", "2025/26"])].copy()
        home_hist = base
        away_hist = base
    elif mode == "Afgelopen 2 volledige seizoenen":
        base = x[x["Season"].isin(["2024/25", "2025/26"])].copy()
        home_hist = base
        away_hist = base
    elif mode == "Alleen vorig seizoen":
        base = x[x["Season"] == "2025/26"].copy()
        home_hist = base
        away_hist = base
    elif mode == "Huidig seizoen":
        base = x[x["Season"] == "2026/27"].copy()
        home_hist = base
        away_hist = base
    elif mode.startswith("Laatste"):
        # Voor de competitie-baseline nemen we huidig + vorig seizoen; de teamsterktes
        # zelf komen uit exact N recente wedstrijden per team.
        base = x[x["Season"].isin(["2025/26", "2026/27"])].copy()
        if base.empty:
            base = x.tail(380)
        home_hist = _last_n_team_matches(x, home_team, n_matches)
        away_hist = _last_n_team_matches(x, away_team, n_matches)
    else:
        base = x.copy()
        home_hist = base
        away_hist = base

    if base.empty:
        base = x.copy()
    return base, home_hist, away_hist


def predict_fixture(
    data: pd.DataFrame,
    competition: str,
    fixture_date,
    home_team: str,
    away_team: str,
    mode: str = "Afgelopen 3 volledige seizoenen",
    n_matches: int = 10,
    pseudo: int = 2,
    max_goals: int = 8,
):
    # Seizoen-catalogus en Football-Data kunnen verschillende clubnamen gebruiken.
    model_home_team = resolve_catalog_team_name(data, competition, home_team)
    model_away_team = resolve_catalog_team_name(data, competition, away_team)

    base, home_hist, away_hist = select_prediction_history(
        data, competition, fixture_date, model_home_team, model_away_team, mode, n_matches
    )
    if base.empty:
        avg_home, avg_away = 1.5, 1.25
    else:
        avg_home = float(base["FTHG"].mean())
        avg_away = float(base["FTAG"].mean())
        avg_home = avg_home if avg_home > 0 else 1.5
        avg_away = avg_away if avg_away > 0 else 1.25

    hs = _team_stats(home_hist, model_home_team, avg_home, avg_away, pseudo)
    aas = _team_stats(away_hist, model_away_team, avg_home, avg_away, pseudo)
    lam_home = float(np.clip(hs["home_attack"] * aas["away_defence"] * avg_home, 0.05, 6.0))
    lam_away = float(np.clip(aas["away_attack"] * hs["home_defence"] * avg_away, 0.05, 6.0))
    matrix = score_matrix(lam_home, lam_away, max_goals)

    p_home = float(np.tril(matrix, -1).sum())
    p_draw = float(np.trace(matrix))
    p_away = float(np.triu(matrix, 1).sum())
    p_btts_no = float(matrix[0, :].sum() + matrix[:, 0].sum() - matrix[0, 0])
    p_btts = 1.0 - p_btts_no
    p_over25 = float(sum(matrix[i, j] for i in range(matrix.shape[0]) for j in range(matrix.shape[1]) if i + j > 2.5))
    p_over15 = float(sum(matrix[i, j] for i in range(matrix.shape[0]) for j in range(matrix.shape[1]) if i + j > 1.5))
    p_over35 = float(sum(matrix[i, j] for i in range(matrix.shape[0]) for j in range(matrix.shape[1]) if i + j > 3.5))

    return {
        "Competition": competition,
        "Date": pd.Timestamp(fixture_date) if pd.notna(fixture_date) else pd.NaT,
        "HomeTeam": home_team,
        "AwayTeam": away_team,
        "HistoryMode": mode,
        "NMatches": n_matches,
        "HistoryRows": len(base),
        "HomeHistoryRows": len(home_hist[(home_hist["HomeTeam"] == home_team) | (home_hist["AwayTeam"] == home_team)]) if not home_hist.empty else 0,
        "AwayHistoryRows": len(away_hist[(away_hist["HomeTeam"] == away_team) | (away_hist["AwayTeam"] == away_team)]) if not away_hist.empty else 0,
        "lambda_home": lam_home,
        "lambda_away": lam_away,
        "P_Home": p_home,
        "P_Draw": p_draw,
        "P_Away": p_away,
        "P_Over1_5": p_over15,
        "P_Over2_5": p_over25,
        "P_Over3_5": p_over35,
        "P_BTTS": p_btts,
    }



def _numeric(row: pd.Series, names: list[str]) -> float:
    for name in names:
        if name in row.index:
            value = pd.to_numeric(pd.Series([row[name]]), errors="coerce").iloc[0]
            if pd.notna(value) and float(value) > 1:
                return float(value)
    return np.nan


def market_odds(row: pd.Series, market: str, odds_source: str = "closing_avg") -> float:
    """
    Geeft een echte historische bookmakerprijs wanneer Football-Data die markt bevat.
    Voor overige markten retourneert de functie NaN.
    """
    maps = {
        "closing_avg": {
            "HOME": ["AvgCH", "PSCH", "B365CH", "AvgH", "B365H"],
            "DRAW": ["AvgCD", "PSCD", "B365CD", "AvgD", "B365D"],
            "AWAY": ["AvgCA", "PSCA", "B365CA", "AvgA", "B365A"],
            "TOTAL_OVER_2.5": ["AvgC>2.5", "PC>2.5", "B365C>2.5", "Avg>2.5", "B365>2.5"],
            "TOTAL_UNDER_2.5": ["AvgC<2.5", "PC<2.5", "B365C<2.5", "Avg<2.5", "B365<2.5"],
        },
        "closing_max": {
            "HOME": ["MaxCH", "AvgCH", "PSCH", "B365CH", "MaxH", "AvgH"],
            "DRAW": ["MaxCD", "AvgCD", "PSCD", "B365CD", "MaxD", "AvgD"],
            "AWAY": ["MaxCA", "AvgCA", "PSCA", "B365CA", "MaxA", "AvgA"],
            "TOTAL_OVER_2.5": ["MaxC>2.5", "AvgC>2.5", "PC>2.5", "B365C>2.5", "Max>2.5"],
            "TOTAL_UNDER_2.5": ["MaxC<2.5", "AvgC<2.5", "PC<2.5", "B365C<2.5", "Max<2.5"],
        },
        "opening_avg": {
            "HOME": ["AvgH", "B365H", "PSH"],
            "DRAW": ["AvgD", "B365D", "PSD"],
            "AWAY": ["AvgA", "B365A", "PSA"],
            "TOTAL_OVER_2.5": ["Avg>2.5", "B365>2.5", "P>2.5"],
            "TOTAL_UNDER_2.5": ["Avg<2.5", "B365<2.5", "P<2.5"],
        },
        "opening_max": {
            "HOME": ["MaxH", "AvgH", "B365H", "PSH"],
            "DRAW": ["MaxD", "AvgD", "B365D", "PSD"],
            "AWAY": ["MaxA", "AvgA", "B365A", "PSA"],
            "TOTAL_OVER_2.5": ["Max>2.5", "Avg>2.5", "B365>2.5", "P>2.5"],
            "TOTAL_UNDER_2.5": ["Max<2.5", "Avg<2.5", "B365<2.5", "P<2.5"],
        },
    }
    names = maps.get(odds_source, maps["closing_avg"]).get(market, [])
    return _numeric(row, names)


def _market_probs_from_odds(row: pd.Series, odds_source: str) -> dict[str, float]:
    """
    No-vig marktprobabilities voor 1X2 en O/U 2.5.
    """
    result = {}

    triple = {
        k: market_odds(row, k, odds_source)
        for k in ["HOME", "DRAW", "AWAY"]
    }
    if all(pd.notna(v) for v in triple.values()):
        inv = {k: 1.0 / v for k, v in triple.items()}
        s = sum(inv.values())
        result.update({k: inv[k] / s for k in inv})

    pair = {
        k: market_odds(row, k, odds_source)
        for k in ["TOTAL_OVER_2.5", "TOTAL_UNDER_2.5"]
    }
    if all(pd.notna(v) for v in pair.values()):
        inv = {k: 1.0 / v for k, v in pair.items()}
        s = sum(inv.values())
        result.update({k: inv[k] / s for k in inv})

    return result


def _team_stats(history: pd.DataFrame, team: str, avg_home: float, avg_away: float, pseudo: int):
    home = history[history["HomeTeam"] == team]
    away = history[history["AwayTeam"] == team]

    hm = len(home)
    am = len(away)

    hgf = float(home["FTHG"].sum())
    hga = float(home["FTAG"].sum())
    agf = float(away["FTAG"].sum())
    aga = float(away["FTHG"].sum())

    # Exact dezelfde pseudo-logica als het Excel-model:
    h_gf_avg = (hgf + avg_home * pseudo) / (hm + pseudo) if hm + pseudo > 0 else avg_home
    h_ga_avg = (hga + avg_away * pseudo) / (hm + pseudo) if hm + pseudo > 0 else avg_away
    a_gf_avg = (agf + avg_away * pseudo) / (am + pseudo) if am + pseudo > 0 else avg_away
    a_ga_avg = (aga + avg_home * pseudo) / (am + pseudo) if am + pseudo > 0 else avg_home

    return {
        "home_attack": h_gf_avg / avg_home if avg_home > 0 else 1.0,
        "home_defence": h_ga_avg / avg_away if avg_away > 0 else 1.0,
        "away_attack": a_gf_avg / avg_away if avg_away > 0 else 1.0,
        "away_defence": a_ga_avg / avg_home if avg_home > 0 else 1.0,
        "home_matches": hm,
        "away_matches": am,
    }


def predict_match(history: pd.DataFrame, home_team: str, away_team: str, pseudo: int = 2, max_goals: int = 8):
    if history.empty:
        avg_home, avg_away = 1.5, 1.25
    else:
        avg_home = float(history["FTHG"].mean())
        avg_away = float(history["FTAG"].mean())
        avg_home = avg_home if avg_home > 0 else 1.5
        avg_away = avg_away if avg_away > 0 else 1.25

    hs = _team_stats(history, home_team, avg_home, avg_away, pseudo)
    as_ = _team_stats(history, away_team, avg_home, avg_away, pseudo)

    lam_home = hs["home_attack"] * as_["away_defence"] * avg_home
    lam_away = as_["away_attack"] * hs["home_defence"] * avg_away

    lam_home = float(np.clip(lam_home, 0.05, 6.0))
    lam_away = float(np.clip(lam_away, 0.05, 6.0))

    matrix = score_matrix(lam_home, lam_away, max_goals)
    return lam_home, lam_away, matrix


def _event_rows(
    row: pd.Series,
    lam_home: float,
    lam_away: float,
    matrix: np.ndarray,
    odds_source: str,
) -> list[dict]:
    hg, ag = int(row["FTHG"]), int(row["FTAG"])
    total = hg + ag
    n = matrix.shape[0]

    base = {
        "Competition": row["Competition"],
        "Season": row["Season"],
        "Date": row["Date"],
        "HomeTeam": row["HomeTeam"],
        "AwayTeam": row["AwayTeam"],
        "FTHG": hg,
        "FTAG": ag,
        "lambda_home": lam_home,
        "lambda_away": lam_away,
    }

    market_probs = _market_probs_from_odds(row, odds_source)
    out = []

    def add(market, label, prob, won, line=np.nan):
        odd = market_odds(row, market, odds_source)
        market_prob = market_probs.get(market, np.nan)
        fair_odd = (1.0 / prob) if prob > 0 else np.nan
        actual_profit = (odd - 1.0) if (pd.notna(odd) and won) else (-1.0 if pd.notna(odd) else np.nan)
        fair_profit = (fair_odd - 1.0) if won else -1.0

        out.append({
            **base,
            "Market": market,
            "Bet": label,
            "Line": line,
            "ModelProb": prob,
            "FairOdd": fair_odd,
            "BookmakerOdd": odd,
            "MarketProbNoVig": market_prob,
            "EdgePP": (prob - market_prob) if pd.notna(market_prob) else np.nan,
            "ModelEV": (prob * odd - 1.0) if pd.notna(odd) else np.nan,
            "Won": bool(won),
            "ActualProfit1u": actual_profit,
            "FairProfit1u": fair_profit,
        })

    # 1X2
    p_home = float(np.tril(matrix, -1).sum())
    p_draw = float(np.trace(matrix))
    p_away = float(np.triu(matrix, 1).sum())
    add("HOME", "Thuiswinst", p_home, hg > ag)
    add("DRAW", "Gelijkspel", p_draw, hg == ag)
    add("AWAY", "Uitwinst", p_away, hg < ag)

    # BTTS
    p_btts_no = float(matrix[0, :].sum() + matrix[:, 0].sum() - matrix[0, 0])
    p_btts_yes = 1.0 - p_btts_no
    add("BTTS_YES", "BTTS - Ja", p_btts_yes, hg > 0 and ag > 0)
    add("BTTS_NO", "BTTS - Nee", p_btts_no, hg == 0 or ag == 0)

    # Totale goals
    for line in TOTAL_LINES:
        p_under = 0.0
        for i in range(n):
            for j in range(n):
                if i + j < line:
                    p_under += matrix[i, j]
        p_over = 1.0 - p_under
        market_o = "TOTAL_OVER_2.5" if line == 2.5 else f"TOTAL_OVER_{line}"
        market_u = "TOTAL_UNDER_2.5" if line == 2.5 else f"TOTAL_UNDER_{line}"
        add(market_o, f"Over {line:g} goals", p_over, total > line, line)
        add(market_u, f"Under {line:g} goals", p_under, total < line, line)

    # Teamgoals via marginale Poisson-kansen
    max_calc = max(20, int(np.ceil(max(lam_home, lam_away) + 10)))
    home_dist = np.array([poisson_pmf(k, lam_home) for k in range(max_calc + 1)])
    away_dist = np.array([poisson_pmf(k, lam_away) for k in range(max_calc + 1)])
    home_dist = home_dist / home_dist.sum()
    away_dist = away_dist / away_dist.sum()

    for line in TEAM_LINES:
        cutoff = int(np.floor(line))
        p_home_under = float(home_dist[: cutoff + 1].sum())
        p_away_under = float(away_dist[: cutoff + 1].sum())

        add(f"HOME_OVER_{line}", f"Thuisteam over {line:g}", 1 - p_home_under, hg > line, line)
        add(f"HOME_UNDER_{line}", f"Thuisteam under {line:g}", p_home_under, hg < line, line)
        add(f"AWAY_OVER_{line}", f"Uitteam over {line:g}", 1 - p_away_under, ag > line, line)
        add(f"AWAY_UNDER_{line}", f"Uitteam under {line:g}", p_away_under, ag < line, line)

    # Exact scores 0-0 t/m max_goals-max_goals
    for i in range(n):
        for j in range(n):
            add(f"SCORE_{i}_{j}", f"Exact {i}-{j}", float(matrix[i, j]), hg == i and ag == j)

    return out


def build_backtest(data: pd.DataFrame, config: BacktestConfig):
    all_bets = []
    match_rows = []

    for (competition, season), season_df in data.groupby(["Competition", "Season"], sort=False):
        season_df = season_df.sort_values(
            ["Date"] + (["Time"] if "Time" in season_df.columns else []),
            kind="stable",
        ).reset_index(drop=True)

        for idx, row in season_df.iterrows():
            if idx < config.warmup_matches:
                continue

            history = season_df.iloc[:idx]
            lam_home, lam_away, matrix = predict_match(
                history,
                str(row["HomeTeam"]),
                str(row["AwayTeam"]),
                pseudo=config.pseudo_matches,
                max_goals=config.max_goals,
            )

            match_rows.append({
                "Competition": competition,
                "Season": season,
                "Date": row["Date"],
                "HomeTeam": row["HomeTeam"],
                "AwayTeam": row["AwayTeam"],
                "FTHG": int(row["FTHG"]),
                "FTAG": int(row["FTAG"]),
                "lambda_home": lam_home,
                "lambda_away": lam_away,
            })
            all_bets.extend(_event_rows(row, lam_home, lam_away, matrix, config.odds_source))

    bets = pd.DataFrame(all_bets)
    matches = pd.DataFrame(match_rows)

    if not bets.empty:
        bets["Week"] = bets["Date"].dt.to_period("W-TUE").astype(str)
        bets["Match"] = bets["HomeTeam"] + " - " + bets["AwayTeam"]
    if not matches.empty:
        matches["Week"] = matches["Date"].dt.to_period("W-TUE").astype(str)
        matches["Match"] = matches["HomeTeam"] + " - " + matches["AwayTeam"]

    return bets, matches


def performance(df: pd.DataFrame, stake: float = 1.0) -> dict:
    if df.empty:
        return {
            "bets": 0, "hit_rate": np.nan, "actual_roi": np.nan, "fair_roi": np.nan,
            "actual_profit": 0.0, "fair_profit": 0.0, "max_drawdown": np.nan,
        }

    hit = float(df["Won"].mean())

    actual = df[df["ActualProfit1u"].notna()].copy()
    actual_roi = float(actual["ActualProfit1u"].mean()) if not actual.empty else np.nan
    actual_profit = float(actual["ActualProfit1u"].sum() * stake) if not actual.empty else np.nan

    fair_roi = float(df["FairProfit1u"].mean())
    fair_profit = float(df["FairProfit1u"].sum() * stake)

    draw_series = actual["ActualProfit1u"].cumsum() if not actual.empty else df["FairProfit1u"].cumsum()
    peak = draw_series.cummax()
    max_dd = float((peak - draw_series).max() * stake) if not draw_series.empty else np.nan

    return {
        "bets": int(len(df)),
        "hit_rate": hit,
        "actual_roi": actual_roi,
        "fair_roi": fair_roi,
        "actual_profit": actual_profit,
        "fair_profit": fair_profit,
        "max_drawdown": max_dd,
        "bets_with_real_odds": int(len(actual)),
    }


def best_home_over_strategy(
    bets: pd.DataFrame,
    threshold: float = 0.85,
    min_value: float = 0.03,
    max_bets_week: int = 8,
) -> pd.DataFrame:
    """
    Repliceert de bestaande strategie:
    1) scherpste thuisteam Over-lijn die de modelgrens haalt;
    2) anders wedstrijd Over 0.5;
    3) maximaal één bet per wedstrijd;
    4) bij echte odds: minimaal opgegeven model-EV;
    5) maximaal N bets per week.

    Let op: historische echte odds zijn voor deze specifieke markten meestal niet beschikbaar.
    De selectie blijft daarom bruikbaar voor hit-rate/fair-odds kalibratie.
    """
    if bets.empty:
        return bets.copy()

    keys = ["Competition", "Season", "Date", "HomeTeam", "AwayTeam"]
    selected = []

    for _, g in bets.groupby(keys, sort=False):
        home_overs = g[g["Market"].astype(str).str.startswith("HOME_OVER_")].copy()
        home_overs = home_overs[home_overs["ModelProb"] >= threshold]
        if not home_overs.empty:
            # Scherpste = hoogste lijn die nog aan de kansgrens voldoet.
            choice = home_overs.sort_values(["Line", "ModelProb"], ascending=[False, False]).iloc[0]
        else:
            fallback = g[(g["Market"] == "TOTAL_OVER_0.5") & (g["ModelProb"] >= threshold)]
            if fallback.empty:
                continue
            choice = fallback.sort_values("ModelProb", ascending=False).iloc[0]

        # Als er echte odds zijn, pas value-filter toe. Zonder odds behouden we hem
        # voor de kalibratiebacktest maar markeren we geen actual ROI.
        if pd.notna(choice["ModelEV"]) and choice["ModelEV"] < min_value:
            continue

        selected.append(choice)

    if not selected:
        return bets.iloc[0:0].copy()

    out = pd.DataFrame(selected).reset_index(drop=True)
    out["RankValue"] = out["ModelEV"].fillna(out["ModelProb"] - threshold)

    kept = []
    for _, week_df in out.groupby(["Competition", "Season", "Week"], sort=False):
        kept.append(
            week_df.sort_values(["RankValue", "ModelProb"], ascending=False)
                   .head(max_bets_week)
        )
    return pd.concat(kept, ignore_index=True).sort_values(["Date", "HomeTeam"]).reset_index(drop=True)


def strategy_sweep(
    bets: pd.DataFrame,
    markets: list[str],
    probabilities: Iterable[float],
    min_evs: Iterable[float],
    require_real_odds: bool = True,
    stake: float = 1.0,
) -> pd.DataFrame:
    base = bets[bets["Market"].isin(markets)].copy()
    rows = []

    for p in probabilities:
        for ev in min_evs:
            x = base[base["ModelProb"] >= p].copy()
            if require_real_odds:
                x = x[x["BookmakerOdd"].notna() & (x["ModelEV"] >= ev)]
            else:
                # EV-filter alleen toepassen waar odds bestaan; zonder odds vooral kansfilter.
                with_odds = x["ModelEV"].notna()
                x = x[(~with_odds) | (x["ModelEV"] >= ev)]

            perf = performance(x, stake=stake)
            rows.append({
                "MinProb": float(p),
                "MinEV": float(ev),
                **perf,
            })

    return pd.DataFrame(rows)


def accumulator_backtest(
    singles: pd.DataFrame,
    legs: int = 2,
    top_n_per_week: int = 8,
    min_prob: float = 0.55,
    min_ev: float = 0.0,
    max_combos_per_week: int = 5000,
    stake: float = 1.0,
) -> pd.DataFrame:
    """
    Maakt doubles/trebles/etc. uit losse bets met echte historische odds.
    Alleen één selectie per wedstrijd wordt toegestaan om afhankelijkheid binnen
    dezelfde wedstrijd te vermijden.
    """
    x = singles.copy()
    x = x[
        x["BookmakerOdd"].notna()
        & (x["ModelProb"] >= min_prob)
        & (x["ModelEV"] >= min_ev)
    ].copy()

    if x.empty:
        return pd.DataFrame()

    # Per wedstrijd alleen de hoogste EV-selectie behouden.
    x = (
        x.sort_values(["Competition", "Season", "Week", "Date", "Match", "ModelEV"], ascending=[True, True, True, True, True, False])
         .drop_duplicates(["Competition", "Season", "Week", "Match"], keep="first")
    )

    combos_out = []

    for (competition, season, week), g in x.groupby(["Competition", "Season", "Week"], sort=False):
        g = g.sort_values(["ModelEV", "ModelProb"], ascending=False).head(top_n_per_week)
        if len(g) < legs:
            continue

        count = 0
        for idxs in combinations(g.index.tolist(), legs):
            if count >= max_combos_per_week:
                break
            legs_df = g.loc[list(idxs)]

            combo_prob = float(legs_df["ModelProb"].prod())
            combo_odds = float(legs_df["BookmakerOdd"].prod())
            won = bool(legs_df["Won"].all())
            profit = (combo_odds - 1.0) if won else -1.0

            combos_out.append({
                "Competition": competition,
                "Season": season,
                "Week": week,
                "Date": legs_df["Date"].max(),
                "Legs": legs,
                "Selections": " | ".join(
                    f'{r.Match}: {r.Bet} @{r.BookmakerOdd:.2f}'
                    for r in legs_df.itertuples()
                ),
                "ComboProb": combo_prob,
                "ComboFairOdd": 1.0 / combo_prob if combo_prob > 0 else np.nan,
                "ComboOdds": combo_odds,
                "ComboEV": combo_prob * combo_odds - 1.0,
                "Won": won,
                "Profit1u": profit,
            })
            count += 1

    out = pd.DataFrame(combos_out)
    if not out.empty:
        out["Profit"] = out["Profit1u"] * stake
    return out


SAME_MATCH_EVENTS = {
    "Thuiswinst": lambda h, a: h > a,
    "Gelijkspel": lambda h, a: h == a,
    "Uitwinst": lambda h, a: h < a,
    "Over 0.5": lambda h, a: h + a > 0.5,
    "Over 1.5": lambda h, a: h + a > 1.5,
    "Over 2.5": lambda h, a: h + a > 2.5,
    "Over 3.5": lambda h, a: h + a > 3.5,
    "Under 2.5": lambda h, a: h + a < 2.5,
    "Under 3.5": lambda h, a: h + a < 3.5,
    "BTTS Ja": lambda h, a: h > 0 and a > 0,
    "BTTS Nee": lambda h, a: h == 0 or a == 0,
    "Thuisteam over 0.5": lambda h, a: h > 0.5,
    "Thuisteam over 1.5": lambda h, a: h > 1.5,
    "Uitteam over 0.5": lambda h, a: a > 0.5,
    "Uitteam over 1.5": lambda h, a: a > 1.5,
}


def same_match_combo(matches: pd.DataFrame, event_a: str, event_b: str, max_goals: int = 8) -> pd.DataFrame:
    if event_a not in SAME_MATCH_EVENTS or event_b not in SAME_MATCH_EVENTS:
        raise ValueError("Onbekende combinatie-event.")

    fa = SAME_MATCH_EVENTS[event_a]
    fb = SAME_MATCH_EVENTS[event_b]
    out = []

    for r in matches.itertuples():
        matrix = score_matrix(r.lambda_home, r.lambda_away, max_goals=max_goals)
        p = 0.0
        for h in range(matrix.shape[0]):
            for a in range(matrix.shape[1]):
                if fa(h, a) and fb(h, a):
                    p += matrix[h, a]

        actual = bool(fa(int(r.FTHG), int(r.FTAG)) and fb(int(r.FTHG), int(r.FTAG)))
        fair_odd = 1.0 / p if p > 0 else np.nan

        out.append({
            "Competition": r.Competition,
            "Season": r.Season,
            "Date": r.Date,
            "Match": r.Match,
            "EventA": event_a,
            "EventB": event_b,
            "ModelProb": p,
            "FairOdd": fair_odd,
            "Won": actual,
            "FairProfit1u": (fair_odd - 1.0) if actual else -1.0,
        })

    return pd.DataFrame(out)

def team_recent_summary(data: pd.DataFrame, competition: str, team: str, before_date=None, n_matches: int = 10) -> dict:
    """Laatste N wedstrijden vóór een datum, altijd gelezen vanuit het gekozen team."""
    x = data[data["Competition"] == competition].copy()
    if before_date is not None and pd.notna(before_date):
        x = x[x["Date"] < pd.Timestamp(before_date)]
    x = x[(x["HomeTeam"] == team) | (x["AwayTeam"] == team)].sort_values("Date").tail(max(int(n_matches), 1))
    x = add_team_perspective(x, team)
    if x.empty:
        return {"Team": team, "Matches": 0}

    is_home = x["HomeTeam"] == team
    gf = pd.Series(np.where(is_home, x["FTHG"], x["FTAG"]), index=x.index, dtype=float)
    ga = pd.Series(np.where(is_home, x["FTAG"], x["FTHG"]), index=x.index, dtype=float)
    wins, draws = gf > ga, gf == ga
    out = {
        "Team": team, "Matches": int(len(x)), "Wins": int(wins.sum()), "Draws": int(draws.sum()),
        "Losses": int((~wins & ~draws).sum()), "PointsPerGame": float((wins.sum()*3 + draws.sum()) / len(x)),
        "GoalsFor": float(gf.mean()), "GoalsAgainst": float(ga.mean()), "TotalGoals": float((gf+ga).mean()),
        "Over2_5": float(((gf+ga)>2.5).mean()), "BTTS": float(((gf>0)&(ga>0)).mean()),
        "CleanSheet": float((ga==0).mean()), "FailedToScore": float((gf==0).mean()),
    }
    for source, target in {
        "TeamShots":"ShotsFor", "OpponentShots":"ShotsAgainst", "TeamShotsOnTarget":"SOTFor",
        "OpponentShotsOnTarget":"SOTAgainst", "TeamCorners":"CornersFor", "OpponentCorners":"CornersAgainst",
        "TeamYellow":"YellowFor", "OpponentYellow":"YellowAgainst", "TeamRed":"RedFor", "OpponentRed":"RedAgainst",
    }.items():
        if source in x.columns:
            s=pd.to_numeric(x[source], errors="coerce")
            out[target]=float(s.mean()) if s.notna().any() else np.nan
    return out


def goal_distribution_summary(data: pd.DataFrame) -> pd.DataFrame:
    if data.empty: return pd.DataFrame()
    rows=[]
    for label,col in [("Totaal goals","TotalGoals"),("Thuisgoals","FTHG"),("Uitgoals","FTAG")]:
        s=pd.to_numeric(data[col],errors="coerce").dropna(); total=len(s)
        if not total: continue
        for bucket in [0,1,2,3,4]:
            count=int((s==bucket).sum()); rows.append({"Type":label,"Goals":str(bucket),"Wedstrijden":count,"Percentage":count/total})
        count=int((s>=5).sum()); rows.append({"Type":label,"Goals":"5+","Wedstrijden":count,"Percentage":count/total})
    return pd.DataFrame(rows)


def _team_perspective_values(row: pd.Series, team: str) -> dict:
    home=str(row["HomeTeam"])==team
    def pick(h,a): return row[h] if home else row[a] if h in row.index and a in row.index else np.nan
    def opp(h,a): return row[a] if home else row[h] if h in row.index and a in row.index else np.nan
    # Explicit version to avoid ambiguous conditional precedence for missing columns.
    def safe(h,a,opponent=False):
        if h not in row.index or a not in row.index: return np.nan
        if opponent: return row[a] if home else row[h]
        return row[h] if home else row[a]
    return {
        "GF":safe("FTHG","FTAG"), "GA":safe("FTHG","FTAG",True),
        "ShotsFor":safe("HS","AS"), "ShotsAgainst":safe("HS","AS",True),
        "SOTFor":safe("HST","AST"), "SOTAgainst":safe("HST","AST",True),
        "CornersFor":safe("HC","AC"), "CornersAgainst":safe("HC","AC",True),
        "YellowFor":safe("HY","AY"), "YellowAgainst":safe("HY","AY",True),
    }


def _rolling_team_features(history: pd.DataFrame, team: str, n: int) -> dict:
    games=history[(history["HomeTeam"]==team)|(history["AwayTeam"]==team)].sort_values("Date").tail(max(int(n),1))
    if games.empty: return {}
    df=pd.DataFrame([_team_perspective_values(r,team) for _,r in games.iterrows()])
    out={"Matches":len(games)}
    for c in df.columns:
        s=pd.to_numeric(df[c],errors="coerce"); out[c]=float(s.mean()) if s.notna().any() else np.nan
    gf=pd.to_numeric(df["GF"],errors="coerce"); ga=pd.to_numeric(df["GA"],errors="coerce")
    out["PPG"]=float((((gf>ga)*3)+(gf==ga)).mean())
    out["Over2_5Rate"]=float(((gf+ga)>2.5).mean()); out["BTTSRate"]=float(((gf>0)&(ga>0)).mean())
    out["CleanSheetRate"]=float((ga==0).mean())
    return out


def build_prematch_feature_table(data: pd.DataFrame, rolling_n: int = 10, min_prior_matches: int = 3) -> pd.DataFrame:
    """Walk-forward tabel: elke feature gebruikt uitsluitend oudere wedstrijden binnen hetzelfde seizoen."""
    rows=[]
    for (competition,season),g in data.groupby(["Competition","Season"],sort=False):
        sort_cols=["Date"]+(["Time"] if "Time" in g.columns else [])
        g=g.sort_values(sort_cols,kind="stable").reset_index(drop=True)
        for i,r in g.iterrows():
            history=g.iloc[:i]
            hf=_rolling_team_features(history,str(r["HomeTeam"]),rolling_n); af=_rolling_team_features(history,str(r["AwayTeam"]),rolling_n)
            if hf.get("Matches",0)<min_prior_matches or af.get("Matches",0)<min_prior_matches: continue
            row={
                "Competition":competition,"Season":season,"Date":r["Date"],"HomeTeam":r["HomeTeam"],"AwayTeam":r["AwayTeam"],
                "Match":f'{r["HomeTeam"]} - {r["AwayTeam"]}',"FTHG":r["FTHG"],"FTAG":r["FTAG"],"TotalGoals":r["FTHG"]+r["FTAG"],
                "Won_HOME":bool(r["FTHG"]>r["FTAG"]),"Won_DRAW":bool(r["FTHG"]==r["FTAG"]),"Won_AWAY":bool(r["FTHG"]<r["FTAG"]),
                "Won_OVER25":bool(r["FTHG"]+r["FTAG"]>2.5),"Won_UNDER25":bool(r["FTHG"]+r["FTAG"]<2.5),
                "Won_BTTS":bool(r["FTHG"]>0 and r["FTAG"]>0),
            }
            for prefix,feat in [("Home",hf),("Away",af)]:
                for key,value in feat.items(): row[f"{prefix}_{key}"]=value
            row["Odd_HOME"]=market_odds(r,"HOME","closing_avg"); row["Odd_DRAW"]=market_odds(r,"DRAW","closing_avg")
            row["Odd_AWAY"]=market_odds(r,"AWAY","closing_avg"); row["Odd_OVER25"]=market_odds(r,"TOTAL_OVER_2.5","closing_avg")
            row["Odd_UNDER25"]=market_odds(r,"TOTAL_UNDER_2.5","closing_avg")
            rows.append(row)
    return pd.DataFrame(rows)


FEATURE_MARKETS={
    "Thuiswinst":("Won_HOME","Odd_HOME"),"Gelijkspel":("Won_DRAW","Odd_DRAW"),"Uitwinst":("Won_AWAY","Odd_AWAY"),
    "Over 2.5":("Won_OVER25","Odd_OVER25"),"Under 2.5":("Won_UNDER25","Odd_UNDER25"),"BTTS Ja":("Won_BTTS",None),
}


def feature_backtest(feature_table: pd.DataFrame, market: str, filters=None, stake: float = 1.0):
    if feature_table.empty or market not in FEATURE_MARKETS:
        return pd.DataFrame(), {"bets":0,"hit_rate":np.nan,"roi":np.nan,"profit":np.nan,"priced_bets":0}
    won_col,odd_col=FEATURE_MARKETS[market]; x=feature_table.copy()
    for col,bounds in (filters or {}).items():
        if col not in x.columns: continue
        lo,hi=bounds; s=pd.to_numeric(x[col],errors="coerce"); mask=s.notna()
        if lo is not None: mask &= s>=float(lo)
        if hi is not None: mask &= s<=float(hi)
        x=x[mask]
    hit=float(x[won_col].mean()) if not x.empty else np.nan; roi=profit=np.nan; priced_bets=0
    if odd_col and odd_col in x.columns:
        odds=pd.to_numeric(x[odd_col],errors="coerce"); valid=odds.notna()&(odds>1); priced_bets=int(valid.sum())
        if valid.any():
            profits=np.where(x.loc[valid,won_col].to_numpy(),odds.loc[valid].to_numpy()-1,-1.0)
            roi=float(np.mean(profits)); profit=float(np.sum(profits)*stake)
    return x,{"bets":int(len(x)),"hit_rate":hit,"roi":roi,"profit":profit,"priced_bets":priced_bets}



# =============================================================================
# v0.5 — uitgebreid historisch variabelenregister en prematch feature-engine
# =============================================================================

# Vanaf 2017/18 zijn wedstrijdstatistieken volgens Football-Data voor alle 22
# Europese divisies beschikbaar. Deze seizoenen zijn daarom uniform selecteerbaar.
SEASON_SLUGS.update({
    "2017/18": "1718",
    "2018/19": "1819",
    "2019/20": "1920",
    "2020/21": "2021",
    "2021/22": "2122",
    "2022/23": "2223",
})
ALL_SUPPORTED_SEASONS = sorted(SEASON_SLUGS.keys(), key=lambda s: int(s[:4]))


def _safe_ratio(num, den):
    num = pd.to_numeric(num, errors="coerce")
    den = pd.to_numeric(den, errors="coerce")
    return np.where(den > 0, num / den, np.nan)


def _no_vig_frame(data: pd.DataFrame, cols: list[str], prefix: str) -> pd.DataFrame:
    if not all(c in data.columns for c in cols):
        return data
    odds = data[cols].apply(pd.to_numeric, errors="coerce")
    valid = odds.gt(1).all(axis=1)
    inv = 1 / odds
    denom = inv.sum(axis=1)
    labels = ["Home", "Draw", "Away"] if len(cols) == 3 else ["Over", "Under"]
    for c, label in zip(cols, labels):
        data[f"{prefix}_{label}RawImplied"] = np.where(valid, inv[c], np.nan)
        data[f"{prefix}_{label}NoVig"] = np.where(valid, inv[c] / denom, np.nan)
    data[f"{prefix}_Overround"] = np.where(valid, denom - 1, np.nan)
    return data


def enrich_raw_data_v05(data: pd.DataFrame) -> pd.DataFrame:
    """Voeg afgeleide wedstrijd-, statistiek- en marktvariabelen toe zonder bronkolommen te verliezen."""
    if data.empty:
        return data
    data = data.copy()

    # Football-Data verandert door de jaren heen van bookmakerkolommen.
    # Probeer daarom alle niet-identificatiekolommen automatisch numeriek te maken
    # wanneer minstens 80% van de gevulde waarden als getal kan worden gelezen.
    text_cols = {
        "Competition","Season","Div","Date","Time","HomeTeam","AwayTeam","Match",
        "FTR","HTR","Referee","HomeResult","HalfTimeResultNL","OpeningFavorite","ClosingFavorite"
    }
    for col in list(data.columns):
        if col in text_cols or pd.api.types.is_numeric_dtype(data[col]) or pd.api.types.is_bool_dtype(data[col]):
            continue
        original_nonnull = int(data[col].notna().sum())
        if original_nonnull == 0:
            continue
        parsed = pd.to_numeric(data[col], errors="coerce")
        if int(parsed.notna().sum()) >= max(1, int(original_nonnull * 0.80)):
            data[col] = parsed

    # Resultaat / goals.
    data["GoalDiffHome"] = data["FTHG"] - data["FTAG"]
    data["AbsGoalDiff"] = data["GoalDiffHome"].abs()
    data["HomeCleanSheet"] = data["FTAG"] == 0
    data["AwayCleanSheet"] = data["FTHG"] == 0
    data["HomeFailedToScore"] = data["FTHG"] == 0
    data["AwayFailedToScore"] = data["FTAG"] == 0
    data["Scoreless"] = data["TotalGoals"] == 0
    data["HomeWin"] = data["FTHG"] > data["FTAG"]
    data["Draw"] = data["FTHG"] == data["FTAG"]
    data["AwayWin"] = data["FTHG"] < data["FTAG"]

    if {"HTHG", "HTAG"}.issubset(data.columns):
        data["FirstHalfGoals"] = pd.to_numeric(data["HTHG"], errors="coerce") + pd.to_numeric(data["HTAG"], errors="coerce")
        data["SecondHalfHomeGoals"] = pd.to_numeric(data["FTHG"], errors="coerce") - pd.to_numeric(data["HTHG"], errors="coerce")
        data["SecondHalfAwayGoals"] = pd.to_numeric(data["FTAG"], errors="coerce") - pd.to_numeric(data["HTAG"], errors="coerce")
        data["SecondHalfGoals"] = data["SecondHalfHomeGoals"] + data["SecondHalfAwayGoals"]
        data["HTGoalDiffHome"] = data["HTHG"] - data["HTAG"]
        data["HTHomeLead"] = data["HTHG"] > data["HTAG"]
        data["HTDraw"] = data["HTHG"] == data["HTAG"]
        data["HTAwayLead"] = data["HTHG"] < data["HTAG"]
        data["HTOver0_5"] = data["FirstHalfGoals"] > 0.5
        data["HTOver1_5"] = data["FirstHalfGoals"] > 1.5
        data["SecondHalfOver0_5"] = data["SecondHalfGoals"] > 0.5
        data["SecondHalfOver1_5"] = data["SecondHalfGoals"] > 1.5
        data["SecondHalfOver2_5"] = data["SecondHalfGoals"] > 2.5

    # Wedstrijdstatistieken.
    pair_defs = {
        "ShotsDiffHome": ("HS", "AS"),
        "SOTDiffHome": ("HST", "AST"),
        "CornersDiffHome": ("HC", "AC"),
        "FoulsDiffHome": ("HF", "AF"),
        "OffsidesDiffHome": ("HO", "AO"),
        "YellowDiffHome": ("HY", "AY"),
        "RedDiffHome": ("HR", "AR"),
        "WoodworkDiffHome": ("HHW", "AHW"),
        "BookingPointsDiffHome": ("HBP", "ABP"),
    }
    for target, (h, a) in pair_defs.items():
        if h in data.columns and a in data.columns:
            data[target] = pd.to_numeric(data[h], errors="coerce") - pd.to_numeric(data[a], errors="coerce")

    if {"HS", "AS"}.issubset(data.columns):
        total = pd.to_numeric(data["HS"], errors="coerce") + pd.to_numeric(data["AS"], errors="coerce")
        data["HomeShotShare"] = np.where(total > 0, pd.to_numeric(data["HS"], errors="coerce") / total, np.nan)
        data["AwayShotShare"] = np.where(total > 0, pd.to_numeric(data["AS"], errors="coerce") / total, np.nan)
        data["HomeGoalConversionShots"] = _safe_ratio(data["FTHG"], data["HS"])
        data["AwayGoalConversionShots"] = _safe_ratio(data["FTAG"], data["AS"])

    if {"HST", "AST"}.issubset(data.columns):
        total = pd.to_numeric(data["HST"], errors="coerce") + pd.to_numeric(data["AST"], errors="coerce")
        data["HomeSOTShare"] = np.where(total > 0, pd.to_numeric(data["HST"], errors="coerce") / total, np.nan)
        data["AwaySOTShare"] = np.where(total > 0, pd.to_numeric(data["AST"], errors="coerce") / total, np.nan)
        data["HomeGoalConversionSOT"] = _safe_ratio(data["FTHG"], data["HST"])
        data["AwayGoalConversionSOT"] = _safe_ratio(data["FTAG"], data["AST"])

    # Kalender.
    if "Date" in data.columns:
        data["Month"] = data["Date"].dt.month
        data["DayOfWeek"] = data["Date"].dt.dayofweek
        data["Weekend"] = data["DayOfWeek"].isin([5, 6])

    # No-vig marktprobabilities.
    data = _no_vig_frame(data, ["AvgH", "AvgD", "AvgA"], "MarketOpen1X2")
    data = _no_vig_frame(data, ["AvgCH", "AvgCD", "AvgCA"], "MarketClose1X2")
    data = _no_vig_frame(data, ["Avg>2.5", "Avg<2.5"], "MarketOpenOU25")
    data = _no_vig_frame(data, ["AvgC>2.5", "AvgC<2.5"], "MarketCloseOU25")

    # Marktbeweging en spreiding.
    for outcome, ocol, ccol in [
        ("Home", "AvgH", "AvgCH"),
        ("Draw", "AvgD", "AvgCD"),
        ("Away", "AvgA", "AvgCA"),
        ("Over25", "Avg>2.5", "AvgC>2.5"),
        ("Under25", "Avg<2.5", "AvgC<2.5"),
    ]:
        if ocol in data.columns and ccol in data.columns:
            data[f"OddsMove_{outcome}"] = pd.to_numeric(data[ccol], errors="coerce") - pd.to_numeric(data[ocol], errors="coerce")
            data[f"OddsMovePct_{outcome}"] = _safe_ratio(
                pd.to_numeric(data[ccol], errors="coerce") - pd.to_numeric(data[ocol], errors="coerce"),
                pd.to_numeric(data[ocol], errors="coerce"),
            )

    for outcome, ocol, ccol in [
        ("Home", "MarketOpen1X2_HomeNoVig", "MarketClose1X2_HomeNoVig"),
        ("Draw", "MarketOpen1X2_DrawNoVig", "MarketClose1X2_DrawNoVig"),
        ("Away", "MarketOpen1X2_AwayNoVig", "MarketClose1X2_AwayNoVig"),
        ("Over25", "MarketOpenOU25_OverNoVig", "MarketCloseOU25_OverNoVig"),
        ("Under25", "MarketOpenOU25_UnderNoVig", "MarketCloseOU25_UnderNoVig"),
    ]:
        if ocol in data.columns and ccol in data.columns:
            data[f"ProbMove_{outcome}"] = data[ccol] - data[ocol]

    for prefix, avg_cols, max_cols in [
        ("Open1X2", ["AvgH", "AvgD", "AvgA"], ["MaxH", "MaxD", "MaxA"]),
        ("Close1X2", ["AvgCH", "AvgCD", "AvgCA"], ["MaxCH", "MaxCD", "MaxCA"]),
    ]:
        if all(c in data.columns for c in avg_cols + max_cols):
            for label, av, mx in zip(["Home", "Draw", "Away"], avg_cols, max_cols):
                data[f"{prefix}_MaxAvgGap_{label}"] = pd.to_numeric(data[mx], errors="coerce") - pd.to_numeric(data[av], errors="coerce")
                data[f"{prefix}_MaxAvgGapPct_{label}"] = _safe_ratio(data[f"{prefix}_MaxAvgGap_{label}"], data[av])

    if "AHh" in data.columns and "AHCh" in data.columns:
        data["AHLineMove"] = pd.to_numeric(data["AHCh"], errors="coerce") - pd.to_numeric(data["AHh"], errors="coerce")

    # Markt-favoriet.
    for prefix, cols in [
        ("Opening", ["AvgH", "AvgD", "AvgA"]),
        ("Closing", ["AvgCH", "AvgCD", "AvgCA"]),
    ]:
        if all(c in data.columns for c in cols):
            odds = data[cols].apply(pd.to_numeric, errors="coerce")
            labels = np.array(["Thuis", "Gelijk", "Uit"])
            valid = odds.gt(1).all(axis=1)
            arr = odds.to_numpy(dtype=float)
            safe = np.where(np.isfinite(arr), arr, np.inf)
            idx = np.argmin(safe, axis=1)
            data[f"{prefix}Favorite"] = np.where(valid, labels[idx], None)
            data[f"{prefix}FavoriteOdd"] = np.where(valid, safe[np.arange(len(safe)), idx], np.nan)

    return data


# Bewaar de v0.4-loader; de v0.5-loader verrijkt zijn resultaat.
_load_data_v04 = load_data
def load_data(
    competitions: Iterable[str],
    seasons: Iterable[str],
    uploaded_files: Optional[dict[tuple[str, str], object]] = None,
) -> pd.DataFrame:
    return enrich_raw_data_v05(_load_data_v04(competitions, seasons, uploaded_files))


def _team_perspective_values_v05(row: pd.Series, team: str) -> dict:
    home = str(row["HomeTeam"]) == team

    def tv(h, a):
        return row.get(h, np.nan) if home else row.get(a, np.nan)
    def ov(h, a):
        return row.get(a, np.nan) if home else row.get(h, np.nan)

    gf = tv("FTHG", "FTAG")
    ga = ov("FTHG", "FTAG")
    ht_gf = tv("HTHG", "HTAG")
    ht_ga = ov("HTHG", "HTAG")
    shots_f = tv("HS", "AS")
    shots_a = ov("HS", "AS")
    sot_f = tv("HST", "AST")
    sot_a = ov("HST", "AST")

    return {
        "GF": gf, "GA": ga,
        "HTGF": ht_gf, "HTGA": ht_ga,
        "SecondHalfGF": (gf - ht_gf) if pd.notna(gf) and pd.notna(ht_gf) else np.nan,
        "SecondHalfGA": (ga - ht_ga) if pd.notna(ga) and pd.notna(ht_ga) else np.nan,
        "ShotsFor": shots_f, "ShotsAgainst": shots_a,
        "SOTFor": sot_f, "SOTAgainst": sot_a,
        "WoodworkFor": tv("HHW", "AHW"), "WoodworkAgainst": ov("HHW", "AHW"),
        "CornersFor": tv("HC", "AC"), "CornersAgainst": ov("HC", "AC"),
        "FoulsFor": tv("HF", "AF"), "FoulsAgainst": ov("HF", "AF"),
        "FreeKicksConceded": tv("HFKC", "AFKC"), "OpponentFreeKicksConceded": ov("HFKC", "AFKC"),
        "OffsidesFor": tv("HO", "AO"), "OffsidesAgainst": ov("HO", "AO"),
        "YellowFor": tv("HY", "AY"), "YellowAgainst": ov("HY", "AY"),
        "RedFor": tv("HR", "AR"), "RedAgainst": ov("HR", "AR"),
        "BookingPointsFor": tv("HBP", "ABP"), "BookingPointsAgainst": ov("HBP", "ABP"),
        "Attendance": row.get("Attendance", np.nan),
        "IsHome": home,
        "Date": row.get("Date", pd.NaT),
    }


def _trailing_streak(flags: list[bool]) -> int:
    count = 0
    for flag in reversed(flags):
        if bool(flag):
            count += 1
        else:
            break
    return count


def _rolling_team_features_v05(history: pd.DataFrame, team: str, n: int, venue: Optional[str] = None) -> dict:
    games = history[(history["HomeTeam"] == team) | (history["AwayTeam"] == team)].sort_values("Date")
    if venue == "home":
        games = games[games["HomeTeam"] == team]
    elif venue == "away":
        games = games[games["AwayTeam"] == team]
    games = games.tail(max(int(n), 1))
    if games.empty:
        return {"Matches": 0}

    vals = pd.DataFrame([_team_perspective_values_v05(r, team) for _, r in games.iterrows()])
    out = {"Matches": int(len(vals))}

    numeric_mean = [
        "GF","GA","HTGF","HTGA","SecondHalfGF","SecondHalfGA",
        "ShotsFor","ShotsAgainst","SOTFor","SOTAgainst","WoodworkFor","WoodworkAgainst",
        "CornersFor","CornersAgainst","FoulsFor","FoulsAgainst","FreeKicksConceded",
        "OpponentFreeKicksConceded","OffsidesFor","OffsidesAgainst","YellowFor","YellowAgainst",
        "RedFor","RedAgainst","BookingPointsFor","BookingPointsAgainst","Attendance",
    ]
    for c in numeric_mean:
        if c in vals.columns:
            s = pd.to_numeric(vals[c], errors="coerce")
            out[c] = float(s.mean()) if s.notna().any() else np.nan

    gf = pd.to_numeric(vals["GF"], errors="coerce")
    ga = pd.to_numeric(vals["GA"], errors="coerce")
    total = gf + ga
    valid_score = gf.notna() & ga.notna()
    if valid_score.any():
        wins = gf > ga
        draws = gf == ga
        losses = gf < ga
        out.update({
            "WinRate": float(wins.mean()),
            "DrawRate": float(draws.mean()),
            "LossRate": float(losses.mean()),
            "PPG": float((wins.astype(int) * 3 + draws.astype(int)).mean()),
            "GoalDiff": float((gf - ga).mean()),
            "TotalGoals": float(total.mean()),
            "BTTSRate": float(((gf > 0) & (ga > 0)).mean()),
            "CleanSheetRate": float((ga == 0).mean()),
            "FailedToScoreRate": float((gf == 0).mean()),
        })
        for line in [0.5,1.5,2.5,3.5,4.5,5.5,6.5]:
            key = str(line).replace(".", "_")
            out[f"Over{key}Rate"] = float((total > line).mean())
        for line in [0.5,1.5,2.5,3.5]:
            key = str(line).replace(".", "_")
            out[f"TeamOver{key}Rate"] = float((gf > line).mean())
            out[f"ConcedeOver{key}Rate"] = float((ga > line).mean())

        out["WinStreak"] = _trailing_streak(wins.tolist())
        out["UnbeatenStreak"] = _trailing_streak((~losses).tolist())
        out["ScoringStreak"] = _trailing_streak((gf > 0).tolist())
        out["ConcedingStreak"] = _trailing_streak((ga > 0).tolist())
        out["CleanSheetStreak"] = _trailing_streak((ga == 0).tolist())
        out["Over2_5Streak"] = _trailing_streak((total > 2.5).tolist())
        out["BTTSStreak"] = _trailing_streak(((gf > 0) & (ga > 0)).tolist())

    hgf = pd.to_numeric(vals.get("HTGF"), errors="coerce")
    hga = pd.to_numeric(vals.get("HTGA"), errors="coerce")
    if isinstance(hgf, pd.Series) and hgf.notna().any() and hga.notna().any():
        htt = hgf + hga
        out["HTLeadRate"] = float((hgf > hga).mean())
        out["HTDrawRate"] = float((hgf == hga).mean())
        out["HTTrailRate"] = float((hgf < hga).mean())
        out["HTOver0_5Rate"] = float((htt > 0.5).mean())
        out["HTOver1_5Rate"] = float((htt > 1.5).mean())

    sf = pd.to_numeric(vals.get("ShotsFor"), errors="coerce")
    sa = pd.to_numeric(vals.get("ShotsAgainst"), errors="coerce")
    stf = pd.to_numeric(vals.get("SOTFor"), errors="coerce")
    sta = pd.to_numeric(vals.get("SOTAgainst"), errors="coerce")
    if isinstance(sf, pd.Series) and sf.notna().any():
        out["ShotAccuracy"] = float((stf / sf.replace(0, np.nan)).mean()) if isinstance(stf, pd.Series) else np.nan
        out["GoalConversionShots"] = float((gf / sf.replace(0, np.nan)).mean())
        denom = sf + sa
        out["ShotShare"] = float((sf / denom.replace(0, np.nan)).mean()) if isinstance(sa, pd.Series) else np.nan
    if isinstance(stf, pd.Series) and stf.notna().any():
        out["GoalConversionSOT"] = float((gf / stf.replace(0, np.nan)).mean())
        denom = stf + sta
        out["SOTShare"] = float((stf / denom.replace(0, np.nan)).mean()) if isinstance(sta, pd.Series) else np.nan

    return out


def _league_features(history: pd.DataFrame, n: int = 90) -> dict:
    g = history.sort_values("Date").tail(n)
    if g.empty:
        return {}
    out = {
        "Matches": len(g),
        "HomeGoals": float(pd.to_numeric(g["FTHG"], errors="coerce").mean()),
        "AwayGoals": float(pd.to_numeric(g["FTAG"], errors="coerce").mean()),
        "TotalGoals": float(pd.to_numeric(g["TotalGoals"], errors="coerce").mean()),
        "HomeWinRate": float((g["FTHG"] > g["FTAG"]).mean()),
        "DrawRate": float((g["FTHG"] == g["FTAG"]).mean()),
        "AwayWinRate": float((g["FTHG"] < g["FTAG"]).mean()),
        "Over2_5Rate": float((g["TotalGoals"] > 2.5).mean()),
        "BTTSRate": float(((g["FTHG"] > 0) & (g["FTAG"] > 0)).mean()),
    }
    for source, target in [
        ("TotalShots","Shots"),("TotalShotsOnTarget","SOT"),("TotalCorners","Corners"),
        ("TotalFouls","Fouls"),("TotalYellow","Yellow"),("TotalRed","Red"),
    ]:
        if source in g.columns:
            s = pd.to_numeric(g[source], errors="coerce")
            out[target] = float(s.mean()) if s.notna().any() else np.nan
    return out


def _h2h_features(history: pd.DataFrame, home_team: str, away_team: str, n: int = 5) -> dict:
    g = history[
        ((history["HomeTeam"] == home_team) & (history["AwayTeam"] == away_team)) |
        ((history["HomeTeam"] == away_team) & (history["AwayTeam"] == home_team))
    ].sort_values("Date").tail(n)
    if g.empty:
        return {"Matches": 0}
    home_gf, away_gf = [], []
    for _, r in g.iterrows():
        if r["HomeTeam"] == home_team:
            home_gf.append(float(r["FTHG"])); away_gf.append(float(r["FTAG"]))
        else:
            home_gf.append(float(r["FTAG"])); away_gf.append(float(r["FTHG"]))
    h = pd.Series(home_gf); a = pd.Series(away_gf)
    return {
        "Matches": len(g),
        "CurrentHomeTeamGF": float(h.mean()),
        "CurrentAwayTeamGF": float(a.mean()),
        "CurrentHomeTeamWinRate": float((h > a).mean()),
        "DrawRate": float((h == a).mean()),
        "CurrentAwayTeamWinRate": float((h < a).mean()),
        "Over2_5Rate": float(((h + a) > 2.5).mean()),
        "BTTSRate": float(((h > 0) & (a > 0)).mean()),
    }


def _referee_features(history: pd.DataFrame, referee, n: int = 20) -> dict:
    if "Referee" not in history.columns or referee is None or pd.isna(referee):
        return {}
    g = history[history["Referee"].astype(str) == str(referee)].sort_values("Date").tail(n)
    if g.empty:
        return {"Matches": 0}
    out = {
        "Matches": len(g),
        "HomeWinRate": float((g["FTHG"] > g["FTAG"]).mean()),
        "DrawRate": float((g["FTHG"] == g["FTAG"]).mean()),
        "AwayWinRate": float((g["FTHG"] < g["FTAG"]).mean()),
        "Goals": float(g["TotalGoals"].mean()),
    }
    for source, target in [
        ("TotalFouls","Fouls"),("TotalYellow","Yellow"),("TotalRed","Red"),("TotalCorners","Corners"),
    ]:
        if source in g.columns:
            s = pd.to_numeric(g[source], errors="coerce")
            out[target] = float(s.mean()) if s.notna().any() else np.nan
    return out


def _rest_days(history: pd.DataFrame, team: str, match_date) -> float:
    g = history[(history["HomeTeam"] == team) | (history["AwayTeam"] == team)].sort_values("Date")
    if g.empty or pd.isna(match_date):
        return np.nan
    last_date = g["Date"].dropna().max()
    if pd.isna(last_date):
        return np.nan
    return float((pd.Timestamp(match_date) - pd.Timestamp(last_date)).days)


def _current_market_features(r: pd.Series) -> dict:
    out = {}
    raw_map = {
        "MarketOpen_HomeOdd":"AvgH","MarketOpen_DrawOdd":"AvgD","MarketOpen_AwayOdd":"AvgA",
        "MarketClose_HomeOdd":"AvgCH","MarketClose_DrawOdd":"AvgCD","MarketClose_AwayOdd":"AvgCA",
        "MarketOpen_Over25Odd":"Avg>2.5","MarketOpen_Under25Odd":"Avg<2.5",
        "MarketClose_Over25Odd":"AvgC>2.5","MarketClose_Under25Odd":"AvgC<2.5",
        "MarketOpen_AHLine":"AHh","MarketClose_AHLine":"AHCh",
        "MarketOpen_AHHomeOdd":"AvgAHH","MarketOpen_AHAwayOdd":"AvgAHA",
        "MarketClose_AHHomeOdd":"AvgCAHH","MarketClose_AHAwayOdd":"AvgCAHA",
    }
    for target, source in raw_map.items():
        if source in r.index:
            out[target] = pd.to_numeric(pd.Series([r[source]]), errors="coerce").iloc[0]

    # no-vig probabilities from current match.
    for prefix, cols, labels in [
        ("MarketOpen", ["AvgH","AvgD","AvgA"], ["HomeProb","DrawProb","AwayProb"]),
        ("MarketClose", ["AvgCH","AvgCD","AvgCA"], ["HomeProb","DrawProb","AwayProb"]),
        ("MarketOpen", ["Avg>2.5","Avg<2.5"], ["Over25Prob","Under25Prob"]),
        ("MarketClose", ["AvgC>2.5","AvgC<2.5"], ["Over25Prob","Under25Prob"]),
    ]:
        vals = [pd.to_numeric(pd.Series([r.get(c, np.nan)]), errors="coerce").iloc[0] for c in cols]
        if all(pd.notna(v) and v > 1 for v in vals):
            inv = np.array([1/v for v in vals], dtype=float)
            probs = inv / inv.sum()
            for label, p in zip(labels, probs):
                out[f"{prefix}_{label}"] = float(p)
            suffix = "1X2Overround" if len(cols) == 3 else "OU25Overround"
            out[f"{prefix}_{suffix}"] = float(inv.sum() - 1)

    for name in ["Home","Draw","Away","Over25","Under25"]:
        op = out.get(f"MarketOpen_{name}Prob")
        cp = out.get(f"MarketClose_{name}Prob")
        if op is not None and cp is not None:
            out[f"MarketMove_{name}Prob"] = cp - op
    if pd.notna(out.get("MarketOpen_AHLine", np.nan)) and pd.notna(out.get("MarketClose_AHLine", np.nan)):
        out["MarketMove_AHLine"] = out["MarketClose_AHLine"] - out["MarketOpen_AHLine"]
    return out


def build_prematch_feature_table(
    data: pd.DataFrame,
    rolling_n: int = 10,
    min_prior_matches: int = 3,
) -> pd.DataFrame:
    """
    Uitgebreide walk-forward featuretabel.
    Iedere feature wordt uitsluitend uit rijen vóór de historische wedstrijd berekend,
    behalve expliciet gemarkeerde MarketClose_* velden: die representeren de slotmarkt.
    """
    if data.empty:
        return pd.DataFrame()

    data = enrich_raw_data_v05(data)
    rows = []

    for competition, comp in data.groupby("Competition", sort=False):
        comp = comp.sort_values(["Date"] + (["Time"] if "Time" in comp.columns else []), kind="stable").reset_index(drop=True)

        for i, r in comp.iterrows():
            history = comp.iloc[:i].copy()
            home_team, away_team = str(r["HomeTeam"]), str(r["AwayTeam"])

            home = _rolling_team_features_v05(history, home_team, rolling_n)
            away = _rolling_team_features_v05(history, away_team, rolling_n)
            if home.get("Matches", 0) < min_prior_matches or away.get("Matches", 0) < min_prior_matches:
                continue

            home_venue = _rolling_team_features_v05(history, home_team, rolling_n, venue="home")
            away_venue = _rolling_team_features_v05(history, away_team, rolling_n, venue="away")
            league = _league_features(history, n=90)
            h2h = _h2h_features(history, home_team, away_team, n=5)
            referee = _referee_features(history, r.get("Referee", None), n=20)

            row = {
                "Competition": competition, "Season": r["Season"], "Date": r["Date"],
                "HomeTeam": home_team, "AwayTeam": away_team,
                "Match": f"{home_team} - {away_team}",
                "FTHG": r["FTHG"], "FTAG": r["FTAG"], "TotalGoals": r["FTHG"] + r["FTAG"],
                "Won_HOME": bool(r["FTHG"] > r["FTAG"]),
                "Won_DRAW": bool(r["FTHG"] == r["FTAG"]),
                "Won_AWAY": bool(r["FTHG"] < r["FTAG"]),
                "Won_OVER25": bool(r["FTHG"] + r["FTAG"] > 2.5),
                "Won_UNDER25": bool(r["FTHG"] + r["FTAG"] < 2.5),
                "Won_BTTS": bool(r["FTHG"] > 0 and r["FTAG"] > 0),
                "Home_RestDays": _rest_days(history, home_team, r["Date"]),
                "Away_RestDays": _rest_days(history, away_team, r["Date"]),
                "Home_SeasonMatchesBefore": int(((history["Season"] == r["Season"]) & ((history["HomeTeam"] == home_team) | (history["AwayTeam"] == home_team))).sum()),
                "Away_SeasonMatchesBefore": int(((history["Season"] == r["Season"]) & ((history["HomeTeam"] == away_team) | (history["AwayTeam"] == away_team))).sum()),
            }

            for prefix, features in [
                ("Home", home), ("Away", away),
                ("HomeVenue", home_venue), ("AwayVenue", away_venue),
                ("League", league), ("H2H", h2h), ("Referee", referee),
            ]:
                for k, v in features.items():
                    row[f"{prefix}_{k}"] = v

            row.update(_current_market_features(r))

            # Verschillen thuisploeg - uitploeg: handig als directe modelinput.
            for feature in [
                "PPG","GF","GA","GoalDiff","TotalGoals","WinRate","BTTSRate","CleanSheetRate",
                "ShotsFor","ShotsAgainst","SOTFor","SOTAgainst","CornersFor","CornersAgainst",
                "YellowFor","RedFor","ShotAccuracy","GoalConversionShots","GoalConversionSOT",
            ]:
                hv, av = row.get(f"Home_{feature}"), row.get(f"Away_{feature}")
                if hv is not None and av is not None and pd.notna(hv) and pd.notna(av):
                    row[f"Delta_{feature}"] = float(hv) - float(av)

            # Echte historische odds voor ROI-output.
            row["Odd_HOME"] = market_odds(r, "HOME", "closing_avg")
            row["Odd_DRAW"] = market_odds(r, "DRAW", "closing_avg")
            row["Odd_AWAY"] = market_odds(r, "AWAY", "closing_avg")
            row["Odd_OVER25"] = market_odds(r, "TOTAL_OVER_2.5", "closing_avg")
            row["Odd_UNDER25"] = market_odds(r, "TOTAL_UNDER_2.5", "closing_avg")
            rows.append(row)

    return pd.DataFrame(rows)


def feature_catalog_table(feature_table: Optional[pd.DataFrame] = None) -> pd.DataFrame:
    """Maak een leesbare catalogus van alle beschikbare prematch-features."""
    fixed_exclude = {
        "Competition","Season","Date","HomeTeam","AwayTeam","Match","FTHG","FTAG","TotalGoals",
        "Won_HOME","Won_DRAW","Won_AWAY","Won_OVER25","Won_UNDER25","Won_BTTS",
        "Odd_HOME","Odd_DRAW","Odd_AWAY","Odd_OVER25","Odd_UNDER25",
    }
    cols = list(feature_table.columns) if feature_table is not None and not feature_table.empty else []
    cols = [c for c in cols if c not in fixed_exclude]

    def category(c):
        if c.startswith("HomeVenue_") or c.startswith("AwayVenue_"): return "Thuis/uit-specifieke vorm"
        if c.startswith("Home_") or c.startswith("Away_"): return "Recente teamvorm"
        if c.startswith("League_"): return "Competitiecontext"
        if c.startswith("H2H_"): return "Head-to-head"
        if c.startswith("Referee_"): return "Scheidsrechter"
        if c.startswith("MarketOpen_"): return "Markt opening"
        if c.startswith("MarketClose_"): return "Markt closing"
        if c.startswith("MarketMove_"): return "Marktbeweging"
        if c.startswith("Delta_"): return "Teamverschil"
        return "Wedstrijdcontext"

    def timing(c):
        if c.startswith("MarketClose_") or c.startswith("MarketMove_"):
            return "Beschikbaar rond slotmarkt"
        return "Prematch / historisch"

    def pretty(c):
        replacements = {
            "_":" ", "PPG":"punten/wedstrijd", "GF":"goals voor", "GA":"goals tegen",
            "SOT":"shots on target", "BTTS":"BTTS", "HT":"1e helft",
            "Over2 5":"Over 2.5", "Over1 5":"Over 1.5", "Over3 5":"Over 3.5",
        }
        s = c
        for a,b in replacements.items():
            s = s.replace(a,b)
        return s

    return pd.DataFrame([
        {"Variabele": c, "Categorie": category(c), "Moment": timing(c), "Omschrijving": pretty(c)}
        for c in cols
    ])


def raw_variable_coverage(data: pd.DataFrame) -> pd.DataFrame:
    """Dekking van belangrijke ruwe Football-Data velden per competitie/seizoen."""
    if data.empty:
        return pd.DataFrame()

    catalog = {
        "Uitslag FT": ["FTHG","FTAG"],
        "Uitslag HT": ["HTHG","HTAG"],
        "Scheidsrechter": ["Referee"],
        "Toeschouwers": ["Attendance"],
        "Shots": ["HS","AS"],
        "Shots on target": ["HST","AST"],
        "Woodwork": ["HHW","AHW"],
        "Corners": ["HC","AC"],
        "Fouls": ["HF","AF"],
        "Free kicks conceded": ["HFKC","AFKC"],
        "Offsides": ["HO","AO"],
        "Gele kaarten": ["HY","AY"],
        "Rode kaarten": ["HR","AR"],
        "Booking points": ["HBP","ABP"],
        "1X2 opening gem.": ["AvgH","AvgD","AvgA"],
        "1X2 closing gem.": ["AvgCH","AvgCD","AvgCA"],
        "1X2 opening max.": ["MaxH","MaxD","MaxA"],
        "1X2 closing max.": ["MaxCH","MaxCD","MaxCA"],
        "O/U2.5 opening": ["Avg>2.5","Avg<2.5"],
        "O/U2.5 closing": ["AvgC>2.5","AvgC<2.5"],
        "Asian Handicap opening": ["AHh","AvgAHH","AvgAHA"],
        "Asian Handicap closing": ["AHCh","AvgCAHH","AvgCAHA"],
    }
    rows = []
    for (comp, season), g in data.groupby(["Competition","Season"], sort=False):
        for label, cols in catalog.items():
            existing = [c for c in cols if c in g.columns]
            if not existing:
                coverage = 0.0
            else:
                coverage = float(g[existing].notna().all(axis=1).mean())
            rows.append({
                "Competition": comp, "Season": season, "Variabele groep": label,
                "Dekking": coverage, "Rijen": len(g),
            })
    return pd.DataFrame(rows)


def raw_column_catalog(data: pd.DataFrame) -> pd.DataFrame:
    """Alle werkelijk aangetroffen bron- en afgeleide kolommen met dekking."""
    if data.empty:
        return pd.DataFrame()
    rows = []
    id_cols = {"Competition","Season","Div","Date","Time","HomeTeam","AwayTeam","Match"}
    for c in data.columns:
        if c in id_cols: cat = "Identificatie"
        elif c in {"FTHG","FTAG","FTR","HTHG","HTAG","HTR"}: cat = "Uitslag"
        elif c in MATCH_STAT_FIELDS: cat = "Matchstatistiek"
        elif c in ODDS_FIELDS or any(t in c for t in ["365","Avg","Max","PS","BFE","AH"]): cat = "Odds / markt"
        else: cat = "Afgeleid"
        rows.append({
            "Kolom": c,
            "Categorie": cat,
            "Dekking": float(data[c].notna().mean()),
            "Type": str(data[c].dtype),
        })
    return pd.DataFrame(rows).sort_values(["Categorie","Kolom"]).reset_index(drop=True)

# =============================================================================
# v0.6 — minimalistische startpagina / bet candidates
# =============================================================================

def fixture_bet_candidates(
    data: pd.DataFrame,
    competition: str,
    fixture_date,
    home_team: str,
    away_team: str,
    mode: str = "Afgelopen 3 volledige seizoenen",
    n_matches: int = 10,
    pseudo: int = 2,
    max_goals: int = 8,
    threshold: float = 0.85,
    market_scope: str = "Compact",
) -> pd.DataFrame:
    """
    Bereken kandidaat-bets voor één komende wedstrijd en retourneer alleen
    bets waarvan de modelkans >= threshold.

    FairOdd = 1 / ModelProb. Dit is een modelprijs, geen bookmakerprijs.
    """
    # Namen uit de seizoen-catalogus kunnen afwijken van Football-Data.
    # Maak daarom eerst expliciet de modelnamen aan en gebruik die vervolgens
    # zowel voor de historische selectie als voor de teamsterktes.
    model_home_team = resolve_catalog_team_name(
        data, competition, home_team
    )
    model_away_team = resolve_catalog_team_name(
        data, competition, away_team
    )

    base, home_hist, away_hist = select_prediction_history(
        data,
        competition,
        fixture_date,
        model_home_team,
        model_away_team,
        mode,
        n_matches,
    )

    if base.empty:
        avg_home, avg_away = 1.5, 1.25
    else:
        avg_home = float(base["FTHG"].mean())
        avg_away = float(base["FTAG"].mean())
        avg_home = avg_home if avg_home > 0 else 1.5
        avg_away = avg_away if avg_away > 0 else 1.25

    hs = _team_stats(home_hist, model_home_team, avg_home, avg_away, pseudo)
    aas = _team_stats(away_hist, model_away_team, avg_home, avg_away, pseudo)
    lam_home = float(np.clip(hs["home_attack"] * aas["away_defence"] * avg_home, 0.05, 6.0))
    lam_away = float(np.clip(aas["away_attack"] * hs["home_defence"] * avg_away, 0.05, 6.0))
    matrix = score_matrix(lam_home, lam_away, max_goals)

    p_home = float(np.tril(matrix, -1).sum())
    p_draw = float(np.trace(matrix))
    p_away = float(np.triu(matrix, 1).sum())

    candidates = []

    def add(category: str, bet: str, prob: float, line=np.nan):
        prob = float(np.clip(prob, 0.0, 1.0))
        candidates.append({
            "Competition": competition,
            "Date": pd.Timestamp(fixture_date) if pd.notna(fixture_date) else pd.NaT,
            "HomeTeam": home_team,
            "AwayTeam": away_team,
            "Match": f"{home_team} - {away_team}",
            "Category": category,
            "Bet": bet,
            "Line": line,
            "ModelProb": prob,
            "FairOdd": (1.0 / prob) if prob > 0 else np.nan,
            "lambda_home": lam_home,
            "lambda_away": lam_away,
            "HistoryMode": mode,
            "NMatches": n_matches,
        })

    # 1X2
    add("1X2", "Thuiswinst", p_home)
    add("1X2", "Gelijkspel", p_draw)
    add("1X2", "Uitwinst", p_away)

    # Dubbele kans
    add("Dubbele kans", "1X (thuis of gelijk)", p_home + p_draw)
    add("Dubbele kans", "X2 (gelijk of uit)", p_draw + p_away)
    add("Dubbele kans", "12 (geen gelijkspel)", p_home + p_away)

    # BTTS
    p_btts_no = float(matrix[0, :].sum() + matrix[:, 0].sum() - matrix[0, 0])
    add("BTTS", "BTTS - Ja", 1.0 - p_btts_no)
    add("BTTS", "BTTS - Nee", p_btts_no)

    # v0.7: alle goal-lijnen standaard van 0.5 t/m 5.5.
    total_lines = [0.5, 1.5, 2.5, 3.5, 4.5, 5.5]
    team_lines = [0.5, 1.5, 2.5, 3.5, 4.5, 5.5]

    # Totaal goals
    for line in total_lines:
        p_under = float(sum(
            matrix[i, j]
            for i in range(matrix.shape[0])
            for j in range(matrix.shape[1])
            if i + j < line
        ))
        add("Totaal goals", f"Over {line:g} goals", 1.0 - p_under, line)
        add("Totaal goals", f"Under {line:g} goals", p_under, line)

    # Teamgoals
    home_marginal = matrix.sum(axis=1)
    away_marginal = matrix.sum(axis=0)

    for line in team_lines:
        cutoff = int(np.floor(line))
        p_home_under = float(home_marginal[:cutoff + 1].sum())
        p_away_under = float(away_marginal[:cutoff + 1].sum())

        add("Teamgoals thuis", f"{home_team} over {line:g}", 1.0 - p_home_under, line)
        add("Teamgoals thuis", f"{home_team} under {line:g}", p_home_under, line)
        add("Teamgoals uit", f"{away_team} over {line:g}", 1.0 - p_away_under, line)
        add("Teamgoals uit", f"{away_team} under {line:g}", p_away_under, line)

    out = pd.DataFrame(candidates)
    if out.empty:
        return out

    out = out[out["ModelProb"] >= float(threshold)].copy()
    return out.sort_values(
        ["ModelProb", "Category", "Bet"],
        ascending=[False, True, True],
        kind="stable",
    ).reset_index(drop=True)


# =============================================================================
# v0.7 — actuele TOTO oddslaag
# =============================================================================

TOTO_LEAGUE_URLS = {
    "Eredivisie": "https://sport.toto.nl/wedden/sport/1176/nederland-eredivisie/overzicht",
    "Premier League": "https://sport.toto.nl/wedden/sport/567/engeland-premier-league/overzicht",
    "La Liga": "https://sport.toto.nl/wedden/sport/570/spanje-laliga/overzicht",
    "Bundesliga": "https://sport.toto.nl/wedden/sport/577/duitsland-bundesliga/overzicht",
    "Serie A": "https://sport.toto.nl/wedden/sport/644/italie-serie-a/overzicht",
    "Ligue 1": "https://sport.toto.nl/wedden/sport/911/frankrijk-ligue-1/wedstrijden",
}

TOTO_TEAM_ALIASES = {
    "psv": ["psv", "psv eindhoven"],
    "ajax": ["ajax"],
    "feyenoord": ["feyenoord"],
    "nec": ["nec", "n.e.c.", "n.e.c. nijmegen"],
    "sparta rotterdam": ["sparta", "sparta rotterdam"],
    "heerenveen": ["heerenveen", "sc heerenveen"],
    "telstar": ["telstar", "sc telstar"],
    "cambuur": ["cambuur", "sc cambuur"],
    "excelsior": ["excelsior", "excelsior rotterdam"],
    "twente": ["twente", "fc twente"],
    "utrecht": ["utrecht", "fc utrecht"],
    "groningen": ["groningen", "fc groningen"],
    "fortuna sittard": ["fortuna sittard"],
    "go ahead eagles": ["go ahead eagles"],
    "ado den haag": ["ado den haag"],
    "willem ii": ["willem ii"],
    "koln": ["koln", "köln", "1. fc koln", "1. fc köln"],
    "bayern munich": ["bayern munich", "bayern münchen", "bayern munchen"],
    "monchengladbach": ["borussia monchengladbach", "borussia mönchengladbach"],
    "paris sg": ["paris saint germain", "paris saint-germain", "psg"],
}


def _toto_norm(value: str) -> str:
    value = "" if value is None else str(value)
    value = unicodedata.normalize("NFKD", value)
    value = "".join(ch for ch in value if not unicodedata.combining(ch))
    value = value.lower().replace("&", " and ")
    value = re.sub(r"\b(fc|afc|sc|cf|ac|as|sv|vfb|rc|ud)\b", " ", value)
    value = re.sub(r"[^a-z0-9]+", " ", value)
    return re.sub(r"\s+", " ", value).strip()


def _team_tokens(team: str) -> list[str]:
    base = _toto_norm(team)
    variants = {base}
    for canonical, aliases in TOTO_TEAM_ALIASES.items():
        norm_aliases = {_toto_norm(a) for a in aliases}
        if base == _toto_norm(canonical) or base in norm_aliases:
            variants |= norm_aliases
            variants.add(_toto_norm(canonical))
    return [v for v in variants if v]


def _match_slug_score(slug: str, home_team: str, away_team: str) -> int:
    slug_norm = _toto_norm(slug.replace("-vs-", " "))
    home_hits = max((len(v) for v in _team_tokens(home_team) if v in slug_norm), default=0)
    away_hits = max((len(v) for v in _team_tokens(away_team) if v in slug_norm), default=0)
    return home_hits + away_hits if home_hits and away_hits else 0


def _http_get(url: str, timeout: int = 10) -> str:
    """HTTP GET met browserheaders en retry voor Streamlit Cloud."""
    headers = {
        "User-Agent": (
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
            "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/126.0 Safari/537.36"
        ),
        "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
        "Accept-Language": "nl-NL,nl;q=0.9,en;q=0.7",
        "Cache-Control": "no-cache",
        "Pragma": "no-cache",
    }
    retry = Retry(
        total=2,
        connect=2,
        read=1,
        backoff_factor=0.35,
        status_forcelist=(429, 500, 502, 503, 504),
        allowed_methods=frozenset(["GET"]),
        raise_on_status=False,
    )
    session = requests.Session()
    session.mount("https://", HTTPAdapter(max_retries=retry))
    response = session.get(
        url,
        timeout=(4, timeout),
        headers=headers,
        allow_redirects=True,
    )
    response.raise_for_status()
    return response.text

def find_toto_match_url(
    competition: str,
    home_team: str,
    away_team: str,
    timeout: int = 12,
) -> Optional[str]:
    """Zoek vanuit de publieke TOTO-competitiepagina de beste wedstrijdlink."""
    league_url = TOTO_LEAGUE_URLS.get(competition)
    if not league_url:
        return None

    try:
        html = _http_get(league_url, timeout)
    except Exception:
        return None

    soup = BeautifulSoup(html, "html.parser")
    links = []
    for a in soup.find_all("a", href=True):
        href = a.get("href", "")
        if "/wedden/wedstrijd/" not in href:
            continue
        if href.startswith("/"):
            href = "https://sport.toto.nl" + href
        links.append(href)

    links = list(dict.fromkeys(links))
    if not links:
        return None

    scored = []
    for url in links:
        slug = url.rstrip("/").split("/")[-1]
        score = _match_slug_score(slug, home_team, away_team)
        if score:
            scored.append((score, url))

    return max(scored, default=(0, None))[1]


def _decimal(text) -> float:
    """Lees zowel TOTO's 1,35 als 1.35 als decimale odd."""
    if text is None:
        return np.nan
    s = str(text).strip().replace(chr(160), " ").replace(",", ".")
    s = s.replace(" ", "")
    if not re.fullmatch(r"\d{1,3}(?:\.\d{1,3})?", s):
        return np.nan
    try:
        value = float(s)
        return value if value >= 1.0 else np.nan
    except Exception:
        return np.nan

def _lines_from_html(html: str) -> list[str]:
    soup = BeautifulSoup(html, "html.parser")
    text = soup.get_text("\n", strip=True)
    return [re.sub(r"\s+", " ", x).strip() for x in text.splitlines() if x.strip()]


def _next_decimal(lines: list[str], start: int, max_steps: int = 8) -> float:
    for j in range(start + 1, min(len(lines), start + 1 + max_steps)):
        raw = lines[j].strip()
        # Odds staan meestal als los numeriek token.
        if re.fullmatch(r"\d{1,3}[,.]\d{1,3}", raw):
            return _decimal(raw)
    return np.nan


def parse_toto_match_odds(html: str, home_team: str, away_team: str) -> dict:
    """
    Best-effort parser voor publieke TOTO wedstrijdpagina.
    Markten: 1X2, dubbele kans, BTTS, totaal goals 0.5 t/m 5.5.
    """
    lines = _lines_from_html(html)
    odds = {}

    # Resultaat - pak eerste voorkomens na de marktnaam.
    for i, line in enumerate(lines):
        if line.lower().startswith("resultaat - vroege uitbetaling"):
            window = lines[i + 1:i + 18]
            nums = [_decimal(x) for x in window if re.fullmatch(r"\d{1,3}[,.]\d{1,3}", x)]
            nums = [x for x in nums if pd.notna(x)]
            if len(nums) >= 3:
                odds["HOME"] = nums[0]
                odds["DRAW"] = nums[1]
                odds["AWAY"] = nums[2]
                break

    # Dubbele kans
    for i, line in enumerate(lines):
        low = _toto_norm(line)
        if low == "dubbele kans":
            # Zoek de eerstvolgende drie keuze/odd paren.
            choices = []
            j = i + 1
            while j < min(len(lines), i + 30) and len(choices) < 3:
                if " of " in lines[j].lower():
                    odd = _next_decimal(lines, j, 4)
                    if pd.notna(odd):
                        choices.append((lines[j], odd))
                j += 1
            for label, odd in choices:
                nl = _toto_norm(label)
                if "gelijkspel" in nl:
                    # 1X of X2
                    home_match = any(v in nl for v in _team_tokens(home_team))
                    away_match = any(v in nl for v in _team_tokens(away_team))
                    if home_match:
                        odds["DC_1X"] = odd
                    elif away_match:
                        odds["DC_X2"] = odd
                else:
                    odds["DC_12"] = odd
            break

    # BTTS
    for i, line in enumerate(lines):
        if _toto_norm(line) == "beide teams scoren":
            for j in range(i + 1, min(len(lines), i + 12)):
                low = _toto_norm(lines[j])
                if low == "ja":
                    odd = _next_decimal(lines, j, 3)
                    if pd.notna(odd):
                        odds["BTTS_YES"] = odd
                elif low == "nee":
                    odd = _next_decimal(lines, j, 3)
                    if pd.notna(odd):
                        odds["BTTS_NO"] = odd
            break

    # Total goals lines 0.5–5.5. Gebruik alle gevonden Over/Under tokens,
    # ook als sommige achter "Bekijk meer" server-side aanwezig zijn.
    line_re = re.compile(r"^(Over|Under)\s+([0-5][.,]5)$", re.I)
    for i, line in enumerate(lines):
        m = line_re.match(line.strip())
        if not m:
            continue
        side = m.group(1).upper()
        gl = float(m.group(2).replace(",", "."))
        odd = _next_decimal(lines, i, 3)
        if pd.notna(odd):
            odds[f"TOTAL_{side}_{gl}"] = odd

    return odds


def fetch_toto_match_odds(
    competition: str,
    home_team: str,
    away_team: str,
    timeout: int = 12,
) -> dict:
    """Vind de TOTO wedstrijdpagina en lees actuele odds uit. Faalt stil met status."""
    url = find_toto_match_url(competition, home_team, away_team, timeout=timeout)
    if not url:
        return {
            "_status": "Geen TOTO-wedstrijdpagina gevonden",
            "_url": None,
        }
    try:
        html = _http_get(url, timeout)
        odds = parse_toto_match_odds(html, home_team, away_team)
        odds["_status"] = "TOTO odds opgehaald" if odds else "TOTO-pagina gevonden, odds niet leesbaar"
        odds["_url"] = url
        return odds
    except Exception as exc:
        return {
            "_status": f"TOTO ophalen mislukt: {type(exc).__name__}",
            "_url": url,
        }


def fetch_toto_competition_index(
    competition: str,
    timeout: int = 10,
) -> dict:
    """Haal één competitiepagina op en verzamel TOTO-wedstrijdlinks."""
    base_url = TOTO_LEAGUE_URLS.get(competition)
    if not base_url:
        return {"_status": "Onbekende TOTO-competitie", "_links": []}

    urls = [base_url]
    if base_url.endswith("/overzicht"):
        urls.append(base_url[:-len("overzicht")] + "wedstrijden")
    elif base_url.endswith("/wedstrijden"):
        urls.append(base_url[:-len("wedstrijden")] + "overzicht")

    html = None
    used_url = None
    errors = []
    for url in urls:
        try:
            candidate = _http_get(url, timeout=timeout)
            if candidate and len(candidate) > 500:
                html = candidate
                used_url = url
                break
        except Exception as exc:
            errors.append(type(exc).__name__)

    if not html:
        return {
            "_status": "TOTO competitiepagina niet bereikbaar"
            + (f" ({'/'.join(errors)})" if errors else ""),
            "_links": [],
            "_source": None,
        }

    soup = BeautifulSoup(html, "html.parser")
    links = []
    for a in soup.find_all("a", href=True):
        href = str(a.get("href", ""))
        if "/wedden/wedstrijd/" not in href:
            continue
        if href.startswith("/"):
            href = "https://sport.toto.nl" + href
        elif not href.startswith("http"):
            href = "https://sport.toto.nl/" + href.lstrip("/")
        links.append(href.split("?")[0])

    links = list(dict.fromkeys(links))
    return {
        "_status": f"{len(links)} TOTO wedstrijdlinks gevonden",
        "_links": links,
        "_source": used_url,
    }


def _find_url_in_index(index: dict, home_team: str, away_team: str):
    scored = []
    for url in index.get("_links", []):
        slug = url.rstrip("/").split("/")[-1]
        score = _match_slug_score(slug, home_team, away_team)
        if score:
            scored.append((score, url))
    return max(scored, default=(0, None))[1]


def fetch_toto_match_odds_from_url(
    url: str,
    home_team: str,
    away_team: str,
    timeout: int = 10,
) -> dict:
    if not url:
        return {"_status": "Geen TOTO-wedstrijdlink", "_url": None}
    try:
        html = _http_get(url, timeout=timeout)
        odds = parse_toto_match_odds(html, home_team, away_team)
        count = sum(
            1
            for key, value in odds.items()
            if not str(key).startswith("_") and pd.notna(value)
        )
        odds["_status"] = (
            f"{count} TOTO odds gelezen"
            if count
            else "TOTO-pagina gevonden, markten niet leesbaar"
        )
        odds["_url"] = url
        return odds
    except Exception as exc:
        return {
            "_status": f"TOTO wedstrijdpagina mislukt ({type(exc).__name__})",
            "_url": url,
        }


def _toto_match_cache_key(home_team: str, away_team: str) -> str:
    return f"{_toto_norm(home_team)}||{_toto_norm(away_team)}"


def fetch_toto_week_odds(
    competition: str,
    matches,
    timeout: int = 10,
    max_workers: int = 6,
) -> dict:
    """Laad TOTO voor een hele competitie/week in één bulk-run."""
    pairs = [(str(home), str(away)) for home, away in matches]
    index = fetch_toto_competition_index(competition, timeout=timeout)

    results = {
        "_status": index.get("_status", ""),
        "_source": index.get("_source"),
        "_matches_requested": len(pairs),
        "_matches_linked": 0,
    }

    jobs = []
    for home, away in pairs:
        url = _find_url_in_index(index, home, away)
        if not url:
            url = find_toto_match_url(
                competition,
                home,
                away,
                timeout=timeout,
            )
        if url:
            results["_matches_linked"] += 1
        jobs.append((home, away, url))

    def worker(job):
        home, away, url = job
        return (
            _toto_match_cache_key(home, away),
            fetch_toto_match_odds_from_url(
                url, home, away, timeout=timeout
            ),
        )

    worker_count = max(1, min(int(max_workers), len(jobs) or 1))
    with ThreadPoolExecutor(max_workers=worker_count) as executor:
        futures = [executor.submit(worker, job) for job in jobs]
        for future in as_completed(futures):
            try:
                key, odds = future.result()
                results[key] = odds
            except Exception:
                continue

    return results


def toto_week_result_for_match(
    week_results: dict,
    home_team: str,
    away_team: str,
) -> dict:
    return week_results.get(
        _toto_match_cache_key(home_team, away_team),
        {
            "_status": "Geen TOTO-data voor deze wedstrijd",
            "_url": None,
        },
    )

def toto_key_for_candidate(row) -> Optional[str]:
    bet = str(row.get("Bet", ""))
    category = str(row.get("Category", ""))

    direct = {
        "Thuiswinst": "HOME",
        "Gelijkspel": "DRAW",
        "Uitwinst": "AWAY",
        "1X (thuis of gelijk)": "DC_1X",
        "X2 (gelijk of uit)": "DC_X2",
        "12 (geen gelijkspel)": "DC_12",
        "BTTS - Ja": "BTTS_YES",
        "BTTS - Nee": "BTTS_NO",
    }
    if bet in direct:
        return direct[bet]

    m = re.fullmatch(r"(Over|Under) ([0-5](?:\.5)) goals", bet)
    if category == "Totaal goals" and m:
        return f"TOTAL_{m.group(1).upper()}_{float(m.group(2))}"

    # Via de gestructureerde odds-feed kunnen team totals wél betrouwbaar
    # worden gekoppeld wanneer TOTO ze voor die wedstrijd aanbiedt.
    tm = re.search(r"\s(over|under)\s([0-5](?:\.5))$", bet, re.I)
    if tm and category == "Teamgoals thuis":
        return f"TEAM_HOME_{tm.group(1).upper()}_{float(tm.group(2))}"
    if tm and category == "Teamgoals uit":
        return f"TEAM_AWAY_{tm.group(1).upper()}_{float(tm.group(2))}"

    return None


def attach_toto_odds(candidates: pd.DataFrame, toto_odds: dict) -> pd.DataFrame:
    if candidates.empty:
        return candidates.copy()

    out = candidates.copy()
    keys = out.apply(toto_key_for_candidate, axis=1)
    out["TotoKey"] = keys
    out["TotoOdd"] = [toto_odds.get(k, np.nan) if k else np.nan for k in keys]
    out["ValuePct"] = np.where(
        pd.to_numeric(out["TotoOdd"], errors="coerce") > 1,
        out["ModelProb"] * pd.to_numeric(out["TotoOdd"], errors="coerce") - 1.0,
        np.nan,
    )
    out["TotoURL"] = toto_odds.get("_url")
    out["TotoStatus"] = toto_odds.get("_status", "")
    return out

# =============================================================================
# v0.8 — volledige 2026/27 weekcatalogus
# =============================================================================

FIXTURE_DOWNLOAD_PAGES = {
    "Eredivisie": "https://fixturedownload.com/results/eredivisie-2026",
    "Premier League": "https://fixturedownload.com/results/epl-2026",
    "La Liga": "https://fixturedownload.com/results/la-liga-2026",
    "Bundesliga": "https://fixturedownload.com/results/bundesliga-2026",
    "Serie A": "https://fixturedownload.com/results/serie-a-2026",
    "Ligue 1": "https://fixturedownload.com/results/ligue-1-2026",
}

# Verschillende databronnen schrijven clubs soms anders. Dit register vangt de
# bekendste verschillen op; daarna gebruiken we een fuzzy fallback.
CATALOG_TEAM_ALIASES = {
    "Eredivisie": {
        "n e c nijmegen": "NEC",
        "sc cambuur": "Cambuur",
        "sc heerenveen": "Heerenveen",
        "excelsior rotterdam": "Excelsior",
        "fc groningen": "Groningen",
        "fc twente": "Twente",
        "fc utrecht": "Utrecht",
        "sparta rotterdam": "Sparta Rotterdam",
        "fortuna sittard": "Fortuna Sittard",
        "go ahead eagles": "Go Ahead Eagles",
        "ado den haag": "ADO Den Haag",
        "willem ii": "Willem II",
    },
    "Premier League": {
        "man city": "Man City",
        "man utd": "Man United",
        "nott m forest": "Nott'm Forest",
        "spurs": "Tottenham",
        "newcastle": "Newcastle",
        "brighton": "Brighton",
        "bournemouth": "Bournemouth",
        "coventry": "Coventry",
        "hull": "Hull",
        "ipswich": "Ipswich",
    },
    "La Liga": {
        "athletic club": "Ath Bilbao",
        "atletico de madrid": "Ath Madrid",
        "deportivo alaves": "Alaves",
        "fc barcelona": "Barcelona",
        "rcd espanyol de barcelona": "Espanyol",
        "real betis": "Betis",
        "real sociedad": "Sociedad",
        "rayo vallecano": "Vallecano",
        "deportivo la coruna": "La Coruna",
        "racing santander": "Santander",
    },
    "Bundesliga": {
        "fc bayern munchen": "Bayern Munich",
        "bayer 04 leverkusen": "Leverkusen",
        "borussia monchengladbach": "M'gladbach",
        "1 fc koln": "FC Koln",
        "1 fc union berlin": "Union Berlin",
        "1 fsv mainz 05": "Mainz",
        "sport club freiburg": "Freiburg",
        "sv werder bremen": "Werder Bremen",
        "fc schalke 04": "Schalke 04",
        "sc paderborn 07": "Paderborn",
        "sv 07 elversberg": "Elversberg",
    },
    "Serie A": {
        "internazionale": "Inter",
        "inter": "Inter",
        "ac milan": "AC Milan",
        "milan": "AC Milan",
        "roma": "Roma",
        "napoli": "Napoli",
        "juventus": "Juventus",
        "fiorentina": "Fiorentina",
        "lecce": "Lecce",
    },
    "Ligue 1": {
        "paris saint germain": "Paris SG",
        "paris saint germain psg": "Paris SG",
        "olympique marseille": "Marseille",
        "olympique lyon": "Lyon",
        "racing club de lens": "Lens",
        "lille osc": "Lille",
        "stade rennais fc": "Rennes",
        "strasbourg alsace": "Strasbourg",
        "as monaco": "Monaco",
        "ogc nice": "Nice",
    },
}


def _catalog_norm(value: str) -> str:
    value = "" if value is None else str(value)
    value = unicodedata.normalize("NFKD", value)
    value = "".join(ch for ch in value if not unicodedata.combining(ch))
    value = value.lower().replace("&", " and ")
    value = re.sub(r"[^a-z0-9]+", " ", value)
    return re.sub(r"\s+", " ", value).strip()


def resolve_catalog_team_name(
    historical_data: pd.DataFrame,
    competition: str,
    schedule_team: str,
) -> str:
    """
    Zet de naam uit de seizoen-kalender om naar de teamnaam uit Football-Data.
    Bij geen exacte alias wordt de beste tekstmatch gebruikt.
    """
    comp = historical_data[historical_data["Competition"] == competition]
    historical_teams = sorted(
        set(comp.get("HomeTeam", pd.Series(dtype=str)).dropna().astype(str))
        | set(comp.get("AwayTeam", pd.Series(dtype=str)).dropna().astype(str))
    )
    if not historical_teams:
        return str(schedule_team)

    target = _catalog_norm(schedule_team)

    # Handmatige alias -> probeer die vervolgens exact tegen historische teams.
    alias = CATALOG_TEAM_ALIASES.get(competition, {}).get(target)
    if alias:
        alias_norm = _catalog_norm(alias)
        for team in historical_teams:
            if _catalog_norm(team) == alias_norm:
                return team

    # Exact genormaliseerd.
    for team in historical_teams:
        if _catalog_norm(team) == target:
            return team

    # Token containment werkt goed voor FC/SC/AC-varianten.
    contained = [
        team for team in historical_teams
        if target in _catalog_norm(team) or _catalog_norm(team) in target
    ]
    if len(contained) == 1:
        return contained[0]

    # Fuzzy laatste redmiddel.
    from difflib import SequenceMatcher
    scored = [
        (SequenceMatcher(None, target, _catalog_norm(team)).ratio(), team)
        for team in historical_teams
    ]
    score, team = max(scored, default=(0.0, str(schedule_team)))
    return team if score >= 0.48 else str(schedule_team)


def _parse_fixture_download_table(
    html: str,
    competition: str,
) -> pd.DataFrame:
    soup = BeautifulSoup(html, "html.parser")
    target_table = None

    for table in soup.find_all("table"):
        headers = [
            _catalog_norm(th.get_text(" ", strip=True))
            for th in table.find_all("th")
        ]
        joined = " | ".join(headers)
        if "home team" in joined and "away team" in joined and "date" in joined:
            target_table = table
            break

    if target_table is None:
        return pd.DataFrame()

    records = []
    for tr in target_table.find_all("tr"):
        tds = [td.get_text(" ", strip=True) for td in tr.find_all("td")]
        if len(tds) < 5:
            continue

        # Fixture Download gebruikt: Round, Date, Location, Home, Away, Result.
        round_no = tds[0] if len(tds) > 0 else ""
        date_raw = tds[1] if len(tds) > 1 else ""
        location = tds[2] if len(tds) > 2 else ""
        home = tds[3] if len(tds) > 3 else ""
        away = tds[4] if len(tds) > 4 else ""
        result = tds[5] if len(tds) > 5 else ""

        dt = pd.to_datetime(date_raw, dayfirst=True, errors="coerce")
        if pd.isna(dt) or not home or not away:
            continue

        records.append({
            "Competition": competition,
            "Round": pd.to_numeric(round_no, errors="coerce"),
            "DateTime": dt,
            "Date": dt.normalize(),
            "Time": dt.strftime("%H:%M"),
            "Location": location,
            "HomeTeam": home,
            "AwayTeam": away,
            "Result": result,
        })

    df = pd.DataFrame(records)
    if df.empty:
        return df

    iso = df["Date"].dt.isocalendar()
    df["ISOYear"] = iso.year.astype(int)
    df["ISOWeek"] = iso.week.astype(int)
    df["WeekKey"] = (
        df["ISOYear"].astype(str)
        + "-W"
        + df["ISOWeek"].astype(str).str.zfill(2)
    )
    df["Match"] = df["HomeTeam"] + " - " + df["AwayTeam"]
    df["Status"] = np.where(
        df["Result"].astype(str).str.contains(r"\d\s*-\s*\d", regex=True),
        "Gespeeld",
        "Gepland",
    )
    return df.sort_values(["DateTime", "Competition", "HomeTeam"], kind="stable").reset_index(drop=True)


def load_full_season_fixture_catalog(
    competitions: Optional[Iterable[str]] = None,
    timeout: int = 15,
) -> pd.DataFrame:
    """
    Haal de volledige 2026/27 kalender op voor onze zes competities.
    Bron: Fixture Download result pages. Bij een mislukte competitie proberen
    we Football-Data fixtures.csv als near-term fallback.
    """
    competitions = list(competitions or FIXTURE_DOWNLOAD_PAGES.keys())
    frames = []

    for competition in competitions:
        url = FIXTURE_DOWNLOAD_PAGES.get(competition)
        if not url:
            continue
        try:
            html = _http_get(url, timeout=timeout)
            parsed = _parse_fixture_download_table(html, competition)
            if not parsed.empty:
                parsed["SourceURL"] = url
                frames.append(parsed)
        except Exception:
            continue

    if frames:
        out = pd.concat(frames, ignore_index=True, sort=False)
    else:
        out = pd.DataFrame()

    # Near-term fallback/aanvulling vanuit Football-Data.
    try:
        near = load_fixtures()
        if not near.empty:
            near = near.copy()
            if "ISOYear" not in near.columns:
                near["ISOYear"] = near["Date"].dt.isocalendar().year.astype("Int64")
            near["WeekKey"] = (
                near["ISOYear"].astype("Int64").astype(str)
                + "-W"
                + near["ISOWeek"].astype("Int64").astype(str).str.zfill(2)
            )
            near["DateTime"] = pd.to_datetime(
                near["Date"].dt.strftime("%Y-%m-%d")
                + " "
                + near.get("Time", pd.Series("00:00", index=near.index)).fillna("00:00").astype(str),
                errors="coerce",
            )
            near["Round"] = np.nan
            near["Location"] = ""
            near["Result"] = ""
            near["Status"] = "Gepland"
            near["SourceURL"] = FIXTURES_URL
            keep = [
                "Competition","Round","DateTime","Date","Time","Location",
                "HomeTeam","AwayTeam","Result","ISOYear","ISOWeek","WeekKey",
                "Match","Status","SourceURL"
            ]
            near = near[[c for c in keep if c in near.columns]]

            if out.empty:
                out = near
            else:
                # Alleen toevoegen als dezelfde competitie/datum/teams nog niet bestaat.
                key_cols = ["Competition","Date","HomeTeam","AwayTeam"]
                merged_keys = set(map(tuple, out[key_cols].astype(str).to_numpy()))
                add_mask = [
                    tuple(row) not in merged_keys
                    for row in near[key_cols].astype(str).to_numpy()
                ]
                if any(add_mask):
                    out = pd.concat([out, near.loc[add_mask]], ignore_index=True, sort=False)
    except Exception:
        pass

    if out.empty:
        return out

    # Gebruikerswens: seizoen tot en met ISO-week 25 van 2027.
    start_date = pd.Timestamp("2026-08-01")
    end_date = pd.Timestamp("2027-06-27")  # einde ISO-week 25 2027
    out = out[(out["Date"] >= start_date) & (out["Date"] <= end_date)].copy()

    return out.sort_values(
        ["DateTime","Competition","HomeTeam"],
        kind="stable"
    ).reset_index(drop=True)


def season_week_keys(
    start_date: str = "2026-08-01",
    end_date: str = "2027-06-27",
) -> list[str]:
    """Alle ISO-weekkeys in de gewenste catalogusperiode, óók lege weken."""
    dates = pd.date_range(start_date, end_date, freq="D")
    iso = dates.isocalendar()
    keys = (
        iso["year"].astype(str)
        + "-W"
        + iso["week"].astype(str).str.zfill(2)
    )
    return list(dict.fromkeys(keys.tolist()))

# =============================================================================
# v1.0.2 — gestructureerde TOTO NL odds-feed
# =============================================================================

from functools import lru_cache
import time as _time

ODDSPAPI_BASE = "https://api.oddspapi.io/v4"

# Naam + land is nodig omdat tournamentnamen niet uniek zijn.
ODDSPAPI_TOURNAMENT_TARGETS = {
    "Eredivisie": ({"eredivisie"}, {"netherlands", "holland"}),
    "Premier League": ({"premier league"}, {"england"}),
    "La Liga": ({"laliga", "la liga"}, {"spain"}),
    "Bundesliga": ({"bundesliga"}, {"germany"}),
    "Serie A": ({"serie a"}, {"italy"}),
    "Ligue 1": ({"ligue 1"}, {"france"}),
}


def _op_norm(value) -> str:
    return _toto_norm("" if value is None else str(value))


def _oddspapi_get(api_key: str, endpoint: str, **params):
    """
    GET met respect voor OddsPapi's 429/retryMs response.
    """
    if not api_key:
        raise ValueError("Geen OddsPapi API key ingesteld")

    params = dict(params)
    params["apiKey"] = api_key
    url = f"{ODDSPAPI_BASE}/{endpoint.lstrip('/')}"

    last_error = None
    for attempt in range(5):
        try:
            r = requests.get(url, params=params, timeout=45)
            if r.status_code == 429:
                retry_ms = 1500
                try:
                    body = r.json()
                    retry_ms = int(
                        body.get("error", {}).get("retryMs", retry_ms)
                    )
                except Exception:
                    pass
                _time.sleep(retry_ms / 1000 + 0.25)
                continue
            if r.status_code == 404:
                return []
            r.raise_for_status()
            return r.json()
        except Exception as exc:
            last_error = exc
            if attempt < 4:
                _time.sleep(0.5 + attempt * 0.5)

    raise RuntimeError(
        f"OddsPapi request mislukt: {type(last_error).__name__ if last_error else 'onbekend'}"
    )


@lru_cache(maxsize=8)
def oddspapi_soccer_markets(api_key: str) -> dict:
    rows = _oddspapi_get(
        api_key, "markets", language="en"
    )
    if isinstance(rows, dict):
        rows = rows.get("data", rows.get("markets", []))
    return {
        str(row.get("marketId")): row
        for row in (rows or [])
        if row.get("sportId") in (10, "10") or row.get("sportId") is None
    }


@lru_cache(maxsize=8)
def oddspapi_tournament_ids(api_key: str) -> dict:
    rows = _oddspapi_get(
        api_key, "tournaments", sportId=10
    )
    if isinstance(rows, dict):
        rows = rows.get("data", rows.get("tournaments", []))

    found = {}
    for competition, (names, countries) in ODDSPAPI_TOURNAMENT_TARGETS.items():
        hits = []
        for row in rows or []:
            name = _op_norm(row.get("tournamentName"))
            country = _op_norm(row.get("categoryName"))
            if name in names and country in countries:
                hits.append(row)
        if hits:
            # Prefer a tournament that actually reports future/upcoming fixtures.
            hits.sort(
                key=lambda x: (
                    int(x.get("futureFixtures") or 0)
                    + int(x.get("upcomingFixtures") or 0)
                ),
                reverse=True,
            )
            found[competition] = int(hits[0]["tournamentId"])
    return found


def _iter_oddspapi_fixtures(payload):
    if payload is None:
        return []
    if isinstance(payload, list):
        return payload
    if isinstance(payload, dict):
        for key in ("data", "fixtures", "events"):
            value = payload.get(key)
            if isinstance(value, list):
                return value
        # Some endpoint responses can be a single fixture object.
        if "fixtureId" in payload:
            return [payload]
        # Or a dict keyed by fixture IDs.
        values = [
            value for value in payload.values()
            if isinstance(value, dict) and "fixtureId" in value
        ]
        if values:
            return values
    return []


def fetch_oddspapi_toto_board(api_key: str) -> dict:
    """
    Eén odds-by-tournaments request haalt de actuele TOTO-prijzen voor alle
    zes competities tegelijk op.
    """
    tournament_ids = oddspapi_tournament_ids(api_key)
    if not tournament_ids:
        return {
            "_status": "Odds-feed: geen competities gevonden",
            "_fixtures": [],
        }

    ids = ",".join(
        str(tournament_ids[c])
        for c in COMPETITIONS
        if c in tournament_ids
    )
    payload = _oddspapi_get(
        api_key,
        "odds-by-tournaments",
        tournamentIds=ids,
        bookmakers="toto.nl",
        language="en",
        verbosity=3,
        oddsFormat="decimal",
    )
    fixtures = _iter_oddspapi_fixtures(payload)
    return {
        "_status": f"Odds-feed: {len(fixtures)} fixtures ontvangen",
        "_fixtures": fixtures,
        "_tournament_ids": tournament_ids,
    }


def _active_player_price(players):
    if not isinstance(players, dict):
        return np.nan, None

    # Prefer the generic player 0 for normal (non-player-prop) markets.
    ordered = []
    if "0" in players:
        ordered.append(players["0"])
    ordered.extend(
        value for key, value in players.items()
        if key != "0"
    )

    for item in ordered:
        # Historical endpoints sometimes return lists; current odds normally dict.
        candidates = item if isinstance(item, list) else [item]
        for quote in candidates:
            if not isinstance(quote, dict):
                continue
            price = pd.to_numeric(
                pd.Series([quote.get("price")]), errors="coerce"
            ).iloc[0]
            if pd.notna(price) and price > 1:
                if quote.get("active", True) is not False:
                    return float(price), quote.get("bookmakerOutcomeId")
    return np.nan, None


def _market_line(meta: dict, bookmaker_outcome_id=None):
    line = pd.to_numeric(
        pd.Series([meta.get("handicap")]), errors="coerce"
    ).iloc[0]
    if pd.notna(line):
        return float(line)

    text = "" if bookmaker_outcome_id is None else str(bookmaker_outcome_id)
    m = re.search(r"([0-9]+(?:\\.[0-9]+)?)", text)
    return float(m.group(1)) if m else np.nan


def _outcome_side(meta: dict, outcome_id: str, bookmaker_outcome_id=None):
    for outcome in meta.get("outcomes", []) or []:
        if str(outcome.get("outcomeId")) == str(outcome_id):
            return _op_norm(outcome.get("outcomeName"))
    return _op_norm(bookmaker_outcome_id)


def _is_team_one_market(name: str) -> bool:
    tokens = (
        "team 1", "team1", "home team", "home total",
        "participant 1", "participant1"
    )
    return any(token in name for token in tokens)


def _is_team_two_market(name: str) -> bool:
    tokens = (
        "team 2", "team2", "away team", "away total",
        "participant 2", "participant2"
    )
    return any(token in name for token in tokens)


def oddspapi_fixture_to_toto_odds(
    fixture: dict,
    market_catalog: dict,
) -> dict:
    """
    Vertaal het volledige TOTO bookmakerOdds object naar dezelfde keys die de
    model-candidates gebruiken. Dit is generiek: line 0.5/1.5/... hoeft niet
    hardcoded op market-ID te worden.
    """
    bookmaker = (fixture.get("bookmakerOdds") or {}).get("toto.nl") or {}
    markets = bookmaker.get("markets") or {}

    odds = {
        "_source": "OddsPapi / TOTO NL",
        "_url": bookmaker.get("fixturePath"),
        "_updated_at": fixture.get("updatedAt"),
        "_status": "TOTO structured feed",
    }

    for market_id, market_data in markets.items():
        meta = market_catalog.get(str(market_id), {})
        market_name = _op_norm(meta.get("marketName"))
        market_type = _op_norm(meta.get("marketType"))
        period = _op_norm(meta.get("period"))

        # Geen player props voor onze startpagina.
        if meta.get("playerProp") is True:
            continue
        if period and period not in ("fulltime", "full time", "result", "match"):
            continue

        for outcome_id, outcome_data in (market_data.get("outcomes") or {}).items():
            price, bookmaker_outcome_id = _active_player_price(
                outcome_data.get("players") or {}
            )
            if pd.isna(price):
                continue

            side = _outcome_side(
                meta, str(outcome_id), bookmaker_outcome_id
            )
            bo = _op_norm(bookmaker_outcome_id)

            # 1X2
            if (
                market_type == "1x2"
                or "full time result" in market_name
                or market_name in ("result", "regular time result")
            ):
                if side in ("1", "home", "team 1"):
                    odds["HOME"] = price
                elif side in ("x", "draw"):
                    odds["DRAW"] = price
                elif side in ("2", "away", "team 2"):
                    odds["AWAY"] = price
                continue

            # Beide teams scoren
            if "both teams to score" in market_name or "both team to score" in market_name:
                if side in ("yes", "ja"):
                    odds["BTTS_YES"] = price
                elif side in ("no", "nee"):
                    odds["BTTS_NO"] = price
                continue

            # Dubbele kans
            if "double chance" in market_name:
                token = side.replace(" ", "")
                token2 = bo.replace(" ", "")
                combined = token or token2
                if combined in ("1x", "homeordraw"):
                    odds["DC_1X"] = price
                elif combined in ("x2", "draworaway"):
                    odds["DC_X2"] = price
                elif combined in ("12", "homeoraway"):
                    odds["DC_12"] = price
                continue

            # Over/Under lijnen.
            is_total_market = (
                market_type == "totals"
                or "over under" in market_name
                or "total" in market_name
            )
            if is_total_market:
                line = _market_line(meta, bookmaker_outcome_id)
                if pd.isna(line):
                    continue
                # Alleen de lijnen die ons model toont.
                if line not in (0.5, 1.5, 2.5, 3.5, 4.5, 5.5):
                    continue

                direction = None
                if side.startswith("over") or "/over" in bo or bo.endswith("over"):
                    direction = "OVER"
                elif side.startswith("under") or "/under" in bo or bo.endswith("under"):
                    direction = "UNDER"
                if direction is None:
                    continue

                if _is_team_one_market(market_name):
                    odds[f"TEAM_HOME_{direction}_{line}"] = price
                elif _is_team_two_market(market_name):
                    odds[f"TEAM_AWAY_{direction}_{line}"] = price
                else:
                    # Exclude obvious half/player/team totals accidentally falling through.
                    if any(
                        text in market_name
                        for text in ("1st half", "first half", "2nd half", "second half", "player")
                    ):
                        continue
                    odds[f"TOTAL_{direction}_{line}"] = price

    count = sum(
        1 for key, value in odds.items()
        if not key.startswith("_") and pd.notna(value)
    )
    odds["_status"] = f"{count} actuele TOTO odds uit structured feed"
    return odds


def _fixture_name_score(fixture: dict, home_team: str, away_team: str) -> float:
    h = _op_norm(fixture.get("participant1Name") or fixture.get("participant1ShortName"))
    a = _op_norm(fixture.get("participant2Name") or fixture.get("participant2ShortName"))
    th = _op_norm(home_team)
    ta = _op_norm(away_team)

    direct = SequenceMatcher(None, h, th).ratio() + SequenceMatcher(None, a, ta).ratio()
    reverse = SequenceMatcher(None, h, ta).ratio() + SequenceMatcher(None, a, th).ratio()
    # Fixture orientation must normally match home/away; penalize reverse.
    return max(direct, reverse - 0.35)


def oddspapi_board_for_matches(
    board: dict,
    competition: str,
    matches,
    api_key: str,
) -> dict:
    market_catalog = oddspapi_soccer_markets(api_key)
    fixtures = board.get("_fixtures", []) if isinstance(board, dict) else []

    results = {
        "_status": board.get("_status", "Odds-feed"),
        "_source": "OddsPapi / TOTO NL",
        "_matches_requested": len(matches),
        "_matches_linked": 0,
    }

    # Restrict tournament when possible.
    tid = (board.get("_tournament_ids") or {}).get(competition)
    if tid is not None:
        fixtures = [
            f for f in fixtures
            if int(f.get("tournamentId") or -1) == int(tid)
        ]

    for home, away in matches:
        scored = [
            (_fixture_name_score(f, home, away), f)
            for f in fixtures
        ]
        score, fixture = max(scored, default=(0.0, None), key=lambda x: x[0])
        key = _toto_match_cache_key(home, away)
        if fixture is not None and score >= 1.15:
            parsed = oddspapi_fixture_to_toto_odds(
                fixture, market_catalog
            )
            results[key] = parsed
            results["_matches_linked"] += 1
        else:
            results[key] = {
                "_status": "Geen TOTO fixture in structured feed",
                "_source": "OddsPapi / TOTO NL",
                "_url": None,
            }
    return results


# Bewaar de directe TOTO-site scraper als fallback.
_fetch_toto_week_odds_website = fetch_toto_week_odds


def fetch_toto_week_odds(
    competition: str,
    matches,
    timeout: int = 10,
    max_workers: int = 6,
    api_key: Optional[str] = None,
    api_board: Optional[dict] = None,
) -> dict:
    """
    Primair: structured TOTO NL feed als API key is ingesteld.
    Fallback: publieke TOTO website voor de direct zichtbare markten.
    Er worden nooit twee bronnen door elkaar gemengd binnen één run.
    """
    if api_key:
        try:
            board = api_board if api_board is not None else safe_fetch_oddspapi_toto_board(api_key)

            if isinstance(board, dict) and board.get("_ok") is False:
                fallback = _fetch_toto_week_odds_website(
                    competition,
                    matches,
                    timeout=timeout,
                    max_workers=max_workers,
                )
                fallback["_provider"] = "website-fallback"
                fallback["_structured_error"] = board.get("_error_code") or "STRUCTURED_UNAVAILABLE"
                fallback["_structured_status"] = board.get(
                    "_status", "Complete TOTO odds-feed niet beschikbaar"
                )
                return fallback

            structured = oddspapi_board_for_matches(
                board, competition, matches, api_key
            )
            structured["_provider"] = "structured"
            structured["_structured_status"] = board.get("_status", "")
            return structured

        except Exception as exc:
            # Een fout in de externe feed mag de Streamlit-app nooit laten crashen.
            fallback = _fetch_toto_week_odds_website(
                competition,
                matches,
                timeout=timeout,
                max_workers=max_workers,
            )
            fallback["_provider"] = "website-fallback"
            fallback["_structured_error"] = type(exc).__name__
            fallback["_structured_status"] = (
                f"Complete odds-feed fout ({type(exc).__name__}); website-fallback actief"
            )
            return fallback

    direct = _fetch_toto_week_odds_website(
        competition,
        matches,
        timeout=timeout,
        max_workers=max_workers,
    )
    direct["_provider"] = "website"
    return direct

# =============================================================================
# v1.0.3 — fail-safe OddsPapi integratie
# =============================================================================

def oddspapi_account_status(api_key: str) -> dict:
    """
    Controleer de API-key via /account.
    Volgens OddsPapi is /account unmetered en blijft dit endpoint beschikbaar
    wanneer de normale maandquota al is bereikt.
    """
    if not api_key:
        return {
            "_ok": False,
            "_status": "Geen API key ingesteld",
            "_error_code": "NO_KEY",
        }

    url = f"{ODDSPAPI_BASE}/account"
    try:
        response = requests.get(
            url,
            params={"apiKey": api_key},
            timeout=20,
        )

        if response.status_code in (401, 403):
            return {
                "_ok": False,
                "_status": "OddsPapi API key is ongeldig of heeft geen toegang",
                "_error_code": "AUTH",
                "_http_status": response.status_code,
            }

        if response.status_code != 200:
            message = ""
            code = f"HTTP_{response.status_code}"
            try:
                body = response.json()
                message = (
                    body.get("message")
                    or body.get("details")
                    or body.get("error", {}).get("message")
                    or ""
                )
                code = body.get("code") or body.get("error", {}).get("code") or code
            except Exception:
                pass
            return {
                "_ok": False,
                "_status": f"OddsPapi accountcontrole mislukt ({code})"
                + (f": {message}" if message else ""),
                "_error_code": code,
                "_http_status": response.status_code,
            }

        account = response.json() or {}
        subscriptions = account.get("subscriptions") or []
        active = [s for s in subscriptions if s.get("is_active") is True]
        subscription = active[0] if active else (subscriptions[0] if subscriptions else {})

        request_limit = pd.to_numeric(
            pd.Series([subscription.get("request_limit")]),
            errors="coerce",
        ).iloc[0]
        request_count = pd.to_numeric(
            pd.Series([subscription.get("request_count")]),
            errors="coerce",
        ).iloc[0]

        quota_exhausted = (
            pd.notna(request_limit)
            and pd.notna(request_count)
            and float(request_count) >= float(request_limit)
        )

        sport_ids = subscription.get("sport_ids") or []
        soccer_allowed = not sport_ids or 10 in sport_ids or "10" in sport_ids

        bookmakers = subscription.get("bookmakers") or {}
        # Een lege bookmakers-map behandelen we als "niet expliciet beperkt".
        toto_explicitly_missing = bool(bookmakers) and "toto.nl" not in bookmakers

        if quota_exhausted:
            return {
                "_ok": False,
                "_status": (
                    f"OddsPapi maandlimiet bereikt: "
                    f"{int(request_count)}/{int(request_limit)} requests"
                ),
                "_error_code": "REQUEST_LIMIT_EXCEEDED",
                "_request_count": int(request_count),
                "_request_limit": int(request_limit),
                "_quota_exhausted": True,
                "_account": account,
            }

        if not soccer_allowed:
            return {
                "_ok": False,
                "_status": "Je OddsPapi-abonnement bevat geen voetbal (sportId 10)",
                "_error_code": "SOCCER_NOT_INCLUDED",
                "_request_count": int(request_count) if pd.notna(request_count) else None,
                "_request_limit": int(request_limit) if pd.notna(request_limit) else None,
                "_account": account,
            }

        status = "OddsPapi API key geldig"
        if pd.notna(request_count) and pd.notna(request_limit):
            status += f" · {int(request_count)}/{int(request_limit)} requests gebruikt"

        if toto_explicitly_missing:
            status += " · TOTO NL staat niet expliciet in je abonnementsboekmakers"

        return {
            "_ok": True,
            "_status": status,
            "_error_code": None,
            "_request_count": int(request_count) if pd.notna(request_count) else None,
            "_request_limit": int(request_limit) if pd.notna(request_limit) else None,
            "_quota_exhausted": False,
            "_toto_explicitly_missing": toto_explicitly_missing,
            "_account": account,
        }

    except Exception as exc:
        return {
            "_ok": False,
            "_status": (
                "OddsPapi accountcontrole kon geen verbinding maken "
                f"({type(exc).__name__})"
            ),
            "_error_code": "CONNECTION",
        }


def safe_fetch_oddspapi_toto_board(api_key: str) -> dict:
    """
    Nooit een exception naar Streamlit laten ontsnappen.
    Geeft bij een externe API-fout een statusobject terug zodat de app daarna
    automatisch de publieke TOTO-site kan gebruiken.
    """
    account = oddspapi_account_status(api_key)

    if not account.get("_ok"):
        return {
            "_ok": False,
            "_status": account.get("_status", "OddsPapi niet beschikbaar"),
            "_error_code": account.get("_error_code"),
            "_request_count": account.get("_request_count"),
            "_request_limit": account.get("_request_limit"),
            "_fixtures": [],
        }

    try:
        board = fetch_oddspapi_toto_board(api_key)
        board = dict(board or {})
        board["_ok"] = True
        board["_account_status"] = account.get("_status")
        return board
    except Exception as exc:
        return {
            "_ok": False,
            "_status": (
                "Complete TOTO odds-feed is tijdelijk niet beschikbaar "
                f"({type(exc).__name__}); website-fallback wordt gebruikt"
            ),
            "_error_code": type(exc).__name__,
            "_request_count": account.get("_request_count"),
            "_request_limit": account.get("_request_limit"),
            "_fixtures": [],
        }

# =============================================================================
# v1.0.4 — directe TOTO bron, zonder externe API
# =============================================================================

def _first_market_index(lines: list[str], exact_names) -> Optional[int]:
    wanted = {_toto_norm(x) for x in exact_names}
    for i, line in enumerate(lines):
        if _toto_norm(line) in wanted:
            return i
    return None


def _choice_odd_pairs(lines: list[str], start: int, stop: int):
    """
    Lees label -> eerstvolgende losse decimale odd. Houdt afstand klein zodat
    odds uit een volgende markt niet per ongeluk worden gekoppeld.
    """
    pairs = []
    i = start
    while i < min(stop, len(lines)):
        text = lines[i].strip()
        if re.fullmatch(r"\d{1,3}[,.]\d{1,3}", text):
            i += 1
            continue
        odd = _next_decimal(lines, i, 3)
        if pd.notna(odd):
            pairs.append((text, odd))
        i += 1
    return pairs


def parse_toto_match_odds_direct(
    html: str,
    home_team: str,
    away_team: str,
) -> dict:
    """
    Rechtstreekse parser voor de publieke TOTO wedstrijdpagina.

    Ondersteund:
    - Resultaat / Resultaat - Vroege uitbetaling
    - Dubbele Kans
    - Beide Teams Scoren
    - Aantal Goals - Over/Under 0.5 t/m 5.5

    Alleen prijzen die daadwerkelijk in de HTML staan worden geretourneerd.
    """
    lines = _lines_from_html(html)
    odds = {}

    # --------------------------------------------------------------
    # 1X2: vroege uitbetaling heeft voorkeur; anders gewone Resultaat.
    # --------------------------------------------------------------
    result_idx = None
    for market_name in [
        "Resultaat - Vroege uitbetaling",
        "Resultaat",
    ]:
        idx = _first_market_index(lines, [market_name])
        if idx is not None:
            result_idx = idx
            break

    if result_idx is not None:
        # Pak eerste 3 odds binnen een beperkt venster.
        vals = []
        for j in range(result_idx + 1, min(len(lines), result_idx + 16)):
            raw = lines[j].strip()
            if re.fullmatch(r"\d{1,3}[,.]\d{1,3}", raw):
                val = _decimal(raw)
                if pd.notna(val):
                    vals.append(val)
            if len(vals) >= 3:
                break
        if len(vals) >= 3:
            odds["HOME"], odds["DRAW"], odds["AWAY"] = vals[:3]

    # --------------------------------------------------------------
    # Dubbele kans
    # --------------------------------------------------------------
    dc_idx = _first_market_index(lines, ["Dubbele Kans"])
    if dc_idx is not None:
        end_markers = {
            "beide teams scoren",
            "correct score",
            "aantal goals over under",
            "half time full time",
            "handicap resultaat",
        }
        stop = min(len(lines), dc_idx + 28)
        for j in range(dc_idx + 1, stop):
            if j > dc_idx + 1 and _toto_norm(lines[j]) in end_markers:
                stop = j
                break

        for label, odd in _choice_odd_pairs(lines, dc_idx + 1, stop):
            norm = _toto_norm(label)
            has_home = any(v in norm for v in _team_tokens(home_team))
            has_away = any(v in norm for v in _team_tokens(away_team))
            has_draw = "gelijkspel" in norm

            if has_home and has_draw and not has_away:
                odds["DC_1X"] = odd
            elif has_away and has_draw and not has_home:
                odds["DC_X2"] = odd
            elif has_home and has_away and not has_draw:
                odds["DC_12"] = odd

    # --------------------------------------------------------------
    # BTTS
    # --------------------------------------------------------------
    btts_idx = _first_market_index(lines, ["Beide Teams Scoren"])
    if btts_idx is not None:
        for j in range(btts_idx + 1, min(len(lines), btts_idx + 10)):
            norm = _toto_norm(lines[j])
            if norm == "ja":
                val = _next_decimal(lines, j, 3)
                if pd.notna(val):
                    odds["BTTS_YES"] = val
            elif norm == "nee":
                val = _next_decimal(lines, j, 3)
                if pd.notna(val):
                    odds["BTTS_NO"] = val

    # --------------------------------------------------------------
    # Totaal goals 0.5 t/m 5.5
    # --------------------------------------------------------------
    total_idx = _first_market_index(lines, ["Aantal Goals - Over/Under"])
    if total_idx is not None:
        # Stop bij volgende bekende markt. Groot genoeg om 6 lijnen te bevatten.
        stop_markers = {
            "correct score",
            "half time full time",
            "handicap resultaat",
            "draw no bet",
            "speler scoort",
            "beide teams scoren 2 of meer doelpunten",
        }
        stop = min(len(lines), total_idx + 48)
        for j in range(total_idx + 1, stop):
            if j > total_idx + 1 and _toto_norm(lines[j]) in stop_markers:
                stop = j
                break

        line_re = re.compile(r"^(Over|Under)\s+([0-5][.,]5)$", re.I)
        for j in range(total_idx + 1, stop):
            m = line_re.fullmatch(lines[j].strip())
            if not m:
                continue
            direction = m.group(1).upper()
            line = float(m.group(2).replace(",", "."))
            val = _next_decimal(lines, j, 3)
            if pd.notna(val):
                odds[f"TOTAL_{direction}_{line}"] = val

    count = sum(
        1 for k, v in odds.items()
        if not k.startswith("_") and pd.notna(v)
    )
    odds["_status"] = f"{count} TOTO odds rechtstreeks gelezen"
    odds["_provider"] = "TOTO direct"
    return odds


def fetch_toto_match_odds_direct(
    url: str,
    home_team: str,
    away_team: str,
    timeout: int = 10,
) -> dict:
    if not url:
        return {
            "_status": "Geen TOTO-wedstrijdlink gevonden",
            "_url": None,
            "_provider": "TOTO direct",
        }
    try:
        html = _http_get(url, timeout=timeout)
        odds = parse_toto_match_odds_direct(
            html, home_team, away_team
        )
        odds["_url"] = url
        return odds
    except Exception as exc:
        return {
            "_status": f"TOTO-pagina niet bereikbaar ({type(exc).__name__})",
            "_url": url,
            "_provider": "TOTO direct",
        }


def fetch_toto_week_odds_direct(
    competition: str,
    matches,
    timeout: int = 10,
    max_workers: int = 6,
) -> dict:
    """
    Rechtstreeks TOTO laden:
    1. competitiepagina één keer;
    2. wedstrijdlinks lokaal koppelen;
    3. wedstrijdpagina's parallel ophalen;
    4. géén externe odds-API.
    """
    pairs = [(str(h), str(a)) for h, a in matches]
    index = fetch_toto_competition_index(
        competition, timeout=timeout
    )

    results = {
        "_status": index.get("_status", ""),
        "_source": index.get("_source"),
        "_provider": "TOTO direct",
        "_matches_requested": len(pairs),
        "_matches_linked": 0,
    }

    jobs = []
    for home, away in pairs:
        url = _find_url_in_index(index, home, away)
        if not url:
            url = find_toto_match_url(
                competition,
                home,
                away,
                timeout=timeout,
            )
        if url:
            results["_matches_linked"] += 1
        jobs.append((home, away, url))

    def worker(job):
        home, away, url = job
        return (
            _toto_match_cache_key(home, away),
            fetch_toto_match_odds_direct(
                url, home, away, timeout=timeout
            ),
        )

    workers = max(1, min(int(max_workers), len(jobs) or 1))
    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures = [executor.submit(worker, job) for job in jobs]
        for future in as_completed(futures):
            try:
                key, value = future.result()
                results[key] = value
            except Exception:
                continue

    found = 0
    for key, value in results.items():
        if key.startswith("_") or not isinstance(value, dict):
            continue
        found += sum(
            1 for k, v in value.items()
            if not k.startswith("_") and pd.notna(v)
        )
    results["_odds_found"] = found
    return results
