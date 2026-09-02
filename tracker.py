from __future__ import annotations

import re
import json
from typing import Iterable

import numpy as np
import pandas as pd


BET_COLUMNS = [
    "BetID", "BetType", "LegCount", "FoldLegsJSON", "CreatedAt",
    "Competition", "Round", "WeekKey", "Date", "Match", "HomeTeam",
    "AwayTeam", "Bet", "ModelProb", "FairOdd", "BookmakerOdd", "Stake",
    "Status", "Return", "Profit",
]


def empty_bet_log() -> pd.DataFrame:
    return pd.DataFrame(columns=BET_COLUMNS)


def normalise_bet_log(df: pd.DataFrame) -> pd.DataFrame:
    if df is None or df.empty:
        return empty_bet_log()

    out = df.copy()
    for c in BET_COLUMNS:
        if c not in out.columns:
            out[c] = np.nan

    # Backwards compatibility: older CSVs are singles.
    out["BetType"] = out["BetType"].fillna("Single").astype(str)
    out.loc[
        ~out["BetType"].isin(["Single", "Fold"]),
        "BetType",
    ] = "Single"

    out["LegCount"] = pd.to_numeric(
        out["LegCount"], errors="coerce"
    )
    out.loc[
        (out["BetType"] == "Single") & out["LegCount"].isna(),
        "LegCount",
    ] = 1

    out["FoldLegsJSON"] = out["FoldLegsJSON"].fillna("").astype(str)
    out = out[BET_COLUMNS]

    for c in [
        "Round", "LegCount", "ModelProb", "FairOdd",
        "BookmakerOdd", "Stake", "Return", "Profit",
    ]:
        out[c] = pd.to_numeric(out[c], errors="coerce")

    out["Date"] = pd.to_datetime(out["Date"], errors="coerce")
    out["Status"] = out["Status"].fillna("Open").astype(str)
    return out

def settle_return(status: str, stake: float, odd: float) -> tuple[float, float]:
    stake = float(stake or 0)
    odd = float(odd or 0)
    if status == "Gewonnen":
        ret = stake * odd
    elif status == "Verloren":
        ret = 0.0
    elif status == "Void":
        ret = stake
    else:
        return np.nan, np.nan
    return ret, ret - stake


def recalculate_log(log: pd.DataFrame) -> pd.DataFrame:
    out = normalise_bet_log(log)
    for idx, row in out.iterrows():
        ret, profit = settle_return(row["Status"], row["Stake"], row["BookmakerOdd"])
        out.at[idx, "Return"] = ret
        out.at[idx, "Profit"] = profit
    return out


def event_won(bet: str, home_team: str, away_team: str, hg: int, ag: int):
    bet = str(bet)
    total = hg + ag

    if bet == "Thuiswinst": return hg > ag
    if bet == "Gelijkspel": return hg == ag
    if bet == "Uitwinst": return hg < ag
    if bet.startswith("1X "): return hg >= ag
    if bet.startswith("X2 "): return ag >= hg
    if bet.startswith("12 "): return hg != ag
    if bet == "BTTS - Ja": return hg > 0 and ag > 0
    if bet == "BTTS - Nee": return not (hg > 0 and ag > 0)

    mt = re.fullmatch(r"(Over|Under) ([0-5](?:\.5)) goals", bet)
    if mt:
        line = float(mt.group(2))
        return total > line if mt.group(1) == "Over" else total < line

    for team, goals in [(home_team, hg), (away_team, ag)]:
        if bet.lower().startswith(team.lower() + " "):
            tail = bet[len(team):].strip()
            mm = re.fullmatch(r"(over|under) ([0-5](?:\.5))", tail, re.I)
            if mm:
                line = float(mm.group(2))
                return goals > line if mm.group(1).lower() == "over" else goals < line
    return None


def _load_fold_legs(value) -> list[dict]:
    if value is None:
        return []
    try:
        if pd.isna(value):
            return []
    except Exception:
        pass
    if isinstance(value, list):
        return value
    try:
        parsed = json.loads(str(value))
        return parsed if isinstance(parsed, list) else []
    except Exception:
        return []


def _settle_fold_status(
    row: pd.Series,
    results: pd.DataFrame,
    resolve_team_fn,
):
    """
    Return:
    - 'Gewonnen' zodra alle legs gespeeld en gewonnen zijn;
    - 'Verloren' zodra één gespeelde leg verloren is;
    - 'Open' zolang minstens één leg nog niet gespeeld is.
    """
    legs = _load_fold_legs(row.get("FoldLegsJSON"))
    if not legs:
        return "Open"

    any_pending = False

    for leg in legs:
        comp = str(leg.get("Competition", ""))
        home_display = str(leg.get("HomeTeam", ""))
        away_display = str(leg.get("AwayTeam", ""))
        bet_name = str(leg.get("Bet", ""))

        home = resolve_team_fn(
            results, comp, home_display
        )
        away = resolve_team_fn(
            results, comp, away_display
        )

        matches = results[
            (results["Competition"] == comp)
            & (results["HomeTeam"] == home)
            & (results["AwayTeam"] == away)
        ].copy()

        date_raw = leg.get("Date")
        if date_raw:
            leg_date = pd.to_datetime(
                date_raw, errors="coerce"
            )
            if pd.notna(leg_date):
                same_date = matches[
                    matches["Date"].dt.normalize()
                    == pd.Timestamp(leg_date).normalize()
                ]
                if not same_date.empty:
                    matches = same_date

        if matches.empty:
            any_pending = True
            continue

        result = matches.sort_values("Date").iloc[-1]
        won = event_won(
            bet_name,
            home_display,
            away_display,
            int(result["FTHG"]),
            int(result["FTAG"]),
        )

        if won is False:
            return "Verloren"
        if won is None:
            any_pending = True

    return "Open" if any_pending else "Gewonnen"

def auto_settle_bets(
    log: pd.DataFrame,
    results: pd.DataFrame,
    resolve_team_fn,
) -> pd.DataFrame:
    out = normalise_bet_log(log)
    if out.empty or results.empty:
        return out

    for idx, bet in out.iterrows():
        if bet["Status"] != "Open":
            continue

        if str(bet.get("BetType", "Single")) == "Fold":
            status = _settle_fold_status(
                bet, results, resolve_team_fn
            )
            if status == "Open":
                continue
            ret, profit = settle_return(
                status,
                bet["Stake"],
                bet["BookmakerOdd"],
            )
            out.at[idx, "Status"] = status
            out.at[idx, "Return"] = ret
            out.at[idx, "Profit"] = profit
            continue

        comp = bet["Competition"]
        home = resolve_team_fn(
            results, comp, bet["HomeTeam"]
        )
        away = resolve_team_fn(
            results, comp, bet["AwayTeam"]
        )

        matches = results[
            (results["Competition"] == comp)
            & (results["HomeTeam"] == home)
            & (results["AwayTeam"] == away)
        ].copy()

        if pd.notna(bet["Date"]):
            same_date = matches[
                matches["Date"].dt.normalize()
                == pd.Timestamp(bet["Date"]).normalize()
            ]
            if not same_date.empty:
                matches = same_date

        if matches.empty:
            continue

        result = matches.sort_values("Date").iloc[-1]
        won = event_won(
            bet["Bet"],
            str(bet["HomeTeam"]),
            str(bet["AwayTeam"]),
            int(result["FTHG"]),
            int(result["FTAG"]),
        )
        if won is None:
            continue

        status = "Gewonnen" if won else "Verloren"
        ret, profit = settle_return(
            status,
            bet["Stake"],
            bet["BookmakerOdd"],
        )
        out.at[idx, "Status"] = status
        out.at[idx, "Return"] = ret
        out.at[idx, "Profit"] = profit

    return out

def bankroll_timeline(
    log: pd.DataFrame,
    start_bankroll: float = 50.0,
    secure_pct: float = 0.05,
    exposure_pct: float = 1.0,
) -> pd.DataFrame:
    """
    Bankrollcyclus = ISO-week over alle competities.
    Het ingestelde percentage wordt van het totale uitbetaalde bedrag
    van een volledig afgerekende week veiliggesteld.
    """
    log = recalculate_log(log)
    if log.empty:
        return pd.DataFrame([{
            "WeekKey": "Start", "Status": "Start", "OpeningBankroll": start_bankroll,
            "StakeBudget": start_bankroll * exposure_pct, "Stake": 0.0, "Return": 0.0,
            "Profit": 0.0, "Secured": 0.0, "SecuredTotal": 0.0,
            "ClosingBankroll": start_bankroll, "RemainingBudget": start_bankroll * exposure_pct,
        }])

    active = float(start_bankroll)
    secured_total = 0.0
    rows = []
    weeks = sorted([w for w in log["WeekKey"].dropna().astype(str).unique() if w])

    for week in weeks:
        g = log[log["WeekKey"].astype(str) == week].copy()
        opening = active
        budget = opening * exposure_pct
        stake = float(pd.to_numeric(g["Stake"], errors="coerce").fillna(0).sum())
        all_settled = bool(g["Status"].isin(["Gewonnen", "Verloren", "Void"]).all())

        if all_settled:
            ret = float(pd.to_numeric(g["Return"], errors="coerce").fillna(0).sum())
            profit = ret - stake

            # Nieuwe regel v1.0.1:
            # veiligstellen over de UITBETALING, niet alleen over de winst.
            secured = max(ret, 0.0) * secure_pct

            # Niet-ingezette bankroll blijft staan; resultaat van de bets wordt
            # toegevoegd en het veilige deel wordt uit de actieve bankroll gehaald.
            active = max(opening + profit - secured, 0.0)
            secured_total += secured
            status = "Afgerekend"
        else:
            settled = g[g["Status"].isin(["Gewonnen", "Verloren", "Void"])]
            ret = float(pd.to_numeric(settled["Return"], errors="coerce").fillna(0).sum())
            profit = np.nan
            secured = 0.0
            status = "Open"

        rows.append({
            "WeekKey": week, "Status": status, "OpeningBankroll": opening,
            "StakeBudget": budget, "Stake": stake, "Return": ret, "Profit": profit,
            "Secured": secured, "SecuredTotal": secured_total, "ClosingBankroll": active,
            "RemainingBudget": max(budget - stake, 0.0),
        })
    return pd.DataFrame(rows)


def current_bankroll_summary(
    log,
    start_bankroll,
    secure_pct,
    exposure_pct,
):
    """
    Conservatieve inzetruimte:
    alle nog open singles en folds worden gezien als reeds vastgezet geld.
    Zo kan een gebruiker niet per ongeluk dezelfde bankroll opnieuw inzetten
    doordat open bets in verschillende ISO-weken staan.
    """
    normalised = recalculate_log(log)
    timeline = bankroll_timeline(
        normalised,
        start_bankroll,
        secure_pct,
        exposure_pct,
    )
    last = timeline.iloc[-1]

    active = float(last["ClosingBankroll"])
    secured = float(last["SecuredTotal"])

    open_stake = float(
        pd.to_numeric(
            normalised.loc[
                normalised["Status"] == "Open",
                "Stake",
            ],
            errors="coerce",
        ).fillna(0).sum()
    )

    gross_budget = active * exposure_pct
    available = max(gross_budget - open_stake, 0.0)

    return {
        "active_bankroll": active,
        "secured_total": secured,
        "total_wealth": active + secured,
        "open_stake": open_stake,
        "gross_stake_budget": gross_budget,
        "available_to_stake": available,
    }

def max_drawdown(values: Iterable[float]) -> float:
    arr = np.asarray(list(values), dtype=float)
    if arr.size == 0:
        return 0.0
    peaks = np.maximum.accumulate(arr)
    return float(np.nanmax(peaks - arr))
