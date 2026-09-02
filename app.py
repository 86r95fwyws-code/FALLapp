from __future__ import annotations

import uuid
import json
from datetime import datetime

import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st

from engine import (
    COMPETITIONS,
    BacktestConfig,
    attach_toto_odds,
    build_backtest,
    fetch_toto_week_odds_direct,
    fixture_bet_candidates,
    load_data,
    load_full_season_fixture_catalog,
    next_pending_round,
    performance,
    predict_fixture,
    resolve_catalog_team_name,
    score_matrix,
    toto_week_result_for_match,
)
from tracker import (
    auto_settle_bets,
    bankroll_timeline,
    current_bankroll_summary,
    empty_bet_log,
    max_drawdown,
    normalise_bet_log,
    recalculate_log,
)


SEASON = "2026/27"
ALL_COMPETITIONS = list(COMPETITIONS.keys())

st.set_page_config(page_title="Current Season Betting Lab", page_icon="⚽", layout="wide")
st.title("Current Season Betting Lab")
st.caption(
    "Alle kansen en backtests gebruiken uitsluitend wedstrijden uit 2026/27 die vóór de betreffende wedstrijd gespeeld waren."
)


def pct_nl(value, decimals=2):
    if pd.isna(value):
        return "–"
    return f"{float(value) * 100:.{decimals}f}%".replace(".", ",")



def most_likely_score_local(
    data,
    competition,
    fixture_date,
    home_team,
    away_team,
    mode="Huidig seizoen",
    n_matches=10,
    pseudo=2,
    max_goals=8,
):
    pred = predict_fixture(
        data=data,
        competition=competition,
        fixture_date=fixture_date,
        home_team=home_team,
        away_team=away_team,
        mode=mode,
        n_matches=n_matches,
        pseudo=pseudo,
        max_goals=max_goals,
    )
    matrix = score_matrix(
        float(pred["lambda_home"]),
        float(pred["lambda_away"]),
        max_goals,
    )
    flat_index = int(np.argmax(matrix))
    home_goals, away_goals = np.unravel_index(flat_index, matrix.shape)
    return {
        "HomeGoals": int(home_goals),
        "AwayGoals": int(away_goals),
        "Probability": float(matrix[home_goals, away_goals]),
        "lambda_home": float(pred["lambda_home"]),
        "lambda_away": float(pred["lambda_away"]),
    }


def eur(value):
    if pd.isna(value):
        return "–"
    return f"€{float(value):,.2f}".replace(",", "X").replace(".", ",").replace("X", ".")


@st.cache_data(ttl=900, show_spinner=False)
def cached_results():
    return load_data(ALL_COMPETITIONS, [SEASON])


@st.cache_data(ttl=21600, show_spinner=False)
def cached_schedule():
    return load_full_season_fixture_catalog(ALL_COMPETITIONS)


@st.cache_data(ttl=300, show_spinner=False)
def cached_toto_week(competition, matches):
    return fetch_toto_week_odds_direct(
        competition,
        matches,
        timeout=10,
        max_workers=6,
    )


# Data verversen
rc1, rc2 = st.sidebar.columns(2)
if rc1.button("↻ Data", use_container_width=True):
    cached_results.clear()
    st.rerun()
if rc2.button("↻ TOTO", use_container_width=True):
    cached_toto_week.clear()
    st.rerun()

with st.spinner("Actuele 2026/27-data laden..."):
    results = cached_results()
    schedule = cached_schedule()

if results.empty:
    st.error("De 2026/27-resultaten konden niet worden geladen. Probeer Data verversen.")
    st.stop()
if schedule.empty:
    st.error("Het volledige 2026/27-programma kon niet worden geladen.")
    st.stop()

# Globale instellingen
st.sidebar.markdown("### TOTO odds")
st.sidebar.success("Rechtstreeks TOTO actief")
st.sidebar.caption(
    "Geen API-key nodig. De app leest alleen odds die daadwerkelijk "
    "op de publieke TOTO-wedstrijdpagina staan."
)

st.sidebar.markdown("### Model")
pseudo = st.sidebar.select_slider(
    "Pseudo-wedstrijden",
    options=list(range(0, 9)),
    value=2,
    help="Meer pseudo-wedstrijden trekken vroege seizoenscijfers sterker richting het huidige competitiegemiddelde.",
)
threshold_sidebar = st.sidebar.slider("Standaard kansgrens (%)", 50, 99, 85, 1)

st.sidebar.markdown("### Bankroll")
start_bankroll = st.sidebar.number_input("Startbankroll", min_value=1.0, value=50.0, step=5.0)
secure_pct_input = st.sidebar.number_input(
    "% van uitbetaald bedrag veiligstellen", min_value=0.0, max_value=100.0, value=5.0, step=1.0
)
secure_pct = secure_pct_input / 100
exposure_pct_input = st.sidebar.slider("Max. bankroll inzetten per week (%)", 10, 100, 100, 5)
exposure_pct = exposure_pct_input / 100

if "bet_log" not in st.session_state:
    st.session_state.bet_log = empty_bet_log()

if "fold_legs" not in st.session_state:
    st.session_state.fold_legs = []

# Bekende bets automatisch afrekenen wanneer Football-Data de uitslag inmiddels bevat.
st.session_state.bet_log = auto_settle_bets(
    st.session_state.bet_log, results, resolve_catalog_team_name
)


tabs = st.tabs(["🏠 Start", "🧾 Mijn bets", "🧪 Backtest", "📊 Dashboard"])


# =============================================================================
# START
# =============================================================================
with tabs[0]:
    c1, c2, c3 = st.columns([1.35, 1, 1])
    competition = c1.selectbox("Competitie", ALL_COMPETITIONS, key="start_comp")
    comp_schedule = schedule[schedule["Competition"] == competition].copy()
    rounds = sorted(
        comp_schedule["Round"].dropna().astype(int).unique().tolist()
    )

    default_round = next_pending_round(
        schedule,
        results,
        competition,
    )
    round_key = f"start_round_{competition}"
    selected_round = c2.selectbox(
        "Speelronde",
        rounds,
        index=(
            rounds.index(default_round)
            if default_round in rounds
            else 0
        ),
        key=round_key,
    )
    threshold_pct = c3.number_input(
        "Min. modelkans (%)", min_value=50, max_value=99, value=int(threshold_sidebar), step=1
    )
    threshold = threshold_pct / 100

    v1, v2 = st.columns([1.1, 1])
    only_value = v1.toggle("Alleen TOTO-value bets", value=False)
    min_value_pct = v2.number_input(
        "Min. TOTO-value (%)", min_value=-20.0, max_value=100.0, value=3.0, step=0.5,
        disabled=not only_value,
    )
    min_value = min_value_pct / 100

    round_df = comp_schedule[comp_schedule["Round"] == selected_round].sort_values("DateTime")
    st.markdown(f"### {competition} · speelronde {selected_round}")
    st.caption(
        f"Pseudo {pseudo} · alleen {SEASON} · goal-lijnen 0.5 t/m 5.5 · modelgrens {threshold_pct}%."
    )

    pairs = tuple((str(r.HomeTeam), str(r.AwayTeam)) for r in round_df.itertuples())
    with st.spinner("TOTO odds voor deze ronde ophalen..."):
        toto_week = cached_toto_week(competition, pairs)

    results_by_match = {}
    score_predictions = {}
    all_bets = []
    summaries = []

    for fixture in round_df.itertuples():
        cands = fixture_bet_candidates(
            results,
            competition=competition,
            fixture_date=fixture.Date,
            home_team=fixture.HomeTeam,
            away_team=fixture.AwayTeam,
            mode="Huidig seizoen",
            n_matches=10,
            pseudo=pseudo,
            max_goals=8,
            threshold=threshold,
            market_scope="0.5-5.5",
        )
        toto = toto_week_result_for_match(toto_week, fixture.HomeTeam, fixture.AwayTeam)
        cands = attach_toto_odds(cands, toto)
        if only_value and not cands.empty:
            cands = cands[
                cands["TotoOdd"].notna()
                & cands["ValuePct"].notna()
                & (cands["ValuePct"] >= min_value)
            ].copy()

        # Gebruikersvoorkeur: laagste kwalificerende kans bovenaan.
        if not cands.empty:
            cands = cands.sort_values(
                ["ModelProb", "Category", "Bet"],
                ascending=[True, True, True],
                kind="stable",
            ).reset_index(drop=True)

        score_pred = most_likely_score_local(
            data=results,
            competition=competition,
            fixture_date=fixture.Date,
            home_team=fixture.HomeTeam,
            away_team=fixture.AwayTeam,
            mode="Huidig seizoen",
            pseudo=pseudo,
            max_goals=8,
        )

        results_by_match[fixture.Match] = cands
        score_predictions[fixture.Match] = score_pred
        all_bets.append(cands)
        summaries.append({"Match": fixture.Match, "Bets": len(cands)})

    s1, s2, s3, s4 = st.columns(4)
    s1.metric("Wedstrijden", len(round_df))
    s2.metric("Geselecteerde bets", int(sum(x["Bets"] for x in summaries)))
    s3.metric("Pseudo", pseudo)
    linked = int(toto_week.get("_matches_linked", 0))
    s4.metric("TOTO matches gekoppeld", f"{linked}/{len(round_df)}")
    st.caption(
        "TOTO bron: rechtstreeks sport.toto.nl. "
        "Alleen daadwerkelijk gevonden prijzen worden getoond."
    )

    team_market_matches = 0
    for fixture in round_df.itertuples():
        toto_info = toto_week_result_for_match(
            toto_week, fixture.HomeTeam, fixture.AwayTeam
        )
        if int(toto_info.get("_team_totals_found", 0) or 0) > 0:
            team_market_matches += 1

    if team_market_matches:
        st.success(
            f"Teamgoal TOTO-markten gevonden bij {team_market_matches}/"
            f"{len(round_df)} wedstrijden."
        )
    else:
        st.info(
            "Voor deze speelronde levert de publieke TOTO-wedstrijdpagina geen "
            "aparte teamgoal Over/Under-markten aan. Het model berekent ze wel, "
            "maar ik vul daarvoor geen geschatte TOTO-odd in."
        )

    for fixture in round_df.itertuples():
        cands = results_by_match[fixture.Match]
        title = (
            f"{pd.Timestamp(fixture.Date).strftime('%d-%m')} · {fixture.Match} · "
            f"{len(cands)} {'bet' if len(cands)==1 else 'bets'}"
        )
        with st.expander(title, expanded=len(cands) > 0):
            score_pred = score_predictions.get(fixture.Match)
            if score_pred:
                st.caption("Meest waarschijnlijke exacte uitslag")
                sc1, sc2 = st.columns([3, 1])
                sc1.markdown(
                    f"### {fixture.HomeTeam} "
                    f"{score_pred['HomeGoals']} – {score_pred['AwayGoals']} "
                    f"{fixture.AwayTeam}"
                )
                sc2.metric(
                    "Kans",
                    pct_nl(score_pred["Probability"]),
                )
                st.progress(
                    min(max(float(score_pred["Probability"]), 0.0), 1.0),
                    text=(
                        f"Exacte scorekans: "
                        f"{pct_nl(score_pred['Probability'])}"
                    ),
                )
                st.caption(
                    f"Verwachte goals: "
                    f"{score_pred['lambda_home']:.2f} – "
                    f"{score_pred['lambda_away']:.2f}"
                )

            if cands.empty:
                st.caption("Geen bets voldoen aan de ingestelde filters.")
            else:
                d = cands[["Category", "Bet", "ModelProb", "FairOdd", "TotoOdd", "ValuePct"]].copy()
                d["Kans"] = d["ModelProb"].apply(pct_nl)
                d["Value"] = d["ValuePct"].apply(pct_nl)
                d = d[["Category", "Bet", "Kans", "FairOdd", "TotoOdd", "Value"]]
                st.dataframe(
                    d,
                    use_container_width=True,
                    hide_index=True,
                    column_config={
                        "Category": "Markt",
                        "Bet": "Bet",
                        "Kans": "Kans",
                        "FairOdd": st.column_config.NumberColumn("Fair odd", format="%.2f"),
                        "TotoOdd": st.column_config.NumberColumn("TOTO odd", format="%.2f"),
                        "Value": "Value",
                    },
                )

                missing_team = cands[
                    cands["Category"].isin(
                        ["Teamgoals thuis", "Teamgoals uit"]
                    )
                    & cands["TotoOdd"].isna()
                ].copy()

                if not missing_team.empty:
                    with st.expander(
                        "Teamgoal TOTO-odds handmatig aanvullen",
                        expanded=False,
                    ):
                        st.caption(
                            "Gebruik dit alleen wanneer je de exacte TOTO-odd zelf "
                            "in TOTO ziet. De app verzint hier geen bookmakerprijs."
                        )
                        manual_view = missing_team[
                            ["Bet", "ModelProb", "FairOdd"]
                        ].copy()
                        manual_view["Kans"] = manual_view["ModelProb"].apply(
                            pct_nl
                        )
                        manual_view["TOTO odd"] = 0.0
                        manual_view = manual_view[
                            ["Bet", "Kans", "FairOdd", "TOTO odd"]
                        ]
                        st.data_editor(
                            manual_view,
                            use_container_width=True,
                            hide_index=True,
                            disabled=["Bet", "Kans", "FairOdd"],
                            key=f"team_toto_manual_{competition}_{selected_round}_{fixture.Match}",
                            column_config={
                                "FairOdd": st.column_config.NumberColumn(
                                    "Fair odd", format="%.2f"
                                ),
                                "TOTO odd": st.column_config.NumberColumn(
                                    "TOTO odd",
                                    min_value=0.0,
                                    step=0.01,
                                    format="%.2f",
                                ),
                            },
                        )

    nonempty = [x for x in all_bets if not x.empty]
    if nonempty:
        combined = pd.concat(nonempty, ignore_index=True)
        combined = combined.sort_values(
            ["ModelProb", "Match", "Category", "Bet"],
            ascending=[True, True, True, True],
            kind="stable",
        ).reset_index(drop=True)
        st.divider()
        st.markdown("#### Alle geselecteerde bets van deze ronde")
        d = combined[["Match", "Category", "Bet", "ModelProb", "FairOdd", "TotoOdd", "ValuePct"]].copy()
        d["Kans"] = d["ModelProb"].apply(pct_nl)
        d["Value"] = d["ValuePct"].apply(pct_nl)
        d = d[["Match", "Category", "Bet", "Kans", "FairOdd", "TotoOdd", "Value"]]
        st.dataframe(
            d,
            use_container_width=True,
            hide_index=True,
            column_config={
                "FairOdd": st.column_config.NumberColumn("Fair odd", format="%.2f"),
                "TotoOdd": st.column_config.NumberColumn("TOTO odd", format="%.2f"),
            },
        )


# =============================================================================
# MIJN BETS
# =============================================================================
with tabs[1]:
    st.subheader("Mijn bets")

    st.info(
        "Je kunt hier losse bets én folds/combi's opslaan. "
        "Open inzet wordt direct van je beschikbare bankroll afgetrokken."
    )

    upload = st.file_uploader(
        "Importeer betlog-backup",
        type=["csv"],
        key="betlog_upload",
    )
    if upload is not None:
        try:
            st.session_state.bet_log = normalise_bet_log(
                pd.read_csv(upload)
            )
            st.success("Betlog geïmporteerd.")
        except Exception as exc:
            st.error(f"Importeren mislukt: {exc}")

    # Altijd bovenaan zichtbaar: bankroll na singles + folds.
    bet_summary = current_bankroll_summary(
        st.session_state.bet_log,
        start_bankroll,
        secure_pct,
        exposure_pct,
    )
    bm1, bm2, bm3, bm4 = st.columns(4)
    bm1.metric(
        "Actieve bankroll",
        eur(bet_summary["active_bankroll"]),
    )
    bm2.metric(
        "Open inzet",
        eur(bet_summary.get("open_stake", 0.0)),
    )
    bm3.metric(
        "Max. inzetbudget",
        eur(bet_summary.get("gross_stake_budget", 0.0)),
    )
    bm4.metric(
        "Nog inzetbaar",
        eur(bet_summary["available_to_stake"]),
    )

    bet_mode = st.radio(
        "Wat wil je toevoegen?",
        ["Single bet", "Fold / combi"],
        horizontal=True,
        key="bet_mode",
    )

    # -----------------------------------------------------------------
    # SINGLE BET
    # -----------------------------------------------------------------
    if bet_mode == "Single bet":
        st.markdown("### Single bet")

        b1, b2 = st.columns(2)
        bet_comp = b1.selectbox(
            "Competitie",
            ALL_COMPETITIONS,
            key="bet_comp",
        )

        bs = schedule[
            schedule["Competition"] == bet_comp
        ].copy()
        bet_rounds = sorted(
            bs["Round"].dropna().astype(int).unique().tolist()
        )
        default_bet_round = next_pending_round(
            schedule,
            results,
            bet_comp,
        )

        bet_round = b2.selectbox(
            "Speelronde",
            bet_rounds,
            index=(
                bet_rounds.index(default_bet_round)
                if default_bet_round in bet_rounds
                else 0
            ),
            key=f"bet_round_{bet_comp}",
        )

        fixtures = bs[
            bs["Round"] == bet_round
        ].sort_values("DateTime")

        if fixtures.empty:
            st.warning("Geen wedstrijden in deze speelronde.")
        else:
            selected_match = st.selectbox(
                "Wedstrijd",
                fixtures["Match"].tolist(),
                key=f"bet_match_{bet_comp}_{bet_round}",
            )
            fixture = fixtures[
                fixtures["Match"] == selected_match
            ].iloc[0]

            choices = fixture_bet_candidates(
                results,
                competition=bet_comp,
                fixture_date=fixture["Date"],
                home_team=fixture["HomeTeam"],
                away_team=fixture["AwayTeam"],
                mode="Huidig seizoen",
                pseudo=pseudo,
                threshold=0.50,
                max_goals=8,
                market_scope="0.5-5.5",
            )

            if choices.empty:
                st.warning(
                    "Geen modelmarkten beschikbaar voor deze wedstrijd."
                )
            else:
                choices = choices.sort_values(
                    ["ModelProb", "Category", "Bet"],
                    ascending=[True, True, True],
                    kind="stable",
                ).reset_index(drop=True)

                single_toto = cached_toto_week(
                    bet_comp,
                    ((fixture["HomeTeam"], fixture["AwayTeam"]),),
                )
                toto = toto_week_result_for_match(
                    single_toto,
                    fixture["HomeTeam"],
                    fixture["AwayTeam"],
                )
                choices = attach_toto_odds(
                    choices,
                    toto,
                )

                labels = []
                for row in choices.itertuples():
                    toto_text = (
                        f" · TOTO {row.TotoOdd:.2f}"
                        if pd.notna(row.TotoOdd)
                        else ""
                    )
                    labels.append(
                        f"{row.Bet} · {pct_nl(row.ModelProb)}"
                        f" · fair {row.FairOdd:.2f}{toto_text}"
                    )

                choice_label = st.selectbox(
                    "Bet",
                    labels,
                    key=f"single_choice_{bet_comp}_{bet_round}_{selected_match}",
                )
                selected = choices.iloc[
                    labels.index(choice_label)
                ]

                toto_odd = pd.to_numeric(
                    pd.Series([selected.get("TotoOdd")]),
                    errors="coerce",
                ).iloc[0]
                default_odd = (
                    float(toto_odd)
                    if pd.notna(toto_odd) and toto_odd > 1
                    else float(selected["FairOdd"])
                )

                m1, m2, m3 = st.columns(3)
                m1.metric(
                    "Nog inzetbaar",
                    eur(bet_summary["available_to_stake"]),
                )
                m2.metric(
                    "Modelkans",
                    pct_nl(selected["ModelProb"]),
                )
                m3.metric(
                    "Fair odd",
                    f"{selected['FairOdd']:.2f}",
                )

                i1, i2 = st.columns(2)
                bookmaker_odd = i1.number_input(
                    "Bookmaker odd",
                    min_value=1.01,
                    max_value=1000.0,
                    value=max(
                        1.01,
                        round(default_odd, 2),
                    ),
                    step=0.01,
                    key=f"single_odd_{bet_comp}_{bet_round}_{selected_match}",
                )

                available = float(
                    bet_summary["available_to_stake"]
                )
                if available <= 0:
                    st.error(
                        "Je hebt momenteel geen inzetruimte meer in je bankroll."
                    )
                    stake = 0.0
                else:
                    stake = i2.number_input(
                        "Inzet",
                        min_value=0.01,
                        max_value=max(0.01, available),
                        value=min(5.0, available),
                        step=0.50,
                        key=f"single_stake_{bet_comp}_{bet_round}_{selected_match}",
                    )

                model_value = (
                    float(selected["ModelProb"])
                    * bookmaker_odd
                    - 1
                )
                iso = pd.Timestamp(
                    fixture["Date"]
                ).isocalendar()
                week_key = (
                    f"{iso.year}-W{int(iso.week):02d}"
                )

                st.caption(
                    f"Model-value: {pct_nl(model_value)} · "
                    f"bankrollweek {week_key}."
                )

                if st.button(
                    "＋ Single toevoegen",
                    type="primary",
                    disabled=available <= 0,
                    key="add_single",
                ):
                    new = pd.DataFrame([{
                        "BetID": str(uuid.uuid4())[:8],
                        "BetType": "Single",
                        "LegCount": 1,
                        "FoldLegsJSON": "",
                        "CreatedAt": datetime.now().isoformat(
                            timespec="seconds"
                        ),
                        "Competition": bet_comp,
                        "Round": int(bet_round),
                        "WeekKey": week_key,
                        "Date": fixture["Date"],
                        "Match": fixture["Match"],
                        "HomeTeam": fixture["HomeTeam"],
                        "AwayTeam": fixture["AwayTeam"],
                        "Bet": selected["Bet"],
                        "ModelProb": float(
                            selected["ModelProb"]
                        ),
                        "FairOdd": float(
                            selected["FairOdd"]
                        ),
                        "BookmakerOdd": bookmaker_odd,
                        "Stake": stake,
                        "Status": "Open",
                        "Return": np.nan,
                        "Profit": np.nan,
                    }])

                    st.session_state.bet_log = pd.concat(
                        [
                            normalise_bet_log(
                                st.session_state.bet_log
                            ),
                            normalise_bet_log(new),
                        ],
                        ignore_index=True,
                    )
                    st.success("Single toegevoegd.")
                    st.rerun()

    # -----------------------------------------------------------------
    # FOLD / COMBI
    # -----------------------------------------------------------------
    else:
        st.markdown("### Fold / combi")
        st.caption(
            "Combineer bets uit verschillende wedstrijden en competities. "
            "De totale bookmakerodd wordt vermenigvuldigd. "
            "De gecombineerde modelkans is een benadering door de kansen "
            "van verschillende wedstrijden te vermenigvuldigen."
        )

        f1, f2 = st.columns(2)
        fold_comp = f1.selectbox(
            "Competitie voor nieuwe leg",
            ALL_COMPETITIONS,
            key="fold_comp",
        )
        fs = schedule[
            schedule["Competition"] == fold_comp
        ].copy()
        fold_rounds = sorted(
            fs["Round"].dropna().astype(int).unique().tolist()
        )
        default_fold_round = next_pending_round(
            schedule,
            results,
            fold_comp,
        )
        fold_round = f2.selectbox(
            "Speelronde",
            fold_rounds,
            index=(
                fold_rounds.index(default_fold_round)
                if default_fold_round in fold_rounds
                else 0
            ),
            key=f"fold_round_{fold_comp}",
        )

        fold_fixtures = fs[
            fs["Round"] == fold_round
        ].sort_values("DateTime")

        if fold_fixtures.empty:
            st.warning("Geen wedstrijden in deze ronde.")
        else:
            fold_match = st.selectbox(
                "Wedstrijd voor nieuwe leg",
                fold_fixtures["Match"].tolist(),
                key=f"fold_match_{fold_comp}_{fold_round}",
            )
            fold_fixture = fold_fixtures[
                fold_fixtures["Match"] == fold_match
            ].iloc[0]

            fold_choices = fixture_bet_candidates(
                results,
                competition=fold_comp,
                fixture_date=fold_fixture["Date"],
                home_team=fold_fixture["HomeTeam"],
                away_team=fold_fixture["AwayTeam"],
                mode="Huidig seizoen",
                pseudo=pseudo,
                threshold=0.50,
                max_goals=8,
                market_scope="0.5-5.5",
            )

            if not fold_choices.empty:
                fold_choices = fold_choices.sort_values(
                    ["ModelProb", "Category", "Bet"],
                    ascending=[True, True, True],
                    kind="stable",
                ).reset_index(drop=True)

                fold_toto_week = cached_toto_week(
                    fold_comp,
                    ((
                        fold_fixture["HomeTeam"],
                        fold_fixture["AwayTeam"],
                    ),),
                )
                fold_toto = toto_week_result_for_match(
                    fold_toto_week,
                    fold_fixture["HomeTeam"],
                    fold_fixture["AwayTeam"],
                )
                fold_choices = attach_toto_odds(
                    fold_choices,
                    fold_toto,
                )

                fold_labels = []
                for row in fold_choices.itertuples():
                    toto_text = (
                        f" · TOTO {row.TotoOdd:.2f}"
                        if pd.notna(row.TotoOdd)
                        else ""
                    )
                    fold_labels.append(
                        f"{row.Bet} · {pct_nl(row.ModelProb)}"
                        f" · fair {row.FairOdd:.2f}{toto_text}"
                    )

                fold_choice_label = st.selectbox(
                    "Bet voor nieuwe leg",
                    fold_labels,
                    key=(
                        f"fold_choice_{fold_comp}_"
                        f"{fold_round}_{fold_match}"
                    ),
                )
                fold_selected = fold_choices.iloc[
                    fold_labels.index(
                        fold_choice_label
                    )
                ]

                fold_toto_odd = pd.to_numeric(
                    pd.Series([
                        fold_selected.get("TotoOdd")
                    ]),
                    errors="coerce",
                ).iloc[0]
                fold_default_odd = (
                    float(fold_toto_odd)
                    if pd.notna(fold_toto_odd)
                    and fold_toto_odd > 1
                    else float(
                        fold_selected["FairOdd"]
                    )
                )

                leg_odd = st.number_input(
                    "Odd van deze leg",
                    min_value=1.01,
                    max_value=1000.0,
                    value=max(
                        1.01,
                        round(fold_default_odd, 2),
                    ),
                    step=0.01,
                    key=(
                        f"fold_leg_odd_{fold_comp}_"
                        f"{fold_round}_{fold_match}"
                    ),
                )

                existing_matches = {
                    str(leg.get("Match"))
                    for leg in st.session_state.fold_legs
                }
                duplicate_match = (
                    str(fold_fixture["Match"])
                    in existing_matches
                )

                if duplicate_match:
                    st.warning(
                        "Deze wedstrijd zit al in je fold. "
                        "Een fold gebruikt hier maximaal één leg per wedstrijd."
                    )

                if st.button(
                    "＋ Leg toevoegen",
                    disabled=duplicate_match,
                    key="add_fold_leg",
                ):
                    iso = pd.Timestamp(
                        fold_fixture["Date"]
                    ).isocalendar()
                    leg_week = (
                        f"{iso.year}-W{int(iso.week):02d}"
                    )

                    st.session_state.fold_legs.append({
                        "LegID": str(uuid.uuid4())[:8],
                        "Competition": fold_comp,
                        "Round": int(fold_round),
                        "WeekKey": leg_week,
                        "Date": pd.Timestamp(
                            fold_fixture["Date"]
                        ).strftime("%Y-%m-%d"),
                        "Match": str(
                            fold_fixture["Match"]
                        ),
                        "HomeTeam": str(
                            fold_fixture["HomeTeam"]
                        ),
                        "AwayTeam": str(
                            fold_fixture["AwayTeam"]
                        ),
                        "Bet": str(
                            fold_selected["Bet"]
                        ),
                        "ModelProb": float(
                            fold_selected["ModelProb"]
                        ),
                        "FairOdd": float(
                            fold_selected["FairOdd"]
                        ),
                        "BookmakerOdd": float(
                            leg_odd
                        ),
                    })
                    st.rerun()
            else:
                st.warning(
                    "Geen modelmarkten beschikbaar voor deze wedstrijd."
                )

        st.divider()
        st.markdown("#### Huidige fold")

        legs = list(st.session_state.fold_legs)
        if not legs:
            st.caption(
                "Voeg minimaal twee legs uit verschillende wedstrijden toe."
            )
        else:
            leg_df = pd.DataFrame(legs)
            leg_df["Kans"] = leg_df[
                "ModelProb"
            ].apply(pct_nl)
            st.dataframe(
                leg_df[[
                    "LegID", "Competition", "Round",
                    "Match", "Bet", "Kans",
                    "BookmakerOdd",
                ]],
                use_container_width=True,
                hide_index=True,
                column_config={
                    "BookmakerOdd":
                        st.column_config.NumberColumn(
                            "Odd", format="%.2f"
                        ),
                },
            )

            remove_ids = st.multiselect(
                "Legs verwijderen",
                [leg["LegID"] for leg in legs],
                format_func=lambda leg_id: next(
                    (
                        f"{leg['Match']} · {leg['Bet']}"
                        for leg in legs
                        if leg["LegID"] == leg_id
                    ),
                    leg_id,
                ),
                key="remove_fold_legs",
            )
            r1, r2 = st.columns(2)
            if remove_ids and r1.button(
                "Verwijder geselecteerde legs",
                key="delete_fold_legs",
            ):
                st.session_state.fold_legs = [
                    leg for leg in legs
                    if leg["LegID"] not in remove_ids
                ]
                st.rerun()

            if r2.button(
                "Wis hele fold",
                key="clear_fold",
            ):
                st.session_state.fold_legs = []
                st.rerun()

            total_odd = float(np.prod([
                float(leg["BookmakerOdd"])
                for leg in legs
            ]))
            combined_prob = float(np.prod([
                float(leg["ModelProb"])
                for leg in legs
            ]))
            combined_fair = (
                1.0 / combined_prob
                if combined_prob > 0
                else np.nan
            )

            fm1, fm2, fm3 = st.columns(3)
            fm1.metric(
                "Totale fold odd",
                f"{total_odd:.2f}",
            )
            fm2.metric(
                "Geschatte modelkans",
                pct_nl(combined_prob),
            )
            fm3.metric(
                "Fair odd gecombineerd",
                (
                    f"{combined_fair:.2f}"
                    if pd.notna(combined_fair)
                    else "–"
                ),
            )

            fold_available = float(
                bet_summary["available_to_stake"]
            )
            if fold_available <= 0:
                st.error(
                    "Je hebt geen inzetruimte meer."
                )
                fold_stake = 0.0
            else:
                fold_stake = st.number_input(
                    "Inzet op deze fold",
                    min_value=0.01,
                    max_value=max(
                        0.01,
                        fold_available,
                    ),
                    value=min(
                        5.0,
                        fold_available,
                    ),
                    step=0.50,
                    key="fold_stake",
                )

            possible_return = (
                fold_stake * total_odd
                if fold_stake > 0
                else 0.0
            )
            st.metric(
                "Mogelijke uitbetaling",
                eur(possible_return),
            )

            leg_weeks = sorted({
                str(leg["WeekKey"])
                for leg in legs
            })
            if len(leg_weeks) > 1:
                st.warning(
                    "Deze fold bevat wedstrijden uit meerdere ISO-weken. "
                    "De inzet wordt voor je bankroll toegewezen aan de week "
                    "van de vroegste leg en blijft open tot de fold beslist is."
                )

            if st.button(
                "＋ Fold toevoegen",
                type="primary",
                disabled=(
                    len(legs) < 2
                    or fold_available <= 0
                ),
                key="save_fold",
            ):
                sorted_legs = sorted(
                    legs,
                    key=lambda x: x["Date"],
                )
                first_leg = sorted_legs[0]

                new_fold = pd.DataFrame([{
                    "BetID": str(uuid.uuid4())[:8],
                    "BetType": "Fold",
                    "LegCount": len(legs),
                    "FoldLegsJSON": json.dumps(
                        legs,
                        ensure_ascii=False,
                    ),
                    "CreatedAt": datetime.now().isoformat(
                        timespec="seconds"
                    ),
                    "Competition": "Fold",
                    "Round": np.nan,
                    "WeekKey": first_leg["WeekKey"],
                    "Date": first_leg["Date"],
                    "Match": f"{len(legs)} legs",
                    "HomeTeam": "",
                    "AwayTeam": "",
                    "Bet": "Fold / combi",
                    "ModelProb": combined_prob,
                    "FairOdd": combined_fair,
                    "BookmakerOdd": total_odd,
                    "Stake": fold_stake,
                    "Status": "Open",
                    "Return": np.nan,
                    "Profit": np.nan,
                }])

                st.session_state.bet_log = pd.concat(
                    [
                        normalise_bet_log(
                            st.session_state.bet_log
                        ),
                        normalise_bet_log(
                            new_fold
                        ),
                    ],
                    ignore_index=True,
                )
                st.session_state.fold_legs = []
                st.success(
                    "Fold toegevoegd. Je resterende bankroll is bijgewerkt."
                )
                st.rerun()

    # -----------------------------------------------------------------
    # OPGESLAGEN BETS / FOLDS
    # -----------------------------------------------------------------
    st.divider()
    log = recalculate_log(
        st.session_state.bet_log
    )
    st.session_state.bet_log = log

    if log.empty:
        st.caption("Nog geen bets opgeslagen.")
    else:
        f1, f2, f3, f4 = st.columns(4)

        ftype = f1.selectbox(
            "Type",
            ["Alle", "Single", "Fold"],
            key="log_type_filter",
        )
        comp_options = (
            ["Alle", "Fold"] + ALL_COMPETITIONS
        )
        fcomp = f2.selectbox(
            "Competitie",
            comp_options,
            key="log_comp_filter",
        )
        round_values = sorted(
            log["Round"].dropna().astype(int).unique().tolist()
        )
        fround = f3.selectbox(
            "Ronde",
            ["Alle"] + round_values,
            key="log_round_filter",
        )
        fstatus = f4.selectbox(
            "Status",
            ["Alle", "Open", "Gewonnen", "Verloren", "Void"],
            key="log_status_filter",
        )

        filtered = log.copy()
        if ftype != "Alle":
            filtered = filtered[
                filtered["BetType"] == ftype
            ]
        if fcomp != "Alle":
            filtered = filtered[
                filtered["Competition"] == fcomp
            ]
        if fround != "Alle":
            filtered = filtered[
                filtered["Round"] == int(fround)
            ]
        if fstatus != "Alle":
            filtered = filtered[
                filtered["Status"] == fstatus
            ]

        editor = filtered.copy()
        editor["Kans"] = editor[
            "ModelProb"
        ].apply(pct_nl)
        editor = editor[[
            "BetID", "BetType", "LegCount",
            "Competition", "Round", "WeekKey",
            "Match", "Bet", "Kans", "FairOdd",
            "BookmakerOdd", "Stake", "Status",
            "Return", "Profit",
        ]]

        edited = st.data_editor(
            editor,
            use_container_width=True,
            hide_index=True,
            disabled=[
                "BetID", "BetType", "LegCount",
                "Competition", "Round", "WeekKey",
                "Match", "Bet", "Kans", "FairOdd",
                "Return", "Profit",
            ],
            column_config={
                "LegCount":
                    st.column_config.NumberColumn(
                        "Legs", format="%d"
                    ),
                "Status":
                    st.column_config.SelectboxColumn(
                        "Status",
                        options=[
                            "Open",
                            "Gewonnen",
                            "Verloren",
                            "Void",
                        ],
                        required=True,
                    ),
                "BookmakerOdd":
                    st.column_config.NumberColumn(
                        "Odd", format="%.2f"
                    ),
                "Stake":
                    st.column_config.NumberColumn(
                        "Inzet", format="€%.2f"
                    ),
                "Return":
                    st.column_config.NumberColumn(
                        "Uitbetaling", format="€%.2f"
                    ),
                "Profit":
                    st.column_config.NumberColumn(
                        "Winst/verlies", format="€%.2f"
                    ),
            },
            key="bet_log_editor",
        )

        if st.button(
            "Wijzigingen opslaan",
            key="save_log_changes",
        ):
            base = st.session_state.bet_log.copy()
            for _, erow in edited.iterrows():
                mask = (
                    base["BetID"].astype(str)
                    == str(erow["BetID"])
                )
                base.loc[
                    mask, "BookmakerOdd"
                ] = erow["BookmakerOdd"]
                base.loc[
                    mask, "Stake"
                ] = erow["Stake"]
                base.loc[
                    mask, "Status"
                ] = erow["Status"]

            st.session_state.bet_log = (
                recalculate_log(base)
            )
            st.rerun()

        # Fold details.
        folds = filtered[
            filtered["BetType"] == "Fold"
        ]
        if not folds.empty:
            st.markdown("#### Fold details")
            for _, fold in folds.iterrows():
                with st.expander(
                    f"{fold['BetID']} · "
                    f"{int(fold['LegCount']) if pd.notna(fold['LegCount']) else 0} legs · "
                    f"odd {fold['BookmakerOdd']:.2f} · "
                    f"{fold['Status']}"
                ):
                    try:
                        legs = json.loads(
                            str(fold["FoldLegsJSON"])
                        )
                    except Exception:
                        legs = []

                    if legs:
                        detail = pd.DataFrame(legs)
                        detail["Kans"] = detail[
                            "ModelProb"
                        ].apply(pct_nl)
                        st.dataframe(
                            detail[[
                                "Competition",
                                "Round",
                                "Match",
                                "Bet",
                                "Kans",
                                "BookmakerOdd",
                            ]],
                            use_container_width=True,
                            hide_index=True,
                            column_config={
                                "BookmakerOdd":
                                    st.column_config.NumberColumn(
                                        "Odd",
                                        format="%.2f",
                                    ),
                            },
                        )

        delete_ids = st.multiselect(
            "Bets/folds verwijderen",
            filtered["BetID"].astype(str).tolist(),
            key="delete_bet_ids",
        )
        if delete_ids and st.button(
            "Verwijder geselecteerde",
            key="delete_bets",
        ):
            st.session_state.bet_log = (
                st.session_state.bet_log[
                    ~st.session_state.bet_log[
                        "BetID"
                    ].astype(str).isin(delete_ids)
                ].reset_index(drop=True)
            )
            st.rerun()

        st.download_button(
            "⬇ Download betlog-backup",
            st.session_state.bet_log.to_csv(
                index=False
            ).encode("utf-8"),
            file_name="current_season_bets.csv",
            mime="text/csv",
            type="primary",
        )


# =============================================================================
# BACKTEST
# =============================================================================
with tabs[2]:
    st.subheader("Backtest · alleen dit seizoen")
    st.caption(
        "Walk-forward: iedere gespeelde wedstrijd wordt berekend met alleen eerdere 2026/27-wedstrijden. "
        "De eerste speelrondes zijn daardoor sterk afhankelijk van pseudo."
    )

    bt1, bt2, bt3 = st.columns(3)
    bt_comp = bt1.selectbox("Competitie", ALL_COMPETITIONS, key="bt_comp")
    bt_threshold = bt2.slider("Min. kans (%)", 50, 99, 70, 1) / 100
    market_group = bt3.selectbox(
        "Markt",
        ["Alle", "1X2", "BTTS", "Totaal goals", "Teamgoals"],
    )
    bt_pseudo = st.select_slider("Pseudo", options=list(range(0, 9)), value=pseudo)
    selection_mode = st.radio(
        "Selectie per wedstrijd",
        ["Alle kwalificerende bets", "Alleen hoogste modelkans"],
        horizontal=True,
    )

    comp_data = results[results["Competition"] == bt_comp].copy()
    cfg = BacktestConfig(pseudo_matches=bt_pseudo, warmup_matches=0, max_goals=8, odds_source="closing_avg")
    with st.spinner("Walk-forward backtest uitvoeren..."):
        bt_all, _ = build_backtest(comp_data, cfg)

    if bt_all.empty:
        st.info("Nog onvoldoende gespeelde wedstrijden voor een backtest.")
    else:
        bt = bt_all[bt_all["ModelProb"] >= bt_threshold].copy()
        if market_group == "1X2":
            bt = bt[bt["Market"].isin(["HOME", "DRAW", "AWAY"])]
        elif market_group == "BTTS":
            bt = bt[bt["Market"].str.startswith("BTTS", na=False)]
        elif market_group == "Totaal goals":
            bt = bt[bt["Market"].str.startswith("TOTAL_", na=False)]
        elif market_group == "Teamgoals":
            bt = bt[bt["Market"].str.startswith(("HOME_", "AWAY_"), na=False) & ~bt["Market"].isin(["HOME", "AWAY"])]

        if selection_mode == "Alleen hoogste modelkans" and not bt.empty:
            bt = bt.sort_values("ModelProb", ascending=False).groupby(["Date", "Match"], as_index=False).head(1)

        perf = performance(bt, stake=5.0)
        k1, k2, k3, k4, k5 = st.columns(5)
        k1.metric("Bets", perf["bets"])
        k2.metric("Hit-rate", pct_nl(perf["hit_rate"]))
        k3.metric("Fair ROI", pct_nl(perf["fair_roi"]))
        k4.metric("Echte odds ROI", pct_nl(perf["actual_roi"]))
        k5.metric("Max drawdown (€5)", eur(perf["max_drawdown"]))

        st.caption(
            f"{perf.get('bets_with_real_odds', 0)} bets hebben een historische marktprijs. "
            "Voor veel 0.5/1.5/3.5/4.5/5.5- en teamgoalmarkten is alleen fair-odds-calibratie beschikbaar."
        )

        d = bt[["Date", "Match", "Bet", "ModelProb", "FairOdd", "BookmakerOdd", "ModelEV", "Won"]].copy()
        d["Kans"] = d["ModelProb"].apply(pct_nl)
        d["Value"] = d["ModelEV"].apply(pct_nl)
        d = d[["Date", "Match", "Bet", "Kans", "FairOdd", "BookmakerOdd", "Value", "Won"]]
        st.dataframe(
            d,
            use_container_width=True,
            hide_index=True,
            column_config={
                "Date": st.column_config.DateColumn("Datum", format="DD-MM-YYYY"),
                "FairOdd": st.column_config.NumberColumn("Fair odd", format="%.2f"),
                "BookmakerOdd": st.column_config.NumberColumn("Hist. odd", format="%.2f"),
                "Won": st.column_config.CheckboxColumn("Gewonnen"),
            },
        )

        st.divider()
        st.markdown("#### Pseudo-vergelijking 0 t/m 8")
        sweep_rows = []
        for p in range(0, 9):
            pcfg = BacktestConfig(pseudo_matches=p, warmup_matches=0, max_goals=8, odds_source="closing_avg")
            pbt, _ = build_backtest(comp_data, pcfg)
            pbt = pbt[pbt["ModelProb"] >= bt_threshold].copy()
            if market_group == "1X2": pbt = pbt[pbt["Market"].isin(["HOME", "DRAW", "AWAY"])]
            elif market_group == "BTTS": pbt = pbt[pbt["Market"].str.startswith("BTTS", na=False)]
            elif market_group == "Totaal goals": pbt = pbt[pbt["Market"].str.startswith("TOTAL_", na=False)]
            elif market_group == "Teamgoals": pbt = pbt[pbt["Market"].str.startswith(("HOME_", "AWAY_"), na=False) & ~pbt["Market"].isin(["HOME", "AWAY"])]
            if selection_mode == "Alleen hoogste modelkans" and not pbt.empty:
                pbt = pbt.sort_values("ModelProb", ascending=False).groupby(["Date", "Match"], as_index=False).head(1)
            pp = performance(pbt, stake=5.0)
            sweep_rows.append({
                "Pseudo": p,
                "Bets": pp["bets"],
                "Hit-rate": pct_nl(pp["hit_rate"]),
                "Fair ROI": pct_nl(pp["fair_roi"]),
                "Echte ROI": pct_nl(pp["actual_roi"]),
                "Echte odds": pp.get("bets_with_real_odds", 0),
            })
        st.dataframe(pd.DataFrame(sweep_rows), use_container_width=True, hide_index=True)

        st.warning(
            "Dit seizoen is nog jong. Een pseudo-instelling die nu bovenaan staat kan door een kleine steekproef makkelijk omslaan."
        )


# =============================================================================
# DASHBOARD
# =============================================================================
with tabs[3]:
    st.subheader("Resultaten & exponentiële bankroll")
    log = recalculate_log(st.session_state.bet_log)
    timeline = bankroll_timeline(log, start_bankroll, secure_pct, exposure_pct)
    summary = current_bankroll_summary(log, start_bankroll, secure_pct, exposure_pct)

    settled = log[log["Status"].isin(["Gewonnen", "Verloren", "Void"])].copy()
    total_stake = float(pd.to_numeric(settled["Stake"], errors="coerce").fillna(0).sum()) if not settled.empty else 0.0
    total_profit = float(pd.to_numeric(settled["Profit"], errors="coerce").fillna(0).sum()) if not settled.empty else 0.0
    decided = settled[settled["Status"].isin(["Gewonnen", "Verloren"])]
    hit = float((decided["Status"] == "Gewonnen").mean()) if len(decided) else np.nan
    roi = total_profit / total_stake if total_stake else np.nan

    q1, q2, q3, q4, q5 = st.columns(5)
    q1.metric("Actieve bankroll", eur(summary["active_bankroll"]))
    q2.metric("Veiliggesteld", eur(summary["secured_total"]))
    q3.metric("Totale waarde", eur(summary["total_wealth"]))
    q4.metric("Nu inzetbaar", eur(summary["available_to_stake"]))
    q5.metric("ROI", pct_nl(roi))

    st.caption(
        f"Start {eur(start_bankroll)} · max. {exposure_pct_input}% per ISO-week · "
        f"{secure_pct_input:.0f}% van iedere weekuitbetaling veilig."
    )

    if log.empty:
        st.info("Voeg eerst bets toe op Mijn bets.")
    else:
        st.markdown("### Wat mag ik per week inzetten?")
        td = timeline.copy()
        st.dataframe(
            td,
            use_container_width=True,
            hide_index=True,
            column_config={
                "WeekKey": "Week",
                "Status": "Status",
                "OpeningBankroll": st.column_config.NumberColumn("Start bankroll", format="€%.2f"),
                "StakeBudget": st.column_config.NumberColumn("Mag inzetten", format="€%.2f"),
                "Stake": st.column_config.NumberColumn("Ingezet", format="€%.2f"),
                "Return": st.column_config.NumberColumn("Uitbetaling", format="€%.2f"),
                "Profit": st.column_config.NumberColumn("Winst/verlies", format="€%.2f"),
                "Secured": st.column_config.NumberColumn("Veilig uit uitbetaling", format="€%.2f"),
                "SecuredTotal": st.column_config.NumberColumn("Veilige pot", format="€%.2f"),
                "ClosingBankroll": st.column_config.NumberColumn("Nieuwe bankroll", format="€%.2f"),
                "RemainingBudget": st.column_config.NumberColumn("Nog inzetbaar", format="€%.2f"),
            },
        )

        chart_df = timeline[timeline["WeekKey"] != "Start"].copy()
        if not chart_df.empty:
            long = chart_df.melt(
                id_vars=["WeekKey"], value_vars=["ClosingBankroll", "SecuredTotal"],
                var_name="Type", value_name="Euro"
            )
            fig = px.line(long, x="WeekKey", y="Euro", color="Type", markers=True, title="Bankroll + veilige pot")
            st.plotly_chart(fig, use_container_width=True)

        st.markdown("### Per competitie")
        rows = []
        for comp, g in settled.groupby("Competition"):
            stake_sum = float(g["Stake"].sum())
            profit_sum = float(g["Profit"].sum())
            dec = g[g["Status"].isin(["Gewonnen", "Verloren"])]
            rows.append({
                "Competitie": comp,
                "Bets": len(g),
                "Inzet": stake_sum,
                "Winst": profit_sum,
                "ROI": pct_nl(profit_sum / stake_sum if stake_sum else np.nan),
                "Hit-rate": pct_nl((dec["Status"] == "Gewonnen").mean() if len(dec) else np.nan),
            })
        if rows:
            st.dataframe(
                pd.DataFrame(rows), use_container_width=True, hide_index=True,
                column_config={
                    "Inzet": st.column_config.NumberColumn(format="€%.2f"),
                    "Winst": st.column_config.NumberColumn(format="€%.2f"),
                }
            )

        st.markdown("### Per speelronde")
        round_rows = []
        for (comp, rnd), g in settled.groupby(["Competition", "Round"]):
            stake_sum = float(g["Stake"].sum())
            profit_sum = float(g["Profit"].sum())
            round_rows.append({
                "Competitie": comp, "Ronde": int(rnd), "Bets": len(g),
                "Inzet": stake_sum, "Winst": profit_sum,
                "ROI": pct_nl(profit_sum / stake_sum if stake_sum else np.nan),
            })
        if round_rows:
            st.dataframe(
                pd.DataFrame(round_rows).sort_values(["Competitie", "Ronde"]),
                use_container_width=True, hide_index=True,
                column_config={
                    "Inzet": st.column_config.NumberColumn(format="€%.2f"),
                    "Winst": st.column_config.NumberColumn(format="€%.2f"),
                }
            )

        values = timeline["ClosingBankroll"].dropna().astype(float).tolist()
        st.caption(
            f"Hit-rate {pct_nl(hit)} · netto resultaat {eur(total_profit)} · "
            f"max bankroll-drawdown {eur(max_drawdown(values))}."
        )

    st.divider()
    st.markdown("#### 5%-regel")
    st.write(
        "Voorbeeld: start bankroll €50. Na een volledig afgerekende week heb je €10 netto winst. "
        "Dan gaat €0,50 naar de veilige pot en wordt je actieve bankroll €59,50. "
        "Bij 100% exposure mag je de volgende week maximaal €59,50 over je eigen geselecteerde bets verdelen. "
        "Bij een verliesweek wordt niets veiliggesteld."
    )
