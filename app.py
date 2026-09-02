from __future__ import annotations

import uuid
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
    fetch_toto_week_odds,
    fetch_oddspapi_toto_board,
    safe_fetch_oddspapi_toto_board,
    oddspapi_account_status,
    fixture_bet_candidates,
    load_data,
    load_full_season_fixture_catalog,
    performance,
    resolve_catalog_team_name,
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


def eur(value):
    if pd.isna(value):
        return "–"
    return f"€{float(value):,.2f}".replace(",", "X").replace(".", ",").replace("X", ".")


def get_toto_api_key():
    """Streamlit Secret heeft voorkeur; handmatige sessie-key is fallback."""
    try:
        secret_key = st.secrets.get("ODDSPAPI_KEY", "")
    except Exception:
        secret_key = ""
    return str(secret_key or st.session_state.get("manual_oddspapi_key", "")).strip()

@st.cache_data(ttl=900, show_spinner=False)
def cached_results():
    return load_data(ALL_COMPETITIONS, [SEASON])


@st.cache_data(ttl=21600, show_spinner=False)
def cached_schedule():
    return load_full_season_fixture_catalog(ALL_COMPETITIONS)


@st.cache_data(ttl=3600, show_spinner=False)
def cached_oddspapi_account(api_key):
    # /account is volgens OddsPapi unmetered en geschikt voor quota/key checks.
    return oddspapi_account_status(api_key)


@st.cache_data(ttl=1800, show_spinner=False)
def cached_oddspapi_board(api_key):
    # Nooit exception naar Streamlit: safe wrapper retourneert altijd een dict.
    return safe_fetch_oddspapi_toto_board(api_key)


@st.cache_data(ttl=1800, show_spinner=False)
def cached_toto_week(competition, matches, api_key=""):
    board = cached_oddspapi_board(api_key) if api_key else None
    return fetch_toto_week_odds(
        competition,
        matches,
        timeout=9,
        max_workers=6,
        api_key=api_key or None,
        api_board=board,
    )


# Data verversen
rc1, rc2 = st.sidebar.columns(2)
if rc1.button("↻ Data", use_container_width=True):
    cached_results.clear()
    st.rerun()
if rc2.button("↻ TOTO", use_container_width=True):
    cached_toto_week.clear()
    cached_oddspapi_board.clear()
    cached_oddspapi_account.clear()
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
toto_api_key = get_toto_api_key()

if toto_api_key:
    api_account = cached_oddspapi_account(toto_api_key)
    if api_account.get("_ok"):
        st.sidebar.success(api_account.get("_status", "OddsPapi API key geldig"))
        st.sidebar.caption(
            "De app probeert de complete TOTO-feed. Als die niet reageert, "
            "wordt automatisch de publieke TOTO-site gebruikt."
        )
    else:
        st.sidebar.warning(api_account.get("_status", "OddsPapi niet beschikbaar"))
        st.sidebar.caption(
            "De app blijft werken via de publieke TOTO-site; er verschijnt geen rood foutscherm."
        )
else:
    st.sidebar.warning(
        "Website-fallback actief: alleen markten die TOTO direct in de publieke pagina laadt."
    )

with st.sidebar.expander("TOTO API-instellingen"):
    st.caption(
        "Je Streamlit Secret ODDSPAPI_KEY heeft voorrang. "
        "Een handmatige key hieronder geldt alleen voor deze sessie."
    )
    manual_key = st.text_input(
        "OddsPapi API key",
        type="password",
        value=st.session_state.get("manual_oddspapi_key", ""),
        key="oddspapi_key_input",
    )
    if st.button("API key gebruiken", key="save_oddspapi_key"):
        st.session_state.manual_oddspapi_key = manual_key.strip()
        cached_toto_week.clear()
        cached_oddspapi_board.clear()
        cached_oddspapi_account.clear()
        st.rerun()

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
    rounds = sorted(comp_schedule["Round"].dropna().astype(int).unique().tolist())

    # Kies standaard eerste ronde met minstens één nog niet gespeelde wedstrijd.
    default_round = rounds[0] if rounds else 1
    comp_results = results[results["Competition"] == competition]
    played = {
        (resolve_catalog_team_name(results, competition, r.HomeTeam),
         resolve_catalog_team_name(results, competition, r.AwayTeam),
         pd.Timestamp(r.Date).normalize())
        for r in comp_results.itertuples()
    }
    for rn in rounds:
        group = comp_schedule[comp_schedule["Round"] == rn]
        pending = False
        for r in group.itertuples():
            h = resolve_catalog_team_name(results, competition, r.HomeTeam)
            a = resolve_catalog_team_name(results, competition, r.AwayTeam)
            if (h, a, pd.Timestamp(r.Date).normalize()) not in played:
                pending = True
                break
        if pending:
            default_round = rn
            break

    selected_round = c2.selectbox(
        "Speelronde", rounds, index=rounds.index(default_round) if default_round in rounds else 0
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
        toto_week = cached_toto_week(competition, pairs, toto_api_key)

    results_by_match = {}
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
        results_by_match[fixture.Match] = cands
        all_bets.append(cands)
        summaries.append({"Match": fixture.Match, "Bets": len(cands)})

    s1, s2, s3, s4 = st.columns(4)
    s1.metric("Wedstrijden", len(round_df))
    s2.metric("Geselecteerde bets", int(sum(x["Bets"] for x in summaries)))
    s3.metric("Pseudo", pseudo)
    linked = int(toto_week.get("_matches_linked", 0))
    s4.metric("TOTO matches gekoppeld", f"{linked}/{len(round_df)}")
    provider = toto_week.get("_provider", "website")
    if provider == "structured":
        st.caption(
            "TOTO bron: structured TOTO NL odds-feed. Ontbrekende odds betekenen dat TOTO "
            "die markt op dit moment niet aanbiedt in de feed."
        )
    elif provider == "website-fallback":
        structured_status = toto_week.get(
            "_structured_status",
            "De complete TOTO-feed kon niet worden gebruikt.",
        )
        st.warning(
            f"{structured_status} "
            "De app gebruikt nu automatisch de publieke TOTO-pagina. "
            "Extra markten achter 'Bekijk meer' kunnen daardoor ontbreken."
        )
    else:
        st.caption(
            "TOTO bron: publieke website. Voor alle beschikbare markten kun je de complete "
            "TOTO odds-feed activeren in de sidebar."
        )

    for fixture in round_df.itertuples():
        cands = results_by_match[fixture.Match]
        title = (
            f"{pd.Timestamp(fixture.Date).strftime('%d-%m')} · {fixture.Match} · "
            f"{len(cands)} {'bet' if len(cands)==1 else 'bets'}"
        )
        with st.expander(title, expanded=len(cands) > 0):
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

    nonempty = [x for x in all_bets if not x.empty]
    if nonempty:
        combined = pd.concat(nonempty, ignore_index=True)
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
    st.subheader("Mijn bets per speelronde")
    st.info(
        "De betlog staat in je huidige Streamlit-sessie. Download na wijzigingen de CSV-backup. "
        "Daarmee voorkom je dat je gegevens kwijtraakt als de gratis app opnieuw opstart."
    )

    upload = st.file_uploader("Importeer betlog-backup", type=["csv"])
    if upload is not None:
        try:
            st.session_state.bet_log = normalise_bet_log(pd.read_csv(upload))
            st.success("Betlog geïmporteerd.")
        except Exception as exc:
            st.error(f"Importeren mislukt: {exc}")

    b1, b2 = st.columns(2)
    bet_comp = b1.selectbox("Competitie", ALL_COMPETITIONS, key="bet_comp")
    bs = schedule[schedule["Competition"] == bet_comp]
    bet_rounds = sorted(bs["Round"].dropna().astype(int).unique().tolist())
    bet_round = b2.selectbox("Speelronde", bet_rounds, key="bet_round")

    fixtures = bs[bs["Round"] == bet_round].sort_values("DateTime")
    selected_match = st.selectbox("Wedstrijd", fixtures["Match"].tolist(), key="bet_match")
    fixture = fixtures[fixtures["Match"] == selected_match].iloc[0]

    # Toon alle markten vanaf 50%, zodat gebruiker zelf kiest.
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
        st.warning("Geen modelmarkten beschikbaar voor deze wedstrijd.")
    else:
        labels = [f"{r.Bet} · {pct_nl(r.ModelProb)} · fair {r.FairOdd:.2f}" for r in choices.itertuples()]
        choice_label = st.selectbox("Bet", labels)
        selected = choices.iloc[labels.index(choice_label)]

        iso = pd.Timestamp(fixture["Date"]).isocalendar()
        week_key = f"{iso.year}-W{int(iso.week):02d}"
        summary = current_bankroll_summary(
            st.session_state.bet_log, start_bankroll, secure_pct, exposure_pct
        )

        # Probeer TOTO voor deze ene match te vullen.
        single_toto = cached_toto_week(bet_comp, ((fixture["HomeTeam"], fixture["AwayTeam"]),), toto_api_key)
        toto = toto_week_result_for_match(single_toto, fixture["HomeTeam"], fixture["AwayTeam"])
        selected_df = attach_toto_odds(pd.DataFrame([selected]), toto)
        toto_odd = pd.to_numeric(selected_df.iloc[0].get("TotoOdd"), errors="coerce")
        default_odd = float(toto_odd) if pd.notna(toto_odd) and toto_odd > 1 else float(selected["FairOdd"])

        m1, m2, m3 = st.columns(3)
        m1.metric("Beschikbaar inzetbudget", eur(summary["available_to_stake"]))
        m2.metric("Modelkans", pct_nl(selected["ModelProb"]))
        m3.metric("Fair odd", f"{selected['FairOdd']:.2f}")

        i1, i2 = st.columns(2)
        bookmaker_odd = i1.number_input(
            "Bookmaker odd", min_value=1.01, max_value=1000.0,
            value=max(1.01, round(default_odd, 2)), step=0.01,
        )
        max_stake = max(float(summary["available_to_stake"]), 0.01)
        stake = i2.number_input(
            "Inzet", min_value=0.01, max_value=max_stake,
            value=min(5.0, max_stake), step=0.50,
        )
        model_value = float(selected["ModelProb"]) * bookmaker_odd - 1
        st.caption(
            f"Model-value bij deze odd: {pct_nl(model_value)} · week {week_key}."
        )

        if st.button("＋ Bet toevoegen", type="primary"):
            new = pd.DataFrame([{
                "BetID": str(uuid.uuid4())[:8],
                "CreatedAt": datetime.now().isoformat(timespec="seconds"),
                "Competition": bet_comp,
                "Round": int(bet_round),
                "WeekKey": week_key,
                "Date": fixture["Date"],
                "Match": fixture["Match"],
                "HomeTeam": fixture["HomeTeam"],
                "AwayTeam": fixture["AwayTeam"],
                "Bet": selected["Bet"],
                "ModelProb": float(selected["ModelProb"]),
                "FairOdd": float(selected["FairOdd"]),
                "BookmakerOdd": bookmaker_odd,
                "Stake": stake,
                "Status": "Open",
                "Return": np.nan,
                "Profit": np.nan,
            }])
            st.session_state.bet_log = pd.concat(
                [st.session_state.bet_log, new], ignore_index=True
            )
            st.success("Bet toegevoegd.")
            st.rerun()

    st.divider()
    log = recalculate_log(st.session_state.bet_log)
    st.session_state.bet_log = log

    if log.empty:
        st.caption("Nog geen bets opgeslagen.")
    else:
        f1, f2, f3 = st.columns(3)
        fcomp = f1.selectbox("Filter competitie", ["Alle"] + ALL_COMPETITIONS)
        fr_values = sorted(log["Round"].dropna().astype(int).unique().tolist())
        fround = f2.selectbox("Filter ronde", ["Alle"] + fr_values)
        fstatus = f3.selectbox("Status", ["Alle", "Open", "Gewonnen", "Verloren", "Void"])

        filtered = log.copy()
        if fcomp != "Alle": filtered = filtered[filtered["Competition"] == fcomp]
        if fround != "Alle": filtered = filtered[filtered["Round"] == int(fround)]
        if fstatus != "Alle": filtered = filtered[filtered["Status"] == fstatus]

        editor = filtered.copy()
        editor["Kans"] = editor["ModelProb"].apply(pct_nl)
        editor = editor[[
            "BetID", "Competition", "Round", "WeekKey", "Match", "Bet", "Kans",
            "FairOdd", "BookmakerOdd", "Stake", "Status", "Return", "Profit"
        ]]
        edited = st.data_editor(
            editor,
            use_container_width=True,
            hide_index=True,
            disabled=["BetID", "Competition", "Round", "WeekKey", "Match", "Bet", "Kans", "FairOdd", "Return", "Profit"],
            column_config={
                "Status": st.column_config.SelectboxColumn(
                    "Status", options=["Open", "Gewonnen", "Verloren", "Void"], required=True
                ),
                "BookmakerOdd": st.column_config.NumberColumn("Odd", format="%.2f"),
                "Stake": st.column_config.NumberColumn("Inzet", format="€%.2f"),
                "Return": st.column_config.NumberColumn("Uitbetaling", format="€%.2f"),
                "Profit": st.column_config.NumberColumn("Winst/verlies", format="€%.2f"),
            },
        )
        if st.button("Wijzigingen opslaan"):
            base = st.session_state.bet_log.copy()
            for _, erow in edited.iterrows():
                mask = base["BetID"].astype(str) == str(erow["BetID"])
                base.loc[mask, "BookmakerOdd"] = erow["BookmakerOdd"]
                base.loc[mask, "Stake"] = erow["Stake"]
                base.loc[mask, "Status"] = erow["Status"]
            st.session_state.bet_log = recalculate_log(base)
            st.rerun()

        delete_ids = st.multiselect("Bets verwijderen", filtered["BetID"].astype(str).tolist())
        if delete_ids and st.button("Verwijder geselecteerde bets"):
            st.session_state.bet_log = st.session_state.bet_log[
                ~st.session_state.bet_log["BetID"].astype(str).isin(delete_ids)
            ].reset_index(drop=True)
            st.rerun()

        st.download_button(
            "⬇ Download betlog-backup",
            st.session_state.bet_log.to_csv(index=False).encode("utf-8"),
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
