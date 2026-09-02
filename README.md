# Current Season Betting Lab v1.0

Een tweede, volledig losse Streamlit-app voor alleen **seizoen 2026/27**.

## Wat de app doet

### Start
- 6 competities: Eredivisie, Premier League, La Liga, Bundesliga, Serie A en Ligue 1.
- Volledig programma per officiële speelronde via Fixture Download.
- Alleen reeds gespeelde wedstrijden uit 2026/27 worden gebruikt voor de berekening.
- Keuze pseudo-wedstrijden 0 t/m 8.
- Instelbare modelkansgrens.
- Markten: 1X2, dubbele kans, BTTS, totaal goals 0.5 t/m 5.5 en teamgoals 0.5 t/m 5.5.
- Fair odd + TOTO odd waar de publieke TOTO-pagina automatisch leesbaar is.
- Optionele TOTO-valuefilter.

### Mijn bets
- Per competitie en speelronde bets vastleggen.
- Modelkans/fair odd automatisch overnemen.
- Bookmakerodd en inzet invullen.
- Open, Gewonnen, Verloren of Void.
- Bekende modelbets worden automatisch afgerekend zodra Football-Data de uitslag bevat.
- CSV-backup importeren/downloaden.

### Backtest
- Alleen 2026/27.
- Walk-forward: iedere historische wedstrijd wordt met eerdere wedstrijden uit hetzelfde seizoen voorspeld.
- Pseudo instelbaar.
- Kansgrens instelbaar.
- Marktgroep instelbaar.
- Alle bets of alleen hoogste modelkans per wedstrijd.
- Hit-rate, fair-odds ROI, echte odds ROI waar Football-Data een historische prijs bevat, max drawdown.
- Pseudo 0 t/m 8 naast elkaar vergelijken.

### Dashboard
- Startbankroll standaard €50.
- Exponentieel compounden.
- Maximaal inzetpercentage per week, standaard 100%.
- 5% van iedere positieve weekwinst veiligstellen.
- Actieve bankroll, veilige pot, totale waarde, inzetbudget, ROI, hit-rate en drawdown.
- Resultaten per competitie en per speelronde.

## Waarom de bankroll per ISO-week wordt afgerekend

Je speelt zes competities tegelijk. Hun speelrondes vallen door elkaar heen. Als iedere competitie-speelronde een afzonderlijke compoundstap zou zijn, zou dezelfde kalenderweek soms vijf of zes keer achter elkaar je bankroll aanpassen terwijl de wedstrijden gelijktijdig lopen.

Daarom:
- bets kun je **per competitie + speelronde** bijhouden;
- bankroll, exponentiële groei en de 5%-veiligstelling worden **per ISO-week** afgerekend.

Zo is `Mag inzetten` daadwerkelijk het bedrag dat je die week over jouw eigen gekozen bets mag verdelen.

## 5%-voorbeeld

- Start week: €50,00
- Ingezet: €50,00 verdeeld over jouw bets
- Uitbetaling totaal: €60,00
- Weekwinst: €10,00
- Veiligstellen: 5% × €10 = €0,50
- Nieuwe actieve bankroll: €50 + €10 - €0,50 = **€59,50**
- Veilige pot: **€0,50**
- Bij 100% exposure mag je de volgende week maximaal **€59,50** inzetten.

Bij een negatieve weekwinst wordt €0 veiliggesteld.

## Data

### Huidig seizoen
Football-Data 2026/27 CSV's:
- Eredivisie N1
- Premier League E0
- La Liga SP1
- Bundesliga D1
- Serie A I1
- Ligue 1 F1

De app laadt **geen vorig seizoen voor het model**. Alleen `2026/27` wordt opgehaald.

De data-cache is 15 minuten. Gebruik `↻ Data` om direct opnieuw te laden. Zodra Football-Data een gespeelde wedstrijd aan de 2026/27 CSV toevoegt, komt die na verversen automatisch in de modeldata.

### Volledig programma
Fixture Download 2026/27 result pages leveren het volledige programma inclusief speelronden. De programmacache is 6 uur.

### TOTO
De app probeert publieke TOTO-wedstrijdpagina's automatisch te koppelen. Omdat TOTO-pagina's dynamisch zijn kan een odd ontbreken. Dit blokkeert de modelberekening niet.

## Belangrijk: opslag van je eigen bets

Streamlit Community Cloud is geen permanente database. Daarom staat de betlog in de huidige gebruikerssessie.

**Download na wijzigingen `current_season_bets.csv` als backup.**

Bij een volgende sessie kun je die CSV weer importeren.

Een volgende versie kan gratis permanente opslag krijgen via bijvoorbeeld Google Sheets of Supabase.

## Als tweede gratis app online zetten

Maak een tweede GitHub repository, bijvoorbeeld:

`current-season-betting-lab`

Upload de losse bestanden uit deze map:
- `app.py`
- `engine.py`
- `tracker.py`
- `requirements.txt`
- `README.md`
- `.streamlit/config.toml`

Ga daarna naar Streamlit Community Cloud en kies **Create app**.

Instellingen:
- Repository: jouw nieuwe repository
- Branch: `main`
- Main file path: `app.py`

Klik **Deploy**. Dit wordt een volledig aparte URL naast je bestaande Football Analysis Lab.


## v1.0.1 — 5% over uitbetaling

De veilige-potregel is aangepast.

Voor een volledig afgerekende ISO-week:

- `Uitbetaling = som van alle returns van gewonnen/void bets`
- `Veiliggesteld = Uitbetaling × 5%`
- `Nieuwe actieve bankroll = Oude bankroll + netto winst/verlies - veiliggesteld`

Voorbeeld:
- Startbankroll: €50
- Totale inzet: €50
- Totale uitbetaling: €90
- Netto resultaat: +€40
- 5% van €90 veilig: €4,50
- Nieuwe actieve bankroll: €85,50
- Veilige pot: €4,50

Als de uitbetaling €40 is op €50 inzet:
- Netto resultaat: -€10
- 5% van €40 veilig: €2
- Nieuwe actieve bankroll: €38
- Veilige pot: €2

Dus de veilige pot kan ook groeien in een verliesweek, zolang er daadwerkelijk een uitbetaling is.
