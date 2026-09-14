# Metro Interstate Traffic Analytics Pipeline (Capstone Part 2)

A reproducible Python pipeline that turns the raw **Metro Interstate Traffic Volume** dataset
(hourly westbound I-94 traffic between Minneapolis and St Paul, Oct 2012 – Sep 2018, with weather
and holiday attributes) into a validated, ML-ready dataset, a set of explanatory figures and a small
command-line analytics application.

It builds on Part 1 (Power BI dashboard): the commuter peaks, weather effects and Low/Medium/High
traffic banding identified there are re-created programmatically, tested and extended here.

---

## Project structure

```
part2_python/
├── pipeline.py               # ENTRY POINT: load -> validate -> clean -> features -> figures
├── feature_engineering.py    # Task 2: time, weather, scaled and target features
├── visualizations.py         # Task 3: seven Matplotlib figures
├── cli_app/
│   └── traffic_app.py        # Task 4: command-line query application (second entry point)
├── data/
│   ├── raw/Metro_Interstate_Traffic_Volume.csv   # original, untouched input
│   └── processed/            # traffic_clean.csv, traffic_features.csv (generated, git-ignored)
├── figures/                  # fig01 ... fig07 PNGs (generated, committed for review)
├── pipeline.log              # sample normal run (INFO) - review without re-running
├── logs/
│   ├── pipeline_debug.log        # sample run with --log-level DEBUG
│   ├── pipeline_error_example.log# sample failed run (missing input file)
│   └── app.log                   # sample CLI session incl. invalid input
├── tests/                    # pytest unit tests for all three stages + CLI
├── REPORT.md                 # 1-2 page methodology and findings report
├── requirements.txt
└── README.md
```

## Setup

Requires Python 3.10+.

```bash
python -m venv .venv
.venv\Scripts\activate          # Windows  (macOS/Linux: source .venv/bin/activate)
pip install -r requirements.txt
```

## How to run

Run all commands from the project root.

**1. Run the full pipeline** (writes `data/processed/`, `figures/`, `pipeline.log`):

```bash
python pipeline.py
```

| Option | Purpose |
|---|---|
| `--log-level DEBUG` | Also log intermediate values (thresholds, medians, scaling parameters) |
| `--log-file PATH` | Write the log somewhere else (default `pipeline.log`) |
| `--input PATH` | Use a different raw CSV |
| `--skip-figures` | Run cleaning and feature engineering only |

The process exits with code `0` on success and `1` if any stage fails (after logging an ERROR).

**2. Query the processed data with the CLI app:**

```bash
python cli_app/traffic_app.py lookup --datetime "2017-12-25 17:00"      # traffic + weather for an hour
python cli_app/traffic_app.py peaks --top 5 --day-type workday           # busiest day/hour slots
python cli_app/traffic_app.py compare                                    # workday vs weekend
python cli_app/traffic_app.py recommend --day friday --start 7 --end 19  # quietest hours to travel
python cli_app/traffic_app.py weather --condition snow                   # weather impact on traffic
python cli_app/traffic_app.py --help                                     # full usage
```

Example output:

```
$ python cli_app/traffic_app.py recommend --day friday --start 7 --end 19
Recommended travel times on Friday between 07:00 and 19:59 (non-holiday):
  1. 19:00-19:59  ~3,664 vehicles/hour  (usually High)
  2. 10:00-10:59  ~4,593 vehicles/hour  (usually High)
  3. 18:00-18:59  ~4,691 vehicles/hour  (usually High)
Avoid 16:00-16:59: busiest hour in this window (~6,113 vehicles/hour).
```

Invalid input (e.g. `--datetime "2016-13-40 25:00"`, `--day funday`, `--top abc`) produces one
clear ERROR log line and exit code `2`, never a raw traceback.

**3. Run the tests:**

```bash
python -m pytest
```

---

## Logging configuration

| Aspect | Implementation |
|---|---|
| Loggers | Every module calls `logger = logging.getLogger(__name__)`; no module logs through the bare root logger. |
| Handlers | Configured **only in the entry points** (`configure_logging()` in `pipeline.py` and `cli_app/traffic_app.py`). Each attaches a console `StreamHandler` and a `FileHandler` to the root logger, so records from `feature_engineering` and `visualizations` propagate to both. |
| Where logs go | Pipeline: console (stdout) + `pipeline.log`, overwritten each run so the file is a single clean audit trail. App: console (stderr, keeping command answers on stdout clean) + `logs/app.log`, appended so it holds a history of queries. |
| Format | `%(asctime)s \| %(levelname)-8s \| %(module)s \| %(message)s`<br>e.g. `2026-09-14 14:04:40 \| WARNING  \| pipeline \| Imputed 10 row(s) in 'temp' with monthly median: ...` |
| Level selection | `--log-level {DEBUG,INFO,WARNING,ERROR}` (default `INFO`). DEBUG records are emitted only when DEBUG is selected. |
| `print()` | Used only in `cli_app/traffic_app.py` to show the answer to a command. All progress/status reporting uses logging. |

### What each level means in this project

| Level | Used for | Examples |
|---|---|---|
| **DEBUG** | Fine-grained internal values useful for troubleshooting, not part of the output | Quartile thresholds `Q1=1255.0, Q2=3430.0, Q3=4952.0`; per-month imputation medians; per-(hour, day type) outage thresholds; z-score mean/std; min/max scaling bounds |
| **INFO** | Normal, expected milestones | `Raw data loaded successfully: 48204 rows, 9 columns`; each cleaning step starting/passing; dataset shape before/after feature engineering; `Saved figure: figures/fig01_...png`; CLI `Command invoked: peaks with arguments {...}` |
| **WARNING** | Unexpected but recoverable data changes, always with **row count and reason** | `Dropped 17 row(s) in duplicate removal: exact duplicate records`; `Imputed 1 row(s) in 'rain_1h' with monthly median: rainfall ... above 305 mm/h`; lookup falling back to the nearest hour |
| **ERROR** | The pipeline or command cannot continue as planned | `Pipeline aborted: Raw data file not found: ...` (with `exc_info=True` traceback in the log); CLI `Invalid input for 'lookup': Malformed date/time ...` |

Sample logs are committed (`pipeline.log` plus debug, error and app samples in `logs/`) so graders can review the behaviour without re-running.

---

## Pipeline details

### Task 1 – Loading, validation and cleaning (`pipeline.py`)

1. **Load** with `pd.read_csv` inside `try/except` for `FileNotFoundError`, `PermissionError`,
   `EmptyDataError`, `ParserError` and `UnicodeDecodeError` (no bare `except:`); failures become a
   `PipelineError` that `main()` logs at ERROR before exiting with code 1.
2. **Validate schema first**: all 9 expected columns present, numeric columns numeric, dataset non-empty.
3. **Clean** in six separately logged steps:

| Step | Issue found in raw data | Action | Rows affected |
|---|---|---|---|
| Standardise categoricals | `Sky is Clear` vs `sky is clear`, `SQUALLS` | Trim, lower-case descriptions, canonical `weather_main` names | 1,730 modified |
| Parse date/time | – | Explicit format `%Y-%m-%d %H:%M:%S`; drop unparseable / out-of-range timestamps | 0 dropped |
| Duplicates | Exact duplicate rows | Dropped | 17 dropped |
| Duplicates | Same hour repeated once per weather condition (volume identical) | Keep primary record, store count in `n_weather_conditions` | 7,612 collapsed |
| Impossible values | `temp` = 0 K (absolute zero) | Impute with **that calendar month's** median (explicit loop over months) | 10 imputed |
| Impossible values | `rain_1h` = 9,831 mm (world record ≈ 305 mm/h) | Impute with monthly median | 1 imputed |
| Traffic outliers | Near-zero counts in daytime (sensor outage / closure) | Loop over 48 (hour × weekday/weekend) groups; counts < 5% of the group median on non-holidays are imputed with the group median, flagged in `traffic_volume_imputed` | 47 imputed |
| Final validation | – | No missing values, unique hourly timestamps | 40,575 rows kept |

A global IQR rule was deliberately **not** used for traffic volume: an early version flagged 754 rows,
most of them genuine low traffic on holidays and in snowstorms, and imputing those would have erased
the weather signal that later analysis depends on.

### Task 2 – Feature engineering (`feature_engineering.py`)

40,575 × 11 → 40,575 × 49 columns.

* **Time**: `hour`, `day_of_week`, `day_name`, `month`, `year`, `is_weekend`, `is_holiday` (the raw
  feed labels only the 00:00 record of a holiday, so the flag is propagated to the whole date),
  `is_rush_hour` (workday 06–09 & 15–18, from Part 1), and cyclical **sin/cos** encodings of hour,
  day of week and month.
* **Weather**: `temp_c`; one-hot `weather_*` columns; `is_precipitation`, `is_low_visibility`,
  `is_severe_weather`, `is_freezing`; ordinal `weather_severity` (0 good, 1 low visibility or
  precipitation, 2 precipitation below freezing, 3 severe).
* **Scaled**: `temp_c_scaled` (z-score), `clouds_all_scaled` (min-max), `rain_1h_scaled`
  (log1p then min-max, because rainfall is extremely skewed), `snow_1h_scaled` (min-max).
* **Target – `congestion_level`** (`congestion_code` 0–3): quartiles of cleaned hourly volume:

| Level | Rule | Threshold (this data) |
|---|---|---|
| Low | volume < Q1 | < 1,255 |
| Moderate | Q1 ≤ volume < median | 1,255 – 3,429 |
| High | median ≤ volume < Q3 | 3,430 – 4,951 |
| Very High | volume ≥ Q3 | ≥ 4,952 |

Quartiles make the classes data-driven and balanced (~10,140 hours each), and extend Part 1's
three bands with a fourth level that isolates peak congestion. For a real train/test split the
thresholds (and scaling parameters) should be fitted on training data only to avoid leakage.

### Task 3 – Visualisations (`visualizations.py`)

| Figure | Interpretation |
|---|---|
| `fig01_hourly_profile_weekday_weekend.png` | Workdays show two sharp commuter peaks (07:00 ≈ 6,180 and 16:00 ≈ 6,330 vehicles/h) with a midday dip; weekends rise slowly to a flat 12:00–16:00 plateau (≈ 4,400). After 19:00 the two profiles converge, so the difference is entirely commuting. |
| `fig02_heatmap_day_hour.png` | The darkest cells are Tuesday–Thursday 07:00 and 16:00; Monday and Friday are slightly lighter, and Saturday/Sunday mornings are the quietest daytime cells. Congestion is a weekday-time problem, not a location-wide one. |
| `fig03_weather_impact_daytime.png` | Comparing like-for-like hours (workdays 07:00–18:59), most weather has almost no effect (±3%). Snow is the clear exception (−8.2% vs clear sky, n = 794). Fog's +6.6% rests on only 52 hours. |
| `fig04_temperature_vs_traffic.png` | Daytime volume is almost flat across temperature (r = 0.11); the median band only drops below −15 °C (≈ 4,700 vs ≈ 5,200). Temperature is a weak predictor on its own, matching Part 1. |
| `fig05_traffic_distribution.png` | Volume is multi-modal: a night-time cluster below 1,000, a shoulder-hour cluster around 2,800 and a daytime cluster at 4,000–6,000. This is why quartile-based congestion classes, not a single mean, are used. |
| `fig06_congestion_share_by_hour.png` | "Very High" hours are almost entirely within 06:00–08:00 and 14:00–17:00 (over 60% of those hours), while 01:00–04:00 is almost always Low. Hour of day alone separates most congestion classes. |
| `fig07_monthly_trend.png` | Monthly averages are stable (≈ 3,000–3,500) with recurring December–January dips; an ~11-month sensor gap (Aug 2014 – Jun 2015) is shown explicitly rather than interpolated. |

---

## Reproducibility and version control

* The raw CSV is committed and never modified; every derived file can be rebuilt with `python pipeline.py`.
* All thresholds (plausibility bounds, outage ratio, rush-hour windows) are named constants at the top of each module.
* The pipeline is deterministic (no random sampling), so reruns produce identical outputs.
* Git history has one descriptive commit per task (scaffold → Task 1 → Task 2 → Task 3 → Task 4 → Task 5).

To publish to GitHub:

```bash
git remote add origin https://github.com/<your-username>/smart-city-traffic-capstone.git
```

```bash
git push -u origin main
```
