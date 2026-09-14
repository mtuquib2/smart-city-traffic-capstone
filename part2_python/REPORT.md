# Traffic Analytics Pipeline – Methodology and Findings

**Capstone Part 2 · Metro Interstate Traffic Volume (I-94 westbound, Minneapolis–St Paul, Oct 2012 – Sep 2018)**

## 1. Objective

Part 1 explored this dataset interactively in Power BI. Part 2 rebuilds that analysis as a
reproducible Python workflow: a logged, multi-stage pipeline that validates and cleans the raw
data, engineers ML-ready features, produces explanatory figures and exposes the results through a
command-line application. Every data change is traceable in `pipeline.log`.

## 2. Methodology

**Pipeline design.** `pipeline.py` is the single entry point and runs five stages:
load → validate schema → clean → feature engineering → visualisation. Each stage is a small,
tested function; failures raise a `PipelineError` that is logged at ERROR (with traceback) before
the program exits with a non-zero code. Logging uses module-level `getLogger(__name__)` loggers,
with console and file handlers configured only in the entry points.

**Data quality assessment and cleaning.** Profiling the 48,204 raw rows found:

* **Inconsistent labels**: 1,730 descriptions differed only by casing (`Sky is Clear`, `SQUALLS`).
  These were standardised.
* **Duplicates**: 17 exact duplicate rows, plus 7,612 rows repeating an hour once per reported weather
  condition (traffic volume identical in every case). The data was collapsed to one record per hour
  (40,575 rows), and the number of conditions was kept as a feature.
* **Impossible values**: 10 temperatures of 0 K and a 9,831 mm hourly rainfall (the world record
  is about 305 mm). These were replaced with the median of the *same calendar month*, computed in an
  explicit loop, so a January fault gets a January value rather than an annual average.
* **Sensor outages**: near-zero traffic counts during busy hours (e.g. 1 vehicle at 09:00). Traffic
  volume is multi-modal, so a single global rule does not work. The pipeline loops over
  48 (hour × weekday/weekend) groups and imputes counts below 5% of that group's median on
  non-holidays (47 rows, flagged in `traffic_volume_imputed`).
  A first attempt with 3×IQR fences flagged 754 rows. Inspection showed most were real holiday
  and snowstorm traffic, so that rule was rejected to preserve genuine signal.

**Feature engineering.** 38 features were added (11 → 49 columns). Time features include hour,
day of week, weekend, whole-day holiday and rush-hour flags, plus sin/cos encodings so that 23:00
is adjacent to 00:00. Weather features include °C temperature, one-hot conditions, precipitation,
low-visibility, severe and freezing indicators, and an ordinal severity score. Four continuous
variables were scaled (z-score or min-max, with log1p applied to skewed rainfall first). The target
`congestion_level` uses traffic-volume quartiles (Low < 1,255 ≤ Moderate < 3,430 ≤ High < 4,952 ≤ Very High),
which gives four balanced, data-driven classes that extend Part 1's three bands.

## 3. Key findings

1. **Commuting drives congestion.** Workdays have two sharp peaks (07:00 ≈ 6,180 and 16:00
   ≈ 6,330 vehicles/h). Weekends build to a flat midday plateau (≈ 4,400). Workdays carry 37% more
   traffic per hour, and 34.8% of workday hours are "Very High" versus 3.1% at weekends. During
   defined rush hours, 80% of hours are Very High. The busiest slots are Tuesday–Thursday 16:00–16:59
   (≈ 6,430–6,460 vehicles/h), confirming Part 1's 7–9 AM / 4–6 PM windows.
2. **Holidays behave like weekends.** Daytime workday traffic falls from ≈ 5,290 to ≈ 3,660
   vehicles/h on public holidays (−31%).
3. **Most weather barely matters once time of day is controlled.** A naive comparison exaggerates
   weather effects because mist and fog occur mostly at night. Within workday daytime hours, rain,
   cloud, drizzle and thunderstorms are all within ±3% of clear-sky traffic. **Snow is the exception
   (−8.2%)**, and precipitation below freezing reduces volume by about 10%. This refines Part 1,
   which attributed low volumes partly to fog and mist.
4. **Temperature is a weak predictor** (daytime r = 0.11). Volume dips only below about −15 °C, and
   winter months (Dec–Jan ≈ 5,000) sit about 8% below spring (≈ 5,450).
5. **Hour of day is the dominant signal for ML.** `hour_cos` correlates −0.77 with volume, versus
   −0.21 for `is_weekend` and ≤ 0.14 for any weather variable. A congestion classifier should be
   built around time features, with weather as a secondary adjustment.

## 4. Recommendations

* Target congestion measures (adaptive signals, ramp metering, transit frequency) at
  **06:00–08:00 and 15:00–17:00, Tuesday–Thursday first**.
* Weather-responsive operations should trigger on **snow and freezing precipitation**. Rain alone does
  not justify intervention on volume grounds, although it may for safety reasons.
* Travellers with flexibility gain the most by shifting to 10:00–11:00 or after 19:00. The
  `recommend` command in `cli_app/traffic_app.py` puts this into practice for any day.

## 5. Limitations and next steps

* One sensor and one direction, with an ~11-month data gap (Aug 2014 – Jun 2015) and a
  single weather record per hour after de-duplication.
* Quartile thresholds and scaling parameters were fitted on the full dataset. A predictive model
  should fit them on the training period only, and use a time-based split.
* Next steps: train a baseline classifier (e.g. gradient boosting) on the engineered features, add
  event and incident data, and schedule the pipeline to refresh as new data arrives.
