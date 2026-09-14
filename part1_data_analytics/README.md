# Part 1 – Data Analytics (SQL, Statistics and Power BI)

Exploratory analysis of the raw Metro Interstate Traffic Volume dataset (48,204 rows).

```
part1_data_analytics/
├── sql/
│   ├── metro_traffic.db          # SQLite database with the raw table Metro_Interstate_Traffic_Volume
│   ├── part1_queries.sql         # the six SQL analyses, with written interpretations as comments
│   └── metro_traffic.sqbpro      # original DB Browser for SQLite project file
├── powerbi/
│   └── CAPSTONE.pbix             # Power BI "Traffic Intelligence" dashboard
├── statistics/
│   ├── statistical_analysis.py   # reproduces the SQL statistics in Python + hypothesis tests
│   ├── statistical_results.md    # generated results with interpretation
│   └── statistical_results.json  # generated results (machine-readable)
├── insights_report.pdf           # insights report for the SmartCity Mobility team
└── insights_report.docx          # editable source of the insights report
```

## Contents

| Item | What it covers |
|---|---|
| `sql/part1_queries.sql` | Data-completeness check, annual trends and year-on-year change, holiday temperatures (New Year's Day, Labor Day), Pearson correlation of temperature vs volume, and probability/odds analysis of congestion (> 5,500 vehicles/h) vs weather |
| `powerbi/CAPSTONE.pbix` | Daily and hourly traffic trends, weather impact, temperature scatter and KPIs |
| `statistics/` | Descriptive statistics; Pearson/Spearman correlation (raw and with 0 K sensor faults removed); joint and conditional probabilities, independence check, odds ratio; chi-square test of weather × congestion with Cramér's V; Welch t-test of weekday vs weekend volume with Cohen's d |
| `insights_report.pdf` | Summary of insights and strategic implications for mobility planning |

## Key statistical findings

| Analysis | Result | Meaning |
|---|---|---|
| Temperature vs volume | Pearson r = 0.13 (r² = 1.8%) | Weak positive association; statistically significant only because n is large |
| P(Congestion) | 0.147 | Congestion (> 5,500 vehicles/h) occurs in about 1 in 7 hours |
| P(Congestion ∩ Clear) vs P(C)·P(Clear) | 0.0366 vs 0.0409 | Close, so congestion and clear weather are roughly independent |
| Odds ratio, clear vs clouds | 0.735 | The odds of congestion are about 26% lower in clear weather |
| Chi-square, weather × congestion | χ² = 165.8, p < 0.001, Cramér's V = 0.059 | Significant but negligible association |
| Weekday vs weekend | 3,534 vs 2,571 vehicles/h, Cohen's d = 0.53 | Medium effect: day type matters far more than weather |

These findings were carried forward: Part 2 made the traffic banding data-driven with quartiles, and Part 3 uses it
as the basis of its classification and recommendation work.

## How to review

* Open `sql/metro_traffic.db` in DB Browser for SQLite (or any SQLite client) and run `sql/part1_queries.sql`.
* Re-run the statistics (Python 3.10+ with pandas, numpy and scipy) from `part1_data_analytics/`:

```bash
python statistics/statistical_analysis.py
```

* Open `powerbi/CAPSTONE.pbix` in Power BI Desktop.
