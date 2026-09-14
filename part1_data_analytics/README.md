# Part 1 – Data Analytics (SQL + Power BI)

Exploratory analysis of the raw Metro Interstate Traffic Volume dataset (48,204 rows).

| Folder | Contents |
|---|---|
| `sql/metro_traffic.db` | SQLite database with the raw table `Metro_Interstate_Traffic_Volume` |
| `sql/part1_queries.sql` | The six SQL analyses, with written interpretations as comments: data completeness, annual trends and year-on-year change, holiday temperatures (New Year's Day, Labor Day), Pearson correlation of temperature vs volume (r ≈ 0.13), and probability/odds analysis of congestion (> 5,500 vehicles/h) vs weather |
| `sql/metro_traffic.sqbpro` | Original DB Browser for SQLite project file |
| `powerbi/CAPSTONE.pbix` | Power BI "Traffic Intelligence" dashboard: daily/hourly trends, weather impact, temperature scatter and KPIs |
| `report/Part 1 Summary.docx` | Insights report for the SmartCity Mobility team |

## Key Part 1 findings carried into Parts 2 and 3

* Clear commuter peaks at 7–9 AM and 4–6 PM, low traffic from midnight to 5 AM.
* Weather and temperature are only weakly related to congestion (odds ratio for clear vs cloudy is 0.74; P(Congestion ∩ Clear) ≈ P(Congestion)·P(Clear)).
* A Low/Medium/High traffic banding was used in the dashboard. Part 2 made this data-driven with quartiles, and Part 3 uses it as the basis of the classification target.

## How to review

* Open `sql/metro_traffic.db` in DB Browser for SQLite (or any SQLite client) and run `sql/part1_queries.sql`.
* Open `powerbi/CAPSTONE.pbix` in Power BI Desktop.
