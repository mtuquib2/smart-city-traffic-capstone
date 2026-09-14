"""
traffic_app.py - Mini command-line application for querying the processed traffic data.

Run `python pipeline.py` first to create data/processed/traffic_features.csv.

Commands
--------
    lookup     --datetime "2016-05-10 08:00"         traffic + weather for a specific hour
    peaks      [--top 5] [--day-type workday]        busiest day/hour slots on average
    compare                                          workday vs weekend traffic
    recommend  --day Friday [--start 6 --end 21]     quietest (recommended) hours to travel
    weather    [--condition Snow]                    how weather conditions affect daytime traffic

Examples
--------
    python cli_app/traffic_app.py lookup --datetime "2017-12-25 17:00"
    python cli_app/traffic_app.py peaks --top 3 --day-type weekend
    python cli_app/traffic_app.py recommend --day monday --start 7 --end 19

Output intended for the user is printed to stdout. Diagnostic events are logged
(console via stderr + logs/app.log): the command and arguments at INFO, and a
clear ERROR message - not a raw traceback - for invalid input.
"""

from __future__ import annotations

import argparse
import logging
import sys
from datetime import datetime
from pathlib import Path

import pandas as pd

logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parents[1]  # part2_python/
DEFAULT_DATA = PROJECT_ROOT / "data" / "processed" / "traffic_features.csv"
DEFAULT_LOG_FILE = PROJECT_ROOT / "logs" / "app.log"
LOG_FORMAT = "%(asctime)s | %(levelname)-8s | %(module)s | %(message)s"
LOG_DATE_FORMAT = "%Y-%m-%d %H:%M:%S"

ACCEPTED_DATETIME_FORMATS = ("%Y-%m-%d %H:%M", "%Y-%m-%d %H:%M:%S", "%Y-%m-%d %H")
DAY_NAMES = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
DAY_TYPES = ("all", "workday", "weekend")
DAYTIME_HOURS = range(7, 19)
MAX_LOOKUP_DISTANCE = pd.Timedelta(hours=3)
REQUIRED_COLUMNS = [
    "date_time", "traffic_volume", "congestion_level", "hour", "day_of_week", "day_name",
    "is_weekend", "is_holiday", "holiday", "weather_main", "weather_description", "temp_c",
    "rain_1h", "snow_1h", "clouds_all",
]


class InvalidInputError(ValueError):
    """Raised when the user supplies an argument the app cannot use."""


class DataUnavailableError(RuntimeError):
    """Raised when the processed dataset cannot be loaded."""


class LoggingArgumentParser(argparse.ArgumentParser):
    """ArgumentParser that logs usage errors at ERROR instead of only printing them."""

    def error(self, message: str) -> None:  # type: ignore[override]
        logger.error("Invalid command-line arguments: %s", message)
        self.print_usage(sys.stderr)
        sys.exit(2)


# ---------------------------------------------------------------------------
# Logging configuration (entry point only)
# ---------------------------------------------------------------------------
def configure_logging(level: str = "INFO", log_file: Path = DEFAULT_LOG_FILE) -> None:
    """Console (stderr, so it never mixes with command output on stdout) + appended log file."""
    log_file = Path(log_file)
    log_file.parent.mkdir(parents=True, exist_ok=True)
    formatter = logging.Formatter(LOG_FORMAT, datefmt=LOG_DATE_FORMAT)

    console_handler = logging.StreamHandler(sys.stderr)
    console_handler.setFormatter(formatter)
    file_handler = logging.FileHandler(log_file, mode="a", encoding="utf-8")
    file_handler.setFormatter(formatter)

    root = logging.getLogger()
    root.handlers.clear()
    root.setLevel(getattr(logging, level.upper()))
    root.addHandler(console_handler)
    root.addHandler(file_handler)


# ---------------------------------------------------------------------------
# Data access and validation helpers
# ---------------------------------------------------------------------------
def load_processed_data(path: Path) -> pd.DataFrame:
    path = Path(path)
    try:
        df = pd.read_csv(path, usecols=REQUIRED_COLUMNS, parse_dates=["date_time"], keep_default_na=False, na_values=[""])
    except FileNotFoundError as exc:
        raise DataUnavailableError(
            f"Processed data not found at {path.name}. Run `python pipeline.py` first."
        ) from exc
    except ValueError as exc:  # raised by usecols when expected columns are missing
        raise DataUnavailableError(f"Processed data at {path.name} is missing expected columns: {exc}") from exc
    except (pd.errors.ParserError, pd.errors.EmptyDataError, OSError) as exc:
        raise DataUnavailableError(f"Could not read processed data at {path.name}: {exc}") from exc
    logger.debug("Loaded processed data: %d rows, %d columns", *df.shape)
    return df


def parse_user_datetime(value: str) -> pd.Timestamp:
    text = value.strip()
    for fmt in ACCEPTED_DATETIME_FORMATS:
        try:
            return pd.Timestamp(datetime.strptime(text, fmt))
        except ValueError:
            continue
    raise InvalidInputError(
        f"Malformed date/time '{value}'. Use 'YYYY-MM-DD HH:MM', e.g. '2016-05-10 08:00'."
    )


def parse_day_name(value: str) -> str:
    text = value.strip().lower()
    matches = [day for day in DAY_NAMES if day.lower().startswith(text)] if len(text) >= 2 else []
    if len(matches) != 1:
        raise InvalidInputError(f"Unrecognised day '{value}'. Use a day name such as 'Monday' or 'fri'.")
    return matches[0]


def validate_hour(value: int, name: str) -> int:
    if not 0 <= value <= 23:
        raise InvalidInputError(f"--{name} must be an hour between 0 and 23 (got {value}).")
    return value


def filter_day_type(df: pd.DataFrame, day_type: str) -> pd.DataFrame:
    if day_type == "workday":
        return df[(df["is_weekend"] == 0) & (df["is_holiday"] == 0)]
    if day_type == "weekend":
        return df[df["is_weekend"] == 1]
    return df


def fmt_hour(hour: int) -> str:
    return f"{int(hour):02d}:00-{int(hour):02d}:59"


# ---------------------------------------------------------------------------
# Commands (each returns the text to show the user)
# ---------------------------------------------------------------------------
def cmd_lookup(df: pd.DataFrame, args: argparse.Namespace) -> str:
    target = parse_user_datetime(args.datetime).floor("h")
    data_min, data_max = df["date_time"].min(), df["date_time"].max()
    if not data_min <= target <= data_max:
        raise InvalidInputError(f"{target:%Y-%m-%d %H:%M} is outside the dataset range {data_min:%Y-%m-%d} to {data_max:%Y-%m-%d}.")

    distance = (df["date_time"] - target).abs()
    nearest_idx = distance.idxmin()
    if distance[nearest_idx] > MAX_LOOKUP_DISTANCE:
        raise InvalidInputError(
            f"No record within {MAX_LOOKUP_DISTANCE.components.hours} hours of {target:%Y-%m-%d %H:%M} "
            "(this falls in a sensor data gap)."
        )
    row = df.loc[nearest_idx]
    typical = filter_day_type(df, "weekend" if row["is_weekend"] else "workday")
    typical_volume = typical.loc[typical["hour"] == row["hour"], "traffic_volume"].median()

    lines = []
    if row["date_time"] != target:
        logger.warning("No record for %s; showing nearest available hour %s", target, row["date_time"])
        lines.append(f"(No record for {target:%Y-%m-%d %H:%M}; showing nearest available hour.)")
    diff_pct = (row["traffic_volume"] / typical_volume - 1) * 100
    holiday = ""
    if row["is_holiday"]:
        # The holiday name is only recorded on the 00:00 row of that date.
        same_day = df[(df["date_time"].dt.normalize() == row["date_time"].normalize()) & (df["holiday"] != "None")]
        holiday = f"  [Public holiday: {same_day['holiday'].iloc[0] if len(same_day) else 'yes'}]"
    lines += [
        f"Traffic on {row['date_time']:%A %d %B %Y, %H:%M}{holiday}",
        f"  Volume           : {int(row['traffic_volume']):,} vehicles/hour",
        f"  Congestion level : {row['congestion_level']}",
        f"  Typical for hour : {typical_volume:,.0f} ({diff_pct:+.0f}% vs typical {'weekend' if row['is_weekend'] else 'workday'})",
        f"  Weather          : {row['weather_main']} ({row['weather_description']}), {row['temp_c']:.1f} °C, "
        f"clouds {int(row['clouds_all'])}%, rain {row['rain_1h']:.2f} mm, snow {row['snow_1h']:.2f} mm",
    ]
    return "\n".join(lines)


def cmd_peaks(df: pd.DataFrame, args: argparse.Namespace) -> str:
    if args.top < 1:
        raise InvalidInputError(f"--top must be a positive integer (got {args.top}).")
    subset = filter_day_type(df, args.day_type)
    slots = (
        subset.assign(is_very_high=subset["congestion_level"].eq("Very High"))
        .groupby(["day_of_week", "day_name", "hour"])
        .agg(mean=("traffic_volume", "mean"), very_high_share=("is_very_high", "mean"))
        .reset_index()
    )
    top = slots.sort_values("mean", ascending=False).head(args.top)
    logger.debug("Computed %d day/hour slots for peaks", len(slots))

    scope = {"all": "all days", "workday": "workdays", "weekend": "weekends"}[args.day_type]
    lines = [f"Top {len(top)} high-traffic periods ({scope}, mean vehicles/hour):"]
    for rank, row in enumerate(top.itertuples(), start=1):
        lines.append(
            f"  {rank:>2}. {row.day_name:<9} {fmt_hour(row.hour)}  {row.mean:>6,.0f}  "
            f"(Very High congestion {row.very_high_share:.0%} of the time)"
        )
    return "\n".join(lines)


def cmd_compare(df: pd.DataFrame, args: argparse.Namespace) -> str:
    groups = {"Workdays": filter_day_type(df, "workday"), "Weekends": filter_day_type(df, "weekend")}
    stats = {}
    for name, group in groups.items():
        hourly = group.groupby("hour")["traffic_volume"].mean()
        stats[name] = {
            "Mean vehicles/hour": f"{group['traffic_volume'].mean():,.0f}",
            "Median vehicles/hour": f"{group['traffic_volume'].median():,.0f}",
            "Mean daily vehicles": f"{hourly.sum():,.0f}",
            "Busiest hour": f"{fmt_hour(hourly.idxmax())} ({hourly.max():,.0f})",
            "Quietest hour": f"{fmt_hour(hourly.idxmin())} ({hourly.min():,.0f})",
            "Hours at Very High": f"{group['congestion_level'].eq('Very High').mean():.1%}",
            "Hours in sample": f"{len(group):,}",
        }
    table = pd.DataFrame(stats)
    workday_mean = groups["Workdays"]["traffic_volume"].mean()
    weekend_mean = groups["Weekends"]["traffic_volume"].mean()
    summary = f"Workdays carry {(workday_mean / weekend_mean - 1):.0%} more traffic per hour than weekends on average."
    return "Workday (Mon-Fri, excluding holidays) vs weekend traffic\n\n" + table.to_string() + "\n\n" + summary


def cmd_recommend(df: pd.DataFrame, args: argparse.Namespace) -> str:
    day = parse_day_name(args.day)
    start, end = validate_hour(args.start, "start"), validate_hour(args.end, "end")
    if start > end:
        raise InvalidInputError(f"--start ({start}) must not be later than --end ({end}).")
    if args.top < 1:
        raise InvalidInputError(f"--top must be a positive integer (got {args.top}).")

    subset = df[(df["day_name"] == day) & (df["is_holiday"] == 0) & df["hour"].between(start, end)]
    hourly = subset.groupby("hour").agg(
        mean=("traffic_volume", "mean"),
        typical_level=("congestion_level", lambda s: s.mode().iloc[0]),
    )
    best = hourly.sort_values("mean").head(args.top)
    worst = hourly["mean"].idxmax()
    logger.debug("Recommendation window %s %02d-%02d: %d hours evaluated", day, start, end, len(hourly))

    lines = [f"Recommended travel times on {day} between {start:02d}:00 and {end:02d}:59 (non-holiday):"]
    for rank, (hour, row) in enumerate(best.iterrows(), start=1):
        lines.append(f"  {rank}. {fmt_hour(hour)}  ~{row['mean']:,.0f} vehicles/hour  (usually {row['typical_level']})")
    lines.append(f"Avoid {fmt_hour(worst)}: busiest hour in this window (~{hourly.loc[worst, 'mean']:,.0f} vehicles/hour).")
    return "\n".join(lines)


def cmd_weather(df: pd.DataFrame, args: argparse.Namespace) -> str:
    daytime = filter_day_type(df, "workday")
    daytime = daytime[daytime["hour"].isin(DAYTIME_HOURS)]
    stats = daytime.groupby("weather_main")["traffic_volume"].agg(["mean", "size"])
    baseline = stats.loc["Clear", "mean"]

    if args.condition:
        lookup = {name.lower(): name for name in stats.index}
        condition = lookup.get(args.condition.strip().lower())
        if condition is None:
            raise InvalidInputError(
                f"Unknown weather condition '{args.condition}'. Choose from: {', '.join(sorted(stats.index))}."
            )
        row = stats.loc[condition]
        all_hours = df[df["weather_main"] == condition]
        return "\n".join([
            f"Weather condition: {condition}",
            f"  Workday daytime mean volume : {row['mean']:,.0f} vehicles/hour ({(row['mean'] / baseline - 1):+.1%} vs Clear)",
            f"  Daytime hours observed      : {int(row['size']):,}",
            f"  All hours observed          : {len(all_hours):,}",
            f"  Average temperature         : {all_hours['temp_c'].mean():.1f} °C",
            f"  Share of hours at Very High : {all_hours['congestion_level'].eq('Very High').mean():.1%}",
        ])

    lines = ["Workday daytime (07:00-18:59) traffic by weather condition, vs Clear:"]
    for condition, row in stats.sort_values("mean", ascending=False).iterrows():
        caution = "  (small sample)" if row["size"] < 50 else ""
        lines.append(f"  {condition:<13} {row['mean']:>6,.0f}  {(row['mean'] / baseline - 1):+6.1%}   n={int(row['size']):,}{caution}")
    return "\n".join(lines)


COMMANDS = {
    "lookup": cmd_lookup,
    "peaks": cmd_peaks,
    "compare": cmd_compare,
    "recommend": cmd_recommend,
    "weather": cmd_weather,
}


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def build_parser() -> argparse.ArgumentParser:
    parser = LoggingArgumentParser(
        prog="cli_app/traffic_app.py",
        description="Query the processed Metro Interstate traffic dataset.",
    )
    parser.add_argument("--data", type=Path, default=DEFAULT_DATA, help="Processed feature CSV (default: %(default)s)")
    parser.add_argument("--log-level", default="INFO", choices=["DEBUG", "INFO", "WARNING", "ERROR"])
    parser.add_argument("--log-file", type=Path, default=DEFAULT_LOG_FILE)
    sub = parser.add_subparsers(dest="command", required=True, metavar="COMMAND", parser_class=LoggingArgumentParser)

    p = sub.add_parser("lookup", help="Traffic and weather for a specific date/hour")
    p.add_argument("--datetime", required=True, help="Date and hour, e.g. '2016-05-10 08:00'")

    p = sub.add_parser("peaks", help="Busiest day-of-week/hour slots")
    p.add_argument("--top", type=int, default=5, help="Number of periods to list (default: 5)")
    p.add_argument("--day-type", choices=DAY_TYPES, default="all")

    sub.add_parser("compare", help="Compare workday and weekend traffic")

    p = sub.add_parser("recommend", help="Quietest hours to travel on a given day")
    p.add_argument("--day", required=True, help="Day of week, e.g. 'Monday' or 'fri'")
    p.add_argument("--start", type=int, default=6, help="Earliest hour considered, 0-23 (default: 6)")
    p.add_argument("--end", type=int, default=21, help="Latest hour considered, 0-23 (default: 21)")
    p.add_argument("--top", type=int, default=3, help="Number of recommendations (default: 3)")

    p = sub.add_parser("weather", help="Effect of weather on daytime traffic")
    p.add_argument("--condition", help="Optional single condition, e.g. 'Snow'")
    return parser


def main(argv: list[str] | None = None) -> int:
    argv = sys.argv[1:] if argv is None else argv
    # Configure logging before parsing so argument errors are logged too.
    pre = argparse.ArgumentParser(add_help=False)
    pre.add_argument("--log-level", default="INFO", choices=["DEBUG", "INFO", "WARNING", "ERROR"])
    pre.add_argument("--log-file", type=Path, default=DEFAULT_LOG_FILE)
    early, _ = pre.parse_known_args(argv)
    configure_logging(early.log_level, early.log_file)

    args = build_parser().parse_args(argv)
    command_args = {k: v for k, v in vars(args).items() if k not in {"command", "data", "log_level", "log_file"}}
    logger.info("Command invoked: %s with arguments %s", args.command, command_args)

    try:
        df = load_processed_data(args.data)
        output = COMMANDS[args.command](df, args)
    except InvalidInputError as exc:
        logger.error("Invalid input for '%s': %s", args.command, exc)
        return 2
    except DataUnavailableError as exc:
        logger.error("Cannot run '%s': %s", args.command, exc)
        return 1
    except Exception:  # unexpected bug: record full details in the log, exit cleanly
        logger.error("Unexpected error while running '%s'", args.command, exc_info=True)
        return 1

    print(output)
    logger.info("Command '%s' completed successfully", args.command)
    return 0


if __name__ == "__main__":
    sys.exit(main())
