"""
pipeline.py - Entry point for the Metro Interstate Traffic analytics pipeline.

Stages
------
1. Load the raw CSV (with explicit I/O and parsing error handling).
2. Validate the schema BEFORE any other processing.
3. Clean the data, one logged step at a time:
     a. standardise inconsistent categorical values
     b. parse and validate the date_time field
     c. remove duplicate rows / duplicate hourly records
     d. detect impossible values and impute them with the MONTHLY median
     e. detect statistical traffic-volume outliers and impute them with the
        median of their (hour, weekday/weekend) group
     f. final validation of the cleaned dataset
4. Feature engineering        (feature_engineering.py)
5. Visualisation              (visualizations.py)

This is the only pipeline module that configures logging handlers. All other
modules simply call ``logging.getLogger(__name__)``.

Usage
-----
    python pipeline.py                      # normal run (INFO and above)
    python pipeline.py --log-level DEBUG    # include intermediate values
    python pipeline.py --input path/to.csv --log-file logs/other.log
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from feature_engineering import build_features

# ---------------------------------------------------------------------------
# Logger (module-level; handlers are attached in configure_logging())
# ---------------------------------------------------------------------------
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Configuration constants
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent
DEFAULT_INPUT = PROJECT_ROOT / "data" / "raw" / "Metro_Interstate_Traffic_Volume.csv"
DEFAULT_CLEAN_OUTPUT = PROJECT_ROOT / "data" / "processed" / "traffic_clean.csv"
DEFAULT_FEATURES_OUTPUT = PROJECT_ROOT / "data" / "processed" / "traffic_features.csv"
DEFAULT_FIGURES_DIR = PROJECT_ROOT / "figures"
DEFAULT_LOG_FILE = PROJECT_ROOT / "logs" / "pipeline.log"

LOG_FORMAT = "%(asctime)s | %(levelname)-8s | %(module)s | %(message)s"
LOG_DATE_FORMAT = "%Y-%m-%d %H:%M:%S"

EXPECTED_COLUMNS = [
    "holiday",
    "temp",
    "rain_1h",
    "snow_1h",
    "clouds_all",
    "weather_main",
    "weather_description",
    "date_time",
    "traffic_volume",
]
NUMERIC_COLUMNS = ["temp", "rain_1h", "snow_1h", "clouds_all", "traffic_volume"]

DATETIME_FORMAT = "%Y-%m-%d %H:%M:%S"
# Valid observation window for this sensor (dataset documentation: Oct 2012 - Sep 2018)
VALID_DATE_MIN = pd.Timestamp("2012-01-01")
VALID_DATE_MAX = pd.Timestamp("2018-12-31 23:59:59")

# Physical plausibility bounds.
# temp is recorded in Kelvin. 0 K is absolute zero; Minnesota's record low is
# about -51 C (222 K) and record high about 46 C (319 K), so we allow a margin.
TEMP_MIN_K, TEMP_MAX_K = 220.0, 325.0
# The world-record one-hour rainfall is ~305 mm, so any reading above this is a
# sensor/recording error (the raw data contains a 9,831 mm reading).
RAIN_MAX_MM_PER_HOUR = 305.0
SNOW_MAX_MM_PER_HOUR = 100.0
CLOUDS_MIN_PCT, CLOUDS_MAX_PCT = 0, 100
# Physical capacity of the monitored carriageway (vehicles/hour). The busiest
# observed hour is ~7,300, so anything beyond 10,000 is not credible.
TRAFFIC_MAX_PER_HOUR = 10_000
# A count below this fraction of the typical (median) volume for the same hour
# and day type is treated as a sensor outage rather than real traffic.
OUTAGE_RATIO = 0.05

# Canonical spellings for categorical values
WEATHER_MAIN_CANONICAL = {
    "clouds": "Clouds",
    "clear": "Clear",
    "mist": "Mist",
    "rain": "Rain",
    "snow": "Snow",
    "drizzle": "Drizzle",
    "haze": "Haze",
    "thunderstorm": "Thunderstorm",
    "fog": "Fog",
    "smoke": "Smoke",
    "squall": "Squall",
    "squalls": "Squall",
}
NO_HOLIDAY_LABEL = "None"


def display_path(path: Path) -> str:
    """Return a path relative to the project root when possible (keeps logs portable)."""
    path = Path(path).resolve()
    try:
        return path.relative_to(PROJECT_ROOT).as_posix()
    except ValueError:
        return str(path)


class PipelineError(Exception):
    """Raised when the pipeline cannot continue."""


class SchemaValidationError(PipelineError):
    """Raised when the input data does not match the expected schema."""


# ---------------------------------------------------------------------------
# Logging configuration (entry point only)
# ---------------------------------------------------------------------------
def configure_logging(level: str = "INFO", log_file: Path = DEFAULT_LOG_FILE) -> None:
    """Attach a console handler and a file handler to the root logger.

    Handlers are configured here - in the entry-point script only - so every
    module's ``logging.getLogger(__name__)`` logger propagates to both outputs.
    The file is opened in write mode so each run produces a clean, reviewable log.
    """
    numeric_level = getattr(logging, level.upper(), None)
    if not isinstance(numeric_level, int):
        raise ValueError(f"Invalid log level: {level}")

    log_file = Path(log_file)
    log_file.parent.mkdir(parents=True, exist_ok=True)

    formatter = logging.Formatter(LOG_FORMAT, datefmt=LOG_DATE_FORMAT)

    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(formatter)
    console_handler.setLevel(numeric_level)

    file_handler = logging.FileHandler(log_file, mode="w", encoding="utf-8")
    file_handler.setFormatter(formatter)
    file_handler.setLevel(numeric_level)

    root = logging.getLogger()
    root.handlers.clear()
    root.setLevel(numeric_level)
    root.addHandler(console_handler)
    root.addHandler(file_handler)

    # Silence noisy third-party debug output so DEBUG mode shows *our* values.
    for noisy in ("matplotlib", "PIL", "fontTools"):
        logging.getLogger(noisy).setLevel(logging.WARNING)


# ---------------------------------------------------------------------------
# Stage 1 - Load
# ---------------------------------------------------------------------------
def load_raw_data(path: Path) -> pd.DataFrame:
    """Load the raw CSV file, translating I/O and parsing failures into PipelineError."""
    path = Path(path)
    logger.info("Loading raw data from %s", display_path(path))
    try:
        # keep_default_na=False stops pandas silently turning the literal
        # holiday label "None" into NaN; genuine blanks are handled explicitly.
        df = pd.read_csv(path, keep_default_na=False, na_values=[""])
    except FileNotFoundError as exc:
        raise PipelineError(f"Raw data file not found: {display_path(path)}") from exc
    except PermissionError as exc:
        raise PipelineError(f"Permission denied when reading: {path}") from exc
    except pd.errors.EmptyDataError as exc:
        raise PipelineError(f"Raw data file is empty: {path}") from exc
    except pd.errors.ParserError as exc:
        raise PipelineError(f"Could not parse CSV file {path}: {exc}") from exc
    except UnicodeDecodeError as exc:
        raise PipelineError(f"File {path} is not valid UTF-8 text: {exc}") from exc

    logger.info("Raw data loaded successfully: %d rows, %d columns", df.shape[0], df.shape[1])
    return df


# ---------------------------------------------------------------------------
# Stage 2 - Validate schema
# ---------------------------------------------------------------------------
def validate_schema(df: pd.DataFrame) -> pd.DataFrame:
    """Check required columns exist and numeric columns are numeric.

    Raises SchemaValidationError if the data cannot be processed.
    """
    logger.info("Validating schema against %d expected columns", len(EXPECTED_COLUMNS))

    if df.empty:
        raise SchemaValidationError("Dataset contains no rows")

    missing = [col for col in EXPECTED_COLUMNS if col not in df.columns]
    if missing:
        raise SchemaValidationError(f"Missing expected columns: {missing}")

    extra = [col for col in df.columns if col not in EXPECTED_COLUMNS]
    if extra:
        logger.warning("Dropping %d unexpected column(s) not in schema: %s", len(extra), extra)
        df = df[EXPECTED_COLUMNS].copy()

    for col in NUMERIC_COLUMNS:
        if not pd.api.types.is_numeric_dtype(df[col]):
            coerced = pd.to_numeric(df[col], errors="coerce")
            n_bad = int(coerced.isna().sum() - df[col].isna().sum())
            if n_bad == len(df):
                raise SchemaValidationError(f"Column '{col}' contains no numeric values")
            logger.warning(
                "Column '%s' is not numeric: %d value(s) could not be converted and were set to NaN",
                col,
                n_bad,
            )
            df[col] = coerced

    logger.info("Schema validation passed: all %d expected columns present", len(EXPECTED_COLUMNS))
    logger.debug("Column dtypes after schema validation: %s", df.dtypes.astype(str).to_dict())
    return df


# ---------------------------------------------------------------------------
# Stage 3 - Cleaning steps
# ---------------------------------------------------------------------------
def standardise_categoricals(df: pd.DataFrame) -> pd.DataFrame:
    """Trim whitespace and map categorical values to one canonical spelling."""
    df = df.copy()

    # --- holiday: blank / "None" variants -> "None"
    before = df["holiday"].copy()
    holiday = df["holiday"].astype("string").str.strip()
    holiday = holiday.mask(holiday.isna() | holiday.str.lower().isin(["", "none", "nan"]), NO_HOLIDAY_LABEL)
    df["holiday"] = holiday.astype(str)
    n_changed = int((before.astype("string").fillna("<NA>") != df["holiday"]).sum())
    if n_changed:
        logger.warning(
            "Standardised 'holiday': %d row(s) modified (blank/variant 'None' labels unified to '%s')",
            n_changed,
            NO_HOLIDAY_LABEL,
        )
    else:
        logger.info("Standardised 'holiday': no inconsistent labels found")

    # --- weather_main: title case canonical names (e.g. 'SQUALLS' -> 'Squall')
    before = df["weather_main"].copy()
    key = df["weather_main"].astype(str).str.strip().str.lower()
    unknown = sorted(set(key) - set(WEATHER_MAIN_CANONICAL))
    for label in unknown:  # keep unseen categories but make their casing consistent
        logger.warning("Unrecognised weather_main category '%s' kept as title case", label)
    df["weather_main"] = key.map(WEATHER_MAIN_CANONICAL).fillna(key.str.title())
    n_changed = int((before != df["weather_main"]).sum())
    if n_changed:
        logger.warning("Standardised 'weather_main': %d row(s) modified (inconsistent casing/spelling)", n_changed)
    else:
        logger.info("Standardised 'weather_main': no inconsistent labels found")

    # --- weather_description: lower case, single-spaced (e.g. 'Sky is Clear' -> 'sky is clear')
    before = df["weather_description"].copy()
    df["weather_description"] = (
        df["weather_description"].astype(str).str.strip().str.lower().str.replace(r"\s+", " ", regex=True)
    )
    n_changed = int((before != df["weather_description"]).sum())
    if n_changed:
        logger.warning(
            "Standardised 'weather_description': %d row(s) modified (inconsistent casing, e.g. 'Sky is Clear', 'SQUALLS')",
            n_changed,
        )
    else:
        logger.info("Standardised 'weather_description': no inconsistent labels found")

    logger.debug("weather_main categories: %s", sorted(df["weather_main"].unique()))
    return df


def parse_datetimes(df: pd.DataFrame) -> pd.DataFrame:
    """Parse date_time with an explicit format and drop unparseable/out-of-range rows."""
    df = df.copy()
    parsed = pd.to_datetime(df["date_time"], format=DATETIME_FORMAT, errors="coerce")

    invalid_mask = parsed.isna()
    if invalid_mask.any():
        logger.warning(
            "Dropped %d row(s) from date_time parsing: value did not match format '%s'",
            int(invalid_mask.sum()),
            DATETIME_FORMAT,
        )
        logger.debug("Examples of unparseable date_time values: %s", df.loc[invalid_mask, "date_time"].head().tolist())

    out_of_range = ~invalid_mask & ((parsed < VALID_DATE_MIN) | (parsed > VALID_DATE_MAX))
    if out_of_range.any():
        logger.warning(
            "Dropped %d row(s) from date_time validation: timestamp outside valid window %s to %s",
            int(out_of_range.sum()),
            VALID_DATE_MIN.date(),
            VALID_DATE_MAX.date(),
        )

    df["date_time"] = parsed
    df = df.loc[~(invalid_mask | out_of_range)].reset_index(drop=True)
    logger.info(
        "Parsed date_time: %d valid timestamps from %s to %s",
        len(df),
        df["date_time"].min(),
        df["date_time"].max(),
    )
    return df


def remove_duplicates(df: pd.DataFrame) -> pd.DataFrame:
    """Remove exact duplicate rows, then collapse repeated hourly records.

    The source feed emits one row per *weather condition* per hour, so a single
    hour can appear several times with identical traffic_volume but different
    weather labels. For hourly analysis we keep one record per timestamp (the
    first, which is the primary condition) and record how many conditions were
    reported in ``n_weather_conditions``.
    """
    df = df.copy()

    n_before = len(df)
    df = df.drop_duplicates().reset_index(drop=True)
    n_exact = n_before - len(df)
    if n_exact:
        logger.warning("Dropped %d row(s) in duplicate removal: exact duplicate records", n_exact)
    else:
        logger.info("Duplicate removal: no exact duplicate rows found")

    # Check that repeated timestamps never disagree on traffic_volume before collapsing.
    conflicts = df.groupby("date_time")["traffic_volume"].nunique()
    n_conflicts = int((conflicts > 1).sum())
    if n_conflicts:
        logger.warning(
            "%d timestamp(s) have conflicting traffic_volume values; keeping the first record for each",
            n_conflicts,
        )

    counts = df.groupby("date_time")["weather_main"].transform("size")
    df["n_weather_conditions"] = counts.astype(int)
    n_before = len(df)
    df = df.drop_duplicates(subset="date_time", keep="first").sort_values("date_time").reset_index(drop=True)
    n_hourly = n_before - len(df)
    if n_hourly:
        logger.warning(
            "Dropped %d row(s) in duplicate removal: repeated hourly timestamps (multiple weather conditions "
            "reported for the same hour; primary condition kept)",
            n_hourly,
        )
    logger.info("Duplicate removal complete: %d unique hourly records remain", len(df))
    return df


def _impute_with_monthly_median(df: pd.DataFrame, column: str, invalid_mask: pd.Series, reason: str) -> pd.DataFrame:
    """Replace flagged values in ``column`` with the median of *valid* values for the same calendar month.

    Uses an explicit loop over months so each month's median is calculated from
    that month's data only, rather than a single global average.
    """
    n_invalid = int(invalid_mask.sum())
    if n_invalid == 0:
        logger.info("Impossible-value check on '%s': no values outside plausible range", column)
        return df

    df = df.copy()
    df.loc[invalid_mask, column] = np.nan
    months = df["date_time"].dt.month
    global_median = df[column].median()

    imputed_total = 0
    for month in sorted(months[invalid_mask].unique()):
        month_mask = months == month
        month_median = df.loc[month_mask, column].median()  # NaNs (the flagged values) are ignored
        if pd.isna(month_median):
            logger.warning(
                "No valid '%s' values in month %d; falling back to global median %.2f", column, month, global_median
            )
            month_median = global_median
        rows_to_fill = month_mask & invalid_mask
        n_fill = int(rows_to_fill.sum())
        df.loc[rows_to_fill, column] = month_median
        imputed_total += n_fill
        logger.debug("Month %02d median for '%s' = %.3f applied to %d row(s)", month, column, month_median, n_fill)

    logger.warning(
        "Imputed %d row(s) in '%s' with monthly median: %s", imputed_total, column, reason
    )
    return df


def handle_impossible_values(df: pd.DataFrame) -> pd.DataFrame:
    """Detect physically impossible sensor readings and impute or drop them."""
    df = df.copy()

    # Temperature (Kelvin)
    temp_mask = df["temp"].isna() | (df["temp"] < TEMP_MIN_K) | (df["temp"] > TEMP_MAX_K)
    logger.debug("temp readings outside [%.0f K, %.0f K]: %d", TEMP_MIN_K, TEMP_MAX_K, int(temp_mask.sum()))
    df = _impute_with_monthly_median(
        df, "temp", temp_mask,
        f"temperature missing or outside plausible range {TEMP_MIN_K:.0f}-{TEMP_MAX_K:.0f} K (e.g. 0 K sensor faults)",
    )

    # Rainfall (mm/hour)
    rain_mask = df["rain_1h"].isna() | (df["rain_1h"] < 0) | (df["rain_1h"] > RAIN_MAX_MM_PER_HOUR)
    df = _impute_with_monthly_median(
        df, "rain_1h", rain_mask,
        f"rainfall missing, negative or above {RAIN_MAX_MM_PER_HOUR:.0f} mm/h (world-record hourly rainfall)",
    )

    # Snowfall (mm/hour)
    snow_mask = df["snow_1h"].isna() | (df["snow_1h"] < 0) | (df["snow_1h"] > SNOW_MAX_MM_PER_HOUR)
    df = _impute_with_monthly_median(
        df, "snow_1h", snow_mask, f"snowfall missing, negative or above {SNOW_MAX_MM_PER_HOUR:.0f} mm/h"
    )

    # Cloud cover (%)
    cloud_mask = df["clouds_all"].isna() | (df["clouds_all"] < CLOUDS_MIN_PCT) | (df["clouds_all"] > CLOUDS_MAX_PCT)
    df = _impute_with_monthly_median(
        df, "clouds_all", cloud_mask, f"cloud cover missing or outside {CLOUDS_MIN_PCT}-{CLOUDS_MAX_PCT}%"
    )

    # Traffic volume: the target variable. Negative, missing or beyond road
    # capacity cannot be trusted, and imputing the target would fabricate labels,
    # so these rows are dropped rather than imputed.
    bad_traffic = df["traffic_volume"].isna() | (df["traffic_volume"] < 0) | (df["traffic_volume"] > TRAFFIC_MAX_PER_HOUR)
    if bad_traffic.any():
        logger.warning(
            "Dropped %d row(s) in impossible-value check: traffic_volume missing, negative or above %d vehicles/hour",
            int(bad_traffic.sum()),
            TRAFFIC_MAX_PER_HOUR,
        )
        df = df.loc[~bad_traffic].reset_index(drop=True)
    else:
        logger.info("Impossible-value check on 'traffic_volume': no values outside 0-%d", TRAFFIC_MAX_PER_HOUR)

    df["clouds_all"] = df["clouds_all"].round().astype(int)
    df["traffic_volume"] = df["traffic_volume"].astype(int)
    return df


def handle_traffic_outliers(df: pd.DataFrame) -> pd.DataFrame:
    """Detect sensor-outage traffic counts per (hour, day type) and impute with that group's median.

    Traffic volume is strongly bimodal (night vs day), so a global rule cannot
    tell a normal 3 a.m. count from a broken sensor at 8 a.m. Each of the 48
    groups (24 hours x weekday/weekend) therefore gets its own threshold: a count
    below OUTAGE_RATIO of the group median (e.g. 1 vehicle at 09:00 on a weekday)
    is treated as a sensor outage / full closure and imputed.

    Deliberately NOT flagged: genuinely low traffic on holidays or during storms
    (typically 30-70% of normal) and unusually high counts (events). These are
    real signal that the weather and holiday analysis depends on.
    """
    df = df.copy()
    hours = df["date_time"].dt.hour
    is_weekend = df["date_time"].dt.dayofweek >= 5

    # The raw feed only labels the 00:00 record of a holiday; exempt the whole date.
    holiday_dates = set(df.loc[df["holiday"] != NO_HOLIDAY_LABEL, "date_time"].dt.normalize())
    on_holiday = df["date_time"].dt.normalize().isin(holiday_dates)

    outlier_mask = pd.Series(False, index=df.index)
    medians: dict[tuple[int, bool], float] = {}

    for (hour, weekend), group in df.groupby([hours, is_weekend])["traffic_volume"]:
        group_median = float(group.median())
        threshold = OUTAGE_RATIO * group_median
        flagged = (group < threshold) & ~on_holiday.loc[group.index]
        q1, q3 = group.quantile([0.25, 0.75])
        medians[(int(hour), bool(weekend))] = float(group[~flagged].median())
        outlier_mask.loc[group.index] = flagged
        logger.debug(
            "Traffic outage check hour=%02d weekend=%s: median=%.0f Q1=%.0f Q3=%.0f threshold=%.0f flagged=%d",
            hour, weekend, group_median, q1, q3, threshold, int(flagged.sum()),
        )

    df["traffic_volume_imputed"] = outlier_mask
    n_outliers = int(outlier_mask.sum())
    if n_outliers == 0:
        logger.info("Traffic outlier check: no sensor-outage counts detected")
        return df

    for idx in df.index[outlier_mask]:
        key = (int(hours[idx]), bool(is_weekend[idx]))
        df.at[idx, "traffic_volume"] = int(round(medians[key]))

    logger.warning(
        "Imputed %d row(s) in 'traffic_volume' with (hour, weekday/weekend) median: count below %.0f%% of the "
        "typical volume for that hour on a non-holiday (sensor outage or full road closure)",
        n_outliers,
        OUTAGE_RATIO * 100,
    )
    return df


def final_validation(df: pd.DataFrame) -> pd.DataFrame:
    """Confirm the cleaned dataset has no missing values and unique timestamps."""
    n_missing = int(df[EXPECTED_COLUMNS].isna().sum().sum())
    if n_missing:
        raise PipelineError(f"Cleaned data still contains {n_missing} missing value(s)")
    if df["date_time"].duplicated().any():
        raise PipelineError("Cleaned data still contains duplicate timestamps")
    if "traffic_volume_imputed" not in df.columns:
        df = df.assign(traffic_volume_imputed=False)
    logger.info(
        "Final validation passed: %d rows, %d columns, 0 missing values, unique hourly timestamps",
        df.shape[0],
        df.shape[1],
    )
    return df


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """Run every cleaning step in order, each logging its own outcome."""
    logger.info("Cleaning step 1/6: standardising categorical values")
    df = standardise_categoricals(df)
    logger.info("Cleaning step 2/6: parsing and validating date_time")
    df = parse_datetimes(df)
    logger.info("Cleaning step 3/6: removing duplicates")
    df = remove_duplicates(df)
    logger.info("Cleaning step 4/6: handling physically impossible values")
    df = handle_impossible_values(df)
    logger.info("Cleaning step 5/6: handling statistical traffic-volume outliers")
    df = handle_traffic_outliers(df)
    logger.info("Cleaning step 6/6: final validation")
    df = final_validation(df)
    return df


def save_dataframe(df: pd.DataFrame, path: Path, description: str) -> None:
    """Write a DataFrame to CSV, wrapping I/O failures in PipelineError."""
    path = Path(path)
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(path, index=False)
    except OSError as exc:
        raise PipelineError(f"Could not write {description} to {path}: {exc}") from exc
    logger.info("Saved %s (%d rows, %d columns) to %s", description, df.shape[0], df.shape[1], display_path(path))


# ---------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------
def run_pipeline(input_path: Path, clean_output: Path, features_output: Path) -> pd.DataFrame:
    """Run load -> validate -> clean -> feature engineering and persist each output."""
    logger.info("Stage 1/4: load raw data")
    raw = load_raw_data(input_path)
    logger.info("Stage 2/4: validate schema")
    validated = validate_schema(raw)
    logger.info("Stage 3/4: clean data")
    cleaned = clean_data(validated)
    save_dataframe(cleaned, clean_output, "cleaned dataset")
    logger.info("Stage 4/4: feature engineering")
    features = build_features(cleaned)
    save_dataframe(features, features_output, "feature dataset")
    return features


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Metro Interstate Traffic data pipeline")
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT, help="Path to the raw CSV file")
    parser.add_argument("--clean-output", type=Path, default=DEFAULT_CLEAN_OUTPUT, help="Where to write cleaned CSV")
    parser.add_argument(
        "--features-output", type=Path, default=DEFAULT_FEATURES_OUTPUT, help="Where to write the feature CSV"
    )
    parser.add_argument("--log-file", type=Path, default=DEFAULT_LOG_FILE, help="Log file path (default logs/pipeline.log)")
    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Minimum log level; DEBUG shows intermediate values",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    configure_logging(args.log_level, args.log_file)
    logger.info("Pipeline started (log level=%s, log file=%s)", args.log_level, display_path(args.log_file))

    try:
        run_pipeline(args.input, args.clean_output, args.features_output)
    except PipelineError as exc:
        logger.error("Pipeline aborted: %s", exc, exc_info=True)
        return 1
    except Exception:  # last-resort guard so the user never sees an unhandled traceback
        logger.error("Pipeline aborted due to an unexpected error", exc_info=True)
        return 1

    logger.info("Pipeline finished successfully")
    return 0


if __name__ == "__main__":
    sys.exit(main())
