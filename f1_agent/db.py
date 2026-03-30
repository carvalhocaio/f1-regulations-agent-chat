"""
SQLite database module for historical F1 data (1950-2024).
Loads 14 Kaggle CSVs into a local SQLite database for fast SQL queries.
"""

import csv
import re
import sqlite3
import threading
from pathlib import Path

from f1_agent.env_utils import get_package_dir

DOCS_DIR = Path(__file__).parent.parent / "docs"


def _resolve_db_dir(base_dir: Path) -> Path:
    """Resolve f1_data directory from either flat or nested package layouts."""
    if (base_dir / "f1_history.db").exists():
        return base_dir

    nested_dir = base_dir / "f1_data"
    if (nested_dir / "f1_history.db").exists():
        return nested_dir

    return base_dir


# When deployed, f1_data is an installed package in site-packages.
# Locally, it's a sibling directory of f1_agent.
try:
    import f1_data as _f1_data_pkg

    _db_base_dir = get_package_dir(_f1_data_pkg)
except ImportError:
    _db_base_dir = Path(__file__).parent.parent / "f1_data"

DB_DIR = _resolve_db_dir(_db_base_dir)
DB_PATH = DB_DIR / "f1_history.db"

_connection: sqlite3.Connection | None = None
_connection_lock = threading.RLock()
_query_lock = threading.Lock()

# Discover CSV directory (name contains a newline from Kaggle export)
_CSV_DIR: Path | None = None
for _p in DOCS_DIR.glob("Formula 1*"):
    if _p.is_dir():
        _CSV_DIR = _p
        break

# Schema: table_name -> (CREATE TABLE SQL, CSV filename)
TABLES: dict[str, str] = {
    "circuits": """
        CREATE TABLE IF NOT EXISTS circuits (
            circuitId INTEGER PRIMARY KEY,
            circuitRef TEXT,
            name TEXT,
            location TEXT,
            country TEXT,
            lat REAL,
            lng REAL,
            alt INTEGER,
            url TEXT
        )
    """,
    "constructors": """
        CREATE TABLE IF NOT EXISTS constructors (
            constructorId INTEGER PRIMARY KEY,
            constructorRef TEXT,
            name TEXT,
            nationality TEXT,
            url TEXT
        )
    """,
    "drivers": """
        CREATE TABLE IF NOT EXISTS drivers (
            driverId INTEGER PRIMARY KEY,
            driverRef TEXT,
            number INTEGER,
            code TEXT,
            forename TEXT,
            surname TEXT,
            dob TEXT,
            nationality TEXT,
            url TEXT
        )
    """,
    "seasons": """
        CREATE TABLE IF NOT EXISTS seasons (
            year INTEGER PRIMARY KEY,
            url TEXT
        )
    """,
    "status": """
        CREATE TABLE IF NOT EXISTS status (
            statusId INTEGER PRIMARY KEY,
            status TEXT
        )
    """,
    "races": """
        CREATE TABLE IF NOT EXISTS races (
            raceId INTEGER PRIMARY KEY,
            year INTEGER,
            round INTEGER,
            circuitId INTEGER,
            name TEXT,
            date TEXT,
            time TEXT,
            url TEXT,
            fp1_date TEXT,
            fp1_time TEXT,
            fp2_date TEXT,
            fp2_time TEXT,
            fp3_date TEXT,
            fp3_time TEXT,
            quali_date TEXT,
            quali_time TEXT,
            sprint_date TEXT,
            sprint_time TEXT,
            FOREIGN KEY (circuitId) REFERENCES circuits(circuitId)
        )
    """,
    "results": """
        CREATE TABLE IF NOT EXISTS results (
            resultId INTEGER PRIMARY KEY,
            raceId INTEGER,
            driverId INTEGER,
            constructorId INTEGER,
            number INTEGER,
            grid INTEGER,
            position INTEGER,
            positionText TEXT,
            positionOrder INTEGER,
            points REAL,
            laps INTEGER,
            time TEXT,
            milliseconds INTEGER,
            fastestLap INTEGER,
            rank INTEGER,
            fastestLapTime TEXT,
            fastestLapSpeed TEXT,
            statusId INTEGER,
            FOREIGN KEY (raceId) REFERENCES races(raceId),
            FOREIGN KEY (driverId) REFERENCES drivers(driverId),
            FOREIGN KEY (constructorId) REFERENCES constructors(constructorId),
            FOREIGN KEY (statusId) REFERENCES status(statusId)
        )
    """,
    "qualifying": """
        CREATE TABLE IF NOT EXISTS qualifying (
            qualifyId INTEGER PRIMARY KEY,
            raceId INTEGER,
            driverId INTEGER,
            constructorId INTEGER,
            number INTEGER,
            position INTEGER,
            q1 TEXT,
            q2 TEXT,
            q3 TEXT,
            FOREIGN KEY (raceId) REFERENCES races(raceId),
            FOREIGN KEY (driverId) REFERENCES drivers(driverId),
            FOREIGN KEY (constructorId) REFERENCES constructors(constructorId)
        )
    """,
    "driver_standings": """
        CREATE TABLE IF NOT EXISTS driver_standings (
            driverStandingsId INTEGER PRIMARY KEY,
            raceId INTEGER,
            driverId INTEGER,
            points REAL,
            position INTEGER,
            positionText TEXT,
            wins INTEGER,
            FOREIGN KEY (raceId) REFERENCES races(raceId),
            FOREIGN KEY (driverId) REFERENCES drivers(driverId)
        )
    """,
    "constructor_standings": """
        CREATE TABLE IF NOT EXISTS constructor_standings (
            constructorStandingsId INTEGER PRIMARY KEY,
            raceId INTEGER,
            constructorId INTEGER,
            points REAL,
            position INTEGER,
            positionText TEXT,
            wins INTEGER,
            FOREIGN KEY (raceId) REFERENCES races(raceId),
            FOREIGN KEY (constructorId) REFERENCES constructors(constructorId)
        )
    """,
    "constructor_results": """
        CREATE TABLE IF NOT EXISTS constructor_results (
            constructorResultsId INTEGER PRIMARY KEY,
            raceId INTEGER,
            constructorId INTEGER,
            points REAL,
            status TEXT,
            FOREIGN KEY (raceId) REFERENCES races(raceId),
            FOREIGN KEY (constructorId) REFERENCES constructors(constructorId)
        )
    """,
    "lap_times": """
        CREATE TABLE IF NOT EXISTS lap_times (
            raceId INTEGER,
            driverId INTEGER,
            lap INTEGER,
            position INTEGER,
            time TEXT,
            milliseconds INTEGER,
            PRIMARY KEY (raceId, driverId, lap),
            FOREIGN KEY (raceId) REFERENCES races(raceId),
            FOREIGN KEY (driverId) REFERENCES drivers(driverId)
        )
    """,
    "pit_stops": """
        CREATE TABLE IF NOT EXISTS pit_stops (
            raceId INTEGER,
            driverId INTEGER,
            stop INTEGER,
            lap INTEGER,
            time TEXT,
            duration TEXT,
            milliseconds INTEGER,
            PRIMARY KEY (raceId, driverId, stop),
            FOREIGN KEY (raceId) REFERENCES races(raceId),
            FOREIGN KEY (driverId) REFERENCES drivers(driverId)
        )
    """,
    "sprint_results": """
        CREATE TABLE IF NOT EXISTS sprint_results (
            resultId INTEGER PRIMARY KEY,
            raceId INTEGER,
            driverId INTEGER,
            constructorId INTEGER,
            number INTEGER,
            grid INTEGER,
            position INTEGER,
            positionText TEXT,
            positionOrder INTEGER,
            points REAL,
            laps INTEGER,
            time TEXT,
            milliseconds INTEGER,
            fastestLap INTEGER,
            fastestLapTime TEXT,
            statusId INTEGER,
            FOREIGN KEY (raceId) REFERENCES races(raceId),
            FOREIGN KEY (driverId) REFERENCES drivers(driverId),
            FOREIGN KEY (constructorId) REFERENCES constructors(constructorId)
        )
    """,
}

INDICES = [
    "CREATE INDEX IF NOT EXISTS idx_races_year ON races(year)",
    "CREATE INDEX IF NOT EXISTS idx_results_raceId ON results(raceId)",
    "CREATE INDEX IF NOT EXISTS idx_results_driverId ON results(driverId)",
    "CREATE INDEX IF NOT EXISTS idx_results_constructorId ON results(constructorId)",
    "CREATE INDEX IF NOT EXISTS idx_qualifying_raceId ON qualifying(raceId)",
    "CREATE INDEX IF NOT EXISTS idx_qualifying_driverId ON qualifying(driverId)",
    "CREATE INDEX IF NOT EXISTS idx_driver_standings_raceId ON driver_standings(raceId)",
    "CREATE INDEX IF NOT EXISTS idx_driver_standings_driverId ON driver_standings(driverId)",
    "CREATE INDEX IF NOT EXISTS idx_constructor_standings_raceId ON constructor_standings(raceId)",
    "CREATE INDEX IF NOT EXISTS idx_constructor_standings_constructorId ON constructor_standings(constructorId)",
    "CREATE INDEX IF NOT EXISTS idx_lap_times_raceId ON lap_times(raceId)",
    "CREATE INDEX IF NOT EXISTS idx_lap_times_driverId ON lap_times(driverId)",
    "CREATE INDEX IF NOT EXISTS idx_pit_stops_raceId ON pit_stops(raceId)",
    "CREATE INDEX IF NOT EXISTS idx_sprint_results_raceId ON sprint_results(raceId)",
]


def _clean_value(value: str | None) -> str | None:
    """Convert CSV placeholders to None."""
    if value in ("\\N", "", None):
        return None
    return value


def build_database() -> None:
    """Build the SQLite database from CSV files."""
    global _connection

    with _connection_lock, _query_lock:
        if _CSV_DIR is None:
            raise FileNotFoundError(
                f"No 'Formula 1 World Championship' directory found in {DOCS_DIR}.\n"
                "Place the Kaggle F1 dataset folder inside the docs/ directory."
            )

        if _connection is not None:
            try:
                _connection.close()
            finally:
                _connection = None

        DB_DIR.mkdir(parents=True, exist_ok=True)

        # Remove existing DB to rebuild cleanly
        if DB_PATH.exists():
            DB_PATH.unlink()

        conn = sqlite3.connect(str(DB_PATH))
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("PRAGMA synchronous=NORMAL")

        try:
            # Create all tables
            for create_sql in TABLES.values():
                conn.execute(create_sql)

            # Load data from CSVs
            for table_name in TABLES:
                csv_path = _CSV_DIR / f"{table_name}.csv"
                if not csv_path.exists():
                    continue

                with open(csv_path, newline="", encoding="utf-8") as f:
                    reader = csv.DictReader(f)
                    columns = reader.fieldnames
                    if not columns:
                        continue

                    placeholders = ", ".join("?" for _ in columns)
                    col_names = ", ".join(columns)
                    sql = (
                        f"INSERT INTO {table_name} ({col_names}) VALUES ({placeholders})"
                    )

                    batch: list[tuple[str | None, ...]] = []
                    for row in reader:
                        batch.append(
                            tuple(_clean_value(row.get(col)) for col in columns)
                        )
                        if len(batch) >= 1000:
                            conn.executemany(sql, batch)
                            batch.clear()
                    if batch:
                        conn.executemany(sql, batch)

            # Create indices
            for idx_sql in INDICES:
                conn.execute(idx_sql)

            conn.execute("ANALYZE")
            conn.commit()
        finally:
            conn.close()


def get_connection() -> sqlite3.Connection:
    """Return a read-only connection to the F1 database (lazy singleton)."""
    global _connection
    if _connection is not None:
        return _connection

    with _connection_lock:
        if _connection is not None:
            return _connection

        if not DB_PATH.exists():
            build_database()

        _connection = sqlite3.connect(
            f"file:{DB_PATH}?mode=ro", uri=True, check_same_thread=False
        )
        _connection.row_factory = sqlite3.Row
        _connection.execute("PRAGMA busy_timeout = 5000")
        return _connection


_TRAILING_LIMIT_RE = re.compile(
    r"(?:\bLIMIT\s+\d+\s*(?:OFFSET\s+\d+)?|\bLIMIT\s+\d+\s*,\s*\d+)\s*;?\s*$",
    re.IGNORECASE,
)
MAX_ROWS = 100


def execute_query(sql: str) -> list[dict]:
    """Execute a read-only SQL query and return results as a list of dicts."""
    conn = get_connection()

    # Add LIMIT unless the outer query already has one at the end.
    if not _TRAILING_LIMIT_RE.search(sql):
        sql = sql.rstrip().rstrip(";")
        sql = f"{sql} LIMIT {MAX_ROWS}"

    with _query_lock:
        cursor = conn.execute(sql)
        rows = cursor.fetchmany(MAX_ROWS)
    return [dict(row) for row in rows]
