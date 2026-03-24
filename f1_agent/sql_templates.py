"""
Pre-computed SQL templates for the most common F1 historical queries.

Each template is a parameterized SQL string that eliminates LLM SQL-generation
errors for predictable question patterns.  The agent receives the template
catalogue in its instruction and can call `query_f1_history_template` with a
template name + parameters instead of writing raw SQL.
"""

TEMPLATES: dict[str, dict] = {
    # ── Champions ───────────────────────────────────────────────────────
    "driver_champions": {
        "description": "World Drivers' Champions. Optional: year, from_year, to_year.",
        "sql": """
            SELECT d.forename || ' ' || d.surname AS driver,
                   r.year,
                   c.name AS constructor,
                   ds.points,
                   ds.wins
            FROM driver_standings ds
            JOIN (SELECT year, MAX(raceId) AS lastRace FROM races GROUP BY year) lr
              ON ds.raceId = lr.lastRace
            JOIN drivers d ON ds.driverId = d.driverId
            JOIN races r ON ds.raceId = r.raceId
            LEFT JOIN results res
              ON res.raceId = r.raceId AND res.driverId = d.driverId
            LEFT JOIN constructors c ON res.constructorId = c.constructorId
            WHERE ds.position = 1 {year_filter}
            ORDER BY r.year DESC
        """,
        "params": {
            "year": "r.year = {value}",
            "from_year": "r.year >= {value}",
            "to_year": "r.year <= {value}",
        },
    },
    "constructor_champions": {
        "description": "World Constructors' Champions. Optional: year, from_year, to_year.",
        "sql": """
            SELECT c.name AS constructor,
                   r.year,
                   cs.points,
                   cs.wins
            FROM constructor_standings cs
            JOIN (SELECT year, MAX(raceId) AS lastRace FROM races GROUP BY year) lr
              ON cs.raceId = lr.lastRace
            JOIN constructors c ON cs.constructorId = c.constructorId
            JOIN races r ON cs.raceId = r.raceId
            WHERE cs.position = 1 {year_filter}
            ORDER BY r.year DESC
        """,
        "params": {
            "year": "r.year = {value}",
            "from_year": "r.year >= {value}",
            "to_year": "r.year <= {value}",
        },
    },
    # ── Race results ────────────────────────────────────────────────────
    "race_winners_by_country": {
        "description": "Race winners for a given country/circuit. Required: country.",
        "sql": """
            SELECT d.forename || ' ' || d.surname AS driver,
                   c.name AS constructor,
                   r.year,
                   r.name AS race_name,
                   ci.name AS circuit
            FROM results res
            JOIN races r ON res.raceId = r.raceId
            JOIN circuits ci ON r.circuitId = ci.circuitId
            JOIN drivers d ON res.driverId = d.driverId
            JOIN constructors c ON res.constructorId = c.constructorId
            WHERE res.position = 1
              AND ci.country LIKE '%' || :country || '%'
            ORDER BY r.year DESC
        """,
        "params": {"country": "literal"},
    },
    "race_results_by_year_country": {
        "description": "Full race result (top 10) for a GP in a given year/country. Required: year, country.",
        "sql": """
            SELECT res.position,
                   d.forename || ' ' || d.surname AS driver,
                   c.name AS constructor,
                   res.points,
                   res.laps,
                   res.time,
                   s.status
            FROM results res
            JOIN races r ON res.raceId = r.raceId
            JOIN circuits ci ON r.circuitId = ci.circuitId
            JOIN drivers d ON res.driverId = d.driverId
            JOIN constructors c ON res.constructorId = c.constructorId
            JOIN status s ON res.statusId = s.statusId
            WHERE r.year = :year
              AND ci.country LIKE '%' || :country || '%'
            ORDER BY res.positionOrder ASC
            LIMIT 10
        """,
        "params": {"year": "literal", "country": "literal"},
    },
    # ── Driver stats ────────────────────────────────────────────────────
    "driver_career_stats": {
        "description": "Career stats for a driver: wins, podiums, poles, championships. Required: driver_name.",
        "sql": """
            SELECT d.forename || ' ' || d.surname AS driver,
                   d.nationality,
                   COUNT(DISTINCT res.raceId) AS races,
                   SUM(CASE WHEN res.position = 1 THEN 1 ELSE 0 END) AS wins,
                   SUM(CASE WHEN res.position IN (1, 2, 3) THEN 1 ELSE 0 END) AS podiums,
                   SUM(CASE WHEN res.grid = 1 THEN 1 ELSE 0 END) AS poles,
                   SUM(res.points) AS total_points,
                   (SELECT COUNT(*) FROM driver_standings ds2
                    JOIN (SELECT year, MAX(raceId) AS lr FROM races GROUP BY year) lr2
                      ON ds2.raceId = lr2.lr
                    WHERE ds2.driverId = d.driverId AND ds2.position = 1
                   ) AS championships
            FROM results res
            JOIN drivers d ON res.driverId = d.driverId
            WHERE d.surname LIKE '%' || :driver_name || '%'
            GROUP BY d.driverId
        """,
        "params": {"driver_name": "literal"},
    },
    "driver_season_results": {
        "description": "All race results for a driver in a season. Required: driver_name, year.",
        "sql": """
            SELECT r.round,
                   r.name AS race,
                   res.grid,
                   res.position,
                   res.points,
                   s.status
            FROM results res
            JOIN races r ON res.raceId = r.raceId
            JOIN drivers d ON res.driverId = d.driverId
            JOIN status s ON res.statusId = s.statusId
            WHERE d.surname LIKE '%' || :driver_name || '%'
              AND r.year = :year
            ORDER BY r.round
        """,
        "params": {"driver_name": "literal", "year": "literal"},
    },
    # ── Comparisons ─────────────────────────────────────────────────────
    "head_to_head_teammates": {
        "description": "Head-to-head race finishes between two drivers at the same constructor in a year. Required: driver1, driver2, year.",
        "sql": """
            SELECT r.round,
                   r.name AS race,
                   d1.forename || ' ' || d1.surname AS driver1,
                   r1.position AS pos1,
                   r1.points AS pts1,
                   d2.forename || ' ' || d2.surname AS driver2,
                   r2.position AS pos2,
                   r2.points AS pts2
            FROM results r1
            JOIN results r2 ON r1.raceId = r2.raceId
                            AND r1.constructorId = r2.constructorId
                            AND r1.driverId != r2.driverId
            JOIN races r ON r1.raceId = r.raceId
            JOIN drivers d1 ON r1.driverId = d1.driverId
            JOIN drivers d2 ON r2.driverId = d2.driverId
            WHERE d1.surname LIKE '%' || :driver1 || '%'
              AND d2.surname LIKE '%' || :driver2 || '%'
              AND r.year = :year
            ORDER BY r.round
        """,
        "params": {"driver1": "literal", "driver2": "literal", "year": "literal"},
    },
    # ── Records ─────────────────────────────────────────────────────────
    "most_wins_all_time": {
        "description": "Top N drivers by race wins all-time. Optional: limit (default 10).",
        "sql": """
            SELECT d.forename || ' ' || d.surname AS driver,
                   d.nationality,
                   COUNT(*) AS wins
            FROM results res
            JOIN drivers d ON res.driverId = d.driverId
            WHERE res.position = 1
            GROUP BY d.driverId
            ORDER BY wins DESC
            LIMIT :limit
        """,
        "params": {"limit": "literal"},
        "defaults": {"limit": 10},
    },
    "most_poles_all_time": {
        "description": "Top N drivers by pole positions all-time. Optional: limit (default 10).",
        "sql": """
            SELECT d.forename || ' ' || d.surname AS driver,
                   d.nationality,
                   COUNT(*) AS poles
            FROM results res
            JOIN drivers d ON res.driverId = d.driverId
            WHERE res.grid = 1
            GROUP BY d.driverId
            ORDER BY poles DESC
            LIMIT :limit
        """,
        "params": {"limit": "literal"},
        "defaults": {"limit": 10},
    },
    "most_podiums_all_time": {
        "description": "Top N drivers by podium finishes all-time. Optional: limit (default 10).",
        "sql": """
            SELECT d.forename || ' ' || d.surname AS driver,
                   d.nationality,
                   COUNT(*) AS podiums
            FROM results res
            JOIN drivers d ON res.driverId = d.driverId
            WHERE res.position IN (1, 2, 3)
            GROUP BY d.driverId
            ORDER BY podiums DESC
            LIMIT :limit
        """,
        "params": {"limit": "literal"},
        "defaults": {"limit": 10},
    },
    "most_constructor_wins": {
        "description": "Top N constructors by race wins all-time. Optional: limit (default 10).",
        "sql": """
            SELECT c.name AS constructor,
                   c.nationality,
                   COUNT(*) AS wins
            FROM results res
            JOIN constructors c ON res.constructorId = c.constructorId
            WHERE res.position = 1
            GROUP BY c.constructorId
            ORDER BY wins DESC
            LIMIT :limit
        """,
        "params": {"limit": "literal"},
        "defaults": {"limit": 10},
    },
    # ── Season overview ─────────────────────────────────────────────────
    "season_calendar": {
        "description": "Race calendar for a season. Required: year.",
        "sql": """
            SELECT r.round,
                   r.name AS race,
                   ci.name AS circuit,
                   ci.country,
                   r.date
            FROM races r
            JOIN circuits ci ON r.circuitId = ci.circuitId
            WHERE r.year = :year
            ORDER BY r.round
        """,
        "params": {"year": "literal"},
    },
    "season_standings_final": {
        "description": "Final driver standings for a season. Required: year.",
        "sql": """
            SELECT ds.position,
                   d.forename || ' ' || d.surname AS driver,
                   ds.points,
                   ds.wins
            FROM driver_standings ds
            JOIN (SELECT MAX(raceId) AS lastRace FROM races WHERE year = :year) lr
              ON ds.raceId = lr.lastRace
            JOIN drivers d ON ds.driverId = d.driverId
            ORDER BY ds.position
        """,
        "params": {"year": "literal"},
    },
    # ── Pit stops & lap times ───────────────────────────────────────────
    "fastest_pit_stops_race": {
        "description": "Fastest pit stops in a race. Required: year, country. Optional: limit (default 10).",
        "sql": """
            SELECT ps.stop,
                   ps.lap,
                   d.forename || ' ' || d.surname AS driver,
                   ps.duration,
                   ps.milliseconds
            FROM pit_stops ps
            JOIN races r ON ps.raceId = r.raceId
            JOIN circuits ci ON r.circuitId = ci.circuitId
            JOIN drivers d ON ps.driverId = d.driverId
            WHERE r.year = :year
              AND ci.country LIKE '%' || :country || '%'
            ORDER BY ps.milliseconds ASC
            LIMIT :limit
        """,
        "params": {"year": "literal", "country": "literal", "limit": "literal"},
        "defaults": {"limit": 10},
    },
    "fastest_laps_race": {
        "description": "Fastest laps in a race. Required: year, country. Optional: limit (default 10).",
        "sql": """
            SELECT lt.lap,
                   d.forename || ' ' || d.surname AS driver,
                   lt.time,
                   lt.milliseconds
            FROM lap_times lt
            JOIN races r ON lt.raceId = r.raceId
            JOIN circuits ci ON r.circuitId = ci.circuitId
            JOIN drivers d ON lt.driverId = d.driverId
            WHERE r.year = :year
              AND ci.country LIKE '%' || :country || '%'
            ORDER BY lt.milliseconds ASC
            LIMIT :limit
        """,
        "params": {"year": "literal", "country": "literal", "limit": "literal"},
        "defaults": {"limit": 10},
    },
}


def get_template_catalogue() -> str:
    """Return a human-readable catalogue for the agent instruction."""
    lines = ["Available SQL templates for query_f1_history_template:\n"]
    for name, t in TEMPLATES.items():
        params = ", ".join(t["params"].keys())
        defaults = t.get("defaults", {})
        default_info = f" (defaults: {defaults})" if defaults else ""
        lines.append(f"- **{name}**({params}){default_info}: {t['description']}")
    return "\n".join(lines)


def resolve_template(template_name: str, **kwargs) -> str:
    """Resolve a template name + parameters into a ready-to-execute SQL string.

    For templates with a ``{year_filter}`` placeholder, the ``year``,
    ``from_year``, and ``to_year`` keyword arguments are converted into SQL
    ``AND`` clauses.  All other parameters use SQLite named-parameter syntax
    (``:``) and are returned as literal values embedded in the query.

    Returns the final SQL string ready for ``db.execute_query()``.

    Raises:
        KeyError: If *template_name* is not found.
        ValueError: If a required parameter is missing.
    """
    if template_name not in TEMPLATES:
        raise KeyError(
            f"Unknown template '{template_name}'. "
            f"Valid templates: {', '.join(TEMPLATES.keys())}"
        )

    tmpl = TEMPLATES[template_name]
    sql = tmpl["sql"]
    defaults = tmpl.get("defaults", {})

    # Merge defaults with provided kwargs
    merged = {**defaults, **kwargs}

    # Handle {year_filter} placeholder (used in champion templates)
    if "{year_filter}" in sql:
        filters: list[str] = []
        for key in ("year", "from_year", "to_year"):
            if key in merged and merged[key] is not None:
                pattern = tmpl["params"][key]
                filters.append("AND " + pattern.format(value=int(merged[key])))
        sql = sql.replace("{year_filter}", " ".join(filters))

    # Handle :named parameters (literal substitution)
    for param_name, param_type in tmpl["params"].items():
        if param_type != "literal":
            continue
        placeholder = f":{param_name}"
        if placeholder in sql:
            if param_name in merged and merged[param_name] is not None:
                value = merged[param_name]
                if isinstance(value, (int, float)):
                    sql = sql.replace(placeholder, str(value))
                else:
                    # Escape single quotes in string values
                    safe_value = str(value).replace("'", "''")
                    sql = sql.replace(placeholder, f"'{safe_value}'")
            elif param_name in tmpl.get("params", {}):
                raise ValueError(
                    f"Required parameter '{param_name}' not provided "
                    f"for template '{template_name}'."
                )

    return sql
