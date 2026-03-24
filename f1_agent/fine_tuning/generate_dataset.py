"""
Auto-generate Q&A training pairs for Gemini fine-tuning.

Uses the actual F1 SQLite database to produce correct answers, paired with
the tool calls the model should learn to make. The output is a JSONL file
ready for Vertex AI supervised fine-tuning.

Usage:
    uv run python -m f1_agent.fine_tuning.generate_dataset [--output dataset.jsonl]
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

from f1_agent import db
from f1_agent.fine_tuning.schema import build_example
from f1_agent.sql_templates import resolve_template

# ── Helpers ──────────────────────────────────────────────────────────────


def _query(sql: str) -> list[dict]:
    """Run a SQL query against the F1 database."""
    return db.execute_query(sql)


def _fmt_driver(row: dict) -> str:
    return row.get(
        "driver", f"{row.get('forename', '')} {row.get('surname', '')}"
    ).strip()


# ── Generators ───────────────────────────────────────────────────────────
# Each generator returns a list of (example_dict) items.


def gen_champion_by_year() -> list[dict]:
    """Generate: 'Who was the F1 world champion in YYYY?'"""
    examples = []
    years = list(range(2000, 2025))
    random.shuffle(years)

    for year in years[:15]:
        sql = resolve_template("driver_champions", year=year)
        rows = _query(sql)
        if not rows:
            continue

        row = rows[0]
        driver = row["driver"]
        constructor = row.get("constructor", "N/A")
        points = row.get("points", "N/A")
        wins = row.get("wins", "N/A")

        # Vary question language
        questions = [
            f"Who was the F1 world champion in {year}?",
            f"Quem foi o campeão mundial de F1 em {year}?",
            f"Who won the {year} Formula 1 championship?",
        ]
        q = random.choice(questions)
        lang = "pt" if "Quem" in q else "en"

        if lang == "pt":
            answer = (
                f"🏆 O campeão mundial de **{year}** foi **{driver}**, "
                f"pela **{constructor}**, com **{points} pontos** e "
                f"**{wins} vitórias** na temporada!\n\n"
                f"---\n**Sources:**\n\n"
                f"📊 *Dados Históricos:*\n"
                f"- **Kaggle — F1 World Championship (1950-2024)**"
            )
        else:
            answer = (
                f"🏆 The **{year}** World Champion was **{driver}**, "
                f"driving for **{constructor}**, with **{points} points** and "
                f"**{wins} wins** in the season!\n\n"
                f"---\n**Sources:**\n\n"
                f"📊 *Historical Data:*\n"
                f"- **Kaggle — F1 World Championship (1950-2024)**"
            )

        examples.append(
            build_example(
                user_message=q,
                function_calls=[
                    {
                        "name": "query_f1_history_template",
                        "args": {
                            "template_name": "driver_champions",
                            "params": json.dumps({"year": year}),
                        },
                    }
                ],
                function_responses=[
                    {
                        "name": "query_f1_history_template",
                        "response": {
                            "status": "success",
                            "results": rows[:1],
                            "row_count": 1,
                        },
                    }
                ],
                model_answer=answer,
            )
        )

    return examples


def gen_driver_career_stats() -> list[dict]:
    """Generate: 'How many wins does X have?'"""
    examples = []
    top_drivers = _query(
        "SELECT d.surname FROM results res "
        "JOIN drivers d ON res.driverId = d.driverId "
        "WHERE res.position = 1 "
        "GROUP BY d.driverId ORDER BY COUNT(*) DESC LIMIT 20"
    )

    for row in top_drivers[:10]:
        surname = row["surname"]
        sql = resolve_template("driver_career_stats", driver_name=surname)
        stats = _query(sql)
        if not stats:
            continue

        s = stats[0]
        questions = [
            f"How many wins does {surname} have?",
            f"What are {surname}'s career stats?",
            f"Quantas vitórias {surname} tem?",
        ]
        q = random.choice(questions)
        lang = "pt" if "Quantas" in q else "en"

        if lang == "pt":
            answer = (
                f"📊 **{s['driver']}** ({s.get('nationality', 'N/A')})\n\n"
                f"- 🏁 **{s['races']}** corridas\n"
                f"- 🏆 **{s['wins']}** vitórias\n"
                f"- 🥇 **{s['podiums']}** pódios\n"
                f"- ⚡ **{s['poles']}** pole positions\n"
                f"- 🏆 **{s['championships']}** campeonatos\n"
                f"- 📊 **{s['total_points']}** pontos\n\n"
                f"---\n**Sources:**\n\n"
                f"📊 *Dados Históricos:*\n"
                f"- **Kaggle — F1 World Championship (1950-2024)**"
            )
        else:
            answer = (
                f"📊 **{s['driver']}** ({s.get('nationality', 'N/A')})\n\n"
                f"- 🏁 **{s['races']}** races\n"
                f"- 🏆 **{s['wins']}** wins\n"
                f"- 🥇 **{s['podiums']}** podiums\n"
                f"- ⚡ **{s['poles']}** pole positions\n"
                f"- 🏆 **{s['championships']}** championships\n"
                f"- 📊 **{s['total_points']}** total points\n\n"
                f"---\n**Sources:**\n\n"
                f"📊 *Historical Data:*\n"
                f"- **Kaggle — F1 World Championship (1950-2024)**"
            )

        examples.append(
            build_example(
                user_message=q,
                function_calls=[
                    {
                        "name": "query_f1_history_template",
                        "args": {
                            "template_name": "driver_career_stats",
                            "params": json.dumps({"driver_name": surname}),
                        },
                    }
                ],
                function_responses=[
                    {
                        "name": "query_f1_history_template",
                        "response": {
                            "status": "success",
                            "results": stats[:1],
                            "row_count": 1,
                        },
                    }
                ],
                model_answer=answer,
            )
        )

    return examples


def gen_race_result_by_country() -> list[dict]:
    """Generate: 'Who won the X GP in YYYY?'"""
    examples = []
    gps = [
        ("Brazil", "GP do Brasil", "Brazilian GP"),
        ("Italy", "GP da Itália", "Italian GP"),
        ("United Kingdom", "GP da Grã-Bretanha", "British GP"),
        ("Monaco", "GP de Mônaco", "Monaco GP"),
        ("Japan", "GP do Japão", "Japanese GP"),
        ("Australia", "GP da Austrália", "Australian GP"),
        ("USA", "GP dos EUA", "US GP"),
        ("Spain", "GP da Espanha", "Spanish GP"),
    ]

    for country, pt_name, en_name in gps:
        year = random.randint(2010, 2024)
        sql = resolve_template(
            "race_results_by_year_country", year=year, country=country
        )
        rows = _query(sql)
        if not rows:
            continue

        questions = [
            f"Who won the {en_name} in {year}?",
            f"Quem ganhou o {pt_name} em {year}?",
        ]
        q = random.choice(questions)
        lang = "pt" if "Quem" in q else "en"

        # Build top-3 podium
        podium_lines = []
        for i, r in enumerate(rows[:3]):
            medal = ["🥇", "🥈", "🥉"][i]
            driver = r.get(
                "driver", f"{r.get('forename', '')} {r.get('surname', '')}"
            ).strip()
            constructor = r.get("constructor", "N/A")
            podium_lines.append(f"{medal} **{driver}** ({constructor})")

        podium = "\n".join(podium_lines)

        if lang == "pt":
            answer = (
                f"🏁 **{pt_name} {year}**\n\n"
                f"{podium}\n\n"
                f"---\n**Sources:**\n\n"
                f"📊 *Dados Históricos:*\n"
                f"- **Kaggle — F1 World Championship (1950-2024)**"
            )
        else:
            answer = (
                f"🏁 **{en_name} {year}**\n\n"
                f"{podium}\n\n"
                f"---\n**Sources:**\n\n"
                f"📊 *Historical Data:*\n"
                f"- **Kaggle — F1 World Championship (1950-2024)**"
            )

        examples.append(
            build_example(
                user_message=q,
                function_calls=[
                    {
                        "name": "query_f1_history_template",
                        "args": {
                            "template_name": "race_results_by_year_country",
                            "params": json.dumps({"year": year, "country": country}),
                        },
                    }
                ],
                function_responses=[
                    {
                        "name": "query_f1_history_template",
                        "response": {
                            "status": "success",
                            "results": rows[:3],
                            "row_count": min(len(rows), 10),
                        },
                    }
                ],
                model_answer=answer,
            )
        )

    return examples


def gen_records() -> list[dict]:
    """Generate: 'Who has the most wins/poles/podiums in F1?'"""
    examples = []
    templates = [
        ("most_wins_all_time", "wins", "race wins", "vitórias"),
        ("most_poles_all_time", "poles", "pole positions", "pole positions"),
        ("most_podiums_all_time", "podiums", "podium finishes", "pódios"),
    ]

    for tmpl, stat_col, en_label, pt_label in templates:
        sql = resolve_template(tmpl, limit=10)
        rows = _query(sql)
        if not rows:
            continue

        questions = [
            f"Who has the most {en_label} in F1 history?",
            f"Top 10 drivers with most {en_label}",
            f"Quem tem mais {pt_label} na história da F1?",
        ]
        q = random.choice(questions)

        lines = []
        for i, r in enumerate(rows, 1):
            driver = r["driver"]
            count = r[stat_col]
            lines.append(f"{i}. **{driver}** — {count}")

        listing = "\n".join(lines)

        answer = (
            f"🏆 **Top 10 — {en_label.title()}**\n\n"
            f"{listing}\n\n"
            f"---\n**Sources:**\n\n"
            f"📊 *Historical Data:*\n"
            f"- **Kaggle — F1 World Championship (1950-2024)**"
        )

        examples.append(
            build_example(
                user_message=q,
                function_calls=[
                    {
                        "name": "query_f1_history_template",
                        "args": {
                            "template_name": tmpl,
                            "params": json.dumps({"limit": 10}),
                        },
                    }
                ],
                function_responses=[
                    {
                        "name": "query_f1_history_template",
                        "response": {
                            "status": "success",
                            "results": rows,
                            "row_count": len(rows),
                        },
                    }
                ],
                model_answer=answer,
            )
        )

    return examples


def gen_season_standings() -> list[dict]:
    """Generate: 'What were the final standings in YYYY?'"""
    examples = []
    years = list(range(2015, 2025))
    random.shuffle(years)

    for year in years[:5]:
        sql = resolve_template("season_standings_final", year=year)
        rows = _query(sql)
        if not rows:
            continue

        questions = [
            f"What were the final F1 standings in {year}?",
            f"Classificação final da F1 em {year}",
        ]
        q = random.choice(questions)

        lines = []
        for r in rows[:10]:
            pos = r["position"]
            driver = r["driver"]
            pts = r["points"]
            lines.append(f"{pos}. **{driver}** — {pts} pts")

        listing = "\n".join(lines)
        answer = (
            f"📊 **{year} Final Standings (Top 10)**\n\n"
            f"{listing}\n\n"
            f"---\n**Sources:**\n\n"
            f"📊 *Historical Data:*\n"
            f"- **Kaggle — F1 World Championship (1950-2024)**"
        )

        examples.append(
            build_example(
                user_message=q,
                function_calls=[
                    {
                        "name": "query_f1_history_template",
                        "args": {
                            "template_name": "season_standings_final",
                            "params": json.dumps({"year": year}),
                        },
                    }
                ],
                function_responses=[
                    {
                        "name": "query_f1_history_template",
                        "response": {
                            "status": "success",
                            "results": rows[:10],
                            "row_count": min(len(rows), 10),
                        },
                    }
                ],
                model_answer=answer,
            )
        )

    return examples


def gen_regulation_lookups() -> list[dict]:
    """Generate regulation search examples (no actual DB needed — synthetic)."""
    examples = []
    queries = [
        (
            "What are the 2026 power unit regulations?",
            "power unit technical specifications",
        ),
        ("What is the cost cap for 2026?", "cost cap financial regulations"),
        ("What are the tire rules for 2026?", "tire regulations specifications"),
        ("What are the safety car rules?", "safety car procedures regulations"),
        ("How does the sprint format work in 2026?", "sprint race format regulations"),
        ("What are the bodywork dimensions?", "bodywork dimensions specifications"),
        ("What are the fuel regulations?", "fuel specifications regulations"),
        ("What are the penalty rules?", "penalty grid sporting regulations"),
        ("Quais são as regras do DRS em 2026?", "DRS drag reduction system"),
        ("Qual é o peso mínimo do carro?", "minimum weight car technical"),
    ]

    for question, search_query in queries:
        # Synthetic response (the actual content comes from the RAG)
        answer = (
            "🔧 Based on the FIA 2026 regulations:\n\n"
            "[Regulation content would be retrieved from the vector store]\n\n"
            "---\n**Sources:**\n\n"
            "📚 *Regulations:*\n"
            "- **Section C — Technical** (relevant article)"
        )

        examples.append(
            build_example(
                user_message=question,
                function_calls=[
                    {
                        "name": "search_regulations",
                        "args": {"query": search_query},
                    }
                ],
                function_responses=[
                    {
                        "name": "search_regulations",
                        "response": {
                            "status": "success",
                            "results": [
                                {
                                    "content": "...",
                                    "section": "Section C — Technical",
                                    "page": 45,
                                }
                            ],
                        },
                    }
                ],
                model_answer=answer,
            )
        )

    return examples


def gen_current_season() -> list[dict]:
    """Generate examples that should use google_search_agent."""
    examples = []
    questions = [
        (
            "Who is leading the 2026 championship?",
            "F1 2026 drivers championship standings",
        ),
        ("When is the next F1 race?", "F1 2026 next race schedule"),
        ("Quem está liderando o campeonato?", "F1 2026 drivers championship standings"),
        ("What happened in the last race?", "F1 2026 latest race results"),
        ("Qual é o calendário da F1 2026?", "F1 2026 race calendar schedule"),
    ]

    for question, search_request in questions:
        answer = (
            "🏁 [Current season information from web search]\n\n"
            "---\n**Sources:**\n\n"
            "🌐 *Web:*\n"
            "- **formula1.com**: Current season information"
        )

        examples.append(
            build_example(
                user_message=question,
                function_calls=[
                    {
                        "name": "google_search_agent",
                        "args": {"request": search_request},
                    }
                ],
                function_responses=[
                    {
                        "name": "google_search_agent",
                        "response": {"status": "success", "results": "..."},
                    }
                ],
                model_answer=answer,
            )
        )

    return examples


def gen_temporal_reasoning() -> list[dict]:
    """Generate examples requiring both DB + web (temporal split)."""
    examples = []

    # "Last 5 champions" in 2026 → 2022-2024 from DB + 2025-2026 from web
    for n, from_year in [(5, 2022), (10, 2017)]:
        sql = resolve_template("driver_champions", from_year=from_year, to_year=2024)
        rows = _query(sql)
        if not rows:
            continue

        questions = [
            f"Who were the last {n} F1 world champions?",
            f"Últimos {n} campeões mundiais de F1",
        ]
        q = random.choice(questions)

        db_lines = []
        for r in rows:
            db_lines.append(
                f"- **{r['year']}**: {r['driver']} ({r.get('constructor', 'N/A')})"
            )

        answer = (
            f"🏆 **Last {n} F1 World Champions**\n\n"
            f"From historical database:\n"
            + "\n".join(db_lines)
            + "\n\nFrom web search:\n"
            "- **2025**: [from google_search_agent]\n"
            "- **2026**: [from google_search_agent]\n\n"
            "---\n**Sources:**\n\n"
            "📊 *Historical Data:*\n"
            "- **Kaggle — F1 World Championship (1950-2024)**\n\n"
            "🌐 *Web:*\n"
            "- **formula1.com**: 2025-2026 champions"
        )

        examples.append(
            build_example(
                user_message=q,
                function_calls=[
                    {
                        "name": "query_f1_history_template",
                        "args": {
                            "template_name": "driver_champions",
                            "params": json.dumps(
                                {"from_year": from_year, "to_year": 2024}
                            ),
                        },
                    },
                    {
                        "name": "google_search_agent",
                        "args": {"request": "F1 world drivers champion 2025 and 2026"},
                    },
                ],
                function_responses=[
                    {
                        "name": "query_f1_history_template",
                        "response": {
                            "status": "success",
                            "results": rows,
                            "row_count": len(rows),
                        },
                    },
                    {
                        "name": "google_search_agent",
                        "response": {"status": "success", "results": "..."},
                    },
                ],
                model_answer=answer,
            )
        )

    return examples


# ── Main ─────────────────────────────────────────────────────────────────

ALL_GENERATORS = [
    gen_champion_by_year,
    gen_driver_career_stats,
    gen_race_result_by_country,
    gen_records,
    gen_season_standings,
    gen_regulation_lookups,
    gen_current_season,
    gen_temporal_reasoning,
]


def generate_all(seed: int = 42) -> list[dict]:
    """Run all generators and return the combined list of examples."""
    random.seed(seed)
    examples = []
    for gen in ALL_GENERATORS:
        examples.extend(gen())
    random.shuffle(examples)
    return examples


def split_dataset(
    examples: list[dict], test_ratio: float = 0.2
) -> tuple[list[dict], list[dict]]:
    """Split examples into train and test sets."""
    split_idx = int(len(examples) * (1 - test_ratio))
    return examples[:split_idx], examples[split_idx:]


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate F1 fine-tuning dataset")
    parser.add_argument(
        "--output",
        type=str,
        default="f1_agent/fine_tuning/dataset.jsonl",
        help="Output JSONL file path",
    )
    parser.add_argument(
        "--seed", type=int, default=42, help="Random seed for reproducibility"
    )
    args = parser.parse_args()

    examples = generate_all(seed=args.seed)
    train, test = split_dataset(examples)

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)

    train_path = output.with_suffix(".train.jsonl")
    test_path = output.with_suffix(".test.jsonl")

    with open(train_path, "w", encoding="utf-8") as f:
        for ex in train:
            f.write(json.dumps(ex, ensure_ascii=False) + "\n")

    with open(test_path, "w", encoding="utf-8") as f:
        for ex in test:
            f.write(json.dumps(ex, ensure_ascii=False) + "\n")

    print(f"Generated {len(examples)} examples total")
    print(f"  Train: {len(train)} → {train_path}")
    print(f"  Test:  {len(test)} → {test_path}")


if __name__ == "__main__":
    main()
