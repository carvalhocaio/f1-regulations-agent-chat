import unittest

from f1_agent.sql_templates import TEMPLATES, get_template_catalogue, resolve_template


class SqlTemplateResolutionTests(unittest.TestCase):
    def test_resolve_driver_champions_no_filter(self):
        sql = resolve_template("driver_champions")
        self.assertIn("driver_standings", sql)
        self.assertIn("position = 1", sql)
        # No year filter should be applied
        self.assertNotIn("r.year =", sql)

    def test_resolve_driver_champions_with_year(self):
        sql = resolve_template("driver_champions", year=2023)
        self.assertIn("r.year = 2023", sql)

    def test_resolve_driver_champions_with_range(self):
        sql = resolve_template("driver_champions", from_year=2020, to_year=2024)
        self.assertIn("r.year >= 2020", sql)
        self.assertIn("r.year <= 2024", sql)

    def test_resolve_driver_career_stats(self):
        sql = resolve_template("driver_career_stats", driver_name="Hamilton")
        self.assertIn("'Hamilton'", sql)
        self.assertIn("championships", sql)

    def test_resolve_race_results_with_country(self):
        sql = resolve_template(
            "race_results_by_year_country", year=2012, country="Brazil"
        )
        self.assertIn("2012", sql)
        self.assertIn("'Brazil'", sql)

    def test_resolve_most_wins_uses_default_limit(self):
        sql = resolve_template("most_wins_all_time")
        self.assertIn("LIMIT 10", sql)

    def test_resolve_most_wins_custom_limit(self):
        sql = resolve_template("most_wins_all_time", limit=5)
        self.assertIn("LIMIT 5", sql)

    def test_resolve_unknown_template_raises(self):
        with self.assertRaises(KeyError):
            resolve_template("nonexistent_template")

    def test_sql_injection_escaped_in_string_params(self):
        sql = resolve_template("driver_career_stats", driver_name="O'Brien")
        # Single quotes should be escaped
        self.assertIn("O''Brien", sql)

    def test_all_templates_are_select_queries(self):
        for name, tmpl in TEMPLATES.items():
            sql = tmpl["sql"].strip().upper()
            self.assertTrue(
                sql.startswith("SELECT"),
                f"Template '{name}' does not start with SELECT",
            )

    def test_catalogue_lists_all_templates(self):
        catalogue = get_template_catalogue()
        for name in TEMPLATES:
            self.assertIn(name, catalogue)

    def test_head_to_head_requires_all_params(self):
        sql = resolve_template(
            "head_to_head_teammates",
            driver1="Verstappen",
            driver2="Pérez",
            year=2023,
        )
        self.assertIn("'Verstappen'", sql)
        self.assertIn("'Pérez'", sql)
        self.assertIn("2023", sql)


if __name__ == "__main__":
    unittest.main()
