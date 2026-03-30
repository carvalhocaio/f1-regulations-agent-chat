import sqlite3
import tempfile
import threading
import time
import unittest
from pathlib import Path
from unittest.mock import patch

import f1_agent.db as db


class _FakeCursor:
    def __init__(self, rows):
        self._rows = rows
        self.fetchmany_size = None

    def fetchmany(self, size):
        self.fetchmany_size = size
        return self._rows[:size]


class _FakeConn:
    def __init__(self, rows):
        self.rows = rows
        self.executed_sql = None
        self.cursor = _FakeCursor(rows)

    def execute(self, sql):
        self.executed_sql = sql
        return self.cursor


class DbQueryExecutionTests(unittest.TestCase):
    def test_execute_query_appends_limit_when_missing(self):
        fake = _FakeConn(rows=[{"x": 1}])

        with patch("f1_agent.db.get_connection", return_value=fake):
            rows = db.execute_query("SELECT 1")

        self.assertEqual(rows, [{"x": 1}])
        self.assertTrue(fake.executed_sql.endswith(f"LIMIT {db.MAX_ROWS}"))
        self.assertEqual(fake.cursor.fetchmany_size, db.MAX_ROWS)

    def test_execute_query_preserves_trailing_limit(self):
        sql = "SELECT 1 LIMIT 5"
        fake = _FakeConn(rows=[{"x": 1}])

        with patch("f1_agent.db.get_connection", return_value=fake):
            db.execute_query(sql)

        self.assertEqual(fake.executed_sql, sql)

    def test_execute_query_appends_limit_when_only_subquery_has_limit(self):
        sql = "SELECT * FROM (SELECT 1 AS x LIMIT 1) t"
        fake = _FakeConn(rows=[{"x": 1}])

        with patch("f1_agent.db.get_connection", return_value=fake):
            db.execute_query(sql)

        self.assertTrue(fake.executed_sql.endswith(f"LIMIT {db.MAX_ROWS}"))


class DbConnectionThreadSafetyTests(unittest.TestCase):
    def tearDown(self):
        if db._connection is not None:
            try:
                db._connection.close()
            finally:
                db._connection = None

    def test_get_connection_initializes_singleton_once_under_concurrency(self):
        with tempfile.TemporaryDirectory() as tmp:
            db_path = Path(tmp) / "f1_history.db"
            sqlite3.connect(str(db_path)).close()

            real_connect = sqlite3.connect
            calls = {"count": 0}
            calls_lock = threading.Lock()

            def tracked_connect(*args, **kwargs):
                time.sleep(0.01)
                with calls_lock:
                    calls["count"] += 1
                return real_connect(*args, **kwargs)

            with (
                patch.object(db, "DB_PATH", db_path),
                patch("f1_agent.db.sqlite3.connect", side_effect=tracked_connect),
            ):
                db._connection = None
                results = []

                def worker():
                    results.append(id(db.get_connection()))

                threads = [threading.Thread(target=worker) for _ in range(10)]
                for t in threads:
                    t.start()
                for t in threads:
                    t.join()

            self.assertEqual(calls["count"], 1)
            self.assertEqual(len(set(results)), 1)


if __name__ == "__main__":
    unittest.main()
