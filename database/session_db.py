"""
Session Database Module.
Handles SQLite storage for session metrics, historical review, and long-term analytics.
"""

import sqlite3
import os
from datetime import datetime
from config import DB_PATH

class SessionDB:
    def __init__(self, db_path=DB_PATH):
        self.db_path = db_path
        os.makedirs(os.path.dirname(self.db_path), exist_ok=True)
        self._init_db()

    def _get_connection(self):
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        return conn

    def _init_db(self):
        """Creates the sessions table schema if not already present."""
        with self._get_connection() as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS sessions (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    date TEXT,
                    start_time TEXT,
                    duration REAL,
                    attention_score REAL,
                    focused_time REAL,
                    distracted_time REAL,
                    drowsy_time REAL,
                    avg_focus_span REAL,
                    longest_focus_span REAL,
                    distraction_count INTEGER,
                    blink_count INTEGER,
                    yawn_count INTEGER,
                    primary_distractor TEXT,
                    experiment_label TEXT
                )
            """)
            conn.commit()

    def save_session(self, data: dict):
        """
        Inserts a completed session record into the database.
        Returns the new session id.
        """
        now = datetime.now()
        date_str = now.strftime("%Y-%m-%d")
        time_str = now.strftime("%H:%M:%S")

        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                INSERT INTO sessions (
                    date, start_time, duration, attention_score,
                    focused_time, distracted_time, drowsy_time,
                    avg_focus_span, longest_focus_span, distraction_count,
                    blink_count, yawn_count, primary_distractor, experiment_label
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                date_str,
                data.get("start_time", time_str),
                data.get("duration", 0.0),
                data.get("attention_score", 0.0),
                data.get("focused_time", 0.0),
                data.get("distracted_time", 0.0),
                data.get("drowsy_time", 0.0),
                data.get("avg_focus_span", 0.0),
                data.get("longest_focus_span", 0.0),
                data.get("distraction_count", 0),
                data.get("blink_count", 0),
                data.get("yawn_count", 0),
                data.get("primary_distractor", "None"),
                data.get("experiment_label", "Standard")
            ))
            conn.commit()
            return cursor.lastrowid

    def get_all_sessions(self):
        """Fetches all sessions sorted by recency."""
        with self._get_connection() as conn:
            rows = conn.execute("SELECT * FROM sessions ORDER BY id DESC").fetchall()
            return [dict(r) for r in rows]

    def get_recent_sessions(self, limit=7):
        """Fetches the N most recent sessions."""
        with self._get_connection() as conn:
            rows = conn.execute("SELECT * FROM sessions ORDER BY id DESC LIMIT ?", (limit,)).fetchall()
            return [dict(r) for r in rows]

    def get_analytics_summary(self):
        """Computes aggregate analytics over stored sessions."""
        sessions = self.get_all_sessions()
        if not sessions:
            return {
                "total_sessions": 0,
                "avg_attention": 0.0,
                "best_attention": 0.0,
                "total_focus_hours": 0.0,
                "avg_focus_span": 0.0
            }

        scores = [s["attention_score"] for s in sessions]
        focus_times = [s["focused_time"] for s in sessions]
        spans = [s["avg_focus_span"] for s in sessions]

        return {
            "total_sessions": len(sessions),
            "avg_attention": round(sum(scores) / len(scores), 1),
            "best_attention": round(max(scores), 1),
            "total_focus_hours": round(sum(focus_times) / 3600.0, 2),
            "avg_focus_span": round(sum(spans) / len(spans), 1)
        }
