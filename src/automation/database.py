"""
Database - SQLite storage for processing logs, audits, and stats
"""

import sqlite3
import json
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional, List, Dict, Any
from dataclasses import dataclass, asdict
import asyncio
from contextlib import contextmanager


@dataclass
class ProcessingLogEntry:
    id: Optional[int] = None
    timestamp: Optional[str] = None
    task_type: str = ""
    file_path: str = ""
    file_hash: Optional[str] = None
    model: str = ""
    backend: str = ""
    was_ab_test: bool = False
    result: Optional[str] = None
    success: bool = True
    error_message: Optional[str] = None
    latency_ms: Optional[int] = None
    input_tokens: Optional[int] = None
    output_tokens: Optional[int] = None
    cost_usd: Optional[float] = None
    audit_score: Optional[float] = None
    audit_notes: Optional[str] = None


@dataclass
class QualityAuditEntry:
    id: Optional[int] = None
    timestamp: Optional[str] = None
    processing_log_id: int = 0
    audit_model: str = ""
    accuracy_score: Optional[int] = None
    usefulness_score: Optional[int] = None
    suggested_improvement: Optional[str] = None
    would_rename: bool = False
    better_name: Optional[str] = None
    audit_latency_ms: Optional[int] = None
    audit_cost_usd: Optional[float] = None


class Database:
    def __init__(self, db_path: str):
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_db()

    @contextmanager
    def _get_connection(self):
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        try:
            yield conn
            conn.commit()
        finally:
            conn.close()

    def _init_db(self):
        """Initialize database schema."""
        with self._get_connection() as conn:
            conn.executescript("""
                -- Processing log
                CREATE TABLE IF NOT EXISTS processing_log (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                    task_type TEXT NOT NULL,
                    file_path TEXT NOT NULL,
                    file_hash TEXT,
                    model TEXT NOT NULL,
                    backend TEXT NOT NULL,
                    was_ab_test BOOLEAN DEFAULT FALSE,
                    result TEXT,
                    success BOOLEAN DEFAULT TRUE,
                    error_message TEXT,
                    latency_ms INTEGER,
                    input_tokens INTEGER,
                    output_tokens INTEGER,
                    cost_usd REAL,
                    audit_score REAL,
                    audit_notes TEXT
                );

                -- Quality audits
                CREATE TABLE IF NOT EXISTS quality_audits (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                    processing_log_id INTEGER REFERENCES processing_log(id),
                    audit_model TEXT NOT NULL,
                    accuracy_score INTEGER,
                    usefulness_score INTEGER,
                    suggested_improvement TEXT,
                    would_rename BOOLEAN,
                    better_name TEXT,
                    audit_latency_ms INTEGER,
                    audit_cost_usd REAL
                );

                -- Model stats cache
                CREATE TABLE IF NOT EXISTS model_stats (
                    model TEXT,
                    task_type TEXT,
                    total_requests INTEGER,
                    success_rate REAL,
                    avg_latency_ms REAL,
                    avg_audit_score REAL,
                    total_cost_usd REAL,
                    last_updated DATETIME,
                    PRIMARY KEY (model, task_type)
                );

                -- Configuration history
                CREATE TABLE IF NOT EXISTS config_history (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                    config_yaml TEXT,
                    changed_by TEXT
                );

                -- Indexes for common queries
                CREATE INDEX IF NOT EXISTS idx_processing_timestamp ON processing_log(timestamp);
                CREATE INDEX IF NOT EXISTS idx_processing_task_type ON processing_log(task_type);
                CREATE INDEX IF NOT EXISTS idx_processing_model ON processing_log(model);
                CREATE INDEX IF NOT EXISTS idx_audit_processing_id ON quality_audits(processing_log_id);
            """)

    # Processing log operations

    def log_processing(self, entry: ProcessingLogEntry) -> int:
        """Log a processing event. Returns the new entry ID."""
        with self._get_connection() as conn:
            cursor = conn.execute(
                """
                INSERT INTO processing_log
                (task_type, file_path, file_hash, model, backend, was_ab_test,
                 result, success, error_message, latency_ms, input_tokens,
                 output_tokens, cost_usd)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    entry.task_type,
                    entry.file_path,
                    entry.file_hash,
                    entry.model,
                    entry.backend,
                    entry.was_ab_test,
                    entry.result,
                    entry.success,
                    entry.error_message,
                    entry.latency_ms,
                    entry.input_tokens,
                    entry.output_tokens,
                    entry.cost_usd,
                ),
            )
            return cursor.lastrowid

    def get_recent_logs(
        self,
        limit: int = 50,
        task_type: Optional[str] = None,
        model: Optional[str] = None,
        hours: Optional[int] = None,
    ) -> List[ProcessingLogEntry]:
        """Get recent processing logs with optional filters."""
        query = "SELECT * FROM processing_log WHERE 1=1"
        params = []

        if task_type:
            query += " AND task_type = ?"
            params.append(task_type)

        if model:
            query += " AND model = ?"
            params.append(model)

        if hours:
            cutoff = datetime.now() - timedelta(hours=hours)
            query += " AND timestamp >= ?"
            params.append(cutoff.isoformat())

        query += " ORDER BY timestamp DESC LIMIT ?"
        params.append(limit)

        with self._get_connection() as conn:
            rows = conn.execute(query, params).fetchall()
            return [ProcessingLogEntry(**dict(row)) for row in rows]

    def get_unaudited_items(self, limit: int = 20) -> List[ProcessingLogEntry]:
        """Get items that haven't been audited yet."""
        with self._get_connection() as conn:
            rows = conn.execute(
                """
                SELECT p.* FROM processing_log p
                LEFT JOIN quality_audits q ON p.id = q.processing_log_id
                WHERE q.id IS NULL AND p.success = TRUE
                ORDER BY p.timestamp DESC
                LIMIT ?
                """,
                (limit,),
            ).fetchall()
            return [ProcessingLogEntry(**dict(row)) for row in rows]

    def update_audit_score(self, log_id: int, score: float, notes: str):
        """Update the audit score on a processing log entry."""
        with self._get_connection() as conn:
            conn.execute(
                "UPDATE processing_log SET audit_score = ?, audit_notes = ? WHERE id = ?",
                (score, notes, log_id),
            )

    # Quality audit operations

    def log_audit(self, entry: QualityAuditEntry) -> int:
        """Log a quality audit. Returns the new entry ID."""
        with self._get_connection() as conn:
            cursor = conn.execute(
                """
                INSERT INTO quality_audits
                (processing_log_id, audit_model, accuracy_score, usefulness_score,
                 suggested_improvement, would_rename, better_name, audit_latency_ms,
                 audit_cost_usd)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    entry.processing_log_id,
                    entry.audit_model,
                    entry.accuracy_score,
                    entry.usefulness_score,
                    entry.suggested_improvement,
                    entry.would_rename,
                    entry.better_name,
                    entry.audit_latency_ms,
                    entry.audit_cost_usd,
                ),
            )

            # Also update the processing log with average score
            avg_score = (
                (entry.accuracy_score or 0) + (entry.usefulness_score or 0)
            ) / 2
            conn.execute(
                "UPDATE processing_log SET audit_score = ?, audit_notes = ? WHERE id = ?",
                (avg_score, entry.suggested_improvement, entry.processing_log_id),
            )

            return cursor.lastrowid

    def get_recent_audits(self, limit: int = 20) -> List[Dict[str, Any]]:
        """Get recent audits with joined processing info."""
        with self._get_connection() as conn:
            rows = conn.execute(
                """
                SELECT q.*, p.file_path, p.task_type, p.model as original_model, p.result
                FROM quality_audits q
                JOIN processing_log p ON q.processing_log_id = p.id
                ORDER BY q.timestamp DESC
                LIMIT ?
                """,
                (limit,),
            ).fetchall()
            return [dict(row) for row in rows]

    # Statistics operations

    def get_stats_by_model(self, days: int = 7) -> List[Dict[str, Any]]:
        """Get aggregated stats by model for the last N days."""
        cutoff = datetime.now() - timedelta(days=days)
        with self._get_connection() as conn:
            rows = conn.execute(
                """
                SELECT
                    model,
                    backend,
                    COUNT(*) as total_requests,
                    SUM(CASE WHEN success THEN 1 ELSE 0 END) * 100.0 / COUNT(*) as success_rate,
                    AVG(latency_ms) as avg_latency_ms,
                    AVG(audit_score) as avg_audit_score,
                    SUM(COALESCE(cost_usd, 0)) as total_cost_usd
                FROM processing_log
                WHERE timestamp >= ?
                GROUP BY model, backend
                ORDER BY total_requests DESC
                """,
                (cutoff.isoformat(),),
            ).fetchall()
            return [dict(row) for row in rows]

    def get_stats_by_task(self, days: int = 7) -> List[Dict[str, Any]]:
        """Get aggregated stats by task type for the last N days."""
        cutoff = datetime.now() - timedelta(days=days)
        with self._get_connection() as conn:
            rows = conn.execute(
                """
                SELECT
                    task_type,
                    COUNT(*) as total_requests,
                    SUM(CASE WHEN success THEN 1 ELSE 0 END) * 100.0 / COUNT(*) as success_rate,
                    AVG(latency_ms) as avg_latency_ms,
                    AVG(audit_score) as avg_audit_score
                FROM processing_log
                WHERE timestamp >= ?
                GROUP BY task_type
                ORDER BY total_requests DESC
                """,
                (cutoff.isoformat(),),
            ).fetchall()
            return [dict(row) for row in rows]

    def get_daily_counts(self, days: int = 7) -> List[Dict[str, Any]]:
        """Get daily processing counts."""
        cutoff = datetime.now() - timedelta(days=days)
        with self._get_connection() as conn:
            rows = conn.execute(
                """
                SELECT
                    DATE(timestamp) as date,
                    task_type,
                    COUNT(*) as count,
                    SUM(CASE WHEN success THEN 1 ELSE 0 END) as success_count
                FROM processing_log
                WHERE timestamp >= ?
                GROUP BY DATE(timestamp), task_type
                ORDER BY date DESC
                """,
                (cutoff.isoformat(),),
            ).fetchall()
            return [dict(row) for row in rows]

    def get_today_summary(self) -> Dict[str, Any]:
        """Get summary stats for today."""
        today = datetime.now().date().isoformat()
        with self._get_connection() as conn:
            row = conn.execute(
                """
                SELECT
                    COUNT(*) as total,
                    SUM(CASE WHEN task_type = 'screenshot_rename' THEN 1 ELSE 0 END) as screenshots,
                    SUM(CASE WHEN task_type = 'audio_transcription' THEN 1 ELSE 0 END) as audio,
                    SUM(CASE WHEN task_type = 'inbox_processing' THEN 1 ELSE 0 END) as inbox,
                    SUM(CASE WHEN success THEN 1 ELSE 0 END) * 100.0 / MAX(COUNT(*), 1) as success_rate,
                    SUM(COALESCE(cost_usd, 0)) as total_cost
                FROM processing_log
                WHERE DATE(timestamp) = ?
                """,
                (today,),
            ).fetchone()
            return dict(row) if row else {}

    # A/B test tracking

    def get_ab_test_results(
        self, task_type: str, days: int = 7
    ) -> List[Dict[str, Any]]:
        """Get A/B test results for a task type."""
        cutoff = datetime.now() - timedelta(days=days)
        with self._get_connection() as conn:
            rows = conn.execute(
                """
                SELECT
                    model,
                    COUNT(*) as total,
                    AVG(audit_score) as avg_score,
                    AVG(latency_ms) as avg_latency,
                    SUM(COALESCE(cost_usd, 0)) as total_cost
                FROM processing_log
                WHERE task_type = ? AND was_ab_test = TRUE AND timestamp >= ?
                GROUP BY model
                """,
                (task_type, cutoff.isoformat()),
            ).fetchall()
            return [dict(row) for row in rows]

    # Configuration history

    def save_config(self, config_yaml: str, changed_by: str = "user"):
        """Save a configuration snapshot."""
        with self._get_connection() as conn:
            conn.execute(
                "INSERT INTO config_history (config_yaml, changed_by) VALUES (?, ?)",
                (config_yaml, changed_by),
            )

    def get_config_history(self, limit: int = 10) -> List[Dict[str, Any]]:
        """Get recent configuration changes."""
        with self._get_connection() as conn:
            rows = conn.execute(
                "SELECT * FROM config_history ORDER BY timestamp DESC LIMIT ?",
                (limit,),
            ).fetchall()
            return [dict(row) for row in rows]
