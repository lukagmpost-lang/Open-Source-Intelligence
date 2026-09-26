"""SQLite store for graphs, metrics, and community assignments.

The database lives at ~/.osi/store.db. Set OSI_STORE to point tests at
another file. Public functions take an optional path keyword for the same reason.
"""

from __future__ import annotations

import json
import os
import sqlite3
import uuid
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterator

import networkx as nx

# Node-link JSON uses "links" because that is what the rest of the pipeline writes.
_LINK_KEY = "links"


def store_path(path: str | Path | None = None) -> Path:
    """Resolve the database file, creating nothing yet."""
    if path is not None:
        return Path(path)
    # Tests and callers can redirect the file without changing the default.
    override = os.environ.get("OSI_STORE")
    if override:
        return Path(override)
    return Path.home() / ".osi" / "store.db"


def _ensure_schema(conn: sqlite3.Connection) -> None:
    conn.executescript(
        """
        CREATE TABLE IF NOT EXISTS runs (
            id TEXT PRIMARY KEY,
            source TEXT,
            created_at TIMESTAMP,
            config_json TEXT,
            notes TEXT
        );
        CREATE TABLE IF NOT EXISTS graphs (
            run_id TEXT,
            layer TEXT,
            graph_json TEXT,
            PRIMARY KEY (run_id, layer)
        );
        CREATE TABLE IF NOT EXISTS metrics (
            run_id TEXT,
            node TEXT,
            metric TEXT,
            value REAL,
            PRIMARY KEY (run_id, node, metric)
        );
        CREATE TABLE IF NOT EXISTS communities (
            run_id TEXT,
            node TEXT,
            algorithm TEXT,
            community_id TEXT,
            PRIMARY KEY (run_id, node, algorithm)
        );
        CREATE TABLE IF NOT EXISTS downloads (
            url TEXT PRIMARY KEY,
            local_path TEXT,
            fetched_at TIMESTAMP
        );
        """
    )


@contextmanager
def _connection(path: str | Path | None = None) -> Iterator[sqlite3.Connection]:
    target = store_path(path)
    # The home directory may not have .osi yet on a fresh machine.
    target.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(target)
    try:
        conn.row_factory = sqlite3.Row
        _ensure_schema(conn)
        yield conn
        conn.commit()
    finally:
        conn.close()


def _node_token(node: Any) -> str:
    # JSON keeps "107" and 107 distinct, which a bare str() would collapse.
    return json.dumps(node)


def _node_from_token(token: str) -> Any:
    return json.loads(token)


def create_run(
    source: str,
    config: dict | str | None,
    notes: str = "",
    run_id: str | None = None,
    path: str | Path | None = None,
) -> str:
    """Insert a run row and return its id. An existing id is replaced."""
    if run_id is None:
        run_id = uuid.uuid4().hex
    if isinstance(config, str):
        config_json = config
    else:
        config_json = json.dumps(config or {})
    created_at = datetime.now(timezone.utc).isoformat(timespec="seconds")
    with _connection(path) as conn:
        # Saving the same name again refreshes the row instead of failing.
        conn.execute(
            """
            INSERT INTO runs (id, source, created_at, config_json, notes)
            VALUES (?, ?, ?, ?, ?)
            ON CONFLICT(id) DO UPDATE SET
                source = excluded.source,
                created_at = excluded.created_at,
                config_json = excluded.config_json,
                notes = excluded.notes
            """,
            (run_id, source, created_at, config_json, notes),
        )
    return run_id


def save_graph(run_id: str, layer: str, graph: nx.Graph, path: str | Path | None = None) -> None:
    """Store one layer as node-link JSON."""
    payload = nx.node_link_data(graph, edges=_LINK_KEY)
    encoded = json.dumps(payload)
    with _connection(path) as conn:
        conn.execute(
            """
            INSERT INTO graphs (run_id, layer, graph_json)
            VALUES (?, ?, ?)
            ON CONFLICT(run_id, layer) DO UPDATE SET graph_json = excluded.graph_json
            """,
            (run_id, layer, encoded),
        )


def load_graph(run_id: str, layer: str, path: str | Path | None = None) -> nx.Graph | None:
    """Return the stored graph, or None when that layer was not saved."""
    with _connection(path) as conn:
        row = conn.execute(
            "SELECT graph_json FROM graphs WHERE run_id = ? AND layer = ?",
            (run_id, layer),
        ).fetchone()
    if row is None:
        return None
    return nx.node_link_graph(json.loads(row["graph_json"]), edges=_LINK_KEY)


def save_metrics(
    run_id: str,
    metric_dict: dict,
    metric: str = "pagerank",
    path: str | Path | None = None,
) -> None:
    """Replace one metric's node scores. metric_dict maps node to value."""
    with _connection(path) as conn:
        # Drop the previous scores so a smaller graph does not leave stale nodes.
        conn.execute(
            "DELETE FROM metrics WHERE run_id = ? AND metric = ?",
            (run_id, metric),
        )
        conn.executemany(
            "INSERT INTO metrics (run_id, node, metric, value) VALUES (?, ?, ?, ?)",
            [
                (run_id, _node_token(node), metric, float(value))
                for node, value in metric_dict.items()
            ],
        )


def load_metrics(run_id: str, metric: str, path: str | Path | None = None) -> dict:
    """Return node to score, highest score first."""
    with _connection(path) as conn:
        rows = conn.execute(
            "SELECT node, value FROM metrics WHERE run_id = ? AND metric = ?",
            (run_id, metric),
        ).fetchall()
    scores = {_node_from_token(row["node"]): float(row["value"]) for row in rows}
    # Same tie break as the analysis functions, so a reload prints the same top 20.
    return dict(sorted(scores.items(), key=lambda item: (-item[1], str(item[0]))))


def save_communities(
    run_id: str,
    algorithm: str,
    comm_dict: dict,
    path: str | Path | None = None,
) -> None:
    """Replace one algorithm's assignment. comm_dict maps node to community id."""
    with _connection(path) as conn:
        conn.execute(
            "DELETE FROM communities WHERE run_id = ? AND algorithm = ?",
            (run_id, algorithm),
        )
        conn.executemany(
            """
            INSERT INTO communities (run_id, node, algorithm, community_id)
            VALUES (?, ?, ?, ?)
            """,
            [
                (run_id, _node_token(node), algorithm, str(community_id))
                for node, community_id in comm_dict.items()
            ],
        )


def load_communities(run_id: str, algorithm: str, path: str | Path | None = None) -> dict:
    """Return node to community id. Numeric ids come back as ints."""
    with _connection(path) as conn:
        rows = conn.execute(
            "SELECT node, community_id FROM communities WHERE run_id = ? AND algorithm = ?",
            (run_id, algorithm),
        ).fetchall()
    loaded: dict[Any, Any] = {}
    for row in rows:
        community_id: Any = row["community_id"]
        # Stored as text. Plain integers should stay integers for the size report.
        if isinstance(community_id, str) and community_id.lstrip("-").isdigit():
            community_id = int(community_id)
        loaded[_node_from_token(row["node"])] = community_id
    return loaded


def list_runs(path: str | Path | None = None) -> list[tuple[str, str, str, str]]:
    """Return (id, source, created_at, notes) oldest first."""
    with _connection(path) as conn:
        rows = conn.execute(
            "SELECT id, source, created_at, notes FROM runs ORDER BY created_at, id"
        ).fetchall()
    return [(row["id"], row["source"], row["created_at"], row["notes"] or "") for row in rows]


def get_run(run_id: str, path: str | Path | None = None) -> dict | None:
    """Return the run metadata, or None when the id is missing."""
    with _connection(path) as conn:
        row = conn.execute("SELECT * FROM runs WHERE id = ?", (run_id,)).fetchone()
    if row is None:
        return None
    config_json = row["config_json"] or "{}"
    return {
        "id": row["id"],
        "source": row["source"],
        "created_at": row["created_at"],
        "notes": row["notes"] or "",
        "config": json.loads(config_json),
    }


def delete_run(run_id: str, path: str | Path | None = None) -> None:
    """Remove a run and the graphs, metrics, and communities attached to it."""
    with _connection(path) as conn:
        conn.execute("DELETE FROM graphs WHERE run_id = ?", (run_id,))
        conn.execute("DELETE FROM metrics WHERE run_id = ?", (run_id,))
        conn.execute("DELETE FROM communities WHERE run_id = ?", (run_id,))
        conn.execute("DELETE FROM runs WHERE id = ?", (run_id,))
