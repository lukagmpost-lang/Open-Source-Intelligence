"""HTTP access to a saved graph.

The command-line tool stays free of this module. Install the ``web`` extra
and run ``uvicorn osi.server:app``.
"""

from __future__ import annotations

import os
import sys
import tempfile
import time
from pathlib import Path
from uuid import uuid4

import networkx as nx
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse
from pydantic import BaseModel

from osi.ask import _execute
from osi.loader import load_edge_list, load_graphml
from osi.loaders.slack import detect_slack_format, load_slack_export
from osi.loaders.telegram import detect_telegram_format, load_telegram_export
from osi.loaders.whatsapp import detect_whatsapp_format, load_whatsapp_export
from osi.store import create_run, get_run, list_runs, load_graph, save_graph, store_path, _connection

app = FastAPI()

_STATIC = Path(__file__).resolve().parent / "static"
_MAX_UPLOAD_BYTES = 50 * 1024 * 1024


class AskRequest(BaseModel):
    run: str
    question: str
    use_llm: bool = True
    use_agent: bool = False


def _static_file(path: str) -> Path:
    root = _STATIC.resolve()
    target = (root / path).resolve()
    if not target.is_relative_to(root) or not target.is_file():
        raise HTTPException(status_code=404, detail="not found")
    return target


@app.get("/api/runs")
def get_runs() -> list[dict]:
    """Return a list of stored runs with details."""
    runs = []
    for run_id, source, created_at, notes in list_runs():
        # Load graph to get node/edge counts
        try:
            # Determine layer from config
            metadata = get_run(run_id)
            config = metadata.get("config") or {}
            layer = config.get("layer") or metadata.get("source")
            graph = load_graph(run_id, layer) if layer else None
            nodes = graph.number_of_nodes() if graph else 0
            edges = graph.number_of_edges() if graph else 0
        except Exception:
            nodes = 0
            edges = 0
        runs.append({
            "run_id": run_id,
            "source": source,
            "nodes": nodes,
            "edges": edges,
            "chat_name": notes or None,
        })
    return runs


@app.get("/api/runs/export")
def export_runs() -> dict:
    """Export the entire store as JSON."""
    data = {}
    tables = [
        "runs",
        "graphs",
        "metrics",
        "communities",
        "downloads",
        "results",
        "answer_cache",
        "metric_cache",
    ]
    with _connection() as conn:
        for table in tables:
            rows = conn.execute(f"SELECT * FROM {table}").fetchall()
            data[table] = [dict(row) for row in rows]
    return data


@app.post("/api/runs/import")
async def import_runs(export_data: dict) -> dict:
    """Import the store from JSON exported by /api/runs/export."""
    with _connection() as conn:
        # Disable foreign key constraints temporarily
        conn.execute("PRAGMA foreign_keys = OFF")
        # Order of deletion: tables that depend on others first
        tables = [
            "metric_cache",
            "answer_cache",
            "results",
            "downloads",
            "communities",
            "metrics",
            "graphs",
            "runs",
        ]
        for table in tables:
            conn.execute(f"DELETE FROM {table}")
        # Insert the provided data
        for table, rows in export_data.items():
            if not rows:
                continue
            columns = ", ".join(rows[0].keys())
            placeholders = ", ".join(["?"] * len(rows[0]))
            insert_sql = f"INSERT INTO {table} ({columns}) VALUES ({placeholders})"
            for row in rows:
                conn.execute(insert_sql, tuple(row.values()))
        # Re-enable foreign keys and commit
        conn.execute("PRAGMA foreign_keys = ON")
        conn.commit()
    return {"status": "success", "message": "Store imported successfully"}
@app.get("/api/runs/{run_id}")
def get_run_summary(run_id: str) -> dict:
    """Return compact graph metadata for the selected run."""
    metadata = get_run(run_id)
    if metadata is None:
        raise HTTPException(status_code=404, detail=f"Run {run_id} was not found.")
    config = metadata.get("config") or {}
    layer = config.get("layer") or metadata.get("source")
    graph = load_graph(run_id, layer) if layer else None
    if graph is None:
        raise HTTPException(status_code=404, detail=f"Graph for run {run_id} was not found.")
    return {
        "run": run_id,
        "source": metadata.get("source") or "unknown",
        "title": config.get("filename") or config.get("path") or run_id,
        "nodes": graph.number_of_nodes(),
        "edges": graph.number_of_edges(),
        "components": nx.number_connected_components(graph) if graph.number_of_nodes() else 0,
    }


def _detect_source(path: Path, requested: str) -> str:
    if requested != "auto":
        return requested
    suffix = path.suffix.casefold()
    if suffix == ".zip" and detect_slack_format(path) != "low":
        return "slack"
    if suffix == ".txt" and detect_whatsapp_format(path) != "low":
        return "whatsapp"
    if suffix == ".json" and detect_telegram_format(path) != "low":
        return "telegram"
    if suffix in {".csv", ".tsv", ".json"}:
        return "file"
    if suffix == ".graphml":
        return "file"
    return "unknown"


def _load_uploaded_graph(
    path: Path,
    source: str,
    edge_rule: str | None,
    window_minutes: float,
    thread_cap: int,
    chat_cap: int,
) -> tuple[nx.Graph, dict]:
    if source == "slack":
        confidence = detect_slack_format(path)
        if confidence == "low":
            raise ValueError("This file does not look like a Slack export ZIP.")
        rule = edge_rule or "thread"
        if rule not in {"thread", "channel", "mention"}:
            raise ValueError("Slack edge rule must be thread, channel, or mention.")
        graph, metadata = load_slack_export(path, edge_rule=rule, thread_cap=thread_cap)
        return graph, {"source": source, "title": metadata["workspace_name"]}
    if source == "telegram":
        confidence = detect_telegram_format(path)
        if confidence == "low":
            raise ValueError("This file does not look like a Telegram export JSON.")
        rule = edge_rule or "reply"
        graph, metadata = load_telegram_export(path, edge_rule=rule, chat_cap=chat_cap)
        return graph, {"source": source, "title": metadata["chat_title"]}
    if source == "whatsapp":
        if detect_whatsapp_format(path) == "low":
            raise ValueError("This file does not look like a WhatsApp text export.")
        rule = edge_rule or "temporal"
        graph, metadata = load_whatsapp_export(
            path,
            edge_rule=rule,
            window_minutes=window_minutes,
            chat_cap=chat_cap,
        )
        return graph, {"source": source, "title": metadata["chat_name"]}
    if source == "file":
        suffix = path.suffix.casefold()
        if suffix in {".csv", ".tsv", ".json"}:
            graph = load_edge_list(path)
        elif suffix == ".graphml":
            graph = load_graphml(path)
        else:
            raise ValueError("Choose a CSV, TSV, JSON, or GraphML file.")
        return graph, {"source": "file", "title": path.stem}
    raise ValueError("Could not recognize this file. Select its export format and try again.")


@app.post("/api/uploads")
async def upload_graph(
    file: UploadFile = File(...),
    source: str = Form("auto"),
    edge_rule: str = Form(""),
    window_minutes: float = Form(5),
    thread_cap: int = Form(30),
    chat_cap: int = Form(30),
) -> dict:
    """Parse an uploaded graph/export and save its graph under a generated run id."""
    allowed_sources = {"auto", "file", "slack", "telegram", "whatsapp"}
    if source not in allowed_sources:
        raise HTTPException(status_code=400, detail="Unknown source format.")
    filename = Path(file.filename or "upload").name
    suffix = Path(filename).suffix.casefold()
    if suffix not in {".csv", ".tsv", ".json", ".graphml", ".txt", ".zip"}:
        raise HTTPException(status_code=400, detail="Unsupported file extension.")

    content = await file.read(_MAX_UPLOAD_BYTES + 1)
    if len(content) > _MAX_UPLOAD_BYTES:
        raise HTTPException(status_code=413, detail="Upload exceeds the 50 MB limit.")

    try:
        with tempfile.TemporaryDirectory(prefix="osi-upload-") as directory:
            path = Path(directory) / filename
            path.write_bytes(content)
            detected_source = _detect_source(path, source)
            graph, metadata = _load_uploaded_graph(
                path,
                detected_source,
                edge_rule or None,
                window_minutes,
                thread_cap,
                chat_cap,
            )
    except (OSError, ValueError, nx.NetworkXError) as error:
        raise HTTPException(status_code=400, detail=str(error)) from error

    if graph.number_of_nodes() == 0:
        raise HTTPException(status_code=400, detail="No senders or graph nodes were found.")

    layer = detected_source
    run_id = f"{detected_source}-{uuid4().hex[:8]}"
    config = {
        "source": detected_source,
        "layer": layer,
        "filename": filename,
        "edge_rule": edge_rule or None,
        "window_minutes": window_minutes,
        "thread_cap": thread_cap,
        "chat_cap": chat_cap,
    }
    create_run(detected_source, config, notes=filename, run_id=run_id)
    save_graph(run_id, layer, graph)
    components = nx.number_connected_components(graph)
    return {
        "run": run_id,
        "source": detected_source,
        "title": metadata.get("title") or filename,
        "filename": filename,
        "nodes": graph.number_of_nodes(),
        "edges": graph.number_of_edges(),
        "components": components,
    }


@app.post("/api/ask")
def post_ask(req: AskRequest) -> dict:
    """Run the pipeline and return the answer plus metadata.

    ``ask`` returns only the sentence. ``_execute`` is the same pipeline
    and also returns intent, params, method, and trust.
    """
    # Determine source, node count, edge count for logging
    try:
        metadata = get_run(req.run)
        source = metadata.get("source") if metadata else "unknown"
        config = metadata.get("config") or {}
        layer = config.get("layer") or metadata.get("source")
        graph = load_graph(req.run, layer) if layer else None
        nodes = graph.number_of_nodes() if graph else 0
        edges = graph.number_of_edges() if graph else 0
    except Exception:
        source = "unknown"
        nodes = 0
        edges = 0
    print(f"[api] ask: run_id={req.run} source={source} nodes={nodes} edges={edges}", file=sys.stderr)

    started = time.perf_counter()
    try:
        text, query = _execute(
            req.run,
            req.question,
            req.use_llm,
            use_agent=req.use_agent,
        )
    except ValueError as error:
        raise HTTPException(status_code=404, detail=str(error)) from error
    runtime_ms = int(round((time.perf_counter() - started) * 1000))
    return {
        "answer": text,
        "intent": query["intent"],
        "params": query["params"],
        "method": query["method"],
        "trust": query["trust"],
        "runtime_ms": runtime_ms,
    }


@app.get("/")
def index() -> FileResponse:
    """Serve the single HTML page."""
    return FileResponse(_static_file("index.html"))


@app.get("/static/{path:path}")
def static(path: str) -> FileResponse:
    """Serve static files from src/osi/static/."""
    return FileResponse(_static_file(path))


def main() -> None:
    """Bind every interface. Render sets PORT; local use defaults to 7860.

    uvicorn reads the port at start. The request handlers do not.
    """
    import uvicorn

    port = int(os.environ.get("PORT", "7860"))
    uvicorn.run(app, host="0.0.0.0", port=port)


if __name__ == "__main__":
    main()
