"""HTTP access to a saved graph.

The command-line tool stays free of this module. Install the ``web`` extra
and run ``uvicorn osi.server:app``.
"""

from __future__ import annotations

import os
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
from osi.store import create_run, get_run, list_runs, load_graph, save_graph

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
def get_runs() -> list[str]:
    """Return a list of stored run ids."""
    return [run_id for run_id, _source, _created_at, _notes in list_runs()]


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
