import networkx as nx

from osi.answer import answer_system_prompt
from osi.executors import rank_nodes
from osi.store import create_run, save_graph
from osi.vocabulary import get_source, vocabulary_for


def test_vocabulary_for_known_and_unknown_sources():
    assert vocabulary_for("slack")["person"] == "employee"
    assert vocabulary_for("telegram")["person"] == "member"
    assert vocabulary_for("csv")["person"] == "account"
    assert vocabulary_for("unknown")["person"] == "account"


def _rank_findings_for_source(tmp_path, monkeypatch, source):
    monkeypatch.setenv("OSI_STORE", str(tmp_path / "store.db"))
    run_id = f"{source}-vocabulary"
    graph = nx.path_graph(["alice", "bob", "carol", "dave"])
    create_run(source, {"layer": source}, run_id=run_id)
    save_graph(run_id, source, graph)
    result = rank_nodes(run_id, metric="degree", top=3)
    return result.values["findings"]


def test_slack_run_findings_use_employee_vocabulary(tmp_path, monkeypatch):
    findings = _rank_findings_for_source(tmp_path, monkeypatch, "slack")

    assert any("employee" in finding for finding in findings)


def test_telegram_run_findings_use_member_vocabulary(tmp_path, monkeypatch):
    findings = _rank_findings_for_source(tmp_path, monkeypatch, "telegram")

    assert any("member" in finding for finding in findings)


def test_get_source_maps_file_csv_and_missing_runs(tmp_path, monkeypatch):
    monkeypatch.setenv("OSI_STORE", str(tmp_path / "store.db"))
    create_run("file", {"path": "edges.csv", "layer": "file"}, run_id="csv-run")

    assert get_source("csv-run") == "csv"
    assert get_source("missing-run") == "general"


def test_answer_system_prompt_includes_source_vocabulary():
    prompt = answer_system_prompt(source="slack")

    assert "This is a workspace network." in prompt
    assert "Refer to people as employee, groups as team." in prompt