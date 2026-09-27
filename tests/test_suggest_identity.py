import json
import sys
from pathlib import Path

import suggest_identity
from osi.store import create_run, save_metrics


def _store(tmp_path, monkeypatch):
    monkeypatch.setenv("OSI_STORE", str(tmp_path / "store.db"))
    path = tmp_path / "store.db"
    create_run("reddit", {"layer": "reddit_user"}, "tiny", run_id="tiny", path=path)
    save_metrics(
        "tiny",
        {
            "akdas": 0.5,
            "Blue": 0.4,
            "bot_user": 0.3,
            "throwaway_x": 0.2,
            "user2": 0.1,
            "malcontent": 0.05,
        },
        metric="pagerank",
        path=path,
    )


def test_matches_skip_generated_names_and_write_the_identity_file(tmp_path, monkeypatch, capsys):
    _store(tmp_path, monkeypatch)
    monkeypatch.setattr(suggest_identity, "_OUTPUT", tmp_path / "identity_candidates.json")
    monkeypatch.setattr(suggest_identity.time, "sleep", lambda _seconds: None)
    requested = []

    def fake_profile(login: str):
        requested.append(login)
        if login == "Blue":
            return None
        if login == "blue":
            return {"login": "blue", "name": "Blue Person", "followers": 12}
        if login == "akdas":
            return {"login": "akdas", "name": "A K Das", "followers": 4}
        if login == "malcontent":
            return None
        raise AssertionError(login)

    monkeypatch.setattr(suggest_identity, "_fetch_profile", fake_profile)
    assert suggest_identity.main(["--run", "tiny", "--top", "10"]) == 0
    # Digits, bot, and throwaway never become a request. Mixed case tries both forms.
    assert requested == ["akdas", "Blue", "blue", "malcontent"]
    text = capsys.readouterr().out
    assert text.splitlines()[0].split() == ["reddit_handle", "github_handle", "gh_followers", "reddit_pagerank"]
    assert "akdas" in text and "0.500000" in text
    assert "bot_user" not in text and "user2" not in text
    payload = json.loads((tmp_path / "identity_candidates.json").read_text())
    assert payload == {
        "akdas": {"github": "akdas", "reddit": "akdas"},
        "Blue": {"github": "blue", "reddit": "Blue"},
    }


def test_rate_limit_stops_and_reports_how_many_were_checked(tmp_path, monkeypatch, capsys):
    _store(tmp_path, monkeypatch)
    monkeypatch.setattr(suggest_identity, "_OUTPUT", tmp_path / "identity_candidates.json")
    monkeypatch.setattr(suggest_identity.time, "sleep", lambda _seconds: None)

    def fake_profile(_login: str):
        raise suggest_identity._RateLimited

    monkeypatch.setattr(suggest_identity, "_fetch_profile", fake_profile)
    assert suggest_identity.main(["--run", "tiny", "--top", "10"]) == 1
    assert "rate limited after checking 1" in capsys.readouterr().err
    assert json.loads((tmp_path / "identity_candidates.json").read_text()) == {}


def test_stops_after_100_matches(tmp_path, monkeypatch):
    monkeypatch.setenv("OSI_STORE", str(tmp_path / "store.db"))
    path = tmp_path / "store.db"
    create_run("reddit", {"layer": "reddit_user"}, "wide", run_id="wide", path=path)
    # Letter-only names stay eligible. A digit in the name would be skipped before any request.
    names = {"".join(chr(ord("a") + (index // 26**power) % 26) for power in range(4)): 1 - index / 1000 for index in range(150)}
    save_metrics("wide", names, metric="pagerank", path=path)
    monkeypatch.setattr(suggest_identity, "_OUTPUT", tmp_path / "identity_candidates.json")
    monkeypatch.setattr(suggest_identity.time, "sleep", lambda _seconds: None)
    calls = {"n": 0}

    def fake_profile(login: str):
        calls["n"] += 1
        return {"login": login, "name": None, "followers": 1}

    monkeypatch.setattr(suggest_identity, "_fetch_profile", fake_profile)
    # Names with digits are skipped, so the scan walks past them until 100 real matches.
    assert suggest_identity.main(["--run", "wide", "--top", "150"]) == 0
    assert calls["n"] == 100
    payload = json.loads((tmp_path / "identity_candidates.json").read_text())
    assert len(payload) == 100
