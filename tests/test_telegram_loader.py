import json

import pytest

from osi.loaders.telegram import detect_telegram_format, load_telegram_export
import main


def _write_export(path, messages, **overrides):
    export = {
        "name": "Activist Group A",
        "type": "group",
        "id": 123,
        "messages": messages,
    }
    export.update(overrides)
    path.write_text(json.dumps(export), encoding="utf-8")
    return path


def _message(message_id, sender, **fields):
    return {
        "id": message_id,
        "type": "message",
        "date": "2024-01-01T12:00:00Z",
        "from": sender.title(),
        "from_id": f"user{sender}",
        "text": "",
        **fields,
    }


def test_two_users_in_a_reply_chain_get_an_edge(tmp_path):
    path = _write_export(
        tmp_path / "result.json",
        [
            _message(1, "alice"),
            _message(2, "bob", reply_to_message_id=1),
            _message(3, "alice", reply_to_message_id=2),
        ],
    )

    graph, metadata = load_telegram_export(path)

    assert graph.has_edge("Alice", "Bob")
    assert graph.nodes["Alice"] == {"user_id": "useralice", "name": "Alice"}
    assert metadata == {
        "source": "telegram",
        "chat_title": "Activist Group A",
        "chat_type": "group",
        "member_count": 2,
        "message_count": 3,
    }


def test_conversation_over_chat_cap_is_skipped(tmp_path):
    messages = [_message(1, "user0")]
    messages.extend(
        _message(index + 1, f"user{index}", reply_to_message_id=1)
        for index in range(1, 31)
    )
    path = _write_export(tmp_path / "result.json", messages)

    graph, metadata = load_telegram_export(path, chat_cap=30)

    assert graph.number_of_nodes() == 31
    assert graph.number_of_edges() == 0
    assert metadata["member_count"] == 31


def test_reply_to_deleted_message_is_skipped_gracefully(tmp_path):
    path = _write_export(
        tmp_path / "result.json",
        [_message(2, "bob", reply_to_message_id=1)],
    )

    graph, _metadata = load_telegram_export(path)

    assert graph.nodes["Bob"]["name"] == "Bob"
    assert graph.number_of_edges() == 0


def test_self_replies_do_not_create_self_edges(tmp_path):
    path = _write_export(
        tmp_path / "result.json",
        [_message(1, "alice"), _message(2, "alice", reply_to_message_id=1)],
    )

    graph, _metadata = load_telegram_export(path)

    assert graph.number_of_nodes() == 1
    assert graph.number_of_edges() == 0


def test_text_list_uses_only_plain_entries_for_mentions(tmp_path):
    path = _write_export(
        tmp_path / "result.json",
        [
            _message(
                1,
                "alice",
                text=[
                    {"type": "plain", "text": "Hello @bob "},
                    {"type": "bold", "text": "@carol"},
                    {"type": "italic", "text": "@carol"},
                ],
            ),
            _message(2, "bob"),
            _message(3, "carol"),
        ],
    )

    graph, _metadata = load_telegram_export(path, edge_rule="mention")

    assert graph.has_edge("Alice", "Bob")
    assert not graph.has_edge("Alice", "Carol")


def test_messages_without_from_id_are_skipped(tmp_path):
    missing_sender = _message(1, "alice")
    missing_sender.pop("from_id")
    path = _write_export(
        tmp_path / "result.json",
        [missing_sender, _message(2, "bob")],
    )

    graph, metadata = load_telegram_export(path)

    assert list(graph.nodes) == ["Bob"]
    assert metadata["message_count"] == 1


def test_temporal_rule_connects_users_within_five_minutes(tmp_path):
    path = _write_export(
        tmp_path / "result.json",
        [
            _message(1, "alice", date="2024-01-01T12:00:00Z"),
            _message(2, "bob", date="2024-01-01T12:04:00Z"),
            _message(3, "carol", date="2024-01-01T12:10:00Z"),
        ],
    )

    graph, _metadata = load_telegram_export(path, edge_rule="temporal")

    assert graph.has_edge("Alice", "Bob")
    assert not graph.has_edge("Bob", "Carol")


def test_display_name_is_node_key_and_user_id_is_preserved(tmp_path):
    path = _write_export(
        tmp_path / "result.json",
        [
            {
                "id": 1,
                "type": "message",
                "from": "alice",
                "from_id": "user100",
                "text": "hello",
            }
        ],
    )

    graph, _metadata = load_telegram_export(path)

    assert list(graph.nodes) == ["alice"]
    assert graph.nodes["alice"]["user_id"] == "user100"
    assert graph.nodes["alice"]["name"] == "alice"


def test_detect_telegram_format_for_file_and_folder(tmp_path):
    path = _write_export(tmp_path / "result.json", [])
    named_sample = _write_export(tmp_path / "telegram-sample.json", [])

    assert detect_telegram_format(path) == "high"
    assert detect_telegram_format(tmp_path) == "high"
    assert detect_telegram_format(named_sample) == "high"


def test_detect_telegram_format_returns_low_for_random_json(tmp_path):
    path = tmp_path / "random.json"
    path.write_text(json.dumps({"items": []}), encoding="utf-8")

    assert detect_telegram_format(path) == "low"


def test_main_loads_and_reports_telegram_source(tmp_path, monkeypatch, capsys):
    monkeypatch.setenv("OSI_STORE", str(tmp_path / "store.db"))
    path = _write_export(
        tmp_path / "result.json",
        [_message(1, "alice"), _message(2, "bob", reply_to_message_id=1)],
    )

    result = main.main(
        [
            "--source", "telegram",
            "--path", str(path),
            "--analyze", "centrality",
            "--save-run", "telegram-test",
            "--out", str(tmp_path / "graph.json"),
        ]
    )

    output = capsys.readouterr().out
    assert result == 0
    assert "Detected: Telegram export" in output
    assert "Chat: Activist Group A" in output
    assert "Type: group" in output
    assert "Members: 2, Messages: 2" in output
    assert "Graph: 2 nodes, 1 edge, 1 component" in output
    assert "saved run telegram-test" in output