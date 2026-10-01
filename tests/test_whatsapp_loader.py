from osi.loaders.whatsapp import detect_whatsapp_format, load_whatsapp_export


def _write_export(path, lines):
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def test_both_timestamp_formats_parse(tmp_path):
    path = _write_export(
        tmp_path / "WhatsApp Chat.txt",
        [
            "Family Group",
            "1/15/24, 10:30:45 AM - Alice: Hello everyone",
            "[1/15/24, 10:31:12 AM] Bob: Hi Alice",
        ],
    )

    graph, metadata = load_whatsapp_export(path)

    assert set(graph) == {"Alice", "Bob"}
    assert metadata["chat_name"] == "Family Group"
    assert metadata["message_count"] == 2
    assert metadata["date_range"] == {
        "first": "2024-01-15T10:30:45",
        "last": "2024-01-15T10:31:12",
    }


def test_multiline_message_continuation_is_combined(tmp_path):
    path = _write_export(
        tmp_path / "chat.txt",
        [
            "1/15/24, 10:30:00 AM - Alice: Hello",
            "@Bob are you there?",
            "1/15/24, 10:31:00 AM - Bob: Yes",
        ],
    )

    graph, _metadata = load_whatsapp_export(path, edge_rule="mention")

    assert graph.has_edge("Alice", "Bob")


def test_system_messages_are_skipped(tmp_path):
    path = _write_export(
        tmp_path / "chat.txt",
        [
            "1/15/24, 10:00:00 AM - Messages and calls are end-to-end encrypted. Only people in this chat can read them.",
            "1/15/24, 10:01:00 AM - Alice added Bob",
            "1/15/24, 10:02:00 AM - Bob left",
            "1/15/24, 10:03:00 AM - Alice changed the subject",
            "1/15/24, 10:04:00 AM - Alice changed this group's icon",
            "1/15/24, 10:05:00 AM - Alice: <Media omitted>",
            "1/15/24, 10:06:00 AM - Bob: This message was deleted",
            "1/15/24, 10:07:00 AM - Carol: Actual message",
        ],
    )

    graph, metadata = load_whatsapp_export(path)

    assert list(graph) == ["Carol"]
    assert graph.nodes["Carol"]["message_count"] == 1
    assert metadata["message_count"] == 1


def test_users_posting_within_temporal_window_get_an_edge(tmp_path):
    path = _write_export(
        tmp_path / "chat.txt",
        [
            "1/15/24, 10:00:00 AM - Alice: Hello",
            "1/15/24, 10:04:59 AM - Bob: Hi",
            "1/15/24, 10:10:00 AM - Carol: Later",
        ],
    )

    graph, _metadata = load_whatsapp_export(path)

    assert graph.has_edge("Alice", "Bob")
    assert not graph.has_edge("Bob", "Carol")


def test_chat_over_cap_keeps_members_but_skips_edges(tmp_path):
    lines = [
        f"1/15/24, 10:{index:02d}:00 AM - User {index}: Message"
        for index in range(31)
    ]
    path = _write_export(tmp_path / "large.txt", lines)

    graph, metadata = load_whatsapp_export(path, chat_cap=30)

    assert graph.number_of_nodes() == 31
    assert graph.number_of_edges() == 0
    assert metadata["member_count"] == 31


def test_sequence_rule_connects_consecutive_senders(tmp_path):
    path = _write_export(
        tmp_path / "chat.txt",
        [
            "1/15/24, 10:00:00 AM - Alice: One",
            "1/15/24, 10:01:00 AM - Bob: Two",
            "1/15/24, 10:02:00 AM - Alice: Three",
        ],
    )

    graph, _metadata = load_whatsapp_export(path, edge_rule="sequence")

    assert graph["Alice"]["Bob"]["weight"] == 2


def test_mention_rule_connects_mentioned_sender(tmp_path):
    path = _write_export(
        tmp_path / "chat.txt",
        [
            "1/15/24, 10:00:00 AM - Alice: @Bob, good morning",
            "1/15/24, 10:01:00 AM - Bob: Hi Alice",
        ],
    )

    graph, _metadata = load_whatsapp_export(path, edge_rule="mention")

    assert graph.has_edge("Alice", "Bob")


def test_detect_whatsapp_format_returns_high_for_export(tmp_path):
    path = _write_export(
        tmp_path / "chat.txt",
        ["1/15/24, 10:30:45 AM - Alice: Hello"],
    )

    assert detect_whatsapp_format(path) == "high"


def test_detect_whatsapp_format_returns_low_for_random_text(tmp_path):
    path = tmp_path / "random.txt"
    path.write_text("A random note with no chat timestamps.\n", encoding="utf-8")

    assert detect_whatsapp_format(path) == "low"


def test_sample_has_six_messages_three_members_and_two_edges():
    graph, metadata = load_whatsapp_export("examples/whatsapp-sample.txt")

    assert metadata["message_count"] == 6
    assert metadata["member_count"] == 3
    assert graph.number_of_edges() == 2