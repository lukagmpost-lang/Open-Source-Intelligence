"""Tests for Slack export loader."""
import json
import tempfile
import zipfile
from pathlib import Path
import pytest
import networkx as nx
import sys

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from osi.loaders.slack import load_slack_export, detect_slack_format


def create_slack_export(tmp_path, include_users=True, include_channels=True, messages=None):
    """Helper to create a synthetic Slack export."""
    tmp_path = Path(tmp_path)
    
    # Create users.json
    if include_users:
        users = [
            {"id": "U001", "name": "alice", "real_name": "Alice Smith", "team": {"name": "TestCorp"}},
            {"id": "U002", "name": "bob", "real_name": "Bob Jones"},
            {"id": "U003", "name": "carol", "real_name": "Carol Davis"},
        ]
        with open(tmp_path / "users.json", 'w') as f:
            json.dump(users, f)
    
    # Create channels.json
    if include_channels:
        channels = [
            {"id": "C001", "name": "general"},
            {"id": "C002", "name": "random"},
        ]
        with open(tmp_path / "channels.json", 'w') as f:
            json.dump(channels, f)
    
    # Create message files
    if messages:
        for filename, msgs in messages.items():
            with open(tmp_path / filename, 'w') as f:
                json.dump(msgs, f)
    
    return tmp_path


def test_three_users_thread_creates_three_edges():
    """3 users in one thread get 3 edges."""
    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)
        
        messages = {
            "2024-01-01.json": [
                {"user": "U001", "ts": "1.0", "thread_ts": "1.0", "text": "Starting thread"},
                {"user": "U002", "ts": "2.0", "thread_ts": "1.0", "text": "Reply 1"},
                {"user": "U003", "ts": "3.0", "thread_ts": "1.0", "text": "Reply 2"},
            ]
        }
        
        create_slack_export(tmp_path, messages=messages)
        graph, metadata = load_slack_export(tmp_path, edge_rule="thread")
        
        assert graph.number_of_nodes() == 3
        assert graph.number_of_edges() == 3  # U001-U002, U001-U003, U002-U003
        assert metadata['user_count'] == 3


def test_thread_over_cap_is_skipped():
    """A thread with 31 users is skipped."""
    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)
        
        # Create 31 users
        messages = {
            "2024-01-01.json": []
        }
        
        msgs = []
        for i in range(31):
            msgs.append({"user": f"U{i:03d}", "ts": f"{i}.0", "thread_ts": "1.0", "text": f"Message {i}"})
        messages["2024-01-01.json"] = msgs
        
        create_slack_export(tmp_path, include_users=False, messages=messages)
        graph, metadata = load_slack_export(tmp_path, edge_rule="thread", thread_cap=30)
        
        # Thread should be skipped, so no edges
        assert graph.number_of_edges() == 0


def test_channel_no_threads_no_edges():
    """A channel with no threads produces no edges."""
    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)
        
        messages = {
            "2024-01-01.json": [
                {"user": "U001", "ts": "1.0", "text": "Message without thread"},
                {"user": "U002", "ts": "2.0", "text": "Another message without thread"},
            ]
        }
        
        create_slack_export(tmp_path, messages=messages)
        graph, metadata = load_slack_export(tmp_path, edge_rule="thread")
        
        # No thread_ts, so no edges
        assert graph.number_of_edges() == 0


def test_two_threads_creates_weight_2():
    """Two users who share 2 threads get weight 2."""
    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)
        
        messages = {
            "2024-01-01.json": [
                # Thread 1
                {"user": "U001", "ts": "1.0", "thread_ts": "1.0", "text": "Thread 1"},
                {"user": "U002", "ts": "2.0", "thread_ts": "1.0", "text": "Reply"},
                # Thread 2
                {"user": "U001", "ts": "3.0", "thread_ts": "3.0", "text": "Thread 2"},
                {"user": "U002", "ts": "4.0", "thread_ts": "3.0", "text": "Reply"},
            ]
        }
        
        create_slack_export(tmp_path, messages=messages)
        graph, metadata = load_slack_export(tmp_path, edge_rule="thread")
        
        assert graph.number_of_nodes() == 2
        assert graph.number_of_edges() == 1
        assert graph['U001']['U002']['weight'] == 2


def test_missing_users_json_fallback():
    """Missing users.json falls back to user IDs."""
    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)
        
        messages = {
            "2024-01-01.json": [
                {"user": "U001", "ts": "1.0", "thread_ts": "1.0", "text": "Message"},
                {"user": "U002", "ts": "2.0", "thread_ts": "1.0", "text": "Reply"},
            ]
        }
        
        create_slack_export(tmp_path, include_users=False, messages=messages)
        graph, metadata = load_slack_export(tmp_path, edge_rule="thread")
        
        assert graph.number_of_nodes() == 2
        # Node attributes should fall back to user IDs
        assert graph.nodes['U001']['display_name'] == 'U001'
        assert graph.nodes['U001']['real_name'] == 'U001'
        assert graph.nodes['U002']['display_name'] == 'U002'


def test_zip_file_same_as_folder():
    """A zip file loads the same as a folder."""
    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)
        
        # Create folder export
        folder_path = tmp_path / "folder_export"
        folder_path.mkdir()
        
        messages = {
            "2024-01-01.json": [
                {"user": "U001", "ts": "1.0", "thread_ts": "1.0", "text": "Message"},
                {"user": "U002", "ts": "2.0", "thread_ts": "1.0", "text": "Reply"},
            ]
        }
        
        create_slack_export(folder_path, messages=messages)
        folder_graph, folder_meta = load_slack_export(folder_path, edge_rule="thread")
        
        # Create zip export
        zip_path = tmp_path / "export.zip"
        with zipfile.ZipFile(zip_path, 'w') as zf:
            for file_path in folder_path.glob("*"):
                zf.write(file_path, file_path.name)
        
        zip_graph, zip_meta = load_slack_export(zip_path, edge_rule="thread")
        
        # Compare results
        assert folder_graph.number_of_nodes() == zip_graph.number_of_nodes()
        assert folder_graph.number_of_edges() == zip_graph.number_of_edges()
        assert folder_meta['user_count'] == zip_meta['user_count']


def test_bot_messages_skipped():
    """Bot messages (user ends in 'bot' or is missing) are skipped."""
    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)
        
        messages = {
            "2024-01-01.json": [
                {"user": "U001", "ts": "1.0", "thread_ts": "1.0", "text": "Human message"},
                {"user": "U001bot", "ts": "2.0", "thread_ts": "1.0", "text": "Bot message"},
                {"ts": "3.0", "thread_ts": "1.0", "text": "Message without user"},
            ]
        }
        
        create_slack_export(tmp_path, messages=messages)
        graph, metadata = load_slack_export(tmp_path, edge_rule="thread")
        
        # Only U001 should be in the graph
        assert graph.number_of_nodes() == 1
        assert 'U001' in graph.nodes()
        assert 'U001bot' not in graph.nodes()


def test_detect_slack_format():
    """Test format detection."""
    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)
        
        # High confidence: both users.json and channels.json
        high_path = tmp_path / "high"
        high_path.mkdir()
        create_slack_export(high_path)
        assert detect_slack_format(high_path) == "high"
        
        # Medium confidence: only channels.json
        medium_path = tmp_path / "medium"
        medium_path.mkdir()
        create_slack_export(medium_path, include_users=False)
        assert detect_slack_format(medium_path) == "medium"
        
        # Low confidence: no JSON files
        low_path = tmp_path / "low"
        low_path.mkdir()
        assert detect_slack_format(low_path) == "low"
        
        # Zip file
        zip_path = tmp_path / "export.zip"
        with zipfile.ZipFile(zip_path, 'w') as zf:
            zf.write(high_path / "users.json", "users.json")
            zf.write(high_path / "channels.json", "channels.json")
        assert detect_slack_format(zip_path) == "high"


def test_channel_edge_rule():
    """Test channel edge rule."""
    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)
        
        messages = {
            "2024-01-01.json": [
                {"user": "U001", "ts": "1.0", "text": "Message 1"},
                {"user": "U002", "ts": "2.0", "text": "Message 2"},
                {"user": "U003", "ts": "3.0", "text": "Message 3"},
            ]
        }
        
        create_slack_export(tmp_path, messages=messages)
        graph, metadata = load_slack_export(tmp_path, edge_rule="channel")
        
        # All 3 users should be connected
        assert graph.number_of_nodes() == 3
        assert graph.number_of_edges() == 3  # 3 choose 2


def test_mention_edge_rule():
    """Test mention edge rule."""
    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)
        
        messages = {
            "2024-01-01.json": [
                {"user": "U001", "ts": "1.0", "text": "Hey <@U002> and <@U003>"},
                {"user": "U002", "ts": "2.0", "text": "Hi <@U001>"},
            ]
        }
        
        create_slack_export(tmp_path, messages=messages)
        graph, metadata = load_slack_export(tmp_path, edge_rule="mention")
        
        # U001 mentions U002 and U003, U002 mentions U001
        assert graph.number_of_nodes() == 3
        assert graph.has_edge('U001', 'U002')
        assert graph.has_edge('U001', 'U003')
        assert graph.has_edge('U002', 'U001')


def test_malformed_messages_skipped():
    """Malformed messages are skipped with a warning."""
    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)
        
        messages = {
            "2024-01-01.json": [
                {"user": "U001", "ts": "1.0", "thread_ts": "1.0", "text": "Valid message"},
                "not a dict",  # Malformed
                None,  # Malformed
                {"user": "U002", "ts": "2.0", "thread_ts": "1.0", "text": "Another valid"},
            ]
        }
        
        create_slack_export(tmp_path, messages=messages)
        graph, metadata = load_slack_export(tmp_path, edge_rule="thread")
        
        # Should still work with valid messages
        assert graph.number_of_nodes() == 2
        assert graph.number_of_edges() == 1


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
