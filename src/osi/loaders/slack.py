"""Slack export loader for OS-INT."""
import json
import os
import re
import zipfile
import gzip
import warnings
from pathlib import Path
from typing import Tuple, Dict, Any, Optional
import networkx as nx


def load_slack_export(path: str, edge_rule: str = "thread", thread_cap: int = 30) -> Tuple[nx.Graph, Dict[str, Any]]:
    """Load a Slack workspace export into a NetworkX graph.
    
    path: a folder OR a .zip file containing:
      - users.json (roster with id, name, real_name, profile)
      - channels.json (list of channels with id, name)
      - per-channel JSON files named YYYY-MM-DD.json
    
    Edge rules:
      thread   — two users who posted in the same thread (default)
      channel  — two users who posted in the same channel
      mention  — two users who @-mentioned each other
    
    Skip threads with more than thread_cap unique users.
    
    Node attributes: user_id, display_name, real_name.
    
    Return (graph, metadata) where metadata includes:
      source: "slack"
      workspace_name: from users.json or folder name
      user_count: unique users
      channel_count: number of channels
      channel_names: list of channel names
    """
    # Determine if path is a zip file or folder
    if zipfile.is_zipfile(path):
        return _load_from_zip(path, edge_rule, thread_cap)
    else:
        return _load_from_folder(path, edge_rule, thread_cap)


def _load_from_folder(path: str, edge_rule: str, thread_cap: int) -> Tuple[nx.Graph, Dict[str, Any]]:
    """Load Slack export from a folder."""
    base_path = Path(path)
    
    # Load users.json if available
    users_info = {}
    users_data = []
    users_file = base_path / "users.json"
    if users_file.exists():
        with open(users_file, 'r', encoding='utf-8') as f:
            users_data = json.load(f)
            for user in users_data:
                users_info[user['id']] = {
                    'display_name': user.get('name', user['id']),
                    'real_name': user.get('real_name', user.get('name', user['id']))
                }
    
    # Load channels.json
    channels_info = {}
    channels_file = base_path / "channels.json"
    if channels_file.exists():
        with open(channels_file, 'r', encoding='utf-8') as f:
            channels_data = json.load(f)
            for channel in channels_data:
                channels_info[channel['id']] = channel.get('name', channel['id'])
    
    # Get workspace name
    workspace_name = users_data[0].get('team', {}).get('name', base_path.name) if users_data else base_path.name
    
    # Build graph
    graph = nx.Graph()
    
    # Process channel files
    for channel_file in base_path.glob("*.json"):
        if channel_file.name in ["users.json", "channels.json"]:
            continue
        
        # Handle .json.gz files
        if channel_file.name.endswith(".json.gz"):
            with gzip.open(channel_file, 'rt', encoding='utf-8') as f:
                messages = json.load(f)
        else:
            with open(channel_file, 'r', encoding='utf-8') as f:
                messages = json.load(f)
        
        _process_messages(graph, messages, users_info, edge_rule, thread_cap)
    
    # Add node attributes
    for node_id in graph.nodes():
        if node_id in users_info:
            graph.nodes[node_id]['user_id'] = node_id
            graph.nodes[node_id]['display_name'] = users_info[node_id]['display_name']
            graph.nodes[node_id]['real_name'] = users_info[node_id]['real_name']
        else:
            graph.nodes[node_id]['user_id'] = node_id
            graph.nodes[node_id]['display_name'] = node_id
            graph.nodes[node_id]['real_name'] = node_id
    
    # Build metadata
    metadata = {
        'source': 'slack',
        'workspace_name': workspace_name,
        'user_count': graph.number_of_nodes(),
        'channel_count': len(channels_info),
        'channel_names': list(channels_info.values())
    }
    
    return graph, metadata


def _load_from_zip(path: str, edge_rule: str, thread_cap: int) -> Tuple[nx.Graph, Dict[str, Any]]:
    """Load Slack export from a zip file."""
    with zipfile.ZipFile(path, 'r') as zf:
        # Load users.json if available
        users_info = {}
        users_data = []
        if "users.json" in zf.namelist():
            with zf.open("users.json") as f:
                users_data = json.load(f)
                for user in users_data:
                    users_info[user['id']] = {
                        'display_name': user.get('name', user['id']),
                        'real_name': user.get('real_name', user.get('name', user['id']))
                    }
        
        # Load channels.json
        channels_info = {}
        if "channels.json" in zf.namelist():
            with zf.open("channels.json") as f:
                channels_data = json.load(f)
                for channel in channels_data:
                    channels_info[channel['id']] = channel.get('name', channel['id'])
        
        # Get workspace name
        workspace_name = users_data[0].get('team', {}).get('name', Path(path).stem) if users_data else Path(path).stem
        
        # Build graph
        graph = nx.Graph()
        
        # Process channel files
        for filename in zf.namelist():
            if filename.endswith("/"):
                continue
            if filename in ["users.json", "channels.json"]:
                continue
            if not (filename.endswith(".json") or filename.endswith(".json.gz")):
                continue
            
            # Get the base filename (without path)
            base_name = Path(filename).name
            if base_name in ["users.json", "channels.json"]:
                continue
            
            try:
                with zf.open(filename) as f:
                    if filename.endswith(".gz"):
                        messages = json.load(gzip.GzipFile(fileobj=f))
                    else:
                        messages = json.load(f)
                _process_messages(graph, messages, users_info, edge_rule, thread_cap)
            except Exception as e:
                warnings.warn(f"Could not load {filename}: {e}")
        
        # Add node attributes
        for node_id in graph.nodes():
            if node_id in users_info:
                graph.nodes[node_id]['user_id'] = node_id
                graph.nodes[node_id]['display_name'] = users_info[node_id]['display_name']
                graph.nodes[node_id]['real_name'] = users_info[node_id]['real_name']
            else:
                graph.nodes[node_id]['user_id'] = node_id
                graph.nodes[node_id]['display_name'] = node_id
                graph.nodes[node_id]['real_name'] = node_id
        
        # Build metadata
        metadata = {
            'source': 'slack',
            'workspace_name': workspace_name,
            'user_count': graph.number_of_nodes(),
            'channel_count': len(channels_info),
            'channel_names': list(channels_info.values())
        }
        
        return graph, metadata


def _process_messages(graph: nx.Graph, messages: list, users_info: dict, edge_rule: str, thread_cap: int):
    """Process a list of messages and add edges to the graph."""
    if edge_rule == "thread":
        _process_threads(graph, messages, users_info, thread_cap)
    elif edge_rule == "channel":
        _process_channel(graph, messages, users_info)
    elif edge_rule == "mention":
        _process_mentions(graph, messages, users_info)
    else:
        raise ValueError(f"Unknown edge rule: {edge_rule}")


def _is_bot(user_id: str) -> bool:
    """Check if a user ID is a bot."""
    return user_id.endswith("bot") or not user_id


def _process_threads(graph: nx.Graph, messages: list, users_info: dict, thread_cap: int):
    """Process messages to create edges between users in the same thread."""
    # Group messages by thread
    threads = {}
    
    for msg in messages:
        if not isinstance(msg, dict):
            continue
        
        user = msg.get('user')
        if not user or _is_bot(user):
            continue
        
        # Add node to graph
        if not graph.has_node(user):
            graph.add_node(user)
        
        # Check if this is a thread reply
        thread_ts = msg.get('thread_ts') or msg.get('ts')
        if not thread_ts:
            continue
        
        if thread_ts not in threads:
            threads[thread_ts] = set()
        threads[thread_ts].add(user)
    
    # Create edges for each thread
    for thread_ts, users in threads.items():
        if len(users) > thread_cap:
            continue  # Skip threads with too many users
        
        users_list = list(users)
        for i in range(len(users_list)):
            for j in range(i + 1, len(users_list)):
                u1, u2 = users_list[i], users_list[j]
                if graph.has_edge(u1, u2):
                    graph[u1][u2]['weight'] = graph[u1][u2].get('weight', 1) + 1
                else:
                    graph.add_edge(u1, u2, weight=1)


def _process_channel(graph: nx.Graph, messages: list, users_info: dict):
    """Process messages to create edges between users in the same channel."""
    users_in_channel = set()
    
    for msg in messages:
        if not isinstance(msg, dict):
            continue
        
        user = msg.get('user')
        if not user or _is_bot(user):
            continue
        
        # Add node to graph
        if not graph.has_node(user):
            graph.add_node(user)
        
        users_in_channel.add(user)
    
    users_list = list(users_in_channel)
    for i in range(len(users_list)):
        for j in range(i + 1, len(users_list)):
            u1, u2 = users_list[i], users_list[j]
            if graph.has_edge(u1, u2):
                graph[u1][u2]['weight'] = graph[u1][u2].get('weight', 1) + 1
            else:
                graph.add_edge(u1, u2, weight=1)


def _process_mentions(graph: nx.Graph, messages: list, users_info: dict):
    """Process messages to create edges between users who mention each other."""
    for msg in messages:
        if not isinstance(msg, dict):
            continue
        
        user = msg.get('user')
        if not user or _is_bot(user):
            continue
        
        # Add node to graph
        if not graph.has_node(user):
            graph.add_node(user)
        
        text = msg.get('text', '')
        if not text:
            continue
        
        # Find mentions in the text (Slack format: <@USERID>)
        mentions = re.findall(r'<@([A-Z0-9]+)>', text)
        
        for mentioned_user in mentions:
            if mentioned_user and mentioned_user != user and not _is_bot(mentioned_user):
                # Add mentioned user to graph
                if not graph.has_node(mentioned_user):
                    graph.add_node(mentioned_user)
                if graph.has_edge(user, mentioned_user):
                    graph[user][mentioned_user]['weight'] = graph[user][mentioned_user].get('weight', 1) + 1
                else:
                    graph.add_edge(user, mentioned_user, weight=1)


def detect_slack_format(path: str) -> str:
    """Check if a path is a Slack export.
    Look for channels.json and users.json in a folder or zip.
    Return confidence: "high", "medium", or "low".
    """
    # Check if it's a zip file
    if zipfile.is_zipfile(path):
        try:
            with zipfile.ZipFile(path, 'r') as zf:
                namelist = zf.namelist()
                has_users = any("users.json" in name for name in namelist)
                has_channels = any("channels.json" in name for name in namelist)
                has_json_files = any(name.endswith(".json") for name in namelist)
                
                if has_users and has_channels:
                    return "high"
                elif has_channels or has_json_files:
                    return "medium"
                else:
                    return "low"
        except Exception:
            return "low"
    
    # Check if it's a folder
    base_path = Path(path)
    if not base_path.is_dir():
        return "low"
    
    has_users = (base_path / "users.json").exists()
    has_channels = (base_path / "channels.json").exists()
    has_json_files = any(f.suffix == ".json" for f in base_path.glob("*.json"))
    
    if has_users and has_channels:
        return "high"
    elif has_channels or has_json_files:
        return "medium"
    else:
        return "low"
