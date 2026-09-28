"""Public dataset loaders."""

from __future__ import annotations

import gzip
import urllib.error
import urllib.request

import networkx as nx

from osi.limits import USER_AGENT

# Exact file linked from https://snap.stanford.edu/data/ego-Facebook.html
# ("facebook_combined.txt.gz", edges from all egonets combined).
# https://snap.stanford.edu/data/facebook_combined.txt.gz
SNAP_FACEBOOK_URL = "https://snap.stanford.edu/data/facebook_combined.txt.gz"
_MAX_BYTES = 5_000_000


def load_snap_facebook() -> nx.Graph:
    """Download the SNAP Facebook Social Circles graph.

    Each edge weight is 1.0. Raises RuntimeError with a clear message when
    the download fails (offline, blocked, or a non-200 response).
    """
    request = urllib.request.Request(SNAP_FACEBOOK_URL, headers={"User-Agent": USER_AGENT})
    try:
        with urllib.request.urlopen(request, timeout=30) as response:
            payload = response.read(_MAX_BYTES + 1)
    except (urllib.error.URLError, TimeoutError, OSError) as error:
        raise RuntimeError(
            "Could not download the SNAP Facebook Social Circles edge list "
            f"({SNAP_FACEBOOK_URL}). The network may be offline or blocking "
            f"snap.stanford.edu. ({type(error).__name__}: {error})"
        ) from None
    if len(payload) > _MAX_BYTES:
        raise RuntimeError(
            f"SNAP Facebook download from {SNAP_FACEBOOK_URL} exceeded {_MAX_BYTES} bytes."
        )
    try:
        text = gzip.decompress(payload).decode("utf-8")
    except (OSError, gzip.BadGzipFile, UnicodeError) as error:
        raise RuntimeError(
            "Downloaded SNAP Facebook file could not be read as a gzip-compressed edge list "
            f"from {SNAP_FACEBOOK_URL}. ({type(error).__name__})"
        ) from None

    graph = nx.Graph()
    for line in text.splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        parts = line.split()
        if len(parts) < 2:
            continue
        left, right = int(parts[0]), int(parts[1])
        if left == right:
            continue
        graph.add_edge(left, right, weight=1.0)
    return graph
