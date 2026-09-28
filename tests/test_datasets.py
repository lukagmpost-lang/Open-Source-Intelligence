import gzip
import io
import urllib.error

import pytest

from osi.datasets import SNAP_FACEBOOK_URL, load_snap_facebook


def test_url_is_the_snap_combined_edge_list():
    assert SNAP_FACEBOOK_URL == "https://snap.stanford.edu/data/facebook_combined.txt.gz"


def test_load_snap_facebook_parses_edges_with_unit_weight(monkeypatch):
    body = gzip.compress(b"# comment\n0 1\n1 2\n2 2\n")

    class Response(io.BytesIO):
        def __enter__(self):
            return self

        def __exit__(self, *args):
            return False

    monkeypatch.setattr("osi.datasets.urllib.request.urlopen", lambda *args, **kwargs: Response(body))
    graph = load_snap_facebook()
    assert set(graph.edges) == {(0, 1), (1, 2)}
    assert graph.edges[0, 1]["weight"] == 1.0
    assert graph.edges[1, 2]["weight"] == 1.0


def test_load_snap_facebook_reports_a_blocked_download(monkeypatch):
    def broken(*args, **kwargs):
        raise urllib.error.URLError("network unreachable")

    monkeypatch.setattr("osi.datasets.urllib.request.urlopen", broken)
    with pytest.raises(RuntimeError, match="offline or blocking"):
        load_snap_facebook()
