from osi.executors import anomaly_scan


def _fold(anomaly: dict) -> float:
    ratio = anomaly.get("ratio")
    if ratio is None or float(ratio) == 0:
        return 0.0
    magnitude = abs(float(ratio))
    return magnitude if magnitude >= 1 else 1 / magnitude


def test_reddit_2012_top_anomaly_is_fragmentation():
    result = anomaly_scan("r2012-v2")
    anomalies = result.values["anomalies"]
    assert len(anomalies) >= 3
    top = anomalies[0]
    blob = f"{top['metric']} {top['finding']}".lower()
    assert "component" in blob or "fragment" in blob
    assert result.values["findings"]


def test_snap_is_closer_to_the_baselines_than_reddit_2012():
    reddit = anomaly_scan("r2012-v2").values["anomalies"]
    snap = anomaly_scan("snap-v1").values["anomalies"]
    assert snap
    assert _fold(reddit[0]) > _fold(snap[0])
    assert reddit[0]["metric"] != snap[0]["metric"] or reddit[0]["this"] != snap[0]["this"]
