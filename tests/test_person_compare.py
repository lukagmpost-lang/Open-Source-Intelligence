import person_compare


def test_page_rank_percentile_sets_the_four_classes():
    # 124 of 1247 is inside the top 10%. 125 is just outside.
    assert person_compare._above_percentile(124, 1247)
    assert not person_compare._above_percentile(125, 1247)
    # 511 of 5110 is exactly the 90th-percentile boundary.
    assert person_compare._above_percentile(511, 5110)
    assert not person_compare._above_percentile(512, 5110)
    assert person_compare.classify(4, 1247, 377, 5110) == "UNIVERSAL"
    assert person_compare.classify(4, 1247, 1615, 5110) == "GITHUB-DOMINANT"
    assert person_compare.classify(400, 1247, 10, 5110) == "REDDIT-DOMINANT"
    assert person_compare.classify(400, 1247, 1615, 5110) == "PERIPHERAL"


def test_placement_follows_score_then_name():
    scores = {"b": 1.0, "a": 1.0, "c": 0.5}
    # Equal scores keep the dict order the store already sorted: name ascending after the score.
    ordered = dict(sorted(scores.items(), key=lambda item: (-item[1], str(item[0]))))
    assert person_compare._placement(ordered, "a")[0] == 1
    assert person_compare._placement(ordered, "c") == (3, 3, 0.5)
