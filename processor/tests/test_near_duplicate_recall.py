import json
import os
import sys

sys.path.insert(
    0,
    os.path.abspath(os.path.join(os.path.dirname(__file__), "..")),
)

from src.deduplication import LSH_THRESHOLD, create_minhash, deduplicate_articles


FIXTURE = os.path.join(
    os.path.dirname(__file__), "fixtures", "missed_near_duplicates.json"
)


def test_near_duplicates_missed_by_lsh_banding_are_caught():
    """
    These 5 pairs (about 95% identical) were missed by the previous
    LSH-only lookup in the 100k load test. Each pair must now be flagged.
    """
    with open(FIXTURE) as f:
        pairs = json.load(f)

    for pair in pairs:
        processed, _ = deduplicate_articles(
            [dict(pair["original"]), dict(pair["near_duplicate"])]
        )
        by_id = {a["article_id"]: a for a in processed}
        original_id = pair["original"]["article_id"]
        copy_id = pair["near_duplicate"]["article_id"]

        assert by_id[original_id]["is_duplicate"] is False
        assert by_id[copy_id]["is_duplicate"] is True
        assert by_id[copy_id]["canonical_article_id"] == original_id


def test_candidates_below_threshold_are_not_merged():
    """
    Articles that share a lot of text but are below LSH_THRESHOLD must
    stay separate, even though the looser candidate lookup may propose them.
    """
    shared = (
        "council approved budget transit plan residents district hospital "
        "school teacher student market price energy power grid storm rain "
        "flood river bridge road traffic police court judge law policy "
    )
    first = shared + (
        "election candidate campaign voter company profit revenue worker "
        "union strike factory farm crop harvest export import trade tariff"
    )
    second = shared + (
        "museum artist festival music film theater library science research "
        "study scientist data report survey health doctor nurse patient"
    )

    similarity = create_minhash(first).jaccard(create_minhash(second))
    assert similarity < LSH_THRESHOLD

    processed, _ = deduplicate_articles(
        [
            {"article_id": "first", "body_text": first},
            {"article_id": "second", "body_text": second},
        ]
    )

    assert all(not a["is_duplicate"] for a in processed)
