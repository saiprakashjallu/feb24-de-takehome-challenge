import os
import sys

# Ensure processor directory is in sys.path
processor_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if processor_dir not in sys.path:
    sys.path.insert(0, processor_dir)

try:
    from src.deduplication import deduplicate_articles
except ImportError:
    from deduplication import deduplicate_articles


def test_controlled_minhash_deduplication():
    articles = [
        {
            "article_id": "article-001",
            "title": "Technology News",
            "body_text": (
                "The technology company announced "
                "a major product update today. "
                "The new product improves performance "
                "and provides better reliability for customers."
            ),
        },
        {
            "article_id": "article-002",
            "title": "Technology News Update",
            "body_text": (
                "The technology company announced "
                "a major product update today. "
                "The new product improves performance "
                "and provides better reliability for customers."
            ),
        },
        {
            "article_id": "article-003",
            "title": "Sports News",
            "body_text": (
                "The local football team won the championship "
                "after a closely contested final match."
            ),
        },
        {
            "article_id": "article-004",
            "title": "Weather News",
            "body_text": (
                "Heavy rainfall is expected across the region "
                "during the weekend."
            ),
        },
    ]

    processed_articles, groups = deduplicate_articles(articles)

    # Two articles should be recognized as duplicates.
    duplicate_articles = [
        article
        for article in processed_articles
        if article["is_duplicate"]
    ]

    assert len(duplicate_articles) == 1

    # Filter non-empty duplicate groups (groups that actually have duplicates)
    non_empty_groups = {k: v for k, v in groups.items() if v}
    assert len(non_empty_groups) == 1

    # Four input articles - one duplicate = three canonical articles.
    canonical_articles = [
        article
        for article in processed_articles
        if not article["is_duplicate"]
    ]

    assert len(canonical_articles) == 3

    # The first article should be the canonical article.
    article_001 = next(
        article
        for article in processed_articles
        if article["article_id"] == "article-001"
    )

    article_002 = next(
        article
        for article in processed_articles
        if article["article_id"] == "article-002"
    )

    assert article_001["is_duplicate"] is False
    assert article_002["is_duplicate"] is True

    assert (
        article_001["canonical_article_id"]
        == "article-001"
    )

    assert (
        article_002["canonical_article_id"]
        == "article-001"
    )



def test_canonical_is_first_to_arrive_not_lowest_id():
    """
    The record that arrived first has the alphabetically LATER id and
    is listed second in the input. It must still be the canonical
    article, and the later near-copy must be its duplicate.
    """
    body = (
        "City officials approved a new transit plan on Tuesday that will "
        "add three bus rapid transit lines across the northern districts. "
        "The plan was debated for more than six hours before the council "
        "voted to fund the first phase of construction next spring."
    )

    articles = [
        {
            "article_id": "aaa-near-copy",
            "body_text": body.replace("next spring.", "next summer."),
            "_source_timestamp": "2026-10-12T13:00:02+00:00",
            "_source_shard_id": "shardId-000000000000",
            "_source_sequence_number": "200",
        },
        {
            "article_id": "zzz-original",
            "body_text": body,
            "_source_timestamp": "2026-10-12T13:00:01+00:00",
            "_source_shard_id": "shardId-000000000000",
            "_source_sequence_number": "100",
        },
    ]

    processed, _ = deduplicate_articles(articles)
    by_id = {a["article_id"]: a for a in processed}

    assert by_id["zzz-original"]["is_duplicate"] is False
    assert by_id["aaa-near-copy"]["is_duplicate"] is True
    assert by_id["aaa-near-copy"]["canonical_article_id"] == "zzz-original"
