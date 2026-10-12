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


def test_show_duplicate_groups():

    articles = [
        {
            "article_id": "article-001",
            "title": "Microsoft announces new AI data center",
            "body_text": (
                "Microsoft announced a new AI data center in Texas. "
                "The facility will support artificial intelligence "
                "and cloud computing workloads."
            ),
        },
        {
            "article_id": "article-002",
            "title": "Microsoft announces new AI data center",
            "body_text": (
                "Microsoft announced a new AI data center in Texas. "
                "The facility will support artificial intelligence "
                "and cloud computing workloads."
            ),
        },
        {
            "article_id": "article-003",
            "title": "Apple releases new iPhone",
            "body_text": (
                "Apple released a new iPhone with an improved camera "
                "system and a faster processor."
            ),
        },
    ]

    processed_articles, groups = deduplicate_articles(
        articles
    )

    print("\n" + "=" * 70)
    print("DUPLICATE GROUPS")
    print("=" * 70)

    for group_id, article_ids in groups.items():

        print(f"\nGroup: {group_id}")

        print("Articles:")

        for article_id in article_ids:
            print(f"  - {article_id}")

    print("\n" + "=" * 70)
    print("ARTICLE STATUS")
    print("=" * 70)

    for article in processed_articles:

        print(
            f"\narticle_id: "
            f"{article['article_id']}"
        )

        print(
            f"is_duplicate: "
            f"{article.get('is_duplicate')}"
        )

        print(
            f"canonical_article_id: "
            f"{article.get('canonical_article_id')}"
        )

        print(
            f"duplicate_group_id: "
            f"{article.get('duplicate_group_id')}"
        )

    non_empty_groups = {k: v for k, v in groups.items() if v}
    assert len(non_empty_groups) == 1
