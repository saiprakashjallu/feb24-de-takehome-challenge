from datasketch import MinHash, MinHashLSH
import re


# -------------------------------------------------------
# Configuration
# -------------------------------------------------------

NUM_PERMUTATIONS = 128
SHINGLE_SIZE = 3
LSH_THRESHOLD = 0.85

# LSH only proposes candidate pairs; it can miss true matches because its
# banding is probabilistic. Index at a looser threshold so near-duplicates
# well above LSH_THRESHOLD are almost always proposed, then confirm each
# candidate with the MinHash Jaccard estimate against LSH_THRESHOLD.
CANDIDATE_THRESHOLD = 0.7


# -------------------------------------------------------
# Text normalization
# -------------------------------------------------------

def normalize_text(text):
    """
    Normalize article text before generating shingles.
    """

    if not text:
        return ""

    text = text.lower()

    text = re.sub(
        r"\s+",
        " ",
        text,
    )

    text = re.sub(
        r"[^\w\s]",
        "",
        text,
    )

    return text.strip()


# -------------------------------------------------------
# Shingle generation
# -------------------------------------------------------

def generate_shingles(
    text,
    shingle_size=SHINGLE_SIZE,
):
    """
    Generate word-level shingles.
    """

    normalized = normalize_text(text)

    words = normalized.split()

    if len(words) < shingle_size:
        return {
            normalized
        } if normalized else set()

    return {
        " ".join(
            words[i:i + shingle_size]
        )
        for i in range(
            len(words) - shingle_size + 1
        )
    }


# -------------------------------------------------------
# MinHash
# -------------------------------------------------------

def create_minhash(
    text,
    num_perm=NUM_PERMUTATIONS,
):
    """
    Create a MinHash signature for article text.
    """

    minhash = MinHash(
        num_perm=num_perm
    )

    shingles = generate_shingles(
        text
    )

    for shingle in shingles:
        minhash.update(
            shingle.encode("utf-8")
        )

    return minhash


# -------------------------------------------------------
# Deduplicate articles
# -------------------------------------------------------

def deduplicate_articles(
    articles,
    threshold=LSH_THRESHOLD,
):
    """
    Identify semantically similar articles using
    MinHash + LSH.

    Returns
    -------
    processed_articles : list
        All input articles with deduplication metadata.

    groups : dict
        Only actual duplicate groups are returned.
    """

    if not articles:
        return [], {}

    lsh = MinHashLSH(
        threshold=threshold,
        num_perm=NUM_PERMUTATIONS,
    )

    minhashes = {}

    # ---------------------------------------------------
    # 1. Create MinHash signatures
    # ---------------------------------------------------

    for article in articles:

        article_id = article.get(
            "article_id"
        )

        body_text = article.get(
            "body_text",
            "",
        )

        minhash = create_minhash(
            body_text
        )

        minhashes[
            article_id
        ] = minhash

        lsh.insert(
            article_id,
            minhash,
        )

    # ---------------------------------------------------
    # 2. Build duplicate groups
    # ---------------------------------------------------

    all_groups = {}

    for article in articles:

        article_id = article.get(
            "article_id"
        )

        similar_articles = (
            lsh.query(
                minhashes[
                    article_id
                ]
            )
        )

        # Remove the article itself.
        similar_articles = [
            candidate
            for candidate in similar_articles
            if candidate != article_id
        ]

        if similar_articles:

            group_members = set(
                similar_articles
            )

            group_members.add(
                article_id
            )

            group_key = min(
                group_members
            )

            if group_key not in all_groups:

                all_groups[
                    group_key
                ] = set()

            all_groups[
                group_key
            ].update(
                group_members
            )

    # ---------------------------------------------------
    # 3. Merge overlapping groups
    # ---------------------------------------------------

    merged_groups = []

    for members in all_groups.values():

        members = set(members)

        merged = False

        for existing_group in merged_groups:

            if members.intersection(
                existing_group
            ):

                existing_group.update(
                    members
                )

                merged = True

                break

        if not merged:

            merged_groups.append(
                members
            )

    # ---------------------------------------------------
    # 4. Keep ONLY actual duplicate groups
    # ---------------------------------------------------

    duplicate_groups = []

    for group in merged_groups:

        if len(group) > 1:

            duplicate_groups.append(
                group
            )

    # ---------------------------------------------------
    # 5. Create lookup metadata
    # ---------------------------------------------------

    article_to_group = {}

    for group in duplicate_groups:

        group_id = min(group)

        for article_id in group:

            article_to_group[
                article_id
            ] = (
                group_id,
                group,
            )

    # ---------------------------------------------------
    # 6. Mark articles
    # ---------------------------------------------------

    processed_articles = []

    for article in articles:

        article_copy = dict(
            article
        )

        article_id = article_copy.get(
            "article_id"
        )

        if article_id in article_to_group:

            (
                group_id,
                group_members,
            ) = article_to_group[
                article_id
            ]

            canonical_article_id = min(
                group_members
            )

            article_copy[
                "duplicate_group_id"
            ] = group_id

            article_copy[
                "canonical_article_id"
            ] = canonical_article_id

            article_copy[
                "is_duplicate"
            ] = (
                article_id
                != canonical_article_id
            )

        else:

            # Unique article
            article_copy[
                "duplicate_group_id"
            ] = None

            article_copy[
                "canonical_article_id"
            ] = article_id

            article_copy[
                "is_duplicate"
            ] = False

        processed_articles.append(
            article_copy
        )

    # ---------------------------------------------------
    # 7. Convert groups to readable format
    # ---------------------------------------------------

    groups = {}

    for group in duplicate_groups:

        group_id = min(group)

        groups[
            group_id
        ] = sorted(
            group
        )

    return (
        processed_articles,
        groups,
    )
    """
    Normalize article text before creating MinHash signatures.
    """
    if not text:
        return ""

    text = text.lower()
    text = re.sub(r"\s+", " ", text)
    text = re.sub(r"[^\w\s]", "", text)

    return text.strip()


def create_minhash(text):
    """
    Create a MinHash signature using 3-word shingles.
    """

    normalized = normalize_text(text)

    tokens = normalized.split()

    minhash = MinHash(
        num_perm=NUM_PERMUTATIONS
    )

    # Create 3-word shingles
    for i in range(
        max(0, len(tokens) - 2)
    ):
        shingle = " ".join(
            tokens[i:i + 3]
        )

        minhash.update(
            shingle.encode("utf-8")
        )

    return minhash


def arrival_order(article):
    """
    Sort key for the order articles arrived on the stream:
    arrival timestamp, then shard, then sequence number.

    Articles without a sequence number (e.g. unit-test input)
    sort after those that have one and keep their input order.
    """

    sequence = str(article.get("_source_sequence_number") or "")

    if not sequence.isdigit():
        return (1, "", "", 0)

    return (
        0,
        article.get("_source_timestamp") or "",
        article.get("_source_shard_id") or "",
        int(sequence),
    )


def deduplicate_articles(articles):
    """
    Detect semantically similar articles using MinHash LSH.

    Articles are processed in arrival order, so the copy that
    reached the stream first becomes the canonical article and
    later near-copies are marked as its duplicates.

    Important:
    We DO NOT remove duplicate articles here.

    Every article is returned with:
        - duplicate_group_id
        - canonical_article_id
        - is_duplicate

    This allows the Silver layer to preserve the complete dataset.

    The Gold layer can later filter:
        is_duplicate == False
    """

    lsh = MinHashLSH(
        threshold=CANDIDATE_THRESHOLD,
        num_perm=NUM_PERMUTATIONS,
    )

    # MinHash of every canonical article, for confirming candidates.
    canonical_minhashes = {}

    processed_articles = []

    groups = {}

    for article in sorted(articles, key=arrival_order):

        article_id = article.get(
            "article_id"
        )

        # New schema uses body_text
        body_text = article.get(
            "body_text",
            ""
        )

        # Create MinHash
        minhash = create_minhash(
            body_text
        )

        # LSH proposes candidates; keep only those whose estimated
        # Jaccard similarity really reaches LSH_THRESHOLD.
        scored = [
            (minhash.jaccard(canonical_minhashes[candidate]), candidate)
            for candidate in lsh.query(minhash)
        ]

        matches = [
            candidate
            for similarity, candidate in sorted(scored, reverse=True)
            if similarity >= LSH_THRESHOLD
        ]

        if matches:

            # Most similar existing canonical article
            canonical_id = matches[0]

            article["duplicate_group_id"] = (
                canonical_id
            )

            article["canonical_article_id"] = (
                canonical_id
            )

            article["is_duplicate"] = True

            groups.setdefault(
                canonical_id,
                []
            ).append(
                article_id
            )

        else:

            # This is a new canonical article
            lsh.insert(
                article_id,
                minhash
            )

            canonical_minhashes[article_id] = minhash

            article["duplicate_group_id"] = (
                article_id
            )

            article["canonical_article_id"] = (
                article_id
            )

            article["is_duplicate"] = False

            groups.setdefault(
                article_id,
                []
            )

        # IMPORTANT:
        # Keep ALL articles.
        processed_articles.append(
            article
        )

    return (
        processed_articles,
        groups
    )
