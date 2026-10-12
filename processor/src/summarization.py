import re
from collections import Counter
from math import log

from pyspark.sql import functions as F
from pyspark.sql.types import StringType
from pyspark.sql.functions import udf


@udf(returnType=StringType())
def generate_extractive_summary(text):
    """
    Spark UDF for generating an extractive summary.
    """
    return extractive_summary(
        text,
        summary_ratio=0.25,
        min_sentences=2,
        max_sentences=5,
    )


def add_extractive_summary(df):
    """
    Add an extractive_summary column to a Spark DataFrame.
    """
    return df.withColumn(
        "extractive_summary",
        generate_extractive_summary(
            F.col("body_text")
        ),
    )


def split_into_sentences(text):
    """
    Split article text into sentences.
    """
    if not text:
        return []

    text = text.strip()

    sentences = re.split(
        r"(?<=[.!?])\s+",
        text,
    )

    return [
        sentence.strip()
        for sentence in sentences
        if sentence.strip()
    ]


def tokenize(text):
    """
    Convert text into normalized word tokens.
    """
    return re.findall(
        r"\b[a-zA-Z][a-zA-Z0-9'-]*\b",
        text.lower(),
    )


def calculate_word_scores(sentences):
    """
    Calculate TF-IDF-inspired importance scores for words.
    """

    tokenized_sentences = [
        tokenize(sentence)
        for sentence in sentences
    ]

    document_frequency = Counter()

    for tokens in tokenized_sentences:
        unique_tokens = set(tokens)

        for token in unique_tokens:
            document_frequency[token] += 1

    total_sentences = len(sentences)

    word_scores = {}

    for tokens in tokenized_sentences:
        for token in tokens:
            if token in document_frequency:
                idf = log(
                    (total_sentences + 1)
                    / (document_frequency[token] + 1)
                )

                word_scores[token] = idf

    return word_scores


def score_sentence(sentence, word_scores):
    """
    Calculate the importance score of a sentence.
    """

    tokens = tokenize(sentence)

    if not tokens:
        return 0.0

    scores = [
        word_scores.get(token, 0.0)
        for token in tokens
    ]

    return sum(scores) / len(tokens)


def extractive_summary(
    text,
    summary_ratio=0.25,
    min_sentences=2,
    max_sentences=5,
):
    """
    Generate an extractive summary.

    Parameters
    ----------
    text : str
        Article body text.

    summary_ratio : float
        Fraction of article sentences to include.

    min_sentences : int
        Minimum number of sentences.

    max_sentences : int
        Maximum number of sentences.

    Returns
    -------
    str
        Extractive summary.
    """

    if not text or not text.strip():
        return ""

    sentences = split_into_sentences(text)

    if len(sentences) <= min_sentences:
        return " ".join(sentences)

    word_scores = calculate_word_scores(sentences)

    scored_sentences = []

    for index, sentence in enumerate(sentences):
        score = score_sentence(
            sentence,
            word_scores,
        )

        scored_sentences.append(
            (
                index,
                sentence,
                score,
            )
        )

    number_of_sentences = max(
        min_sentences,
        int(len(sentences) * summary_ratio),
    )

    number_of_sentences = min(
        number_of_sentences,
        max_sentences,
        len(sentences),
    )

    top_sentences = sorted(
        scored_sentences,
        key=lambda x: x[2],
        reverse=True,
    )[:number_of_sentences]

    # Restore original article order.
    top_sentences = sorted(
        top_sentences,
        key=lambda x: x[0],
    )

    return " ".join(
        sentence
        for _, sentence, _ in top_sentences
    )
