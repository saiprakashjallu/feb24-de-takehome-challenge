def log_summarization_baseline(metrics):
    """
    Log extractive summarization baseline metrics to MLflow.

    These metrics measure operational and compression behavior.
    ROUGE is intentionally not logged because the current dataset
    does not contain human-authored reference summaries.
    """

    import mlflow

    experiment_name = "news-summarization-baseline"

    mlflow.set_experiment(
        experiment_name
    )

    with mlflow.start_run(
        run_name="extractive-baseline"
    ):

        # -------------------------------------------------
        # Dataset metrics
        # -------------------------------------------------

        if "articles_evaluated" in metrics:
            mlflow.log_metric(
                "articles_evaluated",
                float(
                    metrics["articles_evaluated"]
                ),
            )

        if "valid_summaries" in metrics:
            mlflow.log_metric(
                "valid_summaries",
                float(
                    metrics["valid_summaries"]
                ),
            )

        # -------------------------------------------------
        # Summary quality / operational metrics
        # -------------------------------------------------

        if "summary_success_rate" in metrics:
            mlflow.log_metric(
                "summary_success_rate",
                float(
                    metrics["summary_success_rate"]
                ),
            )

        if "average_summary_words" in metrics:
            mlflow.log_metric(
                "average_summary_words",
                float(
                    metrics["average_summary_words"]
                ),
            )

        if "average_compression_ratio" in metrics:
            mlflow.log_metric(
                "average_compression_ratio",
                float(
                    metrics["average_compression_ratio"]
                ),
            )

        if "average_word_reduction" in metrics:
            mlflow.log_metric(
                "average_word_reduction",
                float(
                    metrics["average_word_reduction"]
                ),
            )

        if "min_compression_ratio" in metrics:
            mlflow.log_metric(
                "min_compression_ratio",
                float(
                    metrics["min_compression_ratio"]
                ),
            )

        if "max_compression_ratio" in metrics:
            mlflow.log_metric(
                "max_compression_ratio",
                float(
                    metrics["max_compression_ratio"]
                ),
            )

        # -------------------------------------------------
        # Parameters
        # -------------------------------------------------

        mlflow.log_param(
            "summarization_method",
            "extractive",
        )

        mlflow.log_param(
            "algorithm",
            "TF-IDF sentence scoring",
        )

        mlflow.log_param(
            "summary_ratio",
            0.25,
        )

        mlflow.log_param(
            "min_sentences",
            2,
        )

        mlflow.log_param(
            "max_sentences",
            5,
        )

        mlflow.log_param(
            "reference_summaries_available",
            False,
        )

        mlflow.log_param(
            "evaluation_method",
            "operational_and_compression_metrics",
        )

        print(
            "Summarization baseline logged successfully to MLflow."
        )# pyrefly: ignore [missing-import]
import mlflow


EXPERIMENT_NAME = "news-summarization"


def initialize_mlflow():

    mlflow.set_experiment(
        EXPERIMENT_NAME
    )


def log_pipeline_run(metrics):

    initialize_mlflow()

    with mlflow.start_run():

        # -----------------------------------------
        # Log pipeline metrics
        # -----------------------------------------

        for metric_name, value in metrics.items():

            if isinstance(
                value,
                (int, float),
            ):

                mlflow.log_metric(
                    metric_name,
                    float(value),
                )

        # -----------------------------------------
        # Log pipeline parameters
        # -----------------------------------------

        mlflow.log_param(
            "pipeline_version",
            metrics.get(
                "pipeline_version",
                "unknown",
            ),
        )

        mlflow.log_param(
            "pipeline_type",
            "news_article_etl",
        )

        mlflow.log_param(
            "deduplication_method",
            "MinHash_LSH",
        )

        mlflow.log_param(
            "data_processing_engine",
            "Apache_Spark",
        )

        print(
            "MLflow run logged successfully."
        )
