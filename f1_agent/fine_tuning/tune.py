"""
Launch a Gemini supervised fine-tuning job on Vertex AI.

Prerequisites:
    1. Generate the dataset first:
       uv run python -m f1_agent.fine_tuning.generate_dataset

    2. Upload the training JSONL to GCS:
       gsutil cp f1_agent/fine_tuning/dataset.train.jsonl \
           gs://<YOUR_BUCKET>/fine_tuning/dataset.train.jsonl

    3. Run this script:
       uv run python -m f1_agent.fine_tuning.tune \
           --project <PROJECT_ID> \
           --training-data gs://<BUCKET>/fine_tuning/dataset.train.jsonl

Usage:
    uv run python -m f1_agent.fine_tuning.tune --help
"""

from __future__ import annotations

import argparse


def launch_tuning(
    project: str,
    location: str,
    training_data: str,
    base_model: str,
    tuned_model_name: str,
    epochs: int,
    learning_rate_multiplier: float,
    validation_data: str | None = None,
) -> str:
    """Launch a supervised fine-tuning job and return the tuning job name.

    Returns:
        The Vertex AI tuning job resource name.
    """
    import vertexai
    from vertexai.tuning import sft

    vertexai.init(project=project, location=location)

    tuning_job = sft.train(
        source_model=base_model,
        train_dataset=training_data,
        validation_dataset=validation_data,
        tuned_model_display_name=tuned_model_name,
        epochs=epochs,
        learning_rate_multiplier=learning_rate_multiplier,
    )

    print(f"Tuning job launched: {tuning_job.resource_name}")
    print(f"Base model: {base_model}")
    print(f"Training data: {training_data}")
    if validation_data:
        print(f"Validation data: {validation_data}")
    print(f"Epochs: {epochs}")
    print(f"Learning rate multiplier: {learning_rate_multiplier}")

    return tuning_job.resource_name


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Launch Gemini supervised fine-tuning on Vertex AI"
    )
    parser.add_argument("--project", required=True, help="GCP project ID")
    parser.add_argument("--location", default="us-central1", help="GCP region")
    parser.add_argument(
        "--training-data",
        required=True,
        help="GCS URI for training JSONL (gs://bucket/path.jsonl)",
    )
    parser.add_argument(
        "--validation-data",
        default=None,
        help="GCS URI for validation JSONL (optional)",
    )
    parser.add_argument(
        "--base-model",
        default="gemini-2.5-flash",
        help="Base model to fine-tune (e.g. gemini-2.5-flash, gemini-1.5-flash-002)",
    )
    parser.add_argument(
        "--tuned-model-name",
        default="f1-agent-tuned",
        help="Display name for the tuned model",
    )
    parser.add_argument(
        "--epochs", type=int, default=3, help="Number of training epochs"
    )
    parser.add_argument(
        "--learning-rate-multiplier",
        type=float,
        default=1.0,
        help="Learning rate multiplier",
    )
    args = parser.parse_args()

    resource_name = launch_tuning(
        project=args.project,
        location=args.location,
        training_data=args.training_data,
        validation_data=args.validation_data,
        base_model=args.base_model,
        tuned_model_name=args.tuned_model_name,
        epochs=args.epochs,
        learning_rate_multiplier=args.learning_rate_multiplier,
    )

    print(f"\nTo check status:\n  gcloud ai tuning-jobs describe {resource_name}")
    print(
        "\nOnce done, update agent.py to use the tuned model endpoint "
        "instead of gemini-2.5-pro."
    )


if __name__ == "__main__":
    main()
