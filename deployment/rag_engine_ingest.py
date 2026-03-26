"""Create/import a Vertex RAG corpus for FIA regulation PDFs."""

import argparse
import sys

import vertexai


def _load_rag_module():
    try:
        from vertexai import rag

        return rag
    except Exception:
        from vertexai.preview import rag

        return rag


def _find_corpus_by_display_name(rag_module, display_name: str):
    for corpus in rag_module.list_corpora():
        if getattr(corpus, "display_name", None) == display_name:
            return corpus
    return None


def main():
    parser = argparse.ArgumentParser(description="Bootstrap Vertex RAG corpus")
    parser.add_argument("--project-id", required=True)
    parser.add_argument("--location", default="us-central1")
    parser.add_argument("--display-name", default="f1-regulations-rag")
    parser.add_argument("--description", default="FIA 2026 regulations corpus")
    parser.add_argument(
        "--paths",
        nargs="+",
        required=True,
        help="GCS paths/prefixes for documents (ex.: gs://bucket/fia/)",
    )
    parser.add_argument("--chunk-size", type=int, default=1024)
    parser.add_argument("--chunk-overlap", type=int, default=200)
    parser.add_argument(
        "--import-result-sink",
        default="",
        help="Optional GCS sink for import report NDJSON",
    )
    args = parser.parse_args()

    vertexai.init(project=args.project_id, location=args.location)
    rag = _load_rag_module()

    corpus = _find_corpus_by_display_name(rag, args.display_name)
    if corpus is None:
        try:
            corpus = rag.create_corpus(
                display_name=args.display_name,
                description=args.description,
            )
            print(f"Created corpus: {corpus.name}")
        except Exception as exc:
            message = str(exc)
            if "allowlisted projects" in message and args.location in {
                "us-central1",
                "us-east1",
                "us-east4",
            }:
                print(
                    "RAG Engine is currently allowlisted in "
                    f"{args.location} for new projects.\n"
                    "Try a GA non-allowlist region first, for example: "
                    "europe-west4 or europe-west3.\n"
                    "Example:\n"
                    "  uv run python deployment/rag_engine_ingest.py "
                    f"--project-id {args.project_id} --location europe-west4 "
                    "--display-name f1-regulations-rag "
                    "--paths 'gs://<BUCKET>/regulations/*.pdf'"
                )
                raise SystemExit(2) from exc
            raise
    else:
        print(f"Using existing corpus: {corpus.name}")

    import_kwargs = {
        "corpus_name": corpus.name,
        "paths": args.paths,
        "transformation_config": rag.TransformationConfig(
            rag.ChunkingConfig(
                chunk_size=max(1, args.chunk_size),
                chunk_overlap=max(0, args.chunk_overlap),
            )
        ),
    }
    if args.import_result_sink:
        import_kwargs["import_result_sink"] = args.import_result_sink

    try:
        response = rag.import_files(**import_kwargs)
    except Exception as exc:
        message = str(exc)
        if "internal error" in message.lower() and any(
            p.startswith("gs://") for p in args.paths
        ):
            print(
                "RAG import failed with an internal error. This often happens when "
                "GCS paths are empty/unreadable.\n"
                "Verify objects exist with:\n"
                "  gcloud storage ls 'gs://<BUCKET>/<PREFIX>/*.pdf'\n"
                "Then retry import with valid paths."
            )
        raise

    imported = getattr(response, "imported_rag_files_count", None)
    if imported is not None:
        print(f"Imported files: {imported}")

    print(f"RAG corpus ready: {corpus.name}")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        sys.exit(130)
