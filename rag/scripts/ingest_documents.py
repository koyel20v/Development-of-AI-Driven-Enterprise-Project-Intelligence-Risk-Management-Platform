#!/usr/bin/env python3
"""Index project documents into the persistent ChromaDB collection.

Run from the ``rag/`` directory:

    python scripts/ingest_documents.py                 # index documents/ (incremental, safe to repeat)
    python scripts/ingest_documents.py --stats         # show what is indexed, no ingestion
    python scripts/ingest_documents.py --force         # re-embed everything (ignores the manifest)
    python scripts/ingest_documents.py --prune         # also drop sources whose files were deleted
    python scripts/ingest_documents.py --workers 4     # parse/chunk changed files in 4 processes
    python scripts/ingest_documents.py --reset --yes   # drop the COLLECTION (and manifest), then re-ingest

Exit codes: 0 = success, 1 = fatal error, 2 = finished but some files failed.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.config import ConfigError, RAGSettings  # noqa: E402
from src.embeddings import EmbeddingError  # noqa: E402
from src.rag_pipeline import RAGPipeline  # noqa: E402
from src.vector_store import VectorStoreError  # noqa: E402


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Ingest documents into ChromaDB.")
    parser.add_argument("--documents-dir", help="Override RAG_DOCUMENTS_DIR")
    parser.add_argument("--chroma-dir", help="Override RAG_CHROMA_DIR")
    parser.add_argument("--collection", help="Override RAG_COLLECTION_NAME")
    parser.add_argument("--chunk-size", type=int, help="Override RAG_CHUNK_SIZE_TOKENS")
    parser.add_argument("--chunk-overlap", type=int, help="Override RAG_CHUNK_OVERLAP_TOKENS")
    parser.add_argument("--batch-size", type=int, help="Override RAG_EMBEDDING_BATCH_SIZE")
    parser.add_argument("--workers", type=int, help="Override RAG_LOAD_WORKERS (processes for parsing/chunking)")
    parser.add_argument("--force", action="store_true", help="Re-embed all documents even if unchanged")
    parser.add_argument("--prune", action="store_true", help="Remove indexed sources whose files no longer exist")
    parser.add_argument("--stats", action="store_true", help="Print index statistics and exit")
    parser.add_argument("--reset", action="store_true", help="Drop the collection before ingesting (needs --yes)")
    parser.add_argument("--yes", action="store_true", help="Confirm destructive actions such as --reset")
    parser.add_argument("--json", action="store_true", help="Print the report as JSON")
    parser.add_argument("--no-progress", action="store_true", help="Disable the progress bar")
    parser.add_argument("-v", "--verbose", action="store_true", help="Debug logging")
    return parser


def make_progress_callback(enabled: bool):
    """Return ``(callback, close)``. Uses tqdm when installed (it ships with sentence-transformers)."""
    if not enabled:
        return None, lambda: None
    try:
        from tqdm import tqdm
    except ImportError:
        return None, lambda: None
    state = {"bar": None}

    def callback(stage: str, done: int, total: int, detail: str) -> None:
        if stage == "plan":
            if total > 0:
                state["bar"] = tqdm(total=total, unit="file", desc=f"Processing ({detail})", file=sys.stderr)
        elif state["bar"] is not None:
            if stage == "file":
                state["bar"].n = done
                state["bar"].refresh()
            elif stage == "embed":
                state["bar"].set_postfix(chunks_embedded=done)

    def close() -> None:
        if state["bar"] is not None:
            state["bar"].close()

    return callback, close


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    use_bar = not args.no_progress and sys.stderr.isatty()
    if use_bar and not args.verbose:
        logging.getLogger().setLevel(logging.WARNING)  # keep the bar readable; the report is printed at the end
    callback, close_bar = make_progress_callback(use_bar)
    try:
        settings = RAGSettings.from_env().with_overrides(
            documents_dir=args.documents_dir,
            chroma_dir=args.chroma_dir,
            collection_name=args.collection,
            chunk_size_tokens=args.chunk_size,
            chunk_overlap_tokens=args.chunk_overlap,
            embedding_batch_size=args.batch_size,
            load_workers=args.workers,
        )
        pipeline = RAGPipeline(settings)

        if args.stats:
            stats = pipeline.store.stats()
            print(json.dumps(stats, indent=2))
            return 0

        if args.reset:
            if not args.yes:
                print("--reset deletes every vector in the collection. Re-run with --yes to confirm.", file=sys.stderr)
                return 1
            pipeline.reset_index()
            print(f"Collection '{settings.collection_name}' was reset.")

        try:
            report = pipeline.ingest(force=args.force, prune=args.prune, progress=callback)
        finally:
            close_bar()
    except (ConfigError, EmbeddingError, VectorStoreError, FileNotFoundError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1

    if args.json:
        print(json.dumps(report.to_dict(), indent=2))
    else:
        print("\n=== Ingestion report ===")
        print(f"Mode                     : {report.mode}")
        print(f"Documents directory      : {settings.documents_dir}")
        print(f"ChromaDB directory       : {settings.chroma_dir}  (collection '{settings.collection_name}')")
        print(f"Files found              : {report.load_stats.get('files_found', 0)}  (supported: {report.files_discovered})")
        print(f"Documents loaded         : {report.documents_loaded}")
        print(f"  unchanged (skipped)    : {report.documents_unchanged}  (of which not even parsed: {report.documents_skipped_without_parsing})")
        print(f"  indexed / updated      : {report.documents_indexed}")
        print(f"  known-empty (cached)   : {report.documents_empty_cached}")
        print(f"Chunks generated         : {report.chunks_generated}")
        print(f"Chunks embedded          : {report.chunks_embedded}   ({report.embed_calls} embed calls, {report.model_batches} model batches)")
        print(f"Records upserted         : {report.records_upserted}")
        print(f"Stale chunks deleted     : {report.stale_chunks_deleted}")
        print(f"Chunks pruned            : {report.pruned_chunks}")
        print(f"Total chunks in ChromaDB : {report.total_chunks_in_store}")
        print(f"Manifest                 : {report.manifest_note or 'in use'}")
        timings = report.timings_s
        print(
            "Time (s)                 : "
            f"hash {timings.get('hash', 0)} | load {timings.get('load', 0)} | chunk {timings.get('chunk', 0)} | "
            f"embed {timings.get('embed', 0)} | database {timings.get('db', 0)}   (load/chunk are summed across workers)"
        )
        print(f"Duration (wall clock)    : {report.duration_s}s")
        for label, items in (("Unsupported files", report.unsupported_files), ("Empty files", report.empty_files)):
            if items:
                print(f"\n{label} ({len(items)}):")
                for item in items:
                    print(f"  - {item}")
        if report.failures:
            print(f"\nFailed files ({len(report.failures)}):")
            for path, reason in report.failures:
                print(f"  - {path}: {reason}")
    return 2 if report.failures else 0


if __name__ == "__main__":
    sys.exit(main())