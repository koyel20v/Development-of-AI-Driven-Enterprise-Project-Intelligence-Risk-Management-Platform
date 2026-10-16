#!/usr/bin/env python3
"""Ask a question against the indexed project documents.

Run from the ``rag/`` directory:

    python scripts/query_rag.py "What are the major risks of Project Alpha?"
    python scripts/query_rag.py "Why is the schedule delayed?" --project-id project_alpha
    python scripts/query_rag.py "Which risks could cause budget overruns?" --top-k 8 --show-passages
    python scripts/query_rag.py "What mitigation actions are documented?" --json

Exit codes: 0 = answered (or "insufficient evidence"), 1 = fatal error, 2 = provider failure.
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
from src.llm import LLMError  # noqa: E402
from src.rag_pipeline import RAGPipeline  # noqa: E402
from src.vector_store import VectorStoreError  # noqa: E402


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Query the project-document RAG index.")
    parser.add_argument("question", help="Question to ask (quote it)")
    parser.add_argument("--project-id", action="append", dest="project_ids",
                        help="Restrict to a project id (repeat for several), e.g. project_alpha")
    parser.add_argument("--doc-type", action="append", dest="doc_types", choices=["pdf", "docx", "txt", "csv"],
                        help="Restrict to a document type (repeatable)")
    parser.add_argument("--top-k", type=int, help="Passages to retrieve (default RAG_TOP_K)")
    parser.add_argument("--max-distance", type=float, help="Drop passages with cosine distance above this value")
    parser.add_argument("--collection", help="Override RAG_COLLECTION_NAME")
    parser.add_argument("--chroma-dir", help="Override RAG_CHROMA_DIR")
    parser.add_argument("--show-passages", action="store_true", help="Print the retrieved passage text")
    parser.add_argument("--json", action="store_true", help="Print the full structured result as JSON")
    parser.add_argument("-v", "--verbose", action="store_true", help="Debug logging")
    return parser


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    logging.basicConfig(level=logging.DEBUG if args.verbose else logging.WARNING,
                        format="%(levelname)s %(name)s: %(message)s")
    try:
        settings = RAGSettings.from_env().with_overrides(collection_name=args.collection, chroma_dir=args.chroma_dir)
        pipeline = RAGPipeline(settings)
        result = pipeline.ask(
            args.question,
            top_k=args.top_k,
            project_ids=args.project_ids,
            doc_types=args.doc_types,
            max_distance=args.max_distance,
        )
    except (ConfigError, EmbeddingError, VectorStoreError, LLMError, ValueError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1

    if args.json:
        print(json.dumps(result.to_dict(), indent=2, ensure_ascii=False))
        return 2 if result.error else 0

    print(f"\nQuestion: {result.question}\n")
    print(result.answer)
    if result.error:
        print(f"\n[provider error] {result.error}", file=sys.stderr)
    for warning in result.warnings:
        print(f"\n[warning] {warning}")
    if result.sources:
        print("\nSources:")
        for source in result.sources:
            mark = "*" if source.cited else " "
            where = f", {source.location}" if source.location else ""
            print(f" {mark}[{source.citation_id}] {source.source_path}{where} "
                  f"(project={source.project_id}, distance={source.distance:.3f})")
            if args.show_passages:
                snippet = source.text if len(source.text) < 700 else source.text[:700] + " ..."
                print("      " + snippet.replace("\n", "\n      "))
        print("\n(* = cited in the answer; distance = cosine distance, lower is closer; not a confidence score)")
    return 2 if result.error else 0


if __name__ == "__main__":
    sys.exit(main())
