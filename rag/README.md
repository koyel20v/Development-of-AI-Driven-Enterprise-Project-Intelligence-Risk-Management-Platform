# RAG Module: Enterprise Project Intelligence & Risk Management Platform

Retrieval-Augmented Generation over project documents (PDF, DOCX, TXT/MD, CSV) with a persistent
**ChromaDB** index and a local **sentence-transformers** embedding model. Answers come **only from the indexed
documents**, with `[n]` source citations, and the system says so when the evidence is missing.

> **Scope.** This module retrieves and summarises documented information. It does not predict risk
> probabilities, costs or delays. Probabilities, budgets, dates, owners and statuses are reported only when a
> document states them.

## 1. How it works

```text
Streamlit upload (app.py)  or  CLI (scripts/ingest_documents.py)
   -> document_loader.py   extract text + metadata, report failures
   -> text_chunker.py      clean + chunk (CSV: whole rows, header repeated), deterministic chunk ids
   -> embeddings.py        local model, batched; embedding_cache.py reuses vectors for identical text
   -> vector_store.py      ChromaDB (cosine), metadata filters, upsert
question -> query embedding -> similarity search (+ project filter) -> llm.py (Groq/Anthropic) -> answer + citations
```

| File | Purpose |
|---|---|
| `app.py` | Streamlit UI: upload, validation, preview, indexing progress, retrieval test, confirmed index reset |
| `src/config.py` | All settings from environment / `.env`, validated |
| `src/document_loader.py` | PDF/DOCX/TXT/CSV extraction, project id from folder, load statistics |
| `src/text_chunker.py` | Cleaning, chunking, stable chunk ids |
| `src/embeddings.py` | Sentence-transformers wrapper (batching, retries, output validation) |
| `src/embedding_cache.py` | Model loaded once and shared; SQLite vector cache **keyed by model name** |
| `src/vector_store.py` | ChromaDB wrapper; refuses to mix embedding models or dimensions in one collection |
| `src/ingest_manifest.py` | SHA-256 manifest: unchanged files are skipped without being parsed |
| `src/ingest_worker.py` | Parse + chunk one file (safe for a process pool) |
| `src/ingest_job.py` | Saves uploads, detects duplicates, runs one background indexing job at a time |
| `src/rag_pipeline.py` | `ingest()`, `retrieve()`, `ask()` |
| `src/llm.py` | Groq (default) or Anthropic, injection-resistant grounded prompt |
| `scripts/ingest_documents.py` | CLI indexing (scans a whole folder, see section 5) |
| `scripts/query_rag.py` | CLI question answering |
| `tests/` | Offline tests (fake embedder and LLM, temporary ChromaDB folders) |

## 2. Setup (Windows PowerShell, from the `rag/` folder)

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
pip install streamlit            # only if requirements.txt does not already list it
copy .env.example .env           # then edit .env
```

Put your Groq key in `.env` (`GROQ_API_KEY=...`). `.env` is git-ignored: never commit it, and never put a real key
in `.env.example`. The key is only needed to answer questions, not to index documents.
The first indexing run needs internet to download the embedding model.

## 3. Using the Streamlit app

```powershell
python -m streamlit run app.py --server.fileWatcherType none
```

1. Choose one or more files. A table shows name, type, size, a SHA-256 prefix and a check result
   (unsupported type, empty, too large, duplicate).
2. Optionally tick **Preview extracted text and chunks**. This parses and chunks only; nothing is embedded or stored.
3. Click **Extract and index**. Nothing is embedded before this click, and Streamlit reruns do not start indexing.
4. Watch the progress, then read the results: per-file status, chunks, and a timing table
   (hashing, extraction, chunking, embedding, ChromaDB).
5. Use the retrieval section to ask a question and inspect passages, sources and distances.

## 4. Re-uploading, duplicates and changed files

- **Only the files you select are processed.** `ingest(only_paths=[...])` restricts a run to those files; nothing
  else in `documents/` is hashed, parsed or embedded, and pruning is disabled for such runs.
- **Same file again:** reported as "unchanged (reused)", nothing is embedded.
- **Same content under another name:** detected by SHA-256 and reported as a duplicate; it is not saved or indexed.
- **Same name, changed content:** the new chunks are written and the obsolete chunks of that file are deleted.
  Other documents are never touched.
- Chunk ids are deterministic (path + position + text hash), so indexing is idempotent.

## 5. Command line

```powershell
python scripts/ingest_documents.py --stats          # what is indexed (no ingestion)
python scripts/ingest_documents.py                  # index the whole documents/ folder (incremental)
python scripts/query_rag.py "What are the major risks of Project Alpha?" --show-passages
```

`ingest_documents.py` scans the **entire** `documents/` folder. Run it deliberately, never as part of startup.
The app never calls it. `--force` re-embeds everything and `--prune` removes entries of deleted files.

## 6. Configuration (`.env`)

| Variable | Default | Meaning |
|---|---|---|
| `RAG_EMBEDDING_MODEL` | `BAAI/bge-small-en-v1.5` | Any sentence-transformers model |
| `RAG_COLLECTION_NAME` | `project_documents` | One embedding model per collection |
| `RAG_CHUNK_SIZE_TOKENS` / `RAG_CHUNK_OVERLAP_TOKENS` | 350 / 50 | Approximate tokens |
| `RAG_EMBEDDING_BATCH_SIZE` | 32 | Texts per model batch |
| `RAG_LOAD_WORKERS` | 1 | Processes for parsing PDFs (try 2-4 if PDF loading dominates) |
| `RAG_EMBED_FLUSH_CHUNKS` | 256 | Chunks collected before one embedding call and DB write |
| `RAG_USE_MANIFEST` / `RAG_EMBEDDING_CACHE` | true / true | Hash manifest, vector cache |
| `RAG_TOP_K` / `RAG_MAX_DISTANCE` | 5 / unset | Retrieval size and optional distance cut-off |
| `RAG_DOCUMENTS_DIR` / `RAG_CHROMA_DIR` | `documents` / `chroma_db` | Relative to `rag/` |
| `LLM_PROVIDER` | `groq` | `groq` or `anthropic` |
| `GROQ_MODEL`, `GROQ_API_KEY` | `openai/gpt-oss-120b`, unset | Groq settings |

The project folder level (`RAG_PROJECT_FOLDER_LEVEL=2`) maps `documents/<type>/<project>/file` to a project id.

## 7. Switching the embedding model

The collection records its embedding model and dimension; opening it with a different model raises an error.
To try another model (for example `sentence-transformers/all-MiniLM-L6-v2`):

1. Set `RAG_EMBEDDING_MODEL` to the new model.
2. Set a **new** `RAG_COLLECTION_NAME`, so the old index stays untouched.
3. Match the chunk size to the model's limit. MiniLM truncates at 256 word-pieces, so use about 250 (overlap 40).
4. Restart the app and upload the files you want indexed again.

Which model is better for your documents is not decided here. Compare retrieval quality and speed on your own files
before choosing.

## 8. Deleting an index

Your files in `documents/` are never deleted by any of these. Stop Streamlit first.

- **Current collection:** in the app, open *Advanced: delete the index and start over*, tick the box, type `RESET`.
- **Another collection:**
  ```powershell
  python -c "import chromadb; c=chromadb.PersistentClient(path='chroma_db'); c.delete_collection('OLD_NAME')"
  ```
- **Everything:** delete the `chroma_db` folder (this also removes the manifests and the vector cache).

## 9. Tests

```powershell
python -m pytest -q
```

The tests use a fake embedder and a fake LLM, so they download no model and make no network calls.
They cover extraction, chunking, batching, model-keyed caching, duplicate detection, selected-files-only ingestion,
changed-file stale-chunk handling, unsupported and corrupt files, and retrieval after indexing.

## 10. Troubleshooting

| Symptom | Likely cause and fix |
|---|---|
| `Collection ... was built with embedding model ...` | Model changed. Use a new `RAG_COLLECTION_NAME` or reset the collection |
| Scanned PDF reported as "no extractable text" | The PDF has no text layer. OCR is not included |
| First run is slow | Model download and load (once per server start) |
| Answers fail with an authentication error | Missing or revoked `GROQ_API_KEY` in `.env` |
| Windows reports a locked file when deleting `chroma_db` | Stop Streamlit first |

## 11. Security notes

- Retrieved passages are treated as untrusted data in the prompt; instructions inside documents are not followed.
- Distances are cosine distances for ranking only. They are not probabilities or confidence scores.
- Documents and vectors stay on your machine; only the question and the retrieved passages are sent to the LLM provider.