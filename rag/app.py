"""AI Project Intelligence & Risk Advisor - Milestone 1 UI (restyled, non-blocking indexing).

Place this file in the rag/ folder (next to src/ and scripts/), then run:

    python -m pip install "streamlit>=1.37"
    python -m streamlit run app.py

Flow
----
1. Upload one or more PDF / DOCX / CSV / TXT files.
2. "Extract and index" saves them under documents/uploads/<project>/ (files already on disk with the same
   bytes are not rewritten) and starts ONE background indexing job. The page stays responsive and shows live
   progress, counts and timings. Unchanged files are skipped without being parsed; unchanged chunks are
   never embedded again; failed files stay listed with the reason.
3. Type a question / search phrase and see the most relevant chunks (retrieval preview).

Why this is fast on repeat runs
-------------------------------
* The embedding model loads once per server process, in the background, as soon as the app starts.
* Each file is parsed exactly once (the old page parsed every upload twice, and again on every click).
* Change detection uses SHA-256 content hashes + the chunk counts stored in ChromaDB (see src/rag_pipeline.py).
* Vectors for text that was embedded before are served from a SQLite cache (src/embedding_cache.py).

Tip: for a consistent look in every browser, create .streamlit/config.toml with
    [theme]
    base = "light"
"""

import hashlib
import html
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path

import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parent))

from src.config import RAGSettings  # noqa: E402
from src.document_loader import DEFAULT_MAX_FILE_MB, load_file, slugify_project_id  # noqa: E402
from src.embedding_cache import EmbeddingCache, ManagedEmbedder  # noqa: E402
from src.ingest_job import IngestManager, prepare_uploads  # noqa: E402
from src.llm import describe_location  # noqa: E402
from src.rag_pipeline import RAGPipeline  # noqa: E402
from src.text_chunker import ChunkingConfig, chunk_document  # noqa: E402
from src.vector_store import ChromaVectorStore  # noqa: E402

SUPPORTED_TYPES = ["pdf", "docx", "csv", "txt"]

st.set_page_config(page_title="AI Project Intelligence & Risk Advisor", page_icon="◭", layout="centered")

if not hasattr(st, "fragment"):
    st.error("This app needs Streamlit 1.37 or newer: python -m pip install -U streamlit")
    st.stop()


# --------------------------------------------------------------------------- styling
# Palette: chart paper #EDF1F3, ink navy #14213D, signal amber #F2A900,
# steel teal #0F766E, rule grey #C9D3D9, muted slate #5B6B7A
CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Bricolage+Grotesque:opsz,wght@12..96,500;12..96,700;12..96,800&family=IBM+Plex+Sans:wght@400;500;600&display=swap');

:root {
  --paper: #EDF1F3; --card: #FFFFFF; --ink: #14213D; --slate: #5B6B7A;
  --rule: #C9D3D9; --amber: #F2A900; --teal: #0F766E; --red: #B42318;
}

html, body, [class*="css"], .stApp, .stMarkdown, label, input, textarea, button {
  font-family: 'IBM Plex Sans', system-ui, sans-serif;
}
.stApp {
  background-color: var(--paper);
  background-image:
    linear-gradient(rgba(20,33,61,.045) 1px, transparent 1px),
    linear-gradient(90deg, rgba(20,33,61,.045) 1px, transparent 1px);
  background-size: 32px 32px;
  color: var(--ink);
}
#MainMenu, footer, [data-testid="stHeader"], [data-testid="stToolbar"] { display: none !important; }
.block-container { max-width: 860px; padding-top: 2.2rem; padding-bottom: 5rem; }

.stApp p, .stApp label, .stApp span, .stApp li { color: var(--ink); }
[data-testid="stCaptionContainer"], [data-testid="stCaptionContainer"] * { color: var(--slate) !important; }

/* hero */
.hero { background: var(--ink); border-radius: 6px; padding: 2.2rem 2.2rem 1.9rem; position: relative; overflow: hidden; }
.hero::after {
  content: ""; position: absolute; right: -60px; top: -60px; width: 220px; height: 220px;
  border: 1px solid rgba(242,169,0,.55); border-radius: 50%;
  box-shadow: 0 0 0 34px rgba(242,169,0,.07), 0 0 0 68px rgba(242,169,0,.04);
}
.hero h1 {
  font-family: 'Bricolage Grotesque', sans-serif; font-weight: 800; color: #fff !important;
  font-size: clamp(1.9rem, 5vw, 2.9rem); line-height: 1.05; letter-spacing: -0.02em; margin: 0 0 .7rem; padding: 0;
}
.hero p { color: #B9C4D4 !important; max-width: 520px; margin: 0; font-size: 1.02rem; line-height: 1.55; }
.hero .tag {
  display: inline-block; background: var(--amber); color: var(--ink) !important; font-weight: 600;
  font-size: .78rem; padding: .18rem .6rem; border-radius: 3px; margin-bottom: 1rem;
}

/* progress track */
.track { display: flex; margin: 1.1rem 0 .4rem; background: var(--card); border: 1px solid var(--rule); border-radius: 6px; }
.track .stop { flex: 1; padding: .85rem 1rem; border-right: 1px solid var(--rule); display: flex; gap: .7rem; align-items: center; }
.track .stop:last-child { border-right: 0; }
.track .dot {
  width: 26px; height: 26px; border-radius: 50%; border: 2px solid var(--rule); color: var(--slate);
  display: grid; place-items: center; font-weight: 600; font-size: .8rem; flex: none;
}
.track .name { font-weight: 600; font-size: .92rem; line-height: 1.2; }
.track .sub { color: var(--slate); font-size: .78rem; }
.track .stop.done .dot { background: var(--teal); border-color: var(--teal); color: #fff; }
.track .stop.now .dot { border-color: var(--amber); background: var(--amber); color: var(--ink); }
@media (max-width: 640px) { .track { flex-direction: column; } .track .stop { border-right: 0; border-bottom: 1px solid var(--rule); } }

/* section headings */
.sec { margin: 2.4rem 0 .3rem; display: flex; align-items: center; gap: .8rem; }
.sec h2 { font-family: 'Bricolage Grotesque', sans-serif; font-weight: 700; font-size: 1.55rem; letter-spacing: -0.01em; margin: 0; padding: 0; color: var(--ink) !important; }
.sec .bar { flex: 1; height: 1px; background: var(--rule); }

/* inputs */
.stTextInput input {
  background: var(--card) !important; border: 1px solid var(--rule) !important; border-radius: 4px !important;
  color: var(--ink) !important; padding: .65rem .8rem !important;
}
.stTextInput input:focus { border-color: var(--ink) !important; box-shadow: 0 0 0 3px rgba(242,169,0,.35) !important; }
.stTextInput label p, .stSlider label p, .stCheckbox label p { font-weight: 600; font-size: .9rem; }

[data-testid="stFileUploaderDropzone"] {
  background: var(--card); border: 2px dashed #8FA0AE; border-radius: 6px; padding: 1.6rem;
  transition: border-color .15s, background .15s;
}
[data-testid="stFileUploaderDropzone"]:hover { border-color: var(--amber); background: #FFFBF0; }
[data-testid="stFileUploaderDropzone"] * { color: var(--ink) !important; }

/* "Upload" button inside the dropzone: teal with white text (overrides the rule above) */
[data-testid="stFileUploaderDropzone"] button,
[data-testid="stFileUploaderDropzone"] [data-testid="stBaseButton-secondary"] {
  background: var(--teal) !important; border: 0 !important; border-radius: 4px !important;
}
[data-testid="stFileUploaderDropzone"] button *,
[data-testid="stFileUploaderDropzone"] [data-testid="stBaseButton-secondary"] * { color: #fff !important; fill: #fff !important; }
[data-testid="stFileUploaderDropzone"] button:hover { background: #0B5A54 !important; }

/* uploaded file chips: light card, dark readable text */
[data-testid="stFileUploaderFile"], [data-testid="stFileChip"] {
  background: #F3F6F8 !important; border: 1px solid var(--rule) !important; border-radius: 4px !important;
}
[data-testid="stFileUploaderFile"] *, [data-testid="stFileChip"] *,
[data-testid="stFileUploaderFileName"], [data-testid="stFileChipName"] {
  color: var(--ink) !important; opacity: 1 !important;
}
[data-testid="stFileUploaderFile"] small, [data-testid="stFileChip"] small { color: var(--slate) !important; }
[data-testid="stFileUploaderFile"] svg, [data-testid="stFileChip"] svg { fill: var(--ink) !important; }

/* buttons */
.stButton > button {
  background: var(--ink); color: #fff !important; border: 0; border-radius: 4px; padding: .65rem 1.5rem;
  font-weight: 600; letter-spacing: .01em; transition: transform .1s, background .15s;
}
.stButton > button p { color: #fff !important; }
.stButton > button:hover:not(:disabled) { background: #1F3260; transform: translateY(-1px); }
.stButton > button:focus-visible { outline: 3px solid var(--amber); outline-offset: 2px; }
.stButton > button:disabled { background: #AAB6C0; }

/* result rows */
.res { display: flex; align-items: center; justify-content: space-between; gap: 1rem; flex-wrap: wrap;
  background: var(--card); border: 1px solid var(--rule); border-left: 5px solid var(--teal);
  border-radius: 4px; padding: .75rem 1rem; margin: .5rem 0; }
.res.bad { border-left-color: var(--red); }
.res .fname { font-weight: 600; word-break: break-all; }
.res .meta { color: var(--slate); font-size: .88rem; }
.res .err { color: var(--red); font-size: .88rem; }

.stat { display: inline-block; background: var(--ink); color: #fff; border-radius: 4px; padding: .35rem .8rem; margin-top: .4rem; font-size: .88rem; }
.stat b { color: var(--amber); font-family: 'Bricolage Grotesque', sans-serif; font-size: 1.05rem; }

/* notices */
.note { background: #FFF6DB; border: 1px solid #EBCB6B; border-radius: 4px; padding: .8rem 1rem; margin: .8rem 0; }

/* answer */
.answer { background: var(--card); border: 1px solid var(--ink); border-radius: 6px; padding: 1.2rem 1.4rem; margin: .6rem 0 1rem; box-shadow: 6px 6px 0 var(--amber); }
.answer .lbl { font-weight: 600; color: var(--slate); font-size: .85rem; margin-bottom: .4rem; }

/* expanders = evidence cards */
[data-testid="stExpander"] { background: var(--card); border: 1px solid var(--rule) !important; border-radius: 4px; margin-bottom: .55rem; }
[data-testid="stExpander"] summary { font-weight: 600; }
[data-testid="stExpander"] summary:hover { background: #F6F8F9; }
.prox { height: 6px; border-radius: 3px; background: #E3E9ED; margin: .2rem 0 .7rem; overflow: hidden; }
.prox > div { height: 100%; background: linear-gradient(90deg, var(--teal), var(--amber)); }

[data-testid="stSlider"] [role="slider"] { background: var(--ink); }
:focus-visible { outline-color: var(--amber); }
@media (prefers-reduced-motion: reduce) { * { transition: none !important; } }
</style>
"""
st.markdown(CSS, unsafe_allow_html=True)

EXTRA_CSS = """
<style>
/* indexing stages + counters */
.stages { display: flex; gap: .4rem; flex-wrap: wrap; margin: .4rem 0 .6rem; }
.stages .chip { padding: .25rem .7rem; border: 1px solid var(--rule); border-radius: 999px; background: var(--card);
  font-size: .8rem; color: var(--slate); }
.stages .chip.on { background: var(--amber); border-color: var(--amber); color: var(--ink); font-weight: 600; }
.stat + .stat { margin-left: .4rem; }
</style>
"""
st.markdown(EXTRA_CSS, unsafe_allow_html=True)


def section(title: str) -> None:
    st.markdown(f'<div class="sec"><h2>{html.escape(title)}</h2><div class="bar"></div></div>', unsafe_allow_html=True)


def proximity_bar(distance: float) -> str:
    """Visual closeness: distance 0 fills the bar, distance >= 1 empties it."""
    pct = max(4, min(100, round((1 - min(max(distance, 0.0), 1.0)) * 100)))
    return f'<div class="prox"><div style="width:{pct}%"></div></div>'


def render_track(has_files: bool, indexed: bool, searched: bool) -> str:
    steps = [
        ("Upload", "Add your documents", has_files or indexed),
        ("Index", "Chunk and embed", indexed),
        ("Ask", "Search the evidence", searched),
    ]
    first_open = next((i for i, s in enumerate(steps) if not s[2]), -1)
    parts = []
    for i, (name, sub, done) in enumerate(steps):
        cls = "done" if done else ("now" if i == first_open else "")
        mark = "✓" if done else str(i + 1)
        parts.append(
            f'<div class="stop {cls}"><div class="dot">{mark}</div>'
            f'<div><div class="name">{name}</div><div class="sub">{sub}</div></div></div>'
        )
    return f'<div class="track">{"".join(parts)}</div>'


# --------------------------------------------------------------------------- shared resources
@dataclass
class Services:
    pipeline: RAGPipeline
    embedder: ManagedEmbedder
    manager: IngestManager


@st.cache_resource(show_spinner=False)
def get_services(settings_key: str, _settings: RAGSettings) -> Services:
    """Build the embedder, pipeline and job manager ONCE per server process.

    * The cache key is ``settings_key`` (the repr of every setting): changing the model, collection or folders
      builds fresh objects; ordinary reruns reuse the same ones. ``_settings`` is excluded from hashing.
    * One shared object for all browser sessions: the model is a few hundred MB, and ChromaDB, the manifest
      and the vector cache must not be opened by several competing copies. The embedder serialises model
      calls with a lock and the manager allows only one indexing job at a time.
    * The model loads in a background thread right away, so the delay overlaps with choosing files.
    * The vector dimension is read from the collection metadata, so opening the index never needs the model.
    """
    cache = EmbeddingCache(_settings.chroma_dir / "embedding_cache.sqlite") if _settings.embedding_cache else None
    meta = ChromaVectorStore.peek_metadata(_settings.chroma_dir, _settings.collection_name) or {}
    known_dimension = int(meta["embedding_dim"]) if meta.get("embedding_dim") else None
    embedder = ManagedEmbedder(_settings, cache=cache, known_dimension=known_dimension)
    pipeline = RAGPipeline(_settings, embedder=embedder)
    manager = IngestManager(pipeline, embedder)
    embedder.warm_up_async()
    return Services(pipeline, embedder, manager)


def fmt_secs(value: float) -> str:
    value = max(0.0, float(value))
    if value < 60:
        return f"{value:.1f}s"
    minutes, seconds = divmod(int(round(value)), 60)
    return f"{minutes}m {seconds:02d}s"


def fmt_size(num_bytes: int) -> str:
    return f"{num_bytes / 1_048_576:.1f} MB" if num_bytes >= 1_048_576 else f"{max(1, num_bytes // 1024)} KB"


STAGE_STEPS = ["Hash & scan", "Extract & chunk", "Embed", "Store vectors"]
STAGE_INDEX = {
    "opening the index": 0,
    "scanning & hashing files": 0,
    "extracting text & chunking": 1,
    "nothing to parse (all files unchanged)": 1,
    "loading embedding model": 2,
    "embedding": 2,
    "storing vectors": 3,
    "finalising": 3,
}


def stage_chips(stage: str) -> str:
    active = STAGE_INDEX.get(stage, -1)
    chips = "".join(
        f'<div class="chip{" on" if index == active else ""}">{html.escape(name)}</div>'
        for index, name in enumerate(STAGE_STEPS)
    )
    return f'<div class="stages">{chips}</div>'


def stat_chips(pairs) -> str:
    return "".join(f'<div class="stat">{html.escape(label)}: <b>{html.escape(str(value))}</b></div>' for label, value in pairs)


def current_indexed_count():
    """Chunks in the collection, or ``None`` while the vector dimension is unknown (model still loading)."""
    if not embedder.dimension_known:
        return None
    try:
        return pipeline.store.count()
    except Exception:
        return 0



# --------------------------------------------------------------------------- upload validation + preview
def validate_selection(files) -> list:
    """Cheap checks on the selected files (no parsing): type, size, empty, duplicate content within the selection."""
    rows, seen = [], {}
    for f in files:
        data = f.getvalue()
        sha = hashlib.sha256(data).hexdigest()
        ext = Path(f.name).suffix.lower().lstrip(".")
        if ext not in SUPPORTED_TYPES:
            status = "unsupported type"
        elif not data:
            status = "empty file"
        elif len(data) / 1_048_576 > DEFAULT_MAX_FILE_MB:
            status = f"too large (limit {DEFAULT_MAX_FILE_MB} MB)"
        elif sha in seen:
            status = f"duplicate of {seen[sha]}"
        else:
            status, seen[sha] = "ok", f.name
        rows.append({"File": f.name, "Type": ext or "?", "Size": fmt_size(len(data)), "SHA-256": sha[:12], "Check": status})
    return rows


@st.cache_data(show_spinner=False, max_entries=16)
def preview_file(name: str, data: bytes, chunk_size: int, overlap: int, csv_overlap: int):
    """Extract + chunk ONE file for display only. Nothing is embedded or stored. Cached by content."""
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / Path(name).name
        path.write_bytes(data)
        doc = load_file(path, Path(tmp))
    chunks = chunk_document(doc, ChunkingConfig(chunk_size, overlap, csv_overlap))
    pages = len({s.page_number for s in doc.sections if s.page_number}) or None
    records = len(doc.sections) if doc.doc_type == "csv" else None
    first_text = "\n\n".join(s.text for s in doc.sections[:2])[:1200]
    return {
        "pages": pages, "records": records, "sections": len(doc.sections), "chunks": len(chunks),
        "text": first_text,
        "samples": [(c.metadata.get("page_number"), c.metadata.get("section"), c.text) for c in chunks[:3]],
    }

# --------------------------------------------------------------------------- live activity panel (fragment)
def render_activity_panel(embedder: ManagedEmbedder, manager: IngestManager) -> None:
    """Model status + live job progress. Runs as a fragment: it refreshes itself while work is in progress
    without re-running the rest of the page, and hands control back to the full page once everything is idle."""
    if embedder.ready:
        st.caption(
            f"Embedding model ready: {embedder.model_name} on {embedder.device} "
            f"(loaded in {fmt_secs(embedder.load_seconds)})."
        )
    elif embedder.loading:
        st.caption("Loading the embedding model in the background (once per server start). You can already choose files.")
    elif embedder.load_error:
        st.error(f"The embedding model could not be loaded: {embedder.load_error}")
        if st.button("Retry loading the model", key="retry_model"):
            embedder.warm_up_async()
            st.rerun()
    else:
        st.caption("The embedding model is loaded on first use.")

    job = manager.current
    if job is not None:
        snap = job.snapshot()
        if snap["status"] in ("pending", "running"):
            st.markdown(f"**{snap['label']}** &nbsp;|&nbsp; {html.escape(snap['stage'])} &nbsp;|&nbsp; {fmt_secs(snap['elapsed_s'])} elapsed")
            st.markdown(stage_chips(snap["stage"]), unsafe_allow_html=True)
            total, done = snap["files_total"], snap["files_done"]
            if total:
                current = f" - {Path(snap['current_file']).name}" if snap["current_file"] else ""
                st.progress(min(1.0, done / total), text=f"Files extracted and chunked: {done} of {total}{current}")
            else:
                st.progress(0.0, text="Scanning files and checking content hashes...")
            rate = snap["model_texts"] / snap["model_seconds"] if snap["model_seconds"] else 0.0
            st.markdown(
                stat_chips(
                    [
                        ("Chunks stored", snap["chunks_committed"]),
                        ("Embedded by model", snap["model_texts"]),
                        ("Reused from cache", snap["cache_hits"] + snap["deduplicated"]),
                        ("Model speed", f"{rate:.1f} chunks/s"),
                    ]
                ),
                unsafe_allow_html=True,
            )
            if st.button("Cancel indexing", key=f"cancel_{job.id}"):
                job.cancel()
        else:
            st.caption(f"{snap['label']} {snap['status']} in {fmt_secs(snap['elapsed_s'])}.")

    if st.session_state.get("poll_busy") and not (manager.busy or embedder.loading):
        st.session_state["poll_busy"] = False
        st.rerun()  # full page rerun: results, counts and the retrieval section pick up the finished state


# --------------------------------------------------------------------------- results of the last run
GOOD_STATUSES = {"indexed (new)", "updated", "unchanged (reused)"}


def render_results(manager: IngestManager) -> None:
    job = manager.current
    if job is None or job.status in ("pending", "running"):
        return
    snap = job.snapshot()
    if snap["status"] == "failed":
        st.error(f"{snap['label']} failed: {snap['error']}")
        with st.expander("Technical details"):
            st.code(snap["error_detail"])
        return
    if snap["status"] == "cancelled":
        st.warning(
            f"{snap['label']} was cancelled after {fmt_secs(snap['elapsed_s'])}. Everything already stored is kept; "
            "run it again to continue (vectors already computed are reused)."
        )
        return
    report = snap["report"]
    st.markdown(
        stat_chips(
            [
                (f"{snap['label']} ({report['mode']})", fmt_secs(snap["elapsed_s"])),
                ("Chunks in the ChromaDB collection", report["total_chunks_in_store"]),
            ]
        ),
        unsafe_allow_html=True,
    )
    for row in snap["file_results"]:
        name = html.escape(row["name"])
        if row["status"] in GOOD_STATUSES:
            extra = f" | {html.escape(row['detail'])}" if row["detail"] else ""
            right = (
                f'<span class="meta">{row["chunks"]} chunks &nbsp;|&nbsp; {html.escape(row["status"])} &nbsp;|&nbsp; '
                f'{fmt_size(row["size"])} &nbsp;|&nbsp; sha256 {row["sha256"]}{extra}</span>'
            )
            st.markdown(f'<div class="res"><span class="fname">{name}</span>{right}</div>', unsafe_allow_html=True)
        else:
            st.markdown(
                f'<div class="res bad"><span class="fname">{name}</span>'
                f'<span class="err">Could not be indexed ({html.escape(row["status"])}): {html.escape(row["detail"])}</span></div>',
                unsafe_allow_html=True,
            )
    in_batch = {row["relative"] for row in snap["file_results"]}
    others = [(path, why) for path, why in report["failures"] if path not in in_batch]
    others += [(path, "no extractable text") for path in report["empty_files"] if path not in in_batch]
    if others:
        with st.expander(f"{len(others)} other file(s) in the documents folder could not be indexed"):
            for path, why in others:
                st.write(f"- `{path}`: {why}")

    timings = report["timings_s"]
    section("Where the time went")
    st.dataframe(
        [
            {"Stage": "Hash files (change detection)", "Seconds": timings.get("hash", 0)},
            {"Stage": "Extract text (summed over workers)", "Seconds": timings.get("load", 0)},
            {"Stage": "Chunk text", "Seconds": timings.get("chunk", 0)},
            {"Stage": "Embed (cache lookup + model; includes model load if it happened here)", "Seconds": timings.get("embed", 0)},
            {"Stage": "ChromaDB reads and writes", "Seconds": timings.get("db", 0)},
            {"Stage": "Model load during this run", "Seconds": snap["model_load_s"]},
            {"Stage": "Total wall clock", "Seconds": snap["elapsed_s"]},
        ],
        hide_index=True,
    )
    st.caption(
        f"Chunks parsed: {report['chunks_generated']} | embedded by the model: {snap['model_texts']} | "
        f"reused from the vector cache: {snap['cache_hits'] + snap['deduplicated']} | "
        f"files not parsed because unchanged: {report['documents_skipped_without_parsing']}"
    )
    history = manager.history
    if len(history) >= 2:
        section("First run vs later runs")
        st.dataframe(history, hide_index=True)
        first, last = history[0], history[-1]
        st.caption(
            f"{last['run']} vs {first['run']}: wall clock {last['wall s']}s vs {first['wall s']}s; "
            f"chunks embedded by the model {last['embedded by model']} vs {first['embedded by model']}."
        )
    elif history:
        st.caption("Run the indexing again with unchanged files to compare it with this first run.")


# --------------------------------------------------------------------------- hero
st.markdown(
    """
    <div class="hero">
      <span class="tag">Milestone 1 prototype</span>
      <h1>AI Project Intelligence &amp; Risk Advisor</h1>
      <p>Drop in your project documents and surface the passages that matter: pending tasks, open risks and decisions, each traced back to its source.</p>
    </div>
    """,
    unsafe_allow_html=True,
)
track_slot = st.empty()

# --------------------------------------------------------------------------- start the RAG services
try:
    settings = RAGSettings.from_env()
    services = get_services(repr(settings), settings)
except Exception as exc:  # configuration errors
    st.error(f"Could not start the RAG pipeline: {exc}")
    st.stop()
pipeline, embedder, manager = services.pipeline, services.embedder, services.manager

# --------------------------------------------------------------------------- 1. upload
section("Upload project documents")
st.caption("PDF, DOCX, CSV or TXT")

project_name = st.text_input(
    "Project name (optional)",
    placeholder="Project Alpha",
    help="Used to tag the chunks so you can filter by project. CSV files with a Project column override this per row.",
)
project_id = slugify_project_id(project_name) if project_name.strip() else "unassigned"

uploaded_files = st.file_uploader(
    "Upload documents",
    type=SUPPORTED_TYPES,
    accept_multiple_files=True,
    label_visibility="collapsed",
)

if uploaded_files:
    checks = validate_selection(uploaded_files)
    st.dataframe(checks, hide_index=True, use_container_width=True)
    problems = [r for r in checks if r["Check"] != "ok"]
    if problems:
        st.warning(f"{len(problems)} of {len(checks)} selected file(s) will be skipped or rejected (see the Check column).")
    if st.checkbox("Preview extracted text and chunks (no embedding, nothing is stored)", key="show_preview"):
        for f in uploaded_files[:5]:
            try:
                info = preview_file(f.name, f.getvalue(), settings.chunk_size_tokens, settings.chunk_overlap_tokens, settings.csv_overlap_rows)
            except Exception as exc:
                st.error(f"{f.name}: could not be extracted ({exc})")
                continue
            counts = [f"{info['chunks']} chunks"] + ([f"{info['pages']} pages"] if info["pages"] else []) + ([f"{info['records']} records"] if info["records"] else [])
            with st.expander(f"{f.name} - " + ", ".join(counts)):
                st.caption("Extracted text (start)")
                st.text(info["text"] or "(no text)")
                for number, (page, sec, text) in enumerate(info["samples"], start=1):
                    st.caption(f"Chunk {number}" + (f" - page {page}" if page else "") + (f" - section {sec}" if sec else ""))
                    st.text(text[:900])
        if len(uploaded_files) > 5:
            st.caption(f"Preview limited to the first 5 of {len(uploaded_files)} files.")

# Indexing only ever starts from this button; a plain rerun (typing in a box, moving a slider) never indexes.
if st.button("Extract and index", disabled=not uploaded_files or manager.busy):
    try:
        with st.spinner("Saving files and computing content hashes..."):
            items = prepare_uploads(
                uploaded_files, settings.documents_dir / "uploads" / project_id, settings.documents_dir,
                known_hashes=manager.indexed_hashes(),
            )
        manager.start(uploads=items)
    except Exception as exc:
        st.error(f"Could not start indexing: {exc}")

busy = manager.busy or embedder.loading
st.session_state["poll_busy"] = busy
st.fragment(run_every=1.0 if busy else None)(render_activity_panel)(embedder, manager)
render_results(manager)

# --------------------------------------------------------------------------- full re-index (explicit, confirmed)
with st.expander("Advanced: delete the index and start over"):
    st.warning(
        f"This removes every vector in collection '{settings.collection_name}'. Your files on disk are NOT deleted and "
        "nothing is re-indexed automatically: upload the files you want again and select Extract and index."
    )
    confirm_box = st.checkbox("I understand that the indexed data of this collection will be deleted", key="confirm_reset")
    typed = st.text_input("Type RESET to confirm", key="confirm_reset_text")
    if st.button("Delete index", disabled=not (confirm_box and typed.strip() == "RESET") or manager.busy or embedder.loading):
        try:
            pipeline.reset_index()
            st.success("The collection was reset. Upload documents and select Extract and index to rebuild it.")
        except Exception as exc:
            st.error(f"Reset failed: {exc}")

# --------------------------------------------------------------------------- 2. retrieval
section("Test retrieval from indexed documents")
st.write("This is a retrieval preview, not the final conversational AI assistant.")

indexed_count = current_indexed_count()

if indexed_count is None:
    st.markdown(
        '<div class="note">The index status appears once the embedding model has loaded (first start only).</div>',
        unsafe_allow_html=True,
    )
elif indexed_count == 0:
    st.markdown(
        '<div class="note">Nothing is indexed yet. Upload documents above and select <b>Extract and index</b>.</div>',
        unsafe_allow_html=True,
    )

query = st.text_input(
    "Enter a question or search phrase",
    placeholder="What are the pending project tasks?",
)

col1, col2 = st.columns(2)
top_k = col1.slider("Chunks to retrieve", 1, 15, 5)
filter_project = col2.text_input("Filter by project id (optional)", placeholder="project_alpha")
generate_answer = st.checkbox("Also generate an answer with the LLM (needs GROQ_API_KEY / ANTHROPIC_API_KEY)")

if "searched" not in st.session_state:
    st.session_state.searched = False

if st.button("Find relevant chunks", disabled=not indexed_count or not query.strip()):
    project_ids = [filter_project.strip()] if filter_project.strip() else None
    try:
        if generate_answer:
            with st.spinner("Searching and generating answer..."):
                result = pipeline.ask(query, top_k=top_k, project_ids=project_ids)
            st.session_state.searched = True
            if result.error:
                st.error(result.error)
            section("Answer")
            st.markdown(
                f'<div class="answer"><div class="lbl">Generated from the passages below</div>'
                f"{html.escape(result.answer or '').replace(chr(10), '<br>')}</div>",
                unsafe_allow_html=True,
            )
            for warning in result.warnings:
                st.warning(warning)
            passages = result.sources
            section(f"Relevant chunks ({len(passages)})")
            if not passages:
                st.info("No relevant chunks found.")
            for s in passages:
                label = f"[{s.citation_id}] {s.source_file}" + (f" - {s.location}" if s.location else "")
                if s.cited:
                    label += "  (cited)"
                with st.expander(label):
                    st.markdown(proximity_bar(s.distance), unsafe_allow_html=True)
                    st.caption(f"project: {s.project_id} | distance: {s.distance:.3f} (lower = closer)")
                    st.text(s.text)
        else:
            with st.spinner("Searching..."):
                hits = pipeline.retrieve(query, top_k=top_k, project_ids=project_ids)
            st.session_state.searched = True
            section(f"Relevant chunks ({len(hits)})")
            if not hits:
                st.info("No relevant chunks found.")
            for number, hit in enumerate(hits, start=1):
                meta = hit.metadata
                location = describe_location(meta)
                label = f"[{number}] {meta.get('source_file', '')}" + (f" - {location}" if location else "")
                with st.expander(label, expanded=number == 1):
                    st.markdown(proximity_bar(hit.distance), unsafe_allow_html=True)
                    st.caption(
                        f"project: {meta.get('project_id', '')} | type: {meta.get('doc_type', '')} | "
                        f"distance: {hit.distance:.3f} (lower = closer)"
                    )
                    st.text(hit.text)
    except Exception as exc:
        st.error(str(exc))

# progress track at the top reflects the final state of this run
track_slot.markdown(
    render_track(bool(uploaded_files), bool(indexed_count), st.session_state.searched),
    unsafe_allow_html=True,
)