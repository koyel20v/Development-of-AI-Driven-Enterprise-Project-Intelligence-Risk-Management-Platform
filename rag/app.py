# """AI Project Intelligence & Risk Advisor - Milestone 1 UI (restyled).

# Place this file in the rag/ folder (next to src/ and scripts/), then run:

#     python -m pip install streamlit
#     python -m streamlit run app.py

# Flow
# ----
# 1. Upload one or more PDF / DOCX / CSV / TXT files.
# 2. "Extract and index" saves them under documents/uploads/<project>/, shows how many
#    characters and chunks each file produced, and indexes them into ChromaDB.
# 3. Type a question / search phrase and see the most relevant chunks (retrieval preview).

# Tip: for a consistent look in every browser, create .streamlit/config.toml with
#     [theme]
#     base = "light"
# """

# import html
# import sys
# from pathlib import Path

# import streamlit as st

# sys.path.insert(0, str(Path(__file__).resolve().parent))

# from src.document_loader import DocumentLoadError, EmptyDocumentError, load_file, slugify_project_id  # noqa: E402
# from src.llm import describe_location  # noqa: E402
# from src.rag_pipeline import RAGPipeline  # noqa: E402
# from src.text_chunker import ChunkingConfig, chunk_document  # noqa: E402

# SUPPORTED_TYPES = ["pdf", "docx", "csv", "txt"]

# st.set_page_config(page_title="AI Project Intelligence & Risk Advisor", page_icon="◭", layout="centered")

# # --------------------------------------------------------------------------- styling
# # Palette: chart paper #EDF1F3, ink navy #14213D, signal amber #F2A900,
# # steel teal #0F766E, rule grey #C9D3D9, muted slate #5B6B7A
# CSS = """
# <style>
# @import url('https://fonts.googleapis.com/css2?family=Bricolage+Grotesque:opsz,wght@12..96,500;12..96,700;12..96,800&family=IBM+Plex+Sans:wght@400;500;600&display=swap');

# :root {
#   --paper: #EDF1F3; --card: #FFFFFF; --ink: #14213D; --slate: #5B6B7A;
#   --rule: #C9D3D9; --amber: #F2A900; --teal: #0F766E; --red: #B42318;
# }

# html, body, [class*="css"], .stApp, .stMarkdown, label, input, textarea, button {
#   font-family: 'IBM Plex Sans', system-ui, sans-serif;
# }
# .stApp {
#   background-color: var(--paper);
#   background-image:
#     linear-gradient(rgba(20,33,61,.045) 1px, transparent 1px),
#     linear-gradient(90deg, rgba(20,33,61,.045) 1px, transparent 1px);
#   background-size: 32px 32px;
#   color: var(--ink);
# }
# #MainMenu, footer, [data-testid="stHeader"], [data-testid="stToolbar"] { display: none !important; }
# .block-container { max-width: 860px; padding-top: 2.2rem; padding-bottom: 5rem; }

# .stApp p, .stApp label, .stApp span, .stApp li { color: var(--ink); }
# [data-testid="stCaptionContainer"], [data-testid="stCaptionContainer"] * { color: var(--slate) !important; }

# /* hero */
# .hero { background: var(--ink); border-radius: 6px; padding: 2.2rem 2.2rem 1.9rem; position: relative; overflow: hidden; }
# .hero::after {
#   content: ""; position: absolute; right: -60px; top: -60px; width: 220px; height: 220px;
#   border: 1px solid rgba(242,169,0,.55); border-radius: 50%;
#   box-shadow: 0 0 0 34px rgba(242,169,0,.07), 0 0 0 68px rgba(242,169,0,.04);
# }
# .hero h1 {
#   font-family: 'Bricolage Grotesque', sans-serif; font-weight: 800; color: #fff !important;
#   font-size: clamp(1.9rem, 5vw, 2.9rem); line-height: 1.05; letter-spacing: -0.02em; margin: 0 0 .7rem; padding: 0;
# }
# .hero p { color: #B9C4D4 !important; max-width: 520px; margin: 0; font-size: 1.02rem; line-height: 1.55; }
# .hero .tag {
#   display: inline-block; background: var(--amber); color: var(--ink) !important; font-weight: 600;
#   font-size: .78rem; padding: .18rem .6rem; border-radius: 3px; margin-bottom: 1rem;
# }

# /* progress track */
# .track { display: flex; margin: 1.1rem 0 .4rem; background: var(--card); border: 1px solid var(--rule); border-radius: 6px; }
# .track .stop { flex: 1; padding: .85rem 1rem; border-right: 1px solid var(--rule); display: flex; gap: .7rem; align-items: center; }
# .track .stop:last-child { border-right: 0; }
# .track .dot {
#   width: 26px; height: 26px; border-radius: 50%; border: 2px solid var(--rule); color: var(--slate);
#   display: grid; place-items: center; font-weight: 600; font-size: .8rem; flex: none;
# }
# .track .name { font-weight: 600; font-size: .92rem; line-height: 1.2; }
# .track .sub { color: var(--slate); font-size: .78rem; }
# .track .stop.done .dot { background: var(--teal); border-color: var(--teal); color: #fff; }
# .track .stop.now .dot { border-color: var(--amber); background: var(--amber); color: var(--ink); }
# @media (max-width: 640px) { .track { flex-direction: column; } .track .stop { border-right: 0; border-bottom: 1px solid var(--rule); } }

# /* section headings */
# .sec { margin: 2.4rem 0 .3rem; display: flex; align-items: center; gap: .8rem; }
# .sec h2 { font-family: 'Bricolage Grotesque', sans-serif; font-weight: 700; font-size: 1.55rem; letter-spacing: -0.01em; margin: 0; padding: 0; color: var(--ink) !important; }
# .sec .bar { flex: 1; height: 1px; background: var(--rule); }

# /* inputs */
# .stTextInput input {
#   background: var(--card) !important; border: 1px solid var(--rule) !important; border-radius: 4px !important;
#   color: var(--ink) !important; padding: .65rem .8rem !important;
# }
# .stTextInput input:focus { border-color: var(--ink) !important; box-shadow: 0 0 0 3px rgba(242,169,0,.35) !important; }
# .stTextInput label p, .stSlider label p, .stCheckbox label p { font-weight: 600; font-size: .9rem; }

# [data-testid="stFileUploaderDropzone"] {
#   background: var(--card); border: 2px dashed #8FA0AE; border-radius: 6px; padding: 1.6rem;
#   transition: border-color .15s, background .15s;
# }
# [data-testid="stFileUploaderDropzone"]:hover { border-color: var(--amber); background: #FFFBF0; }
# [data-testid="stFileUploaderDropzone"] * { color: var(--ink) !important; }

# /* buttons */
# .stButton > button {
#   background: var(--ink); color: #fff !important; border: 0; border-radius: 4px; padding: .65rem 1.5rem;
#   font-weight: 600; letter-spacing: .01em; transition: transform .1s, background .15s;
# }
# .stButton > button p { color: #fff !important; }
# .stButton > button:hover:not(:disabled) { background: #1F3260; transform: translateY(-1px); }
# .stButton > button:focus-visible { outline: 3px solid var(--amber); outline-offset: 2px; }
# .stButton > button:disabled { background: #AAB6C0; }

# /* result rows */
# .res { display: flex; align-items: center; justify-content: space-between; gap: 1rem; flex-wrap: wrap;
#   background: var(--card); border: 1px solid var(--rule); border-left: 5px solid var(--teal);
#   border-radius: 4px; padding: .75rem 1rem; margin: .5rem 0; }
# .res.bad { border-left-color: var(--red); }
# .res .fname { font-weight: 600; word-break: break-all; }
# .res .meta { color: var(--slate); font-size: .88rem; }
# .res .err { color: var(--red); font-size: .88rem; }

# .stat { display: inline-block; background: var(--ink); color: #fff; border-radius: 4px; padding: .35rem .8rem; margin-top: .4rem; font-size: .88rem; }
# .stat b { color: var(--amber); font-family: 'Bricolage Grotesque', sans-serif; font-size: 1.05rem; }

# /* notices */
# .note { background: #FFF6DB; border: 1px solid #EBCB6B; border-radius: 4px; padding: .8rem 1rem; margin: .8rem 0; }

# /* answer */
# .answer { background: var(--card); border: 1px solid var(--ink); border-radius: 6px; padding: 1.2rem 1.4rem; margin: .6rem 0 1rem; box-shadow: 6px 6px 0 var(--amber); }
# .answer .lbl { font-weight: 600; color: var(--slate); font-size: .85rem; margin-bottom: .4rem; }

# /* expanders = evidence cards */
# [data-testid="stExpander"] { background: var(--card); border: 1px solid var(--rule) !important; border-radius: 4px; margin-bottom: .55rem; }
# [data-testid="stExpander"] summary { font-weight: 600; }
# [data-testid="stExpander"] summary:hover { background: #F6F8F9; }
# .prox { height: 6px; border-radius: 3px; background: #E3E9ED; margin: .2rem 0 .7rem; overflow: hidden; }
# .prox > div { height: 100%; background: linear-gradient(90deg, var(--teal), var(--amber)); }

# [data-testid="stSlider"] [role="slider"] { background: var(--ink); }
# :focus-visible { outline-color: var(--amber); }
# @media (prefers-reduced-motion: reduce) { * { transition: none !important; } }
# </style>
# """
# st.markdown(CSS, unsafe_allow_html=True)


# def section(title: str) -> None:
#     st.markdown(f'<div class="sec"><h2>{html.escape(title)}</h2><div class="bar"></div></div>', unsafe_allow_html=True)


# def proximity_bar(distance: float) -> str:
#     """Visual closeness: distance 0 fills the bar, distance >= 1 empties it."""
#     pct = max(4, min(100, round((1 - min(max(distance, 0.0), 1.0)) * 100)))
#     return f'<div class="prox"><div style="width:{pct}%"></div></div>'


# def render_track(has_files: bool, indexed: bool, searched: bool) -> str:
#     steps = [
#         ("Upload", "Add your documents", has_files or indexed),
#         ("Index", "Chunk and embed", indexed),
#         ("Ask", "Search the evidence", searched),
#     ]
#     first_open = next((i for i, s in enumerate(steps) if not s[2]), -1)
#     parts = []
#     for i, (name, sub, done) in enumerate(steps):
#         cls = "done" if done else ("now" if i == first_open else "")
#         mark = "✓" if done else str(i + 1)
#         parts.append(
#             f'<div class="stop {cls}"><div class="dot">{mark}</div>'
#             f'<div><div class="name">{name}</div><div class="sub">{sub}</div></div></div>'
#         )
#     return f'<div class="track">{"".join(parts)}</div>'


# # --------------------------------------------------------------------------- pipeline helpers
# @st.cache_resource(show_spinner="Loading embedding model...")
# def get_pipeline() -> RAGPipeline:
#     return RAGPipeline()


# def save_uploads(uploaded_files, target_dir: Path) -> list[Path]:
#     """Write the uploaded files to disk and return their paths."""
#     target_dir.mkdir(parents=True, exist_ok=True)
#     paths = []
#     for uploaded in uploaded_files:
#         safe_name = Path(uploaded.name).name  # strip any directory parts
#         path = target_dir / safe_name
#         path.write_bytes(uploaded.getbuffer())
#         paths.append(path)
#     return paths


# def process_file(path: Path, root: Path, config: ChunkingConfig) -> dict:
#     """Extract text and count chunks for one file. Never raises: errors are returned."""
#     try:
#         doc = load_file(path, root)
#         chunks = chunk_document(doc, config)
#         chars = sum(len(section.text) for section in doc.sections)
#         return {"name": path.name, "ok": True, "chars": chars, "chunks": len(chunks), "project": doc.project_id}
#     except EmptyDocumentError as exc:
#         return {"name": path.name, "ok": False, "error": f"no extractable text ({exc})"}
#     except (DocumentLoadError, OSError) as exc:
#         return {"name": path.name, "ok": False, "error": str(exc)}
#     except Exception as exc:  # unexpected parser failure
#         return {"name": path.name, "ok": False, "error": f"{type(exc).__name__}: {exc}"}


# # --------------------------------------------------------------------------- hero
# st.markdown(
#     """
#     <div class="hero">
#       <span class="tag">Milestone 1 prototype</span>
#       <h1>AI Project Intelligence &amp; Risk Advisor</h1>
#       <p>Drop in your project documents and surface the passages that matter: pending tasks, open risks and decisions, each traced back to its source.</p>
#     </div>
#     """,
#     unsafe_allow_html=True,
# )
# track_slot = st.empty()

# try:
#     pipeline = get_pipeline()
# except Exception as exc:  # config / model-load errors
#     st.error(f"Could not start the RAG pipeline: {exc}")
#     st.stop()

# settings = pipeline.settings if hasattr(pipeline, "settings") else None
# if settings is None:  # fall back to a fresh settings object if the pipeline does not expose it
#     from src.config import RAGSettings

#     settings = RAGSettings.from_env()

# # --------------------------------------------------------------------------- 1. upload
# section("Upload project documents")
# st.caption("PDF, DOCX, CSV or TXT")

# project_name = st.text_input(
#     "Project name (optional)",
#     placeholder="Project Alpha",
#     help="Used to tag the chunks so you can filter by project. CSV files with a Project column override this per row.",
# )
# project_id = slugify_project_id(project_name) if project_name.strip() else "unassigned"

# uploaded_files = st.file_uploader(
#     "Upload documents",
#     type=SUPPORTED_TYPES,
#     accept_multiple_files=True,
#     label_visibility="collapsed",
# )

# if "index_results" not in st.session_state:
#     st.session_state.index_results = []

# if st.button("Extract and index", disabled=not uploaded_files):
#     upload_dir = settings.documents_dir / "uploads" / project_id
#     config = ChunkingConfig(
#         chunk_size_tokens=settings.chunk_size_tokens,
#         chunk_overlap_tokens=settings.chunk_overlap_tokens,
#         csv_overlap_rows=settings.csv_overlap_rows,
#     )
#     with st.spinner("Extracting text, chunking and embedding..."):
#         paths = save_uploads(uploaded_files, upload_dir)
#         results = [process_file(p, settings.documents_dir, config) for p in paths]
#         try:
#             report = pipeline.ingest()
#             st.session_state.ingest_failures = dict(report.failures)
#             st.session_state.total_chunks = report.total_chunks_in_store
#         except Exception as exc:
#             st.session_state.ingest_failures = {}
#             st.session_state.total_chunks = None
#             st.error(f"Indexing failed: {exc}")
#             results = []
#     st.session_state.index_results = results

# for res in st.session_state.index_results:
#     name = html.escape(res["name"])
#     if res["ok"]:
#         st.markdown(
#             f'<div class="res"><span class="fname">{name}</span>'
#             f'<span class="meta">{res["chars"]:,} characters &nbsp;|&nbsp; {res["chunks"]} chunks indexed</span></div>',
#             unsafe_allow_html=True,
#         )
#     else:
#         st.markdown(
#             f'<div class="res bad"><span class="fname">{name}</span>'
#             f'<span class="err">Could not be indexed: {html.escape(str(res["error"]))}</span></div>',
#             unsafe_allow_html=True,
#         )

# if st.session_state.get("total_chunks") is not None and st.session_state.index_results:
#     st.markdown(
#         f'<div class="stat">Chunks in the ChromaDB collection: <b>{st.session_state.total_chunks}</b></div>',
#         unsafe_allow_html=True,
#     )

# # --------------------------------------------------------------------------- 2. retrieval
# section("Test retrieval from indexed documents")
# st.write("This is a retrieval preview, not the final conversational AI assistant.")

# try:
#     indexed_count = pipeline.store.count()
# except Exception:
#     indexed_count = 0

# if indexed_count == 0:
#     st.markdown(
#         '<div class="note">Nothing is indexed yet. Upload documents above and select <b>Extract and index</b>.</div>',
#         unsafe_allow_html=True,
#     )

# query = st.text_input(
#     "Enter a question or search phrase",
#     placeholder="What are the pending project tasks?",
# )

# col1, col2 = st.columns(2)
# top_k = col1.slider("Chunks to retrieve", 1, 15, 5)
# filter_project = col2.text_input("Filter by project id (optional)", placeholder="project_alpha")
# generate_answer = st.checkbox("Also generate an answer with the LLM (needs GROQ_API_KEY / ANTHROPIC_API_KEY)")

# if "searched" not in st.session_state:
#     st.session_state.searched = False

# if st.button("Find relevant chunks", disabled=indexed_count == 0 or not query.strip()):
#     project_ids = [filter_project.strip()] if filter_project.strip() else None
#     try:
#         if generate_answer:
#             with st.spinner("Searching and generating answer..."):
#                 result = pipeline.ask(query, top_k=top_k, project_ids=project_ids)
#             st.session_state.searched = True
#             if result.error:
#                 st.error(result.error)
#             section("Answer")
#             st.markdown(
#                 f'<div class="answer"><div class="lbl">Generated from the passages below</div>'
#                 f"{html.escape(result.answer or '').replace(chr(10), '<br>')}</div>",
#                 unsafe_allow_html=True,
#             )
#             for warning in result.warnings:
#                 st.warning(warning)
#             passages = result.sources
#             section(f"Relevant chunks ({len(passages)})")
#             if not passages:
#                 st.info("No relevant chunks found.")
#             for s in passages:
#                 label = f"[{s.citation_id}] {s.source_file}" + (f" - {s.location}" if s.location else "")
#                 if s.cited:
#                     label += "  (cited)"
#                 with st.expander(label):
#                     st.markdown(proximity_bar(s.distance), unsafe_allow_html=True)
#                     st.caption(f"project: {s.project_id} | distance: {s.distance:.3f} (lower = closer)")
#                     st.text(s.text)
#         else:
#             with st.spinner("Searching..."):
#                 hits = pipeline.retrieve(query, top_k=top_k, project_ids=project_ids)
#             st.session_state.searched = True
#             section(f"Relevant chunks ({len(hits)})")
#             if not hits:
#                 st.info("No relevant chunks found.")
#             for number, hit in enumerate(hits, start=1):
#                 meta = hit.metadata
#                 location = describe_location(meta)
#                 label = f"[{number}] {meta.get('source_file', '')}" + (f" - {location}" if location else "")
#                 with st.expander(label, expanded=number == 1):
#                     st.markdown(proximity_bar(hit.distance), unsafe_allow_html=True)
#                     st.caption(
#                         f"project: {meta.get('project_id', '')} | type: {meta.get('doc_type', '')} | "
#                         f"distance: {hit.distance:.3f} (lower = closer)"
#                     )
#                     st.text(hit.text)
#     except Exception as exc:
#         st.error(str(exc))

# # progress track at the top reflects the final state of this run
# track_slot.markdown(
#     render_track(bool(uploaded_files), indexed_count > 0, st.session_state.searched),
#     unsafe_allow_html=True,
# )

"""AI Project Intelligence & Risk Advisor - Milestone 1 UI (restyled).

Place this file in the rag/ folder (next to src/ and scripts/), then run:

    python -m pip install streamlit
    python -m streamlit run app.py

Flow
----
1. Upload one or more PDF / DOCX / CSV / TXT files.
2. "Extract and index" saves them under documents/uploads/<project>/, shows how many
   characters and chunks each file produced, and indexes them into ChromaDB.
3. Type a question / search phrase and see the most relevant chunks (retrieval preview).

Tip: for a consistent look in every browser, create .streamlit/config.toml with
    [theme]
    base = "light"
"""

import html
import sys
from pathlib import Path

import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parent))

from src.document_loader import DocumentLoadError, EmptyDocumentError, load_file, slugify_project_id  # noqa: E402
from src.llm import describe_location  # noqa: E402
from src.rag_pipeline import RAGPipeline  # noqa: E402
from src.text_chunker import ChunkingConfig, chunk_document  # noqa: E402

SUPPORTED_TYPES = ["pdf", "docx", "csv", "txt"]

st.set_page_config(page_title="AI Project Intelligence & Risk Advisor", page_icon="◭", layout="centered")

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


# --------------------------------------------------------------------------- pipeline helpers
@st.cache_resource(show_spinner="Loading embedding model...")
def get_pipeline() -> RAGPipeline:
    return RAGPipeline()


def save_uploads(uploaded_files, target_dir: Path) -> list[Path]:
    """Write the uploaded files to disk and return their paths."""
    target_dir.mkdir(parents=True, exist_ok=True)
    paths = []
    for uploaded in uploaded_files:
        safe_name = Path(uploaded.name).name  # strip any directory parts
        path = target_dir / safe_name
        path.write_bytes(uploaded.getbuffer())
        paths.append(path)
    return paths


def process_file(path: Path, root: Path, config: ChunkingConfig) -> dict:
    """Extract text and count chunks for one file. Never raises: errors are returned."""
    try:
        doc = load_file(path, root)
        chunks = chunk_document(doc, config)
        chars = sum(len(section.text) for section in doc.sections)
        return {"name": path.name, "ok": True, "chars": chars, "chunks": len(chunks), "project": doc.project_id}
    except EmptyDocumentError as exc:
        return {"name": path.name, "ok": False, "error": f"no extractable text ({exc})"}
    except (DocumentLoadError, OSError) as exc:
        return {"name": path.name, "ok": False, "error": str(exc)}
    except Exception as exc:  # unexpected parser failure
        return {"name": path.name, "ok": False, "error": f"{type(exc).__name__}: {exc}"}


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

try:
    pipeline = get_pipeline()
except Exception as exc:  # config / model-load errors
    st.error(f"Could not start the RAG pipeline: {exc}")
    st.stop()

settings = pipeline.settings if hasattr(pipeline, "settings") else None
if settings is None:  # fall back to a fresh settings object if the pipeline does not expose it
    from src.config import RAGSettings

    settings = RAGSettings.from_env()

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

if "index_results" not in st.session_state:
    st.session_state.index_results = []

if st.button("Extract and index", disabled=not uploaded_files):
    upload_dir = settings.documents_dir / "uploads" / project_id
    config = ChunkingConfig(
        chunk_size_tokens=settings.chunk_size_tokens,
        chunk_overlap_tokens=settings.chunk_overlap_tokens,
        csv_overlap_rows=settings.csv_overlap_rows,
    )
    with st.spinner("Extracting text, chunking and embedding..."):
        paths = save_uploads(uploaded_files, upload_dir)
        results = [process_file(p, settings.documents_dir, config) for p in paths]
        try:
            report = pipeline.ingest()
            st.session_state.ingest_failures = dict(report.failures)
            st.session_state.total_chunks = report.total_chunks_in_store
        except Exception as exc:
            st.session_state.ingest_failures = {}
            st.session_state.total_chunks = None
            st.error(f"Indexing failed: {exc}")
            results = []
    st.session_state.index_results = results

for res in st.session_state.index_results:
    name = html.escape(res["name"])
    if res["ok"]:
        st.markdown(
            f'<div class="res"><span class="fname">{name}</span>'
            f'<span class="meta">{res["chars"]:,} characters &nbsp;|&nbsp; {res["chunks"]} chunks indexed</span></div>',
            unsafe_allow_html=True,
        )
    else:
        st.markdown(
            f'<div class="res bad"><span class="fname">{name}</span>'
            f'<span class="err">Could not be indexed: {html.escape(str(res["error"]))}</span></div>',
            unsafe_allow_html=True,
        )

if st.session_state.get("total_chunks") is not None and st.session_state.index_results:
    st.markdown(
        f'<div class="stat">Chunks in the ChromaDB collection: <b>{st.session_state.total_chunks}</b></div>',
        unsafe_allow_html=True,
    )

# --------------------------------------------------------------------------- 2. retrieval
section("Test retrieval from indexed documents")
st.write("This is a retrieval preview, not the final conversational AI assistant.")

try:
    indexed_count = pipeline.store.count()
except Exception:
    indexed_count = 0

if indexed_count == 0:
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

if st.button("Find relevant chunks", disabled=indexed_count == 0 or not query.strip()):
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
    render_track(bool(uploaded_files), indexed_count > 0, st.session_state.searched),
    unsafe_allow_html=True,
)