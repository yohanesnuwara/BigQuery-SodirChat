import os, sys, re
from typing import List, Tuple, Optional
import pandas as pd
import streamlit as st
from google.cloud import bigquery

# ---- .env (optional) ----
try:
    from dotenv import load_dotenv
    load_dotenv()
except Exception:
    pass

# ---- paths / imports ----
ROOT = os.path.dirname(os.path.abspath(__file__))         # src/
PROJECT_ROOT = os.path.dirname(ROOT)                      # repo root
LOGO_PATH = os.path.join(PROJECT_ROOT, "static", "sodirchat-logo.png")

if os.path.dirname(ROOT) not in sys.path:
    sys.path.append(os.path.dirname(ROOT))

from src.utils_bq_vec import (
    get_bq_client, init_vertex, get_embed_model, clip_text,
    vector_search_sql_dual, vector_search_sql_docs_only,
    PROJECT, BQ_LOCATION, VERTEX_LOCATION, DATASET
)

# ================= UI FRAME =================
st.set_page_config(page_title="SodirChat — BigQuery + Vertex AI", page_icon="🛢️", layout="wide")
st.markdown("<h1 style='margin-bottom:0.1rem;'>SodirChat</h1>", unsafe_allow_html=True)
st.caption("Chat for Norwegian Offshore Directorate Sodir Database - Powered by Google BigQuery")

# ---- sidebar (logo + controls) ----
with st.sidebar:
    if os.path.exists(LOGO_PATH):
        st.image(LOGO_PATH, use_container_width=True)
        st.markdown("---")

    top_k = st.slider("Top-K total (per source auto-tuned)", 1, 40, 8)
    show_snippets = st.checkbox("Show snippets for each hit", value=True)
    max_chunk_chars = st.number_input("Max chars per chunk", 200, 4000, 1600, step=100)
    max_total_chars = st.number_input("Max total context chars", 1000, 40000, 15000, step=500)
    answer_tokens = st.slider("Answer max tokens", 256, 8192, 5000, step=128)
    if st.button("🧹 Clear chat"):
        st.session_state.pop("history", None)
        st.rerun()

DO_SYNTH = True  # synthesis on for non-image queries

# ================= SERVICES =================
@st.cache_resource(show_spinner=False)
def _bq():
    return get_bq_client()

@st.cache_resource(show_spinner=False)
def _embed_model():
    init_vertex()
    return get_embed_model()  # text-embedding-005 (Vertex)

def _gemini_model():
    from vertexai import init as vinit
    from vertexai.generative_models import GenerativeModel
    vinit(project=PROJECT, location=VERTEX_LOCATION)
    return GenerativeModel("gemini-2.5-flash")

# ================= SAFE STRING HELPERS =================
def _s(x) -> str:
    """Safe string: NaN/None/non-str -> ''. Avoids 'boolean value of NA is ambiguous'."""
    return x if isinstance(x, str) else ""

def _link(u: str) -> str:
    u = _s(u)
    return f"[link]({u})" if u.startswith("http") else u

# ================= IMAGE INTENT / RENDERING =================
def wants_imgs(q: str) -> bool:
    q = (q or "").lower()
    return any(tok in q for tok in ["corephoto", "core photo", "map", "factmap", "core", "photo"])

def render_img_answer(hits: pd.DataFrame, limit: int = 30) -> str:
    imgs = hits[hits["source"] == "imgs"].copy()
    if imgs.empty:
        return "No core photos or maps found in the top results."
    lines = []
    for well, grp in imgs.groupby(imgs["wellname"].apply(_s)):
        well_hdr = f"Core photos / maps for well {well or '(unknown)'}:"
        lines.append(well_hdr)
        for _, r in grp.head(limit).iterrows():
            typ = _s(r.get("type")).capitalize()
            fn  = _s(r.get("filename"))
            url = _s(r.get("url"))
            label = f"- {typ} {fn}".strip()
            lines.append(f"{label} {url}")
        lines.append("")
    return "\n".join(lines).strip()

# ================= WELLNAME DETECTION =================
WELL_RE = re.compile(r"\b\d+/\d+-\d+\s*[A-Z]?\b")  # e.g., 15/9-13, 31/2-22 S
def extract_wellname(q: str) -> Optional[str]:
    m = WELL_RE.search(q or "")
    if not m:
        return None
    return m.group(0).strip()

# ================= VECTOR SEARCH HELPERS =================
def embed_question(q: str) -> List[float]:
    model = _embed_model()
    q = clip_text(q)
    resp = model.get_embeddings([q])[0]
    return [float(x) for x in resp.values]

def _pick_k(user_q: str, top_k_total: int, has_well: bool) -> Tuple[int, int]:
    if wants_imgs(user_q):
        k_docs = max(4, top_k_total // 3)
        k_imgs = max(8, top_k_total)       # favor images a lot
    else:
        # If a specific well is detected, grab more docs from that well
        k_docs = max(8, top_k_total + (8 if has_well else 0))
        k_imgs = max(3, top_k_total // 3)
    return k_docs, k_imgs

def run_vector_search(q_emb: List[float], user_q: str, top_k_total: int) -> pd.DataFrame:
    wn = extract_wellname(user_q)
    has_well = wn is not None
    k_docs, k_imgs = _pick_k(user_q, top_k_total, has_well)

    sql = vector_search_sql_dual(top_k_docs=k_docs, top_k_imgs=k_imgs, filter_by_well=bool(wn))
    params = [bigquery.ArrayQueryParameter("q", "FLOAT64", q_emb)]
    if wn:
        params.append(bigquery.ScalarQueryParameter("wn", "STRING", wn))

    bq = _bq()
    try:
        job = bq.query(
            sql,
            job_config=bigquery.QueryJobConfig(query_parameters=params),
        )
        df = job.result().to_dataframe(create_bqstorage_client=True)
        return df
    except Exception as e:
        st.info(f"Images search unavailable or schema mismatch ({e}). Falling back to docs only.")
        sql_docs = vector_search_sql_docs_only(top_k=top_k_total)
        job = bq.query(
            sql_docs,
            job_config=bigquery.QueryJobConfig(
                query_parameters=[bigquery.ArrayQueryParameter("q", "FLOAT64", q_emb)]
            ),
        )
        return job.result().to_dataframe(create_bqstorage_client=True)

# ================= LLM SYNTHESIS (for non-image queries) =================
def _strip_empty_bullets(md: str) -> str:
    cleaned = []
    for line in (md or "").splitlines():
        L = line.strip()
        if L in {"-", "*", "•"}:
            continue
        cleaned.append(line)
    return "\n".join(cleaned)

def build_context(hits_df: pd.DataFrame, max_chunk_chars=1600, max_total_chars=15000) -> str:
    pieces = []
    for _, r in hits_df.iterrows():
        url = _s(r.get("url"))
        title = _s(r.get("title"))
        content = _s(r.get("content"))[:max_chunk_chars]
        pieces.append(f"URL: {url}\nTITLE: {title}\nTEXT: {content}")
    ctx = "\n\n---\n\n".join(pieces)
    return ctx[:max_total_chars]

def synthesize_answer(question: str, hits_df: pd.DataFrame,
                      max_chunk_chars: int, max_total_chars: int,
                      max_output_tokens: int) -> str:
    from vertexai.generative_models import Part
    ctx = build_context(hits_df, max_chunk_chars=max_chunk_chars, max_total_chars=max_total_chars)

    # More guided prompt for reservoir-style Qs
    reservoir_hint = ""
    if extract_wellname(question):
        reservoir_hint = (
            "- If the context includes reservoir info for that well, summarize formation(s), "
            "depths (MD/TVD where available), lithology, fluids, tests (DST), porosity/permeability, "
            "and any notable results. Keep it compact but cover key numbers.\n"
        )

    system = (
        "You are a petroleum data analyst. Answer using ONLY the provided context. "
        "If the answer isn't in the context, say you don't know."
    )
    prompt = f"""{system}

Question:
{question}

Context:
{ctx}

Instructions:
- Cite URLs inline by pasting the URL after the fact it supports.
{reservoir_hint}- Prefer bullet points, but avoid empty bullets.
"""

    model = _gemini_model()
    resp = model.generate_content(
        [Part.from_text(prompt)],
        generation_config=dict(temperature=0.1, max_output_tokens=max_output_tokens)
    )
    return _strip_empty_bullets(resp.text or "")

# ================= RESULTS TABLE =================
def render_hits(hits: pd.DataFrame, show_snippets: bool):
    tbl = hits.copy()
    pd.set_option("display.max_colwidth", None)
    tbl["URL"] = tbl["url"].apply(_link)

    def _title_row(r):
        title = _s(r.get("title")).strip()
        if title:
            return title
        t = _s(r.get("type")).strip()
        fn = _s(r.get("filename")).strip()
        if t or fn:
            return f"{t} — {fn}".strip(" —")
        return "(no title)"

    tbl["display_title"] = tbl.apply(_title_row, axis=1)
    base_cols = ["source", "URL", "display_title", "wellname", "distance"]

    if show_snippets:
        tbl["snippet"] = tbl["content"].apply(lambda x: _s(x)[:700])
        cols = base_cols + ["snippet"]
    else:
        cols = base_cols

    st.dataframe(tbl[cols], use_container_width=True, hide_index=True)

    with st.expander("Show full results"):
        for i, r in tbl.iterrows():
            st.markdown(f"**[{i+1}] {_s(r.get('source'))} — distance = {float(r['distance']):.6f}**")
            st.write("Well:", _s(r.get("wellname")))
            st.write("URL:", _s(r.get("url")))
            st.write("Title:", tbl.iloc[i]["display_title"])
            st.write(_s(r.get("content")))

# ================= CHAT =================
if "history" not in st.session_state:
    st.session_state.history = []

for msg in st.session_state.history:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

user_q = st.chat_input("Ask about SODIR/NOPIMS…")
if user_q:
    st.session_state.history.append({"role": "user", "content": user_q})
    with st.chat_message("user"):
        st.markdown(user_q)

    with st.chat_message("assistant"):
        try:
            with st.spinner("Embedding + searching…"):
                q_emb = _embed_model().get_embeddings([clip_text(user_q)])[0].values
                q_emb = [float(x) for x in q_emb]
                hits = run_vector_search(q_emb, user_q, top_k_total=top_k)

            if hits.empty:
                st.warning("No results returned. Ensure `docs`/`imgs` exist and have `emb` vectors.")
                st.session_state.history.append({"role": "assistant", "content": "No results found."})
            else:
                st.subheader("Top matches")
                render_hits(hits, show_snippets=show_snippets)

                if wants_imgs(user_q):
                    answer_text = render_img_answer(hits)
                    st.subheader("Answer")
                    st.markdown(answer_text)
                elif DO_SYNTH:
                    with st.spinner("Synthesizing concise answer…"):
                        answer_text = synthesize_answer(
                            user_q, hits,
                            max_chunk_chars=max_chunk_chars,
                            max_total_chars=max_total_chars,
                            max_output_tokens=answer_tokens,
                        )
                        st.subheader("Answer")
                        st.markdown(answer_text or "_(empty)_")
                else:
                    answer_text = ""

                # Save compact assistant summary
                citation_lines = []
                for i, r in hits.head(3).iterrows():
                    title = _s(r.get('title')).strip()
                    if not title:
                        title = f"{_s(r.get('type')).strip()} — {_s(r.get('filename')).strip()}".strip(" —")
                    title = title or "(no title)"
                    citation_lines.append(f"- {title} — {_s(r.get('url'))}")
                summary_md = ("**Top citations:**\n" + "\n".join(citation_lines))
                combined = (("**Answer**\n\n" + (answer_text or "") + "\n\n") if answer_text else "") + summary_md
                st.session_state.history.append({"role": "assistant", "content": combined})

        except Exception as e:
            st.error(f"Error: {e}")
