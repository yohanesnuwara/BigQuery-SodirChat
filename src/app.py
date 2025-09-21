import os, sys
from typing import List
import pandas as pd
import streamlit as st

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
    get_bq_client, init_vertex, get_embed_model, clip_text, vector_search_sql,
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

    top_k = st.slider("Top-K retrieval", 1, 20, 8)
    show_snippets = st.checkbox("Show snippets for each hit", value=True)
    max_chunk_chars = st.number_input("Max chars per chunk", 200, 3000, 1200, step=100)
    max_total_chars = st.number_input("Max total context chars", 1000, 20000, 8000, step=500)
    if st.button("🧹 Clear chat"):
        st.session_state.pop("history", None)
        st.rerun()

DO_SYNTH = True  # synthesis always on

# ================= SERVICES =================
@st.cache_resource(show_spinner=False)
def _bq():
    return get_bq_client()

@st.cache_resource(show_spinner=False)
def _embed_model():
    init_vertex()
    return get_embed_model()  # text-embedding-005

def _gemini_model():
    from vertexai import init as vinit
    from vertexai.generative_models import GenerativeModel
    vinit(project=PROJECT, location=VERTEX_LOCATION)
    return GenerativeModel("gemini-2.5-flash")

# ================= HELPERS =================
def embed_question(q: str) -> List[float]:
    model = _embed_model()
    q = clip_text(q)
    resp = model.get_embeddings([q])[0]
    return [float(x) for x in resp.values]

def run_vector_search(q_emb: List[float], top_k: int) -> pd.DataFrame:
    sql = vector_search_sql(top_k=top_k)
    bq = _bq()
    job = bq.query(
        sql,
        job_config=__import__("google.cloud.bigquery").cloud.bigquery.QueryJobConfig(
            query_parameters=[
                __import__("google.cloud.bigquery").cloud.bigquery.ArrayQueryParameter("q", "FLOAT64", q_emb)
            ]
        ),
    )
    return job.result().to_dataframe(create_bqstorage_client=True)

def build_context(hits_df: pd.DataFrame, max_chunk_chars=1200, max_total_chars=8000) -> str:
    pieces = []
    for _, r in hits_df.iterrows():
        url = str(r.get("url") or "")
        title = str(r.get("title") or "")
        content = str(r.get("content") or "")[:max_chunk_chars]
        pieces.append(f"URL: {url}\nTITLE: {title}\nTEXT: {content}")
    ctx = "\n\n---\n\n".join(pieces)
    return ctx[:max_total_chars]

def synthesize_answer(question: str, hits_df: pd.DataFrame,
                      max_chunk_chars: int, max_total_chars: int) -> str:
    from vertexai.generative_models import Part
    ctx = build_context(hits_df, max_chunk_chars=max_chunk_chars, max_total_chars=max_total_chars)
    system = (
        "You are a petroleum data analyst. "
        "Answer concisely using ONLY the provided context. "
        "If the answer isn't in the context, say you don't know."
    )
    prompt = f"""{system}

Question:
{question}

Context:
{ctx}

Instructions:
- Cite URLs inline by pasting the URL after the fact it supports.
- Keep the answer under 10 bullet points.
"""
    model = _gemini_model()
    resp = model.generate_content([Part.from_text(prompt)],
                                  generation_config=dict(temperature=0.1, max_output_tokens=512))
    return resp.text

def render_hits(hits: pd.DataFrame, show_snippets: bool):
    tbl = hits.copy()
    pd.set_option("display.max_colwidth", None)
    tbl["URL"] = tbl["url"].apply(lambda u: f"[link]({u})" if isinstance(u, str) and u.startswith("http") else u)
    if show_snippets:
        tbl["snippet"] = tbl["content"].str.slice(0, 600)
        st.dataframe(tbl[["URL", "title", "distance", "snippet"]],
                     use_container_width=True, hide_index=True)
    else:
        st.dataframe(tbl[["URL", "title", "distance"]],
                     use_container_width=True, hide_index=True)

    with st.expander("Show full results"):
        for i, r in tbl.iterrows():
            st.markdown(f"**[{i+1}] distance = {r['distance']:.6f}**")
            st.write("URL:", r["url"])
            st.write("Title:", r["title"])
            st.write(r["content"])

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
                q_emb = embed_question(user_q)
                hits = run_vector_search(q_emb, top_k=top_k)

            if hits.empty:
                st.warning("No results returned. Ensure `oilqna.docs` exists and has `emb` vectors.")
                st.session_state.history.append({"role": "assistant", "content": "No results found."})
            else:
                st.subheader("Top matches")
                render_hits(hits, show_snippets=show_snippets)

                answer_text = ""
                if DO_SYNTH:
                    with st.spinner("Synthesizing concise answer…"):
                        answer_text = synthesize_answer(
                            user_q, hits,
                            max_chunk_chars=max_chunk_chars,
                            max_total_chars=max_total_chars,
                        )
                        st.subheader("Answer")
                        st.write(answer_text or "_(empty)_")

                # Save compact assistant summary
                citation_lines = []
                for i, r in hits.head(3).iterrows():
                    citation_lines.append(f"- {r['title']} — {r['url']}")
                summary_md = ("**Top citations:**\n" + "\n".join(citation_lines))
                combined = (("**Answer**\n\n" + answer_text + "\n\n") if answer_text else "") + summary_md
                st.session_state.history.append({"role": "assistant", "content": combined})

        except Exception as e:
            st.error(f"Error: {e}")
