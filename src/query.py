import argparse
import pandas as pd
from google.cloud import bigquery
from utils_bq_vec import (
    get_bq_client, init_vertex, get_embed_model, clip_text,
    vector_search_sql
)

def ask(question: str, top_k: int = 8, synthesize: bool = False):
    bq = get_bq_client()
    init_vertex()
    model = get_embed_model()

    # Embed question
    q_emb = [float(x) for x in model.get_embeddings([clip_text(question)])[0].values]

    # Retrieve
    sql = vector_search_sql(top_k=top_k)
    hits = bq.query(
        sql,
        job_config=bigquery.QueryJobConfig(
            query_parameters=[bigquery.ArrayQueryParameter("q", "FLOAT64", q_emb)]
        )
    ).result().to_dataframe()
    pd.set_option("display.max_colwidth", None)
    hits_view = hits.assign(snippet=hits["content"].str.slice(0, 600))
    print("\n=== Top matches ===")
    print(hits_view[["url","title","distance","snippet"]])

    if not synthesize:
        return

    # Synthesize with Gemini 1.5 Pro
    from vertexai.generative_models import GenerativeModel, Part

    def build_context(hdf, max_chunk_chars=1200, max_total_chars=8000):
        pieces = []
        for _, r in hdf.iterrows():
            txt = (r["content"] or "")[:max_chunk_chars]
            pieces.append(f"URL: {r['url']}\nTITLE: {r['title']}\nTEXT: {txt}")
        ctx = "\n\n---\n\n".join(pieces)
        return ctx[:max_total_chars]

    ctx = build_context(hits)
    prompt = (
        "You are a petroleum data analyst. Answer concisely using ONLY the provided context. "
        "If the answer isn't in the context, say you don't know.\n\n"
        f"Question:\n{question}\n\nContext:\n{ctx}\n\n"
        "Instructions:\n- Cite URLs inline by pasting the URL after the fact it supports.\n"
        "- Keep the answer under 10 bullet points.\n"
    )
    gen = GenerativeModel("gemini-1.5-pro")
    resp = gen.generate_content([Part.from_text(prompt)],
                                generation_config=dict(temperature=0.1, max_output_tokens=512))
    print("\n=== Answer ===")
    print(resp.text)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--question", "-q", required=True, help="Your question")
    ap.add_argument("--top_k", type=int, default=8)
    ap.add_argument("--synthesize", action="store_true", help="Use Gemini to generate a concise answer")
    args = ap.parse_args()
    ask(args.question, args.top_k, args.synthesize)

if __name__ == "__main__":
    main()
