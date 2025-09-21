import pandas as pd
from google.cloud import bigquery
from google.api_core.exceptions import NotFound
from utils_bq_vec import (
    get_bq_client, init_vertex, get_embed_model, embed_texts,
    PROJECT, DATASET
)

def main():
    bq = get_bq_client()
    try:
        bq.get_table(f"{PROJECT}.{DATASET}.docs_raw")
    except NotFound:
        raise SystemExit(f"docs_raw not found at {PROJECT}.{DATASET}.docs_raw — load your NDJSON first.")

    print("Reading docs_raw …")
    df = bq.query(f"""
      SELECT id, url, kind, title, content, lang, source_ts
      FROM `{PROJECT}.{DATASET}.docs_raw`
      ORDER BY id
    """).result().to_dataframe(create_bqstorage_client=True)
    print("Rows:", len(df))

    print("Init Vertex + model …")
    init_vertex()
    model = get_embed_model()

    print("Embedding (token-safe batching) …")
    embeddings = embed_texts(model, df["content"].astype(str).tolist())

    print("Writing oilqna.docs with emb …")
    out = df.copy()
    out["emb"] = embeddings  # list[float] -> ARRAY<FLOAT64>

    job = bq.load_table_from_dataframe(
        out, f"{PROJECT}.{DATASET}.docs",
        job_config=bigquery.LoadJobConfig(write_disposition="WRITE_TRUNCATE"),
    )
    job.result()
    t = bq.get_table(f"{PROJECT}.{DATASET}.docs")
    print("✅ Done. Rows in docs:", t.num_rows)

if __name__ == "__main__":
    main()
