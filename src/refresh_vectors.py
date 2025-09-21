import pandas as pd
from google.cloud import bigquery
from google.api_core.exceptions import NotFound
from utils_bq_vec import (
    get_bq_client, init_vertex, get_embed_model, embed_texts,
    PROJECT, DATASET
)

# optional: skip image embeddings by setting this to False
EMBED_IMAGES = True

# defensively ignore UI assets if they slipped in
_IMG_BLACKLIST = ("/Images/factpages-logo", "/Images/faktasider-logo", "/Images/file-pdf.png")

def _caption_for_img(row: pd.Series) -> str:
    t = (row.get("type") or "").lower()
    well = (row.get("wellname") or "").strip()
    fname = (row.get("filename") or "").strip()
    if t == "map":
        return f"Factmap for well {well}".strip()
    if well and fname:
        return f"Core photo {fname} for well {well}".strip()
    if well:
        return f"Core photo for well {well}".strip()
    return "Core photo"

def main():
    bq = get_bq_client()

    # ---------- DOCS ----------
    try:
        bq.get_table(f"{PROJECT}.{DATASET}.docs_raw")
    except NotFound:
        raise SystemExit(f"docs_raw not found at {PROJECT}.{DATASET}.docs_raw — load your NDJSON first.")

    print("Reading docs_raw …")
    df = bq.query(f"""
      SELECT id, url, kind, title, content, lang, source_ts, wellname
      FROM `{PROJECT}.{DATASET}.docs_raw`
      ORDER BY id
    """).result().to_dataframe(create_bqstorage_client=True)
    print("Rows (docs_raw):", len(df))

    print("Init Vertex + model …")
    init_vertex()
    model = get_embed_model()

    print("Embedding docs (token-safe batching) …")
    doc_embeddings = embed_texts(model, df["content"].astype(str).tolist())

    print("Writing docs with embeddings …")
    docs_out = df.copy()
    docs_out["emb"] = doc_embeddings  # ARRAY<FLOAT64>
    # keep same table name as before; schema will now include wellname
    job = bq.load_table_from_dataframe(
        docs_out, f"{PROJECT}.{DATASET}.docs",
        job_config=bigquery.LoadJobConfig(write_disposition="WRITE_TRUNCATE"),
    )
    job.result()
    t = bq.get_table(f"{PROJECT}.{DATASET}.docs")
    print("✅ Done. Rows in docs:", t.num_rows)

    # ---------- IMGS (optional) ----------
    if EMBED_IMAGES:
        try:
            bq.get_table(f"{PROJECT}.{DATASET}.imgs_raw")
        except NotFound:
            print(f"⚠️ imgs_raw not found at {PROJECT}.{DATASET}.imgs_raw — skip image embeddings.")
            return

        print("Reading imgs_raw …")
        imgs = bq.query(f"""
          SELECT url, origin, filename, wellname, type
          FROM `{PROJECT}.{DATASET}.imgs_raw`
          WHERE url IS NOT NULL
        """).result().to_dataframe(create_bqstorage_client=True)

        if imgs.empty:
            print("No rows in imgs_raw — skipping.")
            return

        # filter out UI assets defensively
        mask = ~imgs["url"].str.lower().str.contains("|".join(s.lower() for s in _IMG_BLACKLIST), na=False)
        imgs = imgs[mask].copy()

        # build tiny captions for retrieval
        imgs["text"] = imgs.apply(_caption_for_img, axis=1)
        imgs["id"] = "img::" + imgs["url"].astype(str)

        print("Embedding imgs …")
        img_embeddings = embed_texts(model, imgs["text"].astype(str).tolist())
        imgs["emb"] = img_embeddings  # ARRAY<FLOAT64>

        # write to a dedicated vectors table for images
        job2 = bq.load_table_from_dataframe(
            imgs[["id","text","emb","url","origin","filename","type","wellname"]],
            f"{PROJECT}.{DATASET}.imgs",
            job_config=bigquery.LoadJobConfig(write_disposition="WRITE_TRUNCATE"),
        )
        job2.result()
        t2 = bq.get_table(f"{PROJECT}.{DATASET}.imgs")
        print("✅ Done. Rows in imgs:", t2.num_rows)

if __name__ == "__main__":
    main()
