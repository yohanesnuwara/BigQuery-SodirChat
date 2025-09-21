import os
from typing import Iterable, List, Optional
import google.auth
from google.cloud import bigquery
from google.api_core.exceptions import NotFound
from vertexai import init as vertex_init
from vertexai.preview.language_models import TextEmbeddingModel
from tqdm import tqdm

# ---- config from environment (.env) ----
PROJECT = os.getenv("GOOGLE_CLOUD_PROJECT", "gen-lang-client-0106917803")
BQ_LOCATION = os.getenv("BQ_LOCATION", "US")
VERTEX_LOCATION = os.getenv("VERTEX_LOCATION", "us-central1")
DATASET = os.getenv("DATASET", "oilqna2")
EMBED_MODEL_NAME = os.getenv("EMBED_MODEL_NAME", "text-embedding-005")

# conservative embedding caps to avoid 20k tokens/request
TOKENS_PER_CHAR = 0.25
MAX_ITEM_TOKENS = int(os.getenv("MAX_ITEM_TOKENS", "800"))   # ~800 tokens/row
MAX_ITEM_CHARS  = int(MAX_ITEM_TOKENS / TOKENS_PER_CHAR)     # ~3200 chars
BATCH_ITEMS     = int(os.getenv("BATCH_ITEMS", "20"))         # 20*800=16k tokens worst-case

def get_bq_client() -> bigquery.Client:
    scopes = ["https://www.googleapis.com/auth/cloud-platform"]
    creds, _ = google.auth.default(scopes=scopes)
    return bigquery.Client(project=PROJECT, credentials=creds, location=BQ_LOCATION)

def ensure_dataset(bq: bigquery.Client, dataset_id: str = DATASET):
    ds_ref = bigquery.Dataset(f"{PROJECT}.{dataset_id}")
    ds_ref.location = BQ_LOCATION
    try:
        bq.get_dataset(ds_ref)
    except NotFound:
        bq.create_dataset(ds_ref)

def init_vertex():
    vertex_init(project=PROJECT, location=VERTEX_LOCATION)

def get_embed_model() -> TextEmbeddingModel:
    return TextEmbeddingModel.from_pretrained(EMBED_MODEL_NAME)

def batched(seq: List[str], n: int) -> Iterable[List[str]]:
    for i in range(0, len(seq), n):
        yield seq[i:i+n]

def clip_text(s: str) -> str:
    s = "" if s is None else str(s)
    return s[:MAX_ITEM_CHARS] if len(s) > MAX_ITEM_CHARS else s

def embed_texts(model: TextEmbeddingModel, texts: List[str]) -> List[List[float]]:
    clipped = [clip_text(t) for t in texts]
    out: List[List[float]] = []
    for batch in tqdm(list(batched(clipped, BATCH_ITEMS)), desc="Embedding"):
        resp = model.get_embeddings(batch)
        for r in resp:
            out.append([float(x) for x in r.values])
    assert len(out) == len(texts), f"Embeddings {len(out)} != rows {len(texts)}"
    return out

# -----------------------------
# Vector search SQL helpers
# -----------------------------

def vector_search_sql_docs_only(top_k: int = 8) -> str:
    """Original docs-only search (kept as fallback)."""
    return f"""
WITH vs AS (
  SELECT *
  FROM VECTOR_SEARCH(
    (SELECT id, emb, url, title, content, wellname FROM `{PROJECT}.{DATASET}.docs`),
    'emb',
    (SELECT 'q1' AS qid, @q AS emb),
    top_k => {top_k},
    distance_type => 'COSINE'
  )
)
SELECT
  'docs'                  AS source,
  vs.base.id              AS id,
  vs.base.url             AS url,
  vs.base.title           AS title,
  vs.base.content         AS content,
  vs.base.wellname        AS wellname,
  CAST(vs.distance AS FLOAT64) AS distance,
  NULL                    AS type,
  NULL                    AS filename
FROM vs
ORDER BY distance
"""

def vector_search_sql_dual(top_k_docs: int = 8, top_k_imgs: int = 8, filter_by_well: bool = False,
                           limit_total: int | None = None) -> str:
    """
    Combined search: docs + imgs. Assumes tables:
      - `{PROJECT}.{DATASET}.docs`  with: id, emb, url, title, content, wellname
      - `{PROJECT}.{DATASET}.imgs`  with: id, emb, url, filename, type, text, wellname
    If filter_by_well=True, both sources are prefiltered by wellname = @wn.
    """
    lim = f"\nLIMIT {limit_total}" if limit_total else ""
    where_docs = "WHERE wellname = @wn" if filter_by_well else ""
    where_imgs = "WHERE wellname = @wn" if filter_by_well else ""
    return f"""
WITH docs_vs AS (
  SELECT *
  FROM VECTOR_SEARCH(
    (SELECT id, emb, url, title, content, wellname
       FROM `{PROJECT}.{DATASET}.docs` {where_docs}),
    'emb',
    (SELECT 'q1' AS qid, @q AS emb),
    top_k => {top_k_docs},
    distance_type => 'COSINE'
  )
),
imgs_vs AS (
  SELECT *
  FROM VECTOR_SEARCH(
    (SELECT id, emb, url, filename, type, text, wellname
       FROM `{PROJECT}.{DATASET}.imgs` {where_imgs}),
    'emb',
    (SELECT 'q1' AS qid, @q AS emb),
    top_k => {top_k_imgs},
    distance_type => 'COSINE'
  )
)
SELECT
  'docs'                  AS source,
  d.base.id               AS id,
  d.base.url              AS url,
  d.base.title            AS title,
  d.base.content          AS content,
  d.base.wellname         AS wellname,
  CAST(d.distance AS FLOAT64) AS distance,
  NULL                    AS type,
  NULL                    AS filename
FROM docs_vs d
UNION ALL
SELECT
  'imgs'                  AS source,
  i.base.id               AS id,
  i.base.url              AS url,
  NULL                    AS title,
  i.base.text             AS content,
  i.base.wellname         AS wellname,
  CAST(i.distance AS FLOAT64) AS distance,
  i.base.type             AS type,
  i.base.filename         AS filename
FROM imgs_vs i
ORDER BY distance{lim}
"""
