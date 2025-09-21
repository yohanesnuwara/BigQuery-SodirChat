import os
from typing import Iterable, List
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
DATASET = os.getenv("DATASET", "oilqna")
EMBED_MODEL_NAME = os.getenv("EMBED_MODEL_NAME", "text-embedding-005")

# conservative embedding caps to avoid 20k tokens/request
TOKENS_PER_CHAR = 0.25
MAX_ITEM_TOKENS = int(os.getenv("MAX_ITEM_TOKENS", "800"))   # ~800 tokens/row
MAX_ITEM_CHARS = int(MAX_ITEM_TOKENS / TOKENS_PER_CHAR)      # ~3200 chars
BATCH_ITEMS = int(os.getenv("BATCH_ITEMS", "20"))             # 20*800=16k tokens worst-case

def get_bq_client() -> bigquery.Client:
    """Create a BigQuery client using ADC (user or service account)."""
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

def vector_search_sql(top_k: int = 8) -> str:
    # Your environment returns (query, base, distance) from VECTOR_SEARCH
    return f"""
WITH vs AS (
  SELECT *
  FROM VECTOR_SEARCH(
    (SELECT id, emb, url, title, content FROM `{PROJECT}.{DATASET}.docs`),
    'emb',
    (SELECT 'q1' AS qid, @q AS emb),
    top_k => {top_k},
    distance_type => 'COSINE'
  )
)
SELECT
  vs.base.id      AS id,
  vs.base.url     AS url,
  vs.base.title   AS title,
  vs.base.content AS content,
  vs.distance     AS distance
FROM vs
ORDER BY distance
"""
