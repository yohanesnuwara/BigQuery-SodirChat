from google.cloud import storage
import os

PROJECT_ID = "gen-lang-client-0106917803"
BUCKET = "oilqna2"

client = storage.Client(project=PROJECT_ID)
bucket = client.bucket(BUCKET)

def upload(local_path, gcs_path):
    blob = bucket.blob(gcs_path)
    blob.upload_from_filename(local_path)
    print("Uploaded", local_path, "->", f"gs://{BUCKET}/{gcs_path}")

os.makedirs("out", exist_ok=True)
upload("/home/yohanuwa/code/BigQuery-SodirChat/out/docs_raw.ndjson", "stage/docs_raw.ndjson")
upload("/home/yohanuwa/code/BigQuery-SodirChat/out/imgs.csv",        "stage/imgs.csv")
