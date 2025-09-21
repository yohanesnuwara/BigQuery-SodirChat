from google.api_core.exceptions import NotFound
from utils_bq_vec import get_bq_client, ensure_dataset, PROJECT, BQ_LOCATION, DATASET

def main():
    bq = get_bq_client()
    print(f"Auth OK → project={bq.project}  bq_location={BQ_LOCATION}")
    ensure_dataset(bq, DATASET)
    print(f"Dataset ensured: {PROJECT}.{DATASET}")

    # Optional: confirm docs_raw exists
    try:
        t = bq.get_table(f"{PROJECT}.{DATASET}.docs_raw")
        print(f"Found table: {t.full_table_id}  rows={t.num_rows}")
    except NotFound:
        print(f"⚠️  {PROJECT}.{DATASET}.docs_raw not found. Load your crawl outputs first.")

if __name__ == "__main__":
    main()
