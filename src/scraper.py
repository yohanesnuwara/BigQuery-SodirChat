#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
crawl_ingest.py
Crawl 50 SODIR Factpages:
  https://factpages.sodir.no/en/wellbore/PageView/Exploration/All/{2..51}

Outputs (combined across all pages):
  - out/docs_raw.ndjson   (text/table chunks ready for BigQuery)  + wellname
  - out/imgs.csv          (image-like URLs) columns: url,origin,filename,wellname,type
      * type = "map" for the Factmaps link
      * type = "corephoto" for everything else

Notes:
  - Collects <img>, <source srcset>, CSS background images, and "image-like" links
    (including handler routes like /image, /photo, /corephoto).
  - Tables are flattened to CSV text in "content".
  - Be kind to remote servers: rate-limit is included.
"""

import os
import re
import csv
import json
import time
import hashlib
from typing import List, Tuple, Dict, Iterable, Optional
from urllib.parse import urljoin, urlparse

import requests
import pandas as pd
from bs4 import BeautifulSoup

# -----------------------------
# Config
# -----------------------------
BASES: List[str] = [
    f"https://factpages.sodir.no/en/wellbore/PageView/Exploration/All/{i}"
    for i in range(2, 52)  # 2..51 inclusive -> 50 wells
]

OUT_DIR = "out"
TEXT_PATH = os.path.join(OUT_DIR, "docs_raw.ndjson")
IMG_CSV   = os.path.join(OUT_DIR, "imgs.csv")
DOWNLOAD_DIR = os.path.join(OUT_DIR, "downloads")

# Chunking for long page text
CHUNK_SIZE = 1200
CHUNK_OVERLAP = 150

# Politeness
REQUEST_TIMEOUT = 45
SLEEP_BETWEEN_REQUESTS = 1.0  # seconds

# Optional: download discovered image/PDF assets locally
DOWNLOAD_ASSETS = False  # set True if you want local copies


# -----------------------------
# Session w/ headers & retries
# -----------------------------
session = requests.Session()
session.headers.update({
    "User-Agent": "kaggle-bigquery-ai-hackathon/1.0 (+semantic-detective)"
})
adapter = requests.adapters.HTTPAdapter(
    max_retries=requests.packages.urllib3.util.retry.Retry(
        total=3, backoff_factor=0.5, status_forcelist=[429, 500, 502, 503, 504]
    )
)
session.mount("http://", adapter)
session.mount("https://", adapter)


# -----------------------------
# Helpers
# -----------------------------
IMG_HINTS = (".jpg", ".jpeg", ".png", ".gif", ".webp", ".tif", ".tiff")
# handler-style patterns we’ll accept as "image-like" even without extension
IMG_HANDLER_KEYS = ("/image", "showimage", "photo", "corephoto", "imageid", "imgid")

# image URLs we never want (UI assets)
IMG_BLACKLIST_SUBSTR = [
    "/Images/faktasider-logo",   # both ...-sentrert.svg and ...-logo.svg
    "/Images/file-pdf.png",
]

def clean_text(s: str) -> str:
    return re.sub(r"\s+", " ", s or "").strip()

def chunk_text(text: str, size: int = CHUNK_SIZE, overlap: int = CHUNK_OVERLAP) -> Iterable[str]:
    text = text or ""
    if not text:
        return []
    i, n = 0, len(text)
    step = max(1, size - overlap)
    while i < n:
        yield text[i:i+size]
        i += step

def extract_image_links(soup: BeautifulSoup, base_url: str) -> List[str]:
    imgs = set()

    # <img src> and lazy <img data-src>
    for img in soup.select("img[src], img[data-src]"):
        src = img.get("src") or img.get("data-src")
        if src:
            imgs.add(urljoin(base_url, src))

    # <source srcset="..."> (take first candidate)
    for srcset in soup.select("source[srcset]"):
        s = srcset.get("srcset", "")
        if s:
            first = s.split(",")[0].strip().split(" ")[0]
            if first:
                imgs.add(urljoin(base_url, first))

    # CSS background-image: url(...)
    for el in soup.select("[style*='background-image']"):
        m = re.search(r"url\(['\"]?(.*?)['\"]?\)", el.get("style", ""))
        if m:
            imgs.add(urljoin(base_url, m.group(1)))

    # <a href=...> ending with known image extension OR containing handler-like keywords
    for a in soup.select("a[href]"):
        href = a["href"]
        full = urljoin(base_url, href)
        low = full.lower()
        if low.endswith(IMG_HINTS):
            imgs.add(full)
        elif any(k in low for k in IMG_HANDLER_KEYS):
            imgs.add(full)

    # keep http(s) only, drop blacklisted UI assets
    out = [
        u for u in imgs
        if urlparse(u).scheme in ("http", "https")
        and not any(b.lower() in u.lower() for b in IMG_BLACKLIST_SUBSTR)
    ]
    # dedupe but stable
    out.sort()
    return out

def read_tables(html_text: str) -> List[pd.DataFrame]:
    try:
        if "<table" in html_text.lower():
            return pd.read_html(html_text)
    except ValueError:
        # no tables found
        pass
    except Exception:
        # unexpected parse issue
        pass
    return []

def hash_id(*parts: str) -> str:
    h = hashlib.md5()
    for p in parts:
        h.update((p or "").encode("utf-8"))
    return h.hexdigest()

def write_ndjson(path: str, rows: List[Dict]):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

def write_csv(path: str, rows: List[Dict], fieldnames: List[str]):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k, "") for k in fieldnames})

def safe_filename(url: str, default: str = "file"):
    name = urlparse(url).path.split("/")[-1] or default
    # strip query fragments and keep a simple basename
    name = name.split("?")[0].split("#")[0] or default
    return name

def download_asset(url: str, dst_dir: str):
    try:
        os.makedirs(dst_dir, exist_ok=True)
        fname = safe_filename(url, "asset")
        # If no extension at all, try to keep the route name
        if "." not in fname:
            # tack on .jpg as a hint; real type is unknown without HEAD
            fname = f"{fname}.jpg"
        out_path = os.path.join(dst_dir, fname)
        r = session.get(url, timeout=REQUEST_TIMEOUT, stream=True)
        r.raise_for_status()
        with open(out_path, "wb") as f:
            for chunk in r.iter_content(8192):
                if chunk:
                    f.write(chunk)
        return out_path
    except Exception as e:
        print(f"[download] Failed {url}: {e}")
        return None


# -----------------------------
# SODIR-specific: wellname + factmap URL
# -----------------------------
def extract_wellname_and_map(url: str) -> Tuple[Optional[str], Optional[str]]:
    """
    Extracts wellname from 'General information' table and the
    'Factmaps in new window' -> 'link to map' URL.
    Robust to EN/NO; falls back to header like '31/3-2' or '31/2-24 A'.
    """
    wellname = None
    map_url = None
    try:
        r = session.get(url, timeout=REQUEST_TIMEOUT)
        r.raise_for_status()
        soup = BeautifulSoup(r.text, "html.parser")

        # 1) wellname from table
        label_variants = {
            "wellbore name", "well name",
            "borehullsnavn", "brønnhullsnavn", "br\u00f8nnhullsnavn",
        }
        for cell in soup.select("table td, table th"):
            txt = clean_text(cell.get_text()).lower()
            if txt in label_variants:
                sib = cell.find_next("td")
                if sib:
                    candidate = clean_text(sib.get_text())
                    if candidate:
                        wellname = candidate
                        break

        # fallback: header pattern
        if not wellname:
            h = soup.find(["h1", "h2"])
            if h:
                htxt = clean_text(h.get_text())
                m = re.search(r"\b\d{1,2}/\d{1,2}-[0-9A-Z]+(?: [A-Z])?\b", htxt)
                if m:
                    wellname = m.group(0)

        # 2) factmap URL
        map_row_labels = {"factmaps in new window", "faktakart i nytt vindu"}
        for cell in soup.select("table td, table th"):
            txt = clean_text(cell.get_text()).lower()
            if txt in map_row_labels:
                a = cell.find_next("a", href=True)
                if a and a["href"]:
                    map_url = urljoin(url, a["href"])
                    break

        if not map_url:
            a = soup.find("a", string=re.compile(r"link to map|kart", re.I))
            if a and a.get("href"):
                map_url = urljoin(url, a["href"])

    except Exception as e:
        print(f"[sodir] extraction failed for {url}: {e}")

    return wellname, map_url


# -----------------------------
# Scrapers
# -----------------------------
def scrape_generic(url: str) -> Tuple[str, List[pd.DataFrame], List[str]]:
    """Generic: fetch, parse text + tables + images."""
    r = session.get(url, timeout=REQUEST_TIMEOUT)
    r.raise_for_status()
    soup = BeautifulSoup(r.text, "html.parser")
    text = clean_text(soup.get_text(" "))
    tables = read_tables(r.text)
    images = extract_image_links(soup, url)
    return text, tables, images

def scrape_factpages(url: str) -> Tuple[str, List[pd.DataFrame], List[str], Optional[str], Optional[str]]:
    text, tables, images = scrape_generic(url)
    wellname, map_url = extract_wellname_and_map(url)
    return text, tables, images, wellname, map_url


# -----------------------------
# Main
# -----------------------------
def main():
    all_text_rows: List[Dict] = []
    all_img_rows: List[Dict] = []

    for base in BASES:
        print(f"[fetch] {base}")
        try:
            text, tables, images, wellname, map_url = scrape_factpages(base)
        except Exception as e:
            print(f"[error] {base}: {e}")
            time.sleep(SLEEP_BETWEEN_REQUESTS)
            continue

        wellname = wellname or ""

        # chunk main page text
        for i, ch in enumerate(chunk_text(text, CHUNK_SIZE, CHUNK_OVERLAP)):
            all_text_rows.append({
                "id": hash_id(base, f"chunk{i}"),
                "url": base,
                "kind": "page",
                "title": urlparse(base).path.strip("/").split("/")[-1] or "index",
                "content": ch,
                "lang": "auto",
                "source_ts": int(time.time()),
                "wellname": wellname,
            })

        # tables -> CSV text in content
        for ti, df in enumerate(tables):
            csv_text = df.to_csv(index=False)
            all_text_rows.append({
                "id": hash_id(base, f"table{ti}"),
                "url": base,
                "kind": "table",
                "title": f"table_{ti}",
                "content": csv_text,
                "lang": "auto",
                "source_ts": int(time.time()),
                "wellname": wellname,
            })

        # images -> imgs.csv (with wellname + type)
        for u in images:
            all_img_rows.append({
                "url": u,
                "origin": base,
                "filename": safe_filename(u, "image"),
                "wellname": wellname,
                "type": "corephoto",
            })

        # add the map URL as a dedicated row
        if map_url:
            all_img_rows.append({
                "url": map_url,
                "origin": base,
                "filename": safe_filename(map_url, "map.png"),
                "wellname": wellname,
                "type": "map",
            })

        # politeness
        time.sleep(SLEEP_BETWEEN_REQUESTS)

    # Write combined outputs
    write_ndjson(TEXT_PATH, all_text_rows)
    write_csv(IMG_CSV, all_img_rows, fieldnames=["url", "origin", "filename", "wellname", "type"])

    print(f"[done] wrote {len(all_text_rows)} text/table rows -> {TEXT_PATH}")
    print(f"[done] wrote {len(all_img_rows)} image rows -> {IMG_CSV}")

    # Optional: download assets locally
    if DOWNLOAD_ASSETS and all_img_rows:
        print("[download] fetching assets locally…")
        ok, fail = 0, 0
        for row in all_img_rows:
            path = download_asset(row["url"], DOWNLOAD_DIR)
            if path:
                ok += 1
            else:
                fail += 1
            time.sleep(0.2)
        print(f"[download] completed: {ok} ok, {fail} failed. Saved under {DOWNLOAD_DIR}")


if __name__ == "__main__":
    os.makedirs(OUT_DIR, exist_ok=True)
    main()
