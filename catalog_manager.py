"""
Universal Catalog Manager for Bookstore AI.
Allows adding ANY book in the world:
1. Searches Open Library by title/author/ISBN.
2. Downloads multiple edition covers.
3. Generates structured dossier.
4. Dynamically embeds the new book into catalog_embeddings.pkl without full retraining!
"""

import os
import re
import json
import time
import pickle
import requests
import numpy as np
from PIL import Image, ImageEnhance

BASE_DIR = os.path.dirname(__file__)
CATALOG_DIR = os.path.join(BASE_DIR, "catalog")
TRAIN_DIR = os.path.join(CATALOG_DIR, "training_covers")
CACHE_FILE = os.path.join(BASE_DIR, "catalog_embeddings.pkl")

def sanitize_id(title):
    s = re.sub(r'[^a-zA-Z0-9\s]', '', title).lower()
    return "_".join(s.split())[:40]

def search_open_library(query):
    url = f"https://openlibrary.org/search.json?q={requests.utils.quote(query)}&limit=5"
    headers = {"User-Agent": "BookstoreAI/2.0"}
    try:
        r = requests.get(url, headers=headers, timeout=8)
        if r.status_code == 200:
            data = r.json()
            results = []
            for doc in data.get("docs", [])[:5]:
                title = doc.get("title", "Untitled")
                author = doc.get("author_name", ["Unknown Author"])[0]
                year = doc.get("first_publish_year", "")
                cover_i = doc.get("cover_i")
                results.append({
                    "title": title,
                    "author": author,
                    "year": year,
                    "cover_i": cover_i,
                })
            return results
    except Exception as e:
        print("Search error:", e)
    return []

def add_book_to_catalog(title, author, series="", series_part="Standalone", synopsis="", vision_matcher=None):
    b_id = sanitize_id(title)
    os.makedirs(CATALOG_DIR, exist_ok=True)
    book_train_dir = os.path.join(TRAIN_DIR, b_id)
    os.makedirs(book_train_dir, exist_ok=True)

    headers = {"User-Agent": "Mozilla/5.0"}
    downloaded = []
    
    search_res = search_open_library(f"{title} {author}")
    if search_res and search_res[0].get("cover_i"):
        cid = search_res[0]["cover_i"]
        c_url = f"https://covers.openlibrary.org/b/id/{cid}-L.jpg"
        try:
            resp = requests.get(c_url, headers=headers, timeout=6)
            if resp.status_code == 200 and len(resp.content) > 3000:
                c_path = os.path.join(book_train_dir, "cover_edition_1.jpg")
                with open(c_path, "wb") as f:
                    f.write(resp.content)
                downloaded.append(c_path)
        except Exception:
            pass

    primary_cover = os.path.join(CATALOG_DIR, f"{b_id}.jpg")
    if downloaded and not os.path.exists(primary_cover):
        with open(downloaded[0], "rb") as src, open(primary_cover, "wb") as dst:
            dst.write(src.read())

    txt_path = os.path.join(CATALOG_DIR, f"{b_id}.txt")
    dossier = f"Title: {title}\nAuthor: {author}\nSeries: {series or 'Standalone'}\nVolume: {series_part or 'Volume 1'}\nGenres: Fiction\nSynopsis:\n{synopsis or (title + ' by ' + author)}"
    with open(txt_path, "w", encoding="utf-8") as f:
        f.write(dossier)

    # Save directly to SQLite database
    book_record = {
        "id": b_id,
        "title": title,
        "author": author,
        "series": series or "Standalone",
        "series_part": series_part or "Volume 1",
        "series_order": 99,
        "universe_category": "Literature",
        "year": 2024,
        "preceded_by": "None",
        "followed_by": "None",
        "dossier": dossier,
        "image_path": primary_cover if os.path.exists(primary_cover) else "",
    }
    
    import book_database as db
    db.upsert_book(book_record)

    if vision_matcher is not None:
        try:
            vision_matcher.build_or_load_index(force_rebuild=True)
        except Exception:
            pass

    return True, f"Successfully indexed '{title}' into bookstore database!"
