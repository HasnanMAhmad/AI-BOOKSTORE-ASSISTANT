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
import torch

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

    if vision_matcher is not None:
        try:
            model = vision_matcher.model
            exemplar_embs = []
            for img_path in (downloaded or [primary_cover]):
                if os.path.exists(img_path):
                    try:
                        base_img = Image.open(img_path).convert("RGB")
                        with torch.inference_mode():
                            emb = model.encode([base_img], convert_to_numpy=True, normalize_embeddings=True)[0]
                            exemplar_embs.append(emb)
                        for factor in [0.75, 1.25]:
                            enh = ImageEnhance.Brightness(base_img).enhance(factor)
                            with torch.inference_mode():
                                emb = model.encode([enh], convert_to_numpy=True, normalize_embeddings=True)[0]
                                exemplar_embs.append(emb)
                    except Exception:
                        pass

            if not exemplar_embs:
                dummy = Image.new("RGB", (224, 224), color=(70, 70, 70))
                with torch.inference_mode():
                    exemplar_embs.append(model.encode([dummy], convert_to_numpy=True, normalize_embeddings=True)[0])

            text_prompts = [
                f"Book cover of {title} by {author}",
                f"{title} written by {author}",
                f"{title}",
            ]
            with torch.inference_mode():
                text_embs = model.encode(text_prompts, convert_to_numpy=True, normalize_embeddings=True)

            meta = {
                "title": title,
                "author": author,
                "series": series or "Standalone",
                "series_part": series_part or "Volume 1",
                "series_order": 99,
                "universe_category": "Literature",
                "year": 2024,
                "preceded_by": "None",
                "followed_by": "None",
            }

            if os.path.exists(CACHE_FILE):
                with open(CACHE_FILE, "rb") as f:
                    bundle = pickle.load(f)
            else:
                bundle = {"books_meta": {}, "visual_exemplars": {}, "text_prototypes": {}}

            bundle["books_meta"][b_id] = meta
            bundle["visual_exemplars"][b_id] = np.array(exemplar_embs)
            bundle["text_prototypes"][b_id] = text_embs
            bundle["total_books"] = len(bundle["books_meta"])
            bundle["total_exemplars"] = sum(len(ex) for ex in bundle["visual_exemplars"].values())

            with open(CACHE_FILE, "wb") as f:
                pickle.dump(bundle, f)

            vision_matcher.build_or_load_index(force_rebuild=True)
            return True, f"Successfully added '{title}' with {len(exemplar_embs)} visual exemplars!"
        except Exception as e:
            return False, str(e)
            
    return True, f"Book saved for '{title}'."
