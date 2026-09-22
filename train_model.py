"""
Machine Learning Training & Multi-Exemplar Indexing Pipeline for Bookstore AI.
Trains the visual-semantic model by:
1. Downloading multiple editions/covers for each book.
2. Generating data-augmented visual exemplars (crops, brightness, contrast).
3. Precomputing multi-modal text prototype embeddings (zero-shot title & author matching).
4. Training a k-Nearest Exemplar (k-NN) + Zero-Shot Text Hybrid Classifier.
5. Saving the trained model index into catalog_embeddings.pkl.
"""

import os
import sys
import time
import pickle
import requests
import numpy as np
from PIL import Image, ImageEnhance

if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

BASE_DIR = os.path.dirname(__file__)
CATALOG_DIR = os.path.join(BASE_DIR, "catalog")
TRAIN_DIR = os.path.join(CATALOG_DIR, "training_covers")
CACHE_FILE = os.path.join(BASE_DIR, "catalog_embeddings.pkl")

# Multi-edition ISBN dataset for Dune universe and catalog books
EDITIONS_DATASET = {
    "hunters_of_dune": [
        ("9780765351494", "Tor Mass Market Paperback (Stephen Youll)"),
        ("9780765312938", "Tor Hardcover 2006"),
        ("9780340837498", "Hodder & Stoughton UK Edition (Chris Moore)"),
    ],
    "children_of_dune": [
        ("9780441104024", "Berkley / Ace Standard Edition"),
        ("9780593098240", "Ace Deluxe Trade Edition"),
        ("9780340960189", "Gollancz / Hodder UK Edition"),
    ],
    "dune": [
        ("9780441172719", "Ace Classic 1965 Mass Market"),
        ("9780593099322", "Ace Deluxe Hardcover Edition"),
        ("9780340960196", "Hodder UK Modern Edition"),
    ],
    "dune_messiah": [
        ("9780441172696", "Ace Classic Paperback"),
        ("9780593098233", "Ace Deluxe Edition"),
    ],
    "god_emperor_of_dune": [
        ("9780441294671", "Ace Classic Edition"),
        ("9780593098257", "Ace Deluxe Edition"),
    ],
    "heretics_of_dune": [
        ("9780441328000", "Ace Classic Edition"),
        ("9780593098264", "Ace Deluxe Edition"),
    ],
    "chapterhouse_dune": [
        ("9780441102679", "Ace Classic Edition"),
        ("9780593098271", "Ace Deluxe Edition"),
    ],
    "sandworms_of_dune": [
        ("9780765351500", "Tor Mass Market Paperback"),
        ("9780765312945", "Tor Hardcover 2007"),
    ],
    "dune_house_atreides": [
        ("9780553580273", "Bantam Spectra Paperback"),
        ("9780553110616", "Bantam Hardcover 1999"),
    ],
    "atomic_habits": [
        ("9780735211292", "Avery Hardcover 2018"),
        ("9781847941831", "Random House Business UK"),
    ],
    "project_hail_mary": [
        ("9780593135204", "Ballantine Books Hardcover"),
        ("9781529157468", "Del Rey UK Paperback"),
    ],
    "the_hobbit": [
        ("9780547928227", "Houghton Mifflin Harcourt 75th Anniversary"),
        ("9780261102217", "HarperCollins UK Edition"),
    ],
    "fourth_wing": [
        ("9781649374042", "Red Tower Books Hardcover"),
        ("9780349437002", "Piatkus UK Edition"),
    ],
}

def download_training_editions():
    """Downloads alternative editions for each book to build a multi-exemplar dataset."""
    os.makedirs(TRAIN_DIR, exist_ok=True)
    headers = {"User-Agent": "BookstoreAI-Trainer/2.0 (Educational ML Project)"}
    total_downloaded = 0

    print("--- STEP 1: Collecting Multi-Edition Training Data ---")
    for book_id, editions in EDITIONS_DATASET.items():
        book_folder = os.path.join(TRAIN_DIR, book_id)
        os.makedirs(book_folder, exist_ok=True)

        # Copy the master catalog image if it exists
        master_img = os.path.join(CATALOG_DIR, f"{book_id}.jpg")
        master_dest = os.path.join(book_folder, "master_catalog.jpg")
        if os.path.exists(master_img) and not os.path.exists(master_dest):
            with open(master_img, "rb") as f_in, open(master_dest, "wb") as f_out:
                f_out.write(f_in.read())

        for idx, (isbn, label) in enumerate(editions, 1):
            dest_file = os.path.join(book_folder, f"edition_{isbn}.jpg")
            if os.path.exists(dest_file):
                continue
            url = f"https://covers.openlibrary.org/b/isbn/{isbn}-L.jpg"
            try:
                r = requests.get(url, headers=headers, timeout=5)
                if r.status_code == 200 and len(r.content) > 5000:
                    with open(dest_file, "wb") as f:
                        f.write(r.content)
                    with Image.open(dest_file) as im:
                        im.verify()
                    print(f"  [+] Downloaded: {book_id} -> {label} ({len(r.content)} bytes)")
                    total_downloaded += 1
            except Exception as e:
                pass

    print(f"Downloaded {total_downloaded} new edition covers into training pool.\n")

def augment_image(pil_img):
    """Generates synthetic data augmentations (crops & lighting variations)."""
    w, h = pil_img.size
    variants = [pil_img]

    # Crop 1: Center 88%
    c1 = pil_img.crop((int(w * 0.06), int(h * 0.06), int(w * 0.94), int(h * 0.94)))
    variants.append(c1)

    # Brightness adjustment (dim lighting simulation)
    enhancer_dim = ImageEnhance.Brightness(pil_img)
    variants.append(enhancer_dim.enhance(0.85))

    # Brightness adjustment (bright lighting simulation)
    enhancer_bright = ImageEnhance.Brightness(pil_img)
    variants.append(enhancer_bright.enhance(1.15))

    return variants

def train_and_index():
    """Trains the k-NN multi-exemplar visual classifier and text alignment model."""
    download_training_editions()

    from sentence_transformers import SentenceTransformer
    print("--- STEP 2: Initializing Neural Feature Extractor (CLIP ViT-B-32) ---")
    model = SentenceTransformer("clip-ViT-B-32")

    print("\n--- STEP 3: Training Multi-Modal Exemplar Model ---")
    
    # Read catalog metadata
    books_meta = {}
    for fname in os.listdir(CATALOG_DIR):
        if fname.endswith(".txt"):
            b_id = os.path.splitext(fname)[0]
            info = {"id": b_id, "title": b_id.replace("_", " ").title(), "author": "Unknown", "series": "", "series_part": ""}
            with open(os.path.join(CATALOG_DIR, fname), "r", encoding="utf-8") as f:
                for line in f:
                    l = line.strip()
                    if l.startswith("Title:"): info["title"] = l.split("Title:", 1)[1].strip()
                    elif l.startswith("Author:"): info["author"] = l.split("Author:", 1)[1].strip()
                    elif l.startswith("Series:"): info["series"] = l.split("Series:", 1)[1].strip()
                    elif l.startswith("Series Part:"): info["series_part"] = l.split("Series Part:", 1)[1].strip()
                    elif l.startswith("Genre:"): info["genre"] = l.split("Genre:", 1)[1].strip()
                    elif l.startswith("Pacing:"): info["pacing"] = l.split("Pacing:", 1)[1].strip()
                    elif l.startswith("Universe Category:"): info["universe_category"] = l.split("Universe Category:", 1)[1].strip()
                    elif l.startswith("Preceded By:"): info["preceded_by"] = l.split("Preceded By:", 1)[1].strip()
                    elif l.startswith("Followed By:"): info["followed_by"] = l.split("Followed By:", 1)[1].strip()
            books_meta[b_id] = info

    # 1. Visual Exemplar Embeddings per book
    book_exemplar_embeddings = {}
    total_exemplars = 0

    for b_id in sorted(books_meta.keys()):
        exemplar_images = []
        folder = os.path.join(TRAIN_DIR, b_id)
        
        # Collect base images
        raw_images = []
        if os.path.exists(folder):
            for img_name in os.listdir(folder):
                if img_name.lower().endswith((".jpg", ".png", ".jpeg")):
                    try:
                        im = Image.open(os.path.join(folder, img_name)).convert("RGB")
                        raw_images.append(im)
                    except Exception:
                        pass
        
        # Fallback to catalog image
        if not raw_images:
            cat_img = os.path.join(CATALOG_DIR, f"{b_id}.jpg")
            if os.path.exists(cat_img):
                raw_images.append(Image.open(cat_img).convert("RGB"))

        # Add data augmentations
        for base_img in raw_images:
            exemplar_images.extend(augment_image(base_img))

        # Compute normalized visual embeddings for all exemplars of this book
        if exemplar_images:
            embs = model.encode(exemplar_images, convert_to_numpy=True, normalize_embeddings=True, show_progress_bar=False)
            book_exemplar_embeddings[b_id] = embs
            total_exemplars += len(exemplar_images)
            print(f"  • {books_meta[b_id]['title']:25} -> Trained on {len(exemplar_images)} visual exemplars")

    # 2. Text Prototype Embeddings per book
    print("\n--- STEP 4: Computing Zero-Shot Multi-Modal Text Prototypes ---")
    book_text_embeddings = {}
    for b_id, meta in books_meta.items():
        title = meta["title"]
        author = meta["author"]
        series_part = meta.get("series_part", "")
        
        prompts = [
            f"a book cover with the title '{title}'",
            f"the novel '{title}' by {author}",
            f"cover art for {title}",
            title,
        ]
        if series_part:
            prompts.append(f"{title} {series_part}")
            prompts.append(f"the book cover of {title} ({series_part})")

        t_embs = model.encode(prompts, convert_to_numpy=True, normalize_embeddings=True, show_progress_bar=False)
        book_text_embeddings[b_id] = t_embs
        print(f"  • {title:25} -> {len(prompts)} text alignment prototypes")

    # Save trained model bundle
    trained_bundle = {
        "timestamp": time.time(),
        "total_books": len(books_meta),
        "total_exemplars": total_exemplars,
        "books_meta": books_meta,
        "visual_exemplars": book_exemplar_embeddings,
        "text_prototypes": book_text_embeddings,
    }

    with open(CACHE_FILE, "wb") as f:
        pickle.dump(trained_bundle, f)

    print(f"\n==================================================================")
    print(f"TRAINING COMPLETE! Model successfully trained on {total_exemplars} exemplars across {len(books_meta)} books.")
    print(f"Trained model index saved to: {CACHE_FILE}")
    print(f"==================================================================")

if __name__ == "__main__":
    train_and_index()
