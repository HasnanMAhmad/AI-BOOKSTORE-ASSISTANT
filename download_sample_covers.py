"""
Download or generate high-quality starter book covers for the bookstore catalog,
including the complete Dune saga (Frank Herbert original hexalogy + Brian Herbert & Kevin J. Anderson expanded universe).
"""

import os
import requests
from PIL import Image, ImageDraw, ImageFont

CATALOG_DIR = os.path.join(os.path.dirname(__file__), "catalog")
os.makedirs(CATALOG_DIR, exist_ok=True)

# Complete list of catalog books with Open Library ISBNs and stylized visual fallbacks
BOOKS = [
    # --- FRANK HERBERT ORIGINAL HEXALOGY (BOOKS 1 - 6) ---
    {
        "id": "dune",
        "title": "DUNE",
        "author": "FRANK HERBERT",
        "subtitle": "BOOK 1: THE MASTERPIECE OF SCIENCE FICTION",
        "url": "https://covers.openlibrary.org/b/isbn/9780441172719-L.jpg",
        "bg_color": (214, 126, 44),     # Desert spice orange
        "fg_color": (255, 245, 220),
        "accent": (148, 64, 18),
    },
    {
        "id": "dune_messiah",
        "title": "DUNE MESSIAH",
        "author": "FRANK HERBERT",
        "subtitle": "BOOK 2 OF THE DUNE CHRONICLES",
        "url": "https://covers.openlibrary.org/b/isbn/9780441172696-L.jpg",
        "bg_color": (160, 48, 48),      # Crimson / imperial blood
        "fg_color": (255, 240, 230),
        "accent": (95, 20, 20),
    },
    {
        "id": "children_of_dune",
        "title": "CHILDREN OF DUNE",
        "author": "FRANK HERBERT",
        "subtitle": "BOOK 3 OF THE DUNE CHRONICLES",
        "url": "https://covers.openlibrary.org/b/isbn/9780441104024-L.jpg",
        "bg_color": (180, 120, 40),     # Golden amber sands
        "fg_color": (255, 250, 235),
        "accent": (110, 70, 15),
    },
    {
        "id": "god_emperor_of_dune",
        "title": "GOD EMPEROR OF DUNE",
        "author": "FRANK HERBERT",
        "subtitle": "BOOK 4 OF THE DUNE CHRONICLES",
        "url": "https://covers.openlibrary.org/b/isbn/9780441294671-L.jpg",
        "bg_color": (34, 100, 70),      # Terraformed green oasis
        "fg_color": (245, 255, 240),
        "accent": (15, 60, 40),
    },
    {
        "id": "heretics_of_dune",
        "title": "HERETICS OF DUNE",
        "author": "FRANK HERBERT",
        "subtitle": "BOOK 5 OF THE DUNE CHRONICLES",
        "url": "https://covers.openlibrary.org/b/isbn/9780441328000-L.jpg",
        "bg_color": (90, 40, 110),      # Bene Gesserit royal purple
        "fg_color": (250, 235, 255),
        "accent": (50, 15, 70),
    },
    {
        "id": "chapterhouse_dune",
        "title": "CHAPTERHOUSE: DUNE",
        "author": "FRANK HERBERT",
        "subtitle": "BOOK 6: THE CLIMAX OF FRANK HERBERT'S VISION",
        "url": "https://covers.openlibrary.org/b/isbn/9780441102679-L.jpg",
        "bg_color": (40, 60, 100),      # Deep twilight indigo
        "fg_color": (235, 245, 255),
        "accent": (20, 30, 65),
    },

    # --- EXPANDED DUNE UNIVERSE (BRIAN HERBERT & KEVIN J. ANDERSON) ---
    {
        "id": "hunters_of_dune",
        "title": "HUNTERS OF DUNE",
        "author": "BRIAN HERBERT & KEVIN J. ANDERSON",
        "subtitle": "DUNE 7: SEQUEL TO CHAPTERHOUSE DUNE",
        "url": "https://covers.openlibrary.org/b/isbn/9780765351494-L.jpg",
        "bg_color": (25, 30, 50),       # Cosmic deep space obsidian
        "fg_color": (240, 220, 120),
        "accent": (200, 70, 30),
    },
    {
        "id": "sandworms_of_dune",
        "title": "SANDWORMS OF DUNE",
        "author": "BRIAN HERBERT & KEVIN J. ANDERSON",
        "subtitle": "DUNE 8: THE GRAND FINALE OF THE DUNE CHRONICLES",
        "url": "https://covers.openlibrary.org/b/isbn/9780765351500-L.jpg",
        "bg_color": (140, 70, 30),      # Fiery scorched earth
        "fg_color": (255, 235, 200),
        "accent": (80, 25, 10),
    },
    {
        "id": "dune_house_atreides",
        "title": "DUNE: HOUSE ATREIDES",
        "author": "BRIAN HERBERT & KEVIN J. ANDERSON",
        "subtitle": "PRELUDE TO DUNE: BOOK 1",
        "url": "https://covers.openlibrary.org/b/isbn/9780553580273-L.jpg",
        "bg_color": (30, 75, 45),       # Caladan sea & hawk green
        "fg_color": (255, 240, 210),
        "accent": (160, 40, 40),
    },

    # --- OTHER POPULAR CATALOG TITLES ---
    {
        "id": "atomic_habits",
        "title": "ATOMIC HABITS",
        "author": "JAMES CLEAR",
        "subtitle": "TINY CHANGES, REMARKABLE RESULTS",
        "url": "https://covers.openlibrary.org/b/isbn/9780735211292-L.jpg",
        "bg_color": (245, 243, 238),
        "fg_color": (20, 20, 20),
        "accent": (218, 56, 32),
    },
    {
        "id": "project_hail_mary",
        "title": "PROJECT HAIL MARY",
        "author": "ANDY WEIR",
        "subtitle": "BY THE AUTHOR OF THE MARTIAN",
        "url": "https://covers.openlibrary.org/b/isbn/9780593135204-L.jpg",
        "bg_color": (244, 185, 34),
        "fg_color": (15, 15, 20),
        "accent": (40, 40, 40),
    },
    {
        "id": "the_hobbit",
        "title": "THE HOBBIT",
        "author": "J.R.R. TOLKIEN",
        "subtitle": "THERE AND BACK AGAIN",
        "url": "https://covers.openlibrary.org/b/isbn/9780547928227-L.jpg",
        "bg_color": (26, 77, 46),
        "fg_color": (240, 235, 200),
        "accent": (180, 140, 60),
    },
    {
        "id": "fourth_wing",
        "title": "FOURTH WING",
        "author": "REBECCA YARROS",
        "subtitle": "FLY OR DIE",
        "url": "https://covers.openlibrary.org/b/isbn/9781649374042-L.jpg",
        "bg_color": (20, 20, 24),
        "fg_color": (240, 200, 90),
        "accent": (190, 40, 40),
    },
]

def generate_styled_cover(book, output_path):
    """Generates an aesthetic, distinct book cover image using Pillow."""
    width, height = 600, 900
    img = Image.new("RGB", (width, height), color=book["bg_color"])
    draw = ImageDraw.Draw(img)

    margin = 30
    draw.rectangle(
        [(margin, margin), (width - margin, height - margin)],
        outline=book["accent"],
        width=4
    )
    draw.rectangle(
        [(margin + 12, margin + 12), (width - margin - 12, height - margin - 12)],
        outline=book["accent"],
        width=2
    )

    center_y = height // 2
    emblem_w = 220
    draw.ellipse(
        [(width // 2 - emblem_w // 2, center_y - 80), (width // 2 + emblem_w // 2, center_y + 80)],
        outline=book["fg_color"],
        width=3,
        fill=book["accent"]
    )
    draw.ellipse(
        [(width // 2 - emblem_w // 2 + 25, center_y - 55), (width // 2 + emblem_w // 2 - 25, center_y + 55)],
        outline=book["fg_color"],
        width=2
    )

    try:
        font_title = ImageFont.truetype("arial.ttf", 40)
        font_sub = ImageFont.truetype("arial.ttf", 18)
        font_author = ImageFont.truetype("arial.ttf", 28)
    except Exception:
        font_title = font_sub = font_author = ImageFont.load_default()

    draw.text((width // 2, 160), book["title"], fill=book["fg_color"], font=font_title, anchor="mm")
    draw.text((width // 2, 230), book["subtitle"], fill=book["fg_color"], font=font_sub, anchor="mm")
    draw.text((width // 2, height - 160), book["author"], fill=book["fg_color"], font=font_author, anchor="mm")

    img.save(output_path, "JPEG", quality=95)
    print(f"  [Created Styled Cover] {output_path}")

def setup_covers():
    headers = {"User-Agent": "BookstoreAI/1.0 (Educational Project)"}
    print("Setting up catalog covers...")
    for book in BOOKS:
        dest_path = os.path.join(CATALOG_DIR, f"{book['id']}.jpg")
        downloaded = False
        try:
            resp = requests.get(book["url"], headers=headers, timeout=5)
            if resp.status_code == 200 and len(resp.content) > 5000:
                with open(dest_path, "wb") as f:
                    f.write(resp.content)
                with Image.open(dest_path) as img:
                    img.verify()
                print(f"  [Downloaded] {book['id']}.jpg ({len(resp.content)} bytes)")
                downloaded = True
        except Exception as e:
            print(f"  [Network fetch skipped for {book['id']}: {e}]")

        if not downloaded:
            generate_styled_cover(book, dest_path)

if __name__ == "__main__":
    setup_covers()
