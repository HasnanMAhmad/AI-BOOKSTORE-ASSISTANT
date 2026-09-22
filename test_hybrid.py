import sys
import numpy as np
from PIL import Image
from sentence_transformers import SentenceTransformer

if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

model = SentenceTransformer("clip-ViT-B-32")

books = ["dune", "dune_messiah", "children_of_dune", "hunters_of_dune", "sandworms_of_dune"]
titles = {
    "dune": "Dune by Frank Herbert",
    "dune_messiah": "Dune Messiah by Frank Herbert",
    "children_of_dune": "Children of Dune by Frank Herbert",
    "hunters_of_dune": "Hunters of Dune by Brian Herbert and Kevin J. Anderson",
    "sandworms_of_dune": "Sandworms of Dune by Brian Herbert and Kevin J. Anderson"
}

# Encode catalog visual covers
cover_embs = {b: model.encode([Image.open(f"catalog/{b}.jpg").convert("RGB")], normalize_embeddings=True)[0] for b in books}

# Encode text prototypes
text_prompts = {}
for b in books:
    clean_title = b.replace("_", " ").title()
    p = [
        f"a book cover of {titles[b]}",
        f"the cover of the novel {titles[b]}",
        f"book titled {clean_title}",
        clean_title
    ]
    text_prompts[b] = model.encode(p, normalize_embeddings=True)

# Test query: catalog/hunters_of_dune.jpg
q_img = Image.open("catalog/hunters_of_dune.jpg").convert("RGB")
q_emb = model.encode([q_img], normalize_embeddings=True)[0]

print("Query: catalog/hunters_of_dune.jpg")
for b in books:
    vis_score = float(np.dot(cover_embs[b], q_emb))
    text_score = float(np.max(np.dot(text_prompts[b], q_emb)))
    hybrid = 0.5 * vis_score + 0.5 * text_score
    print(f"  {b:18}: Visual={vis_score:.4f}, Text={text_score:.4f}, Hybrid={hybrid:.4f}")
