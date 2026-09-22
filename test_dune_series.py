import sys
from vision_matcher import BookVisionMatcher

if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

m = BookVisionMatcher()
m.build_or_load_index()

dune_books = [
    "dune", "dune_messiah", "children_of_dune", "god_emperor_of_dune",
    "heretics_of_dune", "chapterhouse_dune", "hunters_of_dune",
    "sandworms_of_dune", "dune_house_atreides"
]

print(f"Testing {len(dune_books)} Dune Universe books recognition:\n")
all_correct = True
for b in dune_books:
    res = m.match_book(f"catalog/{b}.jpg")
    top = res["top_match"]
    title = top["title"]
    part = top.get("series_part", "N/A")
    conf = top["confidence_pct"]
    is_match = (top["id"] == b)
    if not is_match:
        all_correct = False
    status = "✅ MATCH" if is_match else "❌ MISMATCH"
    print(f"  {status} | Query: {b:20} -> {title:24} | {part[:30]:30} | {conf}%")

print(f"\nAll Dune series books correctly distinguished: {all_correct}")
