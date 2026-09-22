import time
import sys
from vision_matcher import BookVisionMatcher

if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

print("1. Initializing matcher...")
m = BookVisionMatcher()
m.build_or_load_index()

print("2. Running warmup (preloading weights into RAM)...")
t_w0 = time.time()
m.warmup()
t_w1 = time.time()
print(f"   Warmup complete in {t_w1 - t_w0:.2f}s")

print("\n3. Testing user upload matching speed:")
for book_id in ["dune", "atomic_habits", "fourth_wing", "project_hail_mary", "the_hobbit"]:
    path = f"catalog/{book_id}.jpg"
    t0 = time.time()
    res = m.match_book(path)
    t1 = time.time()
    top = res["top_match"]
    print(f"   Image: {book_id:18} -> Matched: {top['title']:18} | Conf: {top['confidence_pct']}% | Time: {t1-t0:.4f}s")
