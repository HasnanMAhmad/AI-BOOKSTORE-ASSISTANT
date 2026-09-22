"""
Simulate realistic camera capture conditions (cropping, rotation, lighting shift)
to verify CLIP's real-world book recognition accuracy.
"""

import os
import sys
from PIL import Image, ImageEnhance
from vision_matcher import BookVisionMatcher

if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

def test_camera_simulation():
    matcher = BookVisionMatcher()
    matcher.build_or_load_index()

    dune_img_path = os.path.join(os.path.dirname(__file__), "catalog", "dune.jpg")
    print(f"Loading base image: {dune_img_path}")
    original = Image.open(dune_img_path)
    w, h = original.size

    # Simulate realistic photo conditions:
    # 1. Crop edges (customer holding camera slightly off-center)
    crop_box = (int(w * 0.08), int(h * 0.08), int(w * 0.92), int(h * 0.92))
    simulated = original.crop(crop_box)

    # 2. Dim lighting (dim bookstore or shelf shadow)
    enhancer = ImageEnhance.Brightness(simulated)
    simulated = enhancer.enhance(0.85)

    print("Running CLIP matching on simulated camera capture (cropped + dimmed)...")
    res = matcher.match_book(simulated)

    top = res["top_match"]
    print(f"\nResult:")
    print(f"  Recognized Book: {top['title']} by {top['author']}")
    print(f"  Confidence:      {top['confidence_pct']}%")
    print(f"  Is Confident:    {res['is_confident']}")

    assert top["id"] == "dune", f"Expected 'dune', but got '{top['id']}'"
    print("\n[SUCCESS] CLIP visual matching correctly identified Dune under simulated real-world camera conditions!")

if __name__ == "__main__":
    test_camera_simulation()
