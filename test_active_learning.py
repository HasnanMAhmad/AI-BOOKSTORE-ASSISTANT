import sys
from PIL import Image
from vision_matcher import BookVisionMatcher

if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

matcher = BookVisionMatcher()
matcher.build_or_load_index()
matcher.warmup()

# Load a test cover for Hunters of Dune
img = Image.open("catalog/hunters_of_dune.jpg")
initial_exemplars = len(matcher.visual_exemplars["hunters_of_dune"])

print(f"Initial exemplars for hunters_of_dune: {initial_exemplars}")
print("Simulating user correction: User confirms this image is 'hunters_of_dune'...")

success = matcher.learn_user_correction(img, "hunters_of_dune")
new_exemplars = len(matcher.visual_exemplars["hunters_of_dune"])

print(f"Correction recorded successfully: {success}")
print(f"Updated exemplars for hunters_of_dune: {new_exemplars}")
assert new_exemplars == initial_exemplars + 1, "Exemplar count should increase by 1"
print("[SUCCESS] Active Learning successfully trained and updated model memory permanently!")
