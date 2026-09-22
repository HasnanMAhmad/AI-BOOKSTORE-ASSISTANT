import sys
from vision_matcher import BookVisionMatcher
from chat_brain import BookstoreConcierge

if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

matcher = BookVisionMatcher()
matcher.build_or_load_index()

concierge = BookstoreConcierge()

test_cases = [
    {
        "image": "catalog/children_of_dune.jpg",
        "question": "Is this more of a political thriller or dynastic fantasy sci-fi?",
        "expected_part": "Book 3"
    },
    {
        "image": "catalog/dune_messiah.jpg",
        "question": "Is this an action space opera or a psychological tragedy?",
        "expected_part": "Book 2"
    },
    {
        "image": "catalog/hunters_of_dune.jpg",
        "question": "Which book of the series is this, who wrote it, and is it Frank or Brian?",
        "expected_part": "Book 7"
    },
]

print("==================================================================")
print("TESTING SERIES RECOGNITION & CUSTOM DYNAMIC CONVERSATION")
print("==================================================================\n")

for i, tc in enumerate(test_cases, 1):
    res = matcher.match_book(tc["image"])
    top = res["top_match"]
    print(f"--- Case {i}: Scanned {top['title']} ---")
    print(f"🎯 Recognized Title:  {top['title']} by {top['author']}")
    print(f"📚 Series:            {top.get('series', 'N/A')}")
    print(f"🔢 Volume / Part:     {top.get('series_part', 'N/A')}")
    print(f"🏛️ Canon:             {top.get('universe_category', 'N/A')}")
    print(f"⏮️ Previous:          {top.get('preceded_by', 'None')}")
    print(f"⏭️ Next:              {top.get('followed_by', 'None')}")
    print(f"⚡ Match Time:         {top.get('elapsed_sec')}s (Confidence: {top['confidence_pct']}%)")
    
    # Generate custom dynamic answer
    q = tc["question"]
    reply = concierge.generate_response(top, q)
    print(f"\n💬 Customer: \"{q}\"")
    print(f"🤖 Concierge Dynamic Response:\n{reply}\n")
    print("-" * 66 + "\n")
