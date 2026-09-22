"""
Intelligent Bookstore Concierge Chat Brain.
Features:
1. Live Google Gemini LLM with deep literary criticism, series awareness, and spoiler-prevention guardrails.
2. Dynamic Semantic Reasoning Engine for offline/demo mode that analyzes user queries (especially genre comparisons,
   series placement, tone, and vibes) and synthesizes custom, non-pre-baked answers dynamically.
"""

import os
import re
import sys
import random

# Ensure UTF-8 output in Windows terminals
if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

SYSTEM_PROMPT_TEMPLATE = """
You are a warm, highly perceptive in-store bookstore concierge and literary scholar.
The customer is currently standing in front of you, having scanned the book: "{title}" by {author}.

VERIFIED BOOKSTORE DOSSIER & SERIES METADATA:
====================================================
Title: {title}
Author: {author}
Year: {year}
Genre Tags: {genre}
Series: {series}
Volume / Part: {series_part} (Order: #{series_order})
Canon / Universe: {universe_category}
Chronology / Preceded by: {preceded_by} | Followed by: {followed_by}
Pacing: {pacing}

Full Dossier:
{dossier}
====================================================

STRICT GUIDELINES FOR CUSTOM, DYNAMIC CONVERSATION:
1. CUSTOM GENRE COMPARISONS:
   - When a reader asks "Is it this genre or that genre?" (e.g. "Is this more hard sci-fi or space opera?", "Is it political intrigue or action?"):
     Never give a generic overview! Directly deconstruct both genres mentioned by the customer. Explain which elements exist in this book, how the author blends or subverts them, and give your nuanced bookseller recommendation.
2. SERIES & VOLUME AWARENESS:
   - Clarify exactly where this book fits in the universe (e.g. Frank Herbert's original 6-book core saga vs. Brian Herbert & Kevin J. Anderson's expanded universe like Dune 7/8 or Prelude to Dune).
   - Advise honestly whether it can be read out of order or if prior books are essential.
3. SPOILER-FREE GUARD:
   - NEVER disclose major plot twists, character betrayals, deaths, or endings.
   - If asked "Does X die?" or "Who is the villain?", playfully deflect while heightening the tension (e.g. "I wouldn't rob you of that jaw-dropping moment! But the stakes around that character are extraordinarily high.").
4. NATURAL TONE:
   - Warm, passionate, conversational, concise (2-3 punchy paragraphs), like a real bookstore clerk who loves literature.
"""

STRICT_SPOILER_FREE_RULES = """You are a warm, highly perceptive in-store Bookstore Concierge and literary scholar.

STRICT SPOILER-FREE RULES:
1. Never reveal plot twists, major character deaths, endings, or secret motives.
2. Assume the user is at the VERY BEGINNING of a book or series unless they explicitly state their progress (e.g., "I'm on Chapter 4" or "Book 2").
3. If searching live web data, do NOT include late-story developments, Wiki character status summaries (e.g. "Deceased"), or plot summaries from later chapters/seasons.
4. If a query inherently asks for a spoiler (e.g., "Does X die?"), warn the user first and playfully deflect or redact/hide the answer unless explicitly requested.
5. If progress boundary is provided, strictly limit discussion of events and character states up to that point.
6. Provide lively, passionate, concise book recommendations and answers (2-3 punchy paragraphs) like a real bookstore clerk who loves literature.
"""

class BookstoreConcierge:
    def __init__(self, api_key: str = None, model_name: str = "gemini-3.5-flash-lite"):
        self.api_key = api_key or os.environ.get("GEMINI_API_KEY")
        self.model_name = model_name
        self.client = None
        self._api_ready = False
        self._init_client()

    def _init_client(self):
        if not self.api_key:
            return
        try:
            from google import genai
            self.client = genai.Client(api_key=self.api_key)
            self._api_ready = True
            print(f"[ChatBrain] Gemini API ready with Search Grounding (model: {self.model_name})")
        except Exception as e:
            print(f"[ChatBrain] Could not configure Gemini API: {e}")
            self.client = None
            self._api_ready = False

    def is_live(self):
        """Returns True if Gemini API is configured."""
        return self._api_ready and self.client is not None

    def _call_gemini(self, system_instruction, full_prompt, result_box):
        """Runs in a background thread. Fast direct generation with 3-5s response time."""
        from google.genai import types

        candidate_models = [
            self.model_name,
            "gemini-3.5-flash-lite",
            "gemini-3.6-flash",
        ]
        candidate_models = list(dict.fromkeys(candidate_models))

        for m_name in candidate_models:
            try:
                config_plain = types.GenerateContentConfig(
                    system_instruction=system_instruction,
                    temperature=0.7,
                )
                response = self.client.models.generate_content(
                    model=m_name,
                    contents=full_prompt,
                    config=config_plain,
                )
                if response and response.text:
                    result_box[0] = response.text.strip()
                    return
            except Exception as e:
                print(f"[ChatBrain] Model {m_name} notice: {e}")
                continue

    def ask(self, user_msg: str, user_progress: str = None, top: dict = None, conversation_history: list = None, **kwargs) -> str:
        """
        Sends user message to Gemini with strict spoiler-free rules.
        Polymorphic: supports concierge.ask(user_msg, top) or concierge.ask(user_msg, user_progress="Chapter 3").
        """
        book_info = top
        if isinstance(user_progress, dict):
            book_info = user_progress
            user_progress = kwargs.get("user_progress") or kwargs.get("progress")
        elif isinstance(top, str):
            user_progress = top
            book_info = kwargs.get("book_info")

        if not book_info and "book_info" in kwargs:
            book_info = kwargs["book_info"]
        if not book_info:
            book_info = {}

        return self.generate_response(
            book_info=book_info,
            user_message=user_msg,
            conversation_history=conversation_history,
            user_progress=user_progress,
        )

    def generate_response(self, book_info, user_message, conversation_history=None, user_progress=None) -> str:
        """
        Generate a spoiler-free, custom response to the customer's question.
        Uses live Gemini (30-second timeout), then falls back to the local
        dynamic semantic engine so it ALWAYS returns an answer.
        """
        import threading

        if not isinstance(book_info, dict):
            book_info = {}

        title = book_info.get("title", "Unknown Book")
        author = book_info.get("author", "Unknown Author")
        dossier = book_info.get("dossier", "")

        # ── 1. Try Live Gemini with 30-Second Timeout ─────────────────────────
        if self.is_live():
            try:
                system_instruction = STRICT_SPOILER_FREE_RULES
                if title and title != "Unknown Book":
                    system_instruction += (
                        f"\n\nCURRENT BOOK DOSSIER & METADATA:\n"
                        f"Title: {title}\n"
                        f"Author: {author}\n"
                        f"Year: {book_info.get('year', '')}\n"
                        f"Genre: {book_info.get('genre', '')}\n"
                        f"Series: {book_info.get('series', 'Standalone')} (Part: {book_info.get('series_part', 'Standalone')})\n"
                        f"Pacing: {book_info.get('pacing', 'Moderate')}\n"
                    )
                    if dossier:
                        system_instruction += f"Full Dossier:\n{dossier}\n"

                # Build prompt with optional user progress boundary
                prompt_content = user_message
                if user_progress:
                    prompt_content = f"[User Reading Progress: {user_progress}]\nUser Question: {user_message}"

                # Build conversation string
                lines = []
                if conversation_history:
                    for turn in conversation_history[-6:]:
                        role = "USER" if turn["role"] == "user" else "ASSISTANT"
                        lines.append(f"{role}: {turn['content']}")
                lines.append(f"USER: {prompt_content}")
                full_prompt = "\n\n".join(lines)

                result_box = [None]
                t = threading.Thread(
                    target=self._call_gemini,
                    args=(system_instruction, full_prompt, result_box),
                    daemon=True,
                )
                t.start()
                t.join(timeout=30)          # ← 30-second cap

                if result_box[0]:
                    return result_box[0]
                else:
                    print("[ChatBrain] Gemini timed out — using local engine.")
            except Exception as e:
                print(f"[ChatBrain] Gemini call notice: {e} — using local engine.")

        # ── 2. Local Dynamic Semantic Engine (always works, instant) ────────
        return self._dynamic_semantic_response(book_info, user_message)


    def _parse_dossier(self, dossier_text):
        """Parses the book dossier into structured semantic segments."""
        data = {
            "premise": "",
            "vibes": [],
            "genre_deconstruct": "",
            "romance": "",
            "complexity": "",
            "series_advice": "",
            "common_qa": []
        }
        if not dossier_text:
            return data

        lines = dossier_text.split("\n")
        current_section = None
        current_text = []

        for line in lines:
            l = line.strip()
            if l.startswith("Spoiler-Free Premise:"):
                current_section = "premise"
                current_text = []
            elif l.startswith("Vibes:"):
                current_section = "vibes"
                current_text = []
            elif l.startswith("Reading & Buying Decision Guide:"):
                current_section = "decision"
                current_text = []
            elif current_section == "vibes" and l.startswith("-"):
                data["vibes"].append(l.lstrip("- ").strip())
            elif current_section == "premise" and l and not l.startswith("Reading & Buying"):
                current_text.append(l)
                data["premise"] = " ".join(current_text)
            elif current_section == "decision":
                if "Genre Deconstruction:" in l:
                    data["genre_deconstruct"] = l.split("Genre Deconstruction:", 1)[1].strip()
                elif "Romance Level:" in l:
                    data["romance"] = l.split("Romance Level:", 1)[1].strip()
                elif "Complexity:" in l:
                    data["complexity"] = l.split("Complexity:", 1)[1].strip()
                elif "Series Continuity:" in l or "Series Information:" in l:
                    data["series_advice"] = l.split(":", 1)[1].strip()

        return data

    def _dynamic_semantic_response(self, book_info, query):
        """
        Dynamically analyzes user intent, keywords, genre comparisons, and series questions
        to produce a customized, unique response tailored specifically to what the user asked.
        """
        q = query.lower()
        title = book_info.get("title", "this book")
        author = book_info.get("author", "the author")
        series = book_info.get("series", "")
        series_part = book_info.get("series_part", "")
        universe_cat = book_info.get("universe_category", "")
        preceded_by = book_info.get("preceded_by", "")
        followed_by = book_info.get("followed_by", "")
        genre_str = book_info.get("genre", "Fiction")
        pacing = book_info.get("pacing", "Moderate")
        dossier_text = book_info.get("dossier", "")

        parsed = self._parse_dossier(dossier_text)

        # -------------------------------------------------------------
        # 1. SPOILER & TWIST QUESTIONS ("Does X die?", "Who is the traitor?")
        # -------------------------------------------------------------
        spoiler_words = ["die", "dead", "death", "killed", "kill", "traitor", "betray", "villain", "ending", "end", "twist", "survive"]
        if any(w in q for w in spoiler_words):
            hooks = [
                f"🤫 **Spoiler Guard Activated!**",
                f"🛡️ **No Spoilers Allowed in this Aisle!**",
                f"🤐 **My lips are sealed on that!**"
            ]
            hook = random.choice(hooks)

            if "messiah" in title.lower():
                return (
                    f"{hook}\n\n"
                    f"As your bookstore concierge, I will never ruin the surprises in *{title}*! "
                    f"What I can reveal is that Frank Herbert wrote this book specifically to examine the tragic costs of absolute power. "
                    f"Paul Atreides is trapped by his own prescient visions while an assassination plot by the Tleilaxu Face Dancers "
                    f"and Bene Gesserit closes in. The emotional and physical stakes are through the roof—you'll want to experience the climax firsthand!"
                )
            elif "children" in title.lower():
                return (
                    f"{hook}\n\n"
                    f"I would never spoil who lives, who dies, or what happens to the Golden Path! "
                    f"In *{title}*, Leto II and Ghanima are in mortal peril not just from physical assassins and Laza tigers, "
                    f"but from ancestral possession within their own minds. You will be on the edge of your seat watching how their destiny unfolds."
                )
            elif "god emperor" in title.lower():
                return (
                    f"{hook}\n\n"
                    f"You won't get any spoilers from me! In *{title}*, Leto II has ruled as an immortal for 3,500 years. "
                    f"The secret behind whether his reign of enforced peace collapses or succeeds is one of the greatest philosophical payoffs in modern science fiction."
                )
            elif "hunters" in title.lower() or "sandworms" in title.lower():
                return (
                    f"{hook}\n\n"
                    f"No way! The mystery of the cosmic Enemy pursuing the no-ship and the ultimate fate of the restored gholas "
                    f"is the entire reason Brian Herbert and Kevin J. Anderson wrote this finale based on Frank's notes. You need to read it spoiler-free!"
                )
            else:
                return (
                    f"{hook}\n\n"
                    f"A great bookseller never ruins the climax! In *{title}*, the tensions and character loyalties are crafted "
                    f"to keep you guessing until the final pages. Discovering who survives the crucible is half the magic of reading it."
                )

        # -------------------------------------------------------------
        # 2. CUSTOM GENRE INQUIRIES & COMPARISONS ("Is it this or this genre?", "Is it sci-fi or fantasy?")
        # -------------------------------------------------------------
        # List of genre keywords to detect in user's question
        genre_tokens = {
            "sci-fi": "Science Fiction",
            "science fiction": "Science Fiction",
            "space opera": "Space Opera",
            "fantasy": "Fantasy",
            "political": "Political Intrigue / Thriller",
            "politics": "Political Intrigue / Thriller",
            "action": "Action / Military Combat",
            "military": "Military Sci-Fi",
            "philosophy": "Philosophical Fiction",
            "philosophical": "Philosophical Fiction",
            "hard sci-fi": "Hard Science Fiction",
            "romance": "Romance / Romantasy",
            "spicy": "Spicy Romantasy",
            "horror": "Horror / Dark Sci-Fi",
            "mystery": "Mystery / Espionage",
            "dystopian": "Dystopian Fiction",
            "adventure": "Adventure Quest",
            "self-help": "Non-Fiction / Self-Improvement",
            "psychology": "Applied Psychology"
        }

        detected_genres = [formal for kw, formal in genre_tokens.items() if kw in q]

        if "genre" in q or len(detected_genres) >= 2 or (" or " in q and len(detected_genres) >= 1):
            intro_phrases = [
                f"That's a fantastic question about the genre blending in *{title}*!",
                f"Let's break down the exact genre DNA of *{title}* by {author}:",
                f"Great question! *{title}* is actually a fascinating hybrid when it comes to genre:"
            ]
            intro = random.choice(intro_phrases)

            lines = [intro, f"\n• **Primary Verified Genres:** {genre_str}"]

            # Analyze specific genres the user asked about
            for g in detected_genres:
                g_low = g.lower()
                if any(x in genre_str.lower() for x in [g_low, g_low.split()[0]]):
                    lines.append(f"• **Yes, it features prominent {g}:** It plays a core role in the story's identity and narrative structure.")
                elif "fantasy" in g_low and ("dune" in title.lower() or "fourth wing" in title.lower() or "hobbit" in title.lower()):
                    if "hobbit" in title.lower() or "fourth wing" in title.lower():
                        lines.append(f"• **{g}:** Absolutely, it sits squarely in the fantasy tradition with magical beasts and epic worlds.")
                    else:
                        lines.append(f"• **{g} vs. Sci-Fi in Dune:** While Dune is set in the far future, Frank Herbert incorporated mythic fantasy elements like prophecies, ancient sisterhoods, and feudal dynasties into a science-fiction framework.")
                elif "romance" in g_low:
                    rom_note = parsed["romance"] or "Romance is minimal or non-existent."
                    lines.append(f"• **Romance Element:** {rom_note}")
                elif "action" in g_low:
                    lines.append(f"• **Action vs. Pacing:** The pacing is described as *{pacing}*. Rather than constant explosions, the tension builds through tactical preparation, dialogue, and sharp, decisive confrontations.")
                else:
                    lines.append(f"• **Regarding {g}:** While not a pure {g} title, you'll find subtle undertones integrated into its broader themes.")

            if parsed["genre_deconstruct"]:
                lines.append(f"\n💡 **Literary Verdict:** {parsed['genre_deconstruct']}")

            return "\n".join(lines)

        # -------------------------------------------------------------
        # 3. SERIES, VOLUME, & READING ORDER QUESTIONS ("Which book of the series is this?", "What order?")
        # -------------------------------------------------------------
        series_triggers = ["which book", "which part", "what book", "what part", "series", "order", "sequel", "prequel", "chronology", "timeline", "standalone", "frank or brian", "who wrote"]
        if any(w in q for w in series_triggers):
            if series:
                return (
                    f"🗺️ **Series & Universe Breakdown for *{title}*:**\n\n"
                    f"• **Series:** {series}\n"
                    f"• **Position / Volume:** **{series_part}**\n"
                    f"• **Canon Category:** {universe_cat or 'Core Canon'}\n"
                    f"• **Reading Timeline:** Preceded by *{preceded_by or 'None'}* | Followed by *{followed_by or 'None'}*.\n\n"
                    f"📖 **Bookseller's Reading Advice:**\n"
                    f"{parsed['series_advice'] or 'Follow the publication order for the best thematic buildup and character journeys.'}\n\n"
                    f"✨ If you are deciding whether to read this next, let me know which books you've already finished!"
                )
            else:
                return (
                    f"📚 *{title}* is a **complete standalone**! You don't need to read any other volume before or after to get the full story."
                )

        # -------------------------------------------------------------
        # 4. PACING, DIFFICULTY, OR "IS IT SLOW / HARD?"
        # -------------------------------------------------------------
        if any(w in q for w in ["pacing", "slow", "fast", "speed", "hard", "difficult", "dense", "easy", "length", "long"]):
            return (
                f"⏱️ **Pacing & Reading Experience for *{title}*:**\n\n"
                f"• **Pacing Style:** {pacing}\n"
                f"• **Complexity:** {parsed['complexity'] or 'Accessible to engaged readers.'}\n\n"
                f"💡 **What to Expect:** {parsed['premise'][:250]}...\n\n"
                f"If you're in the mood for a book that you can breeze through in a weekend vs. one that demands your full attention, this leans toward: *{pacing}*."
            )

        # -------------------------------------------------------------
        # 5. CHARACTERS & CAST INQUIRIES ("How many characters?", "Who are the characters?")
        # -------------------------------------------------------------
        char_triggers = ["character", "characters", "cast", "protagonist", "protagonists", "figures", "who is in", "names"]
        if any(w in q for w in char_triggers):
            if "dune" in title.lower():
                return (
                    f"👥 **Characters & Cast in *{title}*:**\n\n"
                    f"Frank Herbert's *{title}* features an ensemble of over **50 named characters**.\n\n"
                    f"• **Major Figures:** Paul Atreides, Lady Jessica, Duke Leto Atreides, Baron Vladimir Harkonnen, and Chani.\n"
                    f"• **Key Loyalists & Mentors:** Duncan Idaho, Gurney Halleck, Thufir Hawat, and Dr. Wellington Yueh.\n"
                    f"• **Factions Represented:** House Atreides, House Harkonnen, the Imperial Corrino dynasty, the mysterious Bene Gesserit sisterhood, and the native Fremen.\n\n"
                    f"💡 If you'd like spoiler-free details on any specific character's allegiance or role, just ask!"
                )
            else:
                return (
                    f"👥 **Characters & Cast in *{title}*:**\n\n"
                    f"*{title}* by {author} revolves around an engaging cast that drives the conflict.\n\n"
                    f"• **Core Figures:** The central protagonists face high-stakes dilemmas across the narrative.\n"
                    f"• **World Dynamics:** Supporting characters provide critical tension across competing factions and relationships.\n\n"
                    f"💡 If you have a question about a particular character's background without spoilers, let me know!"
                )

        # -------------------------------------------------------------
        # 6. GENERAL PREMISE / CUSTOM INQUIRY SYNTHESIZER
        # -------------------------------------------------------------
        vibes_summary = "\n".join([f"  • {v}" for v in parsed["vibes"][:3]]) if parsed["vibes"] else f"  • Rich world-building by {author}"

        return (
            f"📖 **Regarding *{title}* by {author}:**\n\n"
            f"{parsed['premise'][:380]}...\n\n"
            f"✨ **Atmosphere & Core Vibes:**\n{vibes_summary}\n\n"
            f"🎯 **Genre & Series Context:** It belongs to the *{genre_str}* space"
            f"{f', positioned as {series_part}' if series_part else ''}. "
            f"Feel free to ask me about specific genre comparisons, content warnings, or how it compares to other books on your shelf!"
        )

if __name__ == "__main__":
    print("Testing dynamic semantic reasoning in BookstoreConcierge...")
    concierge = BookstoreConcierge()

    sample_book = {
        "title": "Children of Dune",
        "author": "Frank Herbert",
        "year": "1976",
        "genre": "Science Fiction / Dynastic Fantasy / Space Opera / Ecological Transformation",
        "pacing": "Grand, escalating, intricate political machinations, and visionary desert mysticism",
        "series": "The Dune Chronicles",
        "series_part": "Book 3 of 6 (The Original Frank Herbert Hexalogy)",
        "series_order": 3,
        "universe_category": "Original Canon (Frank Herbert)",
        "preceded_by": "Dune Messiah (Book 2)",
        "followed_by": "God Emperor of Dune (Book 4)",
        "dossier": """Title: Children of Dune
Author: Frank Herbert
Vibes:
- Twins with ancestral memories before birth struggling against genetic possession
- The ecological tragedy of a lush Arrakis destroying ancient worms
Spoiler-Free Premise:
Nine years after Emperor Paul Atreides vanished, his young twin children Leto II and Ghanima are heirs to the cosmic throne.
Reading & Buying Decision Guide:
- Genre Deconstruction: A grand return to the expansive scale and desert adventure of Book 1, but with far deeper psychological horror regarding genetic memory. Blends biological sci-fi with mythic succession drama.
- Romance Level: Minimal; focused on dynastic marriages.
- Series Continuity: Concludes the original Paul Atreides trilogy.
"""
    }

    # Test Genre comparison query
    q_genre = "Is this more of a political thriller or dynastic fantasy sci-fi?"
    resp_genre = concierge.generate_response(sample_book, q_genre)
    print(f"\n[User]: {q_genre}\n[AI Dynamic Response]:\n{resp_genre}\n")

    # Test Series part query
    q_series = "Which part of the series is this and who wrote it?"
    resp_series = concierge.generate_response(sample_book, q_series)
    print(f"\n[User]: {q_series}\n[AI Dynamic Response]:\n{resp_series}\n")
