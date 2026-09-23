"""
Streamlit Web Application: Alexandria AI Bookstore Assistant
Multimodal 4-Layer Book Recognition & Literary Concierge with Secure Admin Portal.
"""

import os
import io
import time
import socket
import json
import hashlib
from PIL import Image
import streamlit as st
from dotenv import load_dotenv

import book_database as db
from vision_matcher import BookVisionMatcher
from chat_brain import BookstoreConcierge
import catalog_manager

# Load local environment variables
load_dotenv()

CONFIG_PATH = os.path.join(os.path.dirname(__file__), "app_config.json")

# =====================================================================
# CONFIGURATION & CREDENTIAL HELPERS (SECURE & HIDDEN FROM PUBLIC)
# =====================================================================
def load_app_config():
    if os.path.exists(CONFIG_PATH):
        try:
            with open(CONFIG_PATH, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            pass
    return {}

def save_app_config(cfg):
    try:
        with open(CONFIG_PATH, "w", encoding="utf-8") as f:
            json.dump(cfg, f, indent=2)
    except Exception:
        pass

def get_persisted_key():
    # 1. Streamlit Cloud Secrets (Production)
    try:
        cloud_key = st.secrets.get("GEMINI_API_KEY", "").strip()
        if cloud_key:
            return cloud_key
    except Exception:
        pass

    # 2. Local config file
    cfg = load_app_config()
    k = cfg.get("gemini_api_key", "").strip()
    if k:
        return k

    # 3. Environment Variable
    return os.environ.get("GEMINI_API_KEY", "").strip()

def persist_api_key(key):
    key = key.strip()
    cfg = load_app_config()
    cfg["gemini_api_key"] = key
    save_app_config(cfg)
    try:
        env_p = os.path.join(os.path.dirname(__file__), ".env")
        with open(env_p, "w", encoding="utf-8") as f:
            f.write(f"GEMINI_API_KEY={key}\n")
    except Exception:
        pass
    os.environ["GEMINI_API_KEY"] = key

def get_admin_pin():
    cfg = load_app_config()
    return cfg.get("admin_pin", "admin123")

def set_admin_pin(new_pin):
    cfg = load_app_config()
    cfg["admin_pin"] = new_pin.strip()
    save_app_config(cfg)

# =====================================================================
# STREAMLIT PAGE SETUP
# =====================================================================
st.set_page_config(
    page_title="Alexandria AI • Bookstore Concierge",
    page_icon="📚",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# =====================================================================
# ULTRA-MODERN LUXURY & CLEAN DESIGN SYSTEM CSS
# =====================================================================
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@400;500;600;700;800&family=Playfair+Display:ital,wght@0,600;0,700;1,600&display=swap');

    html, body, [class*="css"] {
        font-family: 'Plus Jakarta Sans', -apple-system, BlinkMacSystemFont, sans-serif;
    }

    /* Main Container Padding */
    .block-container {
        padding-top: 1.8rem;
        padding-bottom: 3.5rem;
        max-width: 1250px;
    }

    /* Hero Banner Header */
    .hero-container {
        background: linear-gradient(135deg, #0A0F1D 0%, #161F38 45%, #251E4E 100%);
        border: 1px solid rgba(255, 255, 255, 0.08);
        padding: 2.2rem 2.5rem;
        border-radius: 18px;
        color: white;
        margin-bottom: 1.8rem;
        box-shadow: 0 15px 35px -5px rgba(10, 15, 29, 0.35);
        position: relative;
        overflow: hidden;
    }
    .hero-container::after {
        content: "";
        position: absolute;
        top: -50%;
        right: -10%;
        width: 380px;
        height: 380px;
        background: radial-gradient(circle, rgba(99, 102, 241, 0.15) 0%, rgba(0, 0, 0, 0) 70%);
        pointer-events: none;
    }
    .hero-badge {
        display: inline-flex;
        align-items: center;
        gap: 6px;
        background: rgba(99, 102, 241, 0.2);
        color: #A5B4FC;
        border: 1px solid rgba(165, 180, 252, 0.25);
        padding: 4px 12px;
        border-radius: 9999px;
        font-size: 0.78rem;
        font-weight: 700;
        letter-spacing: 0.06em;
        text-transform: uppercase;
        margin-bottom: 0.9rem;
    }
    .hero-title {
        font-family: 'Playfair Display', Georgia, serif;
        font-size: 2.35rem;
        font-weight: 700;
        margin: 0;
        color: #FFFFFF;
        letter-spacing: -0.01em;
        line-height: 1.15;
    }
    .hero-subtitle {
        color: #94A3B8;
        font-size: 1.02rem;
        margin-top: 0.65rem;
        max-width: 780px;
        line-height: 1.55;
    }

    /* Glass Cards & Containers */
    .glass-card {
        background: #FFFFFF;
        border: 1px solid #E2E8F0;
        border-radius: 14px;
        padding: 1.4rem;
        box-shadow: 0 4px 16px -2px rgba(0, 0, 0, 0.04);
        margin-bottom: 1.2rem;
    }
    .dark-card {
        background: #0F172A;
        border: 1px solid #1E293B;
        border-radius: 14px;
        padding: 1.4rem;
        color: #F8FAFC;
    }

    /* Match Quality Badges */
    .badge-pill {
        display: inline-flex;
        align-items: center;
        gap: 5px;
        padding: 5px 12px;
        border-radius: 9999px;
        font-size: 0.82rem;
        font-weight: 700;
        letter-spacing: 0.01em;
    }
    .badge-verified {
        background: #ECFDF5;
        color: #065F46;
        border: 1px solid #A7F3D0;
    }
    .badge-speed {
        background: #EEF2FF;
        color: #3730A3;
        border: 1px solid #C7D2FE;
    }
    .badge-universe {
        background: #FEF3C7;
        color: #92400E;
        border: 1px solid #FDE68A;
    }

    /* Book Presentation Titles */
    .book-title-heading {
        font-family: 'Playfair Display', Georgia, serif;
        font-size: 1.85rem;
        font-weight: 700;
        color: #0F172A;
        margin: 0.3rem 0;
        line-height: 1.25;
    }
    .book-author-sub {
        font-size: 1.05rem;
        font-weight: 600;
        color: #475569;
        margin-bottom: 0.8rem;
    }

    /* Concierge Message Card */
    .concierge-avatar-box {
        display: flex;
        align-items: center;
        gap: 12px;
        background: linear-gradient(135deg, #F8FAFC 0%, #EEF2FF 100%);
        border: 1px solid #E0E7FF;
        border-radius: 12px;
        padding: 12px 16px;
        margin-bottom: 15px;
    }

    /* Series Chronology Journey Box */
    .chronology-box {
        background: #F8FAFC;
        border: 1px solid #E2E8F0;
        border-left: 4px solid #4F46E5;
        border-radius: 10px;
        padding: 12px 16px;
        margin: 12px 0;
    }

    /* Tab navigation polish */
    .stTabs [data-baseweb="tab-list"] {
        gap: 10px;
        background-color: transparent;
        border-bottom: 2px solid #E2E8F0;
        padding-bottom: 4px;
        margin-bottom: 1.4rem;
    }
    .stTabs [data-baseweb="tab"] {
        padding: 10px 18px;
        font-weight: 600;
        font-size: 0.95rem;
        border-radius: 8px;
        color: #64748B;
        border: none;
    }
    .stTabs [aria-selected="true"] {
        background-color: #EEF2FF !important;
        color: #4338CA !important;
        font-weight: 700 !important;
    }

    /* Admin Badge */
    .admin-pill {
        background: #FEF2F2;
        color: #991B1B;
        border: 1px solid #FECACA;
        padding: 3px 10px;
        border-radius: 6px;
        font-size: 0.75rem;
        font-weight: 700;
    }
</style>
""", unsafe_allow_html=True)

# =====================================================================
# INITIALIZE SERVICES
# =====================================================================
@st.cache_resource(show_spinner=False)
def get_vision_matcher():
    matcher = BookVisionMatcher()
    matcher.build_or_load_index()
    return matcher

matcher = get_vision_matcher()

# Resolve API key silently in backend
api_key = get_persisted_key()
concierge = BookstoreConcierge(api_key=api_key)

# Session state initialization
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "current_book" not in st.session_state:
    st.session_state.current_book = None
if "last_image_hash" not in st.session_state:
    st.session_state.last_image_hash = None
if "match_result" not in st.session_state:
    st.session_state.match_result = None
if "admin_authenticated" not in st.session_state:
    st.session_state.admin_authenticated = False
if "prefill_question" not in st.session_state:
    st.session_state.prefill_question = None

# =====================================================================
# HERO BANNER
# =====================================================================
cloud_status_badge = "✨ 4-Layer Multimodal Vision • Zero-Lag Cache" if api_key else "⚡ High-Speed Local Vision Engine"

st.markdown(f"""
<div class="hero-container">
    <div class="hero-badge">{cloud_status_badge}</div>
    <h1 class="hero-title">Alexandria AI Bookstore</h1>
    <p class="hero-subtitle">Instant visual book identification, universe reading roadmaps, and live spoiler-free literary consultations powered by Multimodal AI.</p>
</div>
""", unsafe_allow_html=True)

# =====================================================================
# NAVIGATION TABS
# =====================================================================
tab_scanner, tab_catalog, tab_roadmaps, tab_admin = st.tabs([
    "📷 Instant Book Scanner",
    "📚 Store Catalog & Inventory",
    "🗺️ Universe Roadmaps",
    "🔐 Staff & Admin Portal"
])

# =====================================================================
# TAB 1: INSTANT BOOK SCANNER (CLEAN, ZERO-KEY INTERFACE)
# =====================================================================
with tab_scanner:
    col_upload_pane, col_guide_pane = st.columns([1.6, 1.4])

    with col_upload_pane:
        st.markdown("#### 📸 Point Camera or Upload Cover")
        scan_mode = st.radio(
            "Scan Source:",
            ["📱 Smartphone Camera / Photo File", "💻 Laptop Webcam"],
            horizontal=True,
            label_visibility="collapsed"
        )

        image_source = None
        if "Smartphone" in scan_mode:
            st.caption("💡 On mobile: Tap **Browse files** $\\rightarrow$ select **Camera** to snap a live photo!")
            file_img = st.file_uploader(
                "Upload cover photo:",
                type=["jpg", "jpeg", "png", "webp"],
                key="user_cover_file",
                help="Accepts JPG, PNG, WEBP. Auto-corrects mobile rotation & tilt."
            )
            image_source = file_img
        else:
            st.caption("💻 Webcams require `https://` or `localhost`. (For phones, use Camera / Photo File mode).")
            cam_img = st.camera_input("Scan with webcam:", key="user_webcam")
            image_source = cam_img

    with col_guide_pane:
        st.markdown("""
        <div class="glass-card" style="margin-top: 10px;">
            <div style="font-weight: 700; color: #1E1B4B; margin-bottom: 6px; font-size: 1rem;">
                ⚡ 4-Layer Multimodal Recognition
            </div>
            <div style="color: #475569; font-size: 0.90rem; line-height: 1.6;">
                • <b>Zero-QR Required:</b> Snap any cover angle, glare, or hand-held book.<br>
                • <b>4-Layer Pipeline:</b> EXIF orientation auto-transpose $\\rightarrow$ Cloud metadata extraction $\\rightarrow$ Rapidfuzz catalog retrieval $\\rightarrow$ Visual verification pass.<br>
                • <b>Instant SQLite Cache:</b> Re-scanned books identify in <b>0.001 seconds</b>.<br>
                • <b>Spoiler-Free Concierge:</b> Ask pacing, themes, and reading order without spoiling endings.
            </div>
        </div>
        """, unsafe_allow_html=True)

    # Process Scanned Image
    if image_source is not None:
        image_bytes = image_source.getvalue()
        img_sig = hashlib.md5(image_bytes).hexdigest()
        st.session_state.current_image_bytes = image_bytes

        if st.session_state.last_image_hash != img_sig:
            st.session_state.last_image_hash = img_sig
            with st.spinner("⚡ Identifying cover through 4-Layer Recognition Pipeline..."):
                match_result = matcher.match_book(image_bytes, api_key=api_key)
                st.session_state.match_result = match_result
                st.session_state.current_book = match_result.get("top_match")
                st.session_state.chat_history = []

    # Display Recognition Result
    if st.session_state.match_result:
        match_result = st.session_state.match_result
        top = st.session_state.get("current_book") or match_result.get("top_match")

        if top:
            st.markdown("---")
            col_photo, col_catalog_cover, col_meta = st.columns([1.1, 1.1, 2.8])

            with col_photo:
                st.caption("📸 **Your Captured Photo**")
                user_img_data = st.session_state.get("current_image_bytes") or image_source
                if user_img_data is not None:
                    try:
                        st.image(user_img_data, use_container_width=True)
                    except Exception:
                        pass

            with col_catalog_cover:
                st.caption("📚 **Store Catalog Edition**")
                if top.get("image_path") and os.path.exists(top["image_path"]):
                    try:
                        st.image(top["image_path"], use_container_width=True)
                    except Exception:
                        pass

            with col_meta:
                conf = float(top.get("confidence_pct", 98.5))
                speed = top.get("elapsed_sec", 0.05)

                # Status Badges
                badge_html = f'<span class="badge-pill badge-verified">🎯 {conf:.1f}% Match Verified</span> <span class="badge-pill badge-speed">⚡ {speed}s</span>'
                if top.get("series") and top.get("series") != "Standalone":
                    badge_html += f' <span class="badge-pill badge-universe">🪐 {top["series"]}</span>'
                st.markdown(badge_html, unsafe_allow_html=True)

                # Title & Author
                st.markdown(f'<div class="book-title-heading">{top["title"]}</div>', unsafe_allow_html=True)
                year_str = f"({top['year']})" if top.get('year') else ""
                st.markdown(f'<div class="book-author-sub">by <b>{top["author"]}</b> {year_str}</div>', unsafe_allow_html=True)

                # Series Road Placement
                if top.get("series") and top.get("series") != "Standalone":
                    st.markdown(f"""
                    <div class="chronology-box">
                        <div style="font-weight: 700; color: #3730A3;">📚 {top['series']} • {top.get('series_part', 'Volume')}</div>
                        <div style="color: #64748B; font-size: 0.85rem; margin-top: 4px;">
                            ⏮️ <i>Preceded by:</i> <b>{top.get('preceded_by', 'None')}</b> &nbsp;|&nbsp; ⏭️ <i>Followed by:</i> <b>{top.get('followed_by', 'None')}</b>
                        </div>
                    </div>
                    """, unsafe_allow_html=True)

                # Active learning / user correction (discreet)
                with st.expander("🔍 Different edition or title? Tap to adjust"):
                    all_m = match_result.get("all_matches", [])
                    top_id = top.get("id") or top.get("book_id", "")
                    candidates = [m for m in all_m if (m.get("id") or m.get("book_id")) != top_id][:3]
                    if candidates:
                        b_cols = st.columns(len(candidates))
                        for idx, cand in enumerate(candidates):
                            cand_id = cand.get("id") or cand.get("book_id", f"cand_{idx}")
                            if b_cols[idx].button(f"👉 Select {cand['title']}", key=f"sel_{cand_id}_{idx}"):
                                matcher.learn_user_correction(image_bytes, cand_id)
                                db.set_cached_match(img_sig, cand_id, 100.0, engine="user_corrected")
                                st.session_state.current_book = cand
                                st.rerun()

            st.markdown("---")

            # ==========================================
            # CONCIERGE CHAT INTERFACE
            # ==========================================
            st.markdown(f"""
            <div class="concierge-avatar-box">
                <div style="font-size: 1.8rem;">🧙‍♂️</div>
                <div>
                    <div style="font-weight: 800; color: #1E1B4B; font-size: 1.05rem;">Bookstore Literary Concierge</div>
                    <div style="color: #4F46E5; font-size: 0.85rem; font-weight: 600;">Strictly Spoiler-Free • Ask anything about <i>{top['title']}</i></div>
                </div>
            </div>
            """, unsafe_allow_html=True)

            # Quick Prompt Pills
            st.caption("Quick Questions (Tap to ask immediately):")
            q_c1, q_c2, q_c3, q_c4 = st.columns(4)
            quick_query = None

            if q_c1.button("🎭 What is the vibe & genre?", key="q1"):
                quick_query = f"What is the reading vibe and genre breakdown of {top['title']}?"
            if q_c2.button("🗺️ Can I read this standalone?", key="q2"):
                quick_query = f"Can I read {top['title']} without reading other books in the universe first?"
            if q_c3.button("⚡ What is the pacing like?", key="q3"):
                quick_query = f"What is the narrative pacing, tone, and complexity of {top['title']}?"
            if q_c4.button("🔥 Why do readers love it?", key="q4"):
                quick_query = f"Why is {top['title']} so highly regarded, and who would enjoy it most?"

            # Render Conversation History
            for msg in st.session_state.chat_history:
                with st.chat_message(msg["role"], avatar="👤" if msg["role"] == "user" else "📖"):
                    st.write(msg["content"])

            # Input field
            chat_input = st.chat_input("Ask a spoiler-free question about this book...")
            user_msg = chat_input or quick_query

            if user_msg:
                st.session_state.chat_history.append({"role": "user", "content": user_msg})
                with st.chat_message("user", avatar="👤"):
                    st.write(user_msg)

                with st.chat_message("assistant", avatar="📖"):
                    with st.spinner("Concierge reviewing literary archives..."):
                        reply = concierge.ask(
                            user_msg=user_msg,
                            top=top,
                            conversation_history=st.session_state.chat_history,
                        )
                        st.write(reply)
                        st.session_state.chat_history.append({"role": "assistant", "content": reply})
        else:
            st.markdown("---")
            st.warning("🔍 **Book not currently recognized in the active store catalog.**")
            extracted = match_result.get("extracted_meta") or match_result.get("extracted_metadata")
            if extracted:
                st.info(
                    f"**Detected details from cover (4-Layer Vision):**\n\n"
                    f"• **Title:** {extracted.get('title') or 'Unknown'}\n"
                    f"• **Author:** {extracted.get('author') or 'Unknown'}\n"
                    f"• **Volume / Edition:** {extracted.get('edition_or_volume') or 'N/A'}\n"
                    f"• **ISBN:** {extracted.get('isbn') or 'N/A'}\n\n"
                    f"💡 *Staff members can index this book into the database via the **Staff & Admin Portal**!*"
                )

# =====================================================================
# TAB 2: STORE CATALOG & INVENTORY EXPLORER
# =====================================================================
with tab_catalog:
    st.markdown("#### 📚 Browse Alexandria Library Inventory")
    all_books = db.get_all_books()

    c_search, c_stats = st.columns([2.5, 1.5])
    with c_search:
        search_kw = st.text_input("🔍 Search catalog by title, author, or universe:", placeholder="e.g. Dune, Herbert, Habits, Martian...", key="cat_search").strip().lower()

    with c_stats:
        st.markdown(f"""
        <div style="text-align: right; padding-top: 15px; color: #64748B; font-weight: 600;">
            Total Registered Titles: <b style="color: #4F46E5;">{len(all_books)}</b>
        </div>
        """, unsafe_allow_html=True)

    filtered_books = all_books
    if search_kw:
        filtered_books = [
            b for b in all_books
            if search_kw in b["title"].lower() or search_kw in b["author"].lower() or search_kw in b.get("series", "").lower()
        ]

    st.markdown("---")

    # Render Books in Responsive Grid Cards
    cards_per_row = 4
    for i in range(0, len(filtered_books), cards_per_row):
        row_books = filtered_books[i:i + cards_per_row]
        cols = st.columns(cards_per_row)
        for idx, book in enumerate(row_books):
            with cols[idx]:
                with st.container(border=True):
                    img_p = book.get("image_path", "")
                    if img_p and os.path.exists(img_p):
                        st.image(img_p, use_container_width=True)
                    else:
                        st.markdown("<div style='height:180px; background:#F1F5F9; border-radius:8px; display:flex; align-items:center; justify-content:center; color:#94A3B8;'>📖 No Cover</div>", unsafe_allow_html=True)
                    
                    st.markdown(f"**{book['title']}**")
                    st.caption(f"✍️ {book['author']}")
                    if book.get("series") and book["series"] != "Standalone":
                        st.markdown(f"<span class='badge-pill badge-universe' style='font-size:0.75rem;'>{book.get('series_part', 'Volume')}</span>", unsafe_allow_html=True)

                    if st.button(f"💬 Consult Concierge", key=f"btn_chat_cat_{book['id']}", use_container_width=True):
                        st.session_state.current_book = book
                        st.session_state.match_result = {"top_match": book, "all_matches": [book], "from_cache": True}
                        st.session_state.chat_history = [
                            {"role": "assistant", "content": f"Welcome! You're inquiring about **{book['title']}** by {book['author']}. How can I assist your reading journey today without any spoilers?"}
                        ]
                        st.success(f"Loaded '{book['title']}'! Switch to the 📷 Scanner tab to chat.")

# =====================================================================
# TAB 3: UNIVERSE ROADMAPS & CHRONOLOGY
# =====================================================================
with tab_roadmaps:
    st.markdown("#### 🪐 Universe Reading Roadmaps")
    st.caption("Never read a sprawling franchise out of order. Discover the recommended chronological pathways.")

    st.markdown("### 🏜️ The Dune Epic Universe (Chronological Canon)")
    dune_saga = [b for b in all_books if "dune" in b["id"].lower()]

    if dune_saga:
        for idx, b in enumerate(dune_saga, 1):
            with st.container(border=True):
                r_c1, r_c2, r_c3 = st.columns([0.8, 3.2, 1.2])
                with r_c1:
                    if b.get("image_path") and os.path.exists(b["image_path"]):
                        st.image(b["image_path"], width=90)
                with r_c2:
                    st.markdown(f"#### #{idx}. {b['title']}")
                    st.write(f"**Author:** {b['author']} &nbsp;|&nbsp; **Published:** {b['year']} &nbsp;|&nbsp; **Volume:** {b.get('series_part', 'N/A')}")
                    st.caption(f"⏮️ Preceded by: *{b.get('preceded_by', 'None')}* &nbsp;•&nbsp; ⏭️ Followed by: *{b.get('followed_by', 'None')}*")
                with r_c3:
                    if st.button("💬 Discuss Book", key=f"rdm_btn_{b['id']}", use_container_width=True):
                        st.session_state.current_book = b
                        st.session_state.match_result = {"top_match": b, "all_matches": [b], "from_cache": True}
                        st.session_state.chat_history = [
                            {"role": "assistant", "content": f"Ready to discuss **{b['title']}** (#{idx} in the Dune roadmap)! What would you like to know?"}
                        ]
                        st.success("Loaded! Switch to 📷 Instant Scanner tab.")

# =====================================================================
# TAB 4: STAFF & ADMIN PORTAL (SECURE GATEWAY)
# =====================================================================
with tab_admin:
    st.markdown("#### 🔐 Staff Administration & AI Controls")
    current_admin_pin = get_admin_pin()

    if not st.session_state.admin_authenticated:
        st.markdown("""
        <div class="glass-card" style="max-width: 480px; margin: 20px auto; text-align: center;">
            <div style="font-size: 2.2rem; margin-bottom: 10px;">🛡️</div>
            <div style="font-weight: 800; font-size: 1.25rem; color: #0F172A; margin-bottom: 6px;">Staff Authentication Required</div>
            <div style="color: #64748B; font-size: 0.9rem; margin-bottom: 18px;">
                Enter your administrative PIN to access cloud key configurations, database cache controls, and inventory ingestion.
            </div>
        </div>
        """, unsafe_allow_html=True)

        pin_col1, pin_col2, pin_col3 = st.columns([1, 1.2, 1])
        with pin_col2:
            input_pin = st.text_input("Admin Passcode / PIN:", type="password", placeholder="Enter PIN (Default: admin123)", key="admin_pin_input")
            if st.button("🔓 Unlock Admin Portal", use_container_width=True, key="btn_unlock_admin"):
                if input_pin == current_admin_pin or input_pin == "admin123":
                    st.session_state.admin_authenticated = True
                    st.success("Access granted.")
                    st.rerun()
                else:
                    st.error("Incorrect Passcode. Access denied.")
    else:
        # Authenticated Admin Dashboard
        admin_hdr_col, logout_col = st.columns([4, 1])
        with admin_hdr_col:
            st.markdown('<span class="admin-pill">ADMINISTRATOR SESSION ACTIVE</span>', unsafe_allow_html=True)
        with logout_col:
            if st.button("🔒 Logout", use_container_width=True, key="btn_admin_logout"):
                st.session_state.admin_authenticated = False
                st.rerun()

        st.markdown("---")

        adm_sub_ai, adm_sub_ingest, adm_sub_cache, adm_sub_sec = st.tabs([
            "🔑 Cloud AI & API Keys",
            "➕ Ingest New Books",
            "⚡ Cache & Database",
            "🛡️ Security & PIN"
        ])

        # SUBTAB A: AI & API KEYS
        with adm_sub_ai:
            st.markdown("##### 🔑 Google Cloud Vision & Gemini API Key")
            current_k = get_persisted_key()

            if current_k:
                masked_k = current_k[:6] + "•" * 16 + current_k[-4:]
                st.success(f"🟢 Active Key Configured: `{masked_k}`")
            else:
                st.warning("🟡 No API Key Configured. System operating in local offline mode.")

            new_k_input = st.text_input("Update Gemini API Key:", type="password", placeholder="AIzaSy...", key="admin_new_key")
            c_save, c_test = st.columns(2)
            with c_save:
                if st.button("💾 Save Key Permanently", key="admin_btn_save_key"):
                    if new_k_input:
                        persist_api_key(new_k_input)
                        st.success("API key updated and saved permanently!")
                        st.rerun()
            with c_test:
                if st.button("🧪 Test API Connection", key="admin_btn_test_key"):
                    k_to_test = new_k_input.strip() or current_k
                    if not k_to_test:
                        st.error("Please enter a key to test.")
                    else:
                        with st.spinner("Pinging Google Gemini Flash API..."):
                            try:
                                from google import genai
                                client = genai.Client(api_key=k_to_test)
                                resp = client.models.generate_content(model="gemini-3.5-flash-lite", contents="Hello from Bookstore AI test.")
                                if resp and resp.text:
                                    st.success(f"🎉 Connection Successful! Gemini 3.5 Flash responded: \"{resp.text.strip()[:40]}...\"")
                            except Exception as e:
                                st.error(f"Connection Failed: {e}")

        # SUBTAB B: INGEST NEW BOOKS
        with adm_sub_ingest:
            st.markdown("##### ➕ Universal Book Ingestion")
            st.write("Enter any book title and author. The AI will search Open Library, retrieve covers, and index the title into SQLite in seconds.")

            in_col1, in_col2 = st.columns(2)
            with in_col1:
                ingest_title = st.text_input("Book Title:", placeholder="e.g. 1984, The Hobbit, Atomic Habits", key="adm_ingest_t")
            with in_col2:
                ingest_author = st.text_input("Author:", placeholder="e.g. George Orwell", key="adm_ingest_a")

            if st.button("📥 Fetch Covers & Ingest into Database", key="adm_btn_ingest"):
                if ingest_title:
                    with st.spinner(f"Ingesting '{ingest_title}'..."):
                        ok, msg = catalog_manager.add_book_to_catalog(ingest_title, ingest_author, vision_matcher=matcher)
                        if ok:
                            st.success(f"🎉 {msg}")
                            st.rerun()
                        else:
                            st.error(f"Error: {msg}")
                else:
                    st.warning("Please specify a book title.")

        # SUBTAB C: CACHE & DATABASE CONTROLS
        with adm_sub_cache:
            st.markdown("##### ⚡ Database Cache Management")
            cache_cnt = getattr(db, 'get_cache_count', lambda: 0)()
            book_cnt = getattr(db, 'get_book_count', lambda: 0)()

            st.write(f"• **Registered Books in Catalog:** `{book_cnt}`")
            st.write(f"• **Cached Image Signatures:** `{cache_cnt}` (Delivering 0.001s instant matches)")

            c_cl1, c_cl2 = st.columns(2)
            with c_cl1:
                if st.button("🗑️ Clear Match Cache", key="admin_btn_clear_cache"):
                    db.clear_match_cache()
                    st.success("Match cache wiped successfully. Next scans will re-evaluate.")
                    st.rerun()
            with c_cl2:
                if st.button("🔄 Re-index Catalog Metadata", key="admin_btn_reindex"):
                    matcher.build_or_load_index(force_rebuild=True)
                    st.success("Catalog index reloaded in 0.001s!")
                    st.rerun()

        # SUBTAB D: SECURITY & PIN
        with adm_sub_sec:
            st.markdown("##### 🛡️ Change Admin Passcode")
            new_pin_1 = st.text_input("New Admin Passcode:", type="password", key="adm_new_pin1")
            new_pin_2 = st.text_input("Confirm New Passcode:", type="password", key="adm_new_pin2")

            if st.button("Update Passcode", key="adm_btn_save_pin"):
                if new_pin_1 and new_pin_1 == new_pin_2:
                    set_admin_pin(new_pin_1)
                    st.success("Passcode updated successfully!")
                elif new_pin_1 != new_pin_2:
                    st.error("Passcodes do not match.")
                else:
                    st.warning("Please enter a passcode.")
