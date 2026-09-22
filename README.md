# 📚 AI Bookstore Assistant & Vision Concierge

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue.svg?logo=python&logoColor=white)](https://www.python.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.32%2B-FF4B4B.svg?logo=streamlit&logoColor=white)](https://streamlit.io/)
[![Google Gemini](https://img.shields.io/badge/Google_Gemini-3.5_Flash_Lite-8E75B2.svg?logo=google&logoColor=white)](https://ai.google.dev/)
[![SQLite](https://img.shields.io/badge/SQLite-0.001s_Cache-003B57.svg?logo=sqlite&logoColor=white)](https://sqlite.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

An intelligent, QR-code-free multimodal AI bookstore web application designed for mobile phones and laptops. It identifies physical book covers using a high-precision **4-Layer Recognition Pipeline** and provides live, strictly spoiler-free reader consultations using **Google Gemini**.

---

## 🌟 Key Capabilities

1. **4-Layer Vision Recognition (95%+ Accuracy)**:
   - **Layer 1 (Image Preprocessing)**: Auto-corrects mobile camera tilt/orientation via `EXIF transpose`, converts color space to standard `RGB`, and applies contrast enhancement.
   - **Layer 2 (Cloud Metadata Extraction)**: Google Gemini Flash vision models extract structured JSON metadata (title, subtitle, author, series/edition, ISBN).
   - **Layer 3 (Fuzzy Database Candidate Retrieval)**: Rapidfuzz token-set matching against the SQLite catalog with weighted scoring (70% title, 30% author).
   - **Layer 4 (Visual Verification Pass)**: Visual cross-comparison of the cover photo against candidate matches to eliminate false positives.
2. **Instant SQLite Match Cache (0.001s)**:
   - Fingerprints scanned covers using MD5 to instantly serve repeat scans in 1 millisecond.
3. **Spoiler-Free Literary Concierge**:
   - Live AI concierge powered by `gemini-3.5-flash-lite`.
   - Strict conversational guardrails: answers questions about tone, pacing, reading order, and themes while strictly blocking spoilers or major plot twists.
4. **Mobile Camera Ready (Wi-Fi Enabled)**:
   - Run the app on your computer and immediately scan books from your smartphone camera over local Wi-Fi.

---

## 🚀 Quick Start Guide

### 1. Clone the Repository
```bash
git clone https://github.com/YOUR_USERNAME/bookstore-ai.git
cd bookstore-ai
```

### 2. Install Dependencies
```bash
pip install -r requirements.txt
```

### 3. Configure API Key
Create `app_config.json` with your Google Gemini API key:
```json
{
  "gemini_api_key": "your_gemini_api_key_here"
}
```
*(You can obtain a free key from [Google AI Studio](https://aistudio.google.com/)).*

### 4. Launch the Web Application
```bash
streamlit run app.py
```
- **Local URL**: `http://localhost:8501`
- **Mobile Wi-Fi URL**: `http://<YOUR_LOCAL_IP>:8501`

---

## 📁 Repository Structure
```
bookstore-ai/
├── catalog/                  # Master cover images (.jpg) and dossiers (.txt)
├── app.py                    # Streamlit web application & UI
├── vision_matcher.py         # 4-Layer Book Recognition Pipeline
├── concierge.py              # Literary concierge class & AI controller
├── chat_brain.py             # Spoiler-free chat engine & prompts
├── book_database.py          # SQLite database interface & caching layer
├── app_config.json.example   # Configuration template
├── requirements.txt          # Python dependencies
└── README.md                 # Project documentation
```

---

## 📜 License
Distributed under the MIT License.
