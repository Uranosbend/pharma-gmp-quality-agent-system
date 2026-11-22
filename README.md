# PharmaGMP Plant 04 — 3-Tier Hierarchical Multi-Agent Orchestra & Quality Control Tower

A lightweight, presentation-ready **GMP / ALCOA+ Hierarchical Multi-Agent Maintenance & Quality Management System** built with **LangGraph** and **Streamlit** that runs smoothly locally and on **Streamlit Community Cloud (free tier, 1GB RAM limit)**.

---

## Core Capabilities

- **3-Tier Parallel Orchestra (`LangGraph` Supervisor Pattern):**
  - **Tier 3 Conductor (`orchestrator`):** Central supervisory agent governing Gates 1–4 (`INTAKE`, `DIAGNOSTIC`, `SQL_MUTATION`, `REPAIR_COST`, `VIBRATION_GATE`, `REWORK_COST`, `LINE_CLEARANCE`, `CLOSURE`). Workers never call each other directly.
  - **Tier 2 Parallel Engineers & Specialists (`Send()` Fan-Out):** Concurrent **Reliability & Diagnostic Engineer** (Hybrid TF-IDF + BM25 PDF RAG over **24 pages** of SOPs) + **Batch & Stock Pre-Check**, plus **Stock & Auto-Procurement Agent** (safe read-only SQL + atomic reservations + automatic Purchase Requisitions).
  - **Tier 1 Field Technician & Upward Escalation:** Executes LOTO, parts swaps, and post-repair vibration checks against `<= 2.5 mm/s RMS` (ISO 20816-3). Failures (`4.8 mm/s` or `3.9 mm/s`) automatically escalate **UP** to the Tier 2 Reliability Engineer for corrective laser/thermal alignment specs before rework.
  - **Tier 3 QA Line Clearance & Analytics/CAPA Analyst:** Verifies sterile boundary & tool reconciliation, calculates Downtime/MTTR/Financial impact, opens 5-Why CAPAs on repeat asset failures, and generates **1-page Incident Closure Certificates (.pdf)** + **4-sheet Monthly Reports (.xlsx)**.
- **Continuous Autonomous Drift Detection & Multi-Sensor Variance:**
  - Live 1.5-second oscilloscope monitoring `Radial-X / Y`, `Axial-Z`, `Peak-g`, `Kurtosis`, `Frequency (Hz)`, `Temp (°C)`, and `Amps`.
  - Toggle **Autonomous Fault Drift** to let natural stochastic wear emerge over time and trigger the multi-agent orchestra automatically when `>= 2` consecutive samples exceed ISO limits.
- **Human-in-the-Loop (HITL) Governance & Bilingual UI (English / Türkçe):**
  - Pauses explicitly at the top of the dashboard for binding Human QA Approve/Deny decisions on SQLite inventory mutations (`SQL_MUTATION`) and estimated repair/rework budgets (`REPAIR_COST`, `REWORK_COST`).
  - Full live UI & AI output language toggle (**English** / **Türkçe**).
- **100% Free-Tier LLM Support & Resilience (`src/llm_config.py`):**
  - **Groq Free:** `openai/gpt-oss-120b`, `openai/gpt-oss-20b`, `qwen/qwen3.8-27b`, live `/models` discovery, automatic `404` model fallback, and `429` exponential backoff.
  - **Google Gemini Free (AI Studio)** & **Ollama Local** + deterministic **Offline Safe Demo Mode**.

---

## Local Quickstart

```bash
python -m pip install -r requirements.txt
copy .env.example .env   # Add your GROQ_API_KEY (gitignored cleanly by .gitignore)
python src/seed_data.py
python -m streamlit run app.py
```

Run the automated test suite:

```bash
python tests/test_pipeline.py
```

---

## Streamlit Community Cloud Publishing Guide (Tomorrow's Classroom Demo — 1-Click Public URL)

To run this app directly in any browser on the smartboard **without bringing your laptop**:

1. **Create a GitHub Repository & Push This Folder:**
   - Do **NOT** upload `.env` or `.streamlit/secrets.toml` (already blocked by `.gitignore`).
   - Push `app.py`, `src/`, `requirements.txt`, `.python-version`, `.streamlit/config.toml`, `.gitignore`, and `README.md`.
2. **Deploy on Streamlit Community Cloud (`share.streamlit.io`):**
   - Sign in with GitHub at **https://share.streamlit.io** and click **Create app** -> **Yup, I have an app**.
   - Pick your GitHub repository & branch, and set **Main file path** to `app.py`.
3. **Add Your API Key Securely in Cloud Secrets:**
   - Before or right after clicking **Deploy**, open **App Settings -> Secrets** and paste:
     ```toml
     GROQ_API_KEY = "gsk_your_groq_key_here"
     ```
   - Streamlit injects Secrets directly into `st.secrets` securely on the server without exposing your key in GitHub source code.
4. **Open & Present Anywhere:**
   - Share or bookmark the resulting `https://<your-app>.streamlit.app` URL (for classroom security, keep only a free-tier key with usage limits or clear/rotate it after presentation).
