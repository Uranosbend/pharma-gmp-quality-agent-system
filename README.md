# PharmaGMP Plant 04 — 3-Tier Hierarchical Multi-Agent Orchestra & Quality Control Tower

A lightweight, portfolio-ready **GMP / ALCOA+ Hierarchical Multi-Agent Maintenance & Quality Management System** built with **LangGraph** and **Streamlit** that runs smoothly locally and on **Streamlit Community Cloud (free tier, 1GB RAM limit)**.

- **Live Public Demo:** https://pharma-gmp-quality-agen-hawzbdq3lonptktfsonj4v.streamlit.app/
  - Opens directly in **Offline Safe Demo Mode** (no API key required) with full interactive LangGraph orchestration, Hybrid PDF RAG, SQLite inventory + Auto-PO, Human-in-the-Loop QA gates, 2-page PDF Closure Certificates, and 4-sheet `.xlsx` reports.
  - Supports **Bring-Your-Own-Key (BYOK)** in the sidebar for **Groq Free**, **Google Gemini Free**, or **Ollama Local** if visitors want to run Live Autonomous LLM tool calls.

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
  - **Groq Free:** `openai/gpt-oss-20b` (default), `openai/gpt-oss-120b`, `qwen/qwen3.8-27b`, live `/models` discovery, automatic `404` model fallback, and TPM/TPD rate-limit resilience.
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

## Self-Hosting & Streamlit Community Cloud Deployment

To host your own public or classroom instance:

1. **Create a GitHub Repository & Push This Folder:**
   - Do **NOT** upload `.env` or `.streamlit/secrets.toml` (already blocked by `.gitignore`).
   - Push `app.py`, `src/`, `requirements.txt`, `.python-version`, `.streamlit/config.toml`, `.gitignore`, and `README.md`.
2. **Deploy on Streamlit Community Cloud (`share.streamlit.io`):**
   - Sign in with GitHub at **https://share.streamlit.io** and click **Create app** -> **Yup, I have an app**.
   - Pick your GitHub repository & branch, and set **Main file path** to `app.py`.
3. **API Key Handling (Optional Secrets vs. Public Zero-Key Demo):**
   - Leave **App Settings -> Secrets** empty for public portfolio sharing — the app automatically defaults to **Offline Safe Demo Mode** with no API key required, and visitors can optionally paste their own key (BYOK) in the sidebar.
   - If you want Live LLM Mode pre-enabled for a private demo, add your key in **App Settings -> Secrets**:
     ```toml
     GROQ_API_KEY = "gsk_your_groq_key_here"
     ```
4. **Open & Share:**
   - Share your `https://<your-app>.streamlit.app` link safely with zero personal credentials exposed.
