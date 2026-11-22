"""Specialist tools, lightweight hybrid TF-IDF + BM25 PDF RAG, SQLite ops, and Excel reporting."""

from __future__ import annotations

import io
import json
import re
import sqlite3
import threading
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import openpyxl
from openpyxl.styles import Alignment, Border, Font, PatternFill, Side
from openpyxl.utils import get_column_letter
import pandas as pd
import pypdfium2 as pdfium
from langchain_core.tools import tool
from pypdf import PdfReader
from rank_bm25 import BM25Okapi
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from src.seed_data import (
    DB_PATH,
    DOCS_DIR,
    GMP_VIBRATION_LIMIT_RMS,
    LABOR_RATE_USD_PER_HR,
    REPORTS_DIR,
    SOPS_DIR,
    _compute_audit_hash,
    ensure_seed_data,
)


# ---------------------------------------------------------------------------
# Per-run context tracking (captures authentic tool evidence for eBMR & UI)
# ---------------------------------------------------------------------------


@dataclass
class RunContext:
    """Mutable per-run collector bound to the executing thread."""

    ticket_id: str
    scenario_id: str
    machine_id: str
    alarm_vibration_rms: float
    symptom_summary: str
    inject_post_repair_fail: bool = False
    vibration_test_count: int = 0
    repair_attempt_count: int = 0
    latest_vibration_rms: Optional[float] = None
    part_used: Optional[str] = None
    used_alternative: bool = False
    part_cost_usd: float = 0.0
    sop_citation: str = ""
    rag_chunks: List[Dict[str, Any]] = field(default_factory=list)
    sql_queries: List[Dict[str, Any]] = field(default_factory=list)
    reservations: List[Dict[str, Any]] = field(default_factory=list)
    maintenance_attempts: List[Dict[str, Any]] = field(default_factory=list)
    telemetry_readings: List[Dict[str, Any]] = field(default_factory=list)
    batch_risk_report: Optional[Dict[str, Any]] = None
    clearance_report: Optional[Dict[str, Any]] = None
    analytics_report: Optional[Dict[str, Any]] = None
    excel_path: Optional[str] = None
    audit_step_counter: int = 0

    def next_step_number(self, fallback: int = 1) -> int:
        with _CONTEXT_LOCK:
            if self.audit_step_counter < fallback - 1:
                self.audit_step_counter = fallback - 1
            self.audit_step_counter += 1
            return self.audit_step_counter


_CONTEXT_LOCK = threading.Lock()
_AUDIT_WRITE_LOCK = threading.Lock()
_THREAD_CONTEXTS: Dict[int, RunContext] = {}
_FALLBACK_CONTEXT: Optional[RunContext] = None


def set_active_run_context(ctx: RunContext) -> None:
    """Bind a RunContext to the current worker thread."""
    global _FALLBACK_CONTEXT
    tid = threading.get_ident()
    with _CONTEXT_LOCK:
        _THREAD_CONTEXTS[tid] = ctx
        _FALLBACK_CONTEXT = ctx


def clear_active_run_context() -> None:
    """Unbind the current thread's RunContext after workflow completion."""
    tid = threading.get_ident()
    with _CONTEXT_LOCK:
        _THREAD_CONTEXTS.pop(tid, None)


def get_active_run_context() -> Optional[RunContext]:
    """Retrieve the active RunContext for the current thread (if any)."""
    tid = threading.get_ident()
    with _CONTEXT_LOCK:
        return _THREAD_CONTEXTS.get(tid) or _FALLBACK_CONTEXT


# ---------------------------------------------------------------------------
# SQLite Connection Helpers & ALCOA+ Audit Trail
# ---------------------------------------------------------------------------


def _get_rw_conn() -> sqlite3.Connection:
    ensure_seed_data()
    conn = sqlite3.connect(str(DB_PATH), timeout=15.0)
    conn.row_factory = sqlite3.Row
    return conn


def _get_ro_conn() -> sqlite3.Connection:
    ensure_seed_data()
    uri = f"file:{DB_PATH.resolve().as_posix()}?mode=ro"
    conn = sqlite3.connect(uri, uri=True, timeout=15.0)
    conn.row_factory = sqlite3.Row
    return conn


def append_gmp_audit_log(
    ticket_id: str,
    step_number: int,
    agent_name: str,
    action_summary: str,
    gate_decision: str,
    audit_notes: str,
) -> Dict[str, Any]:
    """Append an immutable ALCOA+ SHA-256 chained audit row to gmp_audit_log (thread-safe)."""
    ctx = get_active_run_context()
    effective_step = ctx.next_step_number(fallback=step_number) if ctx else step_number
    ts = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    with _AUDIT_WRITE_LOCK:
        conn = _get_rw_conn()
        try:
            cur = conn.cursor()
            last = cur.execute(
                "SELECT record_hash FROM gmp_audit_log ORDER BY audit_id DESC LIMIT 1"
            ).fetchone()
            prev_hash = str(last["record_hash"]) if last else "GENESIS"
            rec_hash = _compute_audit_hash(
                prev_hash=prev_hash,
                ticket_id=ticket_id,
                step_number=effective_step,
                timestamp_utc=ts,
                agent_name=agent_name,
                action_summary=action_summary,
                gate_decision=gate_decision,
                audit_notes=audit_notes,
            )
            cur.execute(
                """
                INSERT INTO gmp_audit_log (
                    ticket_id, step_number, timestamp_utc, agent_name,
                    action_summary, gate_decision, audit_notes, prev_hash, record_hash
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    ticket_id,
                    effective_step,
                    ts,
                    agent_name,
                    action_summary,
                    gate_decision,
                    audit_notes,
                    prev_hash,
                    rec_hash,
                ),
            )
            conn.commit()
        finally:
            conn.close()

    return {
        "step_number": effective_step,
        "timestamp_utc": ts,
        "ticket_id": ticket_id,
        "agent_name": agent_name,
        "action_summary": action_summary,
        "gate_decision": gate_decision,
        "audit_notes": audit_notes,
        "record_hash": rec_hash[:12],
    }


# ---------------------------------------------------------------------------
# Lightweight Hybrid TF-IDF + BM25 PDF SOP Retriever (Zero Torch / Embeddings)
# ---------------------------------------------------------------------------


def _tokenize_technical(text: str) -> List[str]:
    """Tokenize text while keeping hyphenated equipment/part codes intact."""
    tokens = re.findall(r"[a-z0-9]+(?:-[a-z0-9]+)*", text.lower())
    expanded: List[str] = []
    for tok in tokens:
        expanded.append(tok)
        if "-" in tok:
            expanded.extend(part for part in tok.split("-") if part)
    return expanded


@dataclass
class SOPChunk:
    filename: str
    sop_code: str
    page_number: int
    text: str


class HybridSOPRetriever:
    """In-memory hybrid BM25 + TF-IDF cosine retriever over local pypdf pages."""

    def __init__(self, sops_dir: Path = SOPS_DIR) -> None:
        self.sops_dir = sops_dir
        self.chunks: List[SOPChunk] = []
        self._bm25: Optional[BM25Okapi] = None
        self._vectorizer: Optional[TfidfVectorizer] = None
        self._tfidf_matrix: Any = None
        self._build_index()

    def _build_index(self) -> None:
        ensure_seed_data()
        chunks: List[SOPChunk] = []
        pdf_files = sorted(self.sops_dir.glob("*.pdf"))

        for pdf_path in pdf_files:
            reader = PdfReader(str(pdf_path))
            code_match = re.search(r"((?:SOP|GMP)-[A-Z]+-\d+)", pdf_path.name)
            default_code = code_match.group(1) if code_match else pdf_path.stem

            for page_idx, page in enumerate(reader.pages, start=1):
                raw_text = (page.extract_text() or "").strip()
                if not raw_text:
                    continue
                # Normalize internal PDF line breaks while preserving paragraph sections
                lines = [ln.strip() for ln in raw_text.splitlines() if ln.strip()]
                # Strip repeated header/footer chrome from snippet clarity
                body_lines = [
                    ln
                    for ln in lines
                    if not ln.startswith("PHARMA-GMP OPS  |")
                    and not re.match(r"^(SOP|GMP)-[A-Z]+-\d+ - .+Page \d+ of \d+$", ln)
                ]
                joined = "\n".join(body_lines) if body_lines else raw_text

                # Create focused overlapping chunks (~700 chars) if long, else full page
                if len(joined) <= 950:
                    chunks.append(
                        SOPChunk(
                            filename=pdf_path.name,
                            sop_code=default_code,
                            page_number=page_idx,
                            text=joined,
                        )
                    )
                else:
                    # Group lines into balanced chunks of ~650 chars
                    buf: List[str] = []
                    buf_len = 0
                    for ln in body_lines:
                        buf.append(ln)
                        buf_len += len(ln) + 1
                        if buf_len >= 650:
                            chunks.append(
                                SOPChunk(
                                    filename=pdf_path.name,
                                    sop_code=default_code,
                                    page_number=page_idx,
                                    text="\n".join(buf),
                                )
                            )
                            buf = buf[-2:] if len(buf) > 2 else []
                            buf_len = sum(len(x) + 1 for x in buf)
                    if buf:
                        chunks.append(
                            SOPChunk(
                                filename=pdf_path.name,
                                sop_code=default_code,
                                page_number=page_idx,
                                text="\n".join(buf),
                            )
                        )

        self.chunks = chunks
        if not self.chunks:
            return

        tokenized_corpus = [_tokenize_technical(c.text) for c in self.chunks]
        self._bm25 = BM25Okapi(tokenized_corpus)
        self._vectorizer = TfidfVectorizer(
            ngram_range=(1, 2),
            token_pattern=r"(?u)\b[a-zA-Z0-9]+(?:-[a-zA-Z0-9]+)*\b",
            stop_words="english",
            sublinear_tf=True,
        )
        self._tfidf_matrix = self._vectorizer.fit_transform([c.text for c in self.chunks])

    @staticmethod
    def _min_max_norm(scores: np.ndarray) -> np.ndarray:
        if scores.size == 0:
            return scores
        s_min = float(np.min(scores))
        s_max = float(np.max(scores))
        if s_max - s_min < 1e-9:
            return np.ones_like(scores) if s_max > 0 else np.zeros_like(scores)
        return (scores - s_min) / (s_max - s_min)

    def search(self, query: str, top_k: int = 4) -> List[Dict[str, Any]]:
        if not self.chunks or self._bm25 is None or self._vectorizer is None:
            return []

        q_tokens = _tokenize_technical(query)
        bm25_raw = np.array(self._bm25.get_scores(q_tokens), dtype=float)
        q_vec = self._vectorizer.transform([query])
        tfidf_raw = cosine_similarity(q_vec, self._tfidf_matrix).flatten()

        bm25_norm = self._min_max_norm(bm25_raw)
        tfidf_norm = self._min_max_norm(tfidf_raw)
        hybrid = 0.55 * bm25_norm + 0.45 * tfidf_norm

        # Weighted precision boost: exact spare part codes rank above broad machine codes
        upper_q = query.upper()
        part_codes = re.findall(
            r"\b(?:BRG-4402(?:-ALT)?|LUB-3010|RLR-4450(?:-ALT)?|SL-8810(?:-ALT)?|IMP-8825|SHM-8805|BLT-2205(?:-ALT)?|SRV-2240|BRG-2290|GKT-1102|SNS-9010)\b",
            upper_q,
        )
        equip_codes = re.findall(r"\b(?:TP-01|RP-04|PK-02|SOP-MNT-001|SOP-MNT-002|GMP-QA-010)\b", upper_q)
        if part_codes or equip_codes:
            for idx, ch in enumerate(self.chunks):
                ch_upper = ch.text.upper()
                if part_codes and any(code in ch_upper for code in part_codes):
                    hybrid[idx] += 0.28
                if equip_codes and any(code in ch_upper for code in equip_codes):
                    hybrid[idx] += 0.12

        top_indices = np.argsort(hybrid)[::-1][:top_k]
        results: List[Dict[str, Any]] = []
        for rank, idx in enumerate(top_indices, start=1):
            chunk = self.chunks[int(idx)]
            results.append(
                {
                    "rank": rank,
                    "score": round(float(hybrid[int(idx)]), 4),
                    "bm25_score": round(float(bm25_norm[int(idx)]), 4),
                    "tfidf_score": round(float(tfidf_norm[int(idx)]), 4),
                    "filename": chunk.filename,
                    "sop_code": chunk.sop_code,
                    "page_number": chunk.page_number,
                    "excerpt": chunk.text,
                }
            )
        return results


_RAG_CACHE_LOCK = threading.Lock()
_RAG_RETRIEVER: Optional[HybridSOPRetriever] = None


def get_sop_retriever() -> HybridSOPRetriever:
    """Return cached HybridSOPRetriever instance (zero RAM bloat)."""
    global _RAG_RETRIEVER
    with _RAG_CACHE_LOCK:
        if _RAG_RETRIEVER is None:
            _RAG_RETRIEVER = HybridSOPRetriever(SOPS_DIR)
        return _RAG_RETRIEVER


def invalidate_rag_cache() -> None:
    """Force rebuild of the PDF RAG index after SOP regeneration."""
    global _RAG_RETRIEVER
    with _RAG_CACHE_LOCK:
        _RAG_RETRIEVER = None


# ---------------------------------------------------------------------------
# Tool Implementations (Shared by LangChain @tool wrappers & Offline Safe Mode)
# ---------------------------------------------------------------------------


def search_sop_manuals_impl(query: str, top_k: int = 4) -> Dict[str, Any]:
    retriever = get_sop_retriever()
    hits = retriever.search(query=query, top_k=top_k)
    ctx = get_active_run_context()
    if ctx is not None:
        existing = {(c["filename"], c["page_number"], c["excerpt"][:80]) for c in ctx.rag_chunks}
        for h in hits:
            key = (h["filename"], h["page_number"], h["excerpt"][:80])
            if key not in existing:
                ctx.rag_chunks.append(h)
                existing.add(key)
        if hits and not ctx.sop_citation:
            ctx.sop_citation = f"{hits[0]['sop_code']} p.{hits[0]['page_number']}"
    return {
        "query": query,
        "retriever": "TF-IDF + BM25 Hybrid (pypdf)",
        "total_results": len(hits),
        "results": hits,
    }


def read_sensor_telemetry_impl(machine_id: str) -> Dict[str, Any]:
    clean_id = machine_id.strip().upper()
    conn = _get_ro_conn()
    try:
        row = conn.execute(
            "SELECT * FROM machines WHERE UPPER(machine_id) = ?", (clean_id,)
        ).fetchone()
    finally:
        conn.close()

    if not row:
        return {"error": f"Unknown machine_id '{machine_id}'"}

    ctx = get_active_run_context()
    from src.state import DEMO_SCENARIOS

    from src.telemetry import classify_iso_20816_zone

    alarm_rms = float(row["baseline_vibration_rms"])
    temp_c = 45.0
    dom_freq = 60.0
    peak_g = 0.45
    kurtosis = 3.0
    motor_amps = 16.0
    symptom = "Nominal operating telemetry."
    status = "NORMAL"

    if ctx is not None and ctx.machine_id.upper() == clean_id:
        alarm_rms = ctx.latest_vibration_rms if ctx.latest_vibration_rms is not None else ctx.alarm_vibration_rms
        if ctx.scenario_id in DEMO_SCENARIOS:
            sc = DEMO_SCENARIOS[ctx.scenario_id]
            temp_c = float(sc.get("temperature_c", 68.0))
            dom_freq = float(sc.get("dominant_frequency_hz", 180.0))
            peak_g = float(sc.get("peak_accel_g", 3.5))
            kurtosis = float(sc.get("kurtosis", 5.2))
            motor_amps = float(sc.get("motor_amps", 20.0))
        elif alarm_rms <= float(row["gmp_vibration_limit_rms"]):
            temp_c = 46.0
            dom_freq = 50.0
            peak_g = 0.65
            kurtosis = 3.1
            motor_amps = 16.2
        else:
            temp_c = 64.0
            dom_freq = 155.0
            peak_g = 3.2
            kurtosis = 4.9
            motor_amps = 21.0
        symptom = ctx.symptom_summary
        status = "CRITICAL_ALARM" if alarm_rms > float(row["gmp_vibration_limit_rms"]) else "WITHIN_GMP_LIMIT"

    payload = {
        "machine_id": row["machine_id"],
        "machine_name": row["machine_name"],
        "area": row["area"],
        "nominal_rpm": int(row["nominal_rpm"]),
        "baseline_vibration_rms": float(row["baseline_vibration_rms"]),
        "measured_vibration_rms": round(alarm_rms, 2),
        "gmp_vibration_limit_rms": float(row["gmp_vibration_limit_rms"]),
        "iso_20816_zone": classify_iso_20816_zone(alarm_rms),
        "housing_temperature_c": temp_c,
        "dominant_frequency_hz": dom_freq,
        "peak_accel_g": peak_g,
        "kurtosis": kurtosis,
        "motor_amps": motor_amps,
        "sensor_status": status,
        "symptom_observation": symptom,
        "downtime_cost_per_min_usd": float(row["downtime_cost_per_min"]),
    }
    if ctx is not None:
        ctx.telemetry_readings.append({"phase": "diagnostic_intake", **payload})
    return payload


_FORBIDDEN_SQL_RE = re.compile(
    r"\b(INSERT|UPDATE|DELETE|DROP|ALTER|CREATE|REPLACE|ATTACH|DETACH|PRAGMA|VACUUM|REINDEX|BEGIN|COMMIT|ROLLBACK)\b",
    flags=re.IGNORECASE,
)


def query_inventory_sql_impl(sql_query: str) -> Dict[str, Any]:
    raw = (sql_query or "").strip()
    if raw.endswith(";"):
        raw = raw[:-1].strip()

    if not raw:
        return {"error": "Empty SQL query rejected."}
    if ";" in raw or "--" in raw or "/*" in raw:
        return {"error": "Security Policy: Multi-statement or commented SQL is forbidden."}
    if not re.match(r"^(SELECT|WITH)\b", raw, flags=re.IGNORECASE):
        return {"error": "Read-Only Policy: Only SELECT / WITH queries are permitted."}
    if _FORBIDDEN_SQL_RE.search(raw):
        return {"error": "Read-Only Policy: Mutation or DDL keywords are blocked."}

    conn = _get_ro_conn()
    try:
        rows = conn.execute(raw).fetchall()
        data = [dict(r) for r in rows[:50]]
    except sqlite3.Error as exc:
        return {"error": f"SQLite query error: {exc}", "sql_query": raw}
    finally:
        conn.close()

    ctx = get_active_run_context()
    if ctx is not None:
        ctx.sql_queries.append(
            {
                "sql": raw,
                "row_count": len(data),
                "rows_preview": data[:6],
                "timestamp_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
            }
        )

    return {"sql_query": raw, "row_count": len(data), "rows": data}


def check_alternative_parts_impl(part_code: str) -> Dict[str, Any]:
    clean_code = part_code.strip().upper()
    sql = (
        "SELECT p.part_code AS primary_part_code, p.part_name AS primary_part_name, "
        "p.stock_qty AS primary_stock_qty, p.compatible_alt_code, p.gmp_qualified_alt, "
        "a.part_name AS alt_part_name, a.stock_qty AS alt_stock_qty, "
        "a.bin_location AS alt_bin_location, a.unit_cost_usd AS alt_unit_cost_usd, "
        "a.notes AS alt_notes "
        "FROM spare_parts_inventory p "
        "LEFT JOIN spare_parts_inventory a ON p.compatible_alt_code = a.part_code "
        "WHERE UPPER(p.part_code) = ?"
    )
    conn = _get_ro_conn()
    try:
        row = conn.execute(sql, (clean_code,)).fetchone()
    finally:
        conn.close()

    ctx = get_active_run_context()
    result = dict(row) if row else {"error": f"Part code '{clean_code}' not found in inventory."}

    if ctx is not None:
        ctx.sql_queries.append(
            {
                "sql": sql.replace("?", f"'{clean_code}'"),
                "row_count": 1 if row else 0,
                "rows_preview": [result] if row else [],
                "timestamp_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
            }
        )

    if not row:
        return result

    alt_code = row["compatible_alt_code"]
    alt_stock = int(row["alt_stock_qty"] or 0)
    return {
        **result,
        "has_qualified_in_stock_alternative": bool(alt_code and int(row["gmp_qualified_alt"]) == 1 and alt_stock > 0),
    }


def reserve_spare_part_impl(part_code: str, quantity: int, ticket_id: str) -> Dict[str, Any]:
    clean_code = part_code.strip().upper()
    qty = max(1, int(quantity))
    ctx = get_active_run_context()
    effective_ticket = ticket_id.strip() if ticket_id and ticket_id.strip() else (ctx.ticket_id if ctx else "MANUAL-TKT")

    conn = _get_rw_conn()
    try:
        cur = conn.cursor()
        # Prevent accidental duplicate reservation for the exact same ticket + part
        existing_res = cur.execute(
            "SELECT * FROM part_reservations WHERE ticket_id = ? AND UPPER(part_code) = ?",
            (effective_ticket, clean_code),
        ).fetchone()
        if existing_res:
            part_row = cur.execute(
                "SELECT * FROM spare_parts_inventory WHERE UPPER(part_code) = ?",
                (clean_code,),
            ).fetchone()
            return {
                "status": "RESERVED",
                "already_reserved_for_ticket": True,
                "ticket_id": effective_ticket,
                "part_code": existing_res["part_code"],
                "part_name": part_row["part_name"] if part_row else existing_res["part_code"],
                "quantity_reserved": int(existing_res["quantity"]),
                "remaining_stock_qty": int(part_row["stock_qty"]) if part_row else None,
                "bin_location": existing_res["bin_location"],
                "unit_cost_usd": float(existing_res["unit_cost_usd"]),
                "is_alternative": bool(existing_res["is_alternative"]),
            }

        part_row = cur.execute(
            "SELECT * FROM spare_parts_inventory WHERE UPPER(part_code) = ?",
            (clean_code,),
        ).fetchone()
        if not part_row:
            return {"status": "NOT_FOUND", "part_code": clean_code, "error": "Part not found."}

        stock_qty = int(part_row["stock_qty"])
        if stock_qty < qty:
            return {
                "status": "OUT_OF_STOCK",
                "part_code": part_row["part_code"],
                "part_name": part_row["part_name"],
                "stock_qty": stock_qty,
                "requested_qty": qty,
                "compatible_alt_code": part_row["compatible_alt_code"],
                "message": (
                    f"Stockout on {part_row['part_code']} (stock_qty={stock_qty}). "
                    f"Run check_alternative_parts('{part_row['part_code']}') to evaluate compatible alternatives."
                ),
            }

        is_alt = 1 if clean_code.endswith("-ALT") else 0
        new_qty = stock_qty - qty
        now_str = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")

        cur.execute(
            "UPDATE spare_parts_inventory SET stock_qty = ? WHERE part_code = ?",
            (new_qty, part_row["part_code"]),
        )
        cur.execute(
            """
            INSERT INTO part_reservations (
                ticket_id, part_code, quantity, unit_cost_usd, bin_location, reserved_at, is_alternative
            ) VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            (
                effective_ticket,
                part_row["part_code"],
                qty,
                float(part_row["unit_cost_usd"]),
                part_row["bin_location"],
                now_str,
                is_alt,
            ),
        )
        conn.commit()
    finally:
        conn.close()

    res_payload = {
        "status": "RESERVED",
        "ticket_id": effective_ticket,
        "part_code": part_row["part_code"],
        "part_name": part_row["part_name"],
        "quantity_reserved": qty,
        "previous_stock_qty": stock_qty,
        "remaining_stock_qty": new_qty,
        "bin_location": part_row["bin_location"],
        "unit_cost_usd": float(part_row["unit_cost_usd"]),
        "is_alternative": bool(is_alt),
        "reserved_at": now_str,
    }
    if ctx is not None:
        ctx.reservations.append(res_payload)
        ctx.part_used = part_row["part_code"]
        ctx.used_alternative = bool(is_alt)
        ctx.part_cost_usd = float(part_row["unit_cost_usd"]) * qty
        ctx.sql_queries.append(
            {
                "sql": f"UPDATE spare_parts_inventory SET stock_qty = {new_qty} WHERE part_code = '{part_row['part_code']}'",
                "row_count": 1,
                "rows_preview": [res_payload],
                "timestamp_utc": now_str,
            }
        )
    return res_payload


def execute_part_replacement_impl(
    machine_id: str, old_part: str, new_part: str, sop_ref: str
) -> Dict[str, Any]:
    from src.state import DEMO_SCENARIOS

    ctx = get_active_run_context()
    attempt_num = (len(ctx.maintenance_attempts) + 1) if ctx else 1
    is_realignment = attempt_num > 1
    sc = DEMO_SCENARIOS.get(ctx.scenario_id, {}) if ctx else {}

    if is_realignment:
        procedure = sc.get("replan_procedure") or (
            "Precision Laser Shaft Realignment & Base Bolt Retorque (55 Nm, "
            "parallel offset <= 0.05 mm, angular <= 0.05 mm/100mm) per SOP-MNT-002 p.3"
        )
        torque_spec = sc.get("replan_tolerance") or "55 Nm base mount + soft-foot shim correction"
    else:
        procedure = (
            f"Executed LOTO, extracted worn {old_part}, installed verified {new_part}, "
            f"torqued & calibrated per {sop_ref}, cleared LOTO for verification run."
        )
        torque_spec = f"Calibrated torque & assembly clearance per {sop_ref}"

    record = {
        "attempt_number": attempt_num,
        "machine_id": machine_id.strip().upper(),
        "old_part": old_part.strip().upper(),
        "new_part": new_part.strip().upper(),
        "sop_ref": sop_ref,
        "is_corrective_realignment": is_realignment,
        "procedure_executed": procedure,
        "torque_spec": torque_spec,
        "executed_at_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
    }
    if ctx is not None:
        ctx.repair_attempt_count = attempt_num
        ctx.maintenance_attempts.append(record)
    return record


def run_post_repair_vibration_test_impl(machine_id: str) -> Dict[str, Any]:
    from src.state import DEMO_SCENARIOS
    from src.telemetry import classify_iso_20816_zone

    clean_id = machine_id.strip().upper()
    ctx = get_active_run_context()

    if ctx is not None:
        ctx.vibration_test_count += 1
        attempt_num = ctx.vibration_test_count
        sc = DEMO_SCENARIOS.get(ctx.scenario_id, {})
        fail_rms = sc.get("fail_attempt1_rms")
        pass_rms = float(sc.get("post_repair_pass_rms") or (1.3 if ctx.scenario_id == "custom_unseen" else 1.4))

        if ctx.inject_post_repair_fail and attempt_num == 1:
            measured_rms = float(fail_rms if fail_rms is not None else 4.8)
            if clean_id == "PK-02" or measured_rms < 4.3:
                diagnosis = (
                    f"THERMAL PLATEN CHATTER ({measured_rms:.1f} mm/s RMS > 2.5 mm/s GMP LIMIT): "
                    "Cold bearing fit omitted 160 deg C hot-soak parallelism leveling per SOP-MNT-001 p.6."
                )
            else:
                diagnosis = (
                    f"EXCESSIVE 1x/2x AXIAL VIBRATION ({measured_rms:.1f} mm/s RMS > 2.5 mm/s GMP LIMIT): "
                    "Residual coupling shaft misalignment / soft-foot distortion detected per SOP-MNT-002 p.3."
                )
        elif ctx.inject_post_repair_fail and attempt_num >= 2:
            measured_rms = pass_rms
            if clean_id == "PK-02":
                diagnosis = (
                    f"Hot-soak platen parallelism (<= 0.04 mm) & 38 Nm hot retorque verified ({measured_rms:.1f} mm/s RMS <= 2.5)."
                )
            else:
                diagnosis = (
                    f"Cold dual-beam laser realignment verified (<= 0.05 mm TIR). Vibration stabilized at {measured_rms:.1f} mm/s RMS."
                )
        else:
            measured_rms = pass_rms
            diagnosis = (
                f"Post-repair verification passed at {measured_rms:.1f} mm/s RMS ({classify_iso_20816_zone(measured_rms)} <= 2.5 GMP limit)."
            )
    else:
        attempt_num = 1
        measured_rms = 1.4
        diagnosis = "Post-repair vibration within ISO 20816-3 / cGMP threshold."

    passed = measured_rms <= GMP_VIBRATION_LIMIT_RMS
    result = {
        "machine_id": clean_id,
        "attempt_number": attempt_num,
        "post_repair_vibration_rms": measured_rms,
        "gmp_limit_rms": GMP_VIBRATION_LIMIT_RMS,
        "iso_20816_zone": classify_iso_20816_zone(measured_rms),
        "gmp_gate_result": "PASS_WITHIN_GMP_LIMIT" if passed else "FAIL_EXCEEDS_GMP_LIMIT",
        "passed_gmp_limit": passed,
        "telemetry_notes": diagnosis,
        "measured_at_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
    }
    if ctx is not None:
        ctx.latest_vibration_rms = measured_rms
        ctx.telemetry_readings.append({"phase": f"post_repair_attempt_{attempt_num}", **result})
        if ctx.maintenance_attempts:
            ctx.maintenance_attempts[-1]["post_repair_vibration_rms"] = measured_rms
            ctx.maintenance_attempts[-1]["passed_gmp_limit"] = passed
    return result


def log_incident_analytics_impl(ticket_data: Any) -> Dict[str, Any]:
    if isinstance(ticket_data, str):
        try:
            parsed = json.loads(ticket_data)
        except json.JSONDecodeError:
            parsed = {}
    elif isinstance(ticket_data, dict):
        parsed = dict(ticket_data)
    else:
        parsed = {}

    ctx = get_active_run_context()
    ticket_id = str(parsed.get("ticket_id") or (ctx.ticket_id if ctx else f"GMP-{datetime.now(timezone.utc).strftime('%Y%m%d%H%M%S')}"))
    machine_id = str(parsed.get("machine_id") or (ctx.machine_id if ctx else "TP-01")).upper()

    conn = _get_rw_conn()
    try:
        m_row = conn.execute(
            "SELECT * FROM machines WHERE UPPER(machine_id) = ?", (machine_id,)
        ).fetchone()
        if not m_row:
            m_row = conn.execute("SELECT * FROM machines LIMIT 1").fetchone()

        machine_name = str(m_row["machine_name"])
        downtime_rate = float(m_row["downtime_cost_per_min"])

        # Reconcile authoritative telemetry & parts from RunContext / DB
        if "repair_attempts" in parsed and parsed["repair_attempts"] is not None:
            repair_attempts = int(parsed["repair_attempts"])
        else:
            repair_attempts = max(1, ctx.vibration_test_count) if ctx else 1
        first_time_fix = 1 if repair_attempts <= 1 else 0

        raw_part = str(parsed.get("part_used") or (ctx.part_used if ctx and ctx.part_used else "BRG-4402")).strip()
        is_no_part = raw_part.upper() in ("NONE", "N/A", "CALIBRATION_ONLY", "NO_PART") or repair_attempts == 0
        part_used = "NONE" if is_no_part else raw_part
        used_alternative = (
            0
            if is_no_part
            else int(bool(parsed.get("used_alternative")) or (ctx.used_alternative if ctx else part_used.endswith("-ALT")))
        )

        # Look up actual unit cost from DB if needed
        if is_no_part:
            part_cost_usd = 0.0
        else:
            p_row = conn.execute(
                "SELECT unit_cost_usd FROM spare_parts_inventory WHERE UPPER(part_code) = ?",
                (part_used.upper(),),
            ).fetchone()
            part_cost_usd = float(
                parsed.get("part_cost_usd")
                or (ctx.part_cost_usd if ctx and ctx.part_cost_usd > 0 else (p_row["unit_cost_usd"] if p_row else 350.0))
            )

        alarm_rms = float(
            parsed.get("alarm_vibration_rms")
            if parsed.get("alarm_vibration_rms") is not None
            else (ctx.alarm_vibration_rms if ctx else 9.0)
        )
        final_rms = float(
            parsed.get("final_vibration_rms")
            if parsed.get("final_vibration_rms") is not None
            else (ctx.latest_vibration_rms if ctx and ctx.latest_vibration_rms is not None else (alarm_rms if is_no_part else 1.5))
        )

        base_downtime_map = {"TP-01": 68, "RP-04": 62, "PK-02": 56}
        if is_no_part:
            downtime_minutes = int(parsed.get("downtime_minutes") or 10)
            labor_hours = 0.25
        else:
            default_dt = base_downtime_map.get(machine_id, 60) + (repair_attempts - 1) * 48
            downtime_minutes = int(parsed.get("downtime_minutes") or default_dt)
            if downtime_minutes < 15:
                downtime_minutes = default_dt
            labor_hours = 1.5 if repair_attempts == 1 else 1.5 + (repair_attempts - 1) * 2.0
        labor_cost_usd = round(labor_hours * LABOR_RATE_USD_PER_HR, 2)
        production_loss_usd = round(downtime_minutes * downtime_rate, 2)
        total_cost_usd = round(part_cost_usd + labor_cost_usd + production_loss_usd, 2)

        if is_no_part:
            severity = "Low"
        elif total_cost_usd >= 8500.0:
            severity = "Critical"
        elif total_cost_usd >= 5000.0 or repair_attempts > 1:
            severity = "High"
        elif total_cost_usd >= 2500.0:
            severity = "Medium"
        else:
            severity = "Low"

        failure_mode = str(
            parsed.get("failure_mode")
            or (ctx.symptom_summary[:120] if ctx else "Mechanical Vibration Alarm")
        )
        sop_ref = str(parsed.get("sop_ref") or (ctx.sop_citation if ctx and ctx.sop_citation else "SOP-MNT-001 p.2"))

        now_utc = datetime.now(timezone.utc)
        inc_date = now_utc.strftime("%Y-%m-%d")
        created_at = now_utc.strftime("%Y-%m-%dT%H:%M:%SZ")

        conn.execute(
            """
            INSERT OR REPLACE INTO incident_analytics (
                ticket_id, incident_date, machine_id, machine_name, failure_mode,
                sop_ref, part_used, used_alternative, alarm_vibration_rms,
                final_vibration_rms, repair_attempts, first_time_fix,
                downtime_minutes, part_cost_usd, labor_cost_usd,
                production_loss_usd, total_cost_usd, damage_severity,
                orchestrator_status, created_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                ticket_id,
                inc_date,
                machine_id,
                machine_name,
                failure_mode,
                sop_ref,
                part_used,
                used_alternative,
                alarm_rms,
                final_rms,
                repair_attempts,
                first_time_fix,
                downtime_minutes,
                round(part_cost_usd, 2),
                labor_cost_usd,
                production_loss_usd,
                total_cost_usd,
                severity,
                "SIGNED_OFF",
                created_at,
            ),
        )
        conn.commit()
    finally:
        conn.close()

    df_month = get_monthly_incidents_dataframe()
    kpis = compute_monthly_kpis(df_month)

    summary = {
        "logged_to_db": True,
        "ticket_id": ticket_id,
        "machine_id": machine_id,
        "part_used": part_used,
        "used_alternative": bool(used_alternative),
        "repair_attempts": repair_attempts,
        "first_time_fix": bool(first_time_fix),
        "downtime_minutes": downtime_minutes,
        "mttr_minutes": downtime_minutes,
        "part_cost_usd": round(part_cost_usd, 2),
        "labor_cost_usd": labor_cost_usd,
        "production_loss_usd": production_loss_usd,
        "total_cost_usd": total_cost_usd,
        "damage_severity": severity,
        "monthly_kpis_after_update": kpis,
    }
    if ctx is not None:
        ctx.analytics_report = summary
    return summary


def assess_batch_risk_impl(machine_id: str) -> Dict[str, Any]:
    """Evaluate active GMP batch contamination & downtime exposure for an alarming asset."""
    clean_id = machine_id.strip().upper()
    profiles = {
        "TP-01": {
            "active_batch_id": "LOT-2026-ATRV-441",
            "product_stage": "Solid Dose Compression (Atorvastatin 20 mg)",
            "contamination_risk": "MEDIUM -- Potential tablet weight variation & metal wear debris if run continues",
            "qa_batch_hold_action": "Pause compression feed hopper; quarantine last 15-min tablet bin for weight/metal check",
            "line_priority": "P1 CRITICAL ($68/min stoppage)",
        },
        "RP-04": {
            "active_batch_id": "LOT-2026-WFI-812",
            "product_stage": "Sterile WFI & Granulation Transfer Loop",
            "contamination_risk": "HIGH (EU GMP Annex 1) -- Compromised seal gland compromises sterile barrier",
            "qa_batch_hold_action": "Isolate WFI transfer loop; hold downstream granulation vessel pending QA clearance",
            "line_priority": "P1 CRITICAL ($52/min stoppage)",
        },
        "PK-02": {
            "active_batch_id": "LOT-2026-BLST-190",
            "product_stage": "Primary Alu-Alu Blister Packaging",
            "contamination_risk": "LOW-MEDIUM -- Registration drift / incomplete foil seal integrity risk",
            "qa_batch_hold_action": "Divert blister strips produced since alarm timestamp to 100% leak-test inspection",
            "line_priority": "P2 HIGH ($44/min stoppage)",
        },
    }
    info = profiles.get(
        clean_id,
        {
            "active_batch_id": "LOT-2026-STD-001",
            "product_stage": "General Production",
            "contamination_risk": "MEDIUM",
            "qa_batch_hold_action": "Hold active lot until QA Line Clearance.",
            "line_priority": "P2 HIGH",
        },
    )
    out = {"machine_id": clean_id, "assessed_at_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"), **info}
    ctx = get_active_run_context()
    if ctx is not None:
        ctx.batch_risk_report = out
    return out


def create_purchase_requisition_impl(part_code: str, ticket_id: str) -> Dict[str, Any]:
    """Auto-draft a Purchase Requisition when stock_qty <= min_reorder_qty."""
    clean_code = part_code.strip().upper()
    conn = _get_rw_conn()
    try:
        cur = conn.cursor()
        row = cur.execute(
            "SELECT * FROM spare_parts_inventory WHERE UPPER(part_code) = ?",
            (clean_code,),
        ).fetchone()
        if not row:
            return {"created": False, "reason": f"Part '{clean_code}' not found."}

        cur_qty = int(row["stock_qty"])
        min_qty = int(row["min_reorder_qty"])
        if cur_qty > min_qty:
            return {
                "created": False,
                "part_code": row["part_code"],
                "current_stock_qty": cur_qty,
                "min_reorder_qty": min_qty,
                "reason": "Stock remains above reorder threshold.",
            }

        existing = cur.execute(
            "SELECT * FROM purchase_requisitions WHERE ticket_id = ? AND part_code = ?",
            (ticket_id, row["part_code"]),
        ).fetchone()
        if existing:
            return {"created": True, "already_exists": True, **dict(existing)}

        order_qty = max(3, min_qty * 2 - cur_qty)
        est_cost = round(order_qty * float(row["unit_cost_usd"]), 2)
        now_utc = datetime.now(timezone.utc)
        tkt_tag = ticket_id.split("-")[-1][:4].upper()
        pr_id = f"PR-{now_utc.strftime('%Y%m%d%H%M%S')}-{clean_code.replace('-', '')[:6]}-{tkt_tag}"
        ts = now_utc.strftime("%Y-%m-%dT%H:%M:%SZ")
        cur.execute(
            """
            INSERT INTO purchase_requisitions (
                pr_id, ticket_id, part_code, part_name, current_stock_qty,
                min_reorder_qty, order_qty, est_order_cost_usd, status, created_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                pr_id,
                ticket_id,
                row["part_code"],
                row["part_name"],
                cur_qty,
                min_qty,
                order_qty,
                est_cost,
                "DRAFT_PENDING_PROCUREMENT",
                ts,
            ),
        )
        conn.commit()
    finally:
        conn.close()

    return {
        "created": True,
        "pr_id": pr_id,
        "ticket_id": ticket_id,
        "part_code": row["part_code"],
        "part_name": row["part_name"],
        "current_stock_qty": cur_qty,
        "min_reorder_qty": min_qty,
        "order_qty": order_qty,
        "est_order_cost_usd": est_cost,
        "status": "DRAFT_PENDING_PROCUREMENT",
    }


def run_qa_line_clearance_impl(machine_id: str, ticket_id: str, final_vibration_rms: float) -> Dict[str, Any]:
    """Execute final QA Line Clearance checklist before returning asset to GMP production."""
    clean_id = machine_id.strip().upper()
    vib_ok = float(final_vibration_rms) <= GMP_VIBRATION_LIMIT_RMS
    checks = [
        {"check": "LOTO Padlocks & Tags Removed", "status": "PASS"},
        {"check": "100% Maintenance Tool & Fastener Reconciliation", "status": "PASS"},
        {
            "check": "Sanitary Wipe / NSF H1 Grease Containment" if clean_id != "RP-04" else "CIP Flush & Sterile Seal Leak Check (0 drops/min)",
            "status": "PASS",
        },
        {"check": f"10-Min Thermal & Vibration Run (<= {GMP_VIBRATION_LIMIT_RMS} mm/s)", "status": "PASS" if vib_ok else "FAIL"},
    ]
    approved = all(c["status"] == "PASS" for c in checks)
    out = {
        "ticket_id": ticket_id,
        "machine_id": clean_id,
        "final_vibration_rms": round(float(final_vibration_rms), 2),
        "clearance_approved": approved,
        "line_status_after_clearance": "RELEASED_TO_PRODUCTION" if approved else "QUARANTINED",
        "checklist": checks,
        "cleared_at_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
    }
    ctx = get_active_run_context()
    if ctx is not None:
        ctx.clearance_report = out
    return out


def create_capa_if_repeat_failure_impl(
    machine_id: str, ticket_id: str, failure_mode: str, sop_ref: str
) -> Dict[str, Any]:
    """Auto-generate a 5-Why CAPA record in SQLite if machine has >= 2 monthly incidents."""
    clean_id = machine_id.strip().upper()
    ym_prefix = datetime.now(timezone.utc).strftime("%Y-%m")
    conn = _get_rw_conn()
    try:
        cur = conn.cursor()
        m_row = cur.execute("SELECT machine_name FROM machines WHERE UPPER(machine_id) = ?", (clean_id,)).fetchone()
        machine_name = str(m_row["machine_name"]) if m_row else clean_id

        cnt_row = cur.execute(
            "SELECT COUNT(*) AS cnt FROM incident_analytics WHERE UPPER(machine_id) = ? AND incident_date LIKE ?",
            (clean_id, f"{ym_prefix}%"),
        ).fetchone()
        repeat_count = int(cnt_row["cnt"]) if cnt_row else 1

        if repeat_count < 2:
            return {
                "capa_triggered": False,
                "machine_id": clean_id,
                "repeat_count_30d": repeat_count,
                "reason": "Below repeat-failure CAPA threshold (< 2 incidents in current month).",
            }

        existing = cur.execute("SELECT * FROM capa_records WHERE ticket_id = ?", (ticket_id,)).fetchone()
        if existing:
            return {"capa_triggered": True, "already_exists": True, **dict(existing)}

        now_utc = datetime.now(timezone.utc)
        tkt_tag = ticket_id.split("-")[-1][:4].upper()
        capa_id = f"CAPA-{now_utc.strftime('%Y%m%d%H%M%S')}-{clean_id.replace('-', '')}-{tkt_tag}"
        ts = now_utc.strftime("%Y-%m-%dT%H:%M:%SZ")

        pm_map = {
            "TP-01": ("Shorten turret bearing ultrasound & auto-lube verification from 30 days to 14 days", "Reliability Engineer & Compression Lead"),
            "RP-04": ("Mandate pre-repair NPSHa strainer cleaning + cold laser alignment <= 0.05 mm every 30 days", "Reliability Engineer & Fluid Maintenance Lead"),
            "PK-02": ("Add weekly sonic belt tension (180-200 N) & safety-stock min=3 for BLT-2205", "Packaging Engineering & Inventory Lead"),
        }
        rec_pm, owner = pm_map.get(clean_id, ("Tighten preventive vibration inspection interval by 50%", "Reliability Engineer"))
        five_why = (
            f"1. Alarm tripped on {clean_id} ({failure_mode}) -> 2. Mechanical wear/misalignment exceeded ISO 20816-3 Zone D -> "
            f"3. Repeat occurrence #{repeat_count} within {ym_prefix} -> 4. Existing PM cadence insufficient to catch early Zone B->C drift -> "
            f"5. Systemic gap requires tightened PM + engineering verification per {sop_ref}."
        )
        prev_action = f"Update CMMS PM plan for {machine_name}: {rec_pm}."

        cur.execute(
            """
            INSERT INTO capa_records (
                capa_id, ticket_id, machine_id, machine_name, repeat_count_30d,
                problem_statement, five_why_root_cause, preventive_action,
                recommended_pm_interval, owner_role, status, created_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                capa_id,
                ticket_id,
                clean_id,
                machine_name,
                repeat_count,
                f"Repeat reliability excursion (#{repeat_count} in {ym_prefix}) on {machine_name}: {failure_mode}",
                five_why,
                prev_action,
                rec_pm,
                owner,
                "OPEN_CAPA_REVIEW",
                ts,
            ),
        )
        conn.commit()
    finally:
        conn.close()

    return {
        "capa_triggered": True,
        "capa_id": capa_id,
        "ticket_id": ticket_id,
        "machine_id": clean_id,
        "repeat_count_30d": repeat_count,
        "five_why_root_cause": five_why,
        "preventive_action": prev_action,
        "recommended_pm_interval": rec_pm,
        "owner_role": owner,
        "status": "OPEN_CAPA_REVIEW",
    }


def generate_monthly_excel_report_impl() -> Dict[str, Any]:
    """Create formatted 4-sheet Monthly Quality & Maintenance Report (.xlsx) with openpyxl."""
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    now_utc = datetime.now(timezone.utc)
    ym = now_utc.strftime("%Y_%m")
    out_path = REPORTS_DIR / f"GMP_Monthly_Quality_Maintenance_Report_{ym}.xlsx"

    df_inc = get_monthly_incidents_dataframe()
    kpis = compute_monthly_kpis(df_inc)
    df_audit = get_audit_log_dataframe()
    df_capa = get_capa_dataframe()
    df_prs = get_purchase_requisitions_dataframe()

    wb = openpyxl.Workbook()

    header_fill = PatternFill(start_color="0F172A", end_color="0F172A", fill_type="solid")
    sub_fill = PatternFill(start_color="0284C7", end_color="0284C7", fill_type="solid")
    zebra_fill = PatternFill(start_color="F8FAFC", end_color="F8FAFC", fill_type="solid")
    header_font = Font(name="Calibri", size=11, bold=True, color="FFFFFF")
    title_font = Font(name="Calibri", size=15, bold=True, color="FFFFFF")
    section_font = Font(name="Calibri", size=12, bold=True, color="0F172A")
    bold_font = Font(name="Calibri", size=10, bold=True, color="0F172A")
    thin_border = Border(
        left=Side(style="thin", color="CBD5E1"),
        right=Side(style="thin", color="CBD5E1"),
        top=Side(style="thin", color="CBD5E1"),
        bottom=Side(style="thin", color="CBD5E1"),
    )

    # Sheet 1: KPI Summary
    ws_kpi = wb.active
    ws_kpi.title = "KPI Summary"
    ws_kpi.merge_cells("A1:F1")
    ws_kpi["A1"] = f"GMP Quality & Maintenance Monthly Report ({now_utc.strftime('%B %Y')})"
    ws_kpi["A1"].fill = header_fill
    ws_kpi["A1"].font = title_font
    ws_kpi["A1"].alignment = Alignment(horizontal="left", vertical="center", indent=1)
    ws_kpi.row_dimensions[1].height = 30

    ws_kpi["A2"] = f"Generated UTC: {now_utc.strftime('%Y-%m-%d %H:%M:%S')} | ISO 20816-3 / cGMP Limit: <= {GMP_VIBRATION_LIMIT_RMS} mm/s RMS"
    ws_kpi["A2"].font = Font(name="Calibri", size=10, italic=True, color="475569")

    ws_kpi["A4"] = "Executive Monthly KPI Summary"
    ws_kpi["A4"].font = section_font

    kpi_rows = [
        ("KPI Metric", "Value", "GMP Target / Notes"),
        ("Total Closed Maintenance Incidents", kpis["total_incidents"], "All eBMR tickets signed off"),
        ("Total Monthly Downtime (minutes)", kpis["total_downtime_min"], "Sum of production stoppage"),
        ("Average MTTR (minutes)", kpis["avg_mttr_min"], "Mean Time To Repair"),
        ("First-Time-Fix Rate (%)", f"{kpis['first_time_fix_rate_pct']:.1f}%", "Target >= 85.0%"),
        ("Total Spare Parts Cost ($)", kpis["total_part_cost_usd"], "Currency USD"),
        ("Total Technician Labor Cost ($)", kpis["total_labor_cost_usd"], f"${LABOR_RATE_USD_PER_HR:.2f}/hr cGMP rate"),
        ("Estimated Production Loss ($)", kpis["total_production_loss_usd"], "Downtime x line rate"),
        ("Total Maintenance Financial Impact ($)", kpis["total_cost_usd"], "Parts + Labor + Production Loss"),
        ("Active CAPA Records", int(len(df_capa)), "Repeat-failure 5-Why tracking"),
        ("Purchase Requisitions Drafted", int(len(df_prs)), "Auto-reorder triggers"),
    ]
    for r_idx, row_vals in enumerate(kpi_rows, start=5):
        for c_idx, val in enumerate(row_vals, start=1):
            cell = ws_kpi.cell(row=r_idx, column=c_idx, value=val)
            cell.border = thin_border
            if r_idx == 5:
                cell.fill = sub_fill
                cell.font = header_font
            else:
                if c_idx == 1:
                    cell.font = bold_font
                if c_idx == 2 and isinstance(val, float) and ("Cost" in row_vals[0] or "Loss" in row_vals[0] or "Impact" in row_vals[0]):
                    cell.number_format = "$#,##0.00"

    ws_kpi["A18"] = "Cost & Downtime Breakdown by Equipment"
    ws_kpi["A18"].font = section_font

    m_headers = ["Machine ID", "Machine Name", "Incidents", "Downtime (min)", "Avg MTTR (min)", "Total Impact ($)"]
    for c_idx, h in enumerate(m_headers, start=1):
        c = ws_kpi.cell(row=19, column=c_idx, value=h)
        c.fill = sub_fill
        c.font = header_font
        c.border = thin_border

    if not df_inc.empty:
        grp = (
            df_inc.groupby(["machine_id", "machine_name"], as_index=False)
            .agg(
                incidents=("ticket_id", "count"),
                downtime=("downtime_minutes", "sum"),
                avg_mttr=("downtime_minutes", "mean"),
                total_cost=("total_cost_usd", "sum"),
            )
            .sort_values("total_cost", ascending=False)
        )
        for offset, (_, r) in enumerate(grp.iterrows(), start=20):
            vals = [
                r["machine_id"],
                r["machine_name"],
                int(r["incidents"]),
                int(r["downtime"]),
                round(float(r["avg_mttr"]), 1),
                round(float(r["total_cost"]), 2),
            ]
            for c_idx, val in enumerate(vals, start=1):
                cell = ws_kpi.cell(row=offset, column=c_idx, value=val)
                cell.border = thin_border
                if c_idx == 6:
                    cell.number_format = "$#,##0.00"

    ws_kpi.column_dimensions["A"].width = 40
    ws_kpi.column_dimensions["B"].width = 30
    ws_kpi.column_dimensions["C"].width = 32
    ws_kpi.column_dimensions["D"].width = 18
    ws_kpi.column_dimensions["E"].width = 18
    ws_kpi.column_dimensions["F"].width = 20

    # Sheet 2: Incident Log
    ws_inc = wb.create_sheet(title="Incident Log")
    inc_cols = [
        "ticket_id", "incident_date", "machine_id", "machine_name", "failure_mode", "sop_ref",
        "part_used", "used_alternative", "alarm_vibration_rms", "final_vibration_rms",
        "repair_attempts", "first_time_fix", "downtime_minutes", "part_cost_usd",
        "labor_cost_usd", "production_loss_usd", "total_cost_usd", "damage_severity", "orchestrator_status",
    ]
    severity_fills = {
        "Critical": PatternFill(start_color="FEE2E2", end_color="FEE2E2", fill_type="solid"),
        "High": PatternFill(start_color="FFEDD5", end_color="FFEDD5", fill_type="solid"),
        "Medium": PatternFill(start_color="FEF9C3", end_color="FEF9C3", fill_type="solid"),
        "Low": PatternFill(start_color="DCFCE7", end_color="DCFCE7", fill_type="solid"),
    }

    for c_idx, col_name in enumerate(inc_cols, start=1):
        cell = ws_inc.cell(row=1, column=c_idx, value=col_name.replace("_", " ").title())
        cell.fill = header_fill
        cell.font = header_font
        cell.border = thin_border
    ws_inc.freeze_panes = "A2"

    currency_cols = {"part_cost_usd", "labor_cost_usd", "production_loss_usd", "total_cost_usd"}
    for r_idx, (_, rec) in enumerate(df_inc.iterrows(), start=2):
        sev = str(rec.get("damage_severity", "Low"))
        for c_idx, col_name in enumerate(inc_cols, start=1):
            val = rec[col_name]
            if col_name in ("used_alternative", "first_time_fix"):
                val = "Yes" if int(val) == 1 else "No"
            cell = ws_inc.cell(row=r_idx, column=c_idx, value=val)
            cell.border = thin_border
            if col_name in currency_cols:
                cell.number_format = "$#,##0.00"
            if col_name == "damage_severity" and sev in severity_fills:
                cell.fill = severity_fills[sev]
                cell.font = bold_font
            elif r_idx % 2 == 1:
                cell.fill = zebra_fill

    for c_idx, col_name in enumerate(inc_cols, start=1):
        max_len = max([len(col_name)] + [len(str(v)) for v in df_inc[col_name].tolist()]) if not df_inc.empty else len(col_name)
        ws_inc.column_dimensions[get_column_letter(c_idx)].width = min(max(max_len + 4, 14), 42)

    # Sheet 3: CAPA & Reorders
    ws_cr = wb.create_sheet(title="CAPA & Reorders")
    ws_cr["A1"] = "Corrective & Preventive Action (CAPA) Register"
    ws_cr["A1"].font = section_font
    capa_cols = ["capa_id", "ticket_id", "machine_id", "repeat_count_30d", "problem_statement", "five_why_root_cause", "preventive_action", "status", "created_at"]
    for c_idx, col_name in enumerate(capa_cols, start=1):
        cell = ws_cr.cell(row=2, column=c_idx, value=col_name.replace("_", " ").title())
        cell.fill = header_fill
        cell.font = header_font
        cell.border = thin_border
    for offset, (_, rec) in enumerate(df_capa.iterrows(), start=3):
        for c_idx, col_name in enumerate(capa_cols, start=1):
            cell = ws_cr.cell(row=offset, column=c_idx, value=rec[col_name])
            cell.border = thin_border

    pr_start = max(5, 4 + len(df_capa))
    ws_cr.cell(row=pr_start, column=1, value="Automated Spare-Part Purchase Requisitions").font = section_font
    pr_cols = ["pr_id", "ticket_id", "part_code", "part_name", "current_stock_qty", "min_reorder_qty", "order_qty", "est_order_cost_usd", "status", "created_at"]
    for c_idx, col_name in enumerate(pr_cols, start=1):
        cell = ws_cr.cell(row=pr_start + 1, column=c_idx, value=col_name.replace("_", " ").title())
        cell.fill = sub_fill
        cell.font = header_font
        cell.border = thin_border
    for offset, (_, rec) in enumerate(df_prs.iterrows(), start=pr_start + 2):
        for c_idx, col_name in enumerate(pr_cols, start=1):
            cell = ws_cr.cell(row=offset, column=c_idx, value=rec[col_name])
            cell.border = thin_border
            if col_name == "est_order_cost_usd":
                cell.number_format = "$#,##0.00"

    for c_idx in range(1, 11):
        ws_cr.column_dimensions[get_column_letter(c_idx)].width = 32 if c_idx in (5, 6, 7) else 18

    # Sheet 4: GMP Audit Trail
    ws_aud = wb.create_sheet(title="GMP Audit Trail")
    aud_cols = ["audit_id", "ticket_id", "step_number", "timestamp_utc", "agent_name", "action_summary", "gate_decision", "audit_notes", "record_hash"]
    for c_idx, col_name in enumerate(aud_cols, start=1):
        cell = ws_aud.cell(row=1, column=c_idx, value=col_name.replace("_", " ").title())
        cell.fill = header_fill
        cell.font = header_font
        cell.border = thin_border
    ws_aud.freeze_panes = "A2"

    for r_idx, (_, rec) in enumerate(df_audit.head(200).iterrows(), start=2):
        for c_idx, col_name in enumerate(aud_cols, start=1):
            cell = ws_aud.cell(row=r_idx, column=c_idx, value=rec[col_name])
            cell.border = thin_border

    for c_idx, col_name in enumerate(aud_cols, start=1):
        ws_aud.column_dimensions[get_column_letter(c_idx)].width = 28 if col_name in ("action_summary", "audit_notes") else 18

    wb.save(str(out_path))

    ctx = get_active_run_context()
    if ctx is not None:
        ctx.excel_path = str(out_path)

    return {
        "generated": True,
        "excel_path": str(out_path),
        "filename": out_path.name,
        "incident_count": int(kpis["total_incidents"]),
        "total_cost_usd": kpis["total_cost_usd"],
    }


# ---------------------------------------------------------------------------
# LangChain @tool Wrappers (for Autonomous LLM Tool-Calling Agents)
# ---------------------------------------------------------------------------


@tool
def search_sop_manuals(query: str) -> str:
    """Search local PDF Standard Operating Procedures using hybrid TF-IDF + BM25 RAG.
    Returns top matching SOP sections with exact PDF filename, SOP code, and page number."""
    return json.dumps(search_sop_manuals_impl(query), indent=2)


@tool
def read_sensor_telemetry(machine_id: str) -> str:
    """Read live machine vibration, temperature, dominant frequency, and GMP limit telemetry."""
    return json.dumps(read_sensor_telemetry_impl(machine_id), indent=2)


@tool
def assess_batch_risk(machine_id: str) -> str:
    """Assess active GMP batch contamination risk and downstream line quarantine requirement."""
    return json.dumps(assess_batch_risk_impl(machine_id), indent=2)


@tool
def query_inventory_sql(sql_query: str) -> str:
    """Execute a read-only SELECT SQLite query against data/factory_ops.db (spare_parts_inventory, machines)."""
    return json.dumps(query_inventory_sql_impl(sql_query), indent=2)


@tool
def reserve_spare_part(part_code: str, quantity: int, ticket_id: str) -> str:
    """Reserve spare parts in SQLite spare_parts_inventory and decrement stock_qty atomically."""
    return json.dumps(reserve_spare_part_impl(part_code, quantity, ticket_id), indent=2)


@tool
def check_alternative_parts(part_code: str) -> str:
    """Check whether an out-of-stock OEM spare part has a GMP-qualified in-stock compatible alternative."""
    return json.dumps(check_alternative_parts_impl(part_code), indent=2)


@tool
def create_purchase_requisition(part_code: str, ticket_id: str) -> str:
    """Auto-draft a Purchase Requisition in SQLite when spare part stock is at or below min_reorder_qty."""
    return json.dumps(create_purchase_requisition_impl(part_code, ticket_id), indent=2)


@tool
def execute_part_replacement(machine_id: str, old_part: str, new_part: str, sop_ref: str) -> str:
    """Execute mechanical part replacement or corrective laser shaft realignment per referenced SOP."""
    return json.dumps(execute_part_replacement_impl(machine_id, old_part, new_part, sop_ref), indent=2)


@tool
def run_post_repair_vibration_test(machine_id: str) -> str:
    """Run post-repair tri-axial vibration test and compare measured mm/s RMS to the 2.5 mm/s GMP limit."""
    return json.dumps(run_post_repair_vibration_test_impl(machine_id), indent=2)


@tool
def run_qa_line_clearance(machine_id: str, ticket_id: str, final_vibration_rms: float) -> str:
    """Run post-repair QA Line Clearance checks (LOTO, tool reconciliation, wipe/CIP, stability)."""
    return json.dumps(run_qa_line_clearance_impl(machine_id, ticket_id, final_vibration_rms), indent=2)


@tool
def log_incident_analytics(ticket_data: str) -> str:
    """Compute MTTR, labor, part, and production loss costs and log closed incident to SQLite incident_analytics."""
    return json.dumps(log_incident_analytics_impl(ticket_data), indent=2)


@tool
def create_capa_if_repeat_failure(machine_id: str, ticket_id: str, failure_mode: str, sop_ref: str) -> str:
    """Create a 5-Why CAPA record in SQLite if machine has >= 2 incidents in current month."""
    return json.dumps(create_capa_if_repeat_failure_impl(machine_id, ticket_id, failure_mode, sop_ref), indent=2)


@tool
def generate_monthly_excel_report() -> str:
    """Generate formatted 4-sheet Monthly Quality & Maintenance Report (.xlsx)."""
    return json.dumps(generate_monthly_excel_report_impl(), indent=2)


# ---------------------------------------------------------------------------
# UI Query Helpers (Read-only SQLite -> Pandas DataFrames)
# ---------------------------------------------------------------------------


def get_inventory_dataframe() -> pd.DataFrame:
    conn = _get_ro_conn()
    try:
        return pd.read_sql_query(
            "SELECT part_code, part_name, machine_id, stock_qty, min_reorder_qty, bin_location, unit_cost_usd, compatible_alt_code, notes FROM spare_parts_inventory ORDER BY machine_id, part_code",
            conn,
        )
    finally:
        conn.close()


def get_monthly_incidents_dataframe() -> pd.DataFrame:
    ym_prefix = datetime.now(timezone.utc).strftime("%Y-%m")
    conn = _get_ro_conn()
    try:
        df = pd.read_sql_query(
            "SELECT * FROM incident_analytics WHERE incident_date LIKE ? ORDER BY incident_date ASC, created_at ASC",
            conn,
            params=(f"{ym_prefix}%",),
        )
        if df.empty:
            df = pd.read_sql_query(
                "SELECT * FROM incident_analytics ORDER BY incident_date ASC, created_at ASC",
                conn,
            )
        return df
    finally:
        conn.close()


def get_audit_log_dataframe(ticket_id: Optional[str] = None) -> pd.DataFrame:
    conn = _get_ro_conn()
    try:
        if ticket_id:
            return pd.read_sql_query(
                "SELECT * FROM gmp_audit_log WHERE ticket_id = ? ORDER BY audit_id ASC",
                conn,
                params=(ticket_id,),
            )
        return pd.read_sql_query(
            "SELECT * FROM gmp_audit_log ORDER BY audit_id DESC",
            conn,
        )
    finally:
        conn.close()


def get_capa_dataframe() -> pd.DataFrame:
    conn = _get_ro_conn()
    try:
        return pd.read_sql_query("SELECT * FROM capa_records ORDER BY created_at DESC", conn)
    finally:
        conn.close()


def get_purchase_requisitions_dataframe() -> pd.DataFrame:
    conn = _get_ro_conn()
    try:
        return pd.read_sql_query("SELECT * FROM purchase_requisitions ORDER BY created_at DESC", conn)
    finally:
        conn.close()


def compute_monthly_kpis(df: pd.DataFrame) -> Dict[str, Any]:
    if df.empty:
        return {
            "total_incidents": 0,
            "total_downtime_min": 0,
            "avg_mttr_min": 0.0,
            "first_time_fix_rate_pct": 0.0,
            "total_part_cost_usd": 0.0,
            "total_labor_cost_usd": 0.0,
            "total_production_loss_usd": 0.0,
            "total_cost_usd": 0.0,
        }
    return {
        "total_incidents": int(len(df)),
        "total_downtime_min": int(df["downtime_minutes"].sum()),
        "avg_mttr_min": round(float(df["downtime_minutes"].mean()), 1),
        "first_time_fix_rate_pct": round(float(df["first_time_fix"].mean() * 100.0), 1),
        "total_part_cost_usd": round(float(df["part_cost_usd"].sum()), 2),
        "total_labor_cost_usd": round(float(df["labor_cost_usd"].sum()), 2),
        "total_production_loss_usd": round(float(df["production_loss_usd"].sum()), 2),
        "total_cost_usd": round(float(df["total_cost_usd"].sum()), 2),
    }


def list_sop_metadata() -> List[Dict[str, Any]]:
    ensure_seed_data()
    docs: List[Dict[str, Any]] = []
    for p in sorted(SOPS_DIR.glob("*.pdf")):
        reader = PdfReader(str(p))
        docs.append({"filename": p.name, "pages": len(reader.pages)})
    return docs


_DOC_META_MAP: Dict[str, Dict[str, str]] = {
    "GUIDE-OPS-000": {
        "badge": "OPERATOR GUIDE & PROBLEM ID",
        "title": "Machine Operation Guides & Problem Identification",
        "description": "Pre-shift machine guides (TP-01, RP-04, PK-02), factory problem/solution matrix, ISO 20816-3 zones, and HITL playbook.",
    },
    "SOP-MNT-001": {
        "badge": "MAINTENANCE SOP",
        "title": "TP-01 Tablet Press & PK-02 Blister Vibration SOP",
        "description": "OSHA LOTO rules, turret bearing spalling (BRG-4402), punch binding, and belt fatigue (BLT-2205 / BLT-2205-ALT).",
    },
    "SOP-MNT-002": {
        "badge": "MAINTENANCE SOP",
        "title": "RP-04 Sanitary Pump Seal & Laser Alignment SOP",
        "description": "3-A/EHEDG seal & cavitation repair (SL-8810), CIP/SIP safety, and cold dual-beam laser shaft realignment tolerances.",
    },
    "GMP-QA-010": {
        "badge": "GMP QA PROTOCOL",
        "title": "eBMR Governance, ALCOA+ & Financial Cost Model",
        "description": "Quality gates 1-4, Human-in-the-Loop SQL & repair budget authorization, SHA-256 audit log, and severity matrix.",
    },
}


def estimate_repair_cost(machine_id: str, part_cost_usd: float, attempt_number: int = 1) -> Dict[str, Any]:
    """Compute itemized repair or realignment cost estimate for Human QA approval."""
    clean_id = machine_id.strip().upper()
    conn = _get_ro_conn()
    try:
        row = conn.execute(
            "SELECT machine_name, downtime_cost_per_min FROM machines WHERE UPPER(machine_id) = ?",
            (clean_id,),
        ).fetchone()
    finally:
        conn.close()

    machine_name = str(row["machine_name"]) if row else clean_id
    dt_rate = float(row["downtime_cost_per_min"]) if row else 50.0
    base_downtime_map = {"TP-01": 68, "RP-04": 62, "PK-02": 56}
    base_dt = base_downtime_map.get(clean_id, 60)

    if attempt_number <= 1:
        est_dt = base_dt
        est_labor_hrs = 1.5
        est_labor = round(est_labor_hrs * LABOR_RATE_USD_PER_HR, 2)
        est_prod_loss = round(est_dt * dt_rate, 2)
        est_part = round(float(part_cost_usd), 2)
        est_total = round(est_part + est_labor + est_prod_loss, 2)
        return {
            "machine_id": clean_id,
            "machine_name": machine_name,
            "attempt_number": 1,
            "est_downtime_min": est_dt,
            "downtime_rate_per_min": dt_rate,
            "est_labor_hours": est_labor_hrs,
            "labor_rate_per_hr": LABOR_RATE_USD_PER_HR,
            "part_cost_usd": est_part,
            "est_labor_cost_usd": est_labor,
            "est_production_loss_usd": est_prod_loss,
            "est_total_cost_usd": est_total,
            "incremental_rework_cost_usd": 0.0,
        }

    # Incremental rework cost for precision laser shaft realignment (Attempt >= 2)
    inc_dt = 48
    inc_labor_hrs = 2.0
    inc_labor = round(inc_labor_hrs * LABOR_RATE_USD_PER_HR, 2)
    inc_prod_loss = round(inc_dt * dt_rate, 2)
    inc_total = round(inc_labor + inc_prod_loss, 2)
    cum_dt = base_dt + inc_dt * (attempt_number - 1)
    cum_labor = round((1.5 + inc_labor_hrs * (attempt_number - 1)) * LABOR_RATE_USD_PER_HR, 2)
    cum_prod_loss = round(cum_dt * dt_rate, 2)
    cum_total = round(float(part_cost_usd) + cum_labor + cum_prod_loss, 2)
    return {
        "machine_id": clean_id,
        "machine_name": machine_name,
        "attempt_number": attempt_number,
        "est_downtime_min": cum_dt,
        "downtime_rate_per_min": dt_rate,
        "est_labor_hours": 1.5 + inc_labor_hrs * (attempt_number - 1),
        "labor_rate_per_hr": LABOR_RATE_USD_PER_HR,
        "part_cost_usd": round(float(part_cost_usd), 2),
        "est_labor_cost_usd": cum_labor,
        "est_production_loss_usd": cum_prod_loss,
        "est_total_cost_usd": cum_total,
        "incremental_downtime_min": inc_dt,
        "incremental_labor_cost_usd": inc_labor,
        "incremental_production_loss_usd": inc_prod_loss,
        "incremental_rework_cost_usd": inc_total,
    }


def list_controlled_documents() -> List[Dict[str, Any]]:
    """Return full metadata for the Operator Guide PDF and all 3 SOP PDFs."""
    ensure_seed_data()
    entries: List[Dict[str, Any]] = []
    candidates = sorted(DOCS_DIR.glob("*.pdf")) + sorted(SOPS_DIR.glob("*.pdf"))
    for p in candidates:
        code_match = re.search(r"((?:GUIDE|SOP|GMP)-[A-Z]+-\d+)", p.name)
        doc_code = code_match.group(1) if code_match else p.stem
        meta = _DOC_META_MAP.get(
            doc_code,
            {
                "badge": "CONTROLLED PDF",
                "title": p.stem.replace("_", " "),
                "description": "Controlled GMP operational document.",
            },
        )
        reader = PdfReader(str(p))
        entries.append(
            {
                "doc_code": doc_code,
                "badge": meta["badge"],
                "title": meta["title"],
                "description": meta["description"],
                "filename": p.name,
                "path": str(p),
                "pages": len(reader.pages),
            }
        )
    return entries


def find_document_by_code_or_name(ref: str) -> Optional[Dict[str, Any]]:
    """Find controlled document metadata by doc_code (e.g., SOP-MNT-001) or filename."""
    clean = (ref or "").strip().upper()
    docs = list_controlled_documents()
    for d in docs:
        if d["doc_code"].upper() == clean or d["filename"].upper() == clean or clean in d["filename"].upper():
            return d
    return docs[0] if docs else None


def render_pdf_page_png(pdf_path: str, page_number: int = 1, scale: float = 1.65) -> bytes:
    """Render a 1-indexed PDF page to crisp PNG bytes via pypdfium2."""
    doc = pdfium.PdfDocument(str(pdf_path))
    try:
        idx = max(0, min(int(page_number) - 1, len(doc) - 1))
        page = doc[idx]
        pil_img = page.render(scale=scale).to_pil()
        buf = io.BytesIO()
        pil_img.save(buf, format="PNG")
        return buf.getvalue()
    finally:
        doc.close()


def extract_pdf_page_text(pdf_path: str, page_number: int = 1) -> str:
    """Extract readable text for a 1-indexed PDF page via pypdf."""
    reader = PdfReader(str(pdf_path))
    if not reader.pages:
        return ""
    idx = max(0, min(int(page_number) - 1, len(reader.pages) - 1))
    return (reader.pages[idx].extract_text() or "").strip()


def build_incident_certificate_pdf(state: Dict[str, Any]) -> bytes:
    """Build 1-page printable Executive GMP Incident Closure Certificate PDF bytes from MaintenanceState."""
    from src.seed_data import generate_incident_certificate_pdf_bytes

    return generate_incident_certificate_pdf_bytes(state)
