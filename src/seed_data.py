"""Auto-generated realistic pharma GMP assets for the Quality Control Tower.

Generates:
1. Publication-grade 3-page PDF Standard Operating Procedures (data/sops/*.pdf)
   and Operator Machine & Problem-ID Guide (data/docs/*.pdf) via fpdf2 tables & callouts.
2. SQLite operational database (data/factory_ops.db) grounded in ISO 20816-3
   Group 2 rigid thresholds, EU GMP Annex 1/11/15, and realistic pharma maintenance costs.
"""

from __future__ import annotations

import hashlib
import sqlite3
import sys
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

from fpdf import FPDF
from fpdf.enums import XPos, YPos
from fpdf.fonts import FontFace

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
SOPS_DIR = DATA_DIR / "sops"
DOCS_DIR = DATA_DIR / "docs"
REPORTS_DIR = DATA_DIR / "reports"
DB_PATH = DATA_DIR / "factory_ops.db"

GMP_VIBRATION_LIMIT_RMS = 2.5
LABOR_RATE_USD_PER_HR = 95.0


def _ascii(text: str) -> str:
    """Normalize text to strict Latin-1 safe characters for fpdf2 built-in fonts."""
    replacements = {
        "\u2014": "--",
        "\u2013": "-",
        "\u2264": "<=",
        "\u2265": ">=",
        "\u00b0": " deg ",
        "\u00b5": "u",
        "\u2022": "-",
        "\u2018": "'",
        "\u2019": "'",
        "\u201c": '"',
        "\u201d": '"',
    }
    out = text
    for k, v in replacements.items():
        out = out.replace(k, v)
    return out.encode("latin-1", errors="replace").decode("latin-1")


class _GMPDocumentPdf(FPDF):
    """Publication-grade cGMP SOP, Engineering Guide & Executive Report PDF generator."""

    def __init__(self, doc_code: str, doc_title: str, revision: str, category: str) -> None:
        super().__init__(orientation="P", unit="mm", format="A4")
        self.doc_code = _ascii(doc_code)
        self.doc_title = _ascii(doc_title)
        self.revision = _ascii(revision)
        self.category = _ascii(category)
        self.set_margins(12, 16, 12)
        self.set_auto_page_break(auto=True, margin=15)

    def header(self) -> None:
        self.set_fill_color(15, 23, 42)
        self.rect(12, 6, 186, 8, style="F")
        self.set_xy(14, 7.2)
        self.set_font("Helvetica", "B", 8.5)
        self.set_text_color(248, 250, 252)
        self.cell(125, 5.5, _ascii(f"PHARMA-GMP PLANT 04  |  {self.doc_code} ({self.revision})"))
        self.set_text_color(56, 189, 248)
        self.cell(57, 5.5, _ascii(self.category), align="R")
        self.set_xy(12, 16)

    def footer(self) -> None:
        self.set_y(-12)
        self.set_draw_color(203, 213, 225)
        self.line(12, self.get_y(), 198, self.get_y())
        self.set_font("Helvetica", "I", 7.6)
        self.set_text_color(100, 116, 139)
        self.cell(
            0,
            6,
            _ascii(f"CONTROLLED eBMR COPY - {self.doc_code}: {self.doc_title}   |   Page {self.page_no()} of {{nb}}"),
            align="C",
        )

    def add_banner_title(self, title: str, subtitle: str) -> None:
        self.set_font("Helvetica", "B", 13.8)
        self.set_text_color(15, 23, 42)
        self.cell(0, 6.8, _ascii(title), new_x=XPos.LMARGIN, new_y=YPos.NEXT)
        self.set_font("Helvetica", "", 9.1)
        self.set_text_color(71, 85, 105)
        self.multi_cell(0, 4.5, _ascii(subtitle), new_x=XPos.LMARGIN, new_y=YPos.NEXT)
        self.ln(2.2)

    def add_section_heading(self, title: str) -> None:
        self.ln(1.8)
        y0 = self.get_y()
        self.set_fill_color(241, 245, 249)
        self.rect(12, y0, 186, 6.6, style="F")
        self.set_fill_color(2, 132, 199)
        self.rect(12, y0, 2.2, 6.6, style="F")
        self.set_xy(16.0, y0 + 0.6)
        self.set_font("Helvetica", "B", 10.6)
        self.set_text_color(15, 23, 42)
        self.cell(180, 5.5, _ascii(title), new_x=XPos.LMARGIN, new_y=YPos.NEXT)
        self.set_y(y0 + 8.2)

    def add_paragraph(self, text: str) -> None:
        self.set_font("Helvetica", "", 9.2)
        self.set_text_color(30, 41, 59)
        self.multi_cell(0, 4.7, _ascii(text), new_x=XPos.LMARGIN, new_y=YPos.NEXT)
        self.ln(1.8)

    def add_doc_control_grid(self, rows: List[Tuple[str, str, str, str]]) -> None:
        self.set_font("Helvetica", "", 8.4)
        self.set_draw_color(203, 213, 225)
        with self.table(
            col_widths=(32, 61, 32, 61),
            first_row_as_headings=False,
            line_height=4.5,
            padding=1.8,
        ) as tbl:
            key_style = FontFace(emphasis="BOLD", color=(15, 23, 42), fill_color=(241, 245, 249))
            val_style = FontFace(color=(30, 41, 59), fill_color=(255, 255, 255))
            for k1, v1, k2, v2 in rows:
                r = tbl.row()
                r.cell(_ascii(k1), style=key_style)
                r.cell(_ascii(v1), style=val_style)
                r.cell(_ascii(k2), style=key_style)
                r.cell(_ascii(v2), style=val_style)
        self.ln(2.5)

    def add_callout(self, kind: str, title: str, body: str) -> None:
        palettes = {
            "danger": ((254, 242, 242), (220, 38, 38), (153, 27, 27)),
            "warning": ((255, 251, 235), (217, 119, 6), (146, 64, 14)),
            "info": ((240, 249, 255), (2, 132, 199), (12, 74, 110)),
        }
        bg, accent, title_col = palettes.get(kind, palettes["info"])
        start_y = self.get_y()
        self.set_font("Helvetica", "", 8.7)
        lines = self.multi_cell(178, 4.4, _ascii(body), dry_run=True, output="LINES")
        box_h = 9.2 + len(lines) * 4.4
        self.set_fill_color(*bg)
        self.set_draw_color(226, 232, 240)
        self.rect(12, start_y, 186, box_h, style="DF")
        self.set_fill_color(*accent)
        self.rect(12, start_y, 2.6, box_h, style="F")

        self.set_xy(16.5, start_y + 1.9)
        self.set_font("Helvetica", "B", 9.1)
        self.set_text_color(*title_col)
        self.cell(0, 4.6, _ascii(title), new_x=XPos.LMARGIN, new_y=YPos.NEXT)
        self.set_x(16.5)
        self.set_font("Helvetica", "", 8.7)
        self.set_text_color(30, 41, 59)
        self.multi_cell(178, 4.4, _ascii(body), new_x=XPos.LMARGIN, new_y=YPos.NEXT)
        self.set_y(start_y + box_h + 2.6)

    def add_styled_table(
        self,
        headers: Sequence[str],
        rows: Sequence[Sequence[str]],
        col_widths: Sequence[float],
        row_fills: Optional[Sequence[Tuple[int, int, int]]] = None,
    ) -> None:
        self.set_font("Helvetica", "", 8.2)
        self.set_draw_color(203, 213, 225)
        self.set_fill_color(255, 255, 255)
        self.set_text_color(30, 41, 59)
        head_style = FontFace(emphasis="BOLD", color=(255, 255, 255), fill_color=(15, 23, 42))
        with self.table(
            col_widths=tuple(col_widths),
            headings_style=head_style,
            text_align="LEFT",
            cell_fill_mode="NONE",
            line_height=4.2,
            padding=1.8,
        ) as tbl:
            hr = tbl.row()
            for h in headers:
                hr.cell(_ascii(h))
            for idx, rdata in enumerate(rows):
                r = tbl.row()
                fill_col = row_fills[idx] if row_fills else ((248, 250, 252) if idx % 2 == 1 else (255, 255, 255))
                r_style = FontFace(color=(30, 41, 59), fill_color=fill_col)
                for val in rdata:
                    r.cell(_ascii(str(val)), style=r_style)
        self.set_fill_color(255, 255, 255)
        self.set_text_color(30, 41, 59)
        self.ln(2.4)

    def add_checklist(self, items: Sequence[Tuple[str, str]]) -> None:
        for idx, (lead, detail) in enumerate(items, start=1):
            self.set_x(12)
            self.set_font("Helvetica", "B", 8.8)
            self.set_text_color(2, 132, 199)
            self.cell(6.5, 4.6, f"{idx}.")
            self.set_font("Helvetica", "B", 8.8)
            self.set_text_color(15, 23, 42)
            self.multi_cell(179.5, 4.6, _ascii(f"{lead}: {detail}"), new_x=XPos.LMARGIN, new_y=YPos.NEXT)
            self.ln(1.0)
        self.ln(1.5)

    def add_kpi_strip(self, items: Sequence[Tuple[str, str, str]]) -> None:
        """Render up to 4 executive metric tiles evenly across the 186mm page width."""
        n = max(1, min(len(items), 4))
        gap = 2.66
        card_w = (186.0 - gap * (n - 1)) / n
        card_h = 18.5
        y0 = self.get_y()
        for idx, (lbl, val, sub) in enumerate(items[:n]):
            x0 = 12.0 + idx * (card_w + gap)
            self.set_fill_color(248, 250, 252)
            self.set_draw_color(203, 213, 225)
            self.rect(x0, y0, card_w, card_h, style="DF")
            self.set_fill_color(2, 132, 199)
            self.rect(x0, y0, card_w, 1.8, style="F")
            self.set_xy(x0 + 2.2, y0 + 2.8)
            self.set_font("Helvetica", "B", 7.1)
            self.set_text_color(100, 116, 139)
            self.cell(card_w - 4.4, 3.6, _ascii(lbl.upper())[:28])
            self.set_xy(x0 + 2.2, y0 + 6.8)
            self.set_font("Helvetica", "B", 10.6)
            self.set_text_color(15, 23, 42)
            self.cell(card_w - 4.4, 5.2, _ascii(val)[:22])
            self.set_xy(x0 + 2.2, y0 + 12.6)
            self.set_font("Helvetica", "", 7.2)
            self.set_text_color(71, 85, 105)
            self.cell(card_w - 4.4, 4.0, _ascii(sub)[:28])
        self.set_y(y0 + card_h + 2.8)

    def add_bar_chart(
        self,
        title: str,
        bars: Sequence[Tuple[str, float, Tuple[int, int, int]]],
        max_val: float,
        limit_val: Optional[float] = None,
        unit: str = "mm/s",
    ) -> None:
        """Render crisp horizontal vector bar comparison chart inside bordered card."""
        row_h = 7.4
        pad_top = 8.0
        box_h = pad_top + len(bars) * row_h + 5.5
        y0 = self.get_y()
        self.set_fill_color(248, 250, 252)
        self.set_draw_color(203, 213, 225)
        self.rect(12, y0, 186, box_h, style="DF")

        self.set_xy(15, y0 + 1.8)
        self.set_font("Helvetica", "B", 8.8)
        self.set_text_color(15, 23, 42)
        self.cell(140, 4.5, _ascii(title))
        if limit_val is not None:
            self.set_font("Helvetica", "B", 7.8)
            self.set_text_color(185, 28, 28)
            self.cell(41, 4.5, _ascii(f"Limit: {limit_val:.1f} {unit}"), align="R")

        track_x = 62.0
        track_w = 100.0
        bars_top_y = y0 + pad_top
        for i, (lbl, val, col) in enumerate(bars):
            ry = bars_top_y + i * row_h
            self.set_xy(15, ry + 0.5)
            self.set_font("Helvetica", "", 8.0)
            self.set_text_color(30, 41, 59)
            self.cell(45, 4.5, _ascii(lbl)[:28])

            self.set_fill_color(226, 232, 240)
            self.rect(track_x, ry + 1.0, track_w, 4.4, style="F")

            frac = max(0.02, min(1.0, float(val) / max(0.1, float(max_val))))
            self.set_fill_color(*col)
            self.rect(track_x, ry + 1.0, track_w * frac, 4.4, style="F")

            self.set_xy(track_x + track_w + 2.5, ry + 0.6)
            self.set_font("Helvetica", "B", 8.0)
            self.set_text_color(15, 23, 42)
            self.cell(31, 4.5, _ascii(f"{float(val):.2f} {unit}"))

        if limit_val is not None:
            lim_x = track_x + track_w * max(0.02, min(1.0, float(limit_val) / max(0.1, float(max_val))))
            self.set_draw_color(220, 38, 38)
            self.set_line_width(0.5)
            self.line(lim_x, bars_top_y - 0.5, lim_x, bars_top_y + len(bars) * row_h)
            self.set_line_width(0.2)

        self.set_y(y0 + box_h + 2.6)

    def add_two_col_cards(
        self,
        left_title: str,
        left_lines: Sequence[str],
        right_title: str,
        right_lines: Sequence[str],
    ) -> None:
        """Render two side-by-side equal-height specification cards (91mm + 4mm gap + 91mm)."""
        self.set_font("Helvetica", "", 8.1)
        left_wrapped: List[List[str]] = []
        right_wrapped: List[List[str]] = []
        for ln in left_lines:
            left_wrapped.append(self.multi_cell(85, 4.1, _ascii(f"- {ln}"), dry_run=True, output="LINES"))
        for ln in right_lines:
            right_wrapped.append(self.multi_cell(85, 4.1, _ascii(f"- {ln}"), dry_run=True, output="LINES"))
        n_l = sum(len(x) for x in left_wrapped) + 0.4 * max(0, len(left_wrapped) - 1)
        n_r = sum(len(x) for x in right_wrapped) + 0.4 * max(0, len(right_wrapped) - 1)
        box_h = 9.0 + max(n_l, n_r, 3) * 4.1 + 2.2
        y0 = self.get_y()

        for x0, title, wrapped in ((12.0, left_title, left_lines), (107.0, right_title, right_lines)):
            self.set_fill_color(248, 250, 252)
            self.set_draw_color(203, 213, 225)
            self.rect(x0, y0, 91.0, box_h, style="DF")
            self.set_fill_color(30, 41, 59)
            self.rect(x0, y0, 91.0, 6.2, style="F")
            self.set_xy(x0 + 2.5, y0 + 1.0)
            self.set_font("Helvetica", "B", 8.4)
            self.set_text_color(248, 250, 252)
            self.cell(86, 4.4, _ascii(title)[:48])

            self.set_xy(x0 + 2.5, y0 + 7.6)
            self.set_font("Helvetica", "", 8.1)
            self.set_text_color(30, 41, 59)
            for item in wrapped:
                self.set_x(x0 + 2.5)
                self.multi_cell(85.5, 4.1, _ascii(f"- {item}"), align="L", new_x=XPos.LMARGIN, new_y=YPos.NEXT)
                self.set_y(self.get_y() + 0.8)

        self.set_y(y0 + box_h + 2.6)

    def add_signoff_block(self, owner_role: str) -> None:
        self.add_section_heading("Electronic QA & Engineering Approval Record (21 CFR Part 11)")
        self.add_styled_table(
            headers=("Role", "Authorizing Entity", "Verification Gate", "Signature State"),
            rows=[
                ("Engineering Owner", owner_role, "Technical tolerances & SOP steps verified", "ELECTRONICALLY SIGNED"),
                ("Quality Assurance", "GMP Quality Supervisor", "EU GMP Annex 11 / ALCOA+ check passed", "APPROVED FOR USE"),
                ("Human-in-the-Loop", "Plant QA Lead", "Required for SQL mutations & repair budgets", "ENFORCED IN UI"),
            ],
            col_widths=(36, 48, 66, 36),
        )


def _iso_20816_table(pdf: _GMPDocumentPdf) -> None:
    pdf.add_styled_table(
        headers=("Zone", "Group 2 Rigid RMS", "Operational Interpretation", "Control Tower Action"),
        rows=[
            ("Zone A (Good)", "< 1.4 mm/s RMS", "Newly commissioned / freshly aligned equipment", "Release to batch production"),
            ("Zone B (Acceptable)", "1.4 - 2.8 mm/s (GMP <= 2.5)", "Long-term unrestricted cGMP service <= 2.5 mm/s", "Gate 3 PASS if <= 2.5 mm/s"),
            ("Zone C (Alert)", "> 2.5 - 4.5 mm/s RMS", "Unsatisfactory; coupling/bearing misalignment", "Gate 3 REJECTED -> Realignment"),
            ("Zone D (Danger)", "> 4.5 mm/s RMS", "Severe damage risk (bearing spall / seal rupture)", "Critical Alarm -> Immediate LOTO"),
        ],
        col_widths=(30, 44, 62, 50),
        row_fills=[
            (236, 253, 245),
            (239, 246, 255),
            (255, 251, 235),
            (254, 242, 242),
        ],
    )


def _build_pdf_file(path: Path, doc_code: str, doc_title: str, revision: str, category: str, builder_fn: Any) -> None:
    pdf = _GMPDocumentPdf(doc_code=doc_code, doc_title=doc_title, revision=revision, category=category)
    pdf.alias_nb_pages()
    builder_fn(pdf)
    path.parent.mkdir(parents=True, exist_ok=True)
    pdf.output(str(path))


def generate_sop_pdfs(force: bool = False) -> List[Path]:
    """Generate 3 publication-grade, densely formatted 6-page cGMP SOP PDFs in data/sops/."""
    SOPS_DIR.mkdir(parents=True, exist_ok=True)

    # 1. SOP-MNT-001 (6 dense pages)
    p1 = SOPS_DIR / "SOP-MNT-001_TabletPress_Vibration_Troubleshooting.pdf"
    if force or not p1.exists():
        def _build_sop1(pdf: _GMPDocumentPdf) -> None:
            # Page 1: Standards, LOTO, Multi-Sensor Envelope & Calibrated Test Instruments
            pdf.add_page()
            pdf.add_banner_title(
                "SOP-MNT-001: Solid-Dose Tablet Press & Blister Packaging Vibration SOP",
                "Preventive diagnostics, 6 fault isolation procedures, and ISO 20816-3 release gates for TP-01 and PK-02.",
            )
            pdf.add_doc_control_grid([
                ("Document ID", "SOP-MNT-001", "Revision", "Rev 6.0 (Effective 2026-03-15)"),
                ("Assets Covered", "TP-01 Tablet Press & PK-02 Blister", "Standard", "ISO 20816-3 Group 2 Rigid / cGMP"),
                ("GMP Limit", "<= 2.5 mm/s RMS post-repair", "Stoppage Rates", "TP-01: $68/min | PK-02: $44/min"),
                ("Cleanroom Area", "Suites A & C (ISO Class 8)", "Data Record", "21 CFR Part 11 / eBMR SHA-256"),
            ])
            pdf.add_callout(
                "danger",
                "MANDATORY OSHA 29 CFR 1910.147 LOCKOUT / TAGOUT (LOTO)",
                "Before opening turret guards, removing punch retainers, adjusting cam tracks, or touching drive belts on TP-01 or PK-02: "
                "isolate 480V main disconnect, apply personal safety padlock and eBMR tag, verify 0V zero-energy state on VFD terminals, "
                "bleed pneumatic accumulators to 0.0 bar, and log LOTO badge ID before physical tool entry.",
            )
            pdf.add_section_heading("1. ISO 20816-3 Multi-Axis Vibration & Spectral Acceptance Criteria")
            pdf.add_paragraph(
                "Continuous tri-axial piezoelectric accelerometers (10-1000 Hz velocity + 10 kHz enveloping) monitor Radial-X, Radial-Y, "
                "and Axial-Z vectors plus Kurtosis (bearing impulsiveness) and Peak-g. While ISO 20816-3 Zone B extends to 2.8 mm/s, "
                "Plant 04 enforces a strict <= 2.5 mm/s RMS cGMP release threshold to protect tablet weight uniformity and foil seal integrity."
            )
            _iso_20816_table(pdf)
            pdf.add_two_col_cards(
                "Pre-Service Cleanroom & Safety Readiness",
                [
                    "Record active lot ID & quarantine WIP drums under EU GMP Annex 15.",
                    "Don Grade D cleanroom gowning, nitrile gloves, and safety shield.",
                    "Confirm zero-energy 0.0 VAC across L1/L2/L3 contactor load lugs.",
                    "Stage calibrated torque wrench & cleanroom 70% IPA lint-free wipes.",
                ],
                "Mandatory Diagnostic Instrumentation",
                [
                    "Fluke 805 / SKF CMXA 48 tri-axial vibration analyzer (NIST traceable).",
                    "Dual-beam laser pulley & coupling analyzer (< 0.02 mm resolution).",
                    "Gates 508C sonic belt tension meter (180-200 N span repeatability).",
                    "Fluke 568 IR surface pyrometer & 0.02 mm dial runout indicator.",
                ],
            )
            pdf.add_styled_table(
                headers=("Sensor Axis / Channel", "Healthy Baseline", "Warning Threshold", "Critical Alarm / Physical Fault Meaning"),
                rows=[
                    ("Radial-X / Radial-Y", "0.9 - 1.3 mm/s", "> 2.5 mm/s", "Outer-race bearing spall, unbalance, or belt whip"),
                    ("Axial-Z Vector", "0.5 - 0.8 mm/s", "> 2.2 mm/s", "Parallel/angular shaft coupling offset or cam track misalignment"),
                    ("Kurtosis / Peak-g", "2.8 - 3.3 | < 0.7g", "> 4.2 | > 2.0g", "Subsurface metal impacts (cam knock or roller flat spot)"),
                ],
                col_widths=(40, 34, 34, 78),
            )

            # Page 2: TP-01 Main Turret Bearing Spalling (BRG-4402)
            pdf.add_page()
            pdf.add_section_heading("2. TP-01 Fault Mode 1: Main Turret Bearing Spalling (BRG-4402)")
            pdf.add_paragraph(
                "Main turret spherical roller bearing BRG-4402 (SKF Explorer 22214 E) supports axial compaction loads up to 60 kN at 45 RPM. "
                "Lubrication film breakdown or extended overload causes outer-race fatigue spalling characterized by high-frequency BPFO impacts."
            )
            pdf.add_bar_chart(
                "TP-01 Turret Bearing Spall Vibration Profile vs cGMP Limit",
                [
                    ("Nominal Baseline (45 RPM)", 1.12, (56, 189, 248)),
                    ("Fault Spike (184 Hz BPFO)", 9.80, (239, 68, 68)),
                    ("Verified Post-Repair (BRG-4402)", 1.30, (16, 185, 129)),
                ],
                max_val=11.0,
                limit_val=2.5,
            )
            pdf.add_styled_table(
                headers=("Parameter", "Alarm Signature (BRG-4402 Spall)", "Root Cause Mechanism", "GMP Target Post-Repair"),
                rows=[
                    ("Tri-Axial RMS Velocity", "8.0 - 12.0 mm/s RMS (typ. 9.8)", "Subsurface raceway spalling & metal flaking", "<= 2.5 mm/s RMS (typ. 1.3)"),
                    ("Dominant Frequency", "184.2 Hz BPFO harmonics + sidebands", "Roller passage over outer-race spall crater", "< 0.4 mm/s at 184.2 Hz"),
                    ("Kurtosis & Peak-g", "Kurtosis 6.2-7.5 | Peak 3.8-4.9 g", "Periodic micro-impacts spike high crest factor", "Kurtosis ~3.0 | Peak < 0.9 g"),
                    ("Temperature & Current", "68-78 deg C | Motor 22.8 A (+18%)", "Coulomb friction & grease thermal degradation", "< 52 deg C | 18.5-19.5 A"),
                ],
                col_widths=(36, 48, 56, 46),
            )
            pdf.add_section_heading("2.1 Step-by-Step Replacement Procedure: BRG-4402 / BRG-4402-ALT")
            pdf.add_checklist([
                ("Extraction & Journal Inspection", "Withdraw worn BRG-4402 using 10-ton hydraulic puller; inspect journal runout (< 0.03 mm TIR) and wipe with 70% IPA."),
                ("Controlled Induction Shrink-Fit", "Heat replacement BRG-4402 (or qualified BRG-4402-ALT) on demagnetizing induction heater to max 110 deg C; seat flush."),
                ("Star Torque & NSF H1 Lubrication", "Torque housing cap bolts to 45 Nm in a diagonal star pattern; pack cavity with 35 g food-grade NSF H1 synthetic grease."),
                ("Dial Runout & Post-Repair Gate", "Verify turret axial/radial runout <= 0.03 mm; remove LOTO; execute 10-min 45 RPM run confirming <= 2.5 mm/s RMS."),
            ])
            pdf.add_two_col_cards(
                "SKF 22214 E / FAG Alternative Engineering Specs",
                [
                    "Dynamic Load C = 224 kN | Static C0 = 240 kN | Bore = 70 mm.",
                    "Radial internal clearance CN (0.050-0.085 mm pre-mount feeler check).",
                    "Maximum induction heater temperature: 110 deg C (demagnetized < 2 A/cm).",
                    "Housing cap bolts: M10 Class 10.9 stainless, torque 45 Nm star sequence.",
                ],
                "FMEA & Product Quality Protection (RPN 216)",
                [
                    "Unchecked raceway spalling sheds metallic fines near compression zone.",
                    "Turbine micro-shudder causes tablet weight RSD > 2.0% and lamination.",
                    "BRG-4402-ALT (FAG X-life) carries equal load rating under Annex 15.",
                    "Post-repair vibration > 2.5 mm/s blocks QA Line Clearance automatically.",
                ],
            )
            pdf.add_callout(
                "info",
                "QUALIFIED ALTERNATIVE RULE (BRG-4402-ALT)",
                "If primary BRG-4402 inventory is depleted, Stock Agent may propose GMP-qualified BRG-4402-ALT ($455) subject to "
                "Human QA SQL_MUTATION approval and EN 10204 3.1 batch traceability logging.",
            )

            # Page 3: TP-01 Punch Binding & Auto-Lube Starvation (LUB-3010)
            pdf.add_page()
            pdf.add_section_heading("3. TP-01 Fault Mode 2: Punch Binding & Auto-Lube Starvation (LUB-3010)")
            pdf.add_paragraph(
                "Clogged automatic turret mist nozzles or depleted food-grade lubricant (LUB-3010) permit excipient granulation fines to "
                "bake into upper and lower punch guide bores, causing erratic turret drag, motor current surges, and weight-variation rejects."
            )
            pdf.add_kpi_strip([
                ("Alarm RMS", "6.4 mm/s", "45 Hz 1x turret drag"),
                ("VFD Torque Surge", "28.4 A", "+45% above 19.2 A norm"),
                ("Mist Supply", "4.2 bar", "0.4 mL/min NSF H1 oil"),
                ("GMP Release Target", "<= 1.4 mm/s", "19.0 A & < 50 deg C"),
            ])
            pdf.add_styled_table(
                headers=("Sensor Metric", "Alarm Envelope (Punch Binding)", "Mechanical Effect", "Verified Corrective Spec"),
                rows=[
                    ("Tri-Axial RMS", "5.5 - 7.8 mm/s RMS (typ. 6.4)", "Stick-slip torsional shudder along cam track", "<= 2.5 mm/s RMS (typ. 1.4)"),
                    ("Spectrum & Kurtosis", "45.0 Hz 1x drag | Kurtosis 4.0-4.6", "Non-sinusoidal punch release jerk peaks", "Kurtosis <= 3.1 | Smooth 1x"),
                    ("Motor Current & Temp", "28.4 A (+45% surge) | 61-66 deg C", "Approaching VFD over-torque trip / bore galling", "19.0 A nominal | < 50 deg C"),
                    ("Spare Part & Media", "LUB-3010 Spray Manifold Kit ($145)", "Restore 4.2 bar atomized NSF H1 film", "0.4 mL/min pulse at all heads"),
                ],
                col_widths=(36, 48, 56, 46),
            )
            pdf.add_section_heading("3.1 Bore Cleaning & Auto-Lube (LUB-3010) Restoration Checklist")
            pdf.add_checklist([
                ("Turret Hand-Crank Torque Audit", "Lock out main drive; rotate turret manually via calibrated torque bar; flag all binding stations exceeding 18 Nm drag."),
                ("Punch Extraction & Roughness Check", "Extract flagged punches; inspect barrel surface finish (quarantine if Ra > 0.4 um or heat tint visible)."),
                ("70% IPA Die Bore Scouring", "Swab upper and lower punch guide bores using non-marring brass/nylon brushes dampened with USP-grade 70% IPA."),
                ("Replace LUB-3010 Spray Manifold", "Install new LUB-3010 nozzle block, purge air traps for 15 sec, and confirm 4.2 bar conical mist coverage."),
                ("Dry Crank Amperage & RMS Check", "Verify tablet ejection force < 120 N, main VFD current <= 19.5 A, and RMS <= 2.5 mm/s before releasing to production."),
            ])
            pdf.add_two_col_cards(
                "Pneumatic Mist & Punch Clearance Tolerances",
                [
                    "Upper/lower punch guide bore diametric clearance: 0.025 - 0.045 mm.",
                    "Auto-lube manifold supply air pressure: 4.0 - 4.4 bar regulated.",
                    "Manifold M6 mounting studs: 12 Nm; push-to-connect 6mm PTFE tubing.",
                ],
                "Granulation Dust & Stick-Slip Root Cause Control",
                [
                    "Inspect dust-extractor vacuum DP (> 1200 Pa) at turret scraper blades.",
                    "Replace worn scraper felt seals if powder bypass exceeds 2 g/hr.",
                    "Record LUB-3010 lot number & NSF H1 certificate in eBMR Step 2.",
                ],
            )
            pdf.add_callout(
                "warning",
                "DO NOT USE MINERAL OIL OR SILICONE SPRAYS ON TABLET PUNCH BORES",
                "Only NSF H1 food-grade synthetic oil dispensed by LUB-3010 is approved for product-contact proximity on TP-01.",
            )

            # Page 4: TP-01 Pre-Compression Cam & Roller Knock (RLR-4450)
            pdf.add_page()
            pdf.add_section_heading("4. TP-01 Fault Mode 3: Pre-Compression Cam & Roller Knock (RLR-4450)")
            pdf.add_paragraph(
                "Pre-compression pressure rollers (RLR-4450) absorb cyclic punch-head impact twice per turret revolution (90.0 Hz). "
                "Pin-needle brinelling or flat-spotted outer rings generate severe transient acceleration spikes (Peak > 5.0 g) and punch-tip chipping."
            )
            pdf.add_styled_table(
                headers=("Diagnostic Metric", "Cam / Roller Knock Fault", "Stock & Auto-PO Rule", "Post-Repair Release Limit"),
                rows=[
                    ("Tri-Axial RMS / Peak", "7.0 - 9.6 mm/s RMS (8.2) | 5.2 g", "Reserve OEM RLR-4450 ($640) or Alt RLR-4450-ALT", "<= 2.5 mm/s RMS (typ. 1.4)"),
                    ("Spectrum / Kurtosis", "90.0 Hz (2x cam hit) | Kurtosis 7.4", "LOW STOCK ALERT: if qty <= min_reorder (2), trigger PO", "Peak < 1.1 g | Kurtosis 3.0"),
                    ("Roller Radial Clearance", "> 0.25 mm worn needle play", "Adjust eccentric carrier to 0.08-0.12 mm clearance", "Torque yoke cap to 52 Nm"),
                    ("Drive Load & Temp", "25.1 A (+31%) | 68.0 deg C pin", "Check bronze carrier bushings for ovality (< 0.03 mm)", "< 55 deg C | 19.2 A"),
                ],
                col_widths=(36, 52, 54, 44),
            )
            pdf.add_callout(
                "warning",
                "TIER 2 AUTOMATIC PURCHASE REQUISITION TRIGGER (RLR-4450)",
                "RLR-4450 carries min_reorder_qty = 2. Because Plant 04 stock starts at qty = 1, reserving RLR-4450 drops warehouse "
                "inventory to 0. Stock Agent MUST automatically insert an OPEN Purchase Requisition (PR) in SQLite while issuing the part.",
            )
            pdf.add_section_heading("4.1 Cam Track & Pre-Compression Roller Replacement Checklist")
            pdf.add_checklist([
                ("Pre-Compression Yoke Isolation", "Confirm 0V LOTO; release pneumatic overload cylinder pressure to 0 bar before removing yoke pin circlips."),
                ("Roller Extraction & Bush Check", "Pull worn RLR-4450 assembly; inspect needle pin surface for brinelling pits and bronze bush bore (< 0.03 mm oval)."),
                ("Install RLR-4450 & Set Clearance", "Fit fresh RLR-4450 (or qualified RLR-4450-ALT); set eccentric stop to 0.08-0.12 mm clearance with brass feeler."),
                ("Calibrated Fastener Torque", "Torque yoke clamp cap bolts to 52 Nm using calibrated click wrench; apply witness paint stripe across bolt head."),
                ("Step-Run Acoustic & Peak-g Verify", "Run turret empty at 20 RPM, then 45 RPM; confirm Peak-g < 1.1 g, Kurtosis <= 3.2, and RMS <= 2.5 mm/s."),
            ])
            pdf.add_two_col_cards(
                "Mechanical & Compaction Calibration Targets",
                [
                    "Pre-compression force symmetry check: <= 0.4 kN variance left vs right.",
                    "Lower punch tip height zeroing via 0.01 mm gauge post-roller change.",
                    "Eccentric pin lubrication: EP-2 NSF H1 high-load synthetic grease.",
                ],
                "Consequence of Delayed RLR-4450 Intervention",
                [
                    "Flat-spotted roller transfers shock loads directly to punch heads.",
                    "High risk of punch head mushrooming and cam-track gouging ($8k+ repair).",
                    "Early Kurtosis (> 6.0) triggers Tier 2 Auto-PO before catastrophic failure.",
                ],
            )
            pdf.add_styled_table(
                headers=("Bolt / Joint Location", "Thread Size & Grade", "Required Torque", "Locking Method & Check"),
                rows=[
                    ("Pre-Comp Yoke Retention Cap", "M12 Class 10.9 SS", "52 Nm (dry clean)", "Nord-Lock washer + red torque stripe"),
                    ("Cam Track Segment Bolts", "M8 Class 8.8 SS", "24 Nm diagonal", "Medium strength blue Loctite 243"),
                ],
                col_widths=(54, 42, 40, 50),
            )

            # Page 5: PK-02 Synchronous Drive Belt (BLT-2205) & Servo Coupling (SRV-2240)
            pdf.add_page()
            pdf.add_section_heading("5. PK-02 Blister Packaging (120 RPM): Belt Fatigue & Servo Coupling Drift")
            pdf.add_styled_table(
                headers=("Fault Mode", "Sensor / Physical Symptoms", "Root Cause & Stock Rule", "Part Code, Torque & GMP Target"),
                rows=[
                    (
                        "Synchronous Drive Belt Fatigue",
                        "6.5-9.5 mm/s RMS (7.9); 240.0 Hz 2x belt slip; 2.9 g peak; Kurtosis 4.9; chirping",
                        "Tensile cord stretch & tooth shear. OEM BLT-2205 is qty=0 -> evaluate BLT-2205-ALT",
                        "Install BLT-2205-ALT ($225); tension 180-200 N sonic; lock hub 38 Nm; post <= 1.5 mm/s",
                    ),
                    (
                        "Servo Indexing Coupling Drift",
                        "4.2-6.2 mm/s RMS (5.1); 120.0 Hz index jitter; eye-mark registration > +/-0.5 mm",
                        "Degraded elastomer spider backlash & fouled optical registration encoder",
                        "Install SRV-2240 ($520); zero-backlash align <= 0.04 mm; clamp 24 Nm; post <= 1.3 mm/s",
                    ),
                ],
                col_widths=(36, 50, 48, 52),
            )
            pdf.add_bar_chart(
                "PK-02 Drive & Servo Faults vs Verified Recovery",
                [
                    ("BLT-2205 Belt Slip Spike", 7.90, (239, 68, 68)),
                    ("SRV-2240 Servo Jitter Spike", 5.10, (245, 158, 11)),
                    ("Verified Belt Fix (BLT-2205-ALT)", 1.50, (16, 185, 129)),
                    ("Verified Servo Fix (SRV-2240)", 1.30, (16, 185, 129)),
                ],
                max_val=10.0,
                limit_val=2.5,
            )
            pdf.add_callout(
                "warning",
                "OEM STOCKOUT EQUIVALENCE RULE (BLT-2205 -> BLT-2205-ALT)",
                "If spare_parts_inventory shows BLT-2205 stock = 0, Stock Agent MUST call check_alternative_parts('BLT-2205'), verify "
                "gmp_qualified_alt = 1, generate an Auto-PO for OEM replenishment, and seek Human QA SQL Approval for BLT-2205-ALT.",
            )
            pdf.add_section_heading("5.1 Drive Belt & Servo Encoder Replacement Checklist")
            pdf.add_checklist([
                ("Pulley Coplanarity & Sprocket Check", "Verify driver/driven pulley faces with laser straight-edge <= 0.25 deg; inspect sprocket teeth for hook wear."),
                ("Damage-Free Belt Mounting", "Slacken motor idler bracket completely and slip BLT-2205 or BLT-2205-ALT onto teeth WITHOUT screwdrivers or prying."),
                ("Sonic Belt Tension & Hub Lock", "Pluck center span to 180-200 N (82 Hz acoustic peak); lock taper-lock hub bolts evenly to 38 Nm."),
                ("Servo Coupling & Eye-Mark Homing", "Fit SRV-2240 zero-backlash spider, clamp split-hub 24 Nm, wipe optical sensor lens with IPA, verify +/-0.15 mm mark."),
            ])
            pdf.add_two_col_cards(
                "PK-02 Sonic Tension & Registration Parameters",
                [
                    "Gates 8MGT-2200 Belt Mass Constant: 0.112 kg/m | Span: 420 mm.",
                    "Target span vibration frequency: 80 - 84 Hz (corresponds to 180-200 N).",
                    "Permissible eye-mark registration drift: <= +/-0.20 mm at 120 RPM.",
                ],
                "Post-Repair Blister Quality Verification",
                [
                    "Run 25 dummy blister cycles; perform vacuum dye leak check (-0.7 bar).",
                    "Confirm zero sheared belt rubber fines inside ISO Class 8 hood.",
                    "Record final RMS (<= 1.5 mm/s) and servo following error in eBMR.",
                ],
            )

            # Page 6: PK-02 Sealing Roller Bearing & Thermal Gate 3 Rework (BRG-2290)
            pdf.add_page()
            pdf.add_section_heading("6. PK-02 Sealing Platen Bearing Bind & Thermal Rework Rule (BRG-2290)")
            pdf.add_paragraph(
                "Alu-PVC/Alu-Alu rotary sealing platen rollers operate at 155-165 deg C. Lubricant carbonization seizes high-temp "
                "PTFE pillow bearings (BRG-2290), raising vibration to 7.1 mm/s RMS (312 Hz platen chatter) and causing blister channel leaks."
            )
            pdf.add_callout(
                "danger",
                "MANDATORY THERMAL GATE 3 REWORK RULE (PK-02 SEALING PLATEN)",
                "If BRG-2290 is installed cold without hot-soak parallelism leveling at operating temperature (160 deg C), thermal frame bow "
                "leaves residual vibration at ~3.9 mm/s RMS (> 2.5 mm/s GMP limit). Orchestrator Gate 3 MUST REJECT completion, escalate "
                "to Tier 2 Reliability Engineer, request Human QA Rework Approval, and enforce Thermal Parallelism Alignment (< 0.04 mm).",
            )
            pdf.add_styled_table(
                headers=("Check Stage", "Target Tolerance / Reading", "Measurement Method", "Rework / Release Action"),
                rows=[
                    ("Initial Failure State", "7.1 mm/s RMS | 312 Hz | 92.4 deg C", "Platen pedestal accelerometer + IR", "LOTO & replace seized BRG-2290 ($395)"),
                    ("Cold Replacement Attempt 1", "3.9 mm/s RMS (EXCEEDS 2.5 LIMIT)", "Post-repair vibration check (Gate 3)", "REJECT -> Escalate to Tier 2 Engineer"),
                    ("Thermal Parallelism Rework", "<= 0.04 mm feeler gap at 160 deg C", "Carbon paper foil imprint + shims", "Re-torque bearing pillow blocks to 38 Nm"),
                    ("Verified Hot-Soak Release", "<= 2.5 mm/s RMS (typ. 1.5 mm/s)", "10-min 120 RPM + vacuum leak test", "Approve Gate 3B -> QA Line Clearance"),
                ],
                col_widths=(40, 48, 50, 48),
                row_fills=[
                    (254, 242, 242),
                    (255, 251, 235),
                    (239, 246, 255),
                    (236, 253, 245),
                ],
            )
            pdf.add_section_heading("6.1 Hot-Soak Platen Parallelism Rework Sequence (Attempt #2)")
            pdf.add_checklist([
                ("160 deg C Thermal Equilibration", "Energize sealing cartridge heaters in Jog/Safe mode for 20 min until surface IR pyrometer stabilizes at 160 +/- 2 deg C."),
                ("Carbon Foil Nip Impression", "Place heat-resistant carbon transfer strip across entire platen width, close pneumatic cylinders to 4.5 bar, and inspect."),
                ("Shim Leveling & 38 Nm Hot Retorque", "Insert 304SS feeler shims under low side until nip gap variance is <= 0.04 mm across span; retorque hubs to 38 Nm."),
                ("Hot-Run Vibration & Leak Check", "Operate web drive 10 min at 120 RPM; confirm RMS <= 2.5 mm/s (1.5 mm/s target) and pass -0.7 bar methylene blue leak test."),
            ])
            pdf.add_signoff_block("Solid Dose & Primary Packaging Engineering Lead")

        _build_pdf_file(p1, "SOP-MNT-001", "Tablet Press & Blister Packaging Vibration SOP", "Rev 6.0", "MAINTENANCE SOP", _build_sop1)

    # 2. SOP-MNT-002 (6 dense pages)
    p2 = SOPS_DIR / "SOP-MNT-002_RotaryPump_Seal_And_Cavitation.pdf"
    if force or not p2.exists():
        def _build_sop2(pdf: _GMPDocumentPdf) -> None:
            # Page 1: Standards, Sterile Boundary, CIP/SIP Burn Safety & NPSHa Rules
            pdf.add_page()
            pdf.add_banner_title(
                "SOP-MNT-002: Sanitary Pump Seal, Impeller, Soft-Foot & Laser Alignment SOP",
                "EU GMP Annex 1 sterile boundary maintenance, cavitation remediation, and precision alignment for RP-04 (1450 RPM).",
            )
            pdf.add_doc_control_grid([
                ("Document ID", "SOP-MNT-002", "Revision", "Rev 5.0 (Effective 2026-03-15)"),
                ("Asset Covered", "RP-04 Sanitary Centrifugal Pump", "Operating Speed", "1450 RPM (24.2 Hz) | Rate: $52.00/min"),
                ("Compliance", "EU GMP Annex 1 / 3-A / EN 10204 3.1", "GMP Limit", "<= 2.5 mm/s RMS (ISO 20816-3)"),
                ("Fluid Service", "Purified Water / WFI Loop B", "CIP / SIP Temp", "85 deg C CIP / 121 deg C SIP cycles"),
            ])
            pdf.add_callout(
                "danger",
                "THERMAL CIP/SIP BURN & STERILITY BOUNDARY HAZARD",
                "RP-04 circulates Purified Water / WFI at 85 deg C CIP and 121 deg C SIP cycles. Isolate suction and discharge diaphragm valves, "
                "vent top sample cock to verify 0.0 bar gauge pressure, confirm volute skin temperature < 35 deg C via IR thermometer, and "
                "lock out 480V VFD isolator prior to unbolting spacer coupling or tri-clamps.",
            )
            pdf.add_section_heading("1. Sanitary Design, NPSHa Margin & ISO 20816-3 Governance")
            pdf.add_paragraph(
                "A breached RP-04 sterile envelope exposes critical WFI loops to biofilm or particulate ingress. Only USP Class VI Viton/EPDM "
                "elastomers, electropolished Ra < 0.5 um 316L wetted internals, and EN 10204 3.1 heat-lot certified spare parts are permitted."
            )
            _iso_20816_table(pdf)
            pdf.add_two_col_cards(
                "EU GMP Annex 1 Sterile Boundary Mandates",
                [
                    "Silicone-free, sterile pharma-grade assembly lube on all Viton O-rings.",
                    "Mandate fresh USP Class VI EPDM gaskets (GKT-1102) on any flange break.",
                    "Perform 6.0 bar hydrostatic + spray riboflavin rinse check before batch use.",
                    "Attach EN 10204 3.1 material heat certificate numbers directly into eBMR.",
                ],
                "Minimum NPSHa Cavitation Margin Standards",
                [
                    "Required margin: NPSHa >= NPSHr + 1.5 m across maximum WFI draw flow.",
                    "Suction basket strainer dP alarm threshold: > 0.25 bar triggers flush.",
                    "Minimum loop supply pressure at pump eye: >= 0.45 bar positive head.",
                    "Verify sterile double-seal barrier flush +1.5 bar above volute pressure.",
                ],
            )
            pdf.add_styled_table(
                headers=("RP-04 Operating Envelope", "Nominal Value", "Alarm Trigger", "Diagnostic Interpretation"),
                rows=[
                    ("1x Shaft Speed (1450 RPM)", "24.2 Hz | 1.3 mm/s", "> 2.5 mm/s at 24.2 Hz", "Impeller mass unbalance or coupling offset"),
                    ("6x Vane-Pass Freq (145 Hz)", "< 0.4 mm/s peak", "> 3.0 mm/s at 145 Hz", "Blade-pass erosion or suction recirculation"),
                    ("High-Freq Enveloping (1.4 kHz)", "< 0.6 g peak", "> 3.5 g | 1420 Hz", "Vapor bubble implosion / SiC seal chatter"),
                ],
                col_widths=(48, 34, 40, 64),
            )

            # Page 2: Fault Mode 1 - Cartridge Seal Cavitation (SL-8810)
            pdf.add_page()
            pdf.add_section_heading("2. RP-04 Fault Mode 1: Cartridge Mechanical Seal Cavitation (SL-8810)")
            pdf.add_paragraph(
                "Upstream suction strainer blockage drops Net Positive Suction Head Available (NPSHa) below NPSHr, inducing violent vapor "
                "bubble collapse against the SiC cartridge faces of SL-8810 (John Crane Type 5620) and shattering brittle seal rings."
            )
            pdf.add_kpi_strip([
                ("Alarm Peak RMS", "9.4 mm/s", "Severe gland weeping"),
                ("Cavitation Noise", "1420 Hz", "5.8 g | Kurtosis 6.1"),
                ("OEM Cartridge", "SL-8810", "$780 (John Crane 5620)"),
                ("GMP Target", "<= 1.6 mm/s", "Post-laser alignment"),
            ])
            pdf.add_styled_table(
                headers=("Diagnostic Metric", "Seal Cavitation Fault Signature", "Root Cause & Part Selection", "Release Verification"),
                rows=[
                    ("Tri-Axial RMS Velocity", "7.5 - 11.5 mm/s RMS (typ. 9.4)", "SiC face micro-fracture + gland weeping", "<= 2.5 mm/s RMS post-align (1.6)"),
                    ("High-Freq Acoustic", "1420 Hz broadband cavitation crackle", "Vapor implosion inside volute suction eye", "Clear suction strainer; steady flow"),
                    ("Kurtosis & Peak-g", "Kurtosis 6.1 | Peak 5.8 g", "High-energy transient bubble shockwaves", "Kurtosis <= 3.1 | Peak < 1.0 g"),
                    ("Spare Part Spec", "SL-8810 ($780) [Alt: SL-8810-ALT $820]", "FDA/3-A cartridge, gland follower 32 Nm", "Barrier flush +1.5 bar; 0 drops/min"),
                ],
                col_widths=(36, 50, 54, 46),
            )
            pdf.add_section_heading("2.1 SL-8810 Cartridge Replacement Checklist")
            pdf.add_checklist([
                ("Coupling Removal & Sleeve Runout", "Unbolt spacer coupling; draw failed SL-8810 with uniform jacking bolts; confirm shaft sleeve TIR < 0.03 mm."),
                ("Sterile Elastomer Lubrication", "Coat new SL-8810 Viton O-rings lightly with USP sterile silicone-free assembly gel; slide gently over shaft chamfer."),
                ("Gland Stud Torque & Drive Collar", "Tighten 4 gland follower nuts in cross sequence to 32 Nm; lock drive collar set screws evenly onto shaft."),
                ("Remove Spacer Setting Clips LAST", "Only after gland and drive collar are torqued, unclip and store the plastic axial setting spacer clips in maintenance tray."),
                ("Suction Basket Strainer Flush", "Open duplex suction strainer, clear captured resin/filter media, confirm NPSHa >= NPSHr + 1.5 m prior to prime."),
            ])
            alt_rule = (
                "EagleBurgmann Cartex-DN (SL-8810-ALT, $820) satisfies identical DIN 24960 sanitary gland envelope and EN 10204 3.1 certification."
            )
            pdf.add_callout(
                "info",
                "QUALIFIED ALTERNATIVE CARTRIDGE RULE (SL-8810-ALT)",
                alt_rule + " Reinstall using identical 32 Nm stud torque and remove setting clips strictly after shaft collar lockup.",
            )
            pdf.add_two_col_cards(
                "Pre-Installation Seal Chamber Verification",
                [
                    "Inspect 316L seal housing bore flatness & clean gasket shoulder with 70% IPA.",
                    "Confirm shaft sleeve Ra <= 0.4 um roughness and radial runout <= 0.03 mm TIR.",
                    "Confirm sterile barrier flush regulator set at +1.5 bar above volute discharge.",
                ],
                "Suction Cavitation Prevention Guardrails",
                [
                    "Verify duplex suction basket DP < 0.25 bar prior to resuming WFI transfer.",
                    "Maintain pump inlet submergence and bleed air from top vent petcock pre-start.",
                    "Escalate immediately to Gate 3 alignment if post-seal RMS exceeds 2.5 mm/s.",
                ],
            )

            # Page 3: Gate 3 Laser Realignment Rule (4.8 mm/s -> 1.6 mm/s)
            pdf.add_page()
            pdf.add_section_heading("3. Mandatory Gate 3 Rejection & Dual-Beam Laser Shaft Realignment")
            pdf.add_callout(
                "warning",
                "ORCHESTRATOR GATE 3 REJECTION & HUMAN REWORK APPROVAL",
                "Replacing SL-8810 alone frequently yields ~4.8 mm/s RMS post-repair vibration (1x/2x axial coupling peaks) if thermal CIP growth "
                "compensation is omitted. Quality Orchestrator MUST REJECT closure, request Human QA rework approval, and mandate Section 3.",
            )
            pdf.add_bar_chart(
                "RP-04 Gate 3 Rejection & Laser Alignment Rework Recovery Path",
                [
                    ("Pre-Repair Seal Rupture", 9.40, (239, 68, 68)),
                    ("Attempt #1 (Cold Coupling Only)", 4.80, (217, 119, 6)),
                    ("Attempt #2 (Dual-Beam Laser)", 1.60, (16, 185, 129)),
                ],
                max_val=10.5,
                limit_val=2.5,
            )
            pdf.add_styled_table(
                headers=("Alignment Parameter", "Cold Target Tolerance (1450 RPM)", "Instrumentation", "85 deg C CIP Thermal Growth Offset"),
                rows=[
                    ("Base Soft-Foot Check", "< 0.05 mm at all 4 motor mounts", "Dial indicator + 304SS feeler gauge", "Eliminate casing distortion first"),
                    ("Parallel Offset (Vertical/Horiz)", "<= 0.05 mm TIR across coupling", "Dual-beam laser shaft analyzer", "Set motor -0.03 mm cold vertical offset"),
                    ("Angular Misalignment", "<= 0.05 mm per 100 mm span", "Dual-beam laser live move screen", "Maintains < 0.02 mm hot coaxial line"),
                    ("Anchor & Base Torque", "55 Nm calibrated torque wrench", "Star torque + final laser verification", "Verify final RMS <= 2.5 mm/s (1.6)"),
                ],
                col_widths=(38, 48, 48, 52),
            )
            pdf.add_section_heading("3.1 Dual-Beam Cold Laser Shaft Realignment Sequence (Attempt #2)")
            pdf.add_checklist([
                ("Eliminate Motor Soft Foot", "Verify <= 0.05 mm vertical lift at all 4 motor feet using magnetic dial gauge; shim per Section 5."),
                ("Mount S & M Laser Heads", "Clamp Stationary (pump) and Movable (motor) prism heads equidistant across coupling; sweep 9-12-3 o'clock arc."),
                ("Dial In Cold Thermal Offset", "Adjust motor vertical shims to -0.03 mm low relative to pump so 85 deg C CIP expansion aligns shafts dead-coaxial."),
                ("Cross-Torque Base & Re-Sweep", "Tighten 4 anchor bolts to 55 Nm in star order; verify zero bolt-bound shift and run 10-min WFI test <= 1.6 mm/s."),
            ])
            pdf.add_two_col_cards(
                "Thermal Growth Calculation (RP-04 316L vs Cast Motor)",
                [
                    "Delta-T = 85 deg C CIP - 22 deg C ambient = +63 deg C frame rise.",
                    "Centerline height h = 160 mm; alpha = 1.6e-5 / K -> rise ~ +0.16 mm.",
                    "Volute runs hotter than motor housing -> target -0.03 mm cold offset.",
                ],
                "Why Unaligned Couplings Destroy New SiC Seals",
                [
                    "0.20 mm parallel offset imposes > 450 N cyclic radial shaft deflection.",
                    "Forces secondary SiC stationary seat chatter every 41 ms at 1450 RPM.",
                    "Rework eliminates 1x/2x axial vibration and triples mean time between seals.",
                ],
            )

            # Page 4: Fault Mode 2 - Impeller Erosion & Dynamic Unbalance (IMP-8825)
            pdf.add_page()
            pdf.add_section_heading("4. RP-04 Fault Mode 2: Impeller Erosion & Blade-Pass Unbalance (IMP-8825)")
            pdf.add_paragraph(
                "Extended micro-cavitation or abrasive slurry carryover erodes leading edges of the 6-vane 316L electropolished open impeller "
                "(IMP-8825), causing severe 1x radial mass unbalance and 6x vane-pass pulsations (145.0 Hz at 1450 RPM)."
            )
            pdf.add_styled_table(
                headers=("Parameter", "Impeller Erosion Fault State", "Corrective Engineering Spec", "Verified Target"),
                rows=[
                    ("Tri-Axial RMS Velocity", "8.0 - 11.0 mm/s RMS (typ. 8.9)", "Replace with ISO 1940-1 G2.5 balanced IMP-8825 ($910)", "<= 2.5 mm/s RMS (1.4)"),
                    ("Spectral Signature", "145.0 Hz (6x vane-pass) + 24.2 Hz 1x", "Remove pitted impeller; check keyway fretting", "6x peak reduced > 75%"),
                    ("Pump Discharge Head", "> 20% discharge pressure drop", "Set impeller-to-backplate axial gap 0.40-0.60 mm", "100% nominal flow / head"),
                    ("Impeller Retainer Torque", "65 Nm w/ Nord-Lock sanitary washer", "Inspect backplate wear & torque calibrated wrench", "Zero axial rub; < 50 deg C"),
                ],
                col_widths=(36, 50, 56, 44),
            )
            pdf.add_section_heading("4.1 Impeller Removal, Axial Gap & Trim-Balance Installation Steps")
            pdf.add_checklist([
                ("Volute Casing Opening", "Unclamp heavy-duty tri-clamp volute band and inspect electropolished Ra < 0.5 um casing interior for cavitation gouging."),
                ("Impeller Pull & Keyway Check", "Extract eroded impeller with hygienic jacking puller; inspect parallel keyway and verify shaft straightness <= 0.02 mm TIR."),
                ("Mount ISO G2.5 IMP-8825", "Slide pre-balanced IMP-8825 onto tapered spline/key; confirm factory dynamic balance tag matches serial lot in eBMR."),
                ("Set Backplate Clearance Gap", "Use calibrated non-scratching brass feeler blades around all 6 vanes to confirm 0.40 - 0.60 mm axial backplate gap."),
                ("Torque Dome Nut to 65 Nm", "Fit enclosed sanitary O-ring cap nut with Nord-Lock washer; torque to 65 Nm; hand-spin fan to confirm zero scrape."),
            ])
            pdf.add_two_col_cards(
                "ISO 1940-1 G2.5 Impeller Dynamic Balance Spec",
                [
                    "Max residual unbalance permissible at 1450 RPM: <= 16.5 g-mm / kg.",
                    "Do NOT polish or weld on balanced vanes in field without rebalancing.",
                    "Open 6-vane sanitary profile eliminates deadlegs for clean CIP drainage.",
                ],
                "Post-Installation Hydraulic Performance Check",
                [
                    "Target WFI loop differential head: 3.8 bar +/- 0.15 bar at 1450 RPM.",
                    "Verify VFD drive current settles from faulted load to 14.8 - 15.6 A.",
                    "Capture Radial-X/Y and 145 Hz vane-pass amplitude before line release.",
                ],
            )
            pdf.add_styled_table(
                headers=("Hydraulic / Mechanical Metric", "Eroded Impeller State", "New IMP-8825 Baseline", "Acceptance Criterion"),
                rows=[
                    ("WFI Flow Rate", "78% nominal (-22% loss)", "100% nominal (18 m3/hr)", ">= 96% rated curve"),
                    ("1x Radial / 6x Vane-Pass", "8.9 mm/s RMS | Peak 3.9g", "1.4 mm/s RMS | Peak 0.5g", "<= 2.5 mm/s RMS"),
                ],
                col_widths=(48, 44, 48, 46),
            )

            # Page 5: Fault Mode 3 - Motor Soft-Foot Distortion & Coupling Drift (SHM-8805)
            pdf.add_page()
            pdf.add_section_heading("5. RP-04 Fault Mode 3: Motor Soft-Foot Distortion & Shim Pack Fix (SHM-8805)")
            pdf.add_paragraph(
                "Uneven skid leveling or corroded mild-steel washers create motor frame twist ('soft foot'), bending bearing housings "
                "every time anchor bolts are torqued and generating strong 2x running speed (48.3 Hz) axial coupling vibration."
            )
            pdf.add_styled_table(
                headers=("Measurement", "Soft-Foot Fault Profile", "SHM-8805 Shim & Alignment Action", "GMP Release State"),
                rows=[
                    ("Tri-Axial RMS & Axial-Z", "4.5 - 6.8 mm/s RMS (5.6); high axial_z", "Install laser-cut 304SS SHM-8805 shim pack ($165)", "<= 2.5 mm/s RMS (1.3)"),
                    ("Dominant Harmonic", "48.3 Hz (2x RPM) axial coupling force", "Max 3 stainless shims per foot; never layer > 3", "48.3 Hz peak < 0.5 mm/s"),
                    ("Dial Soft-Foot Lift", "> 0.12 mm lift when loosening foot bolt", "Correct all 4 feet to < 0.05 mm vertical lift", "Bolt torque 55 Nm verified"),
                    ("Motor Stator Current", "23.0 A (+53% rotor air-gap twist)", "Frame relaxation restores concentric rotor gap", "14.9 A nominal balanced"),
                ],
                col_widths=(38, 48, 56, 44),
            )
            pdf.add_section_heading("5.1 Four-Corner Soft-Foot Sweep & 304SS Shimming Procedure")
            pdf.add_checklist([
                ("Rough Check & Pad Cleaning", "Scrape dried paint/scale under all 4 motor feet with flat stone & IPA; inspect baseplate tapped threads."),
                ("Single-Bolt Dial Lift Audit", "Position 0.01 mm dial gauge vertically above foot #1; loosen ONLY bolt #1; record spring rise; re-torque; repeat #2-#4."),
                ("Classify Parallel vs Angular Foot", "Use 4-point feeler probe under air-gapped foot to distinguish parallel air gap vs skewed/bent mounting foot."),
                ("Install SHM-8805 Laser Shims", "Select thickest single 304SS shim (1.0 mm, 0.5 mm, 0.25 mm) to minimize spring effect (strictly <= 3 shims per corner)."),
                ("Final Laser Verification", "Confirm < 0.05 mm lift across all 4 feet, re-check coupling offset (-0.03 mm cold CIP target), and torque to 55 Nm."),
            ])
            pdf.add_callout(
                "warning",
                "STRICT THREE-SHIM MAXIMUM STACKING RULE",
                "Never stack more than 3 thin shims under a motor foot—excess layers act as a damped spring under 55 Nm bolt tension, "
                "causing resonance drift at 48.3 Hz and voiding alignment calibration.",
            )
            pdf.add_styled_table(
                headers=("SHM-8805 Shim Pack Contents", "Thickness Tier", "Qty Included", "Approved Usage Note"),
                rows=[
                    ("Size-B Pre-Cut 304SS Slot 23mm", "0.05 / 0.10 / 0.25 mm", "4 each", "Fine vertical/angular correction"),
                    ("Size-B Heavy 304SS Spacer", "0.50 / 1.00 / 2.00 mm", "2 each", "Base parallel soft-foot stack consolidation"),
                ],
                col_widths=(56, 42, 30, 58),
            )
            pdf.add_two_col_cards(
                "Common Soft-Foot Fault Mechanisms on RP-04",
                [
                    "Bent welded baseplate gusset creates angular rocking across diagonal corners.",
                    "Galvanic corrosion on old mild-steel shims produces uneven compressive settling.",
                    "External discharge piping strain pulls pump volute laterally when flanges bolt up.",
                ],
                "Anchor Bolt & Piping Stress Discipline",
                [
                    "Always loosen pipe hanger supports before laser check; bolt flange stress-free first.",
                    "Verify zero dial deflection (> 0.02 mm) when loosening/tightening suction flange studs.",
                ],
            )

            # Page 6: Gasket Reassembly (GKT-1102) & Sign-off
            pdf.add_page()
            pdf.add_section_heading("6. Sanitary Flange Reassembly (GKT-1102), Flush Check & Release Sign-Off")
            pdf.add_paragraph(
                "Every fluid-side opening on RP-04 requires fresh USP Class VI EPDM Tri-Clamp flange gaskets (GKT-1102) and sterile flush "
                "pressure verification before QA Line Clearance releases the pump back into cGMP WFI production."
            )
            pdf.add_styled_table(
                headers=("Sanitary Boundary Checkpoint", "Acceptance Criterion", "Governing Standard", "Record Location"),
                rows=[
                    ("Tri-Clamp Gasket Replacement", "Fresh GKT-1102 USP Class VI EPDM, 25 Nm clamp", "ASME BPE / 3-A", "eBMR Material Traceability"),
                    ("Sterile Seal Flush DP", "+1.5 bar differential pressure above volute", "EU GMP Annex 1", "Technician Pre-Run Log"),
                    ("CIP / Hydrostatic Pressure Hold", "6.0 bar hold x 10 min with zero weep/drop", "ISPE Baseline Vol 4", "QA Line Clearance Gate 3B"),
                    ("Final Tri-Axial Vibration", "<= 2.5 mm/s RMS; Kurtosis <= 3.3", "ISO 20816-3 / SOP-MNT-002", "Final Telemetry Snapshot"),
                ],
                col_widths=(48, 56, 40, 42),
            )
            pdf.add_section_heading("6.1 Pre-Clearance Sterile Recommissioning Checklist")
            pdf.add_checklist([
                ("Remove Old Elastomers", "Verify 100% old GKT-1102 gasket recovery; confirm zero black elastomer fragments inside pump head or strainer."),
                ("Torque Sanitary Ferrules", "Seat new GKT-1102 squarely on groove bead; tighten wing-nuts evenly to 25 Nm torque wrench limit."),
                ("Barrier Water & Hydro-Test", "Establish +1.5 bar sterile flush across SL-8810; pressurize WFI loop to 6.0 bar for 10 min with zero drop."),
                ("10-Minute Stability Release", "Record 1450 RPM vibration <= 2.5 mm/s RMS across Radial-X, Radial-Y, Axial-Z before QA Clearance signs off."),
            ])
            pdf.add_two_col_cards(
                "QA Line Clearance Hold Verification",
                [
                    "Verify 100% hand-tool & hardware reconciliation count prior to closeout.",
                    "Confirm 480V LOTO personal lock removal by authorized technician.",
                    "Release LOT-2026-WFI-812 ONLY after CIP rinse conductivity < 1.3 uS/cm.",
                ],
                "WFI Loop Production Release Criteria",
                [
                    "Composite RMS <= 2.5 mm/s AND housing temperature < 52 deg C.",
                    "Zero audible cavitation crackle at suction reducer conical section.",
                    "All eBMR audit records cryptographically linked via SHA-256.",
                ],
            )
            pdf.add_signoff_block("Sterile Utilities & Fluid Maintenance Lead")

        _build_pdf_file(p2, "SOP-MNT-002", "Sanitary Pump Seal & Laser Alignment SOP", "Rev 5.0", "MAINTENANCE SOP", _build_sop2)

    # 3. GMP-QA-010 (6 dense pages)
    p3 = SOPS_DIR / "GMP-QA-010_Maintenance_Protocol_And_Signoff.pdf"
    if force or not p3.exists():
        def _build_qa(pdf: _GMPDocumentPdf) -> None:
            # Page 1: ALCOA+ & Supervisor Principles
            pdf.add_page()
            pdf.add_banner_title(
                "GMP-QA-010: eBMR Protocol, 3-Tier Agent Orchestra & ALCOA+ Governance",
                "Central Quality Supervisor governance, cryptographic hash-chain integrity, and Human-in-the-Loop QA rules.",
            )
            pdf.add_doc_control_grid([
                ("Document ID", "GMP-QA-010", "Revision", "Rev 7.0 (Effective 2026-03-15)"),
                ("Regulatory Basis", "EU GMP Annex 11 / 21 CFR Part 11", "Audit Integrity", "SHA-256 Hash-Chained SQLite Log"),
                ("Architecture", "3-Tier LangGraph Parallel Orchestra", "Human QA Gates", "SQL Mutation + Repair / Rework Cost"),
                ("Site Governance", "Plant 04 Sterile & Solid-Dose", "Retention Policy", "Immutable 10-Year eBMR Archive"),
            ])
            pdf.add_section_heading("1. ALCOA+ Data Integrity & Non-Bypassable Supervisor Principles")
            pdf.add_styled_table(
                headers=("ALCOA+ Principle", "Control Tower Implementation", "Verification Evidence"),
                rows=[
                    ("Attributable", "Every step stamped with named Agent Role, Orchestrator, or Human QA Lead", "gmp_audit_log.agent_name"),
                    ("Legible & Contemporaneous", "Typed MaintenanceState eBMR with live thinking traces & UTC timestamps", "agent_traces + timestamp_utc"),
                    ("Original & Accurate", "Append-only SQLite rows linked with SHA-256 prev_hash -> record_hash", "SHA-256 cryptographic chain"),
                    ("Complete & Consistent", "Zero peer-to-peer shortcuts; all outputs routed through Tier 3 Conductor", "LangGraph StateGraph topology"),
                    ("Enduring & Available", "Auto-seeded SQLite WAL store, 4-sheet Excel export & 2p PDF Closure Report", "incident_analytics + PDF cert"),
                ],
                col_widths=(40, 96, 50),
            )
            pdf.add_callout(
                "info",
                "HUMAN-IN-THE-LOOP (HITL) MANUFACTURING GOVERNANCE",
                "AI agents investigate telemetry, search SOPs, and draft SQL mutations or repair estimates, but a Human QA Supervisor retains "
                "binding permission over inventory database writes (Gate 2A) and repair/rework financial authorizations (Gate 3A).",
            )
            pdf.add_two_col_cards(
                "SHA-256 Tamper-Evident Audit Chain Design",
                [
                    "Step #0 initialized with prev_hash = 'GENESIS' on environment boot.",
                    "Each record hashes: prev_hash|ticket|step|utc|agent|action|gate|notes.",
                    "Any retroactive DB row edit invalidates all downstream record_hash values.",
                    "Thread-locked monotonic step counter handles parallel Send() steps cleanly.",
                ],
                "EU GMP Annex 11 & Part 11 Operational Scope",
                [
                    "Strict read-only SQLite guard blocks DDL/DML by LLM diagnostic queries.",
                    "Only atomic authorized reserve_spare_part() commits decrement inventory.",
                    "All Human QA comments & reviewer IDs permanently bound to record hash.",
                    "Offline Safe Demo Mode uses identical hashing, SQL & PDF engines.",
                ],
            )
            pdf.add_styled_table(
                headers=("eBMR SQLite Table", "Primary Purpose", "Append / Mutation Rule"),
                rows=[
                    ("spare_parts_inventory", "15-part real-time bin counts & qualified alternatives", "Mutated ONLY after Gate 2A Human Approval"),
                    ("gmp_audit_log", "Hash-chained step ledger for all agents & human reviewers", "Strictly INSERT append-only (no UPDATE/DELETE)"),
                    ("capa_records / purchase_requisitions", "Auto-generated 5-Why CAPAs & low-stock supplier POs", "Triggered deterministically by GMP thresholds"),
                ],
                col_widths=(50, 74, 62),
            )

            # Page 2: 3-Tier 7-Role Orchestra Topology
            pdf.add_page()
            pdf.add_section_heading("2. Three-Tier Hierarchical Multi-Agent Orchestra (7 Roles)")
            pdf.add_paragraph(
                "Modeled after regulated pharmaceutical reliability engineering teams, Plant 04 separates command authority, "
                "diagnostic engineering, inventory planning, shop-floor execution, QA release, and financial CAPA oversight."
            )
            pdf.add_styled_table(
                headers=("Tier & Role", "Node Name", "Core Responsibilities", "Reporting & Escalation Path"),
                rows=[
                    ("Tier 3 Supervisor", "orchestrator", "Enforces Gates 1-4; fans out parallel intake; issues commands", "Commands all Tier 2 / Tier 1 / Tier 3 nodes"),
                    ("Human Authority", "human_qa_lead", "Approves/Denies SQL mutations, Repair Cost $, Rework Cost $", "Binding veto on Gate 2A & Gate 3A"),
                    ("Tier 2 Reliability Eng", "diagnostic_engineer", "Hybrid TF-IDF+BM25 SOP RAG + spectral RCA + failure replan", "Reports to Conductor; handles Tech escalations"),
                    ("Tier 2 Material / Batch", "stock_precheck", "Parallel batch impact risk check & early SQL stock query", "Runs concurrently with Engineer via Send()"),
                    ("Tier 2 Stock / Auto-PO", "stock_agent", "Reserves OEM or alt parts; raises Auto-PO if qty <= reorder", "Waits for Gate 2A Human SQL Approval"),
                    ("Tier 1 Field Technician", "field_technician", "Executes LOTO + repair SOP + multi-axis vibration verification", "Escalates UP to Engineer if RMS > 2.5"),
                    ("Tier 3 QA Release", "qa_clearance", "LOTO restoration, cleanliness wipe-down & line clearance", "Issues CLEARED / HOLD verdict to Conductor"),
                    ("Tier 2/3 Analyst", "analyst_capa", "Downtime, MTTR, financial $, 4-sheet .xlsx, 30d repeat CAPA", "Finalizes KPIs before Conductor FINISH"),
                ],
                col_widths=(32, 34, 68, 52),
            )
            pdf.add_two_col_cards(
                "Parallel Send() Intake Fan-Out Rules",
                [
                    "Edge alarm simultaneously triggers diagnostic_engineer + stock_precheck.",
                    "Engineers analyze spectral vibration while QA evaluates active lot hold.",
                    "Parallel state fields use reducer-safe deduplication (_merge_append_unique).",
                    "Zero race conditions: scalar worker keys stay isolated per node.",
                ],
                "Strict Hierarchy & No Peer-Bypass Governance",
                [
                    "Workers NEVER invoke peer agents or mutate downstream workflow routes.",
                    "Tier 1 Technician executes strictly SOP or Engineer-prescribed steps.",
                    "Any LLM orchestrator mismatch is overridden by deterministic interlocks.",
                    "Failed Gate 3 checks ALWAYS bounce through Tier 3 Conductor -> Tier 2 Eng.",
                ],
            )
            pdf.add_styled_table(
                headers=("Orchestra Agent Role", "Permitted Tools", "Restricted / Prohibited Actions"),
                rows=[
                    ("Diagnostic Engineer", "read_sensor_telemetry, search_sop_manuals", "Cannot query or modify SQLite inventory directly"),
                    ("Stock & Auto-PO", "query_inventory_sql, check_alt, create_pr, reserve_part", "Cannot reserve without Gate 2A approval when HITL on"),
                    ("Field Technician", "execute_part_replacement, run_post_repair_vibration_test", "Cannot close ticket or alter engineering torque/RMS limits"),
                ],
                col_widths=(40, 74, 72),
            )

            # Page 3: 6 Mandatory Quality Gates
            pdf.add_page()
            pdf.add_section_heading("3. Mandatory Six-Gate State Machine & Escalation Handling")
            pdf.add_styled_table(
                headers=("Gate", "Owner", "Pass Criteria", "Rejection / Hold / Escalation Action"),
                rows=[
                    ("Gate 1: Diagnostic", "Tier 3 Conductor", "Valid SOP code + page + root cause + exact part code", "REJECTED -> re-query SOP PDFs"),
                    ("Gate 2A: SQL Auth", "Human QA Lead", "Human approves exact UPDATE/INSERT SQL + bin/qty", "DENIED -> zero DB changes; workflow ends"),
                    ("Gate 2B: Material", "Tier 3 Conductor", "OEM reserved OR gmp_qualified_alt approved; PO if low", "ESCALATED if both OEM & alt qty = 0"),
                    ("Gate 3A: Cost Auth", "Human QA Lead", "Human approves Parts $ + Labor $ + Est. Downtime $", "DENIED -> repair blocked; LOTO held"),
                    ("Gate 3B: Vibration", "Tier 3 Conductor", "Post-repair tri-axial RMS strictly <= 2.5 mm/s", "REJECTED -> Escalate Tech->Eng -> Rework $"),
                    ("Gate 4: Release & CAPA", "Tier 3 Conductor", "QA clearance PASSED + analytics SQL + .xlsx exported", "SIGNED_OFF -> Incident Certificate & eBMR"),
                ],
                col_widths=(32, 32, 62, 60),
            )
            pdf.add_callout(
                "warning",
                "UPWARD ESCALATION PROTOCOL (TIER 1 TECHNICIAN -> TIER 2 ENGINEER)",
                "If post-repair verification fails Gate 3B (e.g., 4.8 mm/s on RP-04 coupling or 3.9 mm/s on PK-02 hot sealing platen), the "
                "Tier 1 Technician NEVER improvises an unapproved fix. The fault escalates UP to the Tier 2 Reliability Engineer to issue "
                "a formal SOP realignment procedure (SOP-MNT-002 p.3 or SOP-MNT-001 p.6) prior to rework execution.",
            )
            pdf.add_two_col_cards(
                "Gate 2A: Human QA SQL_MUTATION Payload",
                [
                    "Exact parameterized UPDATE spare_parts_inventory SQL preview.",
                    "Stock delta visibility: current_qty -> projected_qty & BIN location.",
                    "Explicit OEM vs qualified alternative badge (e.g. BLT-2205-ALT).",
                    "Linked Auto-PO requisition ID if replenishment threshold crossed.",
                ],
                "Gate 3A: Human QA REPAIR_COST & REWORK Payload",
                [
                    "Itemized breakdown: Spare Parts $ + Labor ($95/hr) + Stoppage Loss $.",
                    "Incremental rework cost delta shown explicitly on Attempt #2.",
                    "Bound Engineer Replan procedure, SOP page ref & torque tolerance.",
                    "Reviewer name & optional comment hashed straight into gmp_audit_log.",
                ],
            )
            pdf.add_styled_table(
                headers=("Live Anomaly Scenario", "First-Attempt Result", "Conductor Decision", "Second-Attempt Target"),
                rows=[
                    ("RP-04 Seal Cavitation (SL-8810)", "4.8 mm/s RMS (Axial 1x/2x)", "Gate 3 REJECT -> Laser Replan", "1.6 mm/s RMS (PASS <= 2.5)"),
                    ("PK-02 Sealing Platen (BRG-2290)", "3.9 mm/s RMS (160C Thermal Bow)", "Gate 3 REJECT -> Hot Parallelism", "1.5 mm/s RMS (PASS <= 2.5)"),
                ],
                col_widths=(52, 46, 48, 40),
            )

            # Page 4: Multi-Sensor Variance & Persistence Filter
            pdf.add_page()
            pdf.add_section_heading("4. Multi-Sensor Telemetry Variance & Spectral Decision Rules")
            pdf.add_paragraph(
                "Real industrial rotating machinery exhibits continuous multi-harmonic vibration modulated by batch compaction tonnage, WFI "
                "valve cycling, and servo web inertia. The live plant oscilloscope samples six synchronized vectors every tick."
            )
            pdf.add_bar_chart(
                "Representative Spectral Tri-Axial Severity vs ISO 20816-3 Thresholds",
                [
                    ("Healthy Plant Baseline (Nom.)", 1.20, (56, 189, 248)),
                    ("Single-Tick Transient Spike", 2.58, (245, 158, 11)),
                    ("PK-02 Thermal Bow Rework Trigger", 3.90, (217, 119, 6)),
                    ("RP-04 Post-Seal Misalignment", 4.80, (239, 68, 68)),
                    ("TP-01 Turret Bearing Spall", 9.80, (185, 28, 28)),
                ],
                max_val=11.0,
                limit_val=2.5,
            )
            pdf.add_styled_table(
                headers=("Telemetry Vector", "Normal Operating Range", "Diagnostic Role in Plant 04", "Trip / Escalation Indicator"),
                rows=[
                    ("Radial-X / Y & Axial-Z", "0.8 - 1.5 mm/s RMS", "Distinguishes radial unbalance/spall vs axial misalignment", "Max vector > 2.5 mm/s triggers trip"),
                    ("Composite Vector RMS", "1.0 - 1.4 mm/s RMS", "ISO 20816-3 Group 2 Rigid primary compliance figure", "> 2.5 mm/s fails Gate 3B verification"),
                    ("Peak Acceleration (g)", "0.3 - 0.9 g peak", "Detects high-frequency impact/cavitation shockwaves", "> 2.5 g signals metal impact / cavitation"),
                    ("Kurtosis (4th Moment)", "2.7 - 3.3 (Gaussian)", "Early bearing raceway & cam-knock impulsiveness detector", "> 4.5 proves discrete subsurface impacts"),
                    ("Temp & Motor Amps", "42-54 deg C | nominal A", "Confirms friction drag, lube starvation, or platen bind", "> 65 deg C or +25% current surge"),
                ],
                col_widths=(38, 38, 62, 48),
            )
            pdf.add_two_col_cards(
                "Normal vs High-Turbulence Load Profiles",
                [
                    "Normal mode (1.0x): steady-state sinusoidal batching & low Gaussian ripple.",
                    "High Turbulence (1.85x): simulates heavy granulation density transients.",
                    "Both modes constrain healthy noise cleanly while letting real wear emerge.",
                ],
                "2-Tick Persistence Filter Against False Alarms",
                [
                    "Occasional 1-tick load transients (2.52-2.65 mm/s) trigger Zone C warning.",
                    "Filter requires >= 2 consecutive ticks > 2.5 mm/s before dispatching agents.",
                    "Eliminates costly nuisance stops from forklift shock or gate slams.",
                ],
            )
            pdf.add_callout(
                "info",
                "SPECTRAL PERSISTENCE & ZERO-FALSE-POSITIVE PROTOCOL",
                "Plant 04's 2-tick persistence filter rejects transient cleanroom shocks (2.52-2.65 mm/s RMS) while tripping within 3.0 seconds "
                "on genuine continuous bearing, pump, or servo degradation—protecting product quality without nuisance line stops.",
            )

            # Page 5: Cost Model, Severity, Auto-PO & 5-Why CAPA
            pdf.add_page()
            pdf.add_section_heading("5. Financial Cost Model, Damage Severity, Auto-PO & 5-Why CAPA")
            pdf.add_kpi_strip([
                ("Technician Labor", "$95.00 / hr", "Standardized cGMP rate"),
                ("Auto-PO Rule", "qty <= min", "Inserts OPEN PR in SQLite"),
                ("CAPA Trigger", ">= 2 in 30d", "Automatic 5-Why + PM shift"),
                ("Critical Impact", ">= $8,500", "QA Director escalation"),
            ])
            pdf.add_paragraph(
                "Standard Cost Equation: Total Financial Impact = Spare Part Unit Cost + Labor ($95.00/hr * Downtime/60) + "
                "(Downtime Minutes * Machine Line Rate). Any post-repair rework adds precision alignment minutes (typically +48 to +52 min)."
            )
            pdf.add_styled_table(
                headers=("Equipment ID", "Production Suite", "Line Rate ($/min)", "1st-Fix MTTR", "Typical 1st-Fix Total Impact"),
                rows=[
                    ("TP-01 Tablet Press", "Compression Suite A", "$68.00 / min", "52 - 72 min", "$3,900 - $5,600"),
                    ("RP-04 Sanitary Pump", "WFI Granulation Suite B", "$52.00 / min", "58 - 64 min (114m w/ rework)", "$3,600 - $7,200"),
                    ("PK-02 Blister Line", "Primary Packaging Suite C", "$44.00 / min", "48 - 58 min (104m w/ rework)", "$2,700 - $5,200"),
                ],
                col_widths=(38, 40, 30, 42, 36),
            )
            pdf.add_styled_table(
                headers=("Severity / Automation Rule", "Quantitative Trigger Condition", "Mandatory System Output"),
                rows=[
                    ("Low Severity", "Total impact < $2,500 AND First-Time-Fix", "Standard eBMR closure & archive"),
                    ("Medium Severity", "$2,500 <= Total impact < $5,000", "Flag for next monthly PM review"),
                    ("High Severity", "$5,000 <= Impact < $8,500 OR Gate 3 rework", "CAPA alignment review & QA Director CC"),
                    ("Critical Severity", "Impact >= $8,500 OR double stockout line halt", "Plant Director emergency escalation"),
                    ("Auto-PO Trigger", "Post-reservation stock_qty <= min_reorder_qty", "INSERT INTO purchase_requisitions (OPEN)"),
                    ("5-Why CAPA Trigger", ">= 2 incidents on same asset within 30 days", "INSERT INTO capa_records (5-Why + PM change)"),
                ],
                col_widths=(42, 72, 72),
            )
            pdf.add_two_col_cards(
                "Automated 5-Why Root-Cause CAPA Governance",
                [
                    "Queries rolling 30-day incident_analytics count per machine_id.",
                    "When >= 2 incidents detected (e.g. RP-04), drafts formal CAPA record.",
                    "Shortens PM inspection cycle (e.g. 90d -> 30d laser alignment verification).",
                    "Links verified CAPA action directly into Monthly 4-Sheet Excel tab.",
                ],
                "Automated Inventory Replenishment (Auto-PO)",
                [
                    "Triggered when stock <= min_reorder_qty (e.g. RLR-4450=1, BLT-2205=0).",
                    "Calculates optimal safety reorder quantity & supplier PO estimate.",
                    "Prevents silent spare exhaustion across 24/7 pharma manufacturing shifts.",
                    "Logs supplier lead time & PR-ID inside ALCOA+ audit summary.",
                ],
            )
            pdf.add_callout(
                "warning",
                "COST IMPACT TRANSPARENCY & REWORK BUDGET GOVERNANCE",
                "Every Gate 3B rejection automatically recalculates cumulative labor and downtime exposure before requesting incremental "
                "Human QA budget sign-off, ensuring 100% financial accountability across all maintenance rework loops.",
            )

            # Page 6: Executive Closure Package & Signoff
            pdf.add_page()
            pdf.add_section_heading("6. Executive Incident Closure Report, 4-Sheet Excel & Archival")
            pdf.add_paragraph(
                "Upon Conductor Gate 4 SIGN_OFF, the Control Tower immediately presents an Executive Incident Closure Banner right at the top "
                "of View 1 containing an interactive Vibration Recovery Chart, Itemized Cost Table, Root Cause & Orchestra Storyboard, a printable "
                "2-page Executive Incident Closure Report (.pdf), and the updated 4-sheet Monthly Quality & Maintenance Report (.xlsx)."
            )
            pdf.add_styled_table(
                headers=("Export Artifact", "Format & Structure", "Primary Consumer", "Verification Controls"),
                rows=[
                    ("Executive Closure Report", "2-Page Styled PDF + Vector Bar Chart", "Plant Director / Regulatory QA", "21 CFR Part 11 signatures + SHA-256 tail"),
                    ("Monthly Quality Workbook", "4-Sheet .xlsx (KPI, Log, CAPA/PO, Audit)", "Maintenance & Finance Leads", "Auto-styled openpyxl formulas & currency"),
                    ("Immutable SQLite Store", "data/factory_ops.db (WAL mode safe)", "Live Dashboard & BI Pipelines", "SHA-256 hash chain continuity verified"),
                ],
                col_widths=(42, 54, 42, 48),
            )
            pdf.add_section_heading("6.1 Mandatory Post-Closure Quality Archival Sequence")
            pdf.add_checklist([
                ("Verify Gate 4 Line Release", "Confirm qa_clearance reported RELEASED_TO_PRODUCTION with post-repair RMS <= 2.5 mm/s."),
                ("Reconcile Inventory & CAPA Tables", "Confirm SQLite spare_parts_inventory, purchase_requisitions, and capa_records reflect final balances."),
                ("Archive Electronic Deliverables", "Save 2-Page Incident Closure Certificate (.pdf) and regenerated Monthly .xlsx to GMP Document Management."),
            ])
            pdf.add_two_col_cards(
                "Quarterly Audit Readiness Verification",
                [
                    "QA Auditor replays random ticket_id hash sequence against GENESIS.",
                    "Cross-checks Human QA comments against budget variance approvals.",
                    "Verifies Excel KPI Summary formulas reconcile 100% against SQLite logs.",
                ],
                "Document Revision & Training Ownership",
                [
                    "Annual effectiveness review mandatory before Rev 7.0 expiration.",
                    "Updated Bilingual UI (EN/TR) certified for operator training sessions.",
                    "Electronic Part 11 signature records retained for 10 years per Annex 11.",
                ],
            )
            pdf.add_signoff_block("Director of cGMP Quality Systems")

        _build_pdf_file(p3, "GMP-QA-010", "eBMR Governance, 3-Tier Orchestra & ALCOA+", "Rev 7.0", "GMP QA PROTOCOL", _build_qa)

    return [p1, p2, p3]


def generate_guide_pdf(force: bool = False) -> Path:
    """Generate 6-page densely formatted Operator Machine Guide & 9-Fault Identification Manual."""
    DOCS_DIR.mkdir(parents=True, exist_ok=True)
    out_path = DOCS_DIR / "GUIDE-OPS-000_Control_Tower_How_To_Use_And_Problem_Identification.pdf"
    if not force and out_path.exists():
        return out_path

    def _build_guide(pdf: _GMPDocumentPdf) -> None:
        # Page 1: Pre-Shift Machine Guides & Safety Envelopes
        pdf.add_page()
        pdf.add_banner_title(
            "GUIDE-OPS-000: Machine Operation Guides, Watcher Playbook & 9-Fault Matrix",
            "Comprehensive operator manual for TP-01, RP-04, PK-02 and live Control Tower demonstration workflows.",
        )
        pdf.add_doc_control_grid([
            ("Document ID", "GUIDE-OPS-000", "Revision", "Rev 4.0 (9-Fault Orchestra Edition)"),
            ("Scope", "Operators, Engineers & QA Leads", "Standards", "ISO 20816-3 / EU GMP Annex 11 / 15"),
            ("HITL Policy", "Human permission for SQL & $ Cost", "Modes", "Live LLM (Groq/Gemini/Ollama) & Safe Demo"),
            ("Languages", "Bilingual English & Turkish UI", "Total Pages", "6 Pages (24 Pages Across Library)"),
        ])
        pdf.add_section_heading("1. Pre-Shift Machine Operation Guides & Sensor Baselines")
        pdf.add_styled_table(
            headers=("Equipment", "Pre-Shift Start Checks", "Baseline Envelope", "Immediate Auto-Trip Trigger"),
            rows=[
                (
                    "TP-01 Tablet Press (45 RPM)",
                    "Verify clean die bores, auto-lube pressure 4.2 bar, punch tip wear, cam roller play <= 0.12 mm",
                    "1.1 mm/s RMS | 0.4g | Kurt 3.0 | 48C | 19A",
                    "> 2.5 mm/s RMS, temp > 65C, amps > 24A, or cam knock",
                ),
                (
                    "RP-04 Sanitary Pump (1450 RPM)",
                    "Check basket strainer clear, WFI valves open, sterile seal flush +1.5 bar, base bolts 55 Nm",
                    "1.3 mm/s RMS | 0.5g | Kurt 3.0 | 42C | 15A",
                    "> 2.5 mm/s RMS, cavitation crackle, seal weep, soft-foot",
                ),
                (
                    "PK-02 Blister Line (120 RPM)",
                    "Clean eye-mark lens with IPA, inspect belt tooth cord, check sealing platen 160C pyrometer",
                    "1.2 mm/s RMS | 0.4g | Kurt 3.0 | 45C | 14A",
                    "> 2.5 mm/s RMS, belt squeal, index drift, platen chatter",
                ),
            ],
            col_widths=(36, 56, 46, 48),
        )
        pdf.add_kpi_strip([
            ("TP-01 Stoppage", "$68.00 / min", "45 RPM Fette Compression"),
            ("RP-04 Stoppage", "$52.00 / min", "1450 RPM WFI Utilities"),
            ("PK-02 Stoppage", "$44.00 / min", "120 RPM Blister Line"),
            ("GMP Release Limit", "<= 2.5 mm/s", "2-tick persistent trip"),
        ])
        pdf.add_two_col_cards(
            "Operator Shift Walkdown Safety Checklist",
            [
                "Verify machine Lexan interlock guards seated & green E-stop healthy.",
                "Inspect sight-glass oil levels & pneumatic air regulator setpoints.",
                "Confirm magnetic accelerometer puck cables clear of moving cam belts.",
                "Never start press or pump if active red/amber tag hangs on disconnect.",
            ],
            "Cleanroom Diagnostic & Tool Cart Staging",
            [
                "Stage 70% USP IPA squeeze bottle & shed-free polyester cleanroom wipes.",
                "Confirm NIST calibration stickers current on torque wrenches (12-65 Nm).",
                "Carry calibrated IR pyrometer & sonic belt meter on Suite C rounds.",
                "All extracted failed parts must go into red QA Quarantine bins.",
            ],
        )
        pdf.add_callout(
            "warning",
            "NEVER SILENCE OR BYPASS > 2.5 MM/S VIBRATION ALARMS WITHOUT QA CLEARANCE",
            "Resetting a tripped VFD without clearing root cause violates GMP-QA-010 and risks foreign particulate contamination.",
        )

        # Page 2: Animated UI Watcher Playbook
        pdf.add_page()
        pdf.add_section_heading("2. How to Watch & Operate the Control Tower (English & Turkish)")
        pdf.add_checklist([
            ("1. Language & Engine (Sidebar)", "Switch EN / TR anytime at top of sidebar; pick Live Autonomous LLM (Groq/Gemini/Ollama) or Offline Safe Demo."),
            ("2. Watch Always-On Oscilloscope", "Monitor live 1.5s multi-sensor table (Rad-X/Y, Ax-Z, Peak-g, Kurtosis, Temp, Amps) with Auto Fault Drift or High Variance."),
            ("3. Inject or Await Autonomous Drift", "Let Auto Fault Drift naturally degrade an asset OR select 1 of 3 fault modes inside any machine bay and click Inject."),
            ("4. Authorize Top-of-Page HITL Cards", "Approve/Deny exact SQL mutations and Repair/Rework $ budgets right under the navigation bar on any tab."),
            ("5. Watch Animated 3-Tier Orchestra", "Track pulsing nodes, flowing SVG command packets, Live Agent Thinking cards, and upward Engineer escalation curves."),
            ("6. Present Executive Final Closure Report", "Inspect KPI drop, Plotly recovery chart, itemized cost shares, and download the 2-Page PDF Report or 4-Sheet .xlsx."),
        ])
        pdf.add_callout(
            "info",
            "WATCHER-FRIENDLY REASONING & COMMAND STREAM",
            "Every agent emits a live card showing: Incoming Command -> Engineering/QA Thinking -> Tools Executed -> Verdict -> Next Handoff.",
        )
        pdf.add_styled_table(
            headers=("UI Panel / Control", "Location on Screen", "What Presenters & Audiences Should Watch"),
            rows=[
                ("Top HITL Permission Card", "Directly below 3-view tabs", "Shows exact SQL diff or $ cost estimate; workflow freezes until clicked"),
                ("Multi-Sensor Oscilloscope", "View 1 Upper Chart + Table", "Rare 1-tick blips recover cleanly; sustained 2-tick breaches trip Orchestra"),
                ("Animated 3-Tier SVG Diagram", "View 1 Mid-Page Canvas", "Blue pulse = active node; red curved arrow = Tier 1->2 escalation"),
                ("Executive Final Closure Report", "View 1 Hero Closure Card", "Displays recovery curve, cost share table, 5-step story & 2p PDF download"),
            ],
            col_widths=(48, 46, 92),
        )
        pdf.add_two_col_cards(
            "Telemetry Toggles Explained",
            [
                "Auto Fault Drift: spawns organic wear over ~11-18 ticks.",
                "3-Tick Drift Ramp: lets manual injection climb step by step.",
                "High Load Variance: tests stability under 1.85x plant load swings.",
            ],
            "Recommended Presentation Watch Pacing",
            [
                "Set Watch Animation Pacing to 0.25s-0.50s for live audience visibility.",
                "Use Offline Safe Demo Mode if classroom Wi-Fi/firewall blocks APIs.",
                "Toggle Language (English / Turkce) live in sidebar for bilingual audiences.",
            ],
        )
        pdf.add_callout(
            "tip",
            "CLASSROOM SMARTBOARD PRESENTATION MODE & HYBRID SAFETY NET",
            "Present directly from View 1's Executive Closure Report at top of screen, inspect cited PDFs side-by-side in View 2, "
            "and demonstrate zero-API-key fallback determinism anytime via the sidebar Offline Safe Demo toggle.",
        )

        # Page 3: Master 9-Fault Matrix Part 1 (TP-01 & RP-04)
        pdf.add_page()
        pdf.add_section_heading("3. Master Problem Identification Matrix: TP-01 & RP-04 (Faults 1-6)")
        pdf.add_styled_table(
            headers=("Machine & Fault", "Sensor & Spectral Signature", "Root Cause & Stock Behavior", "Verified Action & SOP Ref"),
            rows=[
                (
                    "TP-01 Bearing Spall",
                    "9.8 mm/s RMS | 184.2 Hz BPFO | 4.4g | Kurt 6.8 | 74C",
                    "Raceway fatigue spalling; OEM BRG-4402 ($485) in stock",
                    "Induction fit 110C, 45 Nm, NSF H1 grease [SOP-MNT-001 p.2]",
                ),
                (
                    "TP-01 Punch Binding",
                    "6.4 mm/s RMS | 45 Hz drag | 28.4 A surge | 61.5C",
                    "Fines baked in turret bores; lube spray starvation (LUB-3010)",
                    "IPA bore scour + LUB-3010 nozzle kit ($145) [SOP-MNT-001 p.3]",
                ),
                (
                    "TP-01 Cam Knock",
                    "8.2 mm/s RMS | 90 Hz (2x) | 5.2g | Kurt 7.4 | 68C",
                    "Cam brinelling; RLR-4450 qty=1 <= min=2 triggers Auto-PO",
                    "Replace RLR-4450 ($640) + Auto-PO, 52 Nm [SOP-MNT-001 p.4]",
                ),
                (
                    "RP-04 Seal Cavitation",
                    "9.4 mm/s RMS | 1420 Hz | 5.8g | Kurt 6.1 | 72C",
                    "Strainer block -> NPSHa drop; 1st fix hits 4.8 mm/s Gate 3 Reject",
                    "SL-8810 ($780) + Rework Laser Align <=0.05mm [SOP-MNT-002 p.2-3]",
                ),
                (
                    "RP-04 Impeller Erosion",
                    "8.9 mm/s RMS | 145 Hz (6x vane) | 3.9g | -22% head",
                    "Vane leading-edge pitting; unbalance forces > 2.5 limit",
                    "Install G2.5 IMP-8825 ($910), 0.5mm gap, 65 Nm [SOP-MNT-002 p.4]",
                ),
                (
                    "RP-04 Soft-Foot Drift",
                    "5.6 mm/s RMS | 48.3 Hz (2x axial) | 23.0 A | 66C",
                    "Corroded base shims twisting motor frame at anchor torque",
                    "304SS SHM-8805 shims ($165) + CIP offset align [SOP-MNT-002 p.5]",
                ),
            ],
            col_widths=(34, 48, 50, 54),
        )
        pdf.add_bar_chart(
            "Fault Alarm Severity Spectrum Across TP-01 & RP-04 Scenarios",
            [
                ("TP-01 BRG-4402 Spall", 9.80, (239, 68, 68)),
                ("RP-04 SL-8810 Cavitation", 9.40, (239, 68, 68)),
                ("RP-04 IMP-8825 Impeller", 8.90, (239, 68, 68)),
                ("TP-01 RLR-4450 Cam Knock", 8.20, (239, 68, 68)),
                ("TP-01 LUB-3010 Punch Bind", 6.40, (217, 119, 6)),
                ("RP-04 SHM-8805 Soft-Foot", 5.60, (217, 119, 6)),
            ],
            max_val=11.0,
            limit_val=2.5,
        )
        pdf.add_two_col_cards(
            "Quick Diagnostics: TP-01 Fault Signatures",
            [
                "High BPFO + > 65C temp -> Turret Bearing Spall (BRG-4402).",
                "28A torque surge + 45 Hz drag -> Punch Bore Binding (LUB-3010).",
                "Kurtosis > 7.0 + 90 Hz double hit -> Pre-Comp Cam Roller (RLR-4450).",
            ],
            "Quick Diagnostics: RP-04 Fault Signatures",
            [
                "1420 Hz acoustic hiss + gland weep -> Seal Cavitation (SL-8810).",
                "145 Hz blade-pass + 20% head loss -> Impeller Erosion (IMP-8825).",
                "48.3 Hz axial Z dominant + high Amps -> Motor Soft-Foot (SHM-8805).",
            ],
        )
        pdf.add_callout(
            "info",
            "TP-01 & RP-04 TRIAGE RULE: CORRELATE FREQUENCY BAND + KURTOSIS BEFORE DISPATCH",
            "Diagnostic Agent cross-checks velocity RMS alongside dominant FFT band and Kurtosis so bearing impacts, "
            "lubrication drag, seal cavitation, and motor frame twist always route to the exact SOP section on first dispatch.",
        )

        # Page 4: Master Faults 7-9 (PK-02) + Full 15-Part BOM Table
        pdf.add_page()
        pdf.add_section_heading("4. Master Matrix: PK-02 (Faults 7-9), ISO Limits & 15-Part Warehouse BOM")
        pdf.add_styled_table(
            headers=("Machine & Fault", "Sensor & Spectral Signature", "Root Cause & Stock Behavior", "Verified Action & SOP Ref"),
            rows=[
                (
                    "PK-02 Drive Belt Fatigue",
                    "7.9 mm/s RMS | 240 Hz slip | 2.9g | Kurt 4.9",
                    "Cord stretch/tooth shear; OEM BLT-2205 qty=0 -> BLT-2205-ALT",
                    "Approve BLT-2205-ALT ($225), 180-200 N, 38 Nm [SOP-MNT-001 p.5]",
                ),
                (
                    "PK-02 Servo Drift",
                    "5.1 mm/s RMS | 120 Hz jitter | +/-0.6mm index error",
                    "Elastomer coupling wear & fouled optical registration sensor",
                    "Install SRV-2240 ($520), clamp 24 Nm, zero-home [SOP-MNT-001 p.5]",
                ),
                (
                    "PK-02 Sealing Bind",
                    "7.1 mm/s RMS | 312 Hz platen | 92.4C | Kurt 5.6",
                    "Cold repair leaves 3.9 mm/s thermal bow -> Gate 3 Reject",
                    "BRG-2290 ($395) + Hot-Soak Parallel Align [SOP-MNT-001 p.6]",
                ),
            ],
            col_widths=(34, 48, 50, 54),
        )
        _iso_20816_table(pdf)
        pdf.add_section_heading("4.1 Complete Seeded Warehouse Inventory (`spare_parts_inventory`)")
        pdf.add_styled_table(
            headers=("Code", "Asset", "Qty / Min", "Bin", "Cost", "Compatible Alt / Special Rule"),
            rows=[
                ("BRG-4402", "TP-01", "6 / 2", "BIN-A14", "$485", "Alt: BRG-4402-ALT (qty 4, $455)"),
                ("LUB-3010", "TP-01", "9 / 3", "BIN-A09", "$145", "NSF H1 spray manifold kit"),
                ("RLR-4450", "TP-01", "1 / 2", "BIN-A18", "$640", "LOW STOCK -> Triggers Tier 2 Auto-PO"),
                ("SL-8810", "RP-04", "4 / 2", "BIN-B08", "$780", "Alt: SL-8810-ALT (qty 3, $820)"),
                ("IMP-8825", "RP-04", "2 / 1", "BIN-B12", "$910", "ISO 1940-1 G2.5 316L impeller"),
                ("SHM-8805", "RP-04", "10 / 3", "BIN-B05", "$165", "304SS precision motor shim pack"),
                ("BLT-2205", "PK-02", "0 / 2", "BIN-C03", "$245", "STOCKOUT -> Evaluates BLT-2205-ALT (5, $225)"),
                ("SRV-2240 / BRG-2290", "PK-02", "3/1 | 4/2", "C09/C12", "$520/$395", "Servo kit / 160C PTFE sealing bearing"),
                ("GKT-1102 / SNS-9010", "RP/TP", "18/5 | 3/1", "B02/E01", "$54/$560", "USP EPDM gaskets / Piezo accelerometer"),
            ],
            col_widths=(36, 16, 20, 20, 22, 72),
        )

        # Page 5: HITL, Auto-PO & CAPA Playbook
        pdf.add_page()
        pdf.add_section_heading("5. Human-in-the-Loop (HITL), Auto-PO & Repeat-Failure CAPA")
        pdf.add_callout(
            "warning",
            "WHY HUMAN-IN-THE-LOOP GATES PROTECT REGULATED FACTORIES",
            "AI speeds up root-cause identification across 24 pages of SOPs, but regulated GMP sites require explicit human sign-off before "
            "changing warehouse SQL balances or spending thousands in repair/rework downtime.",
        )
        pdf.add_styled_table(
            headers=("Governance Control", "Trigger Condition", "Visible Approval Payload", "Denied / Automated Outcome"),
            rows=[
                ("Gate 2A: SQL Auth", "Before decrementing spare_parts_inventory", "Exact UPDATE/INSERT SQL + bin + stock delta", "Zero DB writes; eBMR closed safely"),
                ("Gate 3A: Cost Auth", "Before Technician executes LOTO repair", "Parts $ + Labor ($95/hr) + Est. Downtime $", "Work halted; LOTO remains locked"),
                ("Gate 3A: Rework Auth", "After Gate 3 Reject (> 2.5 mm/s RMS)", "Incremental laser/thermal alignment cost $", "Held until QA approves rework"),
                ("Tier 2 Auto-PO", "Remaining stock <= min_reorder_qty", "Auto-generated PR-XXXXXX in SQLite", "Protects spare safety stock automatically"),
                ("5-Why CAPA", ">= 2 incidents on same asset in 30 days", "5-Why RCA + shortened PM interval", "Appended to capa_records & .xlsx"),
            ],
            col_widths=(34, 48, 56, 48),
        )
        pdf.add_two_col_cards(
            "QA Reviewer Guide: Approve vs Deny",
            [
                "Approve SQL only after verifying part code matches SOP citation.",
                "For OEM qty=0, verify gmp_qualified_alt = 1 on compatible alternative.",
                "Click Deny to demo safe regulatory halt (0 DB mutations occur).",
                "Enter reviewer note for permanent inclusion in SHA-256 audit row.",
            ],
            "Stage Demo Quick Reference (Key Scenarios)",
            [
                "Best Full Orchestra Demo: RP-04 Seal Cavitation (Gate 3 Laser Rework).",
                "Best Inventory Demo: PK-02 Drive Belt (Stockout -> Alt Part + Auto-PO).",
                "Best Low-Stock PO Demo: TP-01 Cam Knock (RLR-4450 qty 1 -> 0).",
                "Best Thermal Rework Demo: PK-02 Platen Bind (3.9 -> 1.5 mm/s).",
            ],
        )
        pdf.add_styled_table(
            headers=("Benchmark Scenario", "Attempts", "Est. MTTR", "Typical Total Impact", "Governance Highlights"),
            rows=[
                ("TP-01 BRG-4402 Spall", "1 (Pass 1.3)", "64 min", "~$4,938 (Medium)", "Clean Gate 2A + Gate 3A"),
                ("RP-04 SL-8810 Seal", "2 (4.8 -> 1.6)", "114 min", "~$6,881 (High)", "Escalation + Laser Rework + CAPA"),
                ("PK-02 BLT-2205 Belt", "1 (Pass 1.5)", "54 min", "~$2,687 (Medium)", "Stockout Alt + Auto-PO"),
            ],
            col_widths=(40, 28, 24, 38, 56),
        )
        pdf.add_callout(
            "info",
            "ALCOA+ AUDITABILITY ACROSS ALL HUMAN DECISIONS & REWORK LOOPS",
            "Every Approve or Deny click records the QA reviewer identity, optional written rationale, exact SQL/cost snapshot, and "
            "chained SHA-256 hash in data/factory_ops.db and the 4-sheet Excel report—preserving tamper-evident traceability.",
        )

        # Page 6: Live Demo Script, Cloud Modes & Signoff
        pdf.add_page()
        pdf.add_section_heading("6. Execution Engines, Cloud Secrets Security & Demo Checklist")
        pdf.add_styled_table(
            headers=("Execution Mode", "Provider / Model", "Best Demo Use Case", "Cloud & Offline Resilience"),
            rows=[
                ("Live LLM (Primary)", "Groq: openai/gpt-oss-120b", "Sub-second autonomous structured JSON tool calling", "Automatic 404 fallback + 429 retry"),
                ("Live LLM (Backup)", "Google Gemini: gemini-3.5-flash-lite", "Free AI Studio quota with paced calls", "Zero deprecated gemini-2.5 models"),
                ("Local LLM", "Ollama: qwen2.5:7b", "Fully air-gapped laptop stage demo", "No internet required if Ollama pulled"),
                ("Offline Safe Demo", "Deterministic Orchestra Engine", "100% deterministic classroom backup (real SQL/PDF/XLSX)", "Zero API keys or network needed"),
            ],
            col_widths=(36, 48, 56, 46),
        )
        pdf.add_section_heading("6.1 Emergency LOTO & Classroom Live Presentation Flow")
        pdf.add_checklist([
            ("Open Control Tower URL", "Open the Streamlit Cloud link on the classroom smartboard; confirm live oscilloscope ticks streaming."),
            ("Show Multilingual Agility", "Toggle Language switch (English <-> Turkce) and point out bilingual UI, SVG labels, and LLM reasoning."),
            ("Walk Through RP-04 Rework", "Trigger RP-04 Seal Cavitation; approve SQL -> Repair Cost -> watch Attempt #1 fail (4.8 mm/s) -> Escalate -> 1.6 Pass."),
            ("Export Final Evidence", "Download 2-Page Executive Closure Report (.pdf) and 4-Sheet Monthly Report (.xlsx) directly from View 1."),
        ])
        pdf.add_two_col_cards(
            "API Credential & Git Security Discipline",
            [
                "API keys live strictly in gitignored .env (local) or st.secrets (cloud).",
                "Zero hardcoded credentials in source code or public GitHub commits.",
                "Sidebar password input supports temporary live-session key override safely.",
            ],
            "Streamlit Cloud 1GB RAM Architecture Guardrails",
            [
                "PDF RAG runs pure pypdf + scikit-learn + rank-bm25 (~45 MB footprint).",
                "PDF pages cached via @st.cache_data & pypdfium2 crisp PNG rendering.",
                "SQLite operates in WAL mode to support concurrent dashboard + agent writes.",
            ],
        )
        pdf.add_signoff_block("Plant 04 Operations & QA Training Lead")

    _build_pdf_file(
        out_path,
        "GUIDE-OPS-000",
        "Machine Operation Guides & 9-Fault Problem ID",
        "Rev 4.0",
        "OPERATOR HANDBOOK",
        _build_guide,
    )
    return out_path


def generate_incident_certificate_pdf_bytes(state: Dict[str, Any]) -> bytes:
    """Render a dense, executive 2-page printable GMP Incident Closure Report PDF from MaintenanceState."""
    pdf = _GMPDocumentPdf(
        doc_code=f"CERT-{state.get('ticket_id', 'TKT')}",
        doc_title="GMP Executive Incident Closure Report",
        revision="FINAL eBMR",
        category="EXECUTIVE QA RECORD",
    )
    pdf.set_auto_page_break(auto=True, margin=13)
    pdf.alias_nb_pages()

    ticket_id = str(state.get("ticket_id", "N/A"))
    machine_id = str(state.get("machine_id", "N/A"))
    machine_name = str(state.get("machine_name", machine_id))
    final_status = str(state.get("last_gate_decision", "SIGNED_OFF"))
    closed_at = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")

    diag = state.get("diagnostic_report", {}) or {}
    inv = state.get("inventory_report", {}) or {}
    maint = state.get("maintenance_report", {}) or {}
    clear = state.get("clearance_report", {}) or {}
    analyt = state.get("analytics_report", {}) or {}
    pr_info = state.get("purchase_requisition", {}) or {}
    capa = state.get("capa_report", {}) or {}
    pre = state.get("precheck_report", {}) or {}
    replan = state.get("engineering_replan", {}) or {}

    pre_rms = float(state.get("alarm_vibration_rms", 0.0) or 0.0)
    post_rms = float(maint.get("post_repair_vibration_rms", 0.0) or 0.0)
    part_code = str(inv.get("reserved_part_code") or diag.get("required_part_code") or "N/A")
    stock_flag = "QUALIFIED_ALT" if inv.get("used_alternative") else ("OEM_IN_STOCK" if inv.get("reserved") else "HELD")
    attempts_hist = list(maint.get("attempts_history") or state.get("maintenance_attempts") or [])
    attempts = int(maint.get("attempt_count", len(attempts_hist)) or 0)
    rework_flag = "YES (Gate 3 Realign)" if attempts > 1 else ("NO (First-Time-Fix)" if attempts == 1 else "WORK HELD")
    sop_cite = f"{diag.get('sop_code', 'SOP-MNT-001')} p.{diag.get('page_number', 1)}" if diag.get("sop_code") else "N/A"
    clr_status = str(clear.get("line_status_after_clearance") or ("RELEASED" if state.get("clearance_approved") else "HOLD"))

    part_cost = float(analyt.get("part_cost_usd", inv.get("unit_cost_usd", 0.0)) or 0.0)
    labor_cost = float(analyt.get("labor_cost_usd", 0.0) or 0.0)
    loss_cost = float(analyt.get("production_loss_usd", 0.0) or 0.0)
    total_cost = float(analyt.get("total_cost_usd", 0.0) or 0.0)
    downtime_m = int(analyt.get("downtime_minutes", 0) or 0)
    mttr_m = int(analyt.get("mttr_minutes", 0) or 0)
    severity = str(analyt.get("damage_severity", "N/A"))
    pr_str = f"{pr_info.get('pr_id')} (Qty {pr_info.get('order_qty')})" if pr_info.get("created") else "Not Triggered"
    capa_id = str(capa.get("capa_id") or "Not Triggered")
    batch_risk = f"{pre.get('contamination_risk', 'LOW')} ({pre.get('active_batch_id', 'LOT')})"
    tail_hash = ((state.get("audit_trail") or [{}])[-1]).get("record_hash", "N/A")[:22]

    # ---------------- PAGE 1: Executive Overview, Vibration Recovery & Financials ----------------
    pdf.add_page()
    pdf.add_banner_title(
        f"GMP EXECUTIVE INCIDENT CLOSURE REPORT  |  {ticket_id}",
        f"Asset: {machine_name} ({machine_id})   |   Verdict: {final_status}   |   Closed UTC: {closed_at}",
    )

    drop_pct = ((pre_rms - post_rms) / pre_rms * 100.0) if (pre_rms > 0 and post_rms > 0) else 0.0
    short_clr = "RELEASED" if "RELEASED" in clr_status else clr_status[:18]
    pdf.add_kpi_strip([
        ("Final Verdict", final_status, f"Line: {short_clr}"),
        ("Vibration Drop", f"{pre_rms:.1f} -> {post_rms:.2f}", f"-{drop_pct:.0f}% vs <=2.5 limit"),
        ("Total Financial Impact", f"${total_cost:,.2f}", f"Severity: {severity}"),
        ("Downtime / Attempts", f"{downtime_m}m / {attempts}", rework_flag[:24]),
    ])

    pdf.add_doc_control_grid([
        ("Trigger Source", str(state.get("trigger_source", "EDGE_SENSOR_AUTO_TRIP")), "Execution Engine", f"{str(state.get('execution_mode', 'live')).upper()} ({state.get('llm_provider', '')})"),
        ("SOP Citation", sop_cite, "Spare Part Issued", f"{part_code} [{stock_flag}]"),
        ("Warehouse Bin", str(inv.get("bin_location", "N/A")), "Batch Hold Lot", batch_risk[:36]),
        ("Auto-PO Status", pr_str, "CAPA Reference", capa_id),
    ])

    # Vector Vibration Recovery Chart
    bars: List[Tuple[str, float, Tuple[int, int, int]]] = [
        ("Normal Asset Baseline", 1.20, (56, 189, 248)),
        ("Sensor Alarm Trip Peak", pre_rms, (239, 68, 68)),
    ]
    for att in attempts_hist[:2]:
        a_no = att.get("attempt_number", 1)
        a_rms = float(att.get("post_repair_vibration_rms", post_rms) or 0.0)
        bars.append((f"Repair Attempt #{a_no}", a_rms, (16, 185, 129) if a_rms <= GMP_VIBRATION_LIMIT_RMS else (217, 119, 6)))
    if not attempts_hist and post_rms > 0:
        bars.append(("Final Verified RMS", post_rms, (16, 185, 129) if post_rms <= GMP_VIBRATION_LIMIT_RMS else (239, 68, 68)))

    max_chart = max([b[1] for b in bars] + [3.5])
    pdf.add_section_heading("1. Vibration Recovery Telemetry Progression (ISO 20816-3)")
    pdf.add_bar_chart("Multi-Axis Composite RMS Velocity Before & After Intervention", bars, max_val=max_chart, limit_val=GMP_VIBRATION_LIMIT_RMS)

    # Two-Column Root Cause + Governance Details
    pdf.add_section_heading("2. Verified Root Cause, Engineering Prescription & Material Control")
    left_specs = [
        f"Failure Mode: {diag.get('failure_mode', state.get('alarm_type', 'N/A'))}",
        f"Root Cause: {diag.get('root_cause', 'Verified per SOP RAG and sensor telemetry.')}",
        f"SOP Reference: {sop_cite} ({diag.get('sop_filename', 'SOP PDF')[:34]})",
    ]
    if replan:
        left_specs.append(f"Escalation Spec ({replan.get('sop_ref')}): {replan.get('corrective_procedure')}")
    else:
        left_specs.append(f"Tooling Issued: {', '.join(str(x) for x in (diag.get('required_tools') or [])[:3])}")

    right_specs = [
        f"Requested -> Issued: {diag.get('required_part_code', part_code)} -> {part_code} ({stock_flag})",
        f"Warehouse Location: {inv.get('bin_location', 'N/A')} | Remaining Qty: {inv.get('remaining_stock_qty', 'N/A')}",
        f"Batch Contamination Assessment: {batch_risk}",
        f"CAPA Action: {capa.get('preventive_action', '30d asset count below CAPA threshold.')[:125]}",
    ]
    pdf.add_two_col_cards(
        "Engineering Diagnostics & Realignment",
        left_specs,
        "Material Traceability, Batch Hold & CAPA",
        right_specs,
    )

    pdf.add_section_heading("3. Itemized Financial Impact Accounting (Plant 04 Standard Cost Model)")
    pct_part = (part_cost / total_cost * 100.0) if total_cost > 0 else 0.0
    pct_lab = (labor_cost / total_cost * 100.0) if total_cost > 0 else 0.0
    pct_loss = (loss_cost / total_cost * 100.0) if total_cost > 0 else 0.0
    pdf.add_styled_table(
        headers=("Cost Component", "Basis / Rate", "Share of Impact", "Amount (USD)"),
        rows=[
            ("Spare Part Material", f"1x {part_code} ({stock_flag})", f"{pct_part:.1f}%", f"${part_cost:,.2f}"),
            ("Technician Labor", f"{downtime_m} min @ $95.00/hr rate", f"{pct_lab:.1f}%", f"${labor_cost:,.2f}"),
            ("Lost Line Production", f"{machine_id} line stoppage rate", f"{pct_loss:.1f}%", f"${loss_cost:,.2f}"),
            ("TOTAL FINANCIAL IMPACT", f"Severity: {severity} | MTTR: {mttr_m}m", "100.0%", f"${total_cost:,.2f}"),
        ],
        col_widths=(48, 68, 34, 36),
        row_fills=[
            (248, 250, 252),
            (255, 255, 255),
            (248, 250, 252),
            (239, 246, 255),
        ],
    )

    # ---------------- PAGE 2: 6-Gate Matrix, Orchestra Ledger, Line Clearance & Signoff ----------------
    pdf.add_page()
    pdf.add_section_heading("4. Mandatory Six-Gate GMP Quality Verification Matrix")
    hdec = state.get("human_decisions", {}) or {}
    g2a_status = hdec.get("sql_reservation", {}).get("decision", "AUTO_APPROVED" if not state.get("hitl_enabled", True) else "PENDING")
    g3a_status = hdec.get("repair_attempt_1", {}).get("decision", "AUTO_APPROVED" if not state.get("hitl_enabled", True) else "PENDING")
    rework_st = "REWORK_PASSED" if attempts > 1 else ("FIRST_FIX_PASS" if post_rms > 0 and post_rms <= GMP_VIBRATION_LIMIT_RMS else final_status)
    pdf.add_styled_table(
        headers=("Quality Gate", "Governing Authority", "Verification Evidence", "Final State"),
        rows=[
            ("Gate 1: Diagnostic & RAG", "Tier 3 Conductor", f"Cited {sop_cite}; Root Cause & {part_code}", "APPROVED" if state.get("diagnostic_approved") else final_status),
            ("Gate 2A: SQL Mutation Auth", "Human QA Lead", f"Reserved 1x {part_code} from {inv.get('bin_location', 'N/A')}", str(g2a_status)),
            ("Gate 2B: Material / Auto-PO", "Tier 3 Conductor", f"Stock: {stock_flag} | PR: {pr_str[:22]}", "APPROVED" if state.get("stock_approved") else final_status),
            ("Gate 3A: Budget Auth", "Human QA Lead", f"Authorized Parts + Labor + Stoppage Estimate", str(g3a_status)),
            ("Gate 3B: Vibration Gate", "Tier 3 Conductor", f"Verified {post_rms:.2f} mm/s vs <= {GMP_VIBRATION_LIMIT_RMS} limit", rework_st),
            ("Gate 4: Closure & Analytics", "Tier 3 Conductor", f"Clearance: {clr_status} | Hash: {tail_hash[:12]}", final_status),
        ],
        col_widths=(44, 34, 72, 36),
    )

    pdf.add_section_heading("5. Chronological Multi-Agent Orchestra & Audit Trail")
    audit_rows = list(state.get("audit_trail") or [])[-9:]
    table_audit = []
    for row in audit_rows:
        table_audit.append((
            f"#{row.get('step_number', '')}",
            str(row.get("timestamp_utc", ""))[-9:],
            str(row.get("agent_name", ""))[:24],
            str(row.get("gate_decision", ""))[:14],
            f"{str(row.get('action_summary', ''))[:42]} [{str(row.get('record_hash', ''))[:10]}]",
        ))
    if not table_audit:
        table_audit.append(("#1", closed_at[-9:], "System", final_status, f"Ticket {ticket_id} summary"))
    pdf.add_styled_table(
        headers=("Step", "UTC Time", "Tier / Actor", "Gate", "Action Summary & SHA-256 Prefix"),
        rows=table_audit,
        col_widths=(15, 22, 41, 32, 76),
    )

    clr_list = list(clear.get("checklist") or [])
    if clr_list:
        clr_items = [
            (
                str(c.get("check", "Check")),
                f"{c.get('status', 'PASS')} - {c['notes']}" if c.get("notes") else f"{c.get('status', 'PASS')} (Verified per Annex 15)",
            )
            for c in clr_list[:4]
        ]
    else:
        clr_items = [
            ("LOTO Padlock & Tag Removal", f"Verified zero-energy clearance & 480V isolation restoration on {machine_id}."),
            ("100% Tool Accountability", "All torque wrenches, pullers, and alignment shims accounted for before start."),
            ("Sanitary / IPA Wipe & Stability", f"Verified final RMS {post_rms:.2f} mm/s <= {GMP_VIBRATION_LIMIT_RMS} mm/s cGMP limit."),
        ]
    pdf.add_section_heading("6. Tier 3 QA Line Clearance Verification & Release Sign-Off")
    pdf.add_checklist(clr_items)
    pdf.add_signoff_block("GMP Quality Supervisor & Plant QA Lead")
    return bytes(pdf.output())


def _compute_audit_hash(
    prev_hash: str,
    ticket_id: str,
    step_number: int,
    timestamp_utc: str,
    agent_name: str,
    action_summary: str,
    gate_decision: str,
    audit_notes: str,
) -> str:
    payload = "|".join(
        [
            prev_hash,
            ticket_id,
            str(step_number),
            timestamp_utc,
            agent_name,
            action_summary,
            gate_decision,
            audit_notes,
        ]
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def init_database(force_reset: bool = False) -> Path:
    """Create and seed data/factory_ops.db with schema, parts, and 8 monthly incidents."""
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)

    conn = sqlite3.connect(str(DB_PATH), timeout=15.0)
    try:
        conn.execute("PRAGMA foreign_keys = ON;")
        cur = conn.cursor()

        if force_reset:
            # Drop tables in-place instead of unlinking the SQLite file to avoid Windows WinError 32 locks
            cur.executescript(
                """
                DROP TABLE IF EXISTS capa_records;
                DROP TABLE IF EXISTS purchase_requisitions;
                DROP TABLE IF EXISTS gmp_audit_log;
                DROP TABLE IF EXISTS incident_analytics;
                DROP TABLE IF EXISTS part_reservations;
                DROP TABLE IF EXISTS spare_parts_inventory;
                DROP TABLE IF EXISTS machines;
                """
            )
            conn.commit()

        cur.executescript(
            """
            CREATE TABLE IF NOT EXISTS machines (
                machine_id TEXT PRIMARY KEY,
                machine_name TEXT NOT NULL,
                area TEXT NOT NULL,
                nominal_rpm INTEGER NOT NULL,
                baseline_vibration_rms REAL NOT NULL,
                gmp_vibration_limit_rms REAL NOT NULL,
                downtime_cost_per_min REAL NOT NULL
            );

            CREATE TABLE IF NOT EXISTS spare_parts_inventory (
                part_code TEXT PRIMARY KEY,
                part_name TEXT NOT NULL,
                machine_id TEXT NOT NULL,
                stock_qty INTEGER NOT NULL,
                min_reorder_qty INTEGER NOT NULL,
                bin_location TEXT NOT NULL,
                unit_cost_usd REAL NOT NULL,
                compatible_alt_code TEXT,
                gmp_qualified_alt INTEGER NOT NULL DEFAULT 1,
                notes TEXT
            );

            CREATE TABLE IF NOT EXISTS part_reservations (
                reservation_id INTEGER PRIMARY KEY AUTOINCREMENT,
                ticket_id TEXT NOT NULL,
                part_code TEXT NOT NULL,
                quantity INTEGER NOT NULL,
                unit_cost_usd REAL NOT NULL,
                bin_location TEXT NOT NULL,
                reserved_at TEXT NOT NULL,
                is_alternative INTEGER NOT NULL DEFAULT 0
            );

            CREATE TABLE IF NOT EXISTS incident_analytics (
                ticket_id TEXT PRIMARY KEY,
                incident_date TEXT NOT NULL,
                machine_id TEXT NOT NULL,
                machine_name TEXT NOT NULL,
                failure_mode TEXT NOT NULL,
                sop_ref TEXT NOT NULL,
                part_used TEXT NOT NULL,
                used_alternative INTEGER NOT NULL DEFAULT 0,
                alarm_vibration_rms REAL NOT NULL,
                final_vibration_rms REAL NOT NULL,
                repair_attempts INTEGER NOT NULL,
                first_time_fix INTEGER NOT NULL,
                downtime_minutes INTEGER NOT NULL,
                part_cost_usd REAL NOT NULL,
                labor_cost_usd REAL NOT NULL,
                production_loss_usd REAL NOT NULL,
                total_cost_usd REAL NOT NULL,
                damage_severity TEXT NOT NULL,
                orchestrator_status TEXT NOT NULL,
                created_at TEXT NOT NULL
            );

            CREATE TABLE IF NOT EXISTS gmp_audit_log (
                audit_id INTEGER PRIMARY KEY AUTOINCREMENT,
                ticket_id TEXT NOT NULL,
                step_number INTEGER NOT NULL,
                timestamp_utc TEXT NOT NULL,
                agent_name TEXT NOT NULL,
                action_summary TEXT NOT NULL,
                gate_decision TEXT NOT NULL,
                audit_notes TEXT NOT NULL,
                prev_hash TEXT NOT NULL,
                record_hash TEXT NOT NULL
            );

            CREATE TABLE IF NOT EXISTS purchase_requisitions (
                pr_id TEXT PRIMARY KEY,
                ticket_id TEXT NOT NULL,
                part_code TEXT NOT NULL,
                part_name TEXT NOT NULL,
                current_stock_qty INTEGER NOT NULL,
                min_reorder_qty INTEGER NOT NULL,
                order_qty INTEGER NOT NULL,
                est_order_cost_usd REAL NOT NULL,
                status TEXT NOT NULL,
                created_at TEXT NOT NULL
            );

            CREATE TABLE IF NOT EXISTS capa_records (
                capa_id TEXT PRIMARY KEY,
                ticket_id TEXT NOT NULL,
                machine_id TEXT NOT NULL,
                machine_name TEXT NOT NULL,
                repeat_count_30d INTEGER NOT NULL,
                problem_statement TEXT NOT NULL,
                five_why_root_cause TEXT NOT NULL,
                preventive_action TEXT NOT NULL,
                recommended_pm_interval TEXT NOT NULL,
                owner_role TEXT NOT NULL,
                status TEXT NOT NULL,
                created_at TEXT NOT NULL
            );
            """
        )

        if cur.execute("SELECT COUNT(*) FROM machines").fetchone()[0] == 0:
            cur.executemany(
                """
                INSERT INTO machines (
                    machine_id, machine_name, area, nominal_rpm,
                    baseline_vibration_rms, gmp_vibration_limit_rms, downtime_cost_per_min
                ) VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                [
                    ("TP-01", "TP-01 Rotary Tablet Press (Fette-Class)", "Compression Suite A", 45, 1.1, 2.5, 68.0),
                    ("RP-04", "RP-04 Sanitary Centrifugal Pump (3-A)", "WFI & Granulation Suite B", 1450, 1.3, 2.5, 52.0),
                    ("PK-02", "PK-02 Servo Blister Packaging Line", "Primary Packaging Suite C", 120, 1.2, 2.5, 44.0),
                ],
            )

        if cur.execute("SELECT COUNT(*) FROM spare_parts_inventory").fetchone()[0] == 0:
            cur.executemany(
                """
                INSERT INTO spare_parts_inventory (
                    part_code, part_name, machine_id, stock_qty, min_reorder_qty,
                    bin_location, unit_cost_usd, compatible_alt_code, gmp_qualified_alt, notes
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                [
                    (
                        "BRG-4402",
                        "SKF Explorer 22214 E Spherical Roller Bearing",
                        "TP-01",
                        6,
                        2,
                        "BIN-A14",
                        485.00,
                        "BRG-4402-ALT",
                        1,
                        "OEM turret bearing per SOP-MNT-001 Sec 2 (p.2)",
                    ),
                    (
                        "BRG-4402-ALT",
                        "FAG X-life Heavy-Duty Spherical Roller Bearing",
                        "TP-01",
                        4,
                        2,
                        "BIN-A15",
                        455.00,
                        None,
                        1,
                        "GMP-qualified alternative to BRG-4402",
                    ),
                    (
                        "LUB-3010",
                        "NSF H1 Automatic Turret Spray & Punch Lubrication Kit",
                        "TP-01",
                        9,
                        3,
                        "BIN-A09",
                        145.00,
                        None,
                        1,
                        "Food-grade pneumatic auto-lube manifold per SOP-MNT-001 p.3",
                    ),
                    (
                        "RLR-4450",
                        "Pre-Compression Cam Track & Pressure Roller Assembly",
                        "TP-01",
                        1,
                        2,
                        "BIN-A18",
                        640.00,
                        "RLR-4450-ALT",
                        1,
                        "LOW STOCK DEMO (qty=1 <= min=2): triggers Tier 2 Auto-PO on reserve",
                    ),
                    (
                        "RLR-4450-ALT",
                        "Precision Chrome-Carbide Cam Follower Roller Kit",
                        "TP-01",
                        2,
                        1,
                        "BIN-A19",
                        610.00,
                        None,
                        1,
                        "GMP-qualified alternative to RLR-4450",
                    ),
                    (
                        "SL-8810",
                        "John Crane Type 5620 Sanitary Cartridge Seal (SiC/Viton)",
                        "RP-04",
                        4,
                        2,
                        "BIN-B08",
                        780.00,
                        "SL-8810-ALT",
                        1,
                        "FDA/3-A sanitary cartridge seal per SOP-MNT-002 p.2",
                    ),
                    (
                        "SL-8810-ALT",
                        "EagleBurgmann Cartex-DN Sanitary Cartridge Seal",
                        "RP-04",
                        3,
                        1,
                        "BIN-B09",
                        820.00,
                        None,
                        1,
                        "GMP-qualified alternative to SL-8810 (EN 10204 3.1)",
                    ),
                    (
                        "IMP-8825",
                        "316L Electropolished Open Sanitary Impeller (G2.5 Balanced)",
                        "RP-04",
                        2,
                        1,
                        "BIN-B12",
                        910.00,
                        None,
                        1,
                        "ISO 1940-1 G2.5 balanced sanitary impeller per SOP-MNT-002 p.4",
                    ),
                    (
                        "SHM-8805",
                        "Laser-Cut 304SS Motor Soft-Foot Shim & Anchor Bolt Kit",
                        "RP-04",
                        10,
                        3,
                        "BIN-B05",
                        165.00,
                        None,
                        1,
                        "Precision motor alignment shim pack per SOP-MNT-002 p.5",
                    ),
                    (
                        "BLT-2205",
                        "Gates Poly Chain GT Carbon 8MGT-2200 Drive Belt",
                        "PK-02",
                        0,
                        2,
                        "BIN-C03",
                        245.00,
                        "BLT-2205-ALT",
                        1,
                        "STOCKOUT DEMO: OEM belt depleted; evaluate BLT-2205-ALT",
                    ),
                    (
                        "BLT-2205-ALT",
                        "Optibelt Delta Chain Carbon 8MGT-2200 Belt",
                        "PK-02",
                        5,
                        2,
                        "BIN-C04",
                        225.00,
                        None,
                        1,
                        "GMP/FDA qualified compatible alternative to BLT-2205",
                    ),
                    (
                        "SRV-2240",
                        "Brushless Servo Coupling Spider & Optical Eye-Mark Kit",
                        "PK-02",
                        3,
                        1,
                        "BIN-C09",
                        520.00,
                        None,
                        1,
                        "Zero-backlash servo drive kit per SOP-MNT-001 p.5",
                    ),
                    (
                        "BRG-2290",
                        "Heated Platen Roller High-Temp PTFE Pillow Bearing",
                        "PK-02",
                        4,
                        2,
                        "BIN-C12",
                        395.00,
                        None,
                        1,
                        "160C sealing platen pillow bearing per SOP-MNT-001 p.6",
                    ),
                    (
                        "GKT-1102",
                        "EPDM Sanitary Tri-Clamp Flange Gasket Kit (USP Class VI)",
                        "RP-04",
                        18,
                        5,
                        "BIN-B02",
                        54.00,
                        None,
                        1,
                        "USP Class VI sanitary consumable",
                    ),
                    (
                        "SNS-9010",
                        "Piezo Tri-Axial Accelerometer & Eye-Mark Sensor Kit",
                        "TP-01",
                        3,
                        1,
                        "BIN-E01",
                        560.00,
                        None,
                        1,
                        "ISO 20816-3 calibrated diagnostic sensor",
                    ),
                ],
            )

        if cur.execute("SELECT COUNT(*) FROM incident_analytics").fetchone()[0] == 0:
            now_utc = datetime.now(timezone.utc)
            ym = now_utc.strftime("%Y-%m")
            max_day = max(1, now_utc.day)
            raw_days = [1, 3, 5, 7, 10, 13, 16, 18]
            seed_rows = [
                (
                    f"INC-{ym.replace('-', '')}-001",
                    "TP-01",
                    "TP-01 Rotary Tablet Press (Fette-Class)",
                    "Punch Binding from Granulation Fines in Turret Bores",
                    "SOP-MNT-001 p.2",
                    "GKT-1102",
                    0,
                    3.8,
                    1.1,
                    1,
                    1,
                    32,
                    54.0,
                    142.5,
                    2176.0,
                    2372.5,
                    "Low",
                    "SIGNED_OFF",
                ),
                (
                    f"INC-{ym.replace('-', '')}-002",
                    "RP-04",
                    "RP-04 Sanitary Centrifugal Pump (3-A)",
                    "WFI Tri-Clamp Suction Gasket Weep Post-SIP",
                    "SOP-MNT-002 p.1",
                    "GKT-1102",
                    0,
                    3.4,
                    1.3,
                    1,
                    1,
                    35,
                    54.0,
                    142.5,
                    1820.0,
                    2016.5,
                    "Low",
                    "SIGNED_OFF",
                ),
                (
                    f"INC-{ym.replace('-', '')}-003",
                    "PK-02",
                    "PK-02 Servo Blister Packaging Line",
                    "Optical Eye-Mark Registration Sensor Fouling",
                    "SOP-MNT-001 p.3",
                    "GKT-1102",
                    0,
                    4.2,
                    1.3,
                    1,
                    1,
                    40,
                    54.0,
                    142.5,
                    1760.0,
                    1956.5,
                    "Low",
                    "SIGNED_OFF",
                ),
                (
                    f"INC-{ym.replace('-', '')}-004",
                    "TP-01",
                    "TP-01 Rotary Tablet Press (Fette-Class)",
                    "Accelerometer Cable Micro-Fracture & Signal Drift",
                    "SOP-MNT-001 p.1",
                    "SNS-9010",
                    0,
                    5.9,
                    1.1,
                    1,
                    1,
                    55,
                    560.0,
                    142.5,
                    3740.0,
                    4442.5,
                    "Medium",
                    "SIGNED_OFF",
                ),
                (
                    f"INC-{ym.replace('-', '')}-005",
                    "RP-04",
                    "RP-04 Sanitary Centrifugal Pump (3-A)",
                    "Pump Cavitation & Coupling Laser Realignment",
                    "SOP-MNT-002 p.3",
                    "SL-8810",
                    0,
                    8.7,
                    1.6,
                    2,
                    0,
                    110,
                    780.0,
                    332.5,
                    5720.0,
                    6832.5,
                    "High",
                    "SIGNED_OFF",
                ),
                (
                    f"INC-{ym.replace('-', '')}-006",
                    "PK-02",
                    "PK-02 Servo Blister Packaging Line",
                    "Sealing Roller Thermocouple Drift & Channel Leak",
                    "SOP-MNT-001 p.3",
                    "BLT-2205",
                    0,
                    6.4,
                    1.4,
                    1,
                    1,
                    50,
                    245.0,
                    142.5,
                    2200.0,
                    2587.5,
                    "Medium",
                    "SIGNED_OFF",
                ),
                (
                    f"INC-{ym.replace('-', '')}-007",
                    "TP-01",
                    "TP-01 Rotary Tablet Press (Fette-Class)",
                    "Compression Roller Shaft Bearing Pitting",
                    "SOP-MNT-001 p.2",
                    "BRG-4402",
                    0,
                    9.1,
                    1.3,
                    1,
                    1,
                    68,
                    485.0,
                    142.5,
                    4624.0,
                    5251.5,
                    "High",
                    "SIGNED_OFF",
                ),
                (
                    f"INC-{ym.replace('-', '')}-008",
                    "RP-04",
                    "RP-04 Sanitary Centrifugal Pump (3-A)",
                    "Cartridge Seal Viton O-Ring Hardening (CIP Shock)",
                    "SOP-MNT-002 p.2",
                    "SL-8810",
                    0,
                    7.8,
                    1.4,
                    1,
                    1,
                    60,
                    780.0,
                    142.5,
                    3120.0,
                    4042.5,
                    "Medium",
                    "SIGNED_OFF",
                ),
            ]
            for idx, row in enumerate(seed_rows):
                day = min(raw_days[idx], max_day)
                inc_date = f"{ym}-{day:02d}"
                created_at = f"{inc_date}T10:{10 + idx:02d}:00Z"
                cur.execute(
                    """
                    INSERT INTO incident_analytics (
                        ticket_id, incident_date, machine_id, machine_name, failure_mode,
                        sop_ref, part_used, used_alternative, alarm_vibration_rms,
                        final_vibration_rms, repair_attempts, first_time_fix,
                        downtime_minutes, part_cost_usd, labor_cost_usd,
                        production_loss_usd, total_cost_usd, damage_severity,
                        orchestrator_status, created_at
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        row[0],
                        inc_date,
                        row[1],
                        row[2],
                        row[3],
                        row[4],
                        row[5],
                        row[6],
                        row[7],
                        row[8],
                        row[9],
                        row[10],
                        row[11],
                        row[12],
                        row[13],
                        row[14],
                        row[15],
                        row[16],
                        row[17],
                        created_at,
                    ),
                )

        if cur.execute("SELECT COUNT(*) FROM capa_records").fetchone()[0] == 0:
            now_utc = datetime.now(timezone.utc)
            ym = now_utc.strftime("%Y-%m")
            ts_capa = now_utc.strftime("%Y-%m-%dT10:25:00Z")
            cur.execute(
                """
                INSERT INTO capa_records (
                    capa_id, ticket_id, machine_id, machine_name, repeat_count_30d,
                    problem_statement, five_why_root_cause, preventive_action,
                    recommended_pm_interval, owner_role, status, created_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    f"CAPA-{ym.replace('-', '')}-001",
                    f"INC-{ym.replace('-', '')}-005",
                    "RP-04",
                    "RP-04 Sanitary Centrifugal Pump (3-A)",
                    2,
                    "Recurrent RP-04 sanitary seal wear and post-repair coupling vibration > 2.5 mm/s RMS.",
                    "5-Why: Seal chatter -> Cavitation + coupling offset -> Cold alignment omitted thermal CIP growth.",
                    "Mandate pre-repair NPSHa strainer flush and dual-beam cold laser alignment <= 0.05 mm per SOP-MNT-002 p.3.",
                    "Tighten laser alignment check from 90 days to 30 days",
                    "Reliability Engineer & Fluid Maintenance Lead",
                    "ACTIVE_MONITORING",
                    ts_capa,
                ),
            )

        if cur.execute("SELECT COUNT(*) FROM gmp_audit_log").fetchone()[0] == 0:
            ts = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
            tkt = "SYSTEM-BOOT"
            act = "Initialized GMP-QA-010 eBMR database & ISO 20816-3 baseline"
            gate = "SIGNED_OFF"
            notes = "8 historical monthly incidents seeded; BLT-2205 stock=0 for stockout scenario."
            h = _compute_audit_hash("GENESIS", tkt, 0, ts, "System", act, gate, notes)
            cur.execute(
                """
                INSERT INTO gmp_audit_log (
                    ticket_id, step_number, timestamp_utc, agent_name,
                    action_summary, gate_decision, audit_notes, prev_hash, record_hash
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (tkt, 0, ts, "System", act, gate, notes, "GENESIS", h),
            )

        conn.commit()
    finally:
        conn.close()
    return DB_PATH


_SEED_LOCK = threading.Lock()


def ensure_seed_data() -> None:
    """Idempotent startup seed check: generates missing PDFs or DB tables."""
    with _SEED_LOCK:
        generate_sop_pdfs(force=False)
        generate_guide_pdf(force=False)
        init_database(force_reset=False)


def reset_demo_environment() -> None:
    """Clean full reset for live stage demonstrations (Windows file-lock safe)."""
    generate_sop_pdfs(force=True)
    generate_guide_pdf(force=True)
    init_database(force_reset=True)
    if REPORTS_DIR.exists():
        for f in REPORTS_DIR.glob("*.xlsx"):
            try:
                f.unlink()
            except OSError:
                pass
    try:
        root_str = str(BASE_DIR)
        if root_str not in sys.path:
            sys.path.insert(0, root_str)
        from src.tools import generate_monthly_excel_report_impl, invalidate_rag_cache

        invalidate_rag_cache()
        generate_monthly_excel_report_impl()
    except Exception:
        pass


if __name__ == "__main__":
    reset_demo_environment()
    print(f"Seeded SOP PDFs in: {SOPS_DIR}")
    print(f"Seeded Guide PDF in: {DOCS_DIR}")
    print(f"Seeded SQLite DB at: {DB_PATH}")
