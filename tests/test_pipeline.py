"""Verification suite for Autonomous Drift, Bilingual EN/TR, 9-Fault 24p RAG, Parallel 3-Tier Orchestra, HITL & PDF Cert."""

from __future__ import annotations

import io
import json
import sqlite3
import sys
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import openpyxl
import pypdfium2 as pdfium
from langchain_core.messages import AIMessage
from pypdf import PdfReader


def _measure_pdf_fill(pdf_source: object) -> list[float]:
    """Return lowest inked vertical % (body [6%..94%]) per page for PDF layout density checks."""
    doc = pdfium.PdfDocument(pdf_source)
    fills: list[float] = []
    try:
        for idx in range(len(doc)):
            pil_img = doc[idx].render(scale=1.2).to_pil().convert("RGB")
            arr = np.asarray(pil_img)
            h = arr.shape[0]
            row_ink = np.any(np.any(arr < 246, axis=2), axis=1)
            body_rows = np.where(row_ink[int(h * 0.06) : int(h * 0.94)])[0]
            fills.append((int(h * 0.06) + int(body_rows[-1])) / h * 100.0 if len(body_rows) else 0.0)
    finally:
        doc.close()
    return fills

ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from src.agents import _enforce_orchestra_interlocks
from src.graph import apply_human_decision, build_maintenance_graph, run_maintenance_workflow
from src.i18n import SCENARIO_LABELS_TR, localize_zone, t
from src.llm_config import (
    PROVIDER_CATALOG,
    detect_key_provider_mismatch,
    fetch_live_models,
    invoke_with_retry,
    is_model_not_found_error,
)
from src.seed_data import DB_PATH, DOCS_DIR, SOPS_DIR, reset_demo_environment
from src.state import DEMO_SCENARIOS, OrchestratorDecision, create_initial_state
from src.telemetry import SimulatedPlantTelemetryAdapter
from src.tools import (
    HybridSOPRetriever,
    build_incident_certificate_pdf,
    check_alternative_parts_impl,
    estimate_repair_cost,
    generate_monthly_excel_report_impl,
    get_audit_log_dataframe,
    get_capa_dataframe,
    get_inventory_dataframe,
    get_monthly_incidents_dataframe,
    get_purchase_requisitions_dataframe,
    list_controlled_documents,
    query_inventory_sql_impl,
    render_pdf_page_png,
)


class TestQualityAgentPipeline(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        reset_demo_environment()

    def setUp(self) -> None:
        reset_demo_environment()

    def test_01_always_on_telemetry_variance_progressive_trip_and_auto_drift(self) -> None:
        adapter = SimulatedPlantTelemetryAdapter(seed=42, auto_anomaly_enabled=False)
        for _ in range(5):
            self.assertIsNone(adapter.step_tick())
        latest = adapter.get_latest_all()
        for mid in ("TP-01", "RP-04", "PK-02"):
            self.assertLess(latest[mid]["rms"], 2.5)
            self.assertIn("radial_x", latest[mid])
            self.assertIn("radial_y", latest[mid])
            self.assertIn("axial_z", latest[mid])
            self.assertIn("peak_g", latest[mid])
            self.assertIn("kurtosis", latest[mid])
            self.assertIn("motor_amps", latest[mid])
            self.assertIn("zone", latest[mid])

        adapter.set_turbulence_mode("high")
        for _ in range(5):
            self.assertIsNone(adapter.step_tick())
            self.assertLess(adapter.get_latest_all()["TP-01"]["rms"], 2.5)

        self.assertIsNone(adapter.inject_fault("scenario_2", progressive=True))
        t1 = adapter.step_tick()
        t2 = adapter.step_tick()
        t3 = adapter.step_tick()
        self.assertIsNone(t1)
        self.assertIsNone(t2)
        self.assertEqual(t3, "scenario_2")
        self.assertGreater(adapter.get_latest_all()["RP-04"]["rms"], 4.5)

        adapter.mark_repaired("RP-04", 1.6)
        self.assertLessEqual(adapter.get_latest_all()["RP-04"]["rms"], 2.5)

        # Autonomous drift mode must emerge a real fault naturally without manual inject_fault()
        auto_adapter = SimulatedPlantTelemetryAdapter(seed=7, auto_anomaly_enabled=True)
        tripped = None
        for _ in range(25):
            tripped = auto_adapter.step_tick()
            if tripped:
                break
        self.assertIn(tripped, DEMO_SCENARIOS)

    def test_02_seed_6page_pdfs_guide_and_15part_sqlite(self) -> None:
        pdfs = sorted(SOPS_DIR.glob("*.pdf"))
        self.assertEqual(len(pdfs), 3)
        for p in pdfs:
            self.assertEqual(len(PdfReader(str(p)).pages), 6, f"{p.name} should have 6 pages")
            fills = _measure_pdf_fill(str(p))
            for pg_i, pct in enumerate(fills, start=1):
                self.assertGreaterEqual(pct, 70.0, f"{p.name} p.{pg_i} fill {pct:.1f}% is below 70%")

        guide_pdfs = sorted(DOCS_DIR.glob("*.pdf"))
        self.assertEqual(len(guide_pdfs), 1)
        self.assertEqual(len(PdfReader(str(guide_pdfs[0])).pages), 6)
        for pg_i, pct in enumerate(_measure_pdf_fill(str(guide_pdfs[0])), start=1):
            self.assertGreaterEqual(pct, 70.0, f"GUIDE-OPS-000 p.{pg_i} fill {pct:.1f}% is below 70%")

        docs = list_controlled_documents()
        self.assertEqual(len(docs), 4)
        self.assertEqual(sum(d["pages"] for d in docs), 24)
        self.assertTrue(render_pdf_page_png(docs[0]["path"], page_number=1).startswith(b"\x89PNG"))

        sop1_path = SOPS_DIR / "SOP-MNT-001_TabletPress_Vibration_Troubleshooting.pdf"
        sop1_text = "\n".join(pg.extract_text() or "" for pg in PdfReader(str(sop1_path)).pages)
        self.assertIn("20816-3", sop1_text)
        self.assertIn("BRG-4402", sop1_text)
        self.assertIn("BRG-2290", sop1_text)

        df_inv = get_inventory_dataframe()
        self.assertEqual(len(df_inv), 15)
        inv_map = df_inv.set_index("part_code")["stock_qty"].to_dict()
        self.assertEqual(inv_map["BRG-4402"], 6)
        self.assertEqual(inv_map["LUB-3010"], 9)
        self.assertEqual(inv_map["RLR-4450"], 1)
        self.assertEqual(inv_map["IMP-8825"], 2)
        self.assertEqual(inv_map["SHM-8805"], 10)
        self.assertEqual(inv_map["BLT-2205"], 0)
        self.assertEqual(inv_map["BLT-2205-ALT"], 5)
        self.assertEqual(inv_map["SRV-2240"], 3)
        self.assertEqual(inv_map["BRG-2290"], 4)
        self.assertEqual(len(get_monthly_incidents_dataframe()), 8)
        self.assertEqual(len(get_capa_dataframe()), 1)

    def test_03_windows_safe_reset_with_open_connection(self) -> None:
        conn = sqlite3.connect(str(DB_PATH), timeout=15.0)
        try:
            cnt = conn.execute("SELECT COUNT(*) FROM spare_parts_inventory").fetchone()[0]
            self.assertEqual(cnt, 15)
            reset_demo_environment()
        finally:
            conn.close()
        self.assertEqual(len(get_monthly_incidents_dataframe()), 8)

    def test_04_hybrid_tfidf_bm25_rag_citations_all_9_faults(self) -> None:
        retriever = HybridSOPRetriever(SOPS_DIR)
        queries = [
            ("TP-01 turret bearing vibration spalling BRG-4402", "SOP-MNT-001", 2),
            ("TP-01 punch binding auto-lube spray starvation LUB-3010", "SOP-MNT-001", 3),
            ("TP-01 pre-compression cam roller knock RLR-4450", "SOP-MNT-001", 4),
            ("RP-04 cartridge mechanical seal failure cavitation SL-8810", "SOP-MNT-002", 2),
            ("RP-04 impeller erosion dynamic unbalance blade pass IMP-8825", "SOP-MNT-002", 4),
            ("RP-04 motor soft-foot distortion alignment drift SHM-8805", "SOP-MNT-002", 5),
            ("PK-02 synchronous drive belt tooth fatigue BLT-2205", "SOP-MNT-001", 5),
            ("PK-02 servo indexing coupling drift eye-mark SRV-2240", "SOP-MNT-001", 5),
            ("PK-02 heated sealing platen bearing bind thermal rework BRG-2290", "SOP-MNT-001", 6),
        ]
        for q, exp_code, exp_page in queries:
            hits = retriever.search(q, top_k=3)
            self.assertEqual(hits[0]["sop_code"], exp_code, f"Query failed SOP code: {q}")
            self.assertEqual(hits[0]["page_number"], exp_page, f"Query failed page: {q} (got p.{hits[0]['page_number']})")

    def test_05_sql_guard_cost_and_4sheet_excel(self) -> None:
        ok = query_inventory_sql_impl("SELECT part_code, stock_qty FROM spare_parts_inventory WHERE part_code='BRG-4402'")
        self.assertEqual(ok["rows"][0]["stock_qty"], 6)
        self.assertIn("error", query_inventory_sql_impl("DROP TABLE machines"))

        alt = check_alternative_parts_impl("BLT-2205")
        self.assertTrue(alt["has_qualified_in_stock_alternative"])

        est2 = estimate_repair_cost("RP-04", 780.0, attempt_number=2)
        self.assertGreater(est2["incremental_rework_cost_usd"], 0.0)

        res_xl = generate_monthly_excel_report_impl()
        wb = openpyxl.load_workbook(res_xl["excel_path"])
        self.assertEqual(wb.sheetnames, ["KPI Summary", "Incident Log", "CAPA & Reorders", "GMP Audit Trail"])

    def test_06_llm_config_discovery_mismatch_and_retry(self) -> None:
        for p_cfg in PROVIDER_CATALOG.values():
            for m in p_cfg["models"]:
                self.assertNotIn("gemini-2.5", m)

        self.assertIsNotNone(detect_key_provider_mismatch("groq", "AIzaSyFakeGeminiKey"))
        self.assertIsNotNone(detect_key_provider_mismatch("gemini", "gsk_FakeGroqKey"))
        self.assertTrue(is_model_not_found_error(RuntimeError("Error code: 404 - model_not_found")))

        fake_resp = MagicMock()
        fake_resp.status_code = 200
        fake_resp.json.return_value = {
            "data": [
                {"id": "whisper-large-v3"},
                {"id": "canopylabs/orpheus-v1-english"},
                {"id": "allam-2-7b"},
                {"id": "openai/gpt-oss-120b"},
                {"id": "qwen/qwen3.8-27b"},
            ]
        }
        with patch("src.llm_config.requests.get", return_value=fake_resp):
            models, err = fetch_live_models("groq", "gsk_mock")
        self.assertIsNone(err)
        self.assertEqual(models, ["openai/gpt-oss-120b", "qwen/qwen3.8-27b"])

        attempts = {"n": 0}

        def flaky() -> str:
            attempts["n"] += 1
            if attempts["n"] < 3:
                raise RuntimeError("HTTP 429 rate limit")
            return "ok"

        with patch("src.llm_config.time.sleep", return_value=None):
            self.assertEqual(invoke_with_retry(flaky, pacing_delay=0.0, max_retries=4, base_delay=0.01), "ok")

    def test_07_parallel_orchestra_traces_hitl_capa_and_certificate_scenario_2(self) -> None:
        init = create_initial_state("scenario_2", execution_mode="offline_demo", hitl_enabled=True)

        s_p1, _ = run_maintenance_workflow(init)
        self.assertTrue(s_p1["diagnostic_done"])
        self.assertTrue(s_p1["precheck_done"])
        self.assertEqual(s_p1["precheck_report"]["active_batch_id"], "LOT-2026-WFI-812")
        self.assertTrue(s_p1["awaiting_human_approval"])
        self.assertEqual(s_p1["pending_approval"]["gate_type"], "SQL_MUTATION")

        s_r1 = apply_human_decision(s_p1, "sql_reservation", "APPROVED", reviewer="QA Director")
        s_p2, _ = run_maintenance_workflow(s_r1)
        self.assertTrue(s_p2["awaiting_human_approval"])
        self.assertEqual(s_p2["pending_approval"]["gate_type"], "REPAIR_COST")
        self.assertEqual(get_inventory_dataframe().set_index("part_code")["stock_qty"]["SL-8810"], 3)

        s_r2 = apply_human_decision(s_p2, "repair_attempt_1", "APPROVED", reviewer="QA Director")
        s_p3, _ = run_maintenance_workflow(s_r2)
        self.assertTrue(s_p3["awaiting_human_approval"])
        self.assertEqual(s_p3["pending_approval"]["gate_type"], "REWORK_COST")
        self.assertTrue(s_p3["engineering_replan_done"])
        self.assertIn("Laser", s_p3["engineering_replan"]["corrective_procedure"])
        self.assertEqual(s_p3["maintenance_report"]["post_repair_vibration_rms"], 4.8)

        s_r3 = apply_human_decision(s_p3, "rework_attempt_2", "APPROVED", reviewer="QA Director")
        s_final, _ = run_maintenance_workflow(s_r3)
        self.assertTrue(s_final["protocol_closed"])
        self.assertEqual(s_final["last_gate_decision"], "SIGNED_OFF")
        self.assertTrue(s_final["clearance_approved"])
        self.assertEqual(s_final["maintenance_report"]["attempt_count"], 2)
        self.assertEqual(s_final["maintenance_report"]["post_repair_vibration_rms"], 1.6)
        self.assertTrue(s_final["capa_report"]["capa_triggered"])

        traces = s_final.get("agent_traces", [])
        self.assertGreaterEqual(len(traces), 9)
        trace_nodes = {tr["node"] for tr in traces}
        for req_node in (
            "orchestrator",
            "diagnostic_engineer",
            "stock_precheck",
            "stock_agent",
            "field_technician",
            "qa_clearance",
            "analyst_capa",
            "human_qa",
        ):
            self.assertIn(req_node, trace_nodes, f"Missing trace from node {req_node}")

        cert_bytes = build_incident_certificate_pdf(s_final)
        self.assertTrue(cert_bytes.startswith(b"%PDF-"))
        cert_pdf = PdfReader(io.BytesIO(cert_bytes))
        self.assertEqual(len(cert_pdf.pages), 2)
        cert_text = "\n".join(pg.extract_text() or "" for pg in cert_pdf.pages)
        self.assertIn(s_final["ticket_id"], cert_text)
        self.assertIn("Vibration Recovery", cert_text)
        for pg_i, pct in enumerate(_measure_pdf_fill(cert_bytes), start=1):
            self.assertGreaterEqual(pct, 65.0, f"Certificate p.{pg_i} fill {pct:.1f}% is below 65%")

    def test_08_auto_po_and_all_9_fault_scenarios_end_to_end(self) -> None:
        s_p1, _ = run_maintenance_workflow(create_initial_state("scenario_3", execution_mode="offline_demo", hitl_enabled=True))
        self.assertEqual(s_p1["pending_approval"]["part_code"], "BLT-2205-ALT")
        self.assertTrue(s_p1["purchase_requisition"]["created"])
        held = apply_human_decision(s_p1, "sql_reservation", "REJECTED", reviewer="QA Lead", comment="Hold")
        self.assertTrue(held["protocol_closed"])
        self.assertEqual(held["last_gate_decision"], "HUMAN_REJECTED")
        self.assertEqual(get_inventory_dataframe().set_index("part_code")["stock_qty"]["BLT-2205-ALT"], 5)

        fin_roller, _ = run_maintenance_workflow(
            create_initial_state("scenario_tp01_roller", execution_mode="offline_demo", hitl_enabled=False)
        )
        self.assertTrue(fin_roller["protocol_closed"])
        self.assertEqual(fin_roller["inventory_report"]["reserved_part_code"], "RLR-4450")
        self.assertTrue(fin_roller["purchase_requisition"]["created"])
        self.assertGreaterEqual(len(get_purchase_requisitions_dataframe()), 1)

        fin_seal, _ = run_maintenance_workflow(
            create_initial_state("scenario_pk02_sealing", execution_mode="offline_demo", hitl_enabled=False)
        )
        self.assertTrue(fin_seal["protocol_closed"])
        self.assertTrue(fin_seal["engineering_replan_done"])
        self.assertEqual(fin_seal["maintenance_report"]["attempt_count"], 2)
        self.assertEqual(fin_seal["maintenance_report"]["post_repair_vibration_rms"], 1.5)
        self.assertEqual(fin_seal["maintenance_attempts"][0]["post_repair_vibration_rms"], 3.9)
        self.assertIn("Thermal", fin_seal["engineering_replan"]["corrective_procedure"])

        for sc_id in (
            "scenario_1",
            "scenario_tp01_punch",
            "scenario_rp04_impeller",
            "scenario_rp04_softfoot",
            "scenario_pk02_servo",
        ):
            fin, _ = run_maintenance_workflow(create_initial_state(sc_id, execution_mode="offline_demo", hitl_enabled=False))
            self.assertTrue(fin["protocol_closed"], f"{sc_id} did not close")
            self.assertEqual(fin["last_gate_decision"], "SIGNED_OFF")
            self.assertTrue(fin["clearance_approved"])
            self.assertEqual(
                fin["inventory_report"]["reserved_part_code"],
                DEMO_SCENARIOS[sc_id]["expected_part_code"],
            )

    def test_09_autonomous_orchestra_interlocks_and_mocked_live_404_fallback(self) -> None:
        self.assertIsNotNone(build_maintenance_graph())

        # Verify _enforce_orchestra_interlocks fixes wrong LLM reviewed_stage for HITL REPAIR_COST safety
        st_check = create_initial_state("scenario_1", execution_mode="live", hitl_enabled=True)
        st_check["diagnostic_done"] = True
        st_check["precheck_done"] = True
        st_check["diagnostic_approved"] = True
        st_check["diagnostic_report"] = {"sop_code": "SOP-MNT-001", "page_number": 2, "root_cause": "spall", "required_part_code": "BRG-4402"}
        st_check["inventory_report"] = {"reserved": True, "reserved_part_code": "BRG-4402", "unit_cost_usd": 485.0}
        wrong_llm = OrchestratorDecision(
            next_agent="field_technician",
            instruction="Go repair",
            gate_decision="PENDING",
            reviewed_stage="intake",
            audit_notes="bad stage",
        )
        enforced = _enforce_orchestra_interlocks(wrong_llm, st_check)
        self.assertEqual(enforced.reviewed_stage, "stock")
        self.assertEqual(enforced.gate_decision, "APPROVED")

        init_live = create_initial_state(
            "scenario_1",
            execution_mode="live",
            llm_provider="groq",
            llm_model="llama-3.3-70b-versatile",
            api_key="gsk_mocked",
            pacing_delay_sec=0.0,
            hitl_enabled=False,
        )
        orch_seq = iter(
            [
                OrchestratorDecision(next_agent="parallel_intake", instruction="Go", gate_decision="PENDING", reviewed_stage="intake", audit_notes="ok"),
                OrchestratorDecision(next_agent="stock_agent", instruction="Stock", gate_decision="APPROVED", reviewed_stage="diagnostic", audit_notes="ok"),
                OrchestratorDecision(next_agent="field_technician", instruction="Tech", gate_decision="APPROVED", reviewed_stage="stock", audit_notes="ok"),
                OrchestratorDecision(next_agent="qa_clearance", instruction="Clear", gate_decision="APPROVED", reviewed_stage="maintenance", audit_notes="ok"),
                OrchestratorDecision(next_agent="analyst_capa", instruction="KPI", gate_decision="APPROVED", reviewed_stage="clearance", audit_notes="ok"),
                OrchestratorDecision(next_agent="FINISH", instruction="Close", gate_decision="SIGNED_OFF", reviewed_stage="closure", audit_notes="ok"),
            ]
        )

        def make_mock_llm(*args, **kwargs):
            m_name = kwargs.get("model", "")
            mock_llm = MagicMock()

            def with_structured_output(_schema):
                s_llm = MagicMock()

                def call_s(_msgs):
                    if m_name == "llama-3.3-70b-versatile":
                        raise RuntimeError("Error code: 404 - model_not_found")
                    return next(orch_seq)

                s_llm.invoke.side_effect = call_s
                return s_llm

            def bind_tools(tools):
                t_names = [t.name for t in tools]
                b_llm = MagicMock()
                step = {"n": 0}

                def call_t(_msgs):
                    step["n"] += 1
                    if step["n"] == 1:
                        if "search_sop_manuals" in t_names:
                            return AIMessage(
                                content="",
                                tool_calls=[
                                    {"id": "1", "name": "read_sensor_telemetry", "args": {"machine_id": "TP-01"}},
                                    {"id": "2", "name": "search_sop_manuals", "args": {"query": "TP-01 BRG-4402"}},
                                ],
                            )
                        if "assess_batch_risk" in t_names:
                            return AIMessage(
                                content="",
                                tool_calls=[
                                    {"id": "3", "name": "assess_batch_risk", "args": {"machine_id": "TP-01"}},
                                    {"id": "4", "name": "query_inventory_sql", "args": {"sql_query": "SELECT * FROM spare_parts_inventory WHERE machine_id='TP-01'"}},
                                ],
                            )
                        if "reserve_spare_part" in t_names:
                            return AIMessage(
                                content="",
                                tool_calls=[{"id": "5", "name": "reserve_spare_part", "args": {"part_code": "BRG-4402", "quantity": 1, "ticket_id": init_live["ticket_id"]}}],
                            )
                        if "run_post_repair_vibration_test" in t_names:
                            return AIMessage(
                                content="",
                                tool_calls=[
                                    {"id": "6", "name": "execute_part_replacement", "args": {"machine_id": "TP-01", "old_part": "WORN-BRG-4402", "new_part": "BRG-4402", "sop_ref": "SOP-MNT-001 p.2"}},
                                    {"id": "7", "name": "run_post_repair_vibration_test", "args": {"machine_id": "TP-01"}},
                                ],
                            )
                        if "run_qa_line_clearance" in t_names:
                            return AIMessage(
                                content="",
                                tool_calls=[{"id": "8", "name": "run_qa_line_clearance", "args": {"machine_id": "TP-01", "ticket_id": init_live["ticket_id"], "final_vibration_rms": 1.3}}],
                            )
                        if "generate_monthly_excel_report" in t_names:
                            return AIMessage(
                                content="",
                                tool_calls=[
                                    {"id": "9", "name": "log_incident_analytics", "args": {"ticket_data": json.dumps({"ticket_id": init_live["ticket_id"], "machine_id": "TP-01"})}},
                                    {"id": "10", "name": "create_capa_if_repeat_failure", "args": {"machine_id": "TP-01", "ticket_id": init_live["ticket_id"], "failure_mode": "Bearing wear", "sop_ref": "SOP-MNT-001 p.2"}},
                                    {"id": "11", "name": "generate_monthly_excel_report", "args": {}},
                                ],
                            )
                    return AIMessage(content='{"sop_code": "SOP-MNT-001", "page_number": 2, "required_part_code": "BRG-4402"}')

                b_llm.invoke.side_effect = call_t
                return b_llm

            mock_llm.with_structured_output.side_effect = with_structured_output
            mock_llm.bind_tools.side_effect = bind_tools
            return mock_llm

        with (
            patch("src.agents.get_chat_model", side_effect=make_mock_llm),
            patch("src.llm_config.fetch_live_models", return_value=(["llama-3.1-8b-instant"], None)),
        ):
            fin_live, _ = run_maintenance_workflow(init_live)
        self.assertEqual(fin_live["llm_model"], "llama-3.1-8b-instant")
        self.assertEqual(fin_live["last_gate_decision"], "SIGNED_OFF")
        self.assertGreaterEqual(len(get_audit_log_dataframe(init_live["ticket_id"])), 11)

    def test_10_bilingual_tr_en_localization_and_streamlit_ui_smoke(self) -> None:
        # Bilingual i18n unit assertions
        self.assertEqual(t("tr", "hitl_approve_btn"), "✅ Onayla ve Devam Et")
        self.assertEqual(localize_zone("Zone D (Danger)", "tr"), "Bölge D (Tehlike)")
        self.assertEqual(set(SCENARIO_LABELS_TR.keys()), set(DEMO_SCENARIOS.keys()))

        fin_tr, _ = run_maintenance_workflow(
            create_initial_state("scenario_1", execution_mode="offline_demo", hitl_enabled=False, language="tr")
        )
        self.assertEqual(fin_tr["last_gate_decision"], "SIGNED_OFF")
        self.assertTrue(any("Orkestratör" in str(tr["agent_name"]) for tr in fin_tr["agent_traces"]))

        from streamlit.testing.v1 import AppTest

        at = AppTest.from_file(str(ROOT_DIR / "app.py"), default_timeout=30)
        at.run()
        self.assertFalse(at.exception, f"Initial render raised: {at.exception}")

        at.sidebar.radio(key="exec_mode_radio").set_value("offline_demo")
        at.sidebar.slider(key="watch_pacing_slider").set_value(0.0)
        at.toggle(key="auto_anomaly_toggle").set_value(False).run()
        self.assertFalse(at.exception)

        # Inject TP-01 fault -> Top-of-page HITL SQL_MUTATION card appears immediately
        at.button(key="bay_inject_btn_TP-01").click().run()
        self.assertFalse(at.exception, f"Bay fault injection raised: {at.exception}")
        self.assertTrue(at.session_state["last_state"]["awaiting_human_approval"])
        self.assertEqual(at.session_state["last_state"]["pending_approval"]["gate_type"], "SQL_MUTATION")
        btn_keys = {b.key for b in at.button}
        self.assertIn("hitl_approve_btn", btn_keys)
        self.assertIn("hitl_deny_btn", btn_keys)

        # Approve SQL -> REPAIR_COST card appears immediately
        at.button(key="hitl_approve_btn").click().run()
        self.assertEqual(at.session_state["last_state"]["pending_approval"]["gate_type"], "REPAIR_COST")

        # Approve REPAIR_COST -> SIGNED_OFF
        at.button(key="hitl_approve_btn").click().run()
        self.assertFalse(at.exception, f"Full HITL sign-off raised: {at.exception}")
        self.assertTrue(at.session_state["last_state"]["protocol_closed"])
        self.assertEqual(at.session_state["last_state"]["last_gate_decision"], "SIGNED_OFF")
        self.assertGreaterEqual(len(at.session_state["last_state"]["agent_traces"]), 6)

        # Executive Final Incident Closure Report sits at top of View 1 with cited SOP deep-link
        closed_btn_keys = {b.key for b in at.button}
        self.assertIn("closure_open_sop_btn", closed_btn_keys)

        # Switch to Turkish UI and test Executive Report + cited SOP navigation + all 3 views cleanly
        at.sidebar.radio(key="lang_radio").set_value("🇹🇷 Türkçe").run()
        self.assertFalse(at.exception)
        at.button(key="closure_open_sop_btn").click().run()
        self.assertFalse(at.exception)
        self.assertEqual(at.session_state["active_view_idx"], 1)
        at.button(key="nav_btn_1").click().run()
        self.assertFalse(at.exception)
        at.button(key="lib_open_SOP-MNT-002").click().run()
        self.assertFalse(at.exception)
        at.button(key="nav_btn_2").click().run()
        self.assertFalse(at.exception)


if __name__ == "__main__":
    unittest.main(verbosity=2)
