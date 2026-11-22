"""Typed LangGraph eBMR State Schema and 9-Scenario Industrial Failure Definitions."""

from __future__ import annotations

import uuid
from typing import Annotated, Any, Dict, List, Literal, Optional
from typing_extensions import TypedDict
from pydantic import BaseModel, Field


def _merge_append_unique(
    left: Optional[List[Dict[str, Any]]],
    right: Optional[List[Dict[str, Any]]],
) -> List[Dict[str, Any]]:
    """Reducer that safely merges parallel or sequential lists while deduplicating identical items."""
    base = list(left or [])
    if not right:
        return base
    seen = {repr(x) for x in base}
    for item in right:
        sig = repr(item)
        if sig not in seen:
            base.append(item)
            seen.add(sig)
    return base


def _keep_latest(left: Optional[str], right: Optional[str]) -> str:
    """Reducer selecting non-empty latest string for parallel-safe scalar fields."""
    if right is not None and str(right).strip() != "":
        return str(right)
    return str(left or "")


class OrchestratorDecision(BaseModel):
    """Structured decision contract returned by the Tier 3 GMP Quality Conductor."""

    next_agent: Literal[
        "parallel_intake",
        "diagnostic_engineer",
        "stock_agent",
        "field_technician",
        "qa_clearance",
        "analyst_capa",
        "FINISH",
    ] = Field(..., description="Next LangGraph route chosen strictly by the Quality Conductor.")
    instruction: str = Field(..., description="Actionable cGMP work order or gate directive for the routed node.")
    gate_decision: Literal[
        "PENDING",
        "APPROVED",
        "REJECTED",
        "ESCALATED",
        "AWAITING_HUMAN",
        "SIGNED_OFF",
    ] = Field(..., description="Formal GMP quality gate verdict.")
    reviewed_stage: Literal[
        "intake",
        "diagnostic",
        "stock",
        "maintenance",
        "engineering_replan",
        "clearance",
        "analyst",
        "closure",
    ] = Field(..., description="Stage of the eBMR protocol evaluated at this step.")
    audit_notes: str = Field(..., description="Traceable ALCOA+ justification citing state evidence.")


class MaintenanceState(TypedDict):
    """Shared Electronic Batch / Maintenance Record (eBMR) traversing the LangGraph StateGraph."""

    # Ticket Metadata & Configuration
    ticket_id: str
    scenario_id: str
    machine_id: str
    machine_name: str
    trigger_source: str
    alarm_type: str
    alarm_vibration_rms: float
    symptom_summary: str
    inject_post_repair_fail: bool
    execution_mode: str
    llm_provider: str
    llm_model: Annotated[str, _keep_latest]
    api_key: Optional[str]
    pacing_delay_sec: float
    hitl_enabled: bool
    language: str

    # Conductor Routing & Progression Flags
    next_agent: str
    current_instruction: str
    last_worker: str
    last_gate_decision: str
    iteration_count: int
    max_iterations: int

    diagnostic_done: bool
    precheck_done: bool
    diagnostic_approved: bool
    stock_approved: bool
    escalation_to_engineer: bool
    engineering_replan_done: bool
    maintenance_approved: bool
    clearance_approved: bool
    analyst_approved: bool
    protocol_closed: bool
    escalated: bool

    # Human-in-the-Loop Governance
    awaiting_human_approval: bool
    pending_approval: Dict[str, Any]
    human_decisions: Dict[str, Any]

    # Counters & Errors
    vibration_test_count: int
    repair_attempt_count: int
    error_message: Annotated[str, _keep_latest]

    # Worker Reports
    diagnostic_report: Dict[str, Any]
    precheck_report: Dict[str, Any]
    inventory_report: Dict[str, Any]
    purchase_requisition: Dict[str, Any]
    engineering_replan: Dict[str, Any]
    maintenance_report: Dict[str, Any]
    clearance_report: Dict[str, Any]
    capa_report: Dict[str, Any]
    analytics_report: Dict[str, Any]

    # Parallel-Safe Evidence & Trace Logs (Reducers)
    rag_chunks: Annotated[List[Dict[str, Any]], _merge_append_unique]
    sql_queries: Annotated[List[Dict[str, Any]], _merge_append_unique]
    telemetry_readings: Annotated[List[Dict[str, Any]], _merge_append_unique]
    maintenance_attempts: Annotated[List[Dict[str, Any]], _merge_append_unique]
    tool_calls_log: Annotated[List[Dict[str, Any]], _merge_append_unique]
    audit_trail: Annotated[List[Dict[str, Any]], _merge_append_unique]
    agent_events: Annotated[List[Dict[str, Any]], _merge_append_unique]
    agent_traces: Annotated[List[Dict[str, Any]], _merge_append_unique]


# Nine realistic Plant 04 fault scenarios (3 per asset bay: TP-01, RP-04, PK-02)
DEMO_SCENARIOS: Dict[str, Dict[str, Any]] = {
    # ---------------- TP-01 ROTARY TABLET PRESS ----------------
    "scenario_1": {
        "scenario_id": "scenario_1",
        "machine_id": "TP-01",
        "machine_name": "TP-01 Rotary Tablet Press (Fette-Class)",
        "downtime_rate_usd_min": 68.0,
        "alarm_type": "High Turret Vibration & Bearing Spall",
        "label": "Bearing Spall — BRG-4402 (9.8 mm/s)",
        "short_label": "Bearing Spalling (BRG-4402)",
        "description": "Main turret spherical bearing raceway fatigue spall with 184.2 Hz BPFO harmonics and 74.2 C temp.",
        "initial_vibration_rms": 9.8,
        "temperature_c": 74.2,
        "dominant_frequency_hz": 184.2,
        "peak_accel_g": 4.4,
        "kurtosis": 6.8,
        "motor_amps": 22.8,
        "symptom_summary": "9.8 mm/s RMS tri-axial velocity on TP-01 main turret; 184.2 Hz BPFO harmonics, Kurtosis 6.8, bearing housing 74.2 C, metallic grinding.",
        "expected_part_code": "BRG-4402",
        "recommended_alt_code": "BRG-4402-ALT",
        "expected_sop_code": "SOP-MNT-001",
        "expected_sop_page": 2,
        "failure_mode_title": "Main Turret Spherical Bearing Outer-Race Fatigue Spalling",
        "root_cause_summary": "Subsurface fatigue spalling on turret SKF 22214 E bearing raceway under high compaction load (SOP-MNT-001 p.2).",
        "required_tools": [
            "10-ton hydraulic puller",
            "Induction heater (110 C max)",
            "Calibrated torque wrench (45 Nm)",
            "NSF H1 grease gun",
        ],
        "inject_post_repair_fail": False,
        "fail_attempt1_rms": None,
        "post_repair_pass_rms": 1.3,
        "replan_sop_ref": "",
        "replan_procedure": "",
        "replan_tolerance": "",
    },
    "scenario_tp01_punch": {
        "scenario_id": "scenario_tp01_punch",
        "machine_id": "TP-01",
        "machine_name": "TP-01 Rotary Tablet Press (Fette-Class)",
        "downtime_rate_usd_min": 68.0,
        "alarm_type": "Punch Binding & Auto-Lube Starvation",
        "label": "Punch Bind / Auto-Lube — LUB-3010 (6.4 mm/s)",
        "short_label": "Punch Binding & Lube Starvation (LUB-3010)",
        "description": "Blocked turret oil mist jets causing excipient bore drag, 28.4 A drive current surge, and stick-slip.",
        "initial_vibration_rms": 6.4,
        "temperature_c": 61.5,
        "dominant_frequency_hz": 45.0,
        "peak_accel_g": 2.2,
        "kurtosis": 4.3,
        "motor_amps": 28.4,
        "symptom_summary": "6.4 mm/s RMS stick-slip chatter on TP-01 turret at 45.0 Hz with 28.4 A (+45% VFD current surge) and blocked NSF H1 mist LUB-3010 jets.",
        "expected_part_code": "LUB-3010",
        "recommended_alt_code": None,
        "expected_sop_code": "SOP-MNT-001",
        "expected_sop_page": 3,
        "failure_mode_title": "Punch Binding & Automatic Turret Lubrication Starvation",
        "root_cause_summary": "Granulation fines baked inside turret guide bores due to clogged pneumatic NSF H1 lube spray nozzles LUB-3010 (SOP-MNT-001 p.3).",
        "required_tools": [
            "70% IPA bore cleaning kit",
            "Nylon bore brushes",
            "Pneumatic 4.2 bar mist tester",
            "Digital VFD clamp meter",
        ],
        "inject_post_repair_fail": False,
        "fail_attempt1_rms": None,
        "post_repair_pass_rms": 1.4,
        "replan_sop_ref": "",
        "replan_procedure": "",
        "replan_tolerance": "",
    },
    "scenario_tp01_roller": {
        "scenario_id": "scenario_tp01_roller",
        "machine_id": "TP-01",
        "machine_name": "TP-01 Rotary Tablet Press (Fette-Class)",
        "downtime_rate_usd_min": 68.0,
        "alarm_type": "Pre-Compression Cam & Roller Knock",
        "label": "Cam Roller Knock [Auto-PO] — RLR-4450 (8.2 mm/s)",
        "short_label": "Cam & Roller Knock + Auto-PO (RLR-4450)",
        "description": "Pre-compression roller flat-spotting (Peak 5.2 g, Kurtosis 7.4); reserving last RLR-4450 triggers Tier 2 Auto-PO.",
        "initial_vibration_rms": 8.2,
        "temperature_c": 68.0,
        "dominant_frequency_hz": 90.0,
        "peak_accel_g": 5.2,
        "kurtosis": 7.4,
        "motor_amps": 25.1,
        "symptom_summary": "8.2 mm/s RMS cyclic knocking at 90.0 Hz (2x cam impact), Peak 5.2 g, Kurtosis 7.4 on TP-01 pre-compression roller RLR-4450.",
        "expected_part_code": "RLR-4450",
        "recommended_alt_code": "RLR-4450-ALT",
        "expected_sop_code": "SOP-MNT-001",
        "expected_sop_page": 4,
        "failure_mode_title": "Pre-Compression Cam Track Brinelling & Roller Knock",
        "root_cause_summary": "Cyclic 45 kN pre-compression impacts brinelled RLR-4450 needle-roller outer ring, triggering safety-stock Auto-PO (SOP-MNT-001 p.4).",
        "required_tools": [
            "Cam yoke pin puller",
            "Brass feeler gauge (0.08-0.12 mm)",
            "Calibrated 52 Nm torque wrench",
            "Acoustic stethoscope",
        ],
        "inject_post_repair_fail": False,
        "fail_attempt1_rms": None,
        "post_repair_pass_rms": 1.4,
        "replan_sop_ref": "",
        "replan_procedure": "",
        "replan_tolerance": "",
    },

    # ---------------- RP-04 SANITARY CENTRIFUGAL PUMP ----------------
    "scenario_2": {
        "scenario_id": "scenario_2",
        "machine_id": "RP-04",
        "machine_name": "RP-04 Sanitary Centrifugal Pump (3-A)",
        "downtime_rate_usd_min": 52.0,
        "alarm_type": "Mechanical Seal Rupture & Post-Repair Misalignment",
        "label": "Seal Cavitation [Gate 3 Rework] — SL-8810 (9.4 mm/s)",
        "short_label": "Seal Cavitation + Gate 3 Laser Rework (SL-8810)",
        "description": "SiC seal cavitation shattering; 1st fix hits 4.8 mm/s Gate 3 Reject -> Engineer escalates Laser Alignment -> 1.6 mm/s.",
        "initial_vibration_rms": 9.4,
        "temperature_c": 71.8,
        "dominant_frequency_hz": 1420.0,
        "peak_accel_g": 5.8,
        "kurtosis": 6.1,
        "motor_amps": 21.4,
        "symptom_summary": "9.4 mm/s RMS with 1420 Hz acoustic cavitation crackle, Peak 5.8 g, Kurtosis 6.1, gland weeping & 18% head drop on RP-04 seal SL-8810.",
        "expected_part_code": "SL-8810",
        "recommended_alt_code": "SL-8810-ALT",
        "expected_sop_code": "SOP-MNT-002",
        "expected_sop_page": 2,
        "failure_mode_title": "Sanitary Mechanical Cartridge Seal Cavitation & Face Rupture",
        "root_cause_summary": "Suction strainer clogging dropped NPSHa below NPSHr, shattering SL-8810 SiC faces; CIP thermal growth caused post-seal misalignment (SOP-MNT-002 p.2-3).",
        "required_tools": [
            "Cartridge seal puller",
            "Torque wrench (32 Nm & 55 Nm)",
            "Dual-beam laser shaft analyzer",
            "304SS shim pack",
        ],
        "inject_post_repair_fail": True,
        "fail_attempt1_rms": 4.8,
        "post_repair_pass_rms": 1.6,
        "replan_sop_ref": "SOP-MNT-002 p.3",
        "replan_procedure": "Cold Dual-Beam Laser Shaft Realignment (parallel/angular offset <= 0.05 mm TIR, -0.03 mm CIP cold vertical compensation, star-torque base bolts 55 Nm) per SOP-MNT-002 p.3",
        "replan_tolerance": "Soft foot < 0.05 mm; Parallel & Angular <= 0.05 mm TIR; 55 Nm base torque",
    },
    "scenario_rp04_impeller": {
        "scenario_id": "scenario_rp04_impeller",
        "machine_id": "RP-04",
        "machine_name": "RP-04 Sanitary Centrifugal Pump (3-A)",
        "downtime_rate_usd_min": 52.0,
        "alarm_type": "Impeller Erosion & Blade-Pass Unbalance",
        "label": "Impeller Unbalance — IMP-8825 (8.9 mm/s)",
        "short_label": "Impeller Erosion & Blade-Pass Unbalance (IMP-8825)",
        "description": "Vane leading-edge erosion generating 145 Hz (6x vane-pass) radial vibration and >20% flow head loss.",
        "initial_vibration_rms": 8.9,
        "temperature_c": 64.5,
        "dominant_frequency_hz": 145.0,
        "peak_accel_g": 3.9,
        "kurtosis": 4.4,
        "motor_amps": 18.2,
        "symptom_summary": "8.9 mm/s RMS radial vibration dominated by 145.0 Hz (6x vane-pass) and -22% WFI discharge pressure on RP-04 impeller IMP-8825.",
        "expected_part_code": "IMP-8825",
        "recommended_alt_code": None,
        "expected_sop_code": "SOP-MNT-002",
        "expected_sop_page": 4,
        "failure_mode_title": "316L Sanitary Impeller Cavitation Erosion & Dynamic Unbalance",
        "root_cause_summary": "Cavitation pitting removed impeller vane mass on RP-04, causing ISO G2.5 dynamic unbalance and 145 Hz vane-pass pulsations (SOP-MNT-002 p.4).",
        "required_tools": [
            "Tri-clamp volute spanner",
            "Impeller puller",
            "Brass feeler gauge (0.40-0.60 mm)",
            "Torque wrench (65 Nm)",
        ],
        "inject_post_repair_fail": False,
        "fail_attempt1_rms": None,
        "post_repair_pass_rms": 1.4,
        "replan_sop_ref": "",
        "replan_procedure": "",
        "replan_tolerance": "",
    },
    "scenario_rp04_softfoot": {
        "scenario_id": "scenario_rp04_softfoot",
        "machine_id": "RP-04",
        "machine_name": "RP-04 Sanitary Centrifugal Pump (3-A)",
        "downtime_rate_usd_min": 52.0,
        "alarm_type": "Motor Soft-Foot Distortion & Axial Drift",
        "label": "Soft-Foot Axial Drift — SHM-8805 (5.6 mm/s)",
        "short_label": "Motor Soft-Foot Distortion & Shim Fix (SHM-8805)",
        "description": "Frame twist across pump skid feet creating 48.3 Hz (2x RPM) axial coupling strain and elevated motor current.",
        "initial_vibration_rms": 5.6,
        "temperature_c": 66.2,
        "dominant_frequency_hz": 48.3,
        "peak_accel_g": 2.4,
        "kurtosis": 3.3,
        "motor_amps": 23.0,
        "symptom_summary": "5.6 mm/s RMS high Axial-Z vibration at 48.3 Hz (2x RPM) with 23.0 A motor load from skid soft-foot frame twist needing SHM-8805 shims.",
        "expected_part_code": "SHM-8805",
        "recommended_alt_code": None,
        "expected_sop_code": "SOP-MNT-002",
        "expected_sop_page": 5,
        "failure_mode_title": "Motor Soft-Foot Distortion & Skid Alignment Drift",
        "root_cause_summary": "Uneven skid pads introduced > 0.12 mm motor frame soft-foot twist under 55 Nm base bolt torque (SOP-MNT-002 p.5).",
        "required_tools": [
            "Magnetic dial gauge (<0.05 mm)",
            "304SS pre-cut shim pack SHM-8805",
            "Dual-beam laser alignment kit",
            "Torque wrench (55 Nm)",
        ],
        "inject_post_repair_fail": False,
        "fail_attempt1_rms": None,
        "post_repair_pass_rms": 1.3,
        "replan_sop_ref": "",
        "replan_procedure": "",
        "replan_tolerance": "",
    },

    # ---------------- PK-02 SERVO BLISTER PACKAGING LINE ----------------
    "scenario_3": {
        "scenario_id": "scenario_3",
        "machine_id": "PK-02",
        "machine_name": "PK-02 Servo Blister Packaging Line",
        "downtime_rate_usd_min": 44.0,
        "alarm_type": "Drive Belt Fatigue & Stockout Equivalence",
        "label": "Belt Stockout Alt — BLT-2205 (7.9 mm/s)",
        "short_label": "Drive Belt Fatigue + Stockout Alt (BLT-2205-ALT)",
        "description": "OEM BLT-2205 qty=0 in warehouse; Stock Agent auto-evaluates qualified BLT-2205-ALT + requests Human SQL Auth.",
        "initial_vibration_rms": 7.9,
        "temperature_c": 58.5,
        "dominant_frequency_hz": 240.0,
        "peak_accel_g": 2.9,
        "kurtosis": 4.9,
        "motor_amps": 16.8,
        "symptom_summary": "7.9 mm/s RMS synchronous belt chirp at 240.0 Hz (2x tooth slip), Kurtosis 4.9 on PK-02 drive BLT-2205.",
        "expected_part_code": "BLT-2205",
        "recommended_alt_code": "BLT-2205-ALT",
        "expected_sop_code": "SOP-MNT-001",
        "expected_sop_page": 5,
        "failure_mode_title": "Synchronous Poly-Chain Drive Belt Fatigue & Tooth Shear",
        "root_cause_summary": "Tensile cord elongation caused 2x tooth-skip vibration on PK-02; OEM BLT-2205 is at zero stock, requiring BLT-2205-ALT (SOP-MNT-001 p.5).",
        "required_tools": [
            "Laser pulley straight-edge",
            "Gates Sonic Tension Meter (180-200 N)",
            "Torque wrench (38 Nm)",
        ],
        "inject_post_repair_fail": False,
        "fail_attempt1_rms": None,
        "post_repair_pass_rms": 1.5,
        "replan_sop_ref": "",
        "replan_procedure": "",
        "replan_tolerance": "",
    },
    "scenario_pk02_servo": {
        "scenario_id": "scenario_pk02_servo",
        "machine_id": "PK-02",
        "machine_name": "PK-02 Servo Blister Packaging Line",
        "downtime_rate_usd_min": 44.0,
        "alarm_type": "Servo Coupling Backlash & Eye-Mark Drift",
        "label": "Servo Index Drift — SRV-2240 (5.1 mm/s)",
        "short_label": "Servo Coupling & Eye-Mark Drift (SRV-2240)",
        "description": "Elastomer spider backlash & encoder jitter causing +/-0.6 mm blister foil print misregistration.",
        "initial_vibration_rms": 5.1,
        "temperature_c": 55.0,
        "dominant_frequency_hz": 120.0,
        "peak_accel_g": 2.1,
        "kurtosis": 3.8,
        "motor_amps": 17.6,
        "symptom_summary": "5.1 mm/s RMS indexing jitter at 120.0 Hz with +/-0.6 mm foil eye-mark registration offset on PK-02 servo kit SRV-2240.",
        "expected_part_code": "SRV-2240",
        "recommended_alt_code": None,
        "expected_sop_code": "SOP-MNT-001",
        "expected_sop_page": 5,
        "failure_mode_title": "Servo Drive Coupling Backlash & Optical Registration Drift",
        "root_cause_summary": "Degraded servo spider insert and fouled optical eye-mark encoder introduced 120 Hz indexing slip (SOP-MNT-001 p.5).",
        "required_tools": [
            "70% IPA lens wipes",
            "Split-hub 24 Nm torque wrench",
            "PLC zero-home alignment gauge",
        ],
        "inject_post_repair_fail": False,
        "fail_attempt1_rms": None,
        "post_repair_pass_rms": 1.3,
        "replan_sop_ref": "",
        "replan_procedure": "",
        "replan_tolerance": "",
    },
    "scenario_pk02_sealing": {
        "scenario_id": "scenario_pk02_sealing",
        "machine_id": "PK-02",
        "machine_name": "PK-02 Servo Blister Packaging Line",
        "downtime_rate_usd_min": 44.0,
        "alarm_type": "Heated Platen Bearing Bind & Thermal Bow",
        "label": "Platen Bind [Thermal Rework] — BRG-2290 (7.1 mm/s)",
        "short_label": "Sealing Bearing Bind + Thermal Rework (BRG-2290)",
        "description": "High-temp platen bearing seize (92.4 C); cold fix leaves 3.9 mm/s thermal bow -> Thermal Parallelism Rework -> 1.5 mm/s.",
        "initial_vibration_rms": 7.1,
        "temperature_c": 92.4,
        "dominant_frequency_hz": 312.0,
        "peak_accel_g": 3.6,
        "kurtosis": 5.6,
        "motor_amps": 19.2,
        "symptom_summary": "7.1 mm/s RMS platen chatter at 312 Hz, 92.4 deg C housing soak, Kurtosis 5.6 and blister channel leaks on PK-02 BRG-2290.",
        "expected_part_code": "BRG-2290",
        "recommended_alt_code": None,
        "expected_sop_code": "SOP-MNT-001",
        "expected_sop_page": 6,
        "failure_mode_title": "Heated Sealing Platen Pillow Bearing Seizure & Thermal Bow",
        "root_cause_summary": "Lubricant carbonization at 160 deg C platen heat seized BRG-2290; cold fit omitted hot-soak parallelism leveling (SOP-MNT-001 p.6).",
        "required_tools": [
            "Surface IR pyrometer",
            "Carbon foil imprint film",
            "0.04 mm feeler gauge",
            "38 Nm hot-torque wrench",
        ],
        "inject_post_repair_fail": True,
        "fail_attempt1_rms": 3.9,
        "post_repair_pass_rms": 1.5,
        "replan_sop_ref": "SOP-MNT-001 p.6",
        "replan_procedure": "Thermal Feeler Gauge Parallelism Alignment (<= 0.04 mm carbon foil gap at 160 deg C operating soak) & 38 Nm hot retorque per SOP-MNT-001 p.6",
        "replan_tolerance": "<= 0.04 mm hot platen gap; 38 Nm hot retorque; <= 2.5 mm/s RMS",
    },
}

_BAY_META = {
    "TP-01": {"area": "Compression Suite A", "baseline_rms": 1.12, "nominal_rpm": 45, "downtime_rate_per_min": 68.0},
    "RP-04": {"area": "WFI & Granulation Suite B", "baseline_rms": 1.28, "nominal_rpm": 1450, "downtime_rate_per_min": 52.0},
    "PK-02": {"area": "Primary Packaging Suite C", "baseline_rms": 1.18, "nominal_rpm": 120, "downtime_rate_per_min": 44.0},
}
for _sc in DEMO_SCENARIOS.values():
    _bm = _BAY_META[_sc["machine_id"]]
    _sc.setdefault("area", _bm["area"])
    _sc.setdefault("baseline_rms", _bm["baseline_rms"])
    _sc.setdefault("nominal_rpm", _bm["nominal_rpm"])
    _sc.setdefault("downtime_rate_per_min", _bm["downtime_rate_per_min"])
    _sc.setdefault("alarm_vibration_rms", _sc["initial_vibration_rms"])
    _sc.setdefault("button_label", f"⚡ Inject Fault on {_sc['machine_id']}")


def create_initial_state(
    scenario_id: str,
    execution_mode: str = "live",
    llm_provider: str = "groq",
    llm_model: str = "openai/gpt-oss-120b",
    api_key: Optional[str] = None,
    pacing_delay_sec: float = 0.3,
    hitl_enabled: bool = True,
    trigger_source: str = "EDGE_SENSOR_AUTO_TRIP",
    language: str = "en",
) -> MaintenanceState:
    """Create initial typed MaintenanceState eBMR dictionary for any of the 9 scenarios."""
    if scenario_id not in DEMO_SCENARIOS:
        scenario_id = "scenario_1"
    sc = DEMO_SCENARIOS[scenario_id]
    ticket_id = f"TKT-{uuid.uuid4().hex[:8].upper()}"

    return {
        "ticket_id": ticket_id,
        "scenario_id": scenario_id,
        "machine_id": sc["machine_id"],
        "machine_name": sc["machine_name"],
        "trigger_source": trigger_source,
        "alarm_type": sc["alarm_type"],
        "alarm_vibration_rms": float(sc["initial_vibration_rms"]),
        "symptom_summary": sc["symptom_summary"],
        "inject_post_repair_fail": bool(sc.get("inject_post_repair_fail", False)),
        "execution_mode": execution_mode,
        "llm_provider": llm_provider,
        "llm_model": llm_model,
        "api_key": api_key,
        "pacing_delay_sec": float(pacing_delay_sec),
        "hitl_enabled": bool(hitl_enabled),
        "language": "tr" if str(language).lower() == "tr" else "en",
        "next_agent": "parallel_intake",
        "current_instruction": (
            f"Continuous edge vibration monitor tripped on {sc['machine_id']} ({sc['initial_vibration_rms']:.1f} mm/s RMS). "
            "Dispatch Tier 2 Diagnostic Engineer + Batch/Stock Pre-Check concurrently."
        ),
        "last_worker": "orchestrator",
        "last_gate_decision": "PENDING",
        "iteration_count": 0,
        "max_iterations": 22,
        "diagnostic_done": False,
        "precheck_done": False,
        "diagnostic_approved": False,
        "stock_approved": False,
        "escalation_to_engineer": False,
        "engineering_replan_done": False,
        "maintenance_approved": False,
        "clearance_approved": False,
        "analyst_approved": False,
        "protocol_closed": False,
        "escalated": False,
        "awaiting_human_approval": False,
        "pending_approval": {},
        "human_decisions": {},
        "vibration_test_count": 0,
        "repair_attempt_count": 0,
        "error_message": "",
        "diagnostic_report": {},
        "precheck_report": {},
        "inventory_report": {},
        "purchase_requisition": {},
        "engineering_replan": {},
        "maintenance_report": {},
        "clearance_report": {},
        "capa_report": {},
        "analytics_report": {},
        "rag_chunks": [],
        "sql_queries": [],
        "telemetry_readings": [],
        "maintenance_attempts": [],
        "tool_calls_log": [],
        "audit_trail": [],
        "agent_events": [],
        "agent_traces": [],
    }
