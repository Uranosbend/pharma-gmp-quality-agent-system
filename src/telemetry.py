"""Continuous Multi-Axis Plant Telemetry Stream & Factory Sensor Adapter Interface."""

from __future__ import annotations

import math
import random
from abc import ABC, abstractmethod
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from src.state import DEMO_SCENARIOS


def classify_iso_20816_zone(rms: float) -> str:
    """Classify RMS velocity according to ISO 20816-3 Group 2 Rigid + Plant 04 cGMP 2.5 limit."""
    if rms < 1.4:
        return "Zone A (Good)"
    if rms <= 2.5:
        return "Zone B (GMP Acceptable)"
    if rms <= 4.5:
        return "Zone C (Alert)"
    return "Zone D (Danger)"


class BaseTelemetryAdapter(ABC):
    """Abstract adapter enabling seamless swap between Simulated Physics and OPC-UA / MQTT."""

    @abstractmethod
    def step_tick(self) -> Optional[str]:
        """Advance telemetry by 1 tick; return tripped scenario_id if >=2 consecutive ticks exceed 2.5 mm/s."""

    @abstractmethod
    def get_history(self, machine_id: str) -> List[Dict[str, Any]]:
        """Return rolling multi-sensor history list for the given machine_id."""

    @abstractmethod
    def get_latest_all(self) -> Dict[str, Dict[str, Any]]:
        """Return latest live telemetry reading dictionary keyed by machine_id."""

    @abstractmethod
    def inject_fault(self, scenario_id: str, progressive: bool = True) -> Optional[str]:
        """Inject any of the 9 factory faults (either progressive ramp or immediate trip)."""

    @abstractmethod
    def mark_repaired(self, machine_id: str, post_rms: float) -> None:
        """Transition an equipment bay back into healthy post-repair operating envelope."""

    @abstractmethod
    def reset(self) -> None:
        """Reset all bays and rolling buffers back to clean baseline operation."""


class SimulatedPlantTelemetryAdapter(BaseTelemetryAdapter):
    """Multi-harmonic stochastic telemetry simulator with normal vs high-turbulence variance modes."""

    _BASELINES: Dict[str, Dict[str, Any]] = {
        "TP-01": {
            "machine_name": "TP-01 Rotary Tablet Press (Fette-Class)",
            "nominal_rms": 1.12,
            "freq_hz": 45.0,
            "temp_c": 48.2,
            "motor_amps": 19.2,
            "default_scenario": "scenario_1",
        },
        "RP-04": {
            "machine_name": "RP-04 Sanitary Centrifugal Pump (3-A)",
            "nominal_rms": 1.28,
            "freq_hz": 24.2,
            "temp_c": 42.6,
            "motor_amps": 15.0,
            "default_scenario": "scenario_2",
        },
        "PK-02": {
            "machine_name": "PK-02 Servo Blister Packaging Line",
            "nominal_rms": 1.18,
            "freq_hz": 120.0,
            "temp_c": 45.4,
            "motor_amps": 14.1,
            "default_scenario": "scenario_3",
        },
    }

    def __init__(
        self,
        history_maxlen: int = 45,
        turbulence_mode: str = "normal",
        seed: Optional[int] = None,
        auto_anomaly_enabled: bool = False,
    ) -> None:
        self.history_maxlen = history_maxlen
        self._rng = random.Random(seed)
        self._tick = 0
        self._ticks_since_fault = 0
        self._turbulence_mode = "high" if str(turbulence_mode).lower() == "high" else "normal"
        self._auto_anomaly_enabled = bool(auto_anomaly_enabled)
        self.reset()

    def set_turbulence_mode(self, mode: str) -> None:
        self._turbulence_mode = "high" if str(mode).lower() == "high" else "normal"

    def get_turbulence_mode(self) -> str:
        return self._turbulence_mode

    def set_auto_anomaly(self, enabled: bool) -> None:
        self._auto_anomaly_enabled = bool(enabled)

    def get_auto_anomaly(self) -> bool:
        return self._auto_anomaly_enabled

    def reset(self) -> None:
        self._tick = 0
        self._ticks_since_fault = 0
        self._history: Dict[str, List[Dict[str, Any]]] = {mid: [] for mid in self._BASELINES}
        self._bays: Dict[str, Dict[str, Any]] = {}
        for idx, (mid, cfg) in enumerate(self._BASELINES.items()):
            nom = cfg["nominal_rms"]
            self._bays[mid] = {
                "machine_id": mid,
                "machine_name": cfg["machine_name"],
                "rms": nom,
                "radial_x": round(nom * 0.92, 2),
                "radial_y": round(nom * 1.04, 2),
                "axial_z": round(nom * 0.58, 2),
                "peak_g": 0.42,
                "kurtosis": 2.95,
                "freq_hz": cfg["freq_hz"],
                "temp_c": cfg["temp_c"],
                "motor_amps": cfg["motor_amps"],
                "zone": classify_iso_20816_zone(nom),
                "status": "HEALTHY",
                "active_scenario_id": None,
                "ramp_step": 0,
                "consecutive_above": 0,
                "alarm_triggered": False,
                "phase": idx * 1.1,
            }
        # Pre-seed 16 clean healthy ticks without spawning autonomous anomalies during boot
        orig_auto = self._auto_anomaly_enabled
        self._auto_anomaly_enabled = False
        for _ in range(16):
            self.step_tick()
        self._auto_anomaly_enabled = orig_auto
        self._ticks_since_fault = 0

    def _compute_next_reading(self, mid: str, bay: Dict[str, Any]) -> Dict[str, Any]:
        base = self._BASELINES[mid]
        mult = 1.85 if self._turbulence_mode == "high" else 1.0
        t = self._tick * 0.65 + bay["phase"]
        now_str = datetime.now(timezone.utc).strftime("%H:%M:%S")

        sc_id = bay["active_scenario_id"]
        if sc_id and sc_id in DEMO_SCENARIOS:
            sc = DEMO_SCENARIOS[sc_id]
            step = bay["ramp_step"]
            raw_target = float(sc["initial_vibration_rms"])
            if step == 1:
                frac = 0.22
                target_rms = min(2.35, 1.45 + (raw_target - 1.3) * 0.12) + self._rng.gauss(0.0, 0.05)
            elif step == 2:
                frac = 0.46
                target_rms = max(2.85, raw_target * 0.44) + self._rng.gauss(0.0, 0.08 * mult)
            else:
                frac = 1.0
                target_rms = raw_target + self._rng.gauss(0.0, 0.14 * mult)
            rms = round(max(1.6, target_rms), 2)
            # Axial component dominates on alignment/soft-foot/coupling scenarios
            if sc_id in ("scenario_rp04_softfoot", "scenario_2"):
                axial_z = round(rms * 0.96 + self._rng.uniform(-0.08, 0.12), 2)
                radial_x = round(rms * 0.78 + self._rng.uniform(-0.08, 0.08), 2)
                radial_y = round(rms * 0.82 + self._rng.uniform(-0.08, 0.08), 2)
            else:
                radial_x = round(rms * 1.02 + self._rng.uniform(-0.10, 0.10), 2)
                radial_y = round(rms * 0.95 + self._rng.uniform(-0.10, 0.10), 2)
                axial_z = round(rms * 0.66 + self._rng.uniform(-0.06, 0.08), 2)

            peak_g = round(max(0.5, float(sc.get("peak_accel_g", 3.5)) * frac + self._rng.gauss(0.0, 0.10 * mult)), 2)
            kurtosis = round(max(2.8, 3.0 + (float(sc.get("kurtosis", 5.2)) - 3.0) * frac + self._rng.gauss(0.0, 0.10)), 2)
            freq_hz = round(float(sc["dominant_frequency_hz"]) + self._rng.uniform(-0.4, 0.4), 1)
            temp_c = round(base["temp_c"] + (float(sc["temperature_c"]) - base["temp_c"]) * frac + self._rng.uniform(-0.2, 0.2), 1)
            motor_amps = round(
                base["motor_amps"] + (float(sc.get("motor_amps", base["motor_amps"])) - base["motor_amps"]) * frac + self._rng.gauss(0.0, 0.20),
                1,
            )
            zone = classify_iso_20816_zone(rms)
            status = "ALARM" if rms > 2.5 else "WARNING"
        else:
            # Healthy multi-harmonic + stochastic operational load variance
            wave = 0.09 * math.sin(t) + 0.05 * math.cos(t * 1.73)
            noise = self._rng.gauss(0.0, 0.055 * mult)
            pulse = (0.14 * mult) if (self._tick % 7 == 0 and self._rng.random() < 0.35) else 0.0
            # Under autonomous monitoring, allow rare single-tick load transients >2.5 that self-recover (demonstrating 2-tick filter)
            if self._auto_anomaly_enabled and bay["consecutive_above"] == 0 and self._rng.random() < 0.08:
                rms = round(self._rng.uniform(2.52, 2.65), 2)
                status = "WARNING"
            else:
                rms = round(min(2.15, max(0.75, base["nominal_rms"] + (wave * mult) + noise + pulse)), 2)
                status = "HEALTHY"
            radial_x = round(max(0.65, rms * self._rng.uniform(0.91, 1.05)), 2)
            radial_y = round(max(0.65, rms * self._rng.uniform(0.89, 1.06)), 2)
            axial_z = round(max(0.45, rms * self._rng.uniform(0.54, 0.68)), 2)
            peak_g = round(max(0.25, min(1.25, 0.40 + 0.09 * mult * abs(math.sin(t * 1.3)) + self._rng.uniform(0.0, 0.06))), 2)
            kurtosis = round(min(3.55, max(2.70, 2.96 + self._rng.gauss(0.0, 0.09 * mult))), 2)
            freq_hz = round(base["freq_hz"] + 0.3 * math.sin(t) + self._rng.uniform(-0.2, 0.2), 1)
            temp_c = round(base["temp_c"] + 0.4 * math.sin(t * 0.4) + self._rng.uniform(-0.15, 0.15), 1)
            motor_amps = round(base["motor_amps"] + 0.35 * mult * math.sin(t * 0.8) + self._rng.uniform(-0.15, 0.15), 1)
            zone = classify_iso_20816_zone(rms)

        return {
            "tick": self._tick,
            "time": now_str,
            "timestamp": now_str,
            "machine_id": mid,
            "machine_name": base["machine_name"],
            "rms": rms,
            "radial_x": radial_x,
            "radial_y": radial_y,
            "axial_z": axial_z,
            "peak_g": peak_g,
            "kurtosis": kurtosis,
            "freq_hz": freq_hz,
            "temp_c": temp_c,
            "motor_amps": motor_amps,
            "zone": zone,
            "status": status,
            "active_scenario_id": sc_id,
            "alarm_triggered": bay["alarm_triggered"],
            "turbulence_mode": self._turbulence_mode,
        }

    def step_tick(self) -> Optional[str]:
        self._tick += 1
        # Natural autonomous wear & fault emergence when monitoring live floor without an open fault
        any_active_fault = any(bool(b.get("active_scenario_id")) for b in self._bays.values())
        if self._auto_anomaly_enabled and not any_active_fault:
            self._ticks_since_fault += 1
            min_t = 7 if self._turbulence_mode == "high" else 11
            max_t = 12 if self._turbulence_mode == "high" else 18
            prob = 0.38 if self._turbulence_mode == "high" else 0.24
            if self._ticks_since_fault >= min_t and (self._ticks_since_fault >= max_t or self._rng.random() < prob):
                chosen_sc = self._rng.choice(list(DEMO_SCENARIOS.keys()))
                target_mid = DEMO_SCENARIOS[chosen_sc]["machine_id"]
                target_bay = self._bays[target_mid]
                target_bay["active_scenario_id"] = chosen_sc
                target_bay["ramp_step"] = 0
                target_bay["consecutive_above"] = 0
                target_bay["alarm_triggered"] = False
                self._ticks_since_fault = 0

        tripped_scenario: Optional[str] = None
        for mid, bay in self._bays.items():
            if bay["active_scenario_id"] and bay["ramp_step"] < 3:
                bay["ramp_step"] += 1
            reading = self._compute_next_reading(mid, bay)
            bay.update(
                {
                    "rms": reading["rms"],
                    "radial_x": reading["radial_x"],
                    "radial_y": reading["radial_y"],
                    "axial_z": reading["axial_z"],
                    "peak_g": reading["peak_g"],
                    "kurtosis": reading["kurtosis"],
                    "freq_hz": reading["freq_hz"],
                    "temp_c": reading["temp_c"],
                    "motor_amps": reading["motor_amps"],
                    "zone": reading["zone"],
                    "status": reading["status"],
                }
            )
            if reading["rms"] > 2.5:
                bay["consecutive_above"] += 1
                if bay["consecutive_above"] >= 2 and not bay["alarm_triggered"] and bay["active_scenario_id"]:
                    bay["alarm_triggered"] = True
                    reading["alarm_triggered"] = True
                    tripped_scenario = bay["active_scenario_id"]
            else:
                bay["consecutive_above"] = 0

            self._history[mid].append(reading)
            if len(self._history[mid]) > self.history_maxlen:
                self._history[mid] = self._history[mid][-self.history_maxlen :]
        return tripped_scenario

    def inject_fault(self, scenario_id: str, progressive: bool = True) -> Optional[str]:
        if scenario_id not in DEMO_SCENARIOS:
            scenario_id = "scenario_1"
        sc = DEMO_SCENARIOS[scenario_id]
        mid = sc["machine_id"]
        bay = self._bays[mid]
        self._ticks_since_fault = 0
        bay["active_scenario_id"] = scenario_id
        bay["alarm_triggered"] = False
        bay["consecutive_above"] = 0
        bay["ramp_step"] = 0 if progressive else 3

        if progressive:
            # Progressive ramp waits for subsequent step_tick() calls (tick 1 warning -> tick 2 crossing -> tick 3 trip)
            return None

        # Immediate manual injection sets alarm immediately at 100% fault severity
        bay["consecutive_above"] = 2
        bay["alarm_triggered"] = True
        self._tick += 1
        reading = self._compute_next_reading(mid, bay)
        reading["alarm_triggered"] = True
        bay.update(
            {
                "rms": reading["rms"],
                "radial_x": reading["radial_x"],
                "radial_y": reading["radial_y"],
                "axial_z": reading["axial_z"],
                "peak_g": reading["peak_g"],
                "kurtosis": reading["kurtosis"],
                "freq_hz": reading["freq_hz"],
                "temp_c": reading["temp_c"],
                "motor_amps": reading["motor_amps"],
                "zone": reading["zone"],
                "status": "ALARM",
            }
        )
        self._history[mid].append(reading)
        if len(self._history[mid]) > self.history_maxlen:
            self._history[mid] = self._history[mid][-self.history_maxlen :]
        return scenario_id

    def mark_repaired(self, machine_id: str, post_rms: float) -> None:
        if machine_id not in self._bays:
            return
        self._ticks_since_fault = 0
        bay = self._bays[machine_id]
        base = self._BASELINES[machine_id]
        bay["active_scenario_id"] = None
        bay["ramp_step"] = 0
        bay["consecutive_above"] = 0
        bay["alarm_triggered"] = False
        rms_val = round(float(post_rms), 2)
        self._tick += 1
        now_str = datetime.now(timezone.utc).strftime("%H:%M:%S")
        reading = {
            "tick": self._tick,
            "time": now_str,
            "timestamp": now_str,
            "machine_id": machine_id,
            "machine_name": base["machine_name"],
            "rms": rms_val,
            "radial_x": round(rms_val * 0.94, 2),
            "radial_y": round(rms_val * 1.03, 2),
            "axial_z": round(rms_val * 0.60, 2),
            "peak_g": 0.44,
            "kurtosis": 2.95,
            "freq_hz": base["freq_hz"],
            "temp_c": base["temp_c"],
            "motor_amps": base["motor_amps"],
            "zone": classify_iso_20816_zone(rms_val),
            "status": "HEALTHY",
            "active_scenario_id": None,
            "alarm_triggered": False,
            "turbulence_mode": self._turbulence_mode,
        }
        bay.update(
            {
                "rms": reading["rms"],
                "radial_x": reading["radial_x"],
                "radial_y": reading["radial_y"],
                "axial_z": reading["axial_z"],
                "peak_g": reading["peak_g"],
                "kurtosis": reading["kurtosis"],
                "freq_hz": reading["freq_hz"],
                "temp_c": reading["temp_c"],
                "motor_amps": reading["motor_amps"],
                "zone": reading["zone"],
                "status": "HEALTHY",
            }
        )
        self._history[machine_id].append(reading)
        if len(self._history[machine_id]) > self.history_maxlen:
            self._history[machine_id] = self._history[machine_id][-self.history_maxlen :]

    def get_history(self, machine_id: str) -> List[Dict[str, Any]]:
        return list(self._history.get(machine_id, []))

    def get_latest_all(self) -> Dict[str, Dict[str, Any]]:
        return {mid: dict(bay) for mid, bay in self._bays.items()}


class OPCUATelemetryAdapterStub(BaseTelemetryAdapter):
    """Production OPC-UA integration stub ready for PLC/SCADA endpoints (e.g. Siemens S7 / Ignition)."""

    def __init__(self, endpoint_url: str = "opc.tcp://plant04-scada.local:4840") -> None:
        self.endpoint_url = endpoint_url

    def step_tick(self) -> Optional[str]:
        raise NotImplementedError(f"Connect python-asyncua client to {self.endpoint_url} for live PLC polling.")

    def get_history(self, machine_id: str) -> List[Dict[str, Any]]:
        raise NotImplementedError("OPC-UA historian query not wired in sandbox environment.")

    def get_latest_all(self) -> Dict[str, Dict[str, Any]]:
        raise NotImplementedError("OPC-UA node read not wired in sandbox environment.")

    def inject_fault(self, scenario_id: str, progressive: bool = True) -> Optional[str]:
        raise NotImplementedError("Fault injection disabled on live OPC-UA production endpoint.")

    def mark_repaired(self, machine_id: str, post_rms: float) -> None:
        raise NotImplementedError("Write-back via OPC-UA requires signed plant interlock.")

    def reset(self) -> None:
        pass


class MQTTTelemetryAdapterStub(BaseTelemetryAdapter):
    """Production MQTT Sparkplug B edge-gateway stub for wireless accelerometer gateways."""

    def __init__(self, broker_host: str = "mqtt://plant04-broker.local:1883") -> None:
        self.broker_host = broker_host

    def step_tick(self) -> Optional[str]:
        raise NotImplementedError(f"Subscribe paho-mqtt to {self.broker_host} SparkplugB telemetry topics.")

    def get_history(self, machine_id: str) -> List[Dict[str, Any]]:
        raise NotImplementedError("MQTT time-series cache not wired in sandbox environment.")

    def get_latest_all(self) -> Dict[str, Dict[str, Any]]:
        raise NotImplementedError("MQTT latest message cache not wired in sandbox environment.")

    def inject_fault(self, scenario_id: str, progressive: bool = True) -> Optional[str]:
        raise NotImplementedError("Synthetic injection disabled on live MQTT gateway.")

    def mark_repaired(self, machine_id: str, post_rms: float) -> None:
        raise NotImplementedError("MQTT telemetry clearance ack not wired in sandbox environment.")

    def reset(self) -> None:
        pass
