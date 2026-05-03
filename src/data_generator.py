"""
Laser Parameter Optimization — Synthetic Training Data Generator
================================================================
Generates labeled weld process records using physics-informed rules
derived from pulsed Nd:YAG and LASAG SLS 200 operating windows.

Output columns:
  material, thickness_mm, joint_type,
  power_w, pulse_ms, frequency_hz, travel_speed_mm_s,
  shielding_gas, spot_size_um,
  penetration_um, haz_width_um, defect_prob,
  quality_grade (A / B / C / Reject)
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Optional

import numpy as np
import pandas as pd

from materials import MATERIALS, Material, heat_input, penetration_estimate_um

logger = logging.getLogger(__name__)

JOINT_TYPES = ["Butt", "Lap", "T-Joint", "Edge", "Spot"]
QUALITY_GRADES = ["A", "B", "C", "Reject"]

FEATURE_COLS = [
    "power_w", "pulse_ms", "frequency_hz",
    "travel_speed_mm_s", "spot_size_um", "thickness_mm",
    "absorptivity", "thermal_cond", "melting_point_c",
]
TARGET_COLS = ["penetration_um", "haz_width_um", "defect_prob", "quality_grade"]


@dataclass
class GenConfig:
    n_samples: int = 8000
    random_seed: int = 42


class LaserParamGenerator:
    """
    Generates synthetic laser welding trial records with physics-based outcomes.

    Each record represents one weld trial on the LASAG SLS 200 or FLS Nd:YAG,
    covering LSR Welding's full material repertoire.
    """

    def __init__(self, config: Optional[GenConfig] = None) -> None:
        self.config = config or GenConfig()
        self._rng = np.random.default_rng(self.config.random_seed)

    def generate(self) -> pd.DataFrame:
        n = self.config.n_samples
        material_names = list(MATERIALS.keys())
        rows = []

        for _ in range(n):
            mat_name = self._rng.choice(material_names)
            mat = MATERIALS[mat_name]
            row = self._sample_trial(mat)
            rows.append(row)

        df = pd.DataFrame(rows)
        logger.info(
            "Generated %d weld trials. Quality: %s",
            len(df),
            df["quality_grade"].value_counts().to_dict(),
        )
        return df

    # ------------------------------------------------------------------

    def _sample_trial(self, mat: Material) -> dict:
        rng = self._rng

        # Sample process parameters within realistic ranges
        power_w      = float(rng.uniform(mat.min_power_w, mat.max_power_w))
        pulse_ms     = float(rng.uniform(0.5, 20.0))
        frequency_hz = float(rng.uniform(1.0, 50.0))
        speed_mm_s   = float(rng.uniform(0.5, 25.0))
        spot_um      = float(rng.uniform(40, 600))
        thickness_mm = float(rng.uniform(0.1, 4.0))
        joint        = rng.choice(JOINT_TYPES)

        # Physics-derived outcomes
        pen_um = penetration_estimate_um(
            mat, power_w, pulse_ms, frequency_hz, speed_mm_s
        )
        # Add realistic noise
        pen_um += float(rng.normal(0, pen_um * 0.08))
        pen_um = max(5.0, pen_um)

        # HAZ width: proportional to heat input, inversely to thermal conductivity
        hi = heat_input(power_w, speed_mm_s)
        haz_um = float(np.clip(
            (hi / mat.thermal_cond) * 800 + rng.normal(0, 30),
            20, 2000,
        ))

        # Defect probability model
        defect_prob = self._defect_probability(
            mat, power_w, pulse_ms, frequency_hz, speed_mm_s, spot_um, thickness_mm, pen_um
        )

        quality_grade = self._grade(defect_prob, pen_um, thickness_mm)

        return {
            "material":           mat.name,
            "thickness_mm":       round(thickness_mm, 3),
            "joint_type":         joint,
            "power_w":            round(power_w, 1),
            "pulse_ms":           round(pulse_ms, 2),
            "frequency_hz":       round(frequency_hz, 1),
            "travel_speed_mm_s":  round(speed_mm_s, 2),
            "spot_size_um":       round(spot_um, 0),
            "shielding_gas":      mat.preferred_gas,
            "penetration_um":     round(pen_um, 1),
            "haz_width_um":       round(haz_um, 1),
            "defect_prob":        round(float(np.clip(defect_prob, 0, 1)), 4),
            "quality_grade":      quality_grade,
            # Flat material features for ML
            "absorptivity":       mat.absorptivity,
            "thermal_cond":       mat.thermal_cond,
            "melting_point_c":    mat.melting_point_c,
        }

    def _defect_probability(
        self,
        mat: Material,
        power_w: float, pulse_ms: float, frequency_hz: float,
        speed_mm_s: float, spot_um: float, thickness_mm: float,
        pen_um: float,
    ) -> float:
        hi = heat_input(power_w, speed_mm_s)
        norm_power = power_w / mat.max_power_w
        duty = np.clip(pulse_ms * frequency_hz / 1000, 0.01, 0.95)

        prob = 0.05  # base rate

        # Excessive energy → spatter / burn-through
        if norm_power > 0.85:
            prob += 0.30
        if duty > 0.70:
            prob += 0.20

        # Insufficient energy → lack of fusion / cold weld
        if norm_power < 0.20 and mat.absorptivity < 0.3:
            prob += 0.40  # aluminum cold weld risk
        if pen_um < thickness_mm * 300:  # less than 30% penetration
            prob += 0.25

        # Keyhole instability → porosity (high power density at small spot)
        power_density = power_w / (np.pi * (spot_um / 2e3) ** 2 + 1e-9)
        if power_density > 5e8:
            prob += 0.15

        # Titanium oxidation risk if wrong gas (simplified)
        if mat.name == "Ti-6Al-4V" and "Argon" not in mat.preferred_gas:
            prob += 0.35

        # High travel speed → poor fusion
        if speed_mm_s > 20 and hi < 5:
            prob += 0.20

        return float(np.clip(prob + self._rng.normal(0, 0.04), 0, 1))

    def _grade(self, defect_prob: float, pen_um: float, thickness_mm: float) -> str:
        penetration_ratio = pen_um / (thickness_mm * 1000 + 1)
        if defect_prob < 0.10 and penetration_ratio > 0.30:
            return "A"
        if defect_prob < 0.25 and penetration_ratio > 0.15:
            return "B"
        if defect_prob < 0.50:
            return "C"
        return "Reject"
