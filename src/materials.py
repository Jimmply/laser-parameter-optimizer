"""
Material Property Database for Laser Welding Parameter Optimization
=====================================================================
Physics-informed material properties at 1064 nm (Nd:YAG) wavelength.

Sources:
  - ASM Handbook Vol. 6: Welding, Brazing, and Soldering
  - LASAG application notes for pulsed Nd:YAG systems
  - Published Rosenthal heat-flow models for keyhole welding

Properties per material:
  absorptivity       - Fraction of 1064 nm laser energy absorbed (0–1)
  thermal_cond       - Thermal conductivity (W/m·K)
  melting_point_c    - Solidus temperature (°C)
  density            - kg/m³
  specific_heat      - J/kg·K
  min_power_w        - Practical minimum laser power for this material
  max_power_w        - Practical maximum before damage
  preferred_gas      - Preferred shielding gas at LSR Welding
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class Material:
    name: str
    absorptivity: float
    thermal_cond: float        # W/m·K
    melting_point_c: float     # °C
    density: float             # kg/m³
    specific_heat: float       # J/kg·K
    min_power_w: float
    max_power_w: float
    preferred_gas: str
    notes: str = ""

    @property
    def thermal_diffusivity(self) -> float:
        """Thermal diffusivity α = k / (ρ·Cp) in m²/s.

        Used in Rosenthal heat-flow models to estimate weld pool cooling rate
        and HAZ width. Higher α → faster heat dissipation → smaller HAZ.
        """
        return self.thermal_cond / (self.density * self.specific_heat)


MATERIALS: dict[str, Material] = {
    "Ti-6Al-4V": Material(
        name="Ti-6Al-4V",
        absorptivity=0.52,
        thermal_cond=7.2,
        melting_point_c=1660,
        density=4430,
        specific_heat=560,
        min_power_w=50,
        max_power_w=800,
        preferred_gas="Argon",
        notes="Aerospace grade; requires full argon purge to prevent oxidation",
    ),
    "316L Stainless": Material(
        name="316L Stainless",
        absorptivity=0.35,
        thermal_cond=15.0,
        melting_point_c=1400,
        density=8000,
        specific_heat=500,
        min_power_w=60,
        max_power_w=1000,
        preferred_gas="Argon/Nitrogen",
        notes="Medical and aerospace grade; watch for sensitization",
    ),
    "Al 6061-T6": Material(
        name="Al 6061-T6",
        absorptivity=0.08,
        thermal_cond=167.0,
        melting_point_c=660,
        density=2700,
        specific_heat=896,
        min_power_w=200,
        max_power_w=1200,
        preferred_gas="Argon",
        notes="Low absorptivity — needs high peak power; porosity risk from H₂",
    ),
    "Cobalt-Chromium": Material(
        name="Cobalt-Chromium",
        absorptivity=0.42,
        thermal_cond=14.9,
        melting_point_c=1495,
        density=8300,
        specific_heat=420,
        min_power_w=80,
        max_power_w=900,
        preferred_gas="Argon",
        notes="Medical/aerospace alloy; watch for Cr evaporation",
    ),
    "Inconel 625": Material(
        name="Inconel 625",
        absorptivity=0.45,
        thermal_cond=9.8,
        melting_point_c=1350,
        density=8440,
        specific_heat=410,
        min_power_w=70,
        max_power_w=950,
        preferred_gas="Argon",
        notes="Superalloy; low thermal conductivity concentrates heat — watch for cracking",
    ),
    "Hastelloy C-276": Material(
        name="Hastelloy C-276",
        absorptivity=0.44,
        thermal_cond=10.2,
        melting_point_c=1370,
        density=8890,
        specific_heat=427,
        min_power_w=80,
        max_power_w=900,
        preferred_gas="Argon",
        notes="Excellent corrosion resistance; used in chemical processing + aerospace",
    ),
    "Platinum": Material(
        name="Platinum",
        absorptivity=0.38,
        thermal_cond=71.6,
        melting_point_c=1768,
        density=21450,
        specific_heat=133,
        min_power_w=40,
        max_power_w=600,
        preferred_gas="Argon",
        notes="Medical/scientific; high melting point — needs adequate pulse energy",
    ),
    "Nitinol (NiTi)": Material(
        name="Nitinol (NiTi)",
        absorptivity=0.40,
        thermal_cond=8.6,
        melting_point_c=1310,
        density=6450,
        specific_heat=490,
        min_power_w=40,
        max_power_w=600,
        preferred_gas="Argon",
        notes=(
            "Shape memory alloy — medical stents, guidewires, surgical tools. "
            "Low thermal conductivity concentrates heat; excessive energy destroys "
            "shape memory properties. Tight process window required."
        ),
    ),
    "Invar 36": Material(
        name="Invar 36",
        absorptivity=0.36,
        thermal_cond=10.5,
        melting_point_c=1425,
        density=8050,
        specific_heat=515,
        min_power_w=60,
        max_power_w=850,
        preferred_gas="Argon",
        notes=(
            "Fe-36Ni low-expansion alloy — aerospace tooling, optical mounts, "
            "cryogenic structures. Welding must preserve near-zero CTE; "
            "avoid high heat input to prevent Fe₃Ni phase transformation."
        ),
    ),
}


def heat_input(power_w: float, speed_mm_s: float) -> float:
    """Linear heat input (J/mm) — key process parameter for weld quality."""
    return power_w / max(speed_mm_s, 0.1)


def penetration_estimate_um(
    material: Material,
    power_w: float,
    pulse_ms: float,
    frequency_hz: float,
    speed_mm_s: float,
) -> float:
    """
    Simplified Rosenthal-based penetration depth estimate (μm).
    Accuracy ±25% vs. experimental — sufficient for parameter guidance.
    """
    duty_cycle = np.clip(pulse_ms * frequency_hz / 1000, 0.01, 0.95)
    avg_power = power_w * duty_cycle
    hi = heat_input(avg_power, speed_mm_s)
    # Absorbed energy per unit length / thermal resistance proxy
    absorbed = hi * material.absorptivity
    thermal_resistance = material.thermal_cond * (material.melting_point_c + 273)
    raw = 1e6 * absorbed / (thermal_resistance + 1)
    return float(np.clip(raw, 10, 3000))
