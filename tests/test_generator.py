import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from data_generator import FEATURE_COLS, TARGET_COLS, GenConfig, LaserParamGenerator
from materials import MATERIALS, heat_input, penetration_estimate_um


def test_generator_row_count():
    df = LaserParamGenerator(GenConfig(n_samples=200, random_seed=1)).generate()
    assert len(df) == 200


def test_generator_reproducible():
    a = LaserParamGenerator(GenConfig(n_samples=150, random_seed=7)).generate()
    b = LaserParamGenerator(GenConfig(n_samples=150, random_seed=7)).generate()
    pd.testing.assert_frame_equal(a, b)


def test_generator_columns_present():
    df = LaserParamGenerator(GenConfig(n_samples=50, random_seed=1)).generate()
    for col in FEATURE_COLS + TARGET_COLS:
        assert col in df.columns


def test_quality_grades_are_valid():
    df = LaserParamGenerator(GenConfig(n_samples=500, random_seed=2)).generate()
    assert set(df["quality_grade"].unique()) <= {"A", "B", "C", "Reject"}


def test_materials_have_positive_thermal_diffusivity():
    for name, mat in MATERIALS.items():
        assert mat.thermal_diffusivity > 0, f"{name} has non-positive thermal diffusivity"


def test_heat_input_scales_with_power():
    assert heat_input(400, 10) > heat_input(200, 10)
    assert heat_input(400, 10) < heat_input(400, 5)


def test_penetration_estimate_positive():
    mat = MATERIALS["Ti-6Al-4V"]
    p = penetration_estimate_um(mat, power_w=300, pulse_ms=5, frequency_hz=20, speed_mm_s=5)
    assert p > 0
