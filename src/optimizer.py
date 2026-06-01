"""
Laser Parameter Optimizer
==========================
XGBoost surrogate models predict weld outcomes from process parameters.
scipy.optimize then searches the parameter space for the combination that
achieves a target penetration depth while minimising defect probability.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from sklearn.metrics import mean_absolute_error, classification_report
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from xgboost import XGBClassifier, XGBRegressor

from data_generator import FEATURE_COLS
from materials import MATERIALS, Material

logger = logging.getLogger(__name__)


@dataclass
class OptimizeRequest:
    material_name: str
    target_penetration_um: float
    thickness_mm: float
    joint_type: str = "Butt"
    max_iterations: int = 200


@dataclass
class OptimizeResult:
    power_w: float
    pulse_ms: float
    frequency_hz: float
    travel_speed_mm_s: float
    spot_size_um: float
    predicted_penetration_um: float
    predicted_defect_prob: float
    quality_grade: str
    confidence: float


@dataclass
class TrainResults:
    penetration_mae_um: float
    defect_mae: float
    grade_report: str
    feature_importances: pd.Series


class LaserParameterOptimizer:
    """
    Surrogate-model optimizer for pulsed Nd:YAG laser welding parameters.

    1. Fit XGBoost regressors for penetration depth and defect probability.
    2. Fit XGBoost classifier for quality grade.
    3. Use scipy minimize to find parameters achieving target penetration
       at minimum defect probability for a given material.
    """

    def __init__(self) -> None:
        self._pen_reg   = XGBRegressor(n_estimators=300, max_depth=5, learning_rate=0.05, random_state=42, verbosity=0)
        self._def_reg   = XGBRegressor(n_estimators=300, max_depth=5, learning_rate=0.05, random_state=42, verbosity=0)
        self._grade_clf = XGBClassifier(n_estimators=300, max_depth=5, learning_rate=0.05, random_state=42, eval_metric="mlogloss", verbosity=0)
        self._le        = LabelEncoder()
        self._trained   = False

    def fit(self, df: pd.DataFrame) -> TrainResults:
        X = df[FEATURE_COLS].values
        y_pen   = df["penetration_um"].values
        y_def   = df["defect_prob"].values
        y_grade = self._le.fit_transform(df["quality_grade"])

        X_tr, X_te, yp_tr, yp_te, yd_tr, yd_te, yg_tr, yg_te = train_test_split(
            X, y_pen, y_def, y_grade, test_size=0.20, random_state=42
        )

        self._pen_reg.fit(X_tr, yp_tr)
        self._def_reg.fit(X_tr, yd_tr)
        self._grade_clf.fit(X_tr, yg_tr)
        self._trained = True

        pen_mae  = mean_absolute_error(yp_te, self._pen_reg.predict(X_te))
        def_mae  = mean_absolute_error(yd_te, self._def_reg.predict(X_te))
        report   = classification_report(yg_te, self._grade_clf.predict(X_te), target_names=self._le.classes_)
        fi       = pd.Series(self._pen_reg.feature_importances_, index=FEATURE_COLS).sort_values(ascending=True)

        logger.info("Trained. Penetration MAE=%.1f μm  Defect MAE=%.4f", pen_mae, def_mae)
        return TrainResults(pen_mae, def_mae, report, fi)

    def recommend(self, request: OptimizeRequest) -> OptimizeResult:
        """Find optimal parameters for the requested material and target penetration."""
        if not self._trained:
            raise RuntimeError("Call fit() before recommend().")

        mat = MATERIALS[request.material_name]

        # Build the fixed (non-optimized) material feature row
        def _build_x(params: np.ndarray) -> np.ndarray:
            power_w, pulse_ms, freq_hz, speed, spot = params
            return np.array([[
                power_w, pulse_ms, freq_hz, speed, spot,
                request.thickness_mm,
                mat.absorptivity, mat.thermal_cond, mat.melting_point_c,
            ]])

        def objective(params: np.ndarray) -> float:
            X = _build_x(params)
            pen   = float(self._pen_reg.predict(X)[0])
            defp  = float(self._def_reg.predict(X)[0])
            # Minimise: defect probability + squared error from target penetration
            pen_error = ((pen - request.target_penetration_um) / request.target_penetration_um) ** 2
            return defp + 2.0 * pen_error

        # Parameter bounds: [power, pulse_ms, freq_hz, speed_mm_s, spot_um]
        bounds = [
            (mat.min_power_w, mat.max_power_w),
            (0.5, 20.0),
            (1.0, 50.0),
            (0.5, 20.0),
            (40.0, 600.0),
        ]
        x0 = np.array([
            (mat.min_power_w + mat.max_power_w) / 2,
            5.0, 15.0, 5.0, 200.0,
        ])

        result = minimize(objective, x0, method="L-BFGS-B", bounds=bounds,
                         options={"maxiter": request.max_iterations})

        best = result.x
        X_best = _build_x(best)
        pen_pred  = float(self._pen_reg.predict(X_best)[0])
        def_pred  = float(np.clip(self._def_reg.predict(X_best)[0], 0, 1))
        grade_enc = self._grade_clf.predict(X_best)[0]
        grade     = self._le.inverse_transform([grade_enc])[0]
        proba     = self._grade_clf.predict_proba(X_best)[0].max()

        return OptimizeResult(
            power_w=round(float(best[0]), 1),
            pulse_ms=round(float(best[1]), 2),
            frequency_hz=round(float(best[2]), 1),
            travel_speed_mm_s=round(float(best[3]), 2),
            spot_size_um=round(float(best[4]), 0),
            predicted_penetration_um=round(pen_pred, 1),
            predicted_defect_prob=round(def_pred, 4),
            quality_grade=grade,
            confidence=round(float(proba), 3),
        )

    def sensitivity_analysis(
        self,
        request: OptimizeRequest,
        baseline: OptimizeResult,
        n_steps: int = 20,
    ) -> pd.DataFrame:
        """
        Vary each parameter ±20% around the baseline and record how
        predicted penetration and defect probability respond.

        Returns a DataFrame with columns:
            parameter, value, predicted_penetration_um, predicted_defect_prob
        Useful for identifying which parameters have the most leverage.
        """
        if not self._trained:
            raise RuntimeError("Call fit() before sensitivity_analysis().")

        mat = MATERIALS[request.material_name]
        base_vec = np.array([
            baseline.power_w, baseline.pulse_ms, baseline.frequency_hz,
            baseline.travel_speed_mm_s, baseline.spot_size_um,
        ])
        param_names = ["power_w", "pulse_ms", "frequency_hz", "travel_speed_mm_s", "spot_size_um"]
        bounds = [
            (mat.min_power_w, mat.max_power_w),
            (0.5, 20.0),
            (1.0, 50.0),
            (0.5, 20.0),
            (40.0, 600.0),
        ]

        rows = []
        for i, (name, (lo, hi)) in enumerate(zip(param_names, bounds)):
            sweep_lo = max(lo, base_vec[i] * 0.80)
            sweep_hi = min(hi, base_vec[i] * 1.20)
            for val in np.linspace(sweep_lo, sweep_hi, n_steps):
                vec = base_vec.copy()
                vec[i] = val
                X = np.array([[
                    vec[0], vec[1], vec[2], vec[3], vec[4],
                    request.thickness_mm,
                    mat.absorptivity, mat.thermal_cond, mat.melting_point_c,
                ]])
                pen  = float(self._pen_reg.predict(X)[0])
                defp = float(np.clip(self._def_reg.predict(X)[0], 0, 1))
                rows.append({
                    "parameter": name,
                    "value": round(val, 4),
                    "predicted_penetration_um": round(pen, 2),
                    "predicted_defect_prob": round(defp, 4),
                })

        return pd.DataFrame(rows)

    @property
    def classes_(self) -> list[str]:
        return list(self._le.classes_)
