# Laser Parameter Optimizer

![Python](https://img.shields.io/badge/python-3.11-blue?logo=python)
![License](https://img.shields.io/badge/license-MIT-green)
![CI](https://github.com/Jimmply/laser-parameter-optimizer/workflows/CI/badge.svg)
![XGBoost](https://img.shields.io/badge/model-XGBoost+scipy-orange)

XGBoost surrogate model + scipy optimization that recommends pulsed Nd:YAG and LASAG SLS 200 laser welding parameters for a given material, target penetration depth, and joint type. Covers the full aerospace/medical material repertoire used at precision laser shops: Ti-6Al-4V, Inconel 625, 316L Stainless, Al 6061, Cobalt-Chromium, Hastelloy C-276, and Platinum.

Built from the operational experience of an AS9100-certified laser welding facility serving aerospace OEMs, medical device manufacturers, and space systems integrators.

---

## What it does

| Capability | Detail |
|---|---|
| **Parameter recommendation** | Given material + target depth → optimal power, pulse, frequency, speed, spot size |
| **Surrogate model** | XGBoost trained on 8,000 physics-informed weld trials per material |
| **scipy optimization** | L-BFGS-B search minimises defect probability while hitting penetration target |
| **Process window map** | Power vs speed scatter coloured by quality grade — visualise the operating envelope |
| **Material database** | 7 aerospace/medical alloys with absorptivity, thermal conductivity, melting point |

---

## Material Database

| Material | Absorptivity (1064 nm) | Preferred Gas | Key Challenge |
|---|---|---|---|
| Ti-6Al-4V | 0.52 | Argon | Oxidation — requires full argon purge |
| 316L Stainless | 0.35 | Ar/N₂ | Sensitization in HAZ |
| Al 6061-T6 | 0.08 | Argon | Low absorptivity — H₂ porosity risk |
| Cobalt-Chromium | 0.42 | Argon | Cr evaporation at high power |
| Inconel 625 | 0.45 | Argon | Low k → heat concentration → cracking |
| Hastelloy C-276 | 0.44 | Argon | High-temperature corrosion-resistant superalloy |
| Platinum | 0.38 | Argon | Medical/scientific — high melting point |

---

## Model Performance

Evaluated on 20% held-out test split:

| Metric | Value |
|---|---|
| Penetration depth MAE | **~45 μm** (within ±10% of target for >80% of trials) |
| Quality grade accuracy | **~91%** |
| Defect probability MAE | **~0.06** |

Top predictors: `power_w` → `absorptivity` → `thermal_cond` → `pulse_ms`

---

## Quickstart

```bash
git clone https://github.com/Jimmply/laser-parameter-optimizer
cd laser-parameter-optimizer
pip install -r requirements.txt
streamlit run src/app.py
```

**CLI — generate training data:**
```bash
python scripts/generate_data.py --n-samples 10000 --output data/weld_trials.csv
```

**CLI — train and save model:**
```bash
python scripts/train.py --output models/
```

---

## Project Structure

```
laser-parameter-optimizer/
├── .github/workflows/ci.yml
├── config/settings.yaml
├── data/
├── models/
├── scripts/
│   ├── generate_data.py
│   └── train.py
├── src/
│   ├── materials.py       # 7-material property database + Rosenthal penetration model
│   ├── data_generator.py  # Physics-informed weld trial simulation
│   ├── optimizer.py       # XGBoost surrogates + scipy L-BFGS-B optimization
│   └── app.py             # Streamlit dashboard
├── tests/
│   └── test_generator.py
└── pyproject.toml
```

---

## Methodology

**Physics model** — Penetration depth is estimated via a simplified Rosenthal heat-flow equation: `depth ∝ (power × duty_cycle × absorptivity) / (thermal_conductivity × speed)`. This gives each material a distinct operating envelope that the surrogate model learns.

**Defect probability** — Rule-based model encoding known failure mechanisms: excessive energy density → spatter/burn-through; insufficient power × absorptivity → cold weld; keyhole instability (high power density at small spot) → porosity; high travel speed → incomplete fusion.

**Optimization** — `scipy.optimize.minimize` (L-BFGS-B) searches the bounded parameter space to minimize `defect_probability + 2 × (penetration_error / target)²`, ensuring the recommended parameters hit the depth target while staying in the low-defect regime.

---

## Business Value

In a production laser welding cell, parameter selection for a new material/joint combination currently requires 3–8 trial welds costing 1–3 hours of machine time. An ML-guided starting point typically reduces this to 1–2 trials, recovering **2–6 hours per new job setup** across a shop running 20+ new setups per month.

---

## Tech Stack

Python 3.11 · XGBoost · Scikit-learn · SciPy · Pandas · NumPy · Streamlit · Plotly · Joblib
