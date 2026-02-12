# SoftEng–AlertaBayan — Flood Risk Modeling (Final Submission)

This repository contains the **barangay-level flood risk modeling workflow** used for analysis and reporting, including the notebook, a companion Python script, the methodology write-up, and the generated visualizations.

## What’s inside
- **Main notebook:** `barangay_flood_risk_modeling.ipynb`
- **Companion script:** `barangay_flood_risk_modeling.py`
- **Methodology & validation:** `methodology_and_validation.md`
- **Figures:** `*.png` (clustering, feature distributions, feature importance, diagnostics)
- **Final deliverables (included):**
  - `final_risk_assessment.pdf`
  - `final_methodology_aligned_results.xlsx`

## How to run
### 1) Install dependencies
```bash
pip install -r requirements.txt
```
2) Run the analysis
Notebook: open barangay_flood_risk_modeling.ipynb and run all cells, or
Script: 
```bash
python barangay_flood_risk_modeling.py
```

Reproducibility notes
Some results can vary slightly across machines due to library versions and randomness (where applicable).
This repo includes the final outputs used for submission (PDF/XLSX and key plots) for transparency.
