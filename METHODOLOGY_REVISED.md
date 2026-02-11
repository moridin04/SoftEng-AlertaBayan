# Revised Methodology (Aligned to Current Pipeline)

## Research Design
This study adopts a quantitative, experimental research design to evaluate a hybrid analytic pipeline for disaster risk prioritization at the barangay level in Metro Manila. The software artifact is used strictly as a computational proof-of-concept for methodology validation and benchmarking.

The workflow is sequential:
1. **Data Integration** (geospatial hazard + demographic attributes)
2. **Feature Engineering** (exposure, vulnerability, interaction terms)
3. **Deterministic Benchmarking** (rule-based index construction)
4. **Probabilistic Modeling** (unsupervised archetyping + supervised prediction of index-derived targets)

Importantly, the supervised learning stage is used to learn a mapping from engineered features to **index-derived targets** (DPI and DPI risk classes). In this design, supervised models **do not learn from externally observed flood outcomes** (ground truth) because such event/impact labels are not available in the present dataset.

## Data Sources
The study relies on secondary datasets integrated into a unified barangay-level table:
- **Geospatial Hazard Data**: Project NOAH flood hazard layers for multiple return periods (5-year, 25-year, 100-year), representing varying inundation likelihood/severity scenarios.
- **Demographic Data**: Philippine Statistics Authority (PSA) attributes, including total population and land area.

These inputs are merged into a combined dataset for analysis.

## Feature Engineering
To capture both hazard exposure and social vulnerability, engineered variables include:
- **Flood coverage ratios** per return period (flood area / land area)
- **Population density** (population / land area)
- **Affected population** interaction terms (population × flood coverage)
- **Flood growth rate** between scenarios (e.g., 5-year to 25-year)

These features operationalize the intuition that risk increases when high exposure co-occurs with high vulnerability.

## Deterministic Benchmarking: Disaster Priority Index (DPI)
Because externally observed outcome labels (e.g., event inundation confirmations, damages, casualties) are not available, this study defines a **deterministic benchmark index** used as a proxy target for supervised learning.

1. **Composite Susceptibility Index (CSI)**
   - A weighted combination of flood coverage scores across return periods.
   - Weights reflect increasing severity scenarios and are set a priori (e.g., 50% for 5-year, 30% for 25-year, 20% for 100-year).

2. **Vulnerability Score**
   - A tiered (1–10) score derived from population density thresholds consistent with Metro Manila urban density ranges.

3. **Final DPI**
   - A weighted combination of physical susceptibility and social vulnerability (e.g., 60% CSI + 40% vulnerability).

### DPI-Derived Risk Classes (Proxy Labels)
For classification benchmarks, DPI is thresholded into ordinal risk categories (e.g., **Low / Moderate / High**). These DPI-derived classes are treated as **proxy labels** (benchmark targets), not ground truth.

## Modeling: Hybrid Unsupervised + Supervised Analysis
### Unsupervised Phase (Exploratory)
**K-Means clustering** is applied to standardized engineered features to identify natural groupings in the data. The resulting clusters are interpreted as **“Risk Archetypes”** (e.g., high-density/high-exposure vs low-density/low-exposure).

- Purpose: exploratory structure discovery and qualitative profiling.
- Output: archetype groupings used for interpretation and reporting.
- Note: clusters are **not used as supervised training labels** in the current pipeline.

### Supervised Phase (Index Prediction)
Supervised models are trained to predict:
- **Classification**: DPI-derived risk class (Low/Moderate/High)
- **Regression**: the continuous DPI score

Benchmarked architectures include:
- **Random Forest** (bagging)
- **Gradient Boosting** (boosting)
- **Multi-Layer Perceptron (MLP)** (non-linear neural baseline)

The objective of supervised modeling here is to evaluate how well different learners can approximate the deterministic benchmark (DPI) and to identify non-linear feature interactions that may not be explicit in the hand-crafted index.

## Baseline Comparisons (Fair and Methodology-Consistent)
To ensure fair comparison and to prevent overstating “prediction of real-world risk,” baselines must use the **same target source** (DPI and DPI-derived classes). Recommended baselines:

1. **Non-ML Baseline (Primary Reference)**
   - The **deterministic DPI rule** itself (and its thresholded class) is treated as the core benchmark.

2. **Simple Supervised Baselines (Optional but Strongly Recommended)**
   - **Classification**: majority-class (dummy) predictor; logistic regression or a shallow decision tree.
   - **Regression**: mean predictor (dummy); linear regression/ridge.

All models are trained and evaluated on the same train/test split to compare performance under identical supervision.

## Evaluation and Validation
### Predictive Metrics (Agreement with DPI)
Since targets are DPI-derived, performance metrics quantify **agreement with the deterministic benchmark**, not external flood outcomes:
- **Accuracy** (overall agreement on DPI classes)
- **F1-score (weighted)** to reduce tolerance for false negatives in the high-risk class
- **RMSE** for numeric DPI prediction

### Unsupervised Diagnostics (Clustering Quality)
Where applicable, clustering is reported using standard diagnostics such as silhouette score, Davies–Bouldin index, and stability checks.

### Logical/Domain Consistency Checks
Given the experimental nature and lack of externally labeled outcomes, validation includes **domain logic consistency** checks:
- High-risk flags should align with known flood-prone low-lying areas (e.g., historically flood-affected zones).
- Low-risk flags should align with elevated or less exposed areas.

This step supports methodological plausibility of the DPI construction and outputs, but it should be explicitly described as **construct validation** rather than ground-truth validation.

## Methodological Limitation (Must Be Stated)
The study does not use an external, event-validated ground truth label (e.g., observed inundation per barangay per event, damages, casualties). Therefore:
- Supervised models learn to replicate and generalize a **proxy, index-derived target**.
- Results should be interpreted as benchmarking model agreement with DPI and extracting feature-driven insights, not as definitive prediction of real-world flood impacts.

## Future Work (Ground Truth Path)
To convert DPI benchmarking into true outcome prediction, future work should incorporate event-based ground truth, such as:
- Remote-sensing inundation extents (e.g., radar-derived flood masks) intersected with barangay polygons
- Official barangay-level impact records (affected population, damages) aligned by time/event
- Validated hydrologic/hydraulic model outputs calibrated to observed events
