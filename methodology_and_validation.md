# Revised Methodology (Aligned to Current Pipeline)

## Research Design
This study adopts a quantitative, exploratory-computational research design to evaluate a hybrid analytic pipeline for disaster risk prioritization at the barangay level in Metro Manila. The software artifact demonstrates that a hybrid deterministic-probabilistic approach can produce internally consistent risk prioritizations from publicly available hazard and demographic data, even in the absence of ground-truth disaster impact labels.

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

### Data Limitations
The following data quality caveats must be acknowledged:

1. **Modeled vs. observed hazard data**: Project NOAH flood layers are *modeled* hazard maps based on hydrological simulations, not direct observations of historical inundation events. Their accuracy depends on digital elevation model (DEM) resolution, hydrological model assumptions, and calibration data quality. Results should be interpreted as reflecting modeled flood susceptibility rather than empirically confirmed flood extents.

2. **Temporal gap in demographic data**: PSA population figures are drawn from the 2020 Census. Given rapid urbanization in parts of Metro Manila, actual 2025-2026 populations in some barangays may differ materially from 2020 estimates. This temporal lag may cause the vulnerability component to under- or over-estimate current exposure for fast-growing or depopulating barangays.

3. **Spatial resolution mismatches**: Flood hazard layers and barangay administrative boundaries originate from different geospatial sources with potentially different coordinate systems, resolutions, and boundary definitions. The intersection of flood polygons with barangay polygons may introduce edge-case artifacts for barangays only partially covered by flood layers. No sub-barangay inundation depth differentiation is available in the current dataset.

4. **No damage or casualty records**: The dataset contains no event-level impact data (e.g., reported damages, affected persons per flood event, casualties). All risk classifications are therefore derived from modeled exposure and demographic proxies, not empirical outcomes.

## Feature Engineering
To capture both hazard exposure and social vulnerability, the following engineered variables are computed. A subset of these features is selected for model input based on relevance and collinearity considerations:

- **Flood coverage ratios** per return period (flood area / land area), clipped to [0, 1] as a defensive measure against potential GIS overlay artifacts where flood polygons may slightly exceed barangay boundaries
- **Population density** (population / land area)
- **Affected population** interaction terms (population × flood coverage)
- **Flood growth rate** between scenarios (e.g., 5-year to 25-year)

These features operationalize the intuition that risk increases when high exposure co-occurs with high vulnerability.

### Feature Selection Rationale
Not all engineered features are used as model inputs. Features are selected to balance informativeness with redundancy reduction. For instance, `flood25_coverage` is excluded from the model feature set because it exhibits high collinearity with `flood5_coverage` and `flood100_coverage` (Spearman ρ > 0.9 in both cases), and its inclusion does not improve model performance. Similarly, only `Affected_Pop_100yr` is retained as a representative interaction term, since the three affected-population features are highly correlated. The final feature set used for modeling is:

1. `Pop_Density` — social vulnerability proxy
2. `flood5_coverage` — frequent-event hazard exposure
3. `flood100_coverage` — extreme-event hazard exposure
4. `Affected_Pop_100yr` — population-hazard interaction (worst-case scenario)
5. `Flood_Growth_5to25` — hazard escalation rate
6. `Land Area` — spatial scale control variable

## Deterministic Benchmarking: Disaster Priority Index (DPI)
Because externally observed outcome labels (e.g., event inundation confirmations, damages, casualties) are not available, this study defines a **deterministic benchmark index** used as a proxy target for supervised learning.

1. **Composite Susceptibility Index (CSI)**
   - A weighted combination of flood coverage scores across return periods.
   - Weights reflect increasing severity scenarios and are set a priori (50% for 5-year, 30% for 25-year, 20% for 100-year), following the principle that higher-frequency events contribute more to annual expected risk exposure, consistent with probabilistic risk frameworks used by PAGASA and international disaster risk literature (UNDRR, 2015).
   - **Flood coverage scoring thresholds** (≤5% → 0, >5% → 2.5, ≥20% → 5.0, ≥50% → 7.5, ≥80% → 10.0) are informed by flood hazard mapping conventions from DOST-Project NOAH and PAGASA, where 5% coverage represents negligible localized ponding, 20% corresponds to significant partial inundation requiring attention, 50% indicates majority-flooded conditions associated with severe impacts, and 80% represents near-total inundation. These ordinal tiers are consistent with the qualitative severity categories (negligible/minor/moderate/severe/catastrophic) used in Philippine disaster risk reduction and management (DRRM) frameworks. The step-function design is intentional for ordinal risk tiering; robustness to threshold selection is assessed through the DPI weight sensitivity analysis described below.

2. **Vulnerability Score**
   - A tiered (0-10) score derived from population density thresholds. Four non-zero tiers are defined: density < 0.02 persons/m² → score 2, ≥ 0.02 → 4, ≥ 0.04 → 7, ≥ 0.07 → 10 (score 0 is reserved for missing or zero density). Tier boundaries are calibrated to Metro Manila's urban density distribution: the 0.07 persons/m² threshold (~70,000/km²) corresponds approximately to the 90th percentile of barangay-level densities in NCR (PSA, 2020 Census), the 0.04 persons/m² threshold (~40,000/km²) approximates the 75th percentile, while the 0.02 persons/m² threshold (~20,000/km²) approximates the national urban median. These tiers are intended to distribute Metro Manila barangays across meaningful vulnerability gradations.
   - A score of 0 is assigned when population density data is missing or zero, ensuring that the DPI score can reach a true minimum of 0 for barangays with no population and no flood exposure.

3. **Final DPI**
   - A weighted combination of physical susceptibility and social vulnerability (60% CSI + 40% vulnerability), reflecting the established convention in multi-hazard risk indices where physical exposure typically receives majority weighting (e.g., INFORM Risk Index methodology).

### Sensitivity Analysis of DPI Weights
To assess the robustness of the DPI ranking to weight selection, a sensitivity analysis is conducted by varying the CSI component weights (e.g., 40/30/30, 50/30/20, 60/25/15) and the CSI-vulnerability balance (e.g., 50/50, 60/40, 70/30). The resulting DPI rankings are compared using Spearman rank correlation. If rank-order stability remains high (rho > 0.90) across configurations, the findings are considered robust to the specific weight choice. If not, weight-sensitivity is reported as a limitation.

### DPI-Derived Risk Classes (Proxy Labels)
For classification benchmarks, DPI is thresholded into ordinal risk categories using the following cutoffs: **Low** (DPI < 3.5), **Moderate** (3.5 ≤ DPI < 6.5), and **High** (DPI ≥ 6.5). These threshold values are set to produce a distribution that meaningfully separates the Metro Manila barangay population across risk tiers. These DPI-derived classes are treated as **proxy labels** (benchmark targets), not ground truth.

## Modeling: Hybrid Unsupervised + Supervised Analysis

### Unsupervised Phase (Exploratory)
**K-Means clustering** is applied to standardized engineered features to identify natural groupings in the data. The resulting clusters are interpreted as **"Risk Archetypes"** (e.g., high-density/high-exposure vs low-density/low-exposure).

- Purpose: exploratory structure discovery and qualitative profiling.
- Output: archetype groupings used for interpretation and reporting.
- Note: clusters are **not used as supervised training labels** in the current pipeline.

**Cluster count selection**: The number of clusters is determined empirically using the silhouette method, evaluating K in {2, 3, 4, 5, 6}. The value of K that yields the highest mean silhouette score is selected. Clustering stability is additionally verified by re-running K-Means across 10 random seeds and computing Adjusted Rand Index (ARI) agreement, confirming that the cluster structure is robust to initialization.

**Semantic archetype mapping**: The K clusters identified by silhouette-optimized K-Means are subsequently mapped to 4 semantic risk archetypes based on each cluster's centroid position in the Pop_Density × flood100_coverage (extreme-event exposure) feature space. Using the median of cluster centroid values as the decision boundary, each cluster is labeled along two axes — population density (HighDensity vs. LowDensity) and flood exposure (HighExposure vs. LowExposure) — yielding four possible archetype categories: HighDensity-HighExposure, HighDensity-LowExposure, LowDensity-HighExposure, and LowDensity-LowExposure. This two-step approach (statistical K selection → semantic consolidation) preserves the granularity of the optimal clustering solution while producing interpretable, policy-relevant categories. When K exceeds 4, multiple clusters may map to the same archetype label if their centroids fall in the same quadrant of the feature space. For instance, if silhouette maximization selects K=6, two or more clusters sharing similar density-exposure centroid profiles will be consolidated under a single archetype name, and the specific cluster-to-archetype mapping is reported transparently in the pipeline output.

**Archetype assignment limitations**: Because K-Means assigns labels based on cluster-level centroids rather than individual data point characteristics, individual barangays located near cluster decision boundaries may carry archetype labels that appear inconsistent with their own feature values (e.g., a barangay with high CSI assigned to a predominantly low-exposure cluster). This is an inherent property of centroid-based partitioning and is well-documented in clustering literature (Lloyd, 1982; Jain, 2010). Importantly, archetype assignments do not affect DPI scores or supervised risk classifications, which are computed independently of the clustering step.

**Visualization note**: The archetype scatter plot uses CSI — the weighted multi-return-period composite of 5-year, 25-year, and 100-year flood scores — as the y-axis (physical susceptibility), whereas archetype labels themselves are mapped using only Pop_Density × flood100_coverage centroids. Apparent mismatches between a barangay's position on the CSI axis and its archetype label can occur because CSI heavily weights the higher-frequency 5-year scenario, while the archetype mapping uses only the extreme-event (100-year) layer.

### Supervised Phase (Index Prediction)
Supervised models are trained to predict:
- **Classification**: DPI-derived risk class (Low/Moderate/High)
- **Regression**: the continuous DPI score

Benchmarked architectures include:
- **Random Forest** (bagging)
- **Gradient Boosting** (boosting)
- **Multi-Layer Perceptron (MLP)** (non-linear neural baseline)

The objective of supervised modeling here is to evaluate how well different learners can approximate the deterministic benchmark (DPI) and to identify non-linear feature interactions that may not be explicit in the hand-crafted index.

### Interpreting Supervised Performance on Index-Derived Targets
Because the DPI target is itself computed from a subset of the same engineered features available to the supervised models, high predictive accuracy is expected by construction. The research value of the supervised phase is therefore **not** in demonstrating that models can "predict risk" in the real-world sense, but rather in:
1. **Quantifying how much simpler models (baselines) can already capture the DPI mapping** — establishing a performance floor.
2. **Identifying whether non-linear learners discover feature interactions** beyond the hand-crafted index formula.
3. **Benchmarking relative model complexity vs. marginal accuracy gain**, informing whether the index formula alone is sufficient or whether ML adds interpretive value.

Near-perfect accuracy from complex models (e.g., Gradient Boosting, MLP) on this task should be expected and does not alone constitute a claim of real-world predictive validity.

## Baseline Comparisons (Fair and Methodology-Consistent)
To ensure fair comparison and to prevent overstating "prediction of real-world risk," baselines use the **same target source** (DPI and DPI-derived classes). The following baselines are implemented:

1. **Non-ML Baseline (Primary Reference)**
   - The **deterministic DPI rule** itself (and its thresholded class) is treated as the core benchmark.

2. **Simple Supervised Baselines**
   - **Classification**: majority-class (dummy) predictor, logistic regression, a shallow decision tree (max depth = 2), and an **ordinal baseline** ("Ordinal Ridge (rounded)"). Because DPI-derived risk classes are inherently ordered (Low < Moderate < High), an ordinal-aware baseline is included: a Ridge regression model is trained on the numeric ordinal encoding (0/1/2), and its continuous predictions are rounded to the nearest integer and clipped to [0, 2] to produce class labels. This baseline tests whether a simple linear model exploiting ordinal structure can approximate the DPI classification without an explicit multi-class formulation.
   - **Regression**: mean predictor (dummy) and Ridge regression.

All models — baselines and primary — are trained and evaluated on the same train/test split (80/20, stratified by class) to compare performance under identical conditions. To prevent data leakage, feature imputation and standardization are encapsulated within sklearn `Pipeline` objects, ensuring that preprocessing parameters are fit exclusively on training data within each split or cross-validation fold. Critically, **all baseline and primary models are evaluated under the same cross-validation protocols**: 5-fold stratified cross-validation (classification) and 5-fold KFold (regression) for standard generalization estimates, plus GroupKFold cross-validation grouped by city for spatial robustness checks. This ensures that baseline-to-primary comparisons are made under identical evaluation conditions across all protocols. Model improvement is reported as the gain over the dummy baselines (F1 gain for classification, RMSE reduction for regression), establishing a meaningful performance floor.

## Evaluation and Validation

### Predictive Metrics (Agreement with DPI)
Since targets are DPI-derived, all performance metrics quantify **agreement with the deterministic benchmark**, not external flood outcomes. Reported metrics should therefore be interpreted as measuring how well each model approximates the DPI index, not as indicators of real-world flood prediction accuracy.

The primary performance indicator is:
- **5-fold stratified cross-validated F1-score (weighted)**, reported as mean ± standard deviation across folds. This metric provides the most defensible generalization estimate because it reduces sensitivity to the specific train/test partition and evaluates model performance on held-out data across multiple splits.

**Classification metrics reported** (single-split test set, standard CV, and spatial CV):
- **Accuracy** — overall agreement rate on DPI-derived risk classes
- **Balanced Accuracy** — macro-averaged recall across classes, addressing potential class imbalance
- **F1-score (weighted)** — precision-recall harmonic mean weighted by class support
- **F1-score (macro)** — unweighted average across classes, ensuring minority-class performance is not masked
- **Confusion matrices** — reported for the held-out test split (rows = true labels, columns = predicted labels; label order: Low, Moderate, High) to enable inspection of per-class error patterns and systematic misclassification tendencies

**Regression metrics reported** (single-split test set, standard CV, and spatial CV):
- **RMSE** (root mean squared error) — penalizes large deviations
- **MAE** (mean absolute error) — provides a scale-interpretable average error magnitude

Cross-validated metrics are reported as **mean ± standard deviation** across folds for both baseline and primary models. Improvement over dummy baselines (F1 gain for classification, RMSE reduction for regression) is reported to contextualize model gains.

Full-dataset (resubstitution) agreement rates — where the model is evaluated on data that includes its training set — may be reported as a consistency check but must be explicitly labeled as resubstitution metrics. These should not be interpreted as estimates of generalization performance, as they include training data and will therefore overstate model accuracy.

### Unsupervised Diagnostics (Clustering Quality)
Clustering quality is reported using:
- **Silhouette score** (cohesion vs. separation; higher is better)
- **Davies-Bouldin index** (inter-cluster similarity; lower is better)
- **Calinski-Harabasz index** (variance ratio; higher is better)
- **Stability**: Adjusted Rand Index (ARI) agreement across multiple random initializations

### Logical/Domain Consistency Checks (Construct Validation)
Given the absence of externally labeled outcomes, validation includes **domain logic consistency** checks as a form of **construct validation**:
- High-risk flags should align with known flood-prone low-lying areas referenced in PAGASA and DOST-Project NOAH assessments (e.g., historically flood-affected barangays in Marikina, Malabon, Tondo).
- Low-risk flags should align with elevated or less exposed areas.

This step supports the methodological plausibility of the DPI construction. It is explicitly a **construct validation** exercise — confirming that the index produces outputs consistent with domain knowledge — and is not a substitute for ground-truth validation against observed flood events.

## Methodological Limitations
The following limitations must be acknowledged:

1. **No external ground truth**: The study does not use an external, event-validated ground truth label (e.g., observed inundation per barangay per event, damages, casualties). Supervised models learn to replicate and generalize a **proxy, index-derived target**. Results should be interpreted as benchmarking model agreement with DPI and extracting feature-driven insights, not as definitive prediction of real-world flood impacts.

2. **Circular target construction**: The DPI target is computed from features that overlap with the supervised model inputs. High accuracy is therefore expected by design and does not validate external predictive power. This circularity is mitigated by (a) explicitly benchmarking against dummy baselines to establish a performance floor, (b) reporting whether ML models capture interactions beyond the linear index formula, and (c) framing the supervised phase as index-approximation benchmarking rather than outcome prediction.

3. **Single train/test split supplemented by cross-validation**: The primary evaluation uses a single 80/20 stratified split. This is supplemented by **5-fold StratifiedKFold** cross-validation for classification and **5-fold KFold** for regression to assess performance stability. Additionally, **GroupKFold cross-validation grouped by city** is performed as a spatial robustness check, ensuring that all barangays from a given city are assigned to the same fold to prevent spatial leakage between train and test sets. All three evaluation protocols (single-split, standard CV, and spatial CV) are applied to **both baseline and primary models** under identical conditions. However, both standard and grouped approaches assume that group-level observations are independent, which may still be partially violated due to spatial autocorrelation across adjacent cities (see below).

4. **Weight sensitivity**: DPI weights are set a priori and may influence risk rankings. Sensitivity analysis (described above) is conducted to assess robustness, but the specific weight values are not empirically optimized against observed outcomes.

5. **Spatial autocorrelation**: Neighboring barangays likely exhibit correlated flood exposure and demographic characteristics, violating the i.i.d. assumption underlying standard cross-validation. GroupKFold cross-validation grouped by city (17 groups) is implemented as a spatial robustness check, but it is not equivalent to true spatial blocking: adjacent cities may still share correlated flood and demographic patterns, and district-level grouping is not performed. The degree to which residual spatial correlation inflates apparent model performance is not fully quantified.

6. **Ethical considerations**: Risk scores produced by this pipeline could influence resource allocation decisions. Over-reliance on modeled scores without ground-truth validation may lead to under-prioritization of genuinely at-risk communities or over-prioritization of areas with data artifacts. Users of these results should treat them as one input among multiple decision-support tools, not as definitive risk determinations. Equity implications of threshold-based classification (where small differences in DPI can shift a barangay between risk tiers) should be considered in policy applications.

7. **Comparison with existing frameworks**: The DPI index is constructed independently and has not been formally benchmarked against NDRRMC's existing risk classification system or other established Philippine disaster risk frameworks. Future work should compare DPI rankings against official NDRRMC risk assessments at the barangay level to establish convergent validity.

8. **Negative Flood_Growth_5to25 values**: Because flood scenario layers originate from independent hydrological model runs at different return periods, GIS overlay and resolution artifacts may cause `flood25_sqm_final` to be slightly smaller than `flood5_sqm_final` for some barangays, producing negative growth values. This does not affect DPI or CSI (which use coverage ratios, not growth), but should be noted as a data-quality caveat when interpreting the Flood_Growth_5to25 feature.

## Future Work (Ground Truth Path)
To convert DPI benchmarking into true outcome prediction, future work should incorporate event-based ground truth, such as:
- Remote-sensing inundation extents (e.g., radar-derived flood masks) intersected with barangay polygons
- Official barangay-level impact records (affected population, damages) aligned by time/event
- Validated hydrologic/hydraulic model outputs calibrated to observed events
- Refined spatial cross-validation (e.g., leave-one-district-out or distance-based spatial blocking) beyond the current city-level GroupKFold
- Formal benchmarking against NDRRMC risk classifications for convergent validity assessment
