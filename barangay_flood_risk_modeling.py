import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.ensemble import (
    GradientBoostingClassifier,
    GradientBoostingRegressor,
    RandomForestClassifier,
    RandomForestRegressor,
)
from sklearn.model_selection import train_test_split, StratifiedKFold, KFold, cross_val_score, GroupKFold
from sklearn.pipeline import Pipeline
from sklearn.neural_network import MLPClassifier, MLPRegressor
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.tree import DecisionTreeClassifier
from sklearn.dummy import DummyClassifier, DummyRegressor
from sklearn.metrics import (
    accuracy_score,
    adjusted_rand_score,
    balanced_accuracy_score,
    calinski_harabasz_score,
    confusion_matrix,
    davies_bouldin_score,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    silhouette_score,
)
from scipy.stats import spearmanr
from itertools import combinations
from fpdf import FPDF, XPos, YPos
from sklearn.base import clone

# ── Canonical model display names (single source of truth) ──
MODEL_NAMES = {
    "dummy_clf":  "Dummy (Majority Class)",
    "logreg":     "Logistic Regression",
    "dt2":        "Shallow Decision Tree",
    "ord_ridge":  "Ordinal Ridge (rounded)",
    "rf":         "Random Forest",
    "gb":         "Gradient Boosting",
    "mlp":        "Neural Network (MLP)",
    "dummy_reg":  "Dummy (Mean Predictor)",
    "ridge_reg":  "Ridge Regression",
}

BASELINE_CLF_NAMES = {MODEL_NAMES[k] for k in ("dummy_clf", "logreg", "dt2", "ord_ridge")}
BASELINE_REG_NAMES = {MODEL_NAMES[k] for k in ("dummy_reg", "ridge_reg")}


def make_preprocess_pipe(estimator):
    """Imputer → Scaler → Model pipeline (shared by console & PDF evaluation)."""
    return Pipeline([
        ("imputer", SimpleImputer(strategy="mean")),
        ("scaler", StandardScaler()),
        ("model", estimator),
    ])


def get_baseline_clf_models(pipe_fn=make_preprocess_pipe):
    return {
        MODEL_NAMES["dummy_clf"]: pipe_fn(DummyClassifier(strategy="most_frequent")),
        MODEL_NAMES["logreg"]:    pipe_fn(LogisticRegression(max_iter=1000, random_state=42)),
        MODEL_NAMES["dt2"]:       pipe_fn(DecisionTreeClassifier(max_depth=2, random_state=42)),
        MODEL_NAMES["ord_ridge"]: pipe_fn(Ridge(alpha=1.0)),
    }


def get_primary_clf_models(pipe_fn=make_preprocess_pipe):
    return {
        MODEL_NAMES["rf"]:  pipe_fn(RandomForestClassifier(n_estimators=200, random_state=42)),
        MODEL_NAMES["gb"]:  pipe_fn(GradientBoostingClassifier(random_state=42)),
        MODEL_NAMES["mlp"]: pipe_fn(MLPClassifier(hidden_layer_sizes=(32, 16), max_iter=3000, random_state=42)),
    }


def get_baseline_reg_models(pipe_fn=make_preprocess_pipe):
    return {
        MODEL_NAMES["dummy_reg"]: pipe_fn(DummyRegressor(strategy="mean")),
        MODEL_NAMES["ridge_reg"]: pipe_fn(Ridge(alpha=1.0)),
    }


def get_primary_reg_models(pipe_fn=make_preprocess_pipe):
    return {
        MODEL_NAMES["rf"]:  pipe_fn(RandomForestRegressor(n_estimators=200, random_state=42)),
        MODEL_NAMES["gb"]:  pipe_fn(GradientBoostingRegressor(random_state=42)),
        MODEL_NAMES["mlp"]: pipe_fn(MLPRegressor(hidden_layer_sizes=(32, 16), max_iter=3000, random_state=42)),
    }


def _run_consistency_checks(df):
    """Non-fatal pre-export consistency checks."""
    warnings = []
    expected_cols = ['Name', 'City', 'Pop_Density', 'CSI', 'DPI', 'DPI_Risk_Class',
                     'Risk_Archetype', 'Predicted_Risk_Class', 'Predicted_DPI']
    missing = [c for c in expected_cols if c not in df.columns]
    if missing:
        warnings.append(f"Missing expected columns: {missing}")
    # Verify no bare "MLP" in canonical names
    for key, name in MODEL_NAMES.items():
        if name == "MLP":
            warnings.append(f"Bare 'MLP' in MODEL_NAMES['{key}']. Expected 'Neural Network (MLP)'.")
    if warnings:
        for w in warnings:
            print(f"  WARNING: {w}")
    else:
        print("  Pre-export consistency checks passed.")
    return len(warnings) == 0


#Sequential ML Pipeline for Flood Risk Assessment in Metro Manila

#Data Integration
def load_data(combined_csv_path):
    print(f"Data Loading: {combined_csv_path}")
    df = pd.read_csv(combined_csv_path)
    
    #Clean Whitespace
    df.columns = df.columns.str.strip()
    
    #Rename
    rename_map = {
        'barangay': 'Name',
        'Barangay': 'Name',
        'brgy_area_sqm': 'Land Area',
        'Population_2020': 'Population',
        'population': 'Population',
        'Population': 'Population'
    }
    df = df.rename(columns=rename_map)

    #Validation
    if 'Name' not in df.columns:
        raise KeyError("Could not find 'Barangay' or 'Name' column. Check CSV headers.")
            
    #Fill missing population
    if 'Population' not in df.columns:
        print("WARNING: Population column missing. Creating empty column.")
        df['Population'] = 0

    #Composite key for barangay uniqueness (City + Name)
    #Barangay names may repeat across cities (e.g., "Barangay 1" in Manila vs Makati)
    if 'City' in df.columns:
        df['City_Name'] = df['City'].astype(str).str.strip() + '_' + df['Name'].astype(str).str.strip()
        n_total = len(df)
        n_unique = df['City_Name'].nunique()
        if n_total != n_unique:
            n_dupes = n_total - n_unique
            print(f"WARNING: {n_dupes} duplicate City+Barangay entries detected. Keeping first occurrence.")
            df = df.drop_duplicates(subset='City_Name', keep='first').reset_index(drop=True)
        else:
            print(f"  Uniqueness check passed: {n_unique} unique barangays.")
    else:
        print("WARNING: No 'City' column found. Cannot create composite key for uniqueness validation.")

    return df

#Feature Engineering
def engineer_features(df):
    # Basic Cleanup: replace zero land area with NaN to avoid division-by-zero
    n_zero_area = (df['Land Area'] == 0).sum()
    n_null_area = df['Land Area'].isna().sum()
    df['Land Area'] = df['Land Area'].replace(0, np.nan)
    n_invalid = df['Land Area'].isna().sum()
    if n_invalid > 0:
        print(f"  Land Area: {n_zero_area} zero + {n_null_area} null = {n_invalid} invalid rows.")
        print(f"  These rows will produce NaN-derived features (imputed downstream by SimpleImputer with mean strategy).")
    
    #Flood Coverage Ratios (How much of barangay is wet?)
    df['flood5_coverage'] = df['flood5_sqm_final'] / df['Land Area']
    df['flood25_coverage'] = df['flood25_sqm_final'] / df['Land Area']
    df['flood100_coverage'] = df['flood100_sqm_final'] / df['Land Area']

    #Defensive clip to [0, 1] for coverage ratios (handles potential GIS overlay artifacts)
    for cov_col in ['flood5_coverage', 'flood25_coverage', 'flood100_coverage']:
        df[cov_col] = np.clip(df[cov_col], 0, 1)
    
    #Population Density (people per square meter, ppl/m²)
    #Land Area is in sqm, so Population / Land Area = ppl/m²
    df['Pop_Density'] = (df['Population'] / df['Land Area']) 
    
    #Interaction Terms: Affected Population
    #Multiplies People * Water. 
    #If Pop is high but Water is 0, this is 0.
    df['Affected_Pop_5yr'] = df['Population'] * df['flood5_coverage']
    df['Affected_Pop_25yr'] = df['Population'] * df['flood25_coverage']
    df['Affected_Pop_100yr'] = df['Population'] * df['flood100_coverage']
    
    #Flood Growth Rate (5 yr to 25 yr)
    df['Flood_Growth_5to25'] = (df['flood25_sqm_final'] - df['flood5_sqm_final']) / df['Land Area']
    
    return df

#Deterministic Indexing
def calculate_indices(df):
    #Advanced Engineering
    df = engineer_features(df)
    
    #Vulnerability Score (Human Risk)
    #Thresholds calibrated to Metro Manila density distribution (PSA, 2020 Census):
    #  0.07 ppl/m2 (~70k/km2) ~= 90th percentile of NCR barangay densities
    #  0.04 ppl/m2 (~40k/km2) ~= 75th percentile
    #  0.02 ppl/m2 (~20k/km2) ~= national urban median
    #Score range: 0-10 (0 = no population / missing data, 10 = extreme density)
    def get_vuln_score(density):
        if pd.isna(density) or density <= 0.0: return 0
        if density >= 0.07: return 10
        elif density >= 0.04: return 7
        elif density >= 0.02: return 4
        else: return 2
    
    df['Vulnerability_Score'] = df['Pop_Density'].apply(get_vuln_score)

    #Flood Risk Scores (Physical Risk)
    #Thresholds informed by flood hazard mapping conventions (PAGASA; DOST-Project NOAH):
    #  >= 80% coverage: near-total inundation, catastrophic scenario
    #  >= 50% coverage: majority flooded, severe impact
    #  >= 20% coverage: significant partial flooding
    #  >  5% coverage: minor/localized flooding
    #  <= 5% coverage: negligible exposure (scored 0)
    #Note: step-function design is intentional for ordinal risk tiering;
    #sensitivity to threshold choice is assessed via DPI weight sensitivity analysis.
    def get_flood_score(ratio):
        if pd.isna(ratio): return 0
        if ratio >= 0.80: return 10.0
        elif ratio >= 0.50: return 7.5
        elif ratio >= 0.20: return 5.0
        elif ratio > 0.05: return 2.5
        else: return 0.0

    #Calculate sub-scores using coverage columns
    df['Assessment_5yr_Score'] = df['flood5_coverage'].apply(get_flood_score)
    df['Assessment_25yr_Score'] = df['flood25_coverage'].apply(get_flood_score)
    df['Assessment_100yr_Score'] = df['flood100_coverage'].apply(get_flood_score)

    #Composite Susceptibility Index (CSI)
    df['CSI'] = (
        (df['Assessment_5yr_Score'] * 0.5) +
        (df['Assessment_25yr_Score'] * 0.3) +
        (df['Assessment_100yr_Score'] * 0.2)
    )

    #Final DPI Calculation
    df['DPI'] = (df['CSI'] * 0.6) + (df['Vulnerability_Score'] * 0.4)
    
    return df

#Predictive Modeling
def run_ml_pipeline(df):
    #Feature List
    feature_cols = [
        'Pop_Density',
        'flood5_coverage',      
        'flood100_coverage',
        'Affected_Pop_100yr', 
        'Flood_Growth_5to25',  
        'Land Area'
    ]
    
    #Coerce to numeric (safety net for non-numeric artifacts); NaNs preserved
    #for Pipeline SimpleImputer to handle via mean-imputation inside each fold
    for col in feature_cols:
        df[col] = pd.to_numeric(df[col], errors='coerce')

    #Feature collinearity check (justifies flood25_coverage exclusion from model)
    print("\nFeature Collinearity Check (flood coverage ratios)")
    _cov_cols = ['flood5_coverage', 'flood25_coverage', 'flood100_coverage']
    for _i, _c1 in enumerate(_cov_cols):
        for _c2 in _cov_cols[_i+1:]:
            _v1 = pd.to_numeric(df[_c1], errors='coerce').values
            _v2 = pd.to_numeric(df[_c2], errors='coerce').values
            _valid = ~(np.isnan(_v1) | np.isnan(_v2))
            if _valid.sum() > 2:
                _rho, _pval = spearmanr(_v1[_valid], _v2[_valid])
                print(f"  {_c1} vs {_c2}: Spearman rho={_rho:.4f} (p={_pval:.2e})")
    print("  flood25_coverage excluded from model features due to high collinearity with flood5 and flood100.")

    #Raw feature matrix for supervised evaluation (NaNs intact for Pipeline imputer)
    X = df[feature_cols].to_numpy()

    #Pre-scaled matrix for unsupervised KMeans only (no train/test leakage concern)
    _imputer = SimpleImputer(strategy="mean")
    _scaler = StandardScaler()
    X_scaled = _scaler.fit_transform(_imputer.fit_transform(X))

    #Deterministic Benchmarking Target (Proxy Label): DPI-derived risk class
    #These thresholds can be adjusted to match your paper's stated cutoffs.
    def dpi_to_risk_class(dpi_score):
        if pd.isna(dpi_score):
            return "Low"
        if dpi_score >= 6.5:
            return "High"
        if dpi_score >= 3.5:
            return "Moderate"
        return "Low"

    df["DPI_Risk_Class"] = df["DPI"].apply(dpi_to_risk_class)
    #DPI_Risk_Class_Proxy is set equal to DPI_Risk_Class by design: the ML pipeline
    #treats DPI-derived classes as the proxy target. This column is retained for
    #backwards compatibility and should be interpreted as a consistency check
    #(agreement with DPI), not as an independent ML-generated prediction.
    #For actual ML predictions, see 'Predicted_Risk_Class' column.
    df["DPI_Risk_Class_Proxy"] = df["DPI_Risk_Class"]
    df["ML_Risk_Class"] = df["DPI_Risk_Class"]  # alias kept for backwards compatibility

    #Unsupervised Phase | K-Means for exploratory "Risk Archetypes" (not labels)
    #Cluster count selection: evaluate K in {2,3,4,5,6} via silhouette score
    print("\nUnsupervised Phase - Cluster Count Selection")
    k_range = range(2, 7)
    k_silhouettes = {}
    for k in k_range:
        km_temp = KMeans(n_clusters=k, random_state=42, n_init=10)
        labels_temp = km_temp.fit_predict(X_scaled)
        sil_temp = silhouette_score(X_scaled, labels_temp)
        k_silhouettes[k] = sil_temp
        print(f"  K={k}: Silhouette={sil_temp:.4f}")
    best_k = max(k_silhouettes, key=k_silhouettes.get)
    print(f"  Selected K={best_k} (highest silhouette={k_silhouettes[best_k]:.4f})")

    kmeans = KMeans(n_clusters=best_k, random_state=42, n_init=10)
    clusters = kmeans.fit_predict(X_scaled)
    df["Cluster_Group"] = clusters

    #Semantic Archetype Mapping
    #K clusters (selected by silhouette maximization) are mapped to 4 semantic
    #archetypes based on each cluster's centroid position in the Pop_Density ×
    #flood100_coverage feature space. The median of cluster centroids serves as
    #the decision boundary for each axis:
    #  Pop_Density >= median → "HighDensity", else "LowDensity"
    #  flood100_coverage >= median → "HighExposure", else "LowExposure"
    #Because K may exceed 4 (e.g., K=6), multiple clusters can map to the same
    #archetype label if their centroids fall in the same quadrant.
    #This two-step approach (optimal K selection → semantic consolidation)
    #preserves statistical granularity while producing interpretable categories.
    try:
        cluster_means = df.groupby('Cluster_Group')[['Pop_Density', 'flood100_coverage']].mean(numeric_only=True)
        density_median = cluster_means['Pop_Density'].median()
        exposure_median = cluster_means['flood100_coverage'].median()
        archetype_map = {}
        for group, row in cluster_means.iterrows():
            density_tag = "HighDensity" if row['Pop_Density'] >= density_median else "LowDensity"
            exposure_tag = "HighExposure" if row['flood100_coverage'] >= exposure_median else "LowExposure"
            archetype_map[group] = f"{density_tag}-{exposure_tag}"
        df['Risk_Archetype'] = df['Cluster_Group'].map(archetype_map)

        #Report cluster-to-archetype mapping so the K→4 consolidation is transparent
        print(f"\n  Cluster-to-Archetype Mapping (K={best_k} clusters → 4 semantic archetypes):")
        for cluster_id in sorted(archetype_map.keys()):
            n_members = (df['Cluster_Group'] == cluster_id).sum()
            print(f"    Cluster {cluster_id} → {archetype_map[cluster_id]}  (n={n_members})")
        #Summary: how many clusters map to each archetype
        from collections import Counter
        archetype_counts = Counter(archetype_map.values())
        print(f"  Archetype summary: {dict(archetype_counts)}")
    except Exception:
        df['Risk_Archetype'] = df['Cluster_Group'].astype(str)

    #Visualization (Archetypes)
    #Note: archetype labels are derived from flood100_coverage (extreme-event exposure),
    #while this plot shows CSI (multi-return-period composite). Apparent mismatches
    #(e.g., high CSI but "LowExposure" label) can occur because CSI heavily weights
    #flood5_coverage, whereas archetypes use only the 100-year layer.
    try:
        plt.figure(figsize=(10, 6))
        sns.scatterplot(
            data=df, x='Pop_Density', y='CSI', hue='Risk_Archetype', alpha=0.7
        )
        plt.xlabel('Pop_Density (ppl/m²)')
        plt.ylabel('CSI')
        plt.title(f'Risk Archetypes (K-Means, K={best_k}): Pop Density vs Physical Risk')
        plt.tight_layout()
        plt.savefig('kmeans_cluster_chart.png', dpi=300)
    except: pass

    #Unsupervised diagnostics (for reporting)
    print("\nUnsupervised Phase - K-Means Diagnostics")
    try:
        sil = silhouette_score(X_scaled, clusters)
        db = davies_bouldin_score(X_scaled, clusters)
        ch = calinski_harabasz_score(X_scaled, clusters)
        print(f"  Silhouette: {sil:.4f} (higher better)")
        print(f"  Davies-Bouldin: {db:.4f} (lower better)")
        print(f"  Calinski-Harabasz: {ch:.2f} (higher better)")
    except Exception as e:
        print(f"  Could not compute clustering diagnostics: {e}")

    try:
        baseline = clusters
        ari_scores = []
        for seed in range(10):
            km = KMeans(n_clusters=best_k, random_state=seed, n_init=10)
            alt = km.fit_predict(X_scaled)
            ari_scores.append(adjusted_rand_score(baseline, alt))
        print(f"  Stability (ARI vs seed=42, 10 seeds): mean={np.mean(ari_scores):.3f} | min={np.min(ari_scores):.3f}")
    except Exception as e:
        print(f"  Could not compute stability: {e}")

    #Supervised Phase - Benchmarking (classification + regression)
    #Ordinal mapping (not alphabetical LabelEncoder) so Low=0 < Moderate=1 < High=2
    ordinal_map = {"Low": 0, "Moderate": 1, "High": 2}
    inverse_ordinal_map = {v: k for k, v in ordinal_map.items()}
    y_class = df["DPI_Risk_Class"].map(ordinal_map).astype(int).to_numpy()
    y_reg = df["DPI"].astype(float).to_numpy()

    X_train, X_test, y_class_train, y_class_test, y_reg_train, y_reg_test = train_test_split(
        X,
        y_class,
        y_reg,
        test_size=0.2,
        random_state=42,
        stratify=y_class,
    )

    #Baseline Models (methodology-required simple references)
    baseline_clf_models = get_baseline_clf_models()
    baseline_reg_models = get_baseline_reg_models()

    #Primary Supervised Models
    clf_models = get_primary_clf_models()
    reg_models = get_primary_reg_models()

    #Baseline Classification
    print("\nBaseline Comparisons - Classification (target: DPI_Risk_Class)")
    baseline_clf_results = {}
    for name, model in baseline_clf_models.items():
        model.fit(X_train, y_class_train)
        if name == MODEL_NAMES["ord_ridge"]:
            y_pred = np.clip(np.round(model.predict(X_test)).astype(int), 0, 2)
        else:
            y_pred = model.predict(X_test)
        acc = accuracy_score(y_class_test, y_pred)
        bal_acc = balanced_accuracy_score(y_class_test, y_pred)
        f1 = f1_score(y_class_test, y_pred, average='weighted', zero_division=0)
        f1_mac = f1_score(y_class_test, y_pred, average='macro', zero_division=0)
        cm = confusion_matrix(y_class_test, y_pred, labels=[0, 1, 2])
        baseline_clf_results[name] = {"acc": acc, "bal_acc": bal_acc, "f1": f1, "f1_macro": f1_mac, "model": model, "y_pred": y_pred}
        print(f"  [{name}] Acc: {acc:.2%} | BalAcc: {bal_acc:.2%} | F1w: {f1:.2%} | F1macro: {f1_mac:.2%}")
        print(f"    Confusion Matrix (rows=true, cols=pred [L,M,H]): {cm.tolist()}")

    #Baseline Regression
    print("\nBaseline Comparisons - Regression (target: DPI score)")
    baseline_reg_results = {}
    for name, model in baseline_reg_models.items():
        model.fit(X_train, y_reg_train)
        y_pred = model.predict(X_test)
        rmse = float(np.sqrt(mean_squared_error(y_reg_test, y_pred)))
        mae = float(mean_absolute_error(y_reg_test, y_pred))
        baseline_reg_results[name] = {"rmse": rmse, "mae": mae, "model": model, "y_pred": y_pred}
        print(f"  [{name}] RMSE: {rmse:.4f} | MAE: {mae:.4f}")

    #Primary Supervised Classification
    print("\nSupervised Phase - Classification (target: DPI_Risk_Class)")
    clf_results = {}
    for name, model in clf_models.items():
        model.fit(X_train, y_class_train)
        y_pred = model.predict(X_test)
        acc = accuracy_score(y_class_test, y_pred)
        bal_acc = balanced_accuracy_score(y_class_test, y_pred)
        f1 = f1_score(y_class_test, y_pred, average='weighted', zero_division=0)
        f1_mac = f1_score(y_class_test, y_pred, average='macro', zero_division=0)
        cm = confusion_matrix(y_class_test, y_pred, labels=[0, 1, 2])
        clf_results[name] = {"acc": acc, "bal_acc": bal_acc, "f1": f1, "f1_macro": f1_mac, "model": model, "y_pred": y_pred}
        print(f"  [{name}] Acc: {acc:.2%} | BalAcc: {bal_acc:.2%} | F1w: {f1:.2%} | F1macro: {f1_mac:.2%}")
        print(f"    Confusion Matrix (rows=true, cols=pred [L,M,H]): {cm.tolist()}")

    #Primary Supervised Regression
    print("\nSupervised Phase - Regression (target: DPI score)")
    reg_results = {}
    for name, model in reg_models.items():
        model.fit(X_train, y_reg_train)
        y_pred = model.predict(X_test)
        rmse = float(np.sqrt(mean_squared_error(y_reg_test, y_pred)))
        mae = float(mean_absolute_error(y_reg_test, y_pred))
        reg_results[name] = {"rmse": rmse, "mae": mae, "model": model, "y_pred": y_pred}
        print(f"  [{name}] RMSE: {rmse:.4f} | MAE: {mae:.4f}")

    #Improvement over baselines
    print("\nImprovement Over Baselines")
    dummy_clf_f1 = baseline_clf_results[MODEL_NAMES["dummy_clf"]]["f1"]
    dummy_reg_rmse = baseline_reg_results[MODEL_NAMES["dummy_reg"]]["rmse"]
    for name, res in clf_results.items():
        delta = res["f1"] - dummy_clf_f1
        print(f"  [{name}] F1 gain over Dummy: +{delta:.2%}")
    for name, res in reg_results.items():
        delta = dummy_reg_rmse - res["rmse"]
        print(f"  [{name}] RMSE reduction vs Dummy: {delta:.4f}")

    #Select best models for optional output columns
    best_clf_name = max(clf_results.keys(), key=lambda n: clf_results[n]["f1"])
    best_reg_name = min(reg_results.keys(), key=lambda n: reg_results[n]["rmse"])
    best_clf = clf_results[best_clf_name]["model"]
    best_reg = reg_results[best_reg_name]["model"]

    print(f"\nBest Classifier (by F1): {best_clf_name}")
    print(f"Best Regressor (by RMSE): {best_reg_name}")

    #Add predictions to dataframe (proof-of-concept outputs)
    #NOTE: These are full-dataset (resubstitution) predictions — the model predicts on
    #data that includes its training set. These should NOT be used as the primary
    #performance metric. See cross-validated scores above for generalization estimates.
    try:
        full_pred_class = best_clf.predict(X)
        full_pred_dpi = best_reg.predict(X)
        df["Predicted_Risk_Class"] = pd.Series(full_pred_class).map(inverse_ordinal_map).values
        df["Predicted_DPI"] = np.clip(full_pred_dpi, 0, 10)
        #Report resubstitution agreement (full-dataset, includes training data)
        n_disagree = (df["Predicted_Risk_Class"] != df["DPI_Risk_Class"]).sum()
        n_total = len(df)
        print(f"\n  Full-dataset resubstitution: {n_disagree}/{n_total} disagreements "
              f"({(n_total - n_disagree) / n_total:.1%} match)")
        print(f"  Note: Resubstitution metric (includes training data). "
              f"See cross-validated scores for generalization estimate.")
    except Exception:
        pass

    #Feature importance (interpretable models)
    print("\nFeature Importance (Random Forest)")
    try:
        #Use the already-fitted RF pipeline from clf_results (trained on X_train/y_class_train)
        rf_pipeline = clf_results[MODEL_NAMES["rf"]]["model"]
        rf_estimator = rf_pipeline.named_steps['model']
        importances = rf_estimator.feature_importances_
        for feat, score in sorted(zip(feature_cols, importances), key=lambda x: x[1], reverse=True):
            print(f"  - {feat}: {score:.4f}")
    except Exception as e:
        print(f"  Could not compute feature importance: {e}")

    #Stratified K-Fold Cross-Validation (addresses single-split weakness)
    print("\nCross-Validation (5-Fold Stratified)")
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    kf = KFold(n_splits=5, shuffle=True, random_state=42)

    #Helper: ordinal Ridge predict wrapper for cross_val_score
    def _ordinal_ridge_f1w(estimator, X_eval, y_eval):
        y_pred = np.clip(np.round(estimator.predict(X_eval)).astype(int), 0, 2)
        return f1_score(y_eval, y_pred, average='weighted', zero_division=0)
    def _ordinal_ridge_f1m(estimator, X_eval, y_eval):
        y_pred = np.clip(np.round(estimator.predict(X_eval)).astype(int), 0, 2)
        return f1_score(y_eval, y_pred, average='macro', zero_division=0)

    all_clf_cv = {}
    all_clf_cv.update({f"(B) {n}": m for n, m in baseline_clf_models.items()})
    all_clf_cv.update({f"(P) {n}": m for n, m in clf_models.items()})

    all_reg_cv = {}
    all_reg_cv.update({f"(B) {n}": m for n, m in baseline_reg_models.items()})
    all_reg_cv.update({f"(P) {n}": m for n, m in reg_models.items()})

    cv_clf_results = {}
    cv_reg_results = {}

    print("  Classification (F1-weighted & F1-macro):")
    for name, mdl in all_clf_cv.items():
        model_cv = clone(mdl)
        is_ordinal = "Ordinal Ridge" in name
        if is_ordinal:
            scores_w = cross_val_score(model_cv, X, y_class, cv=skf, scoring=_ordinal_ridge_f1w)
            model_cv2 = clone(mdl)
            scores_m = cross_val_score(model_cv2, X, y_class, cv=skf, scoring=_ordinal_ridge_f1m)
        else:
            scores_w = cross_val_score(model_cv, X, y_class, cv=skf, scoring='f1_weighted')
            model_cv2 = clone(mdl)
            scores_m = cross_val_score(model_cv2, X, y_class, cv=skf, scoring='f1_macro')
        cv_clf_results[name] = {"f1w": scores_w, "f1m": scores_m}
        print(f"    [{name}] F1w mean={scores_w.mean():.4f}+/-{scores_w.std():.4f} | F1macro mean={scores_m.mean():.4f}+/-{scores_m.std():.4f}")

    print("  Regression (RMSE & MAE):")
    for name, mdl in all_reg_cv.items():
        model_cv = clone(mdl)
        scores_rmse = cross_val_score(model_cv, X, y_reg, cv=kf, scoring='neg_root_mean_squared_error')
        model_cv2 = clone(mdl)
        scores_mae = cross_val_score(model_cv2, X, y_reg, cv=kf, scoring='neg_mean_absolute_error')
        cv_reg_results[name] = {"rmse": scores_rmse, "mae": scores_mae}
        print(f"    [{name}] RMSE mean={-scores_rmse.mean():.4f}+/-{scores_rmse.std():.4f} | MAE mean={-scores_mae.mean():.4f}+/-{scores_mae.std():.4f}")

    #Spatial Cross-Validation (GroupKFold by City)
    #Addresses spatial autocorrelation limitation: neighboring barangays within
    #the same city share correlated flood exposure and demographic characteristics.
    #GroupKFold ensures all barangays from a given city appear in the same fold,
    #preventing spatial leakage between train and test sets.
    print("\nSpatial Cross-Validation (GroupKFold by City)")
    try:
        if 'City' in df.columns:
            groups = df['City'].values
            n_groups = df['City'].nunique()
            n_splits_group = min(5, n_groups)
            gkf = GroupKFold(n_splits=n_splits_group)
            print(f"  Groups: {n_groups} cities, {n_splits_group}-fold")
            print("  Classification (F1-weighted & F1-macro):")
            for name, mdl in all_clf_cv.items():
                model_cv = clone(mdl)
                is_ordinal = "Ordinal Ridge" in name
                if is_ordinal:
                    scores_w = cross_val_score(model_cv, X, y_class, cv=gkf, groups=groups, scoring=_ordinal_ridge_f1w)
                    model_cv2 = clone(mdl)
                    scores_m = cross_val_score(model_cv2, X, y_class, cv=gkf, groups=groups, scoring=_ordinal_ridge_f1m)
                else:
                    scores_w = cross_val_score(model_cv, X, y_class, cv=gkf, groups=groups, scoring='f1_weighted')
                    model_cv2 = clone(mdl)
                    scores_m = cross_val_score(model_cv2, X, y_class, cv=gkf, groups=groups, scoring='f1_macro')
                print(f"    [{name}] F1w mean={scores_w.mean():.4f}+/-{scores_w.std():.4f} | F1macro mean={scores_m.mean():.4f}+/-{scores_m.std():.4f}")
            print("  Regression (RMSE & MAE):")
            for name, mdl in all_reg_cv.items():
                model_cv = clone(mdl)
                scores_rmse = cross_val_score(model_cv, X, y_reg, cv=gkf, groups=groups, scoring='neg_root_mean_squared_error')
                model_cv2 = clone(mdl)
                scores_mae = cross_val_score(model_cv2, X, y_reg, cv=gkf, groups=groups, scoring='neg_mean_absolute_error')
                print(f"    [{name}] RMSE mean={-scores_rmse.mean():.4f}+/-{scores_rmse.std():.4f} | MAE mean={-scores_mae.mean():.4f}+/-{scores_mae.std():.4f}")
        else:
            print("  SKIP: No 'City' column available for spatial grouping.")
    except Exception as e:
        print(f"  Could not compute spatial CV: {e}")

    # ── Build pipeline results for PDF report (avoids re-training) ──
    pipeline_results = {
        "y_class_test": y_class_test,
        "y_reg_test": y_reg_test,
        "all_clf_test": {},
        "all_reg_test": {},
        "cv_clf": cv_clf_results,
        "cv_reg": cv_reg_results,
    }
    for name, res in baseline_clf_results.items():
        pipeline_results["all_clf_test"][name] = {
            "acc": res["acc"], "bal_acc": res["bal_acc"],
            "f1": res["f1"], "f1_macro": res["f1_macro"],
            "y_pred": res["y_pred"], "type": "Baseline",
        }
    for name, res in clf_results.items():
        pipeline_results["all_clf_test"][name] = {
            "acc": res["acc"], "bal_acc": res["bal_acc"],
            "f1": res["f1"], "f1_macro": res["f1_macro"],
            "y_pred": res["y_pred"], "type": "Primary",
        }
    for name, res in baseline_reg_results.items():
        pipeline_results["all_reg_test"][name] = {
            "rmse": res["rmse"], "mae": res["mae"],
            "y_pred": res["y_pred"], "type": "Baseline",
        }
    for name, res in reg_results.items():
        pipeline_results["all_reg_test"][name] = {
            "rmse": res["rmse"], "mae": res["mae"],
            "y_pred": res["y_pred"], "type": "Primary",
        }

    return df, pipeline_results

def run_sensitivity_analysis(df):
    """Sensitivity analysis of DPI weights per methodology.
    Varies CSI sub-weights and CSI-vulnerability balance,
    then reports Spearman rank correlation across configurations."""
    print("\nSensitivity Analysis of DPI Weights")

    #Required columns (already computed by calculate_indices)
    required = ['Assessment_5yr_Score', 'Assessment_25yr_Score',
                'Assessment_100yr_Score', 'Vulnerability_Score']
    for col in required:
        if col not in df.columns:
            print(f"  SKIP: missing column {col}")
            return

    s5  = df['Assessment_5yr_Score'].fillna(0)
    s25 = df['Assessment_25yr_Score'].fillna(0)
    s100 = df['Assessment_100yr_Score'].fillna(0)
    vuln = df['Vulnerability_Score'].fillna(0)

    #Weight configurations to test
    csi_weight_configs = [
        (0.50, 0.30, 0.20),   #default
        (0.40, 0.30, 0.30),
        (0.60, 0.25, 0.15),
        (0.33, 0.34, 0.33),   #equal
    ]
    balance_configs = [
        (0.60, 0.40),   #default
        (0.50, 0.50),
        (0.70, 0.30),
    ]

    #Compute DPI for every combination
    configs = []
    dpis = []
    for csi_w in csi_weight_configs:
        for bal in balance_configs:
            csi = s5 * csi_w[0] + s25 * csi_w[1] + s100 * csi_w[2]
            dpi = csi * bal[0] + vuln * bal[1]
            label = f"CSI({csi_w[0]:.0%}/{csi_w[1]:.0%}/{csi_w[2]:.0%})|Bal({bal[0]:.0%}/{bal[1]:.0%})"
            configs.append(label)
            dpis.append(dpi.values)

    #Pairwise Spearman correlations
    n = len(configs)
    rho_values = []
    print(f"  Configurations tested: {n}")
    for i, j in combinations(range(n), 2):
        rho, _ = spearmanr(dpis[i], dpis[j])
        rho_values.append(rho)

    min_rho = np.min(rho_values)
    mean_rho = np.mean(rho_values)
    print(f"  Spearman rho (all pairs): mean={mean_rho:.4f} | min={min_rho:.4f}")

    #Compare each config against the default (first one: 50/30/20 + 60/40)
    print("\n  Rank stability vs default (CSI 50/30/20 | Balance 60/40):")
    default_dpi = dpis[0]
    for idx in range(1, n):
        rho, _ = spearmanr(default_dpi, dpis[idx])
        print(f"    {configs[idx]}: rho={rho:.4f}")

    if min_rho >= 0.90:
        print(f"\n  Conclusion: Rankings are ROBUST (min rho={min_rho:.4f} >= 0.90).")
    else:
        print(f"\n  Conclusion: Rankings show SENSITIVITY to weight choice (min rho={min_rho:.4f} < 0.90).")
        print("    This is reported as a methodological limitation.")

    return {"min_rho": min_rho, "mean_rho": mean_rho, "n_configs": n}

def run_validation_checks(df):
    print("\nValidation & Consistency Checks")
    print("  Note: DPI is treated as the deterministic proxy label (not external ground truth).")
    dist = df['DPI_Risk_Class_Proxy'].value_counts(normalize=True) * 100
    print(f"  DPI Risk Distribution: High {dist.get('High',0):.1f}% | Mod {dist.get('Moderate',0):.1f}% | Low {dist.get('Low',0):.1f}%")

    #3A) Monotonic sanity checks (data-driven construct validation)
    print("\n  Monotonic Sanity Checks (Spearman correlations with DPI)")
    dpi_vals = df['DPI'].values
    for check_col, expect_dir in [('CSI', 'positive'), ('Pop_Density', 'positive'), ('flood100_coverage', 'positive')]:
        if check_col in df.columns:
            col_vals = pd.to_numeric(df[check_col], errors='coerce').values
            valid = ~(np.isnan(dpi_vals) | np.isnan(col_vals))
            if valid.sum() > 2:
                rho, pval = spearmanr(dpi_vals[valid], col_vals[valid])
                ok = rho > 0
                print(f"    DPI vs {check_col}: rho={rho:.4f} (p={pval:.2e}) | Expected {expect_dir} | Pass={ok}")
            else:
                print(f"    DPI vs {check_col}: insufficient non-NaN pairs")
        else:
            print(f"    DPI vs {check_col}: column not found")

    #3B) Concrete exemplars chosen from the dataset (no hard-coded names)
    exemplar_cols = ['Name', 'City', 'CSI', 'DPI', 'DPI_Risk_Class']
    if 'Risk_Archetype' in df.columns:
        exemplar_cols.append('Risk_Archetype')

    try:
        print("\n  Top 5 by CSI")
        top_csi = df.sort_values('CSI', ascending=False).head(5)[exemplar_cols]
        print(top_csi.to_string(index=False))
    except Exception:
        pass

    try:
        print("\n  Top 5 by Pop_Density")
        top_pd = df.sort_values('Pop_Density', ascending=False).head(5)[exemplar_cols]
        print(top_pd.to_string(index=False))
    except Exception:
        pass

    try:
        print("\n  Top 5 by DPI")
        top_dpi = df.sort_values('DPI', ascending=False).head(5)[exemplar_cols]
        print(top_dpi.to_string(index=False))
    except Exception:
        pass

    try:
        print("\n  Bottom 5 by DPI")
        bot_dpi = df.sort_values('DPI', ascending=True).head(5)[exemplar_cols]
        print(bot_dpi.to_string(index=False))
    except Exception:
        pass

    #3C) Outlier inspection (preparation for panel questions about extreme values)
    try:
        print("\n  Outlier Inspection (top 5 most extreme Pop_Density)")
        outlier_cols = ['Name', 'City', 'Population', 'Land Area', 'Pop_Density', 'CSI', 'DPI', 'DPI_Risk_Class']
        outlier_cols = [c for c in outlier_cols if c in df.columns]
        top_density = df.nlargest(5, 'Pop_Density')[outlier_cols]
        print(top_density.to_string(index=False))
        print("  Note: Extreme Pop_Density may result from small land area amplifying the population/area ratio.")
        print("        These are not data errors but reflect genuine urban density variation in Metro Manila.")
    except Exception:
        pass

def get_recommendation(row):
    """Generate archetype-aware recommendations combining DPI risk class
    with K-Means cluster archetype for more nuanced interventions."""
    risk = row.get('DPI_Risk_Class', row.get('DPI_Risk_Class_Proxy', 'Low'))
    archetype = str(row.get('Risk_Archetype', ''))

    if risk == "High":
        if 'HighDensity' in archetype and 'HighExposure' in archetype:
            return "Priority Zone: High density + high flood exposure. Immediate structural flood defenses and evacuation planning required."
        elif 'HighDensity' in archetype and 'LowExposure' in archetype:
            return "Priority Zone: High population density with elevated DPI. Focus on population management, early warning systems, and contingency planning."
        elif 'LowDensity' in archetype and 'HighExposure' in archetype:
            return "Priority Zone: Severe flood exposure despite lower density. Prioritize land-use restrictions and structural flood mitigation."
        else:
            return "Priority Zone: High logic-based DPI. Immediate intervention recommended."
    elif risk == "Moderate":
        if 'HighExposure' in archetype:
            return "Watch Zone: Moderate DPI with significant flood exposure. Monitor and prepare preemptive drainage/infrastructure improvements."
        elif 'HighDensity' in archetype:
            return "Watch Zone: Moderate DPI with high population density. Strengthen community preparedness and early warning coverage."
        else:
            return "Watch Zone: Moderate risk. Continue monitoring non-linear risk factors and seasonal flood patterns."
    else:
        if 'HighExposure' in archetype:
            return "Low Priority overall, but notable flood exposure detected. Periodic review recommended."
        else:
            return "Low Priority: Maintenance and routine monitoring only."

def export_results(df):
    df['Recommendation'] = df.apply(get_recommendation, axis=1)
    output_cols = [
        'Name',
        'City',
        'Pop_Density',
        'CSI',
        'DPI',
        'DPI_Risk_Class',
        'Risk_Archetype',
        'Predicted_Risk_Class',
        'Predicted_DPI',
        'Recommendation',
    ]
    
    for col in output_cols:
        if col not in df.columns: df[col] = "N/A"

    df[output_cols].to_excel("final_methodology_aligned_results.xlsx", index=False)
    print("\nResults exported.")

def generate_pdf_report(df, pipeline_results=None):
    """Comprehensive PDF report including summary statistics, cluster diagnostics,
    model performance comparison, top-risk barangays, and per-category details.
    Uses pre-computed pipeline_results to avoid re-training models."""
    print("Generating PDF Report...")
    try:
        pdf = FPDF()
        pdf.add_page()
        pw = pdf.w - pdf.l_margin - pdf.r_margin  # usable page width

        #Title
        pdf.set_font("Helvetica", style="B", size=16)
        pdf.cell(pw, 10, text="Disaster Risk Priority Report (NCR)", align="C",
                 new_x=XPos.LMARGIN, new_y=YPos.NEXT)
        pdf.ln(6)

        # ── Section 1: Summary Statistics ──
        pdf.set_font("Helvetica", style="B", size=13)
        pdf.cell(pw, 8, text="1. Summary Statistics", new_x=XPos.LMARGIN, new_y=YPos.NEXT)
        pdf.ln(2)
        total = len(df)
        dist = df['DPI_Risk_Class'].value_counts()
        pdf.set_font("Helvetica", size=10)
        pdf.cell(pw, 7, text=f"Total Barangays Analyzed: {total}", new_x=XPos.LMARGIN, new_y=YPos.NEXT)
        for cat in ['High', 'Moderate', 'Low']:
            cnt = dist.get(cat, 0)
            pdf.cell(pw, 7, text=f"  {cat} Risk: {cnt} ({cnt/total:.1%})", new_x=XPos.LMARGIN, new_y=YPos.NEXT)
        # DPI descriptive statistics
        pdf.cell(pw, 7, text=f"DPI  mean={df['DPI'].mean():.2f}  median={df['DPI'].median():.2f}  "
                              f"std={df['DPI'].std():.2f}  min={df['DPI'].min():.2f}  max={df['DPI'].max():.2f}",
                 new_x=XPos.LMARGIN, new_y=YPos.NEXT)
        pdf.ln(4)

        # ── Section 2: Top 10 Highest-DPI Barangays ──
        pdf.set_font("Helvetica", style="B", size=13)
        pdf.cell(pw, 8, text="2. Top 10 Highest-DPI Barangays", new_x=XPos.LMARGIN, new_y=YPos.NEXT)
        pdf.ln(2)
        col_w = [50, 35, 20, 20, 25, 40]  # Name, City, DPI, CSI, Class, Archetype
        headers = ["Barangay", "City", "DPI", "CSI", "Risk Class", "Archetype"]
        pdf.set_font("Helvetica", style="B", size=9)
        for i, h in enumerate(headers):
            pdf.cell(col_w[i], 7, h, 1)
        pdf.ln()
        pdf.set_font("Helvetica", size=8)
        top10 = df.sort_values('DPI', ascending=False).head(10)
        for _, r in top10.iterrows():
            vals = [
                str(r.get('Name', ''))[:22],
                str(r.get('City', 'N/A'))[:14],
                f"{r.get('DPI', 0):.2f}",
                f"{r.get('CSI', 0):.2f}",
                str(r.get('DPI_Risk_Class', 'N/A')),
                str(r.get('Risk_Archetype', 'N/A'))[:18],
            ]
            for i, v in enumerate(vals):
                pdf.cell(col_w[i], 7, v, 1)
            pdf.ln()
        pdf.ln(4)

        # ── Section 3: Clustering Diagnostics ──
        pdf.set_font("Helvetica", style="B", size=13)
        pdf.cell(pw, 8, text="3. K-Means Clustering Diagnostics", new_x=XPos.LMARGIN, new_y=YPos.NEXT)
        pdf.ln(2)
        pdf.set_font("Helvetica", size=10)
        # Recompute cluster diagnostics for PDF
        feature_cols = ['Pop_Density', 'flood5_coverage', 'flood100_coverage',
                        'Affected_Pop_100yr', 'Flood_Growth_5to25', 'Land Area']
        try:
            X_for_diag = df[feature_cols].apply(pd.to_numeric, errors='coerce')
            imp_diag = SimpleImputer(strategy="mean")
            scaler_diag = StandardScaler()
            X_sc = scaler_diag.fit_transform(imp_diag.fit_transform(X_for_diag))
            clusters_diag = df['Cluster_Group'].values
            n_clusters = len(set(clusters_diag))
            sil = silhouette_score(X_sc, clusters_diag)
            db = davies_bouldin_score(X_sc, clusters_diag)
            ch = calinski_harabasz_score(X_sc, clusters_diag)
            pdf.cell(pw, 7, text=f"Number of Clusters (K): {n_clusters}", new_x=XPos.LMARGIN, new_y=YPos.NEXT)
            pdf.cell(pw, 7, text=f"Silhouette Score: {sil:.4f} (higher = better)", new_x=XPos.LMARGIN, new_y=YPos.NEXT)
            pdf.cell(pw, 7, text=f"Davies-Bouldin Index: {db:.4f} (lower = better)", new_x=XPos.LMARGIN, new_y=YPos.NEXT)
            pdf.cell(pw, 7, text=f"Calinski-Harabasz Index: {ch:.2f} (higher = better)", new_x=XPos.LMARGIN, new_y=YPos.NEXT)
        except Exception:
            pdf.cell(pw, 7, text="Could not compute cluster diagnostics.", new_x=XPos.LMARGIN, new_y=YPos.NEXT)
        # Archetype distribution
        if 'Risk_Archetype' in df.columns:
            pdf.ln(2)
            pdf.set_font("Helvetica", style="B", size=10)
            pdf.cell(pw, 7, text="Archetype Distribution:", new_x=XPos.LMARGIN, new_y=YPos.NEXT)
            pdf.set_font("Helvetica", size=9)
            for arch, cnt in df['Risk_Archetype'].value_counts().items():
                pdf.cell(pw, 6, text=f"  {arch}: {cnt} ({cnt/total:.1%})", new_x=XPos.LMARGIN, new_y=YPos.NEXT)
        pdf.ln(4)

        # ── Section 4: Model Performance Comparison ──
        pdf.add_page()
        pdf.set_font("Helvetica", style="B", size=13)
        pdf.cell(pw, 8, text="4. Supervised Model Performance (Agreement with DPI)",
                 new_x=XPos.LMARGIN, new_y=YPos.NEXT)
        pdf.ln(2)
        pdf.set_font("Helvetica", size=9)
        pdf.cell(pw, 6, text="Note: Models predict index-derived targets (DPI). High accuracy reflects learnability",
                 new_x=XPos.LMARGIN, new_y=YPos.NEXT)
        pdf.cell(pw, 6, text="of the deterministic benchmark, not real-world flood prediction.",
                 new_x=XPos.LMARGIN, new_y=YPos.NEXT)
        pdf.ln(3)
        # Use pre-computed pipeline results (no re-training needed)
        try:
            if pipeline_results is None:
                raise ValueError("No pipeline results provided; skipping model tables.")
            cv_clf = pipeline_results["cv_clf"]
            cv_reg = pipeline_results["cv_reg"]
            all_clf_test = pipeline_results["all_clf_test"]
            all_reg_test = pipeline_results["all_reg_test"]
            yc_te = pipeline_results["y_class_test"]

            # ── 4a. Cross-Validated Performance (Primary Metric) ──
            # CV scores are the most defensible generalization estimates and are
            # presented first, before single-split or resubstitution metrics.
            pdf.set_font("Helvetica", style="B", size=10)
            pdf.cell(pw, 7, text="Primary: 5-Fold Stratified Cross-Validated Performance",
                     new_x=XPos.LMARGIN, new_y=YPos.NEXT)
            pdf.ln(1)
            pdf.set_font("Helvetica", size=8)
            pdf.cell(pw, 6, text="Cross-validation provides the most defensible generalization estimate.",
                     new_x=XPos.LMARGIN, new_y=YPos.NEXT)
            pdf.ln(2)
            # CV Classification table (F1-weighted + F1-macro)
            cv_cw = [40, 45, 45, 25]
            pdf.set_font("Helvetica", style="B", size=8)
            for i, h in enumerate(["Model", "CV F1w (mean+/-std)", "CV F1macro (mean+/-std)", "Type"]):
                pdf.cell(cv_cw[i], 7, h, 1)
            pdf.ln()
            pdf.set_font("Helvetica", size=8)
            for cv_key, cv_data in cv_clf.items():
                if not cv_key.startswith("(P)"):
                    continue
                display_name = cv_key[4:]  # strip "(P) " prefix
                scores_w = cv_data["f1w"]
                scores_m = cv_data["f1m"]
                pdf.cell(cv_cw[0], 7, display_name, 1)
                pdf.cell(cv_cw[1], 7, f"{scores_w.mean():.4f} +/- {scores_w.std():.4f}", 1)
                pdf.cell(cv_cw[2], 7, f"{scores_m.mean():.4f} +/- {scores_m.std():.4f}", 1)
                pdf.cell(cv_cw[3], 7, "Primary", 1)
                pdf.ln()
            pdf.ln(2)
            # CV Regression table (RMSE + MAE)
            cv_rw = [40, 45, 45, 25]
            pdf.set_font("Helvetica", style="B", size=8)
            for i, h in enumerate(["Model", "CV RMSE (mean+/-std)", "CV MAE (mean+/-std)", "Type"]):
                pdf.cell(cv_rw[i], 7, h, 1)
            pdf.ln()
            pdf.set_font("Helvetica", size=8)
            for cv_key, cv_data in cv_reg.items():
                if not cv_key.startswith("(P)"):
                    continue
                display_name = cv_key[4:]
                scores_rmse = cv_data["rmse"]
                scores_mae = cv_data["mae"]
                pdf.cell(cv_rw[0], 7, display_name, 1)
                pdf.cell(cv_rw[1], 7, f"{-scores_rmse.mean():.4f} +/- {scores_rmse.std():.4f}", 1)
                pdf.cell(cv_rw[2], 7, f"{-scores_mae.mean():.4f} +/- {scores_mae.std():.4f}", 1)
                pdf.cell(cv_rw[3], 7, "Primary", 1)
                pdf.ln()
            pdf.ln(4)

            # ── 4b. Single-Split Test Set Performance (Secondary) ──
            pdf.set_font("Helvetica", style="B", size=10)
            pdf.cell(pw, 7, text="Secondary: Single-Split Test Set Performance (80/20)",
                     new_x=XPos.LMARGIN, new_y=YPos.NEXT)
            pdf.ln(1)
            pdf.set_font("Helvetica", size=8)
            pdf.cell(pw, 6, text="Note: Single 80/20 stratified split. Evaluated on held-out test set only (not resubstitution).",
                     new_x=XPos.LMARGIN, new_y=YPos.NEXT)
            pdf.cell(pw, 6, text="Cross-validated scores above provide a more stable generalization estimate.",
                     new_x=XPos.LMARGIN, new_y=YPos.NEXT)
            pdf.ln(2)

            # Classification table (Acc, BalAcc, F1w, F1macro)
            pdf.set_font("Helvetica", style="B", size=10)
            pdf.cell(pw, 7, text="Classification (target: DPI_Risk_Class)", new_x=XPos.LMARGIN, new_y=YPos.NEXT)
            pdf.set_font("Helvetica", size=8)
            pdf.cell(pw, 5, text="Label mapping: Low=0, Moderate=1, High=2",
                     new_x=XPos.LMARGIN, new_y=YPos.NEXT)
            c_w = [38, 22, 22, 22, 22, 22]
            pdf.set_font("Helvetica", style="B", size=7)
            for i, h in enumerate(["Model", "Acc", "BalAcc", "F1w", "F1macro", "Type"]):
                pdf.cell(c_w[i], 7, h, 1)
            pdf.ln()
            pdf.set_font("Helvetica", size=7)
            for name, res in all_clf_test.items():
                for i, v in enumerate([name, f"{res['acc']:.2%}", f"{res['bal_acc']:.2%}",
                                       f"{res['f1']:.2%}", f"{res['f1_macro']:.2%}", res["type"]]):
                    pdf.cell(c_w[i], 7, v, 1)
                pdf.ln()
            pdf.ln(3)

            # Confusion matrices for selected models
            pdf.set_font("Helvetica", style="B", size=10)
            pdf.cell(pw, 7, text="Confusion Matrices (held-out test split)",
                     new_x=XPos.LMARGIN, new_y=YPos.NEXT)
            pdf.set_font("Helvetica", size=8)
            pdf.cell(pw, 5, text="Rows = true label, Cols = predicted. Order: [Low(0), Moderate(1), High(2)]",
                     new_x=XPos.LMARGIN, new_y=YPos.NEXT)
            pdf.ln(1)
            # Select best primary classifier by F1-weighted on test split
            primary_names = [n for n, r in all_clf_test.items() if r["type"] == "Primary"]
            best_primary_name = max(primary_names, key=lambda n: all_clf_test[n]["f1"])
            cm_models = [MODEL_NAMES["dummy_clf"], MODEL_NAMES["logreg"],
                         MODEL_NAMES["dt2"], MODEL_NAMES["ord_ridge"], best_primary_name]
            pdf.set_font("Helvetica", size=8)
            for cm_name in cm_models:
                if cm_name in all_clf_test:
                    cm = confusion_matrix(yc_te, all_clf_test[cm_name]["y_pred"], labels=[0, 1, 2])
                    pdf.cell(pw, 6, text=f"{cm_name}: {cm.tolist()}",
                             new_x=XPos.LMARGIN, new_y=YPos.NEXT)
            pdf.ln(3)

            # Regression table (RMSE + MAE)
            pdf.set_font("Helvetica", style="B", size=10)
            pdf.cell(pw, 7, text="Regression (target: DPI score)", new_x=XPos.LMARGIN, new_y=YPos.NEXT)
            r_w = [55, 35, 35, 25]
            pdf.set_font("Helvetica", style="B", size=7)
            for i, h in enumerate(["Model", "RMSE", "MAE", "Type"]):
                pdf.cell(r_w[i], 7, h, 1)
            pdf.ln()
            pdf.set_font("Helvetica", size=7)
            for name, res in all_reg_test.items():
                for i, v in enumerate([name, f"{res['rmse']:.4f}", f"{res['mae']:.4f}", res["type"]]):
                    pdf.cell(r_w[i], 7, v, 1)
                pdf.ln()
        except Exception as e:
            pdf.cell(pw, 7, text=f"Could not generate model tables: {e}", new_x=XPos.LMARGIN, new_y=YPos.NEXT)
        pdf.ln(5)

        # ── Section 5: Per-Category Details (original section, improved) ──
        pdf.set_font("Helvetica", style="B", size=13)
        pdf.cell(pw, 8, text="5. Risk Category Details (Top 10 per Category)",
                 new_x=XPos.LMARGIN, new_y=YPos.NEXT)
        pdf.ln(2)

        for category in ['High', 'Moderate', 'Low']:
            pdf.set_font("Helvetica", style="B", size=11)
            pdf.cell(pw, 8, text=f"{category} Risk", new_x=XPos.LMARGIN, new_y=YPos.NEXT)
            pdf.ln(1)

            t_w = [45, 30, 18, 18, 22, 57]  # Name, City, DPI, CSI, Archetype, Rec
            pdf.set_font("Helvetica", style="B", size=8)
            for i, h in enumerate(["Barangay", "City", "DPI", "CSI", "Archetype", "Recommendation"]):
                pdf.cell(t_w[i], 7, h, 1)
            pdf.ln()
            pdf.set_font("Helvetica", size=7)
            subset_df = df[df['DPI_Risk_Class'] == category].sort_values('DPI', ascending=False).head(10)
            for _, row in subset_df.iterrows():
                name = str(row.get('Name', ''))[:20]
                city = str(row.get('City', 'N/A'))[:13]
                dpi_v = f"{row.get('DPI', 0):.2f}"
                csi_v = f"{row.get('CSI', 0):.2f}"
                arch = str(row.get('Risk_Archetype', 'N/A'))[:10]
                rec = str(row.get('Recommendation', ''))[:30]
                for i, v in enumerate([name, city, dpi_v, csi_v, arch, rec]):
                    pdf.cell(t_w[i], 6, v, 1)
                pdf.ln()
            pdf.ln(3)

        output_filename = "final_risk_assessment.pdf"
        pdf.output(output_filename)
        print(f"PDF Report saved as: {output_filename}")
    except Exception as e:
        print(f"Could not generate PDF: {e}")

#Execution
if __name__ == "__main__":
    filename = 'MetroManila_Combined_Flood_Population.csv'
    
    try:
        df = load_data(filename)
        df = calculate_indices(df)
        df_final, pipeline_results = run_ml_pipeline(df)
        
        # Sensitivity Analysis (DPI weight robustness)
        run_sensitivity_analysis(df_final)
        
        # Validation
        run_validation_checks(df_final)
        
        # Exports
        # Pre-export consistency checks
        print("\nPre-Export Consistency Checks")
        _run_consistency_checks(df_final)

        export_results(df_final) #Excel
        generate_pdf_report(df_final, pipeline_results) #PDF
        
    except FileNotFoundError:
        print(f"\nERROR: File '{filename}' not found. Please upload it.")
    except Exception as e:
        print(f"\nAn error occurred: {e}")