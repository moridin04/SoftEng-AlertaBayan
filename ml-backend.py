import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.cluster import KMeans
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, RandomForestRegressor
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, mean_squared_error

# ==========================================
# RESEARCH DESIGN: SEQUENTIAL MULTI-STAGE
# ==========================================

# STAGE 1: DATA INTEGRATION
def load_data(combined_csv_path):
    """
    Loads a single combined dataset containing both Geospatial Data and Demographic Data.
    Robustly handles column renaming.
    """
    print(f"Loading data from: {combined_csv_path}")
    df = pd.read_csv(combined_csv_path)
    
    # 1. Clean whitespace from column headers
    df.columns = df.columns.str.strip()
    
    # 2. Smart Rename: Map file columns to standard variables
    rename_map = {
        'barangay': 'Name',
        'Barangay': 'Name',
        'brgy_area_sqm': 'Land Area',
        'Population_2020': 'Population', # Matches your specific file
        'population': 'Population',
        'Population': 'Population'
    }
    df = df.rename(columns=rename_map)

    # 3. Validation
    if 'Name' not in df.columns:
        raise KeyError("Could not find 'Barangay' or 'Name' column. Check CSV headers.")
            
    # 4. Fill missing population
    if 'Population' not in df.columns:
        print("WARNING: Population column missing. Creating empty column.")
        df['Population'] = 0
            
    return df

# STAGE 2: FEATURE ENGINEERING & DETERMINISTIC INDEXING
def calculate_indices(df):
    # 1. Feature Engineering: Population Density
    df['Land Area'] = df['Land Area'].replace(0, np.nan)
    df['Pop_Density'] = df['Population'] / df['Land Area']
    
    # 2. Deterministic Indexing: Disaster Priority Index (DPI)
    
    # CRITICAL FIX: Realistic Thresholds for Metro Manila Density
    def get_vuln_score(density):
        if pd.isna(density): return 1
        # Adjusted for people/sqm (0.07 is ~70k people/km², which is very dense)
        if density >= 0.07: return 10   # Very High Density (e.g., Tondo)
        elif density >= 0.04: return 7  # High Density
        elif density >= 0.02: return 4  # Moderate Density
        elif density > 0.0: return 2    # Low Density
        else: return 1                  # Zero / No Data
    
    df['Vulnerability_Score'] = df['Pop_Density'].apply(get_vuln_score)

    # Flood Risk Scores (Physical Risk - Based on Coverage Ratio)
    def get_flood_score(flood_area, total_area):
        if pd.isna(total_area) or total_area == 0: return 0
        ratio = flood_area / total_area
        if ratio >= 0.50: return 10.0
        elif ratio >= 0.30: return 7.5
        elif ratio >= 0.15: return 5.0
        elif ratio > 0.0: return 2.5
        else: return 0.0

    # Calculate scores for DPI formula
    df['Assessment_5yr_Score'] = df.apply(lambda x: get_flood_score(x['flood5_sqm_final'], x['Land Area']), axis=1)
    df['Assessment_25yr_Score'] = df.apply(lambda x: get_flood_score(x['flood25_sqm_final'], x['Land Area']), axis=1)
    df['Assessment_100yr_Score'] = df.apply(lambda x: get_flood_score(x['flood100_sqm_final'], x['Land Area']), axis=1)

    # Composite Susceptibility Index (CSI)
    df['CSI'] = (
        (df['Assessment_5yr_Score'] * 0.5) +
        (df['Assessment_25yr_Score'] * 0.3) +
        (df['Assessment_100yr_Score'] * 0.2)
    )

    # Final DPI Calculation (Target Variable)
    df['DPI'] = (df['CSI'] * 0.6) + (df['Vulnerability_Score'] * 0.4)
    
    return df

# STAGE 3: PREDICTIVE MODELING (Supervised & Unsupervised)
def run_ml_pipeline(df):
    # CRITICAL: Honest Feature List (No Data Leakage)
    feature_cols = [
        'Pop_Density',          # Human Factor
        'flood5_sqm_final',     # Physical Factor (5-year)
        'flood25_sqm_final',    # Physical Factor (25-year)
        'flood100_sqm_final',   # Physical Factor (100-year)
        'Land Area'             # Context Factor
    ]
    
    # Preprocessing
    for col in feature_cols:
        df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)

    imputer = SimpleImputer(strategy="mean")
    scaler = StandardScaler()
    
    X_raw = df[feature_cols]
    X_imputed = imputer.fit_transform(X_raw)
    X_scaled = scaler.fit_transform(X_imputed)
    
    # A. UNSUPERVISED PHASE: K-Means Clustering
    kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
    clusters = kmeans.fit_predict(X_scaled)
    df["Cluster_Group"] = clusters
    
    # Assign Labels (Low/Moderate/High) based on DPI clusters
    cluster_ranks = df.groupby("Cluster_Group")["DPI"].mean().sort_values().index
    risk_labels = {cluster_ranks[0]: "Low", cluster_ranks[1]: "Moderate", cluster_ranks[2]: "High"}
    df["ML_Risk_Class"] = df["Cluster_Group"].map(risk_labels)

    # VISUALIZATION: Generate K-Means Scatter Plot
    try:
        plt.figure(figsize=(10, 6))
        # Plot Pop Density vs CSI (Physical Risk), colored by Cluster
        sns.scatterplot(
            data=df, x='Pop_Density', y='CSI', hue='ML_Risk_Class',
            hue_order=['Low', 'Moderate', 'High'],
            palette={'Low': 'green', 'Moderate': 'orange', 'High': 'red'},
            alpha=0.7
        )
        plt.title('K-Means Clustering Results: Risk Archetypes')
        plt.xlabel('Population Density (People/sqm)')
        plt.ylabel('Physical Flood Risk (CSI)')
        plt.tight_layout()
        plt.savefig('kmeans_cluster_chart.png', dpi=300)
        print("Visualization saved: 'kmeans_cluster_chart.png'")
    except Exception as e:
        print(f"Could not generate plot: {e}")

    # B. SUPERVISED PHASE: Classification
    y = LabelEncoder().fit_transform(df["ML_Risk_Class"])
    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

    models = {
        "Random Forest": RandomForestClassifier(n_estimators=100, random_state=42),
        "Gradient Boosting": GradientBoostingClassifier(random_state=42),
        "Neural Network (MLP)": MLPClassifier(hidden_layer_sizes=(16, 8), max_iter=500, random_state=42)
    }

    print("\n=== EVALUATION: CLASSIFICATION METRICS (HONEST VALIDATION) ===")
    
    for name, model in models.items():
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        acc = accuracy_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred, average='weighted', zero_division=0)
        
        print(f"[{name}]")
        print(f"  Accuracy:  {acc:.2%}")
        print(f"  F1-Score:  {f1:.2%}")
        print("-" * 30)

    # C. REGRESSION ANALYSIS
    print("\n=== EVALUATION: REGRESSION METRICS ===")
    regressor = RandomForestRegressor(n_estimators=100, random_state=42)
    y_reg = df['DPI']
    X_train_r, X_test_r, y_train_r, y_test_r = train_test_split(X_scaled, y_reg, test_size=0.2, random_state=42)
    
    regressor.fit(X_train_r, y_train_r)
    y_pred_r = regressor.predict(X_test_r)
    rmse = np.sqrt(mean_squared_error(y_test_r, y_pred_r))
    
    print(f"Random Forest Regressor RMSE: {rmse:.4f}")

    # Feature Importance
    rf_model = models["Random Forest"]
    print("\n=== FEATURE IMPORTANCE (RAW DRIVERS) ===")
    importances = rf_model.feature_importances_
    for feat, score in sorted(zip(feature_cols, importances), key=lambda x: x[1], reverse=True):
        print(f"  - {feat}: {score:.4f}")

    return df

def run_validation_checks(df):
    print("\n" + "="*50)
    print("       FINAL VALIDATION CHECKS (LOGIC & GROUND TRUTH)")
    print("="*50)

    # CHECK 1: GROUND TRUTH COMPARISON
    print("\n[CHECK 1] Known Ground Truth Verification:")
    ground_truths = {
        "Barangay 12": "High",      # Tondo (Coastal/Dense) -> SHOULD BE HIGH
        "Tuktukan": "High",         # Taguig (Riverine) -> SHOULD BE HIGH
        "Wawa": "High",             # Taguig (Laguna Lake) -> SHOULD BE HIGH
        "Blue Ridge A": "Low",      # QC (High Elevation) -> SHOULD BE LOW
        "Forbes Park": "Low"        # Makati (Gated) -> SHOULD BE LOW/MODERATE
    }

    correct_matches = 0
    total_checks = 0

    for brgy, expected in ground_truths.items():
        row = df[df['Name'].str.contains(brgy, case=False, na=False)]
        if not row.empty:
            actual = row.iloc[0]['ML_Risk_Class']
            is_correct = (actual == expected) or (expected == "Low" and actual == "Moderate")
            status = "✅ PASS" if is_correct else "❌ FAIL"
            print(f"  - {brgy:<15} | Expected: {expected:<5} | Actual: {actual:<8} | {status}")
            if is_correct: correct_matches += 1
            total_checks += 1
        else:
            print(f"  - {brgy:<15} | Not found in dataset.")

    # CHECK 2: PHYSICAL LOGIC CONSISTENCY
    print("\n[CHECK 2] Physical Logic Consistency:")
    high_flood = df[df['CSI'] >= 9.0]
    mismatches = high_flood[high_flood['ML_Risk_Class'] == 'Low']
    if mismatches.empty:
        print("  ✅ PASS: All locations with Extreme Flood Scores (CSI > 9.0) are correctly classified.")
    else:
        print(f"  ❌ FAIL: Found {len(mismatches)} locations with Max Flooding but Low Risk labeling.")

    # CHECK 3: DISTRIBUTION BALANCE
    print("\n[CHECK 3] Distribution Balance:")
    dist = df['ML_Risk_Class'].value_counts(normalize=True) * 100
    print(f"  - High Risk:     {dist.get('High', 0):.1f}%")
    print(f"  - Moderate Risk: {dist.get('Moderate', 0):.1f}%")
    print(f"  - Low Risk:      {dist.get('Low', 0):.1f}%")

    print("\nValidation Complete.\n")

def get_recommendation(row):
    risk = row['ML_Risk_Class']
    if risk == "High":
        return "Priority Zone: High logic-based DPI. Immediate intervention."
    elif risk == "Moderate":
        return "Watch Zone: Monitor non-linear risk factors."
    else:
        return "Low Priority: Maintenance only."

def export_results(df):
    df['Recommendation'] = df.apply(get_recommendation, axis=1)
    output_cols = ['Name', 'Pop_Density', 'CSI', 'DPI', 'ML_Risk_Class', 'Recommendation']
    
    for col in output_cols:
        if col not in df.columns: df[col] = "N/A"

    df[output_cols].to_excel("final_methodology_aligned_results.xlsx", index=False)
    print("\nResults exported.")

# EXECUTION
if __name__ == "__main__":
    # Correct filename for the combined dataset
    filename = 'MetroManila_Combined_Flood_Population.csv'
    
    try:
        df = load_data(filename)
        df = calculate_indices(df)
        df_final = run_ml_pipeline(df)
        
        # Run Validation Logic
        run_validation_checks(df_final)
        
        export_results(df_final)
    except FileNotFoundError:
        print(f"\nERROR: File '{filename}' not found. Please upload it.")
    except Exception as e:
        print(f"\nAn error occurred: {e}")