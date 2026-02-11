import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.cluster import KMeans
from sklearn.ensemble import (
    GradientBoostingClassifier,
    GradientBoostingRegressor,
    RandomForestClassifier,
    RandomForestRegressor,
)
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier, MLPRegressor
from sklearn.metrics import (
    accuracy_score,
    adjusted_rand_score,
    calinski_harabasz_score,
    davies_bouldin_score,
    f1_score,
    mean_squared_error,
    silhouette_score,
)
from fpdf import FPDF, XPos, YPos

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
            
    return df

#Feature Engineering
def engineer_features(df):
    # Basic Cleanup
    df['Land Area'] = df['Land Area'].replace(0, np.nan)
    
    #Flood Coverage Ratios (How much of barangay is wet?)
    df['flood5_coverage'] = df['flood5_sqm_final'] / df['Land Area']
    df['flood25_coverage'] = df['flood25_sqm_final'] / df['Land Area']
    df['flood100_coverage'] = df['flood100_sqm_final'] / df['Land Area']
    
    #Population Density (People per Hectare)
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
    def get_vuln_score(density):
        if pd.isna(density): return 1
        # Thresholds (People per SQM): 0.07 is ~70k/km2
        if density >= 0.07: return 10
        elif density >= 0.04: return 7
        elif density >= 0.02: return 4
        elif density > 0.0: return 2
        else: return 1
    
    df['Vulnerability_Score'] = df['Pop_Density'].apply(get_vuln_score)

    #Flood Risk Scores (Physical Risk)
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
    
    #Handle NaNs
    for col in feature_cols:
        df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)

    imputer = SimpleImputer(strategy="mean")
    scaler = StandardScaler()
    
    X_raw = df[feature_cols]
    X_imputed = imputer.fit_transform(X_raw)
    X_scaled = scaler.fit_transform(X_imputed)
    
    #Deterministic Benchmarking Target ("Ground Truth"): DPI-derived risk class
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
    #Backwards-compatible alias used across exports/PDF
    df["ML_Risk_Class"] = df["DPI_Risk_Class"]

    #Unsupervised Phase | K-Means for exploratory "Risk Archetypes" (not labels)
    kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
    clusters = kmeans.fit_predict(X_scaled)
    df["Cluster_Group"] = clusters

    #Simple archetype naming based on cluster means (density vs exposure)
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
    except Exception:
        df['Risk_Archetype'] = df['Cluster_Group'].astype(str)

    #Visualization (Archetypes)
    try:
        plt.figure(figsize=(10, 6))
        sns.scatterplot(
            data=df, x='Pop_Density', y='CSI', hue='Risk_Archetype', alpha=0.7
        )
        plt.title('Risk Archetypes (K-Means): Pop Density vs Physical Risk')
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
            km = KMeans(n_clusters=3, random_state=seed, n_init=10)
            alt = km.fit_predict(X_scaled)
            ari_scores.append(adjusted_rand_score(baseline, alt))
        print(f"  Stability (ARI vs seed=42, 10 seeds): mean={np.mean(ari_scores):.3f} | min={np.min(ari_scores):.3f}")
    except Exception as e:
        print(f"  Could not compute stability: {e}")

    #Supervised Phase - Benchmarking (classification + regression)
    y_class = LabelEncoder().fit_transform(df["DPI_Risk_Class"].astype(str))
    y_reg = df["DPI"].astype(float).to_numpy()

    X_train, X_test, y_class_train, y_class_test, y_reg_train, y_reg_test = train_test_split(
        X_scaled,
        y_class,
        y_reg,
        test_size=0.2,
        random_state=42,
        stratify=y_class,
    )

    clf_models = {
        "Random Forest": RandomForestClassifier(n_estimators=200, random_state=42),
        "Gradient Boosting": GradientBoostingClassifier(random_state=42),
        "Neural Network (MLP)": MLPClassifier(hidden_layer_sizes=(32, 16), max_iter=3000, random_state=42),
    }

    reg_models = {
        "Random Forest": RandomForestRegressor(n_estimators=200, random_state=42),
        "Gradient Boosting": GradientBoostingRegressor(random_state=42),
        "Neural Network (MLP)": MLPRegressor(hidden_layer_sizes=(32, 16), max_iter=3000, random_state=42),
    }

    print("\nSupervised Phase - Classification (target: DPI_Risk_Class)")
    clf_results = {}
    for name, model in clf_models.items():
        model.fit(X_train, y_class_train)
        y_pred = model.predict(X_test)
        acc = accuracy_score(y_class_test, y_pred)
        f1 = f1_score(y_class_test, y_pred, average='weighted', zero_division=0)
        clf_results[name] = {"acc": acc, "f1": f1, "model": model}
        print(f"  [{name}] Acc: {acc:.2%} | F1: {f1:.2%}")

    print("\nSupervised Phase - Regression (target: DPI score)")
    reg_results = {}
    for name, model in reg_models.items():
        model.fit(X_train, y_reg_train)
        y_pred = model.predict(X_test)
        rmse = float(np.sqrt(mean_squared_error(y_reg_test, y_pred)))
        reg_results[name] = {"rmse": rmse, "model": model}
        print(f"  [{name}] RMSE: {rmse:.4f}")

    #Select best models for optional output columns
    best_clf_name = max(clf_results.keys(), key=lambda n: clf_results[n]["f1"])
    best_reg_name = min(reg_results.keys(), key=lambda n: reg_results[n]["rmse"])
    best_clf = clf_results[best_clf_name]["model"]
    best_reg = reg_results[best_reg_name]["model"]

    print(f"\nBest Classifier (by F1): {best_clf_name}")
    print(f"Best Regressor (by RMSE): {best_reg_name}")

    #Add predictions to dataframe (proof-of-concept outputs)
    try:
        full_pred_class = best_clf.predict(X_scaled)
        full_pred_dpi = best_reg.predict(X_scaled)
        le = LabelEncoder().fit(df["DPI_Risk_Class"].astype(str))
        df["Predicted_Risk_Class"] = le.inverse_transform(full_pred_class.astype(int))
        df["Predicted_DPI"] = np.clip(full_pred_dpi, 0, 10)
    except Exception:
        pass

    #Feature importance (interpretable models)
    print("\nFeature Importance (Random Forest)")
    try:
        rf_model = clf_models["Random Forest"]
        importances = rf_model.feature_importances_
        for feat, score in sorted(zip(feature_cols, importances), key=lambda x: x[1], reverse=True):
            print(f"  - {feat}: {score:.4f}")
    except Exception as e:
        print(f"  Could not compute feature importance: {e}")

    return df

def run_validation_checks(df):
    print("\nValidation & Consistency Checks")
    print("  Note: DPI is treated as the deterministic baseline label (experimental ground truth).")
    dist = df['ML_Risk_Class'].value_counts(normalize=True) * 100
    print(f"  DPI Risk Distribution: High {dist.get('High',0):.1f}% | Mod {dist.get('Moderate',0):.1f}% | Low {dist.get('Low',0):.1f}%")

    #Domain logic consistency checks (example locations; matches your methodology narrative)
    print("\n  Domain Logic Consistency (examples)")
    checks = {
        "Barangay 12": "High",
        "Tuktukan": "High",
        "Blue Ridge": "Low",
        "Tondo": "High",
        "Taguig": "High",
    }
    for needle, expected in checks.items():
        row = df[df['Name'].astype(str).str.contains(needle, case=False, na=False)]
        if row.empty:
            continue
        r = row.iloc[0]
        actual = r.get('DPI_Risk_Class', r.get('ML_Risk_Class', 'N/A'))
        dpi_val = r.get('DPI', np.nan)
        csi_val = r.get('CSI', np.nan)
        ok = (actual == expected) or (expected == "Low" and actual == "Moderate")
        print(f"    {r['Name']}: Expected {expected} | Got {actual} | DPI={dpi_val:.2f} | CSI={csi_val:.2f} | Pass={ok}")

    try:
        top = df.sort_values('DPI', ascending=False).head(5)[['Name', 'City', 'DPI', 'CSI', 'DPI_Risk_Class']]
        print("\n  Top 5 by DPI")
        print(top.to_string(index=False))
    except Exception:
        pass

def get_recommendation(row):
    risk = row.get('DPI_Risk_Class', row.get('ML_Risk_Class', 'Low'))
    if risk == "High":
        return "Priority Zone: High logic-based DPI. Immediate intervention."
    elif risk == "Moderate":
        return "Watch Zone: Monitor non-linear risk factors."
    else:
        return "Low Priority: Maintenance only."

def export_results(df):
    df['Recommendation'] = df.apply(get_recommendation, axis=1)
    output_cols = [
        'Name',
        'Pop_Density',
        'CSI',
        'DPI',
        'DPI_Risk_Class',
        'ML_Risk_Class',
        'Risk_Archetype',
        'Predicted_Risk_Class',
        'Predicted_DPI',
        'Recommendation',
    ]
    
    for col in output_cols:
        if col not in df.columns: df[col] = "N/A"

    df[output_cols].to_excel("final_methodology_aligned_results.xlsx", index=False)
    print("\nResults exported.")

def generate_pdf_report(df):
    print("Generating PDF Report...")
    try:
        pdf = FPDF()
        pdf.add_page()
        pdf.set_font("Helvetica", size=12)

        #Title
        pdf.set_font("Helvetica", style="B", size=16)
        pdf.cell(
            200,
            10,
            text="Disaster Risk Priority Report (NCR)",
            align="C",
            new_x=XPos.LMARGIN,
            new_y=YPos.NEXT,
        )
        pdf.ln(10)

        #Summary Statistics
        total = len(df)
        high_risk = len(df[df['ML_Risk_Class'] == 'High'])
        pdf.set_font("Helvetica", size=12)
        pdf.cell(200, 10, text=f"Total Locations Analyzed: {total}", new_x=XPos.LMARGIN, new_y=YPos.NEXT)
        pdf.cell(
            200,
            10,
            text=f"Critical High Risk Areas: {high_risk} ({high_risk/total:.1%})",
            new_x=XPos.LMARGIN,
            new_y=YPos.NEXT,
        )
        pdf.ln(5)

        #Loop through each Risk Category (High, Moderate, Low)
        for category in ['High', 'Moderate', 'Low']:
            pdf.set_font("Helvetica", style="B", size=14)
            pdf.cell(
                200,
                10,
                text=f"Category: {category} Risk (Top 10 Samples)",
                new_x=XPos.LMARGIN,
                new_y=YPos.NEXT,
            )
            pdf.ln(2)

            #Table Header
            pdf.set_font("Helvetica", style="B", size=10)
            pdf.cell(60, 10, "Location", 1)
            pdf.cell(30, 10, "DPI Class", 1)
            pdf.cell(100, 10, "Recommendation", 1)
            pdf.ln()

            #Table Rows (Top 10 per category)
            pdf.set_font("Helvetica", size=9)
            subset_df = df[df['ML_Risk_Class'] == category].head(10)
            
            for index, row in subset_df.iterrows():
                name = (str(row['Name'])[:25] + '..') if len(str(row['Name'])) > 25 else str(row['Name'])
                rec = (str(row['Recommendation'])[:55] + '..') if len(str(row['Recommendation'])) > 55 else str(row['Recommendation'])
                
                pdf.cell(60, 10, name, 1)
                pdf.cell(30, 10, str(row.get('DPI_Risk_Class', row.get('ML_Risk_Class', 'N/A'))), 1)
                pdf.cell(100, 10, rec, 1)
                pdf.ln()
            
            pdf.ln(5)

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
        df_final = run_ml_pipeline(df)
        
        # Validation
        run_validation_checks(df_final)
        
        # Exports
        export_results(df_final) #Excel
        generate_pdf_report(df_final) #PDF
        
    except FileNotFoundError:
        print(f"\nERROR: File '{filename}' not found. Please upload it.")
    except Exception as e:
        print(f"\nAn error occurred: {e}")