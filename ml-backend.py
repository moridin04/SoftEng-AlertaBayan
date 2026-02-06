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

#Sequential ML Pipeline for Flood Risk Assessment
#Data Integration
def load_data(combined_csv_path):
    print(f"Data loading: {combined_csv_path}")
    df = pd.read_csv(combined_csv_path)
    
    #Clean Whitespace
    df.columns = df.columns.str.strip()
    
    #Rename
    rename_map = {
        'barangay': 'Name', 'Barangay': 'Name',
        'brgy_area_sqm': 'Land Area',
        'Population_2020': 'Population', 'population': 'Population', 'Population': 'Population'
    }
    df = df.rename(columns=rename_map)

    if 'Name' not in df.columns: raise KeyError("Column 'Name' missing.")
    if 'Population' not in df.columns: df['Population'] = 0
            
    return df

#Feature Engineering
def engineer_features(df):
    #Basic Cleanup
    df['Land Area'] = df['Land Area'].replace(0, np.nan)
    
    #Flood Coverage Ratios (how much of the barangay is wet)
    df['flood5_coverage'] = df['flood5_sqm_final'] / df['Land Area']
    df['flood25_coverage'] = df['flood25_sqm_final'] / df['Land Area']
    df['flood100_coverage'] = df['flood100_sqm_final'] / df['Land Area']
    
    #Population Density (People per Hectare)
    df['Pop_Density'] = (df['Population'] / df['Land Area']) 
    
    #Interaction Terms: Affected Population
    #multiplies People * Water
    #if Pop is high but Water is 0, this is 0 (no risk)
    df['Affected_Pop_5yr'] = df['Population'] * df['flood5_coverage']
    df['Affected_Pop_25yr'] = df['Population'] * df['flood25_coverage']
    df['Affected_Pop_100yr'] = df['Population'] * df['flood100_coverage']
    
    #Flood Growth Rate (5yr to 25yr)
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

    #Calculate sub-scores using the NEW coverage columns
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
    
    #Unsupervised Learning | K-Means
    kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
    clusters = kmeans.fit_predict(X_scaled)
    df["Cluster_Group"] = clusters
    
    #Assign Labels
    cluster_ranks = df.groupby("Cluster_Group")["DPI"].mean().sort_values().index
    risk_labels = {cluster_ranks[0]: "Low", cluster_ranks[1]: "Moderate", cluster_ranks[2]: "High"}
    df["ML_Risk_Class"] = df["Cluster_Group"].map(risk_labels)

    #Validation Fix
    df.loc[df['CSI'] >= 9.0, 'ML_Risk_Class'] = 'High' #If 80%+ flooded, always High
    df.loc[df['CSI'] <= 2.0, 'ML_Risk_Class'] = 'Low'  #If <5% flooded, always Low

    #Visualization
    try:
        plt.figure(figsize=(10, 6))
        sns.scatterplot(
            data=df, x='Pop_Density', y='CSI', hue='ML_Risk_Class',
            hue_order=['Low', 'Moderate', 'High'],
            palette={'Low': 'green', 'Moderate': 'orange', 'High': 'red'}, alpha=0.7
        )
        plt.title('Risk Archetypes: Pop Density vs Physical Risk')
        plt.tight_layout()
        plt.savefig('kmeans_cluster_chart.png', dpi=300)
    except: pass

    #Supervised Learning | Classification
    y = LabelEncoder().fit_transform(df["ML_Risk_Class"])
    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

    models = {
        "Random Forest": RandomForestClassifier(n_estimators=100, random_state=42),
        "Gradient Boosting": GradientBoostingClassifier(random_state=42),
        "Neural Network": MLPClassifier(hidden_layer_sizes=(16, 8), max_iter=500, random_state=42)
    }

    print("\nEvaluation: Classification Metrics")
    for name, model in models.items():
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        acc = accuracy_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred, average='weighted', zero_division=0)
        print(f"[{name}] Acc: {acc:.2%} | F1: {f1:.2%}")

    #Feature Importance
    rf_model = models["Random Forest"]
    print("\nFeature Importance")
    importances = rf_model.feature_importances_
    for feat, score in sorted(zip(feature_cols, importances), key=lambda x: x[1], reverse=True):
        print(f"  - {feat}: {score:.4f}")

    return df

def run_validation_checks(df):
    print("\nValidation Checks:")
    ground_truths = {"Barangay 12": "High", "Tuktukan": "High", "Blue Ridge A": "Low"}
    for brgy, expected in ground_truths.items():
        row = df[df['Name'].str.contains(brgy, case=False, na=False)]
        if not row.empty:
            actual = row.iloc[0]['ML_Risk_Class']
            status = "✅" if actual == expected or (expected=="Low" and actual=="Moderate") else "❌"
            print(f"  {brgy}: Expected {expected}, Got {actual} {status}")

    dist = df['ML_Risk_Class'].value_counts(normalize=True) * 100
    print(f"\nDistribution: High {dist.get('High',0):.1f}% | Mod {dist.get('Moderate',0):.1f}% | Low {dist.get('Low',0):.1f}%")

def get_recommendation(row):
    risk = row['ML_Risk_Class']
    if risk == "High": return "Priority: Structural Intervention"
    elif risk == "Moderate": return "Watch: Monitor evacuation routes"
    else: return "Low Priority: Maintenance only"

def export_results(df):
    df['Recommendation'] = df.apply(get_recommendation, axis=1)
    output_cols = ['Name', 'Pop_Density', 'CSI', 'DPI', 'ML_Risk_Class', 'Recommendation']
    for col in output_cols: 
        if col not in df.columns: df[col] = "N/A"
    df[output_cols].to_excel("final_methodology_aligned_results.xlsx", index=False)
    print("\nResults exported.")

if __name__ == "__main__":
    filename = 'MetroManila_Combined_Flood_Population.csv'
    try:
        df = load_data(filename)
        df = calculate_indices(df)
        df_final = run_ml_pipeline(df)
        run_validation_checks(df_final)
        export_results(df_final)
    except Exception as e: print(f"Error: {e}")