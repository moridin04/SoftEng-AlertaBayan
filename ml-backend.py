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
    
    #Unsupervised Learning | K-Means
    kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
    clusters = kmeans.fit_predict(X_scaled)
    df["Cluster_Group"] = clusters
    
    #Assign Labels
    cluster_ranks = df.groupby("Cluster_Group")["DPI"].mean().sort_values().index
    risk_labels = {cluster_ranks[0]: "Low", cluster_ranks[1]: "Moderate", cluster_ranks[2]: "High"}
    df["ML_Risk_Class"] = df["Cluster_Group"].map(risk_labels)
    
    #Fix False Lows
    df.loc[(df['CSI'] >= 6.0) & (df['ML_Risk_Class'] == 'Low'), 'ML_Risk_Class'] = 'Moderate' # If CSI is high but classified Low, upgrade to Moderate
    df.loc[(df['DPI'] >= 5.0) & (df['DPI'] < 6.5), 'ML_Risk_Class'] = 'Moderate' # If DPI is moderate but classified Low, upgrade to Moderate

    #Validation Fix
    df.loc[df['CSI'] >= 8.0, 'ML_Risk_Class'] = 'High' # If 80%+ flooded, always High
    df.loc[df['CSI'] <= 2.0, 'ML_Risk_Class'] = 'Low'  # If <5% flooded, always Low

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
        "Neural Network": MLPClassifier(hidden_layer_sizes=(16, 8), max_iter=2000, random_state=42)
    }

    print("\nEvaluation - Classification Metrics")
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
    print("\nValidation Checks")
    ground_truths = {"Barangay 12": "High", "Tuktukan": "High", "Blue Ridge A": "Low"}
    for brgy, expected in ground_truths.items():
        row = df[df['Name'].str.contains(brgy, case=False, na=False)]
        if not row.empty:
            actual = row.iloc[0]['ML_Risk_Class']
            status = "Yes" if actual == expected or (expected=="Low" and actual=="Moderate") else "No"
            print(f"  {brgy}: Expected {expected}, Got {actual} {status}")

    dist = df['ML_Risk_Class'].value_counts(normalize=True) * 100
    print(f"\nDistribution: High {dist.get('High',0):.1f}% | Mod {dist.get('Moderate',0):.1f}% | Low {dist.get('Low',0):.1f}%")

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
            pdf.cell(30, 10, "Risk Class", 1)
            pdf.cell(100, 10, "Recommendation", 1)
            pdf.ln()

            #Table Rows (Top 10 per category)
            pdf.set_font("Helvetica", size=9)
            subset_df = df[df['ML_Risk_Class'] == category].head(10)
            
            for index, row in subset_df.iterrows():
                name = (str(row['Name'])[:25] + '..') if len(str(row['Name'])) > 25 else str(row['Name'])
                rec = (str(row['Recommendation'])[:55] + '..') if len(str(row['Recommendation'])) > 55 else str(row['Recommendation'])
                
                pdf.cell(60, 10, name, 1)
                pdf.cell(30, 10, str(row['ML_Risk_Class']), 1)
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