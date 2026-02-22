import geopandas as gpd
import pandas as pd
import os

# ==========================================
# 1. CONFIGURATION
# ==========================================
# This makes sure Python looks in the same folder as the script
script_dir = os.path.dirname(os.path.abspath(__file__))
def get_path(filename): return os.path.join(script_dir, filename)

# ADMIN FILE
admin_filename = 'phl_admbnda_adm4_psa_namria_20231106.shp'
admin_file_path = get_path(admin_filename)

# FLOOD FILES (Input) mapped to OUTPUT COLUMN NAMES
flood_scenarios = {
    'flood5_sqm_final': get_path('MetroManila_Flood_5year.shp'),
    'flood25_sqm_final': get_path('MetroManila_Flood_25year.shp'),
    'flood100_sqm_final': get_path('MetroManila_Flood_100year.shp')
}

# COLUMN MAPPING (HDX Standard)
col_barangay = 'ADM4_EN'  # Column for Barangay Name
col_city = 'ADM3_EN'      # Column for City Name
col_district = 'ADM2_EN'  # Column for District/Province Name

# ==========================================
# 2. LOAD AND PREPARE ADMIN MAP
# ==========================================
print("-" * 40)
print(f"Working in: {script_dir}")
print("-" * 40)

if not os.path.exists(admin_file_path):
    print(f"CRITICAL ERROR: Admin file not found.")
    print(f"Please ensure '{admin_filename}' is in the folder.")
    exit()

print("1. Loading Barangay Admin Map...")
admin_gdf = gpd.read_file(admin_file_path)

# REPROJECT TO METERS (EPSG:32651 - Philippines Zone 51N)
print("   -> Converting map to Meters (EPSG:32651)...")
admin_gdf = admin_gdf.to_crs(epsg=32651)

# Calculate Total Area of each Barangay
print("   -> Calculating Barangay total areas...")
admin_gdf['brgy_area_sqm'] = admin_gdf.geometry.area

# Initialize the Master DataFrame
# We include CITY and DISTRICT now to ensure uniqueness
cols_to_keep = [col_district, col_city, col_barangay, 'brgy_area_sqm']
master_df = pd.DataFrame(admin_gdf[cols_to_keep]).drop_duplicates(subset=[col_barangay, col_city])

# Rename columns to output format
master_df.columns = ['district', 'city', 'barangay', 'brgy_area_sqm']

# ==========================================
# 3. PROCESS FLOOD INTERSECTIONS (CORRECTED)
# ==========================================
for output_col, flood_path in flood_scenarios.items():
    print(f"\n2. Processing scenario: {output_col}...")
    
    if os.path.exists(flood_path):
        # Load Flood Map
        flood_gdf = gpd.read_file(flood_path)
        
        # Filter out "0" hazard
        if 'Var' in flood_gdf.columns:
            flood_gdf = flood_gdf[flood_gdf['Var'] > 0]
        elif 'GRIDCODE' in flood_gdf.columns:
             flood_gdf = flood_gdf[flood_gdf['GRIDCODE'] > 0]

        # Reproject to Meters
        flood_gdf = flood_gdf.to_crs(admin_gdf.crs)
        
        # SPATIAL INTERSECTION
        print("   -> Calculating intersection (this is the slow part)...")
        # We only need the geometry from flood_gdf to save memory
        intersection = gpd.overlay(admin_gdf, flood_gdf[['geometry']], how='intersection')
        
        # Calculate the area of these intersecting pieces
        intersection['flooded_part_area'] = intersection.geometry.area
        
        # === THE FIX IS HERE ===
        # We group by BOTH City (col_city) and Barangay (col_barangay)
        # This prevents "Barangay 1" in Manila from mixing with "Barangay 1" in Caloocan
        flood_sum = intersection.groupby([col_city, col_barangay])['flooded_part_area'].sum().reset_index()
        
        # Rename columns to match the Master DF keys
        flood_sum.columns = ['city', 'barangay', output_col]
        
        # Merge on BOTH keys
        master_df = master_df.merge(flood_sum, on=['city', 'barangay'], how='left')
        # =======================
        
        # Fill NaNs with 0.0
        master_df[output_col] = master_df[output_col].fillna(0.0)
        
    else:
        print(f"   [WARNING] File missing: {flood_path}")
        master_df[output_col] = 0.0

# ==========================================
# 4. FINAL CLEANUP AND SAVE
# ==========================================
print("\n3. Cleaning up data...")

numeric_cols = ['brgy_area_sqm', 'flood5_sqm_final', 'flood25_sqm_final', 'flood100_sqm_final']
master_df[numeric_cols] = master_df[numeric_cols].round(2)

# Sanity Check: Ensure Flood Area never exceeds Barangay Area
# (If slight projection errors cause >100%, cap it at the barangay area)
for col in ['flood5_sqm_final', 'flood25_sqm_final', 'flood100_sqm_final']:
    mask = master_df[col] > master_df['brgy_area_sqm']
    if mask.any():
        print(f"   [NOTICE] Capped {mask.sum()} rows where {col} slightly exceeded land area.")
        master_df.loc[mask, col] = master_df.loc[mask, 'brgy_area_sqm']

output_csv = get_path('MetroManila_Flood_Dataset_SQM.csv')
master_df.to_csv(output_csv, index=False)

print("=" * 40)
print("COMPLETED SUCCESSFULLY")
print(f"File saved: {output_csv}")
print("Preview:")
print(master_df.head())