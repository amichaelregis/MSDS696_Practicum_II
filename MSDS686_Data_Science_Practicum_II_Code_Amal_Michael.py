#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Apr 26 09:21:26 2026
this code is for Data science Practicum 2
Title of Project : Analyse Federal funding Impact on Broadband Equity
the project processes federal funding allocated at a censusblock group level
creates a funding efficiency Index
output Tableau data that can be plotted on a Tableau dashboard 
@author: amalmichael
"""
import pandas as pd
from pathlib import Path
import time
import numpy as np
import gc
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import geopandas as gpd

# Set the base folder variable
base_folder = Path("/Users/amalmichael/2026/MSDS/696 Practicum II/Week7/")

# %%
# Step1 Process FCC data to get competitor product information at census block level
# data is from https://broadbandmap.fcc.gov/data-download/nationwide-data
# 2 files per state ( broadband and Fiber ); total 102 files ( 50 states + Wasgington DC) * 2
# ( Weighted availability Score )
# at the end of this step we have a data frame fcc_bdc_df that has about 178 mil rows

# Start a timer (102 files might take a minute or two)
start_time = time.time()

data_folder = base_folder / "Data/FCC"
file_paths = list(data_folder.glob("bdc_*.csv"))

# 1. Update our target columns based on your inspection
columns_to_keep = [
    'block_geoid', 
    'state_usps', 
    'technology', 
    'max_advertised_download_speed', 
    'max_advertised_upload_speed'
]

# 2. Define the Non-Continental states and territories to EXCLUDE
# AK=Alaska, HI=Hawaii, PR=Puerto Rico, VI=Virgin Islands, GU=Guam, AS=American Samoa, MP=Northern Mariana Islands
exclude_states = ['AK', 'HI', 'PR', 'VI', 'GU', 'AS', 'MP']

df_list = []
print(f"Stitching {len(file_paths)} files together...")

# 3. Loop through all files
for i, file in enumerate(file_paths):
    # Print progress every 10 files
    if (i + 1) % 10 == 0:
        print(f"Processing file {i + 1} of {len(file_paths)}...")
        
    temp_df = pd.read_csv(
        file, 
        usecols=columns_to_keep,
        dtype={'block_geoid': str} # CRITICAL: Keeps leading zeros!
    )
    
    # 4. Filter out non-continental data immediately to save RAM
    temp_df = temp_df[~temp_df['state_usps'].isin(exclude_states)]
    
    df_list.append(temp_df)

# 5. Concatenate everything into one massive DataFrame
fcc_bdc_df = pd.concat(df_list, ignore_index=True)

elapsed_time = round(time.time() - start_time, 2)
print(f"\n Nationwide broadband data processed  in {elapsed_time} seconds!")
print(f"Total Rows (Continental US): {len(fcc_bdc_df):,}")

# Inspect the final structure
print(f"Shape: {fcc_bdc_df.shape}")  # Returns (rows, columns)
print(f"Columns: {fcc_bdc_df.columns.tolist()}")
print(fcc_bdc_df.head())

# %%
# Step2 : Aggregare FCC data at Block group level and calculate WAS
# in this step we do feature engineering
# first, competitors are classified into tiers based on provided speeds
# then  we aggreate the censusblock level data to censusblockgroup level.
# this reduces no of rows from 178 mil to about 240,000 rows
# during this aggregation,we calculate tier1,tier2, tier3 competitor counts for each blockgroup.
# Finally we  create  a new metric called WAS ( Weighted Internet Availability Score)
# WAS indicates internet quality.Higher the WAS, better Intrenet.
# the end of this step, we have a dataframe bg_summary that has WAS at blockgroup level 

print("--- 1. Memory-Optimized Feature Engineering ---")
# Using int8 instead of default int64 saves ~4GB of RAM across these 3 columns!
fcc_bdc_df['Tier1'] = np.where(fcc_bdc_df['max_advertised_download_speed'] >= 1000, 1, 0).astype(np.int8)
fcc_bdc_df['Tier2'] = np.where((fcc_bdc_df['max_advertised_download_speed'] >= 100) & (fcc_bdc_df['max_advertised_download_speed'] < 1000), 1, 0).astype(np.int8)
fcc_bdc_df['Tier3'] = np.where((fcc_bdc_df['max_advertised_download_speed'] >= 25) & (fcc_bdc_df['max_advertised_download_speed'] < 100), 1, 0).astype(np.int8)

print("--- 2. FIRST AGGREGATION: Block Level (15-digit) ---")
# This compresses 178M rows down to roughly 8-10 Million rows (Total US populated blocks)
block_summary = fcc_bdc_df.groupby('block_geoid').agg(
    Tier1_Count=('Tier1', 'sum'),
    Tier2_Count=('Tier2', 'sum'),
    Tier3_Count=('Tier3', 'sum'),
    Block_Max_Downspeed=('max_advertised_download_speed', 'max')
).reset_index()

print("--- 3.  MEMORY PURGE ---")
# Delete the 23GB raw dataframe before attempting the next step
del fcc_bdc_df
gc.collect()
print(" Raw data cleared. ")

print("--- 4. SECOND AGGREGATION: Block Group Level (12-digit) ---")
# Extract the 12-digit Block Group ID from the much smaller block_summary
# (Doing this string slice here instead of on the 178M row DF saves massive processing time)
block_summary['bg_geoid'] = block_summary['block_geoid'].str[:12]

# Calculate the Block Group summary
bg_summary = block_summary.groupby('bg_geoid').agg(
    Tier1_Competitors=('Tier1_Count', 'mean'), # Average Tier 1 choices per block
    Tier2_Competitors=('Tier2_Count', 'mean'), # Average Tier 2 choices per block
    Tier3_Competitors=('Tier3_Count', 'mean'), # Average Tier 3 choices per block
    Max_Downspeed=('Block_Max_Downspeed', 'max') # Absolute max speed in the BG
).reset_index()

# Clean up the intermediate Block Summary to save more memory
del block_summary
gc.collect()
print(" Intermediate block summary cleared.")

print("--- 5. Computing National WAS ---")
bg_summary['Tier1_Competitors'] = bg_summary['Tier1_Competitors'].round(2)
bg_summary['Tier2_Competitors'] = bg_summary['Tier2_Competitors'].round(2)
bg_summary['Tier3_Competitors'] = bg_summary['Tier3_Competitors'].round(2)

# --- YOUR DEFINED WEIGHTS ---
weight_tier1 = 5  
weight_tier2 = 3  
weight_tier3 = 1  

bg_summary['WAS'] = (
    (bg_summary['Tier1_Competitors'] * weight_tier1) +
    (bg_summary['Tier2_Competitors'] * weight_tier2) +
    (bg_summary['Tier3_Competitors'] * weight_tier3)
).round(2)

print("---Normalizing WAS to 0-100 Scale ---")

# Calculate min and max
was_min = bg_summary['WAS'].min()
was_max = bg_summary['WAS'].max()

# Overwrite the WAS column directly
bg_summary['WAS'] = (
    (bg_summary['WAS'] - was_min) / (was_max - was_min) * 100
).round(2)

print(f"Normalization complete. Range: {bg_summary['WAS'].min()} to {bg_summary['WAS'].max()}")

print(f" Feature Engineering complete! Total National Block Groups: {len(bg_summary):,}")

# Inspect the final structure
print(f"Shape: {bg_summary.shape}")  # Returns (rows, columns)
print(f"Columns: {bg_summary.columns.tolist()}")
print(bg_summary.head())
# %% Step3 : Process RDOF Data
# in this step, we process RDOF ( Rural Digital Opportunity Fund ) data
# data is from url : https://www.fcc.gov/auction/904
# at the end of this step we have an extra column RDOF_Allocation to dataframe bg_summary
# no of rows ~235000

# Define your exact local file path
csv_path = base_folder / 'Data/RDOF/auction904_updated_cbg.csv'

print("Loading RDOF Block Group Data...")
# 1. Load the data, ensuring the ID column is read as a string
df = pd.read_csv(csv_path, dtype={'census_id': str})

# 2. Rename the columns to match your target schema
df = df.rename(columns={
    'census_id': 'bg_geoid', 
    'reserve_price': 'RDOF_Allocation'
})

print("Merging directly into National bg_summary...")
# 3. Merge only the necessary columns into your main bg_summary dataframe
# Passing df[['bg_geoid', 'RDOF_Allocation']] saves memory during the join
bg_summary = pd.merge(bg_summary, df[['bg_geoid', 'RDOF_Allocation']], on='bg_geoid', how='left')

# 4. Clean up block groups that didn't receive any RDOF funding
bg_summary['RDOF_Allocation'] = bg_summary['RDOF_Allocation'].fillna(0)

# Final validation
total_rows = len(bg_summary)
print(f" Final Validation: The bg_summary dataframe has {total_rows:,} rows.")

# 5. Final memory cleanup
del df
gc.collect()
print(" National RDOF Integration Complete!")

# Inspect the final structure
print(f"Shape: {bg_summary.shape}")  # Returns (rows, columns)
print(f"Columns: {bg_summary.columns.tolist()}")
print(bg_summary.head())

# %% Step4 : Add demographics data - Median Income
# data is from census.gov ; have to get 50 states data by using api calls
# eg : https://api.census.gov/data/2023/acs/acs5
# ?get=NAME,B19013_001E&for=block%20group:*&in=state:36%20county:119&key=xxxxxxxxxxxxxxxxx
# once the files are downloaded, a preprocess step consolidates the files together into a single file.
# in this step here, we combine the preprocesses file data into opur main data frame

# Define File Paths
income_path = base_folder / 'Data/CensusIncome/national_blockgroup_median_income.csv'

print(f" BASELINE: Starting with {len(bg_summary):,} block groups in bg_summary.\n")

## --- 1. INCOME DATA MERGE ---
print("Loading Income Data...")
df_income = pd.read_csv(income_path, dtype={'census_block_group': str})
df_income = df_income.rename(columns={'census_block_group': 'bg_geoid'})
df_income['bg_geoid'] = df_income['bg_geoid'].str.zfill(12)
print(f"    -> Loaded {len(df_income):,} income records.")

bg_summary = pd.merge(bg_summary, df_income, on='bg_geoid', how='left')
del df_income
gc.collect()

# Checkpoint 1 Validation (CORRECTED)
print(f" Post-Income Merge: bg_summary has {len(bg_summary):,} rows.")
# Updated the target column to match your specific dataset
matched_income = bg_summary['median_income'].notna().sum() 
print(f"    -> {matched_income:,} block groups successfully matched to income data.\n")


# Inspect the final structure
print(f"Shape: {bg_summary.shape}")  # Returns (rows, columns)
print(f"Columns: {bg_summary.columns.tolist()}")
print(bg_summary.head())

# %% Step5 : Add demographics data - Population , Minority Population and Minority Pct
# data is from census.gov ; have to get 50 states data by using api calls
# once downloaded a preprocess step consolidates all files and computes Minority Population and Pct
# in this step here, we combine the preprocesses file data into opur main data frame

# --- 2. MINORITY DATA MERGE ---
minority_path = base_folder / 'Data/CensusMinorityPct/national_blockgroup_minority_pct.csv'

print("Loading Minority Data...")
# Loading the file we just created with underscored column names
df_minority = pd.read_csv(minority_path, dtype={'census_block_group': str})

# Rename the joining key to match your main dataframe
df_minority = df_minority.rename(columns={'census_block_group': 'bg_geoid'})

# Ensure the 12-digit GEOID format is preserved
df_minority['bg_geoid'] = df_minority['bg_geoid'].str.zfill(12)
print(f"     -> Loaded {len(df_minority):,} minority records.")

# Merge including all the new columns: 
# Population, White_Population, Minority_Population, and MINORITY_PCT
bg_summary = pd.merge(bg_summary, df_minority, on='bg_geoid', how='left')

# Clean up memory immediately
del df_minority
gc.collect()

# --- Checkpoint 2 Validation ---
print(f" Post-Minority Merge: bg_summary has {len(bg_summary):,} rows.")

# Inspect the final structure
print(f"Shape: {bg_summary.shape}")  # Returns (rows, columns)
print(f"Columns: {bg_summary.columns.tolist()}")
print(bg_summary.head())

# %% Step 6: Data Profiling of Numeric Columns

# 1. Identify all numeric columns in bg_summary
numeric_cols = bg_summary.select_dtypes(include=['number']).columns.tolist()

print(f" Profiling {len(numeric_cols)} numeric columns:")
print(f" Columns: {numeric_cols}\n")

# 2. Generate descriptive statistics
# .T (Transpose) makes it easier to read if you have many columns
profiling_stats = bg_summary[numeric_cols].describe().T

# 3. Add a check for Missing Values (Nulls)
profiling_stats['missing_values'] = bg_summary[numeric_cols].isnull().sum()

# 4. Add Skewness check (important for broadband/population data)
profiling_stats['skewness'] = bg_summary[numeric_cols].skew()

# Display the summary table
print(profiling_stats)

# Optional: Quick check for potential outliers
print("\n🔍 Potential Outlier Check (Max vs 75th percentile):")
for col in numeric_cols:
    max_val = bg_summary[col].max()
    p75 = bg_summary[col].quantile(0.75)
    if max_val > p75 * 5: # Flagging if max is 5x higher than 75th percentile
        print(f"     Column '{col}' has potential extreme outliers (Max: {max_val:,.2f}, 75th%: {p75:,.2f})")


# %% Step 7 : Limit Analysis to only where RDOF ALlocation is atleast a $1000

# --- 3. FINAL SUMMARY ---
df_core = bg_summary[(bg_summary['RDOF_Allocation'] >= 1000) ].copy()

print(" Pipeline Complete. Final Status:")
print(f" Total National Block Groups: {len(bg_summary):,}")
print(f" Core Range ($1000-Max) Block Groups: {len(df_core):,}")

# %% Step 8a. EDA1

# Create a copy or just divide the series for the plot
plt.figure(figsize=(10, 6))

# Plotting with values divided by 1000
sns.histplot(df_core['RDOF_Allocation'] / 1000, bins=50, kde=True, color='teal')

plt.title('Distribution of RDOF Funding Amounts (Funded Areas Only)')
plt.xlabel('Allocation Amount (in Thousands of $)')
plt.ylabel('Frequency')

# Optional: Add a comma separator to the y-axis for better readability
plt.gca().yaxis.set_major_formatter(ticker.StrMethodFormatter('{x:,.0f}'))

plt.show()

# %% Step 8b. EDA2

# Filtering the data for the 'Zoom' - Adjust 10000 to 5000 if you want it even tighter
df_zoom = df_core[df_core['Population'] < 10000]

plt.figure(figsize=(10, 6))

# Plotting the zoomed-in data
sns.scatterplot(
    data=df_zoom, 
    x='Population', 
    y=df_zoom['RDOF_Allocation'] / 1000, 
    alpha=0.3, 
    color='royalblue',
    edgecolor=None # Removes outlines for a cleaner 'cloud' look
)

# Adding the regression line - 'truncate=True' keeps the line within the data limits
sns.regplot(
    data=df_zoom, 
    x='Population', 
    y=df_zoom['RDOF_Allocation'] / 1000, 
    scatter=False, 
    color='red',
    truncate=True 
)

plt.title('Zoomed View: RDOF Funding vs. Low-Density Populations (< 10k)')
plt.xlabel('Total Population')
plt.ylabel('RDOF Allocation (in Thousands of $)')
plt.grid(True, linestyle='--', alpha=0.6) # Adding a light grid for easier reading

plt.show()



# %% Step 8.c: Median Income Segment Analysis

# 1. Create Income Buckets
# Using standard brackets to define the economic landscape
income_bins = [0, 35000, 50000, 75000, 100000, 150000, df_core['median_income'].max()]
income_labels = ['Low (<35k)', 'Low-Mid (35k-50k)', 'Middle (50k-75k)', 
                 'Upper-Mid (75k-100k)', 'High (100k-150k)', 'Affluent (>150k)']

df_core['Income_Segment'] = pd.cut(df_core['median_income'], bins=income_bins, labels=income_labels)

# 2. Plotting Average Funding by Income Segment
plt.figure(figsize=(12, 6))
sns.barplot(
    data=df_core, 
    x='Income_Segment', 
    y=df_core['RDOF_Allocation'] / 1000, 
    palette='viridis',
    estimator='mean', # Crucial: Shows the average funding for that income bracket
    errorbar=None    
)

plt.title('Average RDOF Allocation by Median Household Income Tier')
plt.xlabel('Median Income Segment')
plt.ylabel('Average Allocation (in Thousands of $)')
plt.xticks(rotation=15)
plt.grid(axis='y', linestyle='--', alpha=0.7)

plt.show()



# %% Step 8.d:  Efficiency Heatmap (Large Professional Fonts)

# 1. CLIP AND RE-NORMALIZE
was_99 = df_core['WAS'].quantile(0.99)
df_core['WAS_Clipped'] = df_core['WAS'].clip(upper=was_99)

df_core['WAS_Final'] = (
    (df_core['WAS_Clipped'] - df_core['WAS_Clipped'].min()) / 
    (df_core['WAS_Clipped'].max() - df_core['WAS_Clipped'].min()) * 100
).round(2)

# 2. DEFINE BINS
was_bins = [0, 10, 20, 40, 70, 100]
was_labels = ['0-10', '10-20', '20-40', '40-70', '70-100']

rdof_k = df_core['RDOF_Allocation'] / 1000
rdof_bins = [0, 50, 100, 250, 500, rdof_k.max()]
rdof_labels = ['$0-50k', '$50k-100k', '$100k-250k', '$250k-500k', '$500k+']

# 3. APPLY BINS
df_core['WAS_Bin'] = pd.cut(df_core['WAS_Final'], bins=was_bins, labels=was_labels, include_lowest=True)
df_core['RDOF_Bin'] = pd.cut(rdof_k, bins=rdof_bins, labels=rdof_labels, include_lowest=True)

# 4. CREATE THE MATRIX
matrix_data = pd.crosstab(df_core['RDOF_Bin'], df_core['WAS_Bin']).iloc[::-1]

# 5. PLOT WITH LARGE FONTS ACROSS ALL ELEMENTS
plt.figure(figsize=(16, 14))

# Create the heatmap
ax = sns.heatmap(
    matrix_data, 
    annot=True, 
    fmt="d", 
    cmap="Blues", 
    square=True,
    annot_kws={"size": 18, "weight": "bold"}, # Big, bold numbers inside blocks
    cbar_kws={'shrink': 0.8}
)

# 6. ADJUST AXIS AND LEGEND FONT SIZES
# Title
plt.title('Intervention Matrix: WAS (0-100) vs. RDOF Funding', fontsize=22, pad=30, weight='bold')

# Axis Labels
plt.xlabel('WAS Score (Normalized)', fontsize=20, labelpad=15)
plt.ylabel('RDOF Allocation ($k)', fontsize=20, labelpad=15)

# Tick Labels (The numbers on the axes)
plt.xticks(fontsize=16)
plt.yticks(fontsize=16)

# Legend (Colorbar) Label size
cbar = ax.collections[0].colorbar
cbar.set_label('Count of Block Groups', fontsize=18, labelpad=20)
cbar.ax.tick_params(labelsize=14) # Colorbar scale numbers

plt.tight_layout()
plt.show()

# %% Step 9:  Calculate FEI and also Descriptive Statistics for FEI

# Calculate FEI safely: Points of WAS per $1,000 of RDOF
# We replace 0 funding with NaN to avoid 'inf' values
df_core['FEI'] = df_core['WAS'] / (df_core['RDOF_Allocation'] / 1000).replace(0, np.nan)

print("--- Funding Efficiency Index (FEI) Statistics ---")
# Filter out the NaNs (areas with no funding) for the stats report
fei_stats = df_core['FEI'].dropna()

print(fei_stats.describe(percentiles=[.05, .25, .5, .75, .9, .95, .99]))

# Check for extreme outliers
print(f"\nTotal Block Groups with Funding: {len(fei_stats):,}")
print(f"Absolute Max FEI: {fei_stats.max():.4f}")
print(f"Absolute Min FEI: {fei_stats.min():.4f}")

# %% Step 10. Machine Learning: Random Forest (Corrected Feature Name)

# 1. Update Features List with correct column name: 'Population'
features = [
    'Population',        
    'Tier1_Competitors', 
    'Tier2_Competitors', 
    'Tier3_Competitors', 
    'Max_Downspeed', 
    'median_income', 
    'MINORITY_PCT'
]

# 2. Pre-processing: Ensure all features are numeric
for col in features:
    df_core[col] = pd.to_numeric(df_core[col], errors='coerce')

# 3. Clean Missing Data
ml_df = df_core.dropna(subset=features + ['FEI']).copy()
print(f"Data cleaned. {len(ml_df):,} block groups ready for modeling.\n")

# 4. Train/Test Split
X = ml_df[features]
y = ml_df['FEI']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 5. Train the Random Forest Regressor
print("Training Random Forest Regressor (National Level)...")
rf_model = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
rf_model.fit(X_train, y_train)

# 6. Evaluate
y_pred = rf_model.predict(X_test)
r2 = r2_score(y_test, y_pred)

print(f" Model Training Complete!")
print(f" R-squared (R2) Score: {r2:.4f}\n")

# 7. Extract Feature Importances
print("Feature Importances (What dynamically drives funding efficiency?):")
importances = pd.Series(rf_model.feature_importances_, index=features).sort_values(ascending=False)

for feature, importance in importances.items():
    print(f"   -> {feature}: {importance * 100:.2f}%")
    
    
# %% Step 11: Unsupervised Learning - K-Means Clustering

print("Running Unsupervised Clustering (K-Means)...")

# 1. Prepare Data for Clustering
# We focus on the relationship between Market (WAS) and Funding (RDOF)
cluster_features = ['WAS', 'RDOF_Allocation']
cluster_df = df_core.dropna(subset=cluster_features).copy()

# 2. Scale the Data 
# K-Means is distance-based, so scaling is MANDATORY
scaler = StandardScaler()
scaled_features = scaler.fit_transform(cluster_df[cluster_features])

# 3. Run K-Means
# We'll look for 3 distinct groups (Personas)
kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
cluster_df['Cluster_ID'] = kmeans.fit_predict(scaled_features)

# 4. Analyze the Clusters
cluster_summary = cluster_df.groupby('Cluster_ID')[cluster_features].mean().sort_values(by='WAS')
print("\n--- Cluster Personas (Mean Values) ---")
print(cluster_summary)

# 5. Visualize the "Personas"
plt.figure(figsize=(12, 8))
sns.scatterplot(
    data=cluster_df, 
    x='WAS', 
    y='RDOF_Allocation', 
    hue='Cluster_ID', 
    palette='viridis',
    alpha=0.6
)
plt.title('Unsupervised Personas: WAS vs. RDOF Allocation', fontsize=16)
plt.show()

# %% Step12: Spatial Merge

# 1. Define the national shapefile path
shp_path = base_folder / 'Data/ShapeFiles/national_blockgroup.shp'

print(f" BASELINE: df_core has {len(df_core):,} funded records ready for spatial join.\n")

# --- 1. LOAD SHAPEFILE ---
print("Loading National Shapefile...")
gdf = gpd.read_file(shp_path)
print(f"   -> Loaded {len(gdf):,} spatial polygons.")

# --- 2. THE HIGH-EFFICIENCY SPATIAL MERGE ---
print("Merging spatial geometries with core funded areas...")
# Swapped bg_summary for df_core! 
map_data = gdf.merge(df_core, left_on='GEOID', right_on='bg_geoid', how='inner')

# Drop the redundant ID column to save memory
map_data = map_data.drop(columns=['bg_geoid'])

# CRITICAL MEMORY PURGE: Destroy the raw shapefile immediately
del gdf
gc.collect()

# Checkpoint 1 Validation
print(f" Post-Merge: map_data has {len(map_data):,} integrated spatial records.\n")

# --- 3. REPROJECTION ---
print("Converting CRS to EPSG:4326 (WGS84)...")
# This will now run exponentially faster because it's only processing 15k polygons
map_data = map_data.to_crs(epsg=4326)


print("🧹 Pipeline Complete.")
print(f" Final Map Data is ready! Geometry count: {len(map_data):,}")

# %% Step 13: National Policy Classification (Final Optimized Tiers)

# 1. Scaling the variables
df_core['WAS_scaled'] = (df_core['WAS'] - df_core['WAS'].min()) / (df_core['WAS'].max() - df_core['WAS'].min())
df_core['FEI_scaled'] = (df_core['FEI'] - df_core['FEI'].min()) / (df_core['FEI'].max() - df_core['FEI'].min())
df_core['RDOF_scaled'] = (df_core['RDOF_Allocation'] - df_core['RDOF_Allocation'].min()) / (df_core['RDOF_Allocation'].max() - df_core['RDOF_Allocation'].min())

# 2. Score Calculation (Lower = More Urgent)
df_core['Policy_Score'] = (df_core['FEI_scaled'] * 0.4) + (df_core['WAS_scaled'] * 0.4) - (df_core['RDOF_scaled'] * 0.2)

# 3. Dynamic Thresholds for 5%, 10% (cumulative 15%), and 70% (cumulative 85%)
q02, q15, q85 = df_core['Policy_Score'].quantile([0.02, 0.15, 0.85])

def assign_policy(score):
    if score <= q02: return "P1: Urgent Audit (Bottom 2%)"
    if score <= q15: return "P2: Performance Review (2-15%)"
    if score <= q85: return "P3: Sustainable Growth (15-85%)"
    return "P4: Strategic Leader (Top 15%)"

df_core['Policy_Recommendation'] = df_core['Policy_Score'].apply(assign_policy)

# 4. Final Display: Count and Percentage
summary = df_core['Policy_Recommendation'].value_counts().sort_index().to_frame(name='CBG_Count')
summary['Percentage'] = (summary['CBG_Count'] / len(df_core) * 100).round(1).astype(str) + '%'

print("\n--- National Policy Classification Summary ---")
print(summary)
# %%  STEP 14: TABLEAU SPATIAL EXPORT (WITH PREVIEW)

# Dictionary for State FIPS to Name mapping
fips_to_name = {
    '01': 'Alabama', '02': 'Alaska', '04': 'Arizona', '05': 'Arkansas', '06': 'California',
    '08': 'Colorado', '09': 'Connecticut', '10': 'Delaware', '11': 'District of Columbia',
    '12': 'Florida', '13': 'Georgia', '15': 'Hawaii', '16': 'Idaho', '17': 'Illinois',
    '18': 'Indiana', '19': 'Iowa', '20': 'Kansas', '21': 'Kentucky', '22': 'Louisiana',
    '23': 'Maine', '24': 'Maryland', '25': 'Massachusetts', '26': 'Michigan', '27': 'Minnesota',
    '28': 'Mississippi', '29': 'Missouri', '30': 'Montana', '31': 'Nebraska', '32': 'Nevada',
    '33': 'New Hampshire', '34': 'New Jersey', '35': 'New Mexico', '36': 'New York',
    '37': 'North Carolina', '38': 'North Dakota', '39': 'Ohio', '40': 'Oklahoma', '41': 'Oregon',
    '42': 'Pennsylvania', '44': 'Rhode Island', '45': 'South Carolina', '46': 'South Dakota',
    '47': 'Tennessee', '48': 'Texas', '49': 'Utah', '50': 'Vermont', '51': 'Virginia',
    '53': 'Washington', '54': 'West Virginia', '55': 'Wisconsin', '56': 'Wyoming'
}

print(f"Columns: {map_data.columns.tolist()}")
print("\nPreparing Spatial Data for Tableau Cloud...")

# 1. ADD STATE NAMES TO df_core BEFORE MERGING
# Ensure GEOID is string, take first 2 digits, and map
df_core['State_FIPS'] = df_core['bg_geoid'].astype(str).str.zfill(12).str[:2]
df_core['State_Name'] = df_core['State_FIPS'].map(fips_to_name)

# 2. THE FIX FOR _x / _y: 
cols_to_clear = ['Policy_Recommendation', 'FEI', 'State_Name']
map_data = map_data.drop(columns=[c for c in cols_to_clear if c in map_data.columns])

# 3. PREPARE EXPORT COLUMNS (Including State_Name now)
export_columns = ['bg_geoid', 'Policy_Recommendation', 'FEI', 'State_Name']

# 4. MERGE THE DATA
map_data = map_data.merge(
    df_core[export_columns], 
    left_on='GEOID', 
    right_on='bg_geoid', 
    how='left'
)

# Clean up redundant join key
if 'bg_geoid' in map_data.columns:
    map_data = map_data.drop(columns=['bg_geoid'])

# --- VERIFICATION STEP ---
print("\n--- Map Data Final Column Check ---")
print(map_data.info())
print("\n--- State Distribution Check ---")
print(map_data['State_Name'].value_counts().head(10)) # Verify states are mapped correctly
# -------------------------

# GEOMETRY SIMPLIFICATION
print("\nSimplifying geometries...")
map_data['geometry'] = map_data['geometry'].simplify(0.0005, preserve_topology=True)

# Export A: GeoJSON
print(f"Exporting GeoJSON to {base_folder}...")
geojson_path = base_folder / 'national_broadband_final_map.geojson'
map_data.to_file(geojson_path, driver='GeoJSON')

# Export B: Spatial CSV (WKT)
print(f"Exporting Spatial CSV to {base_folder}...")
wkt_path = base_folder / 'national_broadband_spatial_tableau.csv'
map_data_csv = map_data.copy()
map_data_csv['WKT'] = map_data_csv['geometry'].apply(lambda x: x.wkt)

pd.DataFrame(map_data_csv.drop(columns='geometry')).to_csv(wkt_path, index=False)

print(f"\n Pipeline Complete! Files saved to: {base_folder}")

# Export C: Diagnostic Review CSV (Flat file for QA)
print(f"Exporting Diagnostic Review CSV to {base_folder}...")
review_csv_path = base_folder / 'national_policy_diagnostic_review.csv'

# Selecting key columns to make the review easy to read
review_cols = [
    'bg_geoid', 'State_Name', 'FEI', 'WAS', 'RDOF_Allocation', 
    'Policy_Score', 'Policy_Recommendation'
]

# Save to CSV
df_core[review_cols].to_csv(review_csv_path, index=False)

# Final Count Verification for the Narrative
print("\n--- CBG Count per Policy Tier ---")
print(df_core['Policy_Recommendation'].value_counts().sort_index())

print(f"\nReview file saved to: {review_csv_path}")