import pandas as pd
import numpy as np

# -------------------------
# LOAD DATA
# -------------------------
file_path = "data/environmental/orange dataset.csv"
df = pd.read_csv(file_path)

print("✅ Data Loaded Successfully")
print("Shape:", df.shape)

# -------------------------
# FIX COLUMN NAMES
# -------------------------
df.columns = df.columns.str.strip()

# -------------------------
# NEW SMART LABEL CREATION (Based on Complete Data Analysis)
# -------------------------
def label_stress(row):
    # 1. Water Stress (Bottom 15% moisture data)
    if row['soil_moisture'] < 0.12:  
        return "water_stress"
    
    # 2. Heat Stress (Nagpur Extreme Heat > 37C)
    elif row['air_temperature_C'] > 37:  
        return "heat_stress"
    
    # 3. Vegetation Stress (NDVI below 0.5)
    elif row['NDVI'] < 0.5:  
        return "vegetation_stress"
    
    # 4. Nutrient Stress (Using Chlorophyll < 10 because Nitrogen is constant)
    elif row['chlorophyll_content'] < 10:  
        return "nutrient_stress"
    
    # 5. Healthy
    else:
        return "healthy"

# Apply logic
df['stress_label'] = df.apply(label_stress, axis=1)

# -------------------------
# CLEAN DATA
# -------------------------
df.fillna(df.mean(numeric_only=True), inplace=True)

# -------------------------
# SAVE FILE
# -------------------------
output_path = "data/environmental/processed_data.csv"
df.to_csv(output_path, index=False)

print("\n✅ FILE SAVED SUCCESSFULLY")
print("Saved at:", output_path)

print("\n🔥 FINAL Class Distribution:")
print(df['stress_label'].value_counts())