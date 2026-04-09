import splitfolders
import os

# ---------------------------------------------------------
# 1. TERA EXACT PATH
# ---------------------------------------------------------
# Tera original train folder jisme sab images hain
input_folder = r"C:\Users\Ankush\early_stress_detection\data\images\train" 

# Naya folder jahan split hoke images (train, val, test) aayengi
output_folder = r"C:\Users\Ankush\early_stress_detection\data\images_split"

# ---------------------------------------------------------
# 2. PATH CHECK (Taaki koi error na aaye)
# ---------------------------------------------------------
if not os.path.exists(input_folder):
    print(f"❌ ERROR: Mujhe ye folder nahi mila -> {input_folder}")
    print("👉 Check kar ki kya folder sach mein is jagah par hai!")
    exit()

# ---------------------------------------------------------
# 3. SPLIT DATA (70% Train, 15% Val, 15% Test)
# ---------------------------------------------------------
print(f"⏳ Data check ho gaya hai. Splitting start ho rahi hai...")

splitfolders.ratio(
    input_folder, 
    output=output_folder, 
    seed=42, 
    ratio=(0.7, 0.15, 0.15), 
    group_prefix=None
)

print(f"\n✅ BOOM! Data successfully split ho gaya hai!")
print(f"📂 Jaa kar check kar apna naya folder: {output_folder}")