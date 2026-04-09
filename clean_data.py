import pandas as pd

print("⏳ Data cleaning shuru ho rahi hai...")

try:
    df = pd.read_csv('data\environmental/processed_data.csv')
    
    # 1. Sabse pehle column names ke aage-peeche ke hidden spaces (kachra) hatao
    df.columns = df.columns.str.strip()
    
    print(f"✅ File load ho gayi! Pehle isme {len(df.columns)} columns the.")
    
    # 2. Jo hatana hai uski list
    columns_to_drop = [
        'date', 'orchard_id', 'location', 'district', 
        'region', 'dataset', 'plant_health_status', 'yield_ton_per_hectare'
    ]

    # 3. Check karo ki list wale columns sach mein file mein hain ya nahi
    columns_found = [col for col in columns_to_drop if col in df.columns]
    columns_missing = [col for col in columns_to_drop if col not in df.columns]
    
    if columns_missing:
        print(f"⚠️ DHYAN DEIN: Ye columns file mein nahi mile (ya spelling alag hai): {columns_missing}")

    # 4. Hata do
    df_cleaned = df.drop(columns=columns_found)

    # 5. Save kar do
    df_cleaned.to_csv('cleaned_data.csv', index=False)
    
    print(f"🗑️ Successfully hataye gaye columns: {columns_found}")
    print(f"🎉 Cleaning Done! Ab exactly {len(df_cleaned.columns)} columns bache hain.")

except FileNotFoundError:
    print("❌ Error: 'processed_data.csv' file nahi mili!")
except Exception as e:
    print(f"❌ Ek naya error aaya hai: {e}")