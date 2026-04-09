import pandas as pd
import numpy as np
import pickle
from sklearn.preprocessing import StandardScaler, LabelEncoder
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout

print("⏳ Files generate ho rahi hain...")

# 1. Data load karo
df = pd.read_csv('cleaned_data.csv')
X = df.drop('stress_label', axis=1)
y = df['stress_label']

# 2. Scaler aur Encoder fit karo
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

encoder = LabelEncoder()
y_encoded = encoder.fit_transform(y)
classes = encoder.classes_

# 3. Chota sa model training (Fatafat files banane ke liye)
model = Sequential([
    Dense(64, activation='relu', input_shape=(X_scaled.shape[1],)),
    Dense(32, activation='relu'),
    Dense(len(classes), activation='softmax')
])
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
model.fit(X_scaled, y_encoded, epochs=5, verbose=0) # Sirf 5 rounds

# 4. 🔥 SAB KUCH SAVE KARO
model.save('tabular_model.h5')
with open('scaler.pkl', 'wb') as f:
    pickle.dump(scaler, f)
with open('encoder.pkl', 'wb') as f:
    pickle.dump(encoder, f)

print("\n✅ MUBARAK HO! Ye files ban gayi hain:")
print("- tabular_model.h5")
print("- scaler.pkl")
print("- encoder.pkl")