import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pickle

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout

print("⏳ Step 1: Data Load ho raha hai...")
df = pd.read_csv('cleaned_data.csv')

# Features (X) aur Target (y) ko alag karna
X = df.drop('stress_label', axis=1)
y = df['stress_label']

print("\n⏳ Step 2: Data Preprocessing shuru...")
# Labels (healthy, water_stress) ko numbers (0, 1, 2) mein badalna
encoder = LabelEncoder()
y_encoded = encoder.fit_transform(y)
classes = encoder.classes_
print(f"✅ AI ne in bimaariyon ko pehchana: {classes}")

# Data ko train aur test mein baatna (80% seekhne ke liye, 20% test ke liye)
X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.2, random_state=42)

# Scaling (Sabse important! Sensors ke numbers alag-alag range mein hote hain, inko ek scale pe lana)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

print("\n⏳ Step 3: Tabular Neural Network ban raha hai...")
# Ye SNN (Spiking Neural Network) ki taraf pehla kadam hai
model = Sequential([
    Dense(128, activation='relu', input_shape=(X_train_scaled.shape[1],)),
    Dropout(0.3),  # 30% neurons ko band karna taaki model ratta (overfit) na maare
    Dense(64, activation='relu'),
    Dropout(0.2),
    Dense(32, activation='relu'),
    Dense(len(classes), activation='softmax') # Final output layer
])

model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
print("✅ Model Ready! Training shuru karte hain...\n")

# Training the model
history = model.fit(
    X_train_scaled, y_train, 
    epochs=50, 
    batch_size=32, 
    validation_data=(X_test_scaled, y_test),
    verbose=1
)

print("\n⏳ Step 4: Model ka Test aur Result...")
loss, accuracy = model.evaluate(X_test_scaled, y_test)
print(f"\n🎉 TABULAR MODEL FINAL ACCURACY: {accuracy * 100:.2f}%")

# ---------------------------------------------------------
# Step 5: Graphs Banana (Project Report ke liye bohot zaroori)
# ---------------------------------------------------------
plt.figure(figsize=(12, 5))

# Accuracy ka graph
plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'], label='Train Accuracy')
plt.plot(history.history['val_accuracy'], label='Test Accuracy')
plt.title('Model Accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch (Rounds)')
plt.legend()

# Loss (Galtiyon) ka graph
plt.subplot(1, 2, 2)
plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Test Loss')
plt.title('Model Loss (Galtiyan)')
plt.ylabel('Loss')
plt.xlabel('Epoch (Rounds)')
plt.legend()

plt.show()

# ---------------------------------------------------------
# Step 6: Phase 3 ke liye sab kuch Save karna!
# ---------------------------------------------------------
model.save('tabular_model.h5')
with open('scaler.pkl', 'wb') as f:
    pickle.dump(scaler, f)
with open('encoder.pkl', 'wb') as f:
    pickle.dump(encoder, f)

print("\n💾 SUCCESS! Tera Model ('tabular_model.h5'), Scaler, aur Encoder save ho gaye hain.")
print("In teeno files ko Colab se download kar le, ye aage Hybrid Model mein kaam aayengi!")
import pickle
# Training ke baad ye add karo:
model.save('tabular_model.h5')
with open('scaler.pkl', 'wb') as f:
    pickle.dump(scaler, f)
with open('encoder.pkl', 'wb') as f:
    pickle.dump(encoder, f)
print("✅ Scaler aur Encoder save ho gaye!")