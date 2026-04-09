import torch
import torch.nn as nn
import snntorch as snn
import pandas as pd
import numpy as np
import pickle
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, TensorDataset

print("⏳ Step 1: Loading Sensor Data & Setting Device...")

# 1. Device set karna (GPU hai toh CUDA use hoga, warna CPU)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"🔥 Training on: {device}")

# 2. Data Load & Preprocessing
df = pd.read_csv('cleaned_data.csv')

X = df.drop('stress_label', axis=1).values
y_raw = df['stress_label'].values

encoder = LabelEncoder()
y = encoder.fit_transform(y_raw)

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Scaler aur Encoder save karna
with open('snn_scaler.pkl', 'wb') as f:
    pickle.dump(scaler, f)
with open('snn_encoder.pkl', 'wb') as f:
    pickle.dump(encoder, f)

# Train aur Test mein split karna
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# Data ko PyTorch Tensors mein badalna
X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
y_train_tensor = torch.tensor(y_train, dtype=torch.long)
X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
y_test_tensor = torch.tensor(y_test, dtype=torch.long)

# DataLoader banana batches ke liye
train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)

# 3. SNN Architecture
num_steps = 25  
beta = 0.9      

class SpikingSensorNet(nn.Module):
    def __init__(self, num_inputs, num_classes):
        super().__init__()
        self.fc1 = nn.Linear(num_inputs, 64)
        self.lif1 = snn.Leaky(beta=beta)
        self.fc2 = nn.Linear(64, num_classes)
        self.lif2 = snn.Leaky(beta=beta)

    def forward(self, x):
        mem1 = self.lif1.init_leaky()
        mem2 = self.lif2.init_leaky()
        spk2_rec = [] 

        for step in range(num_steps):
            cur1 = self.fc1(x)
            spk1, mem1 = self.lif1(cur1, mem1)
            cur2 = self.fc2(spk1)
            spk2, mem2 = self.lif2(cur2, mem2)
            spk2_rec.append(spk2)

        return torch.stack(spk2_rec, dim=0)

num_features = X.shape[1] 
num_classes = len(encoder.classes_)

# Model ko GPU/CPU pe bhejna
model = SpikingSensorNet(num_features, num_classes).to(device)

# Loss & Optimizer
loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

# 4. Training Loop
print("🚀 Step 2: Training SNN (Generating Spikes)...")
epochs = 100

for epoch in range(epochs):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    
    # Batches mein train karna
    for batch_X, batch_y in train_loader:
        # Data ko GPU/CPU pe bhejna (IMPORTANT)
        batch_X, batch_y = batch_X.to(device), batch_y.to(device)
        
        optimizer.zero_grad()
        
        # Forward pass
        spk_rec = model(batch_X) 
        
        # Total spikes aur Loss
        total_spikes = spk_rec.sum(dim=0) 
        loss = loss_fn(total_spikes, batch_y)

        # Backward pass & weights update
        loss.backward()
        optimizer.step()
        
        # Metrics calculate karna
        running_loss += loss.item()
        _, predicted = total_spikes.max(1)
        total += batch_y.size(0)
        correct += predicted.eq(batch_y).sum().item()

    if (epoch+1) % 10 == 0:
        avg_loss = running_loss / len(train_loader)
        accuracy = 100. * correct / total
        print(f"Epoch {epoch+1}/{epochs} | Loss: {avg_loss:.4f} | Training Accuracy: {accuracy:.2f}%")

# 5. Model Save Karna
# Save karne se pehle model ko wapas CPU pe lana safe rehta hai
torch.save(model.cpu().state_dict(), 'snn_model.pth')
print("\n✅ Success! SNN Model Saved as 'snn_model.pth'")
print(f"🎯 Recognized Classes: {encoder.classes_}")