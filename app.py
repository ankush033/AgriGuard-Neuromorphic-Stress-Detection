import os
from dotenv import load_dotenv
import requests
import streamlit as st
import numpy as np
import tensorflow as tf
import torch
import torch.nn as nn
import snntorch as snn
from PIL import Image
import pickle

# ==========================================
# 🔥 ENVIRONMENT VARIABLES (Professional Way)
# ==========================================
load_dotenv(override=True)
API_KEY = os.getenv("GEMINI_API_KEY")

# TensorFlow ki faltoo warnings chhupane ke liye
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# ==========================================
# 1. DYNAMIC RECOMMENDATION ENGINE (Severity Based)
# ==========================================
def get_detailed_root_cause(final_label, cnn_label, temp, soil, aqi):
    cnn_clean = str(cnn_label).lower().strip()
    
    # --- Part A: What the Camera (CNN) saw ---
    if 'healthy' in cnn_clean:
        visual_msg = "📸 **Visual Analysis (CNN):** The uploaded leaf image appears **perfectly healthy** with no physical damage or spots."
    elif 'nutrient' in cnn_clean:
        visual_msg = "📸 **Visual Analysis (CNN):** The AI detected leaf discoloration indicating **Nutrient Stress**."
    else:
        visual_msg = f"📸 **Visual Analysis (CNN):** The leaf shows physical symptoms of **{cnn_clean.replace('_', ' ').title()}**."

    # --- Part B: What the Sensors (SNN) felt & VETO ENGINE ---
    sensor_msg = ""
    
    if final_label == 'heat_stress':
        if 'healthy' in cnn_clean:
            sensor_msg = f"\n\n⚠️ **EXPERT OVERRIDE (SNN):** Ignored visual data! Lethal ambient temperature detected (**{temp}°C**). The plant is burning internally before showing physical symptoms."
        else:
            sensor_msg = f"\n\n🌡️ **Environmental Alert:** Extreme heat (**{temp}°C**) is actively accelerating the damage."
            
    elif final_label == 'water_stress':
        if 'healthy' in cnn_clean:
            sensor_msg = f"\n\n⚠️ **EXPERT OVERRIDE (SNN):** Ignored visual data! Soil moisture is critically low (**{soil}%**). The plant is suffering severe underground dehydration."
        else:
            sensor_msg = f"\n\n💧 **Environmental Alert:** Critically low soil moisture (**{soil}%**) is starving the roots."
            
    elif final_label == 'pollution_stress':
        if 'healthy' in cnn_clean:
            sensor_msg = f"\n\n⚠️ **EXPERT OVERRIDE (SNN):** Ignored visual data! Hazardous AQI (**{aqi}**) is suffocating the plant's pores."
        else:
            sensor_msg = f"\n\n🏭 **Environmental Alert:** Hazardous pollution (AQI: **{aqi}**) is severely affecting gas exchange."
            
    elif final_label == 'healthy':
        sensor_msg = "\n\n✅ **Environmental Alert:** All sensor metrics are optimal. The plant environment is completely safe."
    else:
        sensor_msg = f"\n\n⚠️ **Environmental Alert:** Sensors confirm the stress condition."

    return visual_msg + sensor_msg

def get_dynamic_advice(label, temp, hum, soil, light, aqi, ozone):
    advice = {"name": "", "reason": "", "action": ""}
    
    if label == 'water_stress':
        advice["name"] = "Water Stress"
        if soil <= 15:
            advice["action"] = "🚨 IMMEDIATE EMERGENCY RESCUE: \n1. Do NOT flood the soil immediately (it causes root shock). Apply water slowly through drip irrigation over 4 hours.\n2. Apply thick organic mulch immediately to stop any further evaporation.\n3. Consider adding hydrogels to the soil base for future water retention."
        else:
            advice["action"] = "💧 PREVENTATIVE ACTION: \n1. Start your drip irrigation cycle now.\n2. Adjust irrigation sensors to trigger automatically when moisture hits 30%.\n3. Check for any leakages in your water supply lines."
            
    elif label == 'heat_stress':
        advice["name"] = "Heat Stress"
        if temp >= 42.0:
            advice["action"] = "🚨 EMERGENCY PROTOCOL: \n1. Deploy 75% UV shade nets IMMEDIATELY over the crop.\n2. Start heavy overhead misting to artificially bring ambient temperature down.\n3. Strictly stop any fertilizer application until temperature drops below 35°C."
        else:
            advice["action"] = "🌤️ ACTION REQUIRED: \n1. Shift all watering schedules strictly to early morning or late evening.\n2. Provide partial afternoon shading if possible.\n3. Ensure adequate spacing between plants for wind circulation."
            
    elif label == 'pollution_stress':
        advice["name"] = "Air/Soil Pollution Stress"
        if aqi >= 300 or ozone >= 80:
            advice["action"] = "🏭 EMERGENCY CLEANING: \n1. Perform a gentle but thorough overhead washing of the entire plant canopy.\n2. Erect physical agro-shade barriers facing the wind/industrial side.\n3. Use activated biochar in the soil to neutralize heavy metals settling from the air."
        else:
            advice["action"] = "🧹 PREVENTATIVE: \n1. Lightly spray the leaves with water to clear the dust.\n2. Plant tall perimeter trees around the farm to act as natural dust filters."
            
    elif label == 'nutrient_stress':
        advice["name"] = "Nutrient Stress"
        advice["action"] = "🧪 ACTION REQUIRED: \n1. Apply a fast-acting water-soluble foliar spray directly to the leaves for 24-hour absorption.\n2. Conduct a lab soil test to check the exact pH levels.\n3. Add vermicompost to the base to improve organic soil health."
        
    elif label == 'vegetation_stress':
        advice["name"] = "Vegetation Stress (Pathogen/Pest)"
        advice["action"] = "🌿 CONTAINMENT PROTOCOL: \n1. Immediately prune and burn/dispose of the infected leaves away from the farm.\n2. Apply a broad-spectrum organic biopesticide (e.g., Neem oil extract).\n3. Stop overhead watering, as wet leaves encourage fungal growth."
        
    elif label == 'healthy':
        advice["name"] = "Healthy"
        if temp > 35.0 or soil < 30.0:
            advice["action"] = "👀 PROACTIVE CARE: \n1. No immediate rescue needed, but keep a close eye on the temperature.\n2. Maintain your current routine, the plant is showing good resilience."
        else:
            advice["action"] = "🌟 KEEP IT UP: \n1. Continue your current exact watering and nutrient schedules.\n2. Log these environmental parameters as the 'Gold Standard' for this crop cycle."
            
    return advice

# ==========================================
# 2. SNN ARCHITECTURE 
# ==========================================
class SpikingSensorNet(nn.Module):
    def __init__(self, num_inputs, num_classes):
        super().__init__()
        self.fc1 = nn.Linear(num_inputs, 64)
        self.lif1 = snn.Leaky(beta=0.9)
        self.fc2 = nn.Linear(64, num_classes)
        self.lif2 = snn.Leaky(beta=0.9)

    def forward(self, x):
        mem1 = self.lif1.init_leaky()
        mem2 = self.lif2.init_leaky()
        spk2_rec = []
        for step in range(25): 
            cur1 = self.fc1(x)
            spk1, mem1 = self.lif1(cur1, mem1)
            cur2 = self.fc2(spk1)
            spk2, mem2 = self.lif2(cur2, mem2)
            spk2_rec.append(spk2)
        return torch.stack(spk2_rec, dim=0)

# ==========================================
# 3. LOAD ALL MODELS
# ==========================================
@st.cache_resource
def load_all_models():
    cnn_model = tf.keras.models.load_model('leaf_stress_model_final.h5')
    
    with open('snn_scaler.pkl', 'rb') as f:
        scaler = pickle.load(f)
    with open('snn_encoder.pkl', 'rb') as f:
        encoder = pickle.load(f)
        
    num_features = scaler.mean_.shape[0]
    num_classes = len(encoder.classes_)
    snn_model = SpikingSensorNet(num_features, num_classes)
    snn_model.load_state_dict(torch.load('snn_model.pth'))
    snn_model.eval() 
    
    return cnn_model, snn_model, scaler, encoder

cnn_model, snn_model, scaler, encoder = load_all_models()

# ==========================================
# 4. STREAMLIT UI DESIGN
# ==========================================
st.set_page_config(page_title="AgriGuard AI", layout="wide")
st.title("🌱 AgriGuard: Early Stress Detection (CNN + SNN)")
st.markdown("**A Hybrid Artificial Intelligence & Dynamic Expert System for Smart Agriculture**")

col1, col2 = st.columns(2)

with col1:
    st.header("📸 1. Visual Input (CNN)")
    uploaded_file = st.file_uploader("Upload Leaf Image", type=["jpg", "png", "jpeg"])
    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        st.image(image, caption='Uploaded Leaf Sample', use_container_width=True)

with col2:
    st.header("🌡️ 2. Sensor Input (SNN & Logic)")
    temperature = st.slider("Temperature (°C)", 10.0, 50.0, 25.0)
    humidity = st.slider("Humidity (%)", 10.0, 100.0, 60.0)
    soil_moisture = st.slider("Soil Moisture (%)", 0.0, 100.0, 40.0)
    light_intensity = st.slider("Light Intensity (Lux)", 0.0, 1000.0, 500.0)
    
    st.markdown("---")
    st.markdown("**🏭 Air Quality & Pollution Metrics**")
    aqi = st.slider("Air Quality Index (AQI)", 0, 500, 50)
    ozone = st.slider("Ozone Level (ppb)", 0, 150, 20)

# ==========================================
# 5. HYBRID DIAGNOSIS LOGIC
# ==========================================
st.markdown("---")
if st.button("🚀 RUN HYBRID DIAGNOSIS", use_container_width=True):
    if uploaded_file is None:
        st.error("⚠️ Please upload a leaf image first to begin the diagnosis!")
    else:
        with st.spinner("🧠 Analyzing SNN Temporal Spikes, CNN Spatial Features, and Computing Hybrid Probabilities..."):
            
            # --- A. CNN PREDICTION (Vision) ---
            cnn_classes_raw = ['Healthy', 'Nutrients_stress', 'Pollution_stress', 'heat_stress', 'water_stress']
            cnn_mapping = {
                'Healthy': 'healthy',
                'Nutrients_stress': 'nutrient_stress',
                'Pollution_stress': 'pollution_stress',
                'heat_stress': 'heat_stress',
                'water_stress': 'water_stress'
            }
            
            img = image.resize((224, 224)) 
            img_array = np.array(img) / 255.0
            if img_array.shape[-1] == 4: 
                img_array = img_array[..., :3]
            img_array = np.expand_dims(img_array, axis=0)
            
            cnn_preds_raw = cnn_model.predict(img_array)[0] 
            
            confidence_cap = 0.92 
            cnn_preds = cnn_preds_raw * confidence_cap + ((1.0 - confidence_cap) / len(cnn_preds_raw))
            
            cnn_prob_dict = {cnn_mapping[label]: prob for label, prob in zip(cnn_classes_raw, cnn_preds)}
            cnn_prob_dict['vegetation_stress'] = 0.0 
            top_cnn_label = max(cnn_prob_dict, key=cnn_prob_dict.get)
            
            # --- B. SNN PREDICTION (Sensors) ---
            num_features = scaler.mean_.shape[0]
            sensor_data = scaler.mean_.copy().reshape(1, -1)
            
            if num_features >= 15:
                sensor_data[0, 11] = temperature    
                sensor_data[0, 13] = humidity       
                sensor_data[0, 14] = soil_moisture  
            else:
                sensor_data[0, 0] = temperature
                sensor_data[0, 1] = humidity
                sensor_data[0, 2] = soil_moisture

            sensor_scaled = scaler.transform(sensor_data)
            sensor_tensor = torch.tensor(sensor_scaled, dtype=torch.float32)
            
            with torch.no_grad():
                spikes = snn_model(sensor_tensor)
                total_spikes = spikes.sum(dim=0).numpy()[0] 
            
            T = 8.5 
            shifted_spikes = (total_spikes - np.max(total_spikes)) / T
            snn_probs_raw = np.exp(shifted_spikes) / np.sum(np.exp(shifted_spikes))
            
            snn_prob_dict = {}
            for idx, snn_label in enumerate(encoder.classes_):
                label_str = str(snn_label).lower().strip()
                if "heat" in label_str: std_label = 'heat_stress'
                elif "water" in label_str: std_label = 'water_stress'
                elif "pollution" in label_str: std_label = 'pollution_stress'
                elif "nutrient" in label_str: std_label = 'nutrient_stress'
                else: std_label = 'healthy'
                snn_prob_dict[std_label] = snn_probs_raw[idx]
                
            top_snn_label = max(snn_prob_dict, key=snn_prob_dict.get)
                
            # --- C. HYBRID INTEGRATION ---
            final_probabilities = {}
            all_possible_classes = ['healthy', 'nutrient_stress', 'pollution_stress', 'heat_stress', 'water_stress', 'vegetation_stress']
            
            for label in all_possible_classes:
                c_prob = cnn_prob_dict.get(label, 0.0)
                s_prob = snn_prob_dict.get(label, 0.0)
                final_probabilities[label] = (0.4 * c_prob) + (0.6 * s_prob)
                
            final_label = max(final_probabilities, key=final_probabilities.get)
            
            # --- EXPERT VETO ENGINE ---
            veto_triggered = False
            if aqi >= 300 or ozone >= 80:
                final_label = 'pollution_stress'
                veto_triggered = True
            elif temperature >= 40.0 and humidity <= 30.0:
                final_label = 'heat_stress'
                veto_triggered = True
            elif soil_moisture <= 20.0:
                final_label = 'water_stress'
                veto_triggered = True

            # --- D. GET DYNAMIC ADVICE ---
            dynamic_result = get_dynamic_advice(final_label, temperature, humidity, soil_moisture, light_intensity, aqi, ozone)
            dynamic_result['reason'] = get_detailed_root_cause(final_label, top_cnn_label, temperature, soil_moisture, aqi)

            st.session_state.current_diagnosis = {
                'name': dynamic_result['name'],
                'reason': dynamic_result['reason'],
                'action': dynamic_result['action'],
                'cnn_top': top_cnn_label,
                'snn_top': top_snn_label,
                'veto_active': veto_triggered
            }

            # --- E. DISPLAY RESULTS ---
            st.markdown("---")
            st.success(f"### 🎯 Final AI Diagnosis: {dynamic_result['name']}")
            
            st.markdown(f"**AI Confidence Analytics:**")
            st.write(f"- 📸 CNN (Visual) Confidence: **{max(cnn_preds)*100:.1f}%**")
            st.write(f"- 🌡️ SNN (Sensor) Confidence: **{max(snn_probs_raw)*100:.1f}%**")
            
            st.subheader("📊 Root Cause Analysis (Dynamic)")
            st.info(dynamic_result['reason'])
            
            st.subheader("💊 Expert Multi-Step Recommendation")
            st.warning(dynamic_result['action'])

# ==========================================
# 6. AGRIBOT - VIRTUAL AGRONOMIST (Context-Aware LLM API)
# ==========================================
st.markdown("---")
st.header("🤖 AgriBot: AI Expert Agronomist")
st.markdown("Ask anything! I know about your current plant diagnosis.")

if "messages" not in st.session_state:
    st.session_state.messages = [
        {"role": "assistant", "content": "Hello Farmer! 🌱 I am AgriBot. Ask your question, or run the diagnosis above and ask me for a specific solution!"}
    ]

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if prompt := st.chat_input("Type your question here... (e.g., Should I apply fertilizer right now?)"):
    
    with st.chat_message("user"):
        st.markdown(prompt)
    st.session_state.messages.append({"role": "user", "content": prompt})

    with st.chat_message("assistant"):
        with st.spinner("AgriBot is analyzing the vision and sensor data... 🤔"):
            try:
                # API Key check from Environment Variable only
                if not API_KEY:
                    st.error("⚠️ SYSTEM ERROR: Expert System Offline. (Check Backend configuration)")
                    st.stop()

                diagnosis_context = ""
                if "current_diagnosis" in st.session_state:
                    diag = st.session_state.current_diagnosis
                    diagnosis_context = f"""
                    🚨 CRITICAL HYBRID CONTEXT: 
                    The overall system diagnosed the plant with: '{diag['name']}'.
                    - 📸 Visual AI (CNN) saw: {diag['cnn_top']}
                    - 🌡️ Sensor AI (SNN) felt: {diag['snn_top']}
                    - 🚨 Expert Veto Used: {diag['veto_active']}
                    - Reason: {diag['reason']}
                    """
                
                system_instruction = f"""You are AgriBot, a strict and expert agricultural AI. You are a HYBRID AI that uses both vision and sensors.
                {diagnosis_context}
                YOUR RULES:
                1. If the user asks about the image, explain to them that while the image (CNN) might look '{st.session_state.get('current_diagnosis', {}).get('cnn_top', 'unknown')}', the environmental sensors (SNN) detected a critical issue, which is why the system overrode the visual data to save the plant.
                2. If the user asks about applying fertilizer, and the condition is 'Heat Stress' or 'Water Stress', strictly tell them NO and explain that it will burn the roots due to the current condition.
                3. Keep answers short, smart, and in bullet points.
                
                User asks: """
                
                full_prompt = system_instruction + prompt
                
                models_url = f"https://generativelanguage.googleapis.com/v1beta/models?key={API_KEY}"
                models_req = requests.get(models_url)
                
                working_model = "models/gemini-1.5-flash"
                if models_req.status_code == 200:
                    models_data = models_req.json().get("models", [])
                    for m in models_data:
                        if 'generateContent' in m.get('supportedGenerationMethods', []):
                            working_model = m['name'] 
                            break
                
                url = f"https://generativelanguage.googleapis.com/v1beta/{working_model}:generateContent?key={API_KEY}"
                headers = {'Content-Type': 'application/json'}
                data = {
                    "contents": [{"parts": [{"text": full_prompt}]}]
                }
                
                response = requests.post(url, headers=headers, json=data)
                
                if response.status_code == 200:
                    bot_reply = response.json()['candidates'][0]['content']['parts'][0]['text']
                else:
                    bot_reply = f"⚠️ API Error ({response.status_code}): Could not connect to Expert System."
                
                st.markdown(bot_reply)
            except Exception as e:
                bot_reply = "⚠️ System is currently offline for maintenance."
                st.error(bot_reply)
                
    st.session_state.messages.append({"role": "assistant", "content": bot_reply})