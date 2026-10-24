import streamlit as st
import numpy as np
import pickle
import os

st.set_page_config(page_title="CsPbBr3 PD Optimizer", layout="centered")

st.title("CsPbBr₃ Perovskite Performance Predictor")
st.write("Adjust fabrication parameters below to estimate device Specific Detectivity ($D^*$).")



model_exists = os.path.exists("best_perovskite_model.pkl")
scaler_exists = os.path.exists("scaler.pkl")

if not model_exists or not scaler_exists:
    st.error("⚠️ Missing model dependencies!")
    st.warning("Please open 'train.py' in VS Code and click the Play/Run button first to generate the required `.pkl` files.")
else:
    try:
        with open("best_perovskite_model.pkl", "rb") as f:
            model = pickle.load(f)
        with open("scaler.pkl", "rb") as f:
            scaler = pickle.load(f)
        
        st.success("✅ ML Model loaded successfully!")
        
        st.header("Fabrication Variables")
        col1, col2 = st.columns(2)
        
        with col1:
            temp = st.slider("Annealing Temperature (°C)", 70.0, 150.0, 110.0, step=1.0)
            speed = st.slider("Spin-Coating Speed (RPM)", 1500, 4000, 3000, step=100)
        with col2:
            doping = st.slider("Doping/Additive Conc. (%)", 0.0, 5.0, 1.5, step=0.1)
            thickness = st.slider("Target Film Thickness (nm)", 100, 400, 250, step=10)

        # Inference
        input_features = np.array([[temp, speed, doping, thickness]])
        scaled_features = scaler.transform(input_features)
        prediction = model.predict(scaled_features)[0]

        st.markdown("---")
        st.subheader("Predicted Device Metrics")
        st.metric(
            label="Estimated Specific Detectivity ($D^*$)", 
            value=f"{prediction:.3e} Jones"
        )
    except Exception as e:
        st.error(f"An execution error occurred: {e}")