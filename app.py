import streamlit as st
import joblib
import numpy as np

st.set_page_config(page_title="Multiple Disease Prediction", layout="wide", page_icon="🩺")

# ---- Custom CSS for a polished look ----
st.markdown("""
    <style>
    div[data-testid="stForm"] label {
        color: #1e293b !important;
        font-weight: 600 !important;
    }
    div[data-testid="stNumberInput"] input {
        color: white !important;
    }
    .main-header {
        background: linear-gradient(90deg, #1e293b, #0f172a);
        padding: 40px;
        border-radius: 12px;
        color: white;
        margin-bottom: 30px;
    }
    .main-header h1 {
        font-size: 42px;
        font-weight: 800;
        margin-bottom: 10px;
    }
    .main-header p {
        font-size: 18px;
        color: #cbd5e1;
    }
    .stButton>button, .stFormSubmitButton>button {
        background-color: #f97316;
        color: white;
        border-radius: 8px;
        padding: 10px 24px;
        font-weight: 600;
        border: none;
    }
    .stButton>button:hover, .stFormSubmitButton>button:hover {
        background-color: #ea580c;
        color: white;
    }
    div[data-testid="stForm"] {
        background-color: #ffffff;
        padding: 25px;
        border-radius: 12px;
        box-shadow: 0px 2px 10px rgba(0,0,0,0.08);
    }
    </style>
""", unsafe_allow_html=True)

# ---- Header banner ----
st.markdown("""
    <div class="main-header">
        <h1>🩺 Multiple Disease Prediction System</h1>
        <p>Get instant risk predictions for Parkinson's, Kidney Disease, and Liver Disease — powered by Machine Learning.</p>
    </div>
""", unsafe_allow_html=True)

st.sidebar.title("🔍 Navigation")
choice = st.sidebar.radio("Choose Disease", ["Parkinson's", "Kidney Disease", "Liver Disease"])

# ---------------- PARKINSON'S ----------------
if choice == "Parkinson's":
    st.subheader("Parkinson's Disease Prediction")
    model = joblib.load("saved_models/parkinsons_model.pkl")
    scaler = joblib.load("saved_models/parkinsons_scaler.pkl")

    with st.form("parkinsons_form"):
        st.write("Enter the voice measurement values below:")
        col1, col2, col3 = st.columns(3)

        with col1:
            fo = st.number_input("MDVP:Fo(Hz)", value=120.0)
            fhi = st.number_input("MDVP:Fhi(Hz)", value=150.0)
            flo = st.number_input("MDVP:Flo(Hz)", value=80.0)
            jitter_pct = st.number_input("MDVP:Jitter(%)", value=0.005, format="%.5f")
            jitter_abs = st.number_input("MDVP:Jitter(Abs)", value=0.00005, format="%.6f")
            jitter_rap = st.number_input("MDVP:RAP", value=0.003, format="%.5f")
            jitter_ppq = st.number_input("MDVP:PPQ", value=0.003, format="%.5f")
            jitter_ddp = st.number_input("Jitter:DDP", value=0.009, format="%.5f")

        with col2:
            shimmer = st.number_input("MDVP:Shimmer", value=0.03, format="%.5f")
            shimmer_db = st.number_input("MDVP:Shimmer(dB)", value=0.3)
            shimmer_apq3 = st.number_input("Shimmer:APQ3", value=0.015, format="%.5f")
            shimmer_apq5 = st.number_input("Shimmer:APQ5", value=0.017, format="%.5f")
            shimmer_apq = st.number_input("MDVP:APQ", value=0.02, format="%.5f")
            shimmer_dda = st.number_input("Shimmer:DDA", value=0.045, format="%.5f")
            nhr = st.number_input("NHR", value=0.02, format="%.5f")
            hnr = st.number_input("HNR", value=21.0)

        with col3:
            rpde = st.number_input("RPDE", value=0.5)
            dfa = st.number_input("DFA", value=0.7)
            spread1 = st.number_input("spread1", value=-5.5)
            spread2 = st.number_input("spread2", value=0.2)
            d2 = st.number_input("D2", value=2.3)
            ppe = st.number_input("PPE", value=0.2)

        submitted = st.form_submit_button("Predict Parkinson's")

        if submitted:
            input_data = np.array([[fo, fhi, flo, jitter_pct, jitter_abs, jitter_rap,
                                     jitter_ppq, jitter_ddp, shimmer, shimmer_db,
                                     shimmer_apq3, shimmer_apq5, shimmer_apq, shimmer_dda,
                                     nhr, hnr, rpde, dfa, spread1, spread2, d2, ppe]])
            scaled = scaler.transform(input_data)
            result = model.predict(scaled)
            if result[0] == 1:
                st.error("⚠️ The model predicts: Positive for Parkinson's Disease")
            else:
                st.success("✅ The model predicts: Negative for Parkinson's Disease")

# ---------------- KIDNEY DISEASE ----------------
elif choice == "Kidney Disease":
    st.subheader("Kidney Disease Prediction")
    model = joblib.load("saved_models/kidney_model.pkl")
    scaler = joblib.load("saved_models/kidney_scaler.pkl")

    with st.form("kidney_form"):
        col1, col2, col3 = st.columns(3)

        with col1:
            age = st.number_input("Age", value=45.0)
            bp = st.number_input("Blood Pressure", value=80.0)
            sg = st.number_input("Specific Gravity", value=1.02)
            al = st.number_input("Albumin", value=0.0)
            su = st.number_input("Sugar", value=0.0)
            rbc = st.selectbox("Red Blood Cells", ["normal", "abnormal"])
            pc = st.selectbox("Pus Cell", ["normal", "abnormal"])
            pcc = st.selectbox("Pus Cell Clumps", ["present", "notpresent"])

        with col2:
            ba = st.selectbox("Bacteria", ["present", "notpresent"])
            bgr = st.number_input("Blood Glucose Random", value=120.0)
            bu = st.number_input("Blood Urea", value=40.0)
            sc = st.number_input("Serum Creatinine", value=1.2)
            sod = st.number_input("Sodium", value=140.0)
            pot = st.number_input("Potassium", value=4.5)
            hemo = st.number_input("Hemoglobin", value=13.0)
            pcv = st.number_input("Packed Cell Volume", value=40.0)

        with col3:
            wc = st.number_input("White Blood Cell Count", value=8000.0)
            rc = st.number_input("Red Blood Cell Count", value=5.0)
            htn = st.selectbox("Hypertension", ["yes", "no"])
            dm = st.selectbox("Diabetes Mellitus", ["yes", "no"])
            cad = st.selectbox("Coronary Artery Disease", ["yes", "no"])
            appet = st.selectbox("Appetite", ["good", "poor"])
            pe = st.selectbox("Pedal Edema", ["yes", "no"])
            ane = st.selectbox("Anemia", ["yes", "no"])

        submitted = st.form_submit_button("Predict Kidney Disease")

        if submitted:
            binary_map = {"normal": 1, "abnormal": 0, "present": 1, "notpresent": 0,
                           "yes": 1, "no": 0, "good": 1, "poor": 0}
            input_data = np.array([[age, bp, sg, al, su, binary_map[rbc], binary_map[pc],
                                     binary_map[pcc], binary_map[ba], bgr, bu, sc, sod, pot,
                                     hemo, pcv, wc, rc, binary_map[htn], binary_map[dm],
                                     binary_map[cad], binary_map[appet], binary_map[pe],
                                     binary_map[ane]]])
            scaled = scaler.transform(input_data)
            result = model.predict(scaled)
            if result[0] == 1:
                st.error("⚠️ The model predicts: Chronic Kidney Disease Detected")
            else:
                st.success("✅ The model predicts: No Kidney Disease Detected")

# ---------------- LIVER DISEASE ----------------
elif choice == "Liver Disease":
    st.subheader("Liver Disease Prediction")
    model = joblib.load("saved_models/liver_model.pkl")
    scaler = joblib.load("saved_models/liver_scaler.pkl")

    with st.form("liver_form"):
        col1, col2 = st.columns(2)

        with col1:
            age = st.number_input("Age", value=45.0)
            gender = st.selectbox("Gender", ["Male", "Female"])
            total_bilirubin = st.number_input("Total Bilirubin", value=1.0)
            direct_bilirubin = st.number_input("Direct Bilirubin", value=0.3)
            alk_phosphatase = st.number_input("Alkaline Phosphotase", value=200.0)

        with col2:
            alamine = st.number_input("Alamine Aminotransferase", value=30.0)
            aspartate = st.number_input("Aspartate Aminotransferase", value=35.0)
            total_proteins = st.number_input("Total Proteins", value=6.5)
            albumin = st.number_input("Albumin", value=3.2)
            ag_ratio = st.number_input("Albumin and Globulin Ratio", value=1.0)

        submitted = st.form_submit_button("Predict Liver Disease")

        if submitted:
            gender_val = 1 if gender == "Male" else 0
            input_data = np.array([[age, gender_val, total_bilirubin, direct_bilirubin,
                                     alk_phosphatase, alamine, aspartate, total_proteins,
                                     albumin, ag_ratio]])
            scaled = scaler.transform(input_data)
            result = model.predict(scaled)
            if result[0] == 1:
                st.error("⚠️ The model predicts: Liver Disease Detected")
            else:
                st.success("✅ The model predicts: No Liver Disease Detected")