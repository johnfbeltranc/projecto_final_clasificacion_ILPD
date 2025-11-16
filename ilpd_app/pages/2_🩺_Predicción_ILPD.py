import streamlit as st
import pandas as pd
import joblib

st.set_page_config(layout="wide") # Recomendación de formato
st.title("🩺 Predicción ILPD - Enfermedad del Hígado")

# Cargar el modelo (Pipeline completo)
try:
    model = joblib.load("/workspaces/projecto_final_clasificacion_ILPD/notebooks/modelo_rf.pkl")
except FileNotFoundError:
    st.error("Error: El archivo 'modelo_rf.pkl' no se encontró. Asegúrate de que está en el mismo directorio.")
    st.stop()

st.subheader("Ingresa los datos del paciente")

col1, col2, col3 = st.columns(3)

with col1:
    age = st.number_input("Edad (años)", 1, 100, value=35, step=1)
    gender = st.selectbox("Género", ["Male", "Female"])
    tb = st.number_input("Total Bilirubin (0.1 - 1.2 mg/dL)", min_value=0.0, format="%.2f", value=0.1)
    db = st.number_input("Direct Bilirubin (0.0 - 0.3)", min_value=0.0, format="%.2f", value=0.0)

with col2:
    ap = st.number_input("Alkaline Phos (ALP)(44 - 147)", min_value=0.0, format="%.2f", value=44.0)
    aa = st.number_input("ALT (SGPT)(7 - 56)", min_value=0.0, format="%.2f", value=7.0)
    asa = st.number_input("AST (SGOT)(10 - 40)", min_value=0.0, format="%.2f", value=10.0)
    tp = st.number_input("Total Proteins(6.0 - 8.3)", min_value=0.0, format="%.2f", value=6.0)

with col3:
    alb = st.number_input("Albumin(3.4 - 5.4)", min_value=0.0, format="%.2f", value=3.4)
    agr = st.number_input("A/G Ratio( > 1.0)", min_value=0.0, format="%.2f", value=1.0)

# Botón de predicción
if st.button("Predecir", type="primary"):
    # 1. Crear el DataFrame con los nombres de columna correctos (iguales al entrenamiento)
    paciente = pd.DataFrame([{
        "age": age,
        "gender": gender,
        "tb": tb,
        "db": db,
        "alkphos": ap,
        "sgpt": aa,
        "sgot": asa,
        "tp": tp,
        "alb": alb,
        "a_g_ratio": agr
    }])
    
    # Manejo de nulos (si agr es 0, puedes causar problemas si n
    # o manejaste el nulo 
    # como 0, aunque en el notebook usaste la mediana, no está de más.)
    if paciente['a_g_ratio'].iloc[0] == 0:
        st.warning("Advertencia: El 'A/G Ratio' es 0, se recomienda revisarlo o usar la imputación de la mediana.")


    # 2. Hacer la predicción
    pred = model.predict(paciente)[0]

    st.divider()

    # 3. Mapeo CORREGIDO: Asumimos que la clase mayoritaria (que predice casi siempre) es ENFERMO.
    if pred == 1: # Si predice la clase mayoritaria (1 o 'yes' codificado)
        st.error(f"## 🚨 Resultado: Enfermedad del Hígado (Clase Predicha: **{pred}**)")
        st.write("El modelo sugiere un alto riesgo de enfermedad hepática. Consulte a un médico.")
    else: # Si predice la clase minoritaria (0 o 'no' codificado)
        st.success(f"## ✅ Resultado: Hígado Sano (Clase Predicha: **{pred}**)")
        st.write("El modelo sugiere que el paciente no tiene enfermedad hepática.")