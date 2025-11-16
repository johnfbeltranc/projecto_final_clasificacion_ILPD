import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import os
st.title("📊 Análisis y Gráficos ILPD")

try:
    data_path = os.path.join(os.getcwd(), '/workspaces/projecto_final_clasificacion_ILPD/', 'Indian Liver Patient Dataset (ILPD).csv') 
    # Asegúrate de usar el nombre de archivo correcto
    df = pd.read_csv(data_path)
    st.success("Datos cargados correctamente.")
except FileNotFoundError:
    st.error("Error: No se encontró el archivo de datos. Verifique la ruta.")
    st.stop()


#Data del diccionario
data = {
    'Nombre del Campo': ['Age', 'Gender', 'TB', 'DB', 'Alkphos', 'Sgpt', 'Sgot', 'TP', 'ALB', 'A/G Ratio', 'Liver Disease'],
    'Nombre Completo': ['Edad del Paciente', 'Género', 'Total Bilirubin', 'Direct Bilirubin', 'Alkaline Phosphatase', 'ALT — Alanine Aminotransferase', 'AST — Aspartate Aminotransferase', 'Total Proteins', 'Albumin', 'Albumin/Globulin Ratio', 'Indicador de Enfermedad Hepática'],
    'Descripción': [
        'Edad del paciente en años',
        'Sexo biológico del paciente',
        'Nivel total de bilirrubina en sangre',
        'Bilirrubina conjugada; elevada indica daño hepático',
        'Enzima asociada a obstrucción biliar o daño hepático',
        'Enzima hepática, elevada en daño hepático',
        'Enzima hepática, elevada en inflamación hepática o daño muscular',
        'Cantidad total de proteínas en sangre',
        'Proteína producida por el hígado; baja indica fallo hepático',
        'Relación albúmina–globulina; baja en enfermedad hepática',
        'Indica si el paciente tiene enfermedad hepática'
    ],
    'Unidades / Valores': [
        'Años',
        'Male / Female',
        'mg/dL',
        'mg/dL',
        'IU/L',
        'IU/L',
        'IU/L',
        'g/dL',
        'g/dL',
        'Razón',
        '1 = Enfermo / 2 = Sano'
    ]
}
# 2. Crear un DataFrame de Pandas
df_diccionario = pd.DataFrame(data)
# 3. Mostrar la tabla en tu aplicación Streamlit
st.title("Diccionario")
st.dataframe(df_diccionario)

#Tabla de datos
df.columns = ["Age","Gender","TB","DB","Alkphos","Sgpt","Sgot","TP","ALB","A/G Ratio","Target"]
st.subheader("Vista previa de datos:")
st.dataframe(df.head(10))

#Tabla descriptiva
desc_num = df.describe(include='number').T.assign(range=lambda x: x['max']-x['min'], cv=lambda x: x['std']/x['mean'])
desc_cat = df.describe(include='object').T
st.subheader("Resumen Estadístico de las Variables")
st.dataframe(desc_num)
st.dataframe(desc_cat)

#Grafica conteo
st.subheader("=== Conteo de casos positivos y negativos ===")
st.subheader("- Casos positivos 415")
st.subheader("- Casos negativos 167")
st.image("/workspaces/projecto_final_clasificacion_ILPD/notebooks/target_count.png", caption="Conteo de clases")

#GRafica conteo de generos
st.subheader("=== Conteo de géneros del DataSet ===")
st.image("/workspaces/projecto_final_clasificacion_ILPD/notebooks/gender_count.png", caption="Conteo de géneros")

#GRafica conteo de HeatMap
st.subheader("=== Correlación de Pearson entre variables predictoras ===")
st.image("/workspaces/projecto_final_clasificacion_ILPD/notebooks/heatmap.png", caption="Grafica de correlación de Pearson")

st.subheader("Comparación de los modelos probados")
st.image("/workspaces/projecto_final_clasificacion_ILPD/notebooks/rendimiento_modelos.png", caption="Modelos entrenados")
st.write("Realizamos el entrenamiento de distintos modelos de inteligencia articial para hallar el mejor rendimiento para nuestra métrica F1-score, que esta nos permite obtener el balance entre el Recall y Precision.")
st.write("Hemos elegido el RandomForest por mejor balance entre métricas de predicción Recall/Precision.")

st.subheader("Rendimiento parcial del Modelo RandomForest implementado para el análisis.")
data_rf = {
    'precision': [0.49, 0.86, None, 0.67, 0.75],
    'recall': [0.71, 0.70, None, 0.71, 0.70],
    'f1-score': [0.58, 0.77, 0.70, 0.68, 0.72],
    'support': [55, 138, 193, 193, 193]
}
index_names = ['no', 'yes', 'accuracy', 'macro avg', 'weighted avg']
# 3. Crear el DataFrame
df_reporte = pd.DataFrame(data_rf, index=index_names)
# Opcional: Rellenar los valores None (como el de 'accuracy' en precision y recall)
df_reporte = df_reporte.fillna('') 
# 4. Mostrar el DataFrame en Streamlit
st.dataframe(df_reporte)
