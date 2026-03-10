import streamlit as st
import pandas as pd
import numpy as np
import joblib
from tensorflow.keras.models import load_model

# Configuración de la página
st.set_page_config(page_title="Riesgo Crediticio", layout="wide")
st.title("🏦 Predicción de Riesgo Crediticio (ANN Multiclase)")

# 1. Cargar todos los artefactos exportados
@st.cache_resource
def cargar_modelos():
    modelo = load_model('modelo.keras')
    scaler = joblib.load('scaler.pkl')
    pca = joblib.load('pca.pkl')
    encoders = joblib.load('encoders.pkl')
    columnas = joblib.load('columnas.pkl')
    return modelo, scaler, pca, encoders, columnas

try:
    modelo, scaler, pca, encoders, columnas = cargar_modelos()
    st.success("✅ Modelo y transformadores cargados correctamente.")
except Exception as e:
    st.error(f"❌ Error cargando los archivos. Verifica que subiste los .pkl y .keras a GitHub. Detalle: {e}")
    st.stop()

st.divider()
st.subheader("📋 Ingresa los datos del cliente:")

# 2. Crear el formulario dinámico basado en las columnas originales
input_data = {}
cols_visuales = st.columns(3) # Dividir la pantalla en 3 columnas

for i, col in enumerate(columnas):
    with cols_visuales[i % 3]:
        # Si es categórica, mostramos un selectbox con las clases reales
        if col in encoders:
            clases_reales = encoders[col].classes_
            input_data[col] = st.selectbox(f"{col}", clases_reales)
        # Si es numérica, mostramos un number_input
        else:
            input_data[col] = st.number_input(f"{col}", value=0.0)

# 3. Botón para predecir
st.divider()
if st.button("🚀 Evaluar Riesgo de Crédito", use_container_width=True):
    # Convertir a DataFrame
    df_input = pd.DataFrame([input_data])
    
    # Aplicar el preprocesamiento exactamente igual que en Colab
    try:
        # A. Codificar categóricas
        for col in encoders.keys():
            df_input[col] = encoders[col].transform(df_input[col].astype(str))
            
        # B. Escalar
        df_scaled = scaler.transform(df_input)
        
        # C. PCA
        df_pca = pca.transform(df_scaled)
        
        # D. Predecir
        prediccion = modelo.predict(df_pca)
        clase_ganadora = np.argmax(prediccion, axis=1)[0]
        probabilidades = prediccion[0] * 100
        
        # Mostrar resultados
        st.subheader("📊 Resultado de la Evaluación")
        
        if clase_ganadora == 0:
            st.success(f"Categoría asignada: {clase_ganadora} (Bajo Riesgo)")
        elif clase_ganadora == 1:
            st.warning(f"Categoría asignada: {clase_ganadora} (Riesgo Medio)")
        else:
            st.error(f"Categoría asignada: {clase_ganadora} (Alto Riesgo)")
            
        st.write(f"**Probabilidades de la red neuronal:** Clase 0 ({probabilidades[0]:.1f}%) | Clase 1 ({probabilidades[1]:.1f}%) | Clase 2 ({probabilidades[2]:.1f}%)")
        
    except Exception as e:
        st.error(f"Error procesando los datos: {e}")
