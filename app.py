import streamlit as st
import pandas as pd
import numpy as np
import joblib
from tensorflow.keras.models import load_model

# Configuración de la página
st.set_page_config(page_title="Riesgo Crediticio", layout="wide")

# 1. Cargar todos los artefactos
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
except Exception as e:
    st.error(f"❌ Error cargando los archivos: {e}")
    st.stop()

# --- DISEÑO DE LA PANTALLA PRINCIPAL ---
st.title("💳 Clasificador de Riesgo Crediticio (ANN)")
st.write("Esta aplicación utiliza una Red Neuronal Densa para clasificar el perfil crediticio de un cliente.")
st.info("Complete los datos en el panel izquierdo para ver la predicción.")

# --- DISEÑO DE LA BARRA LATERAL (IDÉNTICO A LA FOTO) ---
st.sidebar.header("Parámetros del Cliente")

# Creamos solo los 8 campos visibles
input_ui = {}
input_ui['Age'] = st.sidebar.slider("Edad", min_value=18, max_value=100, value=30)
input_ui['Annual_Income'] = st.sidebar.number_input("Ingreso Anual", min_value=0.0, value=50000.0, step=1000.0)
input_ui['Num_Bank_Accounts'] = st.sidebar.number_input("Número de cuentas bancarias", min_value=0, max_value=20, value=2, step=1)
input_ui['Num_Credit_Card'] = st.sidebar.number_input("Número de tarjetas de crédito", min_value=0, max_value=20, value=3, step=1)
input_ui['Interest_Rate'] = st.sidebar.slider("Tasa de Interés (%)", min_value=0.0, max_value=50.0, value=12.0, step=0.1)
input_ui['Outstanding_Debt'] = st.sidebar.number_input("Deuda Pendiente ($)", min_value=0.0, value=1500.00, step=100.0)

# Para Mix de Crédito
if 'Credit_Mix' in encoders:
    clases_mix = list(encoders['Credit_Mix'].classes_)
    idx = clases_mix.index('Good') if 'Good' in clases_mix else 0
    input_ui['Credit_Mix'] = st.sidebar.selectbox("Mix de Crédito", clases_mix, index=idx)

# Para Paga el mínimo
if 'Payment_of_Min_Amount' in encoders:
    clases_min = list(encoders['Payment_of_Min_Amount'].classes_)
    idx = clases_min.index('Yes') if 'Yes' in clases_min else 0
    input_ui['Payment_of_Min_Amount'] = st.sidebar.selectbox("¿Paga el mínimo?", clases_min, index=idx)

# --- BOTÓN Y PREDICCIÓN ---
if st.button("Realizar Diagnóstico Crediticio"):
    
    # TRUCO INVISIBLE: Llenar las columnas faltantes con valores por defecto 
    # para que la red neuronal no colapse al faltarle datos.
    datos_completos = {}
    for col in columnas:
        if col in input_ui:
            datos_completos[col] = input_ui[col] # Usa lo que pusiste en pantalla
        else:
            # Rellenar en modo oculto
            if col in encoders:
                datos_completos[col] = encoders[col].classes_[0]
            else:
                datos_completos[col] = 0.0
                
    df_input = pd.DataFrame([datos_completos])[columnas]
    
    with st.spinner('Analizando el perfil del cliente...'):
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
            st.divider()
            st.subheader("📊 Resultado del Diagnóstico")
            
            if clase_ganadora == 0:
                st.success(f"**Categoría asignada:** {clase_ganadora} (Bajo Riesgo) ✅")
            elif clase_ganadora == 1:
                st.warning(f"**Categoría asignada:** {clase_ganadora} (Riesgo Medio) ⚠️")
            else:
                st.error(f"**Categoría asignada:** {clase_ganadora} (Alto Riesgo) ❌")
                
            st.write(f"**Probabilidades de la red neuronal:** Clase 0 ({probabilidades[0]:.1f}%) | Clase 1 ({probabilidades[1]:.1f}%) | Clase 2 ({probabilidades[2]:.1f}%)")
            
        except Exception as e:
            st.error(f"Error procesando los datos: {e}")
