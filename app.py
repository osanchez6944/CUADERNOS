import streamlit as st
import pandas as pd
import numpy as np
import joblib
from tensorflow.keras.models import load_model

# Configuración de la página
st.set_page_config(page_title="Riesgo Crediticio", layout="wide")

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
except Exception as e:
    st.error(f"❌ Error cargando los archivos: {e}")
    st.stop()

# --- DISEÑO DE LA PANTALLA PRINCIPAL ---
st.title("💳 Clasificador de Riesgo Crediticio (ANN)")
st.write("Esta aplicación utiliza una Red Neuronal Densa para clasificar el perfil crediticio de un cliente.")

# Diccionario de traducciones
traducciones = {
    "Age": "Edad",
    "Occupation": "Ocupación",
    "Annual_Income": "Ingreso Anual ($)",
    "Monthly_Inhand_Salary": "Salario Mensual Neto ($)",
    "Num_Bank_Accounts": "Número de Cuentas Bancarias",
    "Num_Credit_Card": "Número de Tarjetas de Crédito",
    "Interest_Rate": "Tasa de Interés (%)",
    "Num_of_Loan": "Número de Préstamos Activos",
    "Type_of_Loan": "Tipo de Préstamo",
    "Delay_from_due_date": "Días promedio de retraso",
    "Num_of_Delayed_Payment": "Cantidad de pagos retrasados",
    "Changed_Credit_Limit": "Cambio en Límite de Crédito",
    "Num_Credit_Inquiries": "Consultas de Crédito",
    "Credit_Mix": "Mix de Crédito",
    "Outstanding_Debt": "Deuda Pendiente ($)",
    "Credit_Utilization_Ratio": "Uso del Crédito (%)",
    "Credit_History_Age": "Historial Crediticio (Meses)",
    "Payment_of_Min_Amount": "¿Paga el mínimo?",
    "Total_EMI_per_month": "Cuotas Mensuales (EMI)",
    "Amount_invested_monthly": "Inversión Mensual ($)",
    "Payment_Behaviour": "Comportamiento de Pago",
    "Monthly_Balance": "Balance Mensual Final ($)"
}

# --- DISEÑO DE LA BARRA LATERAL (SIDEBAR) ---
st.sidebar.header("Parámetros del Cliente")
st.sidebar.divider()

input_data = {}

# Crear los campos de entrada dentro de la barra lateral
for col in columnas:
    nombre_visual = traducciones.get(col, col) 
    
    # Si es categórica, mostramos un selectbox
    if col in encoders:
        clases_reales = encoders[col].classes_
        input_data[col] = st.sidebar.selectbox(nombre_visual, clases_reales)
    # Si es numérica, mostramos un number_input
    else:
        input_data[col] = st.sidebar.number_input(nombre_visual, value=0.0)

# --- BOTÓN Y PREDICCIÓN EN LA PANTALLA PRINCIPAL ---
st.info("Complete los datos en el panel izquierdo para ver la predicción.")

if st.button("Realizar Diagnóstico Crediticio"):
    # Convertir a DataFrame
    df_input = pd.DataFrame([input_data])
    
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
