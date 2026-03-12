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
st.info("Complete los datos en el panel izquierdo para ver la predicción.")

# --- DISEÑO DE LA BARRA LATERAL (SIDEBAR) ---
st.sidebar.header("Parámetros del Cliente")

input_data = {}

# 1. Los 8 campos idénticos a la foto
input_data['Age'] = st.sidebar.slider("Edad", min_value=18, max_value=100, value=30)
input_data['Annual_Income'] = st.sidebar.number_input("Ingreso Anual", min_value=0.0, value=50000.0, step=1000.0)
input_data['Num_Bank_Accounts'] = st.sidebar.number_input("Número de cuentas bancarias", min_value=0, max_value=20, value=2, step=1)
input_data['Num_Credit_Card'] = st.sidebar.number_input("Número de tarjetas de crédito", min_value=0, max_value=20, value=3, step=1)
input_data['Interest_Rate'] = st.sidebar.slider("Tasa de Interés (%)", min_value=0.0, max_value=50.0, value=12.0, step=0.1)
input_data['Outstanding_Debt'] = st.sidebar.number_input("Deuda Pendiente ($)", min_value=0.0, value=1500.00, step=100.0)

# Para Mix de Crédito intentamos poner 'Good' por defecto si existe
if 'Credit_Mix' in encoders:
    clases_mix = list(encoders['Credit_Mix'].classes_)
    idx = clases_mix.index('Good') if 'Good' in clases_mix else 0
    input_data['Credit_Mix'] = st.sidebar.selectbox("Mix de Crédito", clases_mix, index=idx)

# Para Paga el mínimo intentamos poner 'Yes' por defecto si existe
if 'Payment_of_Min_Amount' in encoders:
    clases_min = list(encoders['Payment_of_Min_Amount'].classes_)
    idx = clases_min.index('Yes') if 'Yes' in clases_min else 0
    input_data['Payment_of_Min_Amount'] = st.sidebar.selectbox("¿Paga el mínimo?", clases_min, index=idx)


# 2. El resto de las columnas necesarias para el modelo
campos_ya_agregados = ['Age', 'Annual_Income', 'Num_Bank_Accounts', 'Num_Credit_Card', 'Interest_Rate', 'Outstanding_Debt', 'Credit_Mix', 'Payment_of_Min_Amount']

traducciones_extra = {
    "Occupation": "Ocupación",
    "Monthly_Inhand_Salary": "Salario Mensual Neto ($)",
    "Num_of_Loan": "Número de Préstamos Activos",
    "Type_of_Loan": "Tipo de Préstamo",
    "Delay_from_due_date": "Días promedio de retraso",
    "Num_of_Delayed_Payment": "Cantidad de pagos retrasados",
    "Changed_Credit_Limit": "Cambio en Límite de Crédito",
    "Num_Credit_Inquiries": "Consultas de Crédito",
    "Credit_Utilization_Ratio": "Uso del Crédito (%)",
    "Credit_History_Age": "Historial Crediticio (Meses)",
    "Total_EMI_per_month": "Cuotas Mensuales (EMI)",
    "Amount_invested_monthly": "Inversión Mensual ($)",
    "Payment_Behaviour": "Comportamiento de Pago",
    "Monthly_Balance": "Balance Mensual Final ($)"
}

for col in columnas:
    if col not in campos_ya_agregados:
        nombre_visual = traducciones_extra.get(col, col) 
        if col in encoders:
            clases_reales = encoders[col].classes_
            input_data[col] = st.sidebar.selectbox(nombre_visual, clases_reales)
        else:
            input_data[col] = st.sidebar.number_input(nombre_visual, value=0.0)

# --- BOTÓN Y PREDICCIÓN EN LA PANTALLA PRINCIPAL ---
if st.button("Realizar Diagnóstico Crediticio"):
    
    # IMPORTANTE: Reordenar el diccionario para que coincida exactamente con lo que espera el modelo
    df_input = pd.DataFrame([input_data])[columnas]
    
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
