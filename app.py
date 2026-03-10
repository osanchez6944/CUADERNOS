import pandas as pd
import numpy as np
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.decomposition import PCA
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.utils import to_categorical

# 1. Cargar datos y limpiar
url = "https://github.com/adiacla/bigdata/raw/master/riesgo.xlsx"
df = pd.read_excel(url)
df = df.drop(columns=["Customer_ID","Name","SSN"])

# Rellenar valores nulos para evitar errores en Streamlit
for col in df.columns:
    if df[col].dtype == 'object':
        df[col] = df[col].fillna(df[col].mode()[0])
    else:
        df[col] = df[col].fillna(df[col].mean())

# 2. Guardar las columnas originales (sin el target)
X_cols = df.drop("Credit_Score", axis=1).columns
joblib.dump(list(X_cols), 'columnas.pkl')

# 3. Codificar variables categóricas CORRECTAMENTE (Un encoder por columna)
cat_cols = df.select_dtypes(include=["object"]).columns
encoders = {} # Diccionario para guardar todos los encoders

for col in cat_cols:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col].astype(str))
    encoders[col] = le

joblib.dump(encoders, 'encoders.pkl')

# 4. Separar variables
X = df.drop("Credit_Score", axis=1)
y = df["Credit_Score"]
y = to_categorical(y)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 5. Normalizar
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)
joblib.dump(scaler, 'scaler.pkl')

# 6. PCA
pca = PCA(n_components=10)
X_train = pca.fit_transform(X_train)
X_test = pca.transform(X_test)
joblib.dump(pca, 'pca.pkl')

# 7. Red Neuronal
model = Sequential()
model.add(Dense(64, activation='relu', input_shape=(X_train.shape[1],)))
model.add(Dense(32, activation='relu'))
model.add(Dense(3, activation='softmax'))

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
model.fit(X_train, y_train, epochs=50, batch_size=32, validation_split=0.2, verbose=0) # verbose=0 para no llenar la pantalla

# 8. Guardar modelo
model.save('modelo.keras')
print("¡Entrenamiento finalizado y archivos exportados! Descarga los .pkl y el .keras")
