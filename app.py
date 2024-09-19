import streamlit as st
import pandas as pd
import joblib
import os

# Ruta del modelo
model_filename = 'final_model.joblib'

# Verificar si el archivo existe
if not os.path.exists(model_filename):
    st.error(f"No se encontró el archivo del modelo en la ruta: {model_filename}")
else:
    # Cargar el modelo
    loaded_model = joblib.load(model_filename)

# Función para predecir
def predict(input_data):
    input_df = pd.DataFrame(input_data)
    predictions = loaded_model.predict(input_df)
    return predictions

# Título de la aplicación
st.title("Predicción de Resultados Educativos")

# Ingreso de datos
st.header("Ingrese los datos del estudiante")

# Campos de entrada según las tablas
marital_status = st.selectbox("Estado Civil", [1, 2, 3, 4, 5, 6])
nationality = st.selectbox("Nacionalidad", list(range(1, 22)))
application_mode = st.selectbox("Modo de Aplicación", list(range(1, 19)))
application_order = st.number_input("Orden de Aplicación", min_value=1, max_value=5)
course = st.selectbox("Curso", list(range(1, 18)))
previous_qualification = st.selectbox("Calificación Anterior", list(range(1, 14)))
mothers_qualification = st.selectbox("Calificación de la Madre", list(range(1, 35)))
fathers_qualification = st.selectbox("Calificación del Padre", list(range(1, 35)))
mothers_occupation = st.selectbox("Ocupación de la Madre", list(range(1, 48)))
fathers_occupation = st.selectbox("Ocupación del Padre", list(range(1, 48)))
displaced = st.selectbox("Desplazado", [0, 1])  # 0 para No, 1 para Sí
educational_needs = st.selectbox("Necesidades educativas especiales", [0, 1])  # 0 para No, 1 para Sí
debtor = st.selectbox("Deudor", [0, 1])  # 0 para No, 1 para Sí
tuition_fees = st.selectbox("¿Cuotas al día?", [0, 1])  # 0 para No, 1 para Sí
gender = st.selectbox("Género", [1, 0])  # 1 para masculino, 0 para femenino
scholarship_holder = st.selectbox("Becado", [0, 1])  # 0 para No, 1 para Sí
age_at_enrollment = st.number_input("Edad al ingreso", min_value=15, max_value=100)  # Ajusta según el rango
international = st.selectbox("Internacional", [0, 1])  # 0 para No, 1 para Sí
unemployment_rate = st.number_input("Tasa de desempleo")
inflation_rate = st.number_input("Tasa de inflación")
gdp = st.number_input("PIB")

# Botón para hacer la predicción
if st.button("Predecir"):
    input_data = {
        "Marital status": marital_status,
        "Nationality": nationality,
        "Application mode": application_mode,
        "Application order": application_order,
        "Course": course,
        "Daytime/evening attendance": attendance_regime,
        "Previous qualification": previous_qualification,
        "Mother's qualification": mothers_qualification,
        "Father's qualification": fathers_qualification,
        "Mother's occupation": mothers_occupation,
        "Father's occupation": fathers_occupation,
        "Displaced": displaced,
        "Educational special needs": educational_needs,
        "Debtor": debtor,
        "Tuition fees up to date": tuition_fees,
        "Gender": gender,
        "Scholarship holder": scholarship_holder,
        "Age at enrollment": age_at_enrollment,
        "International": international,
        "Unemployment rate": unemployment_rate,
        "Inflation rate": inflation_rate,
        "GDP": gdp,
    }

    # Realizar la predicción
    prediction = predict(input_data)
    
    # Mostrar resultados
    st.success(f"La predicción es: {prediction[0]}")


# Información adicional
st.header("Resultados del Modelo")
st.write("Modelo cargado:", loaded_model.named_steps['classifier'])
