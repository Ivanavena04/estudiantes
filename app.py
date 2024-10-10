import streamlit as st
import pandas as pd
import joblib
import os

# Título de la aplicación
st.title("Predicción de Resultados Educativos")

# Información adicional
st.markdown("""
    **Facultad de Ingeniería UACH**  
    **Maestría en Ingeniería en Computación**  
    **Materia: Machine Learning**  
    **Alumno: Iván Avena Caro**  
    **Matrícula: 193650**  
""")

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
    input_df = pd.DataFrame(input_data, index=[0])  # Crear un DataFrame válido
    predictions = loaded_model.predict(input_df)
    return predictions

# Ingreso de datos
st.header("Ingrese los datos del estudiante")

# Campos de entrada según las características filtradas
application_mode = st.selectbox("Modo de Aplicación", list(range(1, 19)))
age_at_enrollment = st.number_input("Edad al ingreso", min_value=15, max_value=100)  # Ajusta según el rango
tuition_fees = st.selectbox("¿Cuotas al día?", [0, 1])  # 0 para No, 1 para Sí
curricular_units_1st_sem_approved = st.number_input("Unidades curriculares 1er sem (aprobadas)", min_value=0)
curricular_units_1st_sem_grade = st.number_input("Unidades curriculares 1er sem (calificación)", min_value=0.0)
curricular_units_2nd_sem_approved = st.number_input("Unidades curriculares 2do sem (aprobadas)", min_value=0)
curricular_units_2nd_sem_grade = st.number_input("Unidades curriculares 2do sem (calificación)", min_value=0.0)
curricular_units_1st_sem_evaluations = st.number_input("Unidades curriculares 1er sem (evaluaciones)", min_value=0)
curricular_units_2nd_sem_evaluations = st.number_input("Unidades curriculares 2do sem (evaluaciones)", min_value=0)

# Botón para hacer la predicción
if st.button("Predecir"):
    input_data = {
        "Application mode": application_mode,
        "Age at enrollment": age_at_enrollment,
        "Tuition fees up to date": tuition_fees,
        "Curricular units 1st sem (approved)": curricular_units_1st_sem_approved,
        "Curricular units 1st sem (grade)": curricular_units_1st_sem_grade,
        "Curricular units 2nd sem (approved)": curricular_units_2nd_sem_approved,
        "Curricular units 2nd sem (grade)": curricular_units_2nd_sem_grade,
        "Curricular units 1st sem (evaluations)": curricular_units_1st_sem_evaluations,
        "Curricular units 2nd sem (evaluations)": curricular_units_2nd_sem_evaluations,
    }

    # Realizar la predicción
    try:
        prediction = predict(input_data)
        # Mostrar resultados
        st.success(f"La predicción es: {prediction[0]}")
    except Exception as e:
        st.error(f"Ocurrió un error al realizar la predicción: {e}")

# Información adicional
st.header("Resultados del Modelo")
st.write("Modelo cargado:", loaded_model)