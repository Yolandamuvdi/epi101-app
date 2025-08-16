import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import google.generativeai as genai

# --- CONFIGURACIÓN DE GEMINI ---
genai.configure(api_key=st.secrets["GEMINI_API_KEY"])
model = genai.GenerativeModel("gemini-1.5-pro")

# Función para obtener respuesta del modelo
def get_gemini_response(prompt):
    response = model.generate_content(prompt)
    return response.text

# -------------------------------
# CONFIGURACIÓN DE LA APP
# -------------------------------

st.set_page_config(
    page_title="Epidemiología 101",
    page_icon="🧬",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.title("🧬 Epidemiología 101 - Tu Asistente Interactivo")

menu = st.sidebar.radio(
    "📌 Navegación",
    ["Conceptos Básicos", "Medidas de Asociación", "Tablas 2x2", "Ejercicios Prácticos", "Visualización de Datos", "Glosario Interactivo", "Chat con Gemini"]
)

# -------------------------------
# CONCEPTOS BÁSICOS
# -------------------------------
if menu == "Conceptos Básicos":
    st.header("📖 Conceptos Básicos de Epidemiología")
    conceptos = {
        "Incidencia": "Número de casos nuevos de una enfermedad en una población durante un periodo específico.",
        "Prevalencia": "Número total de casos (nuevos y existentes) en una población en un momento dado.",
        "Riesgo Relativo": "Razón de la incidencia de la enfermedad en los expuestos frente a los no expuestos.",
        "Odds Ratio": "Medida de asociación usada principalmente en estudios de casos y controles.",
        "Sesgo": "Error sistemático que puede llevar a conclusiones incorrectas.",
        "Confusión": "Distorsión de la asociación entre exposición y enfermedad por una tercera variable."
    }
    for concepto, definicion in conceptos.items():
        st.subheader(concepto)
        st.write(definicion)

# -------------------------------
# MEDIDAS DE ASOCIACIÓN
# -------------------------------
elif menu == "Medidas de Asociación":
    st.header("📊 Medidas de Asociación")
    st.write("Aquí puedes calcular Riesgo Relativo y Odds Ratio.")

    a = st.number_input("a: Expuestos y enfermos", min_value=0, value=10)
    b = st.number_input("b: Expuestos y no enfermos", min_value=0, value=20)
    c = st.number_input("c: No expuestos y enfermos", min_value=0, value=5)
    d = st.number_input("d: No expuestos y no enfermos", min_value=0, value=30)

    if (a+b) > 0 and (c+d) > 0 and (b+d) > 0 and (a+c) > 0:
        rr = (a / (a+b)) / (c / (c+d)) if (c+d) > 0 else np.nan
        or_ = (a*d) / (b*c) if (b*c) > 0 else np.nan

        st.write(f"**Riesgo Relativo (RR):** {rr:.2f}")
        st.write(f"**Odds Ratio (OR):** {or_:.2f}")

# -------------------------------
# TABLAS 2X2
# -------------------------------
elif menu == "Tablas 2x2":
    st.header("🔢 Generador de Tablas 2x2")

    a = st.number_input("Casos expuestos (a)", min_value=0, value=10)
    b = st.number_input("No casos expuestos (b)", min_value=0, value=20)
    c = st.number_input("Casos no expuestos (c)", min_value=0, value=5)
    d = st.number_input("No casos no expuestos (d)", min_value=0, value=30)

    tabla = pd.DataFrame(
        [[a, b, a+b],
         [c, d, c+d],
         [a+c, b+d, a+b+c+d]],
        index=["Enfermos", "No enfermos", "Total"],
        columns=["Expuestos", "No expuestos", "Total"]
    )

    st.table(tabla)

# -------------------------------
# EJERCICIOS PRÁCTICOS
# -------------------------------
elif menu == "Ejercicios Prácticos":
    st.header("✍️ Ejercicios de Epidemiología")
    ejercicios = [
        "Calcule la prevalencia de hipertensión en una población de 1000 personas donde 200 son hipertensos.",
        "En un estudio de cohorte, 50 de 200 fumadores desarrollaron cáncer de pulmón, mientras que 10 de 300 no fumadores lo desarrollaron. Calcule el Riesgo Relativo.",
        "En un estudio de casos y controles, 40 de 100 casos estuvieron expuestos a un factor de riesgo, y 30 de 200 controles estuvieron expuestos. Calcule el Odds Ratio."
    ]
    for i, ejercicio in enumerate(ejercicios, 1):
        st.subheader(f"Ejercicio {i}")
        st.write(ejercicio)

# -------------------------------
# VISUALIZACIÓN DE DATOS
# -------------------------------
elif menu == "Visualización de Datos":
    st.header("📈 Visualización de Datos")
    st.write("Sube un archivo CSV para graficar sus variables.")

    archivo = st.file_uploader("📂 Cargar CSV", type=["csv"])
    if archivo:
        df = pd.read_csv(archivo)
        st.write("Vista previa de los datos:", df.head())

        variable = st.selectbox("Selecciona una variable para graficar", df.columns)

        fig, ax = plt.subplots()
        sns.histplot(df[variable], kde=True, ax=ax)
        st.pyplot(fig)

# -------------------------------
# GLOSARIO INTERACTIVO
# -------------------------------
elif menu == "Glosario Interactivo":
    st.header("📚 Glosario Interactivo")
    glosario = {
        "Cohorte": "Grupo de individuos que comparten una característica y son seguidos en el tiempo.",
        "Estudio de casos y controles": "Diseño que compara sujetos con la enfermedad (casos) y sin ella (controles).",
        "Validez interna": "Grado en que los resultados de un estudio reflejan la realidad de la población estudiada.",
        "Validez externa": "Grado en que los resultados de un estudio son aplicables a otras poblaciones."
    }
    termino = st.selectbox("Selecciona un término", list(glosario.keys()))
    st.write(glosario[termino])

# -------------------------------
# CHAT CON GEMINI
# -------------------------------
elif menu == "Chat con Gemini":
    st.header("🤖 Chat con Gemini - Epidemiología 101")

    if "messages" not in st.session_state:
        st.session_state.messages = []

    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    if prompt := st.chat_input("Escribe tu pregunta de epidemiología..."):
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        response = get_gemini_response(prompt)
        st.session_state.messages.append({"role": "assistant", "content": response})
        with st.chat_message("assistant"):
            st.markdown(response)
