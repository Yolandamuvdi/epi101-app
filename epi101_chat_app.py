import streamlit as st
import openai
import matplotlib.pyplot as plt
import pandas as pd

st.set_page_config(page_title="Epidemiología 101", page_icon="🧪", layout="wide")

# Estilo visual personalizado
st.markdown("""
    <style>
    body, .block-container {
        background: linear-gradient(135deg, #f0f4f8 0%, #d9e2ec 100%);
        color: #1b2838;
        font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
    }
    .block-container {
        padding: 2rem 4rem 3rem 4rem;
        max-width: 1100px;
        margin: auto;
        background: white;
        border-radius: 12px;
        box-shadow: 0 10px 25px rgba(0,0,0,0.1);
    }
    .title {
        font-size: 3rem;
        font-weight: 900;
        color: #0d3b66;
        margin-bottom: 0.2rem;
    }
    .subtitle {
        font-size: 1.3rem;
        color: #3e5c76;
        margin-bottom: 2rem;
        font-weight: 500;
    }
    </style>
""", unsafe_allow_html=True)

st.markdown('<div class="title">🧠 Epidemiología 101 - Asistente educativo</div>', unsafe_allow_html=True)
st.markdown('<div class="subtitle">Plataforma integral para el aprendizaje de conceptos clave de epidemiología, salud pública y análisis de datos, creada por Yolanda Muvdi.</div>', unsafe_allow_html=True)

if "OPENAI_API_KEY" in st.secrets:
    openai.api_key = st.secrets["OPENAI_API_KEY"]
    st.success("✅ Clave API detectada correctamente.")
else:
    st.error("❌ No se encontró OPENAI_API_KEY. Ve al panel de Secrets en Streamlit Cloud y agrégala.")
    st.stop()

# Pestañas de navegación
tabs = st.tabs([
    "Conceptos Básicos",
    "Medidas de Asociación",
    "Diseños de Estudio",
    "Sesgos y Errores",
    "Glosario Interactivo",
    "Ejercicios Prácticos",
    "Tablas 2x2 y Cálculos",
    "Visualización de Datos",
    "Chat"
])

# Contenidos directamente en el código
with tabs[0]:
    st.header("📌 Conceptos Básicos de Epidemiología")
    st.markdown("""
    **Epidemiología:** Ciencia que estudia la distribución y determinantes de los eventos en salud en poblaciones.

    **Incidencia:** Número de casos nuevos en una población durante un periodo.

    **Prevalencia:** Proporción de personas que presentan una condición de salud en un momento o periodo específico.

    **Tasa:** Expresión matemática que relaciona la frecuencia de un evento con el tiempo/personas en riesgo.

    **Variables:** Características observables que pueden medirse (cualitativas o cuantitativas).

    **Población en riesgo:** Conjunto de individuos susceptibles al evento.

    **Cohorte:** Grupo seguido en el tiempo para observar aparición de eventos.

    **Casos:** Individuos que presentan la enfermedad de interés.

    **Controles:** Individuos sin la enfermedad de interés.

    **Error aleatorio vs sesgo:** Variabilidad por azar vs. error sistemático.
    """)

with tabs[1]:
    st.header("📈 Medidas de Asociación")
    st.markdown("""
    **Riesgo Relativo (RR):** Comparación del riesgo de desarrollar un evento entre dos grupos.

    **Odds Ratio (OR):** Comparación de las probabilidades de exposición entre casos y controles.

    **Riesgo Atribuible (RA):** Diferencia entre tasas de incidencia.

    **Fracción Etiológica:** Proporción de riesgo atribuible a la exposición.

    **Razón de Tasas:** Comparación entre tasas de incidencia por unidad de tiempo/persona.

    **NNT (Número Necesario a Tratar):** Número de pacientes a tratar para evitar un caso.

    **Hazard Ratio (HR):** Comparación de tasas instantáneas en estudios de supervivencia.
    """)

with tabs[2]:
    st.header("📊 Diseños de Estudio Epidemiológico")
    st.markdown("""
    **Estudios Observacionales:**
    - Transversales: Miden prevalencia.
    - Cohorte: Evalúan incidencia y riesgo.
    - Casos y controles: Evalúan asociaciones retrospectivas.

    **Estudios Experimentales:**
    - Ensayos Clínicos Aleatorizados (RCT): Intervención asignada por el investigador.
    - Cuasiexperimentales: No hay aleatorización.

    **Estudios Ecológicos:** Unidades de análisis son grupos poblacionales.
    """)

with tabs[3]:
    st.header("⚠️ Sesgos y Errores")
    st.markdown("""
    **Sesgo de Selección:** Error por la forma de incluir sujetos en el estudio.

    **Sesgo de Información:** Error en la medición de variables (ej. recuerdo, observación).

    **Confusión:** Asociación espuria por efecto de una tercera variable.

    **Error Aleatorio:** Variabilidad por azar, no atribuible a sesgo.

    **Validez interna:** Precisión de los resultados dentro del estudio.

    **Validez externa:** Generalización de los hallazgos.
    """)

with tabs[4]:
    st.header("📚 Glosario Interactivo A–Z")
    glosario = {
        "Incidencia": "Número de casos nuevos en una población durante un periodo específico.",
        "Prevalencia": "Proporción de personas con una condición en un momento dado.",
        "Odds Ratio": "Medida de asociación que compara las probabilidades de exposición.",
        "Riesgo Relativo": "Comparación de riesgo entre expuestos y no expuestos.",
        "Cohorte": "Grupo seguido en el tiempo para observar aparición de eventos.",
        "Ensayo Clínico Aleatorizado": "Estudio experimental con asignación aleatoria de tratamiento.",
        "Tasa de Mortalidad": "Medida de frecuencia de muertes en una población."
    }
    for termino, definicion in glosario.items():
        with st.expander(termino):
            st.write(definicion)

with tabs[5]:
    st.header("🧪 Ejercicios Prácticos")
    preguntas = [
        {
            "pregunta": "¿Cuál es la diferencia entre incidencia y prevalencia?",
            "opciones": [
                "Incidencia mide casos existentes, prevalencia los nuevos.",
                "Incidencia es una proporción, prevalencia una tasa.",
                "Incidencia mide casos nuevos, prevalencia los existentes.",
                "No hay diferencia."
            ],
            "respuesta_correcta": "Incidencia mide casos nuevos, prevalencia los existentes."
        },
        {
            "pregunta": "¿Qué medida se usa comúnmente en estudios de casos y controles?",
            "opciones": ["Riesgo Relativo", "Odds Ratio", "Hazard Ratio", "Riesgo Atribuible"],
            "respuesta_correcta": "Odds Ratio"
        }
    ]
    for i, q in enumerate(preguntas):
        st.subheader(f"Pregunta {i+1}")
        respuesta = st.radio(q['pregunta'], q['opciones'], key=f"q{i}")
        if st.button(f"Verificar {i+1}"):
            if respuesta == q['respuesta_correcta']:
                st.success("✅ Correcto")
            else:
                st.error(f"❌ Incorrecto. Respuesta correcta: {q['respuesta_correcta']}")

with tabs[6]:
    st.header("📊 Tablas 2x2 y Cálculos Epidemiológicos")
    col1, col2 = st.columns(2)
    with col1:
        a = st.number_input("Casos con exposición (a)", min_value=0, step=1)
        b = st.number_input("Casos sin exposición (b)", min_value=0, step=1)
    with col2:
        c = st.number_input("Controles con exposición (c)", min_value=0, step=1)
        d = st.number_input("Controles sin exposición (d)", min_value=0, step=1)
    if st.button("Calcular RR y OR"):
        try:
            rr = (a / (a + b)) / (c / (c + d)) if (a + b) > 0 and (c + d) > 0 else None
            orr = (a * d) / (b * c) if b > 0 and c > 0 else None
            if rr:
                st.success(f"RR: {rr:.2f}")
            if orr:
                st.success(f"OR: {orr:.2f}")
        except:
            st.error("Error en los cálculos")

with tabs[7]:
    st.header("📈 Visualización de Datos")
    st.markdown("Carga tus datos en formato CSV para graficar.")
    uploaded_file = st.file_uploader("Sube un archivo CSV", type=["csv"])
    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)
        st.dataframe(df)
        col_x = st.selectbox("Selecciona variable para eje X", df.columns)
        col_y = st.selectbox("Selecciona variable para eje Y", df.columns)
        tipo = st.selectbox("Tipo de gráfico", ["Barras", "Líneas", "Dispersión"])

        fig, ax = plt.subplots()
        if tipo == "Barras":
            ax.bar(df[col_x], df[col_y])
        elif tipo == "Líneas":
            ax.plot(df[col_x], df[col_y])
        elif tipo == "Dispersión":
            ax.scatter(df[col_x], df[col_y])

        ax.set_xlabel(col_x)
        ax.set_ylabel(col_y)
        ax.set_title(f"{tipo} entre {col_x} y {col_y}")
        st.pyplot(fig)

with tabs[8]:
    st.header("💬 Chat con Epidemiología 101")
    if "messages" not in st.session_state:
        st.session_state["messages"] = [{
            "role": "system",
            "content": "Eres un docente experto en epidemiología. Explica conceptos y resuelve preguntas con claridad y evidencia."
        }]
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])
    if prompt := st.chat_input("Haz tu pregunta de epidemiología..."):
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)
        with st.chat_message("assistant"):
            try:
                response = openai.chat.completions.create(
                    model="gpt-3.5-turbo",
                    messages=st.session_state.messages
                )
                reply = response.choices[0].message.content
                st.markdown(reply)
                st.session_state.messages.append({"role": "assistant", "content": reply})
            except Exception as e:
                st.error(f"Error al comunicarse con OpenAI: {e}")
