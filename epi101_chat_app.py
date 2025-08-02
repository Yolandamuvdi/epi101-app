
import streamlit as st
import openai

st.set_page_config(page_title="Epidemiología 101", page_icon="🧪", layout="wide")

# Estilos visuales con flow
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

    div[data-testid="stChatMessage"] {
        border-radius: 12px;
        padding: 12px 18px;
        margin-bottom: 8px;
        box-shadow: 0 2px 8px rgba(0,0,0,0.08);
        font-size: 1rem;
        line-height: 1.5;
    }

    .stTextInput > div > input {
        border-radius: 10px !important;
        border: 1.5px solid #0d3b66 !important;
        padding: 12px 16px !important;
        font-size: 1rem !important;
    }
    </style>
""", unsafe_allow_html=True)

# Cabecera
st.markdown('<div class="title">🧠 Epidemiología 101 - Asistente educativo</div>', unsafe_allow_html=True)
st.markdown('<div class="subtitle">Desarrollado por Yolanda Muvdi, Enfermera Epidemióloga. Plataforma de formación en conceptos clave de epidemiología y salud pública, con enfoque pedagógico y base científica sólida.</div>', unsafe_allow_html=True)

# Verificación de API
if "OPENAI_API_KEY" in st.secrets:
    openai.api_key = st.secrets["OPENAI_API_KEY"]
    st.success("✅ Clave API detectada correctamente.")
else:
    st.error("❌ No se encontró OPENAI_API_KEY. Ve al panel de Secrets en Streamlit Cloud y agrégala.")
    st.stop()

# Pestañas de contenido
tabs = st.tabs([
    "Conceptos Básicos",
    "Medidas de Asociación",
    "Diseños de Estudio",
    "Sesgos y Errores",
    "Chat"
])

with tabs[0]:
    st.header("📌 Conceptos Básicos de Epidemiología")
    st.markdown("""
    - **Epidemiología:** Ciencia que estudia la distribución y los determinantes de los eventos de salud en poblaciones.
    - **Población:** Conjunto de individuos que comparten una característica en común (ej: edad, lugar de residencia).
    - **Evento de interés:** Enfermedad, condición o suceso relacionado con la salud (ej: diabetes, COVID-19, mortalidad).
    - **Frecuencia:** Cuántas veces ocurre un evento en un grupo.
    - **Tasa:** Medida que relaciona frecuencia con tiempo.
    - **Prevalencia:** Proporción de individuos que tienen una enfermedad en un momento dado.
    - **Incidencia:** Número de casos nuevos de una enfermedad en un periodo específico.
    """)

with tabs[1]:
    st.header("📈 Medidas de Asociación")
    st.markdown("""
    - **Riesgo Relativo (RR):** Compara el riesgo de enfermar entre dos grupos.
      - RR > 1: mayor riesgo en el grupo expuesto
      - RR < 1: el factor puede ser protector
    - **Razón de Momios (OR):** Se usa en estudios de casos y controles.
    - **Reducción del Riesgo Absoluto (RRA):** Diferencia entre dos tasas de incidencia.
    - **Número Necesario a Tratar (NNT):** Cuántas personas necesitas tratar para prevenir un caso.
    - **Interpretación:** Si OR = 2, el grupo expuesto tiene el doble de probabilidad de presentar el evento.
    """)

with tabs[2]:
    st.header("📊 Diseños de Estudio Epidemiológico")
    st.markdown("""
    - **Estudio Transversal:** Se mide exposición y enfermedad al mismo tiempo. Útil para prevalencia.
    - **Estudio de Casos y Controles:** Compara un grupo con la enfermedad vs. uno sin ella. Calcula OR.
    - **Estudio de Cohorte:** Sigue a personas expuestas y no expuestas en el tiempo. Calcula RR.
    - **Ensayo Clínico:** Intervención controlada para probar tratamientos. Alta validez interna.
    - **Ecológicos:** Comparan poblaciones, no individuos.
    """)

with tabs[3]:
    st.header("⚠️ Sesgos y Errores")
    st.markdown("""
    - **Sesgo:** Error sistemático que afecta la validez de los resultados.
        - **Sesgo de selección:** Diferencias entre quienes entran o no al estudio.
        - **Sesgo de información:** Errores en la medición o recolección de datos.
        - **Sesgo de memoria (recall bias):** Frecuente en estudios de casos y controles.
    - **Confusión:** Otro factor que afecta la asociación observada.
    - **Errores aleatorios:** Se minimizan aumentando el tamaño de muestra.
    - **Control del sesgo:** Aleatorización, enmascaramiento, ajustes estadísticos.
    """)

with tabs[4]:
    st.header("💬 Chat con Epidemiología 101")

    if "messages" not in st.session_state:
        st.session_state["messages"] = [{
            "role": "system",
            "content": (
                "Eres un profesor universitario experto en epidemiología con vocación pedagógica. "
                "Tu tarea es explicar conceptos clave de epidemiología (tasas, medidas de asociación, "
                "diseños de estudio, sesgos, interpretación de resultados), resolver ejercicios y generar recursos educativos adaptados al nivel del usuario. "
                "Este GPT se llama 'Epidemiología 101' y fue creado por Yolanda Muvdi."
            )
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

