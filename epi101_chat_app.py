import streamlit as st
import openai

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

# Cabecera
descripcion = (
    "Desarrollado por Yolanda Muvdi, Enfermera Epidemióloga. Plataforma de formación en conceptos clave de epidemiología y salud pública, "
    "con enfoque pedagógico y base científica sólida."
)

st.markdown('<div class="title">🧠 Epidemiología 101 - Asistente educativo</div>', unsafe_allow_html=True)
st.markdown(f'<div class="subtitle">{descripcion}</div>', unsafe_allow_html=True)

# Verificación de API
if "OPENAI_API_KEY" in st.secrets:
    openai.api_key = st.secrets["OPENAI_API_KEY"]
    st.success("✅ Clave API detectada correctamente.")
else:
    st.error("❌ No se encontró OPENAI_API_KEY. Ve al panel de Secrets en Streamlit Cloud y agrégala.")
    st.stop()

# Pestañas
tabs = st.tabs([
    "Conceptos Básicos",
    "Medidas de Asociación",
    "Diseños de Estudio",
    "Sesgos y Errores",
    "Glosario Interactivo",
    "Ejercicios Prácticos",
    "Tablas 2x2 y Cálculos",
    "Chat"
])

with tabs[0]:
    st.header("📌 Conceptos Básicos de Epidemiología")
    st.markdown("""
    - **Epidemiología:** Ciencia que estudia la distribución y los determinantes de los eventos de salud en poblaciones.
    - **Población:** Conjunto de individuos que comparten una característica en común.
    - **Evento de interés:** Enfermedad, condición o suceso relacionado con la salud.
    - **Frecuencia:** Cuántas veces ocurre un evento en un grupo.
    - **Prevalencia:** Proporción de personas que tienen una enfermedad en un momento dado.
    - **Incidencia:** Número de casos nuevos de una enfermedad en un periodo específico.
    """)

with tabs[1]:
    st.header("📈 Medidas de Asociación")
    st.markdown("""
    - **Riesgo Relativo (RR):** Compara el riesgo de enfermar entre dos grupos.
    - **Razón de Momios (OR):** Se usa en estudios de casos y controles.
    - **Reducción del Riesgo Absoluto (RRA):** Diferencia entre dos tasas de incidencia.
    - **Número Necesario a Tratar (NNT):** Cuántas personas necesitas tratar para prevenir un caso.
    """)

with tabs[2]:
    st.header("📊 Diseños de Estudio Epidemiológico")
    st.markdown("""
    - **Estudio Transversal:** Mide exposición y enfermedad al mismo tiempo.
    - **Casos y Controles:** Compara un grupo con la enfermedad vs. uno sin ella. Calcula OR.
    - **Cohorte:** Sigue a personas expuestas y no expuestas. Calcula RR.
    - **Ensayo Clínico:** Intervención para probar tratamientos. Alta validez interna.
    """)

with tabs[3]:
    st.header("⚠️ Sesgos y Errores")
    st.markdown("""
    - **Sesgo:** Error sistemático que afecta la validez de los resultados.
    - **Sesgo de selección:** Diferencias entre quienes entran o no al estudio.
    - **Sesgo de información:** Errores en la medición.
    - **Confusión:** Otro factor afecta la asociación.
    - **Errores aleatorios:** Se minimizan aumentando el tamaño de muestra.
    """)

with tabs[4]:
    st.header("📚 Glosario Interactivo")
    for termino, definicion in {
        "Incidencia": "Número de casos nuevos en una población en riesgo durante un periodo de tiempo.",
        "Prevalencia": "Proporción de individuos con la enfermedad en un momento o periodo determinado.",
        "Riesgo relativo": "Comparación del riesgo entre dos grupos.",
        "Odds ratio": "Medida de asociación utilizada en estudios de casos y controles.",
        "Tasa de mortalidad": "Número de muertes por cada unidad de población en un tiempo determinado."
    }.items():
        with st.expander(termino):
            st.write(definicion)

with tabs[5]:
    st.header("🧪 Ejercicios Prácticos")
    st.subheader("Pregunta 1")
    pregunta = st.radio("¿Cuál de las siguientes es una medida de frecuencia?", 
                        ["Odds ratio", "Riesgo relativo", "Prevalencia", "Sensibilidad"])
    if st.button("Verificar respuesta"):
        if pregunta == "Prevalencia":
            st.success("✅ ¡Correcto! La prevalencia es una medida de frecuencia.")
        else:
            st.error("❌ Incorrecto. La respuesta correcta es 'Prevalencia'.")

with tabs[6]:
    st.header("📊 Tablas 2x2 y Cálculos Epidemiológicos")
    st.markdown("Completa los datos de la tabla 2x2:")

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
                st.success(f"✅ Riesgo Relativo (RR): {rr:.2f}")
            else:
                st.warning("⚠️ No se pudo calcular el RR (división por cero)")

            if orr:
                st.success(f"✅ Odds Ratio (OR): {orr:.2f}")
            else:
                st.warning("⚠️ No se pudo calcular el OR (división por cero)")
        except:
            st.error("❌ Ocurrió un error en los cálculos")

with tabs[7]:
    st.header("💬 Chat con Epidemiología 101")

    if "messages" not in st.session_state:
        st.session_state["messages"] = [{
            "role": "system",
            "content": (
                "Eres un profesor universitario experto en epidemiología con vocación pedagógica. "
                "Tu tarea es explicar conceptos clave de epidemiología (tasas, medidas de asociación, "
                "diseños de estudio, sesgos, interpretación de resultados), resolver ejercicios y generar recursos educativos."
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

