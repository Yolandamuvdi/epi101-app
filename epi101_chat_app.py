import streamlit as st
import openai

st.set_page_config(page_title="Epidemiología 101", page_icon="🧪", layout="wide")

# Estilos personalizados CSS para toda la app
st.markdown("""
    <style>
    /* Fondo degradado suave */
    body, .block-container {
        background: linear-gradient(135deg, #f0f4f8 0%, #d9e2ec 100%);
        color: #1b2838;
        font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
    }

    /* Contenedor de la app con padding y sombra */
    .block-container {
        padding: 2rem 4rem 3rem 4rem;
        max-width: 1100px;
        margin: auto;
        background: white;
        border-radius: 12px;
        box-shadow: 0 10px 25px rgba(0,0,0,0.1);
    }

    /* Título grande y azul oscuro */
    .title {
        font-size: 3rem;
        font-weight: 900;
        color: #0d3b66;
        margin-bottom: 0.2rem;
    }

    /* Subtítulo más pequeño, gris azulado */
    .subtitle {
        font-size: 1.3rem;
        color: #3e5c76;
        margin-bottom: 2rem;
        font-weight: 500;
    }

    /* Pestañas personalizadas */
    .css-1offfwp e1fqkh3o3 {  /* Clase interna, puede cambiar pero funciona ahora */
        background-color: #e3f2fd !important;
        border-radius: 10px !important;
        padding: 0.5rem 1.2rem !important;
        color: #0d3b66 !important;
        font-weight: 700 !important;
    }

    /* Texto dentro de pestañas activo */
    .css-1offfwp.e1fqkh3o3[aria-selected="true"] {
        background-color: #0d3b66 !important;
        color: white !important;
        box-shadow: 0 4px 15px rgba(13,59,102,0.3);
    }

    /* Mensajes de chat con fondo y sombra */
    div[data-testid="stChatMessage"] {
        border-radius: 12px;
        padding: 12px 18px;
        margin-bottom: 8px;
        box-shadow: 0 2px 8px rgba(0,0,0,0.08);
        font-size: 1rem;
        line-height: 1.5;
    }

    /* Input chat */
    .stTextInput > div > input {
        border-radius: 10px !important;
        border: 1.5px solid #0d3b66 !important;
        padding: 12px 16px !important;
        font-size: 1rem !important;
    }

    /* Botones */
    button[kind="primary"] {
        background-color: #0d3b66 !important;
        border: none !important;
        padding: 10px 24px !important;
        border-radius: 10px !important;
        font-weight: 700 !important;
        font-size: 1.1rem !important;
        transition: background-color 0.3s ease;
    }
    button[kind="primary"]:hover {
        background-color: #0b2f55 !important;
    }

    /* Scrollbar estilizado para pestañas */
    ::-webkit-scrollbar {
        width: 10px;
    }
    ::-webkit-scrollbar-track {
        background: #e3f2fd;
        border-radius: 12px;
    }
    ::-webkit-scrollbar-thumb {
        background: #0d3b66;
        border-radius: 12px;
    }
    </style>
""", unsafe_allow_html=True)

# Cabecera
st.markdown('<div class="title">🧠 Epidemiología 101 - Asistente educativo</div>', unsafe_allow_html=True)
st.markdown('<div class="subtitle">🤓 Creado por Yolanda Muvdi. Aprende conceptos clave de epidemiología de forma clara, pro y con flow académico internacional.</div>', unsafe_allow_html=True)

# API Key
if "OPENAI_API_KEY" in st.secrets:
    openai.api_key = st.secrets["OPENAI_API_KEY"]
    st.success("✅ Clave API detectada correctamente.")
else:
    st.error("❌ No se encontró OPENAI_API_KEY. Ve al panel de Secrets en Streamlit Cloud y agrégala.")
    st.stop()

tabs = st.tabs(["Conceptos Básicos", "Medidas de Asociación", "Diseños de Estudio", "Sesgos y Errores", "Chat"])

with tabs[0]:
    st.header("Conceptos Básicos")
    st.write("Aquí explicamos qué es la epidemiología, tipos de variables, población, etc.")

with tabs[1]:
    st.header("Medidas de Asociación")
    st.write("Exploramos odds ratio, riesgo relativo, razones de momios y más.")

with tabs[2]:
    st.header("Diseños de Estudio")
    st.write("Estudio transversal, cohortes, casos y controles, ensayos clínicos.")

with tabs[3]:
    st.header("Sesgos y Errores")
    st.write("Aprende a identificar y corregir sesgos en tus estudios.")

with tabs[4]:
    st.header("Chat con Epidemiología 101")
    
    if "messages" not in st.session_state:
        st.session_state["messages"] = [{
            "role": "system",
            "content": ("Eres un profesor universitario experto en epidemiología con vocación pedagógica. "
                        "Tu tarea es explicar conceptos clave de epidemiología (tasas, medidas de asociación, "
                        "diseños de estudio, sesgos, interpretación de resultados), resolver ejercicios y generar recursos educativos adaptados al nivel del usuario. "
                        "Este GPT se llama 'Epidemiología 101' y fue creado por Yolanda Muvdi.")
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
                response = openai.ChatCompletion.create(
                    model="gpt-3.5-turbo",
                    messages=st.session_state.messages
                )
                reply = response.choices[0].message.content
                st.markdown(reply)
                st.session_state.messages.append({"role": "assistant", "content": reply})
            except Exception as e:
                st.error(f"Error al comunicarse con OpenAI: {e}")
