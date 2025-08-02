
import streamlit as st
import openai

st.set_page_config(page_title="Epidemiología 101", page_icon="🧪", layout="wide")

# Estilo personalizado
st.markdown("""
    <style>
    body {
        background-color: #f4f6f9;
    }
    .block-container {
        padding-top: 2rem;
    }
    .title {
        font-size: 2.8em;
        font-weight: bold;
        color: #2c3e50;
    }
    .subtitle {
        font-size: 1.2em;
        color: #34495e;
        margin-bottom: 2rem;
    }
    </style>
""", unsafe_allow_html=True)

st.markdown('<div class="title">🧠 Epidemiología 101 - Asistente educativo</div>', unsafe_allow_html=True)
st.markdown('<div class="subtitle">🤓 Creado por Yolanda Muvdi. Aprende conceptos clave de epidemiología de forma clara, pro y con flow académico internacional.</div>', unsafe_allow_html=True)

# Verifica que la clave esté presente
if "OPENAI_API_KEY" in st.secrets:
    openai.api_key = st.secrets["OPENAI_API_KEY"]
    st.success("✅ Clave API detectada correctamente.")
else:
    st.error("❌ No se encontró OPENAI_API_KEY. Ve al panel de Secrets en Streamlit Cloud y agrégala.")
    st.stop()

# Inicializa el historial
if "messages" not in st.session_state:
    st.session_state["messages"] = [{
        "role": "system",
        "content": "Eres un profesor universitario experto en epidemiología con vocación pedagógica. Tu tarea es explicar conceptos clave de epidemiología (tasas, medidas de asociación, diseños de estudio, sesgos, interpretación de resultados), resolver ejercicios y generar recursos educativos adaptados al nivel del usuario. Este GPT se llama 'Epidemiología 101' y fue creado por Yolanda Muvdi."
    }]

# Mostrar historial
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# Entrada del usuario
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
            st.error(f"❌ Error al generar respuesta: {e}")
