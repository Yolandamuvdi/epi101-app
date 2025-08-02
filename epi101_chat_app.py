import streamlit as st
import openai

st.set_page_config(page_title="Epidemiología 101", page_icon="🧪")
st.title("🧠 Epidemiología 101 - Asistente educativo")
st.markdown("🤓 Creado por Yolanda Muvdi. Aprende conceptos clave de epidemiología de forma clara y práctica.")

openai.api_key = st.secrets["OPENAI_API_KEY"]

if "messages" not in st.session_state:
    st.session_state["messages"] = [{"role": "system", "content": """
Eres un profesor universitario experto en epidemiología con vocación pedagógica.
Tu tarea es explicar conceptos clave de epidemiología (tasas, medidas de asociación, diseños de estudio, sesgos, interpretación de resultados), resolver ejercicios y generar recursos educativos adaptados al nivel del usuario.
Este GPT se llama 'Epidemiología 101' y fue creado por Yolanda Muvdi.
Sigue estos pasos:
1. Pregunta o identifica el nivel de conocimiento del usuario (básico, intermedio o avanzado).
2. Explica el concepto solicitado con claridad y precisión.
3. Incluye un ejemplo práctico.
4. Propón una pregunta o ejercicio relacionado.
5. Incluye glosario si se usan términos técnicos.
6. Cita fuentes si se usan datos concretos.
"""}]

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

if prompt := st.chat_input("Haz tu pregunta de epidemiología..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=st.session_state.messages
        )
        reply = response.choices[0].message.content
        st.markdown(reply)

    st.session_state.messages.append({"role": "assistant", "content": reply})
