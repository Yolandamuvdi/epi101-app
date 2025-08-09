import streamlit as st
import os
import math
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import chi2_contingency, fisher_exact, norm

# --- CONFIGURACIÓN GENERAL ---
st.set_page_config(page_title="🧠 Epidemiología 101 - Masterclass", page_icon="🧪", layout="wide", initial_sidebar_state="expanded")

# --- ESTILOS CSS ---
st.markdown("""
<style>
    body, .block-container {
        background: #fefefe;
        color: #0d3b66;
        font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
        line-height: 1.5;
    }
    .block-container {
        max-width: 1100px;
        margin: 2rem auto 4rem auto;
        padding: 2rem 3rem;
        background: white;
        border-radius: 12px;
        box-shadow: 0 12px 30px rgba(13,59,102,0.1);
    }
    h1, h2, h3, h4 {
        color: #0d3b66;
        font-weight: 700;
    }
    .stButton>button {
        background-color: #0d3b66;
        color: white;
        border-radius: 7px;
        padding: 10px 20px;
        font-weight: 700;
        font-size: 1.1rem;
        transition: background-color 0.3s ease;
    }
    .stButton>button:hover {
        background-color: #09466b;
        cursor: pointer;
    }
    a {
        color: #0d3b66;
        text-decoration: none;
    }
    a:hover {
        text-decoration: underline;
    }
    /* Botones grandes para móvil */
    @media (max-width: 768px) {
        .stButton>button {
            width: 100% !important;
            font-size: 1.2rem !important;
        }
    }
</style>
""", unsafe_allow_html=True)

# --- ICONOS DE SECCIONES ---
ICONOS = {
    "Conceptos Básicos": "📌",
    "Medidas de Asociación": "📈",
    "Diseños de Estudio": "📊",
    "Sesgos y Errores": "⚠️",
    "Glosario Interactivo": "📚",
    "Ejercicios Prácticos": "🧪",
    "Tablas 2x2 y Cálculos": "📊",
    "Visualización de Datos": "📊",
    "Multimedia YouTube": "🎥",
    "Gamificación": "🎯",
    "Chat Epidemiológico": "🤖",
}

# --- FUNCIONES PARA CARGA DE CONTENIDO ---
@st.cache_data(show_spinner=False)
def cargar_md(ruta):
    try:
        with open(ruta, encoding="utf-8") as f:
            return f.read()
    except Exception:
        return None

@st.cache_data(show_spinner=False)
def cargar_py_variable(ruta_py, var_name):
    ns = {}
    try:
        with open(ruta_py, encoding="utf-8") as f:
            exec(f.read(), ns)
        return ns.get(var_name)
    except Exception:
        return None

# --- CÁLCULOS EPIDEMIOLÓGICOS ---
def corregir_ceros(a,b,c,d):
    if 0 in [a,b,c,d]:
        return a+0.5, b+0.5, c+0.5, d+0.5, True
    return a,b,c,d, False

def ic_riesgo_relativo(a,b,c,d, alpha=0.05):
    risk1 = a / (a + b)
    risk2 = c / (c + d)
    rr = risk1 / risk2
    se_log_rr = math.sqrt(1/a - 1/(a+b) + 1/c - 1/(c+d))
    z = norm.ppf(1 - alpha/2)
    lower = math.exp(math.log(rr) - z * se_log_rr)
    upper = math.exp(math.log(rr) + z * se_log_rr)
    return rr, lower, upper

def ic_odds_ratio(a,b,c,d, alpha=0.05):
    or_ = (a*d)/(b*c)
    se_log_or = math.sqrt(1/a + 1/b + 1/c + 1/d)
    z = norm.ppf(1 - alpha/2)
    lower = math.exp(math.log(or_) - z * se_log_or)
    upper = math.exp(math.log(or_) + z * se_log_or)
    return or_, lower, upper

def diferencia_riesgos(a,b,c,d, alpha=0.05):
    risk1 = a / (a + b)
    risk2 = c / (c + d)
    rd = risk1 - risk2
    se_rd = math.sqrt((risk1*(1-risk1))/(a+b) + (risk2*(1-risk2))/(c+d))
    z = norm.ppf(1 - alpha/2)
    lower = rd - z*se_rd
    upper = rd + z*se_rd
    return rd, lower, upper

def calcular_p_valor(a,b,c,d):
    table = np.array([[a,b],[c,d]])
    chi2, p, dof, expected = chi2_contingency(table, correction=False)
    if (expected < 5).any():
        _, p = fisher_exact(table)
        test_used = "Fisher exact test"
    else:
        test_used = "Chi-cuadrado sin corrección"
    return p, test_used

# --- INTERPRETACIÓN RESULTADOS ---
def interpretar_resultados(rr, rr_l, rr_u, or_, or_l, or_u, rd, rd_l, rd_u, p_val, test_name):
    texto = f"""
*Resultados Epidemiológicos:*

• Riesgo Relativo (RR): {rr:.3f} (IC95% {rr_l:.3f} - {rr_u:.3f})  
• Odds Ratio (OR): {or_:.3f} (IC95% {or_l:.3f} - {or_u:.3f})  
• Diferencia de Riesgos (RD): {rd:.3f} (IC95% {rd_l:.3f} - {rd_u:.3f})  
• Valor p ({test_name}): {p_val:.4f}  
"""
    if p_val < 0.05:
        texto += "\n🎯 La asociación es estadísticamente significativa (p < 0.05)."
    else:
        texto += "\n⚠️ No se encontró asociación estadísticamente significativa (p ≥ 0.05)."
    return texto

# --- GRÁFICOS ---
def plot_forest(rr, rr_l, rr_u, or_, or_l, or_u):
    fig, ax = plt.subplots(figsize=(6,3))
    ax.errorbar(x=[rr, or_], y=[2,1], 
                xerr=[ [rr-rr_l, or_-or_l], [rr_u-rr, or_u-or_] ], 
                fmt='o', color='#0d3b66', capsize=5, markersize=10)
    ax.set_yticks([1,2])
    ax.set_yticklabels(["Odds Ratio (OR)", "Riesgo Relativo (RR)"])
    ax.axvline(1, color='gray', linestyle='--')
    ax.set_xlabel("Medidas de Asociación")
    ax.set_title("Intervalos de Confianza 95%")
    st.pyplot(fig, use_container_width=True)

def plot_barras_expuestos(a,b,c,d):
    labels = ["Casos expuestos", "No casos expuestos", "Casos no expuestos", "No casos no expuestos"]
    valores = [a,b,c,d]
    colores = ['#0d3b66', '#3e5c76', '#82a0bc', '#b0c4de']
    fig, ax = plt.subplots()
    ax.bar(labels, valores, color=colores)
    ax.set_ylabel("Conteo")
    ax.set_title("Distribución de exposición y casos")
    plt.xticks(rotation=15)
    st.pyplot(fig, use_container_width=True)

# --- GAMIFICACIÓN EXTENDIDA ---
def mostrar_insignia(tipo):
    insignias = {
        "inicio": "🎓 Bienvenida a Epidemiología 101. ¡Empecemos la aventura científica! 🧬",
        "ejercicio_correcto": "🏅 ¡Genial! Has desbloqueado una insignia por responder correctamente. Sigue así 🔥",
        "completo": "🌟 ¡Felicidades! Has completado todos los ejercicios y desbloqueado el certificado digital. 📜",
        "nivel_bajo": "📘 Sigue estudiando, el camino de la epidemiología es desafiante, ¡pero tú puedes!",
        "nivel_medio": "👍 Vas muy bien, casi eres un pro en epidemiología. ¡Sigue así!",
        "nivel_alto": "🏆 Eres un PRO en Epidemiología. ¡Eres la envidia del equipo!",
    }
    msg = insignias.get(tipo, "🎉 ¡Bien hecho!")
    st.toast(msg, icon="🎉")

def gamificacion():
    st.header("🎯 Gamificación")

    if "puntos" not in st.session_state:
        st.session_state.puntos = 0
    if "ejercicios_respondidos" not in st.session_state:
        st.session_state.ejercicios_respondidos = 0
    if "ejercicios_correctos" not in st.session_state:
        st.session_state.ejercicios_correctos = 0

    preguntas = cargar_py_variable("contenido/ejercicios_completos.py", "preguntas")
    if not preguntas:
        st.info("Agrega 'contenido/ejercicios_completos.py' con la variable 'preguntas' para activar gamificación.")
        return

    for i, q in enumerate(preguntas):
        st.subheader(f"Pregunta {i+1}")
        respuesta = st.radio(q['pregunta'], q['opciones'], key=f"q{i}")
        if st.button(f"Verificar respuesta {i+1}", key=f"btn_{i}"):
            st.session_state.ejercicios_respondidos += 1
            if respuesta == q['respuesta_correcta']:
                st.session_state.ejercicios_correctos += 1
                st.session_state.puntos += 10
                mostrar_insignia("ejercicio_correcto")
                st.success("✅ Correcto")
            else:
                st.error(f"❌ Incorrecto. La respuesta correcta es: {q['respuesta_correcta']}")

    if st.session_state.ejercicios_respondidos == len(preguntas) and len(preguntas) > 0:
        mostrar_insignia("completo")

    # Mostrar nivel
    puntos = st.session_state.puntos
    st.markdown(f"**Puntos acumulados:** {puntos}")

    if puntos < 30:
        mostrar_insignia("nivel_bajo")
    elif puntos < 70:
        mostrar_insignia("nivel_medio")
    else:
        mostrar_insignia("nivel_alto")

# --- CHAT CON GEMINI ---
try:
    import google.generativeai as genai
    GENAI_AVAILABLE = True
except ImportError:
    GENAI_AVAILABLE = False

GEMINI_KEY = st.secrets.get("GEMINI_API_KEY") or os.environ.get("GEMINI_API_KEY")
if GENAI_AVAILABLE and GEMINI_KEY:
    try:
        genai.configure(api_key=GEMINI_KEY)
    except Exception as e:
        st.warning(f"Error configurando Gemini: {e}")
else:
    if not GENAI_AVAILABLE:
        st.info("⚠️ Gemini no disponible: falta la librería google-generativeai.")
    elif not GEMINI_KEY:
        st.info("⚠️ No configurada GEMINI_API_KEY en secrets o entorno.")

def chat_with_gemini(messages):
    if not GENAI_AVAILABLE:
        return "⚠ La librería google-generativeai no está disponible."
    if not GEMINI_KEY:
        return "⚠ No hay GEMINI_API_KEY configurada."
    prompt = "\n\n".join([f"[{m['role'].upper()}]\n{m['content']}" for m in messages]) + "\n\n[ASSISTANT]\nResponde clara y didácticamente."
    try:
        model = genai.GenerativeModel("gemini-2.5-pro")
        response = model.generate_content(prompt)
        text = getattr(response, "text", None)
        if not text and hasattr(response, "candidates") and response.candidates:
            text = getattr(response.candidates[0], "content", str(response))
        return text or str(response)
    except Exception as e:
        return f"⚠ Error en Gemini: {e}"

def chat_epidemiologico():
    st.header(f"{ICONOS['Chat Epidemiológico']} Chat Epidemiológico con Gemini AI")
    st.info("Pregunta cualquier duda epidemiológica. Gemini responde clara, precisa y con humor inteligente.")

    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []

    pregunta = st.text_input("Escribe tu pregunta aquí:", key="chat_input")

    if st.button("Enviar"):
        if pregunta.strip():
            st.session_state.chat_history.append({"role":"user", "content": pregunta})
            with st.spinner("Gemini está respondiendo..."):
                respuesta = chat_with_gemini(st.session_state.chat_history)
            st.session_state.chat_history.append({"role":"assistant", "content": respuesta})

    for msg in st.session_state.chat_history:
        if msg["role"] == "user":
            st.markdown(f"*Tú:* {msg['content']}")
        else:
            st.markdown(f"*Gemini:* {msg['content']}")

# --- CONTENIDO MULTIMEDIA YOUTUBE ---
VIDEOS = {
    "Introducción a la Epidemiología": "https://www.youtube.com/embed/qVFP-IkyWgQ",
    "Medidas de Asociación": "https://www.youtube.com/embed/d61E24xvRfI",
    "Diseños de Estudios Epidemiológicos": "https://www.youtube.com/embed/y6odn6E8yRs",
    "Sesgos en Epidemiología": "https://www.youtube.com/embed/1kyFIyG37qc"
}

def multimedia_youtube():
    st.header("🎥 Videos Educativos de Epidemiología")
    st.info("Aquí puedes encontrar recursos audiovisuales para complementar tu aprendizaje.")

    for title, url in VIDEOS.items():
        st.subheader(title)
        st.video(url)

# --- FUNCIÓN PRINCIPAL ---
def main():
    if "seccion" not in st.session_state:
        st.session_state.seccion = None

    if st.session_state.seccion is None:
        # Pantalla inicial limpia
        st.title("🧠 Epidemiología 101")
        st.markdown("### ¿Qué quieres aprender hoy?")

        opciones = list(ICONOS.keys())
        opcion = st.selectbox(
            "Selecciona una sección",
            opciones,
            format_func=lambda x: f"{ICONOS[x]}  {x}",
            index=0
        )
        if st.button("Ir a la sección"):
            st.session_state.seccion = opcion
            st.experimental_rerun()
    else:
        # Sidebar con menú y opción para volver al inicio
        st.sidebar.title("Menú")
        opciones = list(ICONOS.keys())
        opcion = st.sidebar.selectbox(
            "Selecciona sección:",
            opciones,
            index=opciones.index(st.session_state.seccion),
            format_func=lambda x: f"{ICONOS[x]}  {x}"
        )

        st.sidebar.markdown("---")
        if st.sidebar.button("Volver al inicio"):
            st.session_state.seccion = None
            st.experimental_rerun()

        st.session_state.seccion = opcion

        # Mostrar contenido según sección
        st.header(f"{ICONOS[opcion]} {opcion}")

        if opcion == "Conceptos Básicos":
            contenido = cargar_md("contenido/conceptosbasicos.md")
            if contenido:
                st.markdown(contenido)
            else:
                st.info("Agrega el archivo 'contenido/conceptosbasicos.md' para mostrar el contenido.")

        elif opcion == "Medidas de Asociación":
            contenido = cargar_md("contenido/medidas_completas.md")
            if contenido:
                st.markdown(contenido)
            else:
                st.info("Agrega el archivo 'contenido/medidas_completas.md' para mostrar el contenido.")

        elif opcion == "Diseños de Estudio":
            contenido = cargar_md("contenido/disenos_completos.md")
            if contenido:
                st.markdown(contenido)
            else:
                st.info("Agrega el archivo 'contenido/disenos_completos.md' para mostrar el contenido.")

        elif opcion == "Sesgos y Errores":
            contenido = cargar_md("contenido/sesgos_completos.md")
            if contenido:
                st.markdown(contenido)
            else:
                st.info("Agrega el archivo 'contenido/sesgos_completos.md' para mostrar el contenido.")

        elif opcion == "Glosario Interactivo":
            glosario = cargar_py_variable("contenido/glosario_completo.py", "glosario")
            if glosario:
                for termino, definicion in glosario.items():
                    with st.expander(termino):
                        st.write(definicion)
            else:
                st.info("Agrega 'contenido/glosario_completo.py' con variable 'glosario'.")

        elif opcion == "Ejercicios Prácticos":
            preguntas = cargar_py_variable("contenido/ejercicios_completos.py", "preguntas")
            if preguntas:
                respuestas_correctas = 0
                for i, q in enumerate(preguntas):
                    st.subheader(f"Pregunta {i+1}")
                    respuesta = st.radio(q['pregunta'], q['opciones'], key=f"q{i}")
                    if st.button(f"Verificar respuesta {i+1}", key=f"btn_{i}"):
                        if respuesta == q['respuesta_correcta']:
                            st.success("✅ Correcto")
                            respuestas_correctas += 1
                            mostrar_insignia("ejercicio_correcto")
                        else:
                            st.error(f"❌ Incorrecto. La respuesta correcta es: {q['respuesta_correcta']}")
                st.markdown(f"**Respuestas correctas:** {respuestas_correctas} de {len(preguntas)}")
            else:
                st.info("Agrega 'contenido/ejercicios_completos.py' con variable 'preguntas'.")

        elif opcion == "Tablas 2x2 y Cálculos":
            st.subheader("Calculadora 2x2 - Medidas de Asociación")
            with st.form("form_2x2"):
                a = st.number_input("Casos expuestos (a)", min_value=0, value=10)
                b = st.number_input("No casos expuestos (b)", min_value=0, value=20)
                c = st.number_input("Casos no expuestos (c)", min_value=0, value=5)
                d = st.number_input("No casos no expuestos (d)", min_value=0, value=40)
                submitted = st.form_submit_button("Calcular")

            if submitted:
                a_, b_, c_, d_, corregido = corregir_ceros(a,b,c,d)
                if corregido:
                    st.warning("Se aplicó corrección de 0.5 para valores cero.")
                rr, rr_l, rr_u = ic_riesgo_relativo(a_, b_, c_, d_)
                or_, or_l, or_u = ic_odds_ratio(a_, b_, c_, d_)
                rd, rd_l, rd_u = diferencia_riesgos(a_, b_, c_, d_)
                p_val, test_name = calcular_p_valor(a_, b_, c_, d_)
                st.markdown(interpretar_resultados(rr, rr_l, rr_u, or_, or_l, or_u, rd, rd_l, rd_u, p_val, test_name))
                plot_forest(rr, rr_l, rr_u, or_, or_l, or_u)
                plot_barras_expuestos(a,b,c,d)

        elif opcion == "Visualización de Datos":
            st.info("Próximamente: gráficos interactivos y visualización avanzada.")

        elif opcion == "Multimedia YouTube":
            multimedia_youtube()

        elif opcion == "Gamificación":
            gamificacion()

        elif opcion == "Chat Epidemiológico":
            chat_epidemiologico()

        else:
            st.info("Sección en construcción.")

if __name__ == "__main__":
    main()
