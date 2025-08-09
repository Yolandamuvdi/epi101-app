import streamlit as st
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import math
import os

# Import Gemini client (Google Generative AI)
try:
    import google.generativeai as genai
    GENAI_AVAILABLE = True
except Exception:
    GENAI_AVAILABLE = False

# SciPy para tests estadísticos
try:
    from scipy.stats import chi2_contingency, fisher_exact, norm
    SCIPY_AVAILABLE = True
except Exception:
    SCIPY_AVAILABLE = False

# -------------------- CONFIGURACIÓN GENERAL --------------------
st.set_page_config(page_title="🧠 Epidemiología 101 - Masterclass", page_icon="🧪", layout="wide", initial_sidebar_state="expanded")

# Estilos CSS (limpio, accesible, responsive)
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

# --- Variables y configuraciones ---
SECCIONES = {
    "🏠 Inicio": "inicio",
    "📌 Conceptos Básicos": "conceptos",
    "📈 Medidas de Asociación": "medidas",
    "📊 Diseños de Estudio": "disenos",
    "⚠️ Sesgos y Errores": "sesgos",
    "📚 Glosario Interactivo": "glosario",
    "🧪 Ejercicios Prácticos": "ejercicios",
    "📊 Tablas 2x2 y Cálculos": "tablas",
    "🎥 Videos Educativos": "videos",
    "🎮 Gamificación": "gamificacion",
    "🤖 Chat IA Epidemiológica": "chat",
}

# --- Inicializar sesión ---
if "seccion_actual" not in st.session_state:
    st.session_state.seccion_actual = None

# -------------------- FUNCIONES DE CARGA --------------------
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

# -------------------- CÁLCULOS EPIDEMIOLÓGICOS --------------------
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
    if not SCIPY_AVAILABLE:
        return None, "scipy no disponible"
    table = np.array([[a,b],[c,d]])
    chi2, p, dof, expected = chi2_contingency(table, correction=False)
    if (expected < 5).any():
        _, p = fisher_exact(table)
        test_used = "Fisher exact test"
    else:
        test_used = "Chi-cuadrado sin corrección"
    return p, test_used

# -------------------- INTERPRETACIÓN --------------------
def interpretar_resultados(rr, rr_l, rr_u, or_, or_l, or_u, rd, rd_l, rd_u, p_val, test_name):
    texto = f"""
**Resultados Epidemiológicos:**

- Riesgo Relativo (RR): {rr:.3f} (IC95% {rr_l:.3f} - {rr_u:.3f})  
- Odds Ratio (OR): {or_:.3f} (IC95% {or_l:.3f} - {or_u:.3f})  
- Diferencia de Riesgos (RD): {rd:.3f} (IC95% {rd_l:.3f} - {rd_u:.3f})  
- Valor p ({test_name}): {p_val:.4f}  

"""
    if p_val is not None and p_val < 0.05:
        texto += "🎯 La asociación es estadísticamente significativa (p < 0.05)."
    elif p_val is not None:
        texto += "⚠️ No se encontró asociación estadísticamente significativa (p ≥ 0.05)."
    else:
        texto += "⚠️ No se pudo calcular valor p (scipy no disponible)."
    return texto

# -------------------- GRÁFICOS --------------------
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
    labels = ["Casos expuestos (a)", "No casos expuestos (b)", "Casos no expuestos (c)", "No casos no expuestos (d)"]
    valores = [a,b,c,d]
    colores = ['#0d3b66', '#3e5c76', '#82a0bc', '#b0c4de']
    fig, ax = plt.subplots()
    ax.bar(labels, valores, color=colores)
    ax.set_ylabel("Conteo")
    ax.set_title("Distribución de exposición y casos")
    plt.xticks(rotation=15)
    st.pyplot(fig, use_container_width=True)

# -------------------- GAMIFICACIÓN --------------------
def mostrar_insignia(tipo):
    insignias = {
        "inicio": "🎓 Bienvenida a Epidemiología 101. ¡Empecemos la aventura científica! 🧬",
        "ejercicio_correcto": "🏅 ¡Genial! Has desbloqueado una insignia por responder correctamente. Sigue así 🔥",
        "completo": "🌟 ¡Felicidades! Has completado todos los ejercicios y desbloqueado el certificado digital. 📜"
    }
    msg = insignias.get(tipo, "🎉 ¡Bien hecho!")
    st.toast(msg, icon="🎉")

# -------------------- CHAT GEMINI --------------------
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

# -------------------- CONTENIDO PRINCIPAL --------------------

if st.session_state.seccion_actual is None:
    # Landing
    st.markdown(
        """
        <h1 style="text-align:center; font-size:4rem; color:#0d3b66; font-weight:900; user-select:none;">
        Bienvenido/a a <br> Epidemiología 101
        </h1>
        <h3 style="text-align:center; margin-top:-10px; font-weight:600; color:#144a91; user-select:none;">
        Hecho por Yolanda Muvdi
        </h3>
        <p style="text-align:center; font-size:1.5rem; color:#0d3b66; margin-top:40px; user-select:none;">
        ¿Qué quieres aprender hoy?
        </p>
        """, unsafe_allow_html=True
    )
    opcion = st.selectbox(
        "Selecciona la sección:",
        options=list(SECCIONES.keys()),
        index=0,
        label_visibility="collapsed",
        help="Elige una sección para comenzar",
    )
    if st.button("Ir a la sección"):
        st.session_state.seccion_actual = SECCIONES[opcion]
        st.experimental_rerun()

else:
    # Sidebar y contenido
    with st.sidebar:
        st.markdown("<h2 style='color:#0d3b66; font-weight:900;'>🧪 Epidemiología 101</h2>", unsafe_allow_html=True)
        if st.button("🏠 Volver al inicio"):
            st.session_state.seccion_actual = None
            st.experimental_rerun()
        st.markdown("---")
        for nombre, key in SECCIONES.items():
            if key == "inicio":
                continue
            style = "font-weight:bold; color:#0d3b66;" if st.session_state.seccion_actual == key else "color:#0d3b66;"
            if st.button(f"{nombre}", key=key):
                st.session_state.seccion_actual = key
                st.experimental_rerun()

    # Mostrar contenido por sección
    if st.session_state.seccion_actual == "conceptos":
        st.header("📌 Conceptos Básicos")
        contenido = cargar_md("contenido/conceptosbasicos.md")
        if contenido:
            st.markdown(contenido)
        else:
            st.info("Agrega el archivo 'contenido/conceptosbasicos.md' para mostrar el contenido.")

    elif st.session_state.seccion_actual == "medidas":
        st.header("📈 Medidas de Asociación")
        contenido = cargar_md("contenido/medidas_completas.md")
        if contenido:
            st.markdown(contenido)
        else:
            st.info("Agrega el archivo 'contenido/medidas_completas.md' para mostrar el contenido.")

    elif st.session_state.seccion_actual == "disenos":
        st.header("📊 Diseños de Estudio")
        contenido = cargar_md("contenido/disenos_completos.md")
        if contenido:
            st.markdown(contenido)
        else:
            st.info("Agrega el archivo 'contenido/disenos_completos.md' para mostrar el contenido.")

    elif st.session_state.seccion_actual == "sesgos":
        st.header("⚠️ Sesgos y Errores")
        contenido = cargar_md("contenido/sesgos_completos.md")
        if contenido:
            st.markdown(contenido)
        else:
            st.info("Agrega el archivo 'contenido/sesgos_completos.md' para mostrar el contenido.")

    elif st.session_state.seccion_actual == "glosario":
        st.header("📚 Glosario Interactivo")
        glosario = cargar_py_variable("contenido/glosario_completo.py", "glosario")
        if glosario:
            for termino, definicion in glosario.items():
                with st.expander(termino):
                    st.write(definicion)
        else:
            st.info("Agrega 'contenido/glosario_completo.py' con variable `glosario`.")

    elif st.session_state.seccion_actual == "ejercicios":
        st.header("🧪 Ejercicios Prácticos")
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
            if respuestas_correctas == len(preguntas) and len(preguntas) > 0:
                mostrar_insignia("completo")
        else:
            st.info("Agrega 'contenido/ejercicios_completos.py' con variable `preguntas`.")

    elif st.session_state.seccion_actual == "tablas":
        st.header("📊 Tablas 2x2 y Cálculos Epidemiológicos")

        if "a" not in st.session_state:
            st.session_state.a = 10
        if "b" not in st.session_state:
            st.session_state.b = 20
        if "c" not in st.session_state:
            st.session_state.c = 5
        if "d" not in st.session_state:
            st.session_state.d = 40

        col1, col2 = st.columns(2)
        with col1:
            st.session_state.a = st.number_input("Casos expuestos (a)", min_value=0, value=st.session_state.a, step=1, key="input_a")
            st.session_state.b = st.number_input("No casos expuestos (b)", min_value=0, value=st.session_state.b, step=1, key="input_b")
        with col2:
            st.session_state.c = st.number_input("Casos no expuestos (c)", min_value=0, value=st.session_state.c, step=1, key="input_c")
            st.session_state.d = st.number_input("No casos no expuestos (d)", min_value=0, value=st.session_state.d, step=1, key="input_d")

        if st.button("Calcular medidas y mostrar gráficos"):
            a, b, c, d = st.session_state.a, st.session_state.b, st.session_state.c, st.session_state.d
            total = a+b+c+d
            if total == 0:
                st.error("Por favor ingresa valores mayores que cero.")
            else:
                a_, b_, c_, d_, corregido = corregir_ceros(a,b,c,d)
                rr, rr_l, rr_u = ic_riesgo_relativo(a_,b_,c_,d_)
                or_, or_l, or_u = ic_odds_ratio(a_,b_,c_,d_)
                rd, rd_l, rd_u = diferencia_riesgos(a_,b_,c_,d_)
                p_val, test_name = calcular_p_valor(int(a_), int(b_), int(c_), int(d_))

                st.markdown(interpretar_resultados(rr, rr_l, rr_u, or_, or_l, or_u, rd, rd_l, rd_u, p_val, test_name))

                plot_forest(rr, rr_l, rr_u, or_, or_l, or_u)
                plot_barras_expuestos(a, b, c, d)

                if corregido:
                    st.warning("Se aplicó corrección de 0.5 para evitar ceros en la tabla.")

    elif st.session_state.seccion_actual == "videos":
        st.header("🎥 Videos Educativos")
        st.info("Próximamente: Videos explicativos para reforzar conceptos epidemiológicos.")

    elif st.session_state.seccion_actual == "gamificacion":
        st.header("🎮 Gamificación")
        st.info("Aquí podrías agregar juegos, quizzes interactivos y retos para aprender más divertido.")

    elif st.session_state.seccion_actual == "chat":
        st.header("🤖 Chat IA Epidemiológica")
        st.write("Haz preguntas sobre epidemiología, cálculos, análisis y más.")

        if "chat_messages" not in st.session_state:
            st.session_state.chat_messages = [{"role":"system","content":"Eres un experto epidemiólogo amable y claro."}]

        with st.form("chat_form", clear_on_submit=True):
            user_msg = st.text_area("Escribe tu pregunta o consulta epidemiológica aquí:", max_chars=512)
            submitted = st.form_submit_button("Enviar")

        if submitted and user_msg.strip():
            st.session_state.chat_messages.append({"role":"user","content":user_msg.strip()})
            respuesta = chat_with_gemini(st.session_state.chat_messages)
            st.session_state.chat_messages.append({"role":"assistant","content":respuesta})

        for msg in st.session_state.chat_messages[1:]:
            if msg["role"] == "user":
                st.markdown(f"**Tú:** {msg['content']}")
            else:
                st.markdown(f"**IA Epidemiológica:** {msg['content']}")

    else:
        st.info("Sección no encontrada. Por favor vuelve al inicio.")



