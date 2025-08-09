import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import math
import os
from scipy.stats import chi2_contingency, fisher_exact, norm

# ----------- CONFIGURACIÓN GENERAL --------------
st.set_page_config(page_title="🧠 Epidemiología 101 - Masterclass", page_icon="🧪", layout="wide", initial_sidebar_state="expanded")

# ----------- ESTILOS CSS --------------
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
    /* Inicio centrado */
    .centered {
        display: flex;
        flex-direction: column;
        align-items: center;
        justify-content: center;
        height: 80vh;
        color: #0d3b66;
        font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
    }
    .titulo {
        font-size: 3rem;
        font-weight: 800;
        margin-bottom: 1rem;
    }
    .subtitulo {
        font-size: 1.5rem;
        margin-bottom: 2rem;
        font-weight: 600;
    }
    .stSelectbox > div > div > select {
        font-size: 1.25rem;
        padding: 10px;
        border-radius: 10px;
        border: 2px solid #0d3b66;
        width: 320px;
    }
</style>
""", unsafe_allow_html=True)

# ----------- MENU CON ICONOS --------------
menu_items = {
    "📌 Conceptos Básicos": "conceptos_basicos",
    "📈 Medidas de Asociación": "medidas_asociacion",
    "📊 Diseños de Estudio": "disenos_estudio",
    "⚠️ Sesgos y Errores": "sesgos_errores",
    "📚 Glosario Interactivo": "glosario_interactivo",
    "🧪 Ejercicios Prácticos": "ejercicios_practicos",
    "📊 Tablas 2x2 y Cálculos": "tablas_2x2",
    "📉 Visualización de Datos": "visualizacion_datos",
    "🎥 Multimedia YouTube": "multimedia_youtube",
    "🤖 Chat Epidemiológico": "chat_gemini",
    "🏆 Gamificación": "gamificacion"
}

# ----------- ESTADO INICIAL ---------
if "seccion_actual" not in st.session_state:
    st.session_state.seccion_actual = "Inicio"

# ----------- FUNCIONES DE CARGA MD/VARIABLES --------------
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

# ----------- FUNCIONES TABLAS 2x2, CÁLCULOS, INTERPRETACIONES --------------
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
    try:
        table = np.array([[a,b],[c,d]])
        chi2, p, dof, expected = chi2_contingency(table, correction=False)
        if (expected < 5).any():
            _, p = fisher_exact(table)
            test_used = "Fisher exact test"
        else:
            test_used = "Chi-cuadrado sin corrección"
        return p, test_used
    except Exception:
        return None, "Error en cálculo p-valor"

def interpretar_resultados(rr, rr_l, rr_u, or_, or_l, or_u, rd, rd_l, rd_u, p_val, test_name):
    texto = f"""
*Resultados Epidemiológicos:*

•⁠  ⁠Riesgo Relativo (RR): {rr:.3f} (IC95% {rr_l:.3f} - {rr_u:.3f})  
•⁠  ⁠Odds Ratio (OR): {or_:.3f} (IC95% {or_l:.3f} - {or_u:.3f})  
•⁠  ⁠Diferencia de Riesgos (RD): {rd:.3f} (IC95% {rd_l:.3f} - {rd_u:.3f})  
•⁠  ⁠Valor p ({test_name}): {p_val if p_val is not None else 'N/A'}  

"""
    if p_val is not None and p_val < 0.05:
        texto += "🎯 La asociación es estadísticamente significativa (p < 0.05)."
    elif p_val is not None:
        texto += "⚠️ No se encontró asociación estadísticamente significativa (p ≥ 0.05)."
    else:
        texto += "⚠️ No se pudo calcular valor p."
    return texto

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

# ----------- GAMIFICACIÓN --------------
def mostrar_insignia(tipo):
    insignias = {
        "inicio": "🎓 Bienvenida a Epidemiología 101. ¡Empecemos la aventura científica! 🧬",
        "ejercicio_correcto": "🏅 ¡Genial! Has desbloqueado una insignia por responder correctamente. Sigue así 🔥",
        "completo": "🌟 ¡Felicidades! Has completado todos los ejercicios y desbloqueado el certificado digital. 📜"
    }
    msg = insignias.get(tipo, "🎉 ¡Bien hecho!")
    st.toast(msg, icon="🎉")

# Gamificación avanzada con niveles
def gamificacion():
    st.subheader("Selecciona tu nivel actual en Epidemiología")
    niveles = ["Básico", "Intermedio", "Avanzado", "Experto/Messi"]
    nivel = st.selectbox("Nivel de epidemiología", niveles, key="nivel_gamificacion")
    
    preguntas_nivel = {
        "Básico": [
            {
                "pregunta": "¿Qué es la epidemiología?",
                "opciones": ["Estudio de animales", "Estudio de enfermedades en poblaciones", "Estudio de plantas"],
                "respuesta_correcta": "Estudio de enfermedades en poblaciones"
            },
            {
                "pregunta": "¿Qué mide la incidencia?",
                "opciones": ["Nuevos casos", "Casos totales", "Casos recuperados"],
                "respuesta_correcta": "Nuevos casos"
            }
        ],
        "Intermedio": [
            {
                "pregunta": "¿Qué es un Odds Ratio?",
                "opciones": ["Medida de asociación", "Tasa de mortalidad", "Variable de confusión"],
                "respuesta_correcta": "Medida de asociación"
            },
            {
                "pregunta": "¿Qué es un sesgo de selección?",
                "opciones": ["Error de medición", "Error en selección de participantes", "Error aleatorio"],
                "respuesta_correcta": "Error en selección de participantes"
            }
        ],
        "Avanzado": [
            {
                "pregunta": "¿Qué método se usa para ajustar confusores?",
                "opciones": ["Regresión logística", "T-test", "ANOVA"],
                "respuesta_correcta": "Regresión logística"
            },
            {
                "pregunta": "¿Qué es un diseño transversal?",
                "opciones": ["Estudio en un momento", "Estudio longitudinal", "Estudio experimental"],
                "respuesta_correcta": "Estudio en un momento"
            }
        ],
        "Experto/Messi": [
            {
                "pregunta": "¿Qué significa p<0.05 en análisis estadístico?",
                "opciones": ["No hay asociación", "Asociación significativa", "Error tipo II"],
                "respuesta_correcta": "Asociación significativa"
            },
            {
                "pregunta": "¿Qué es un efecto modificador?",
                "opciones": ["Variable que modifica asociación", "Variable dependiente", "Variable confusora"],
                "respuesta_correcta": "Variable que modifica asociación"
            }
        ]
    }
    
    preguntas = preguntas_nivel.get(nivel, [])
    correctas = 0
    for i, q in enumerate(preguntas):
        st.markdown(f"**Pregunta {i+1}:** {q['pregunta']}")
        resp = st.radio("", q['opciones'], key=f"gam_{nivel}_{i}")
        if st.button(f"Verificar Pregunta {i+1}", key=f"btn_gam_{nivel}_{i}"):
            if resp == q['respuesta_correcta']:
                st.success("✅ Correcto")
                correctas += 1
            else:
                st.error(f"❌ Incorrecto. La respuesta correcta es: {q['respuesta_correcta']}")

    if correctas == len(preguntas) and len(preguntas) > 0:
        st.balloons()
        st.success(f"¡Excelente, {nivel}! Eres un pro en Epidemiología 🎉")
    elif len(preguntas) > 0:
        st.info("Sigue estudiando para subir de nivel 🚀")

# ----------- CHAT GEMINI --------------
GENAI_AVAILABLE = False
try:
    import google.generativeai as genai
    GENAI_AVAILABLE = True
except Exception:
    pass

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

# ----------- PANTALLA INICIAL --------------
def pantalla_inicio():
    st.markdown(
        """
        <div class="centered">
            <div class="titulo">🧠 Epidemiología 101</div>
            <div class="subtitulo">¿Qué quieres aprender hoy?</div>
        </div>
        """, unsafe_allow_html=True)
    opcion = st.selectbox("", list(menu_items.keys()), key="inicio_dropdown")
    if opcion != "":
        st.session_state.seccion_actual = menu_items[opcion]
        st.experimental_rerun()

# ----------- BARRA LATERAL --------------
def barra_lateral():
    st.sidebar.title("🧪 Epidemiología 101")
    st.sidebar.markdown("""
    👩‍⚕️ Creado por Yolanda Muvdi, Enfermera Epidemióloga  
    📧 [ymuvdi@gmail.com](mailto:ymuvdi@gmail.com)  
    🔗 [LinkedIn](https://www.linkedin.com/in/yolanda-paola-muvdi-muvdi-778b73152/)
    """)
    opcion_sidebar = st.sidebar.radio("Selecciona una sección:", list(menu_items.keys()), index=list(menu_items.values()).index(st.session_state.seccion_actual))
    if menu_items[opcion_sidebar] != st.session_state.seccion_actual:
        st.session_state.seccion_actual = menu_items[opcion_sidebar]
        st.experimental_rerun()

# ----------- SECCIONES --------------
def mostrar_contenido():
    s = st.session_state.seccion_actual
    if s == "Inicio":
        pantalla_inicio()

    elif s == "conceptos_basicos":
        st.header("📌 Conceptos Básicos")
        st.markdown("""
        Ciencia que estudia la frecuencia, distribución y determinantes de las enfermedades en poblaciones humanas, y aplica este conocimiento al control de problemas de salud.
        """)
        st.video("https://www.youtube.com/watch?v=qVFP-IkyWgQ")

    elif s == "medidas_asociacion":
        st.header("📈 Medidas de Asociación")
        st.markdown("Explicación de medidas de asociación en epidemiología.")
        st.video("https://www.youtube.com/watch?v=d61E24xvRfI")

    elif s == "disenos_estudio":
        st.header("📊 Diseños de Estudio")
        st.markdown("Descripción de diseños de estudio epidemiológicos.")
        st.video("https://www.youtube.com/watch?v=y6odn6E8yRs")

    elif s == "sesgos_errores":
        st.header("⚠️ Sesgos y Errores")
        st.markdown("Tipos y ejemplos de sesgos en epidemiología.")
        st.video("https://www.youtube.com/watch?v=1kyFIyG37qc")

    elif s == "glosario_interactivo":
        st.header("📚 Glosario Interactivo")
        st.markdown("Aquí irá tu glosario interactivo con términos epidemiológicos.")

    elif s == "ejercicios_practicos":
        st.header("🧪 Ejercicios Prácticos")
        st.markdown("Ejercicios para practicar tus conocimientos.")

    elif s == "tablas_2x2":
        st.header("📊 Tablas 2x2 y Cálculos")
        st.markdown("Calcula medidas de asociación desde tablas 2x2.")

        col1, col2 = st.columns(2)
        with col1:
            a = st.number_input("Casos Expuestos (a)", min_value=0, value=10)
            b = st.number_input("No Casos Expuestos (b)", min_value=0, value=20)
        with col2:
            c = st.number_input("Casos No Expuestos (c)", min_value=0, value=5)
            d = st.number_input("No Casos No Expuestos (d)", min_value=0, value=30)

        if st.button("Calcular Medidas de Asociación"):
            a_corr, b_corr, c_corr, d_corr, corregido = corregir_ceros(a,b,c,d)
            rr, rr_l, rr_u = ic_riesgo_relativo(a_corr,b_corr,c_corr,d_corr)
            or_, or_l, or_u = ic_odds_ratio(a_corr,b_corr,c_corr,d_corr)
            rd, rd_l, rd_u = diferencia_riesgos(a_corr,b_corr,c_corr,d_corr)
            p_val, test_name = calcular_p_valor(a_corr,b_corr,c_corr,d_corr)

            st.markdown(interpretar_resultados(rr, rr_l, rr_u, or_, or_l, or_u, rd, rd_l, rd_u, p_val, test_name))
            plot_forest(rr, rr_l, rr_u, or_, or_l, or_u)
            plot_barras_expuestos(a,b,c,d)

    elif s == "visualizacion_datos":
        st.header("📉 Visualización de Datos")
        st.markdown("Próximamente: herramientas para crear gráficos interactivos.")

    elif s == "multimedia_youtube":
        st.header("🎥 Multimedia YouTube")
        st.markdown("""
        - Introducción a la Epidemiología: [Video](https://www.youtube.com/watch?v=qVFP-IkyWgQ)  
        - Medidas de Asociación: [Video](https://www.youtube.com/watch?v=d61E24xvRfI)  
        - Diseños de Estudio: [Video](https://www.youtube.com/watch?v=y6odn6E8yRs)  
        - Sesgos y Errores: [Video](https://www.youtube.com/watch?v=1kyFIyG37qc)
        """)

    elif s == "chat_gemini":
        st.header("🤖 Chat Epidemiológico con Gemini")
        if not GENAI_AVAILABLE or not GEMINI_KEY:
            st.warning("⚠️ Gemini no está configurado o no está disponible.")
            return
        chat_history = st.session_state.get("chat_history", [])
        if "input_text" not in st.session_state:
            st.session_state.input_text = ""

        with st.form("form_gemini", clear_on_submit=True):
            user_input = st.text_area("Pregunta a Gemini:", height=100)
            submit = st.form_submit_button("Enviar")
            if submit and user_input.strip():
                chat_history.append({"role": "user", "content": user_input})
                respuesta = chat_with_gemini(chat_history)
                chat_history.append({"role": "assistant", "content": respuesta})
                st.session_state.chat_history = chat_history

        for msg in chat_history:
            if msg["role"] == "user":
                st.markdown(f"**Tú:** {msg['content']}")
            else:
                st.markdown(f"**Gemini:** {msg['content']}")

    elif s == "gamificacion":
        st.header("🏆 Gamificación")
        gamificacion()

# ----------- MAIN ---------
def main():
    if st.session_state.seccion_actual == "Inicio":
        pantalla_inicio()
    else:
        barra_lateral()
        mostrar_contenido()

if __name__ == "__main__":
    main()

