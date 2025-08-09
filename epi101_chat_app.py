# epi101_chat_app.py
import streamlit as st
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import math
import os

# Gemini client (Google Generative AI)
try:
    import google.generativeai as genai
    GENAI_AVAILABLE = True
except Exception:
    GENAI_AVAILABLE = False

# SciPy para pruebas estadísticas y IC exactos
try:
    from scipy.stats import chi2_contingency, fisher_exact, norm
    SCIPY_AVAILABLE = True
except Exception:
    SCIPY_AVAILABLE = False


# --------------------------
# Funciones utilitarias y manejo de errores
# --------------------------

def safe_execute(func, *args, **kwargs):
    """Ejecuta función con manejo de errores amigable para usuario."""
    try:
        return func(*args, **kwargs)
    except Exception as e:
        st.error(f"Error inesperado: {e}")
        return None


@st.cache_data(show_spinner=False)
def cargar_md(ruta):
    """Carga archivo markdown para mostrar en pantalla."""
    try:
        with open(ruta, encoding="utf-8") as f:
            return f.read()
    except Exception as e:
        return f"Error cargando archivo: {e}"


@st.cache_data(show_spinner=False)
def cargar_py_variable(path_py, var_name):
    """Carga variable específica desde script .py"""
    namespace = {}
    try:
        with open(path_py, "r", encoding="utf-8") as f:
            code = f.read()
        exec(code, namespace)
        return namespace.get(var_name, None)
    except Exception:
        return None


# --------------------------
# Configuración base de Streamlit
# --------------------------

st.set_page_config(page_title="Epidemiología 101", page_icon="🧪", layout="wide")

st.markdown("""
    <link rel="icon" href="assets/logo.ico" type="image/x-icon" />
""", unsafe_allow_html=True)


# --------------------------
# Modo oscuro toggle con estilos
# --------------------------

modo_oscuro = st.sidebar.checkbox("Modo oscuro / Dark mode")

if modo_oscuro:
    st.markdown("""
    <style>
        body, .block-container {
            background-color: #1e1e1e;
            color: #cfcfcf;
            font-family: 'Montserrat', sans-serif;
            padding: 2rem 4rem;
            max-width: 1100px;
            margin: auto;
        }
        .block-container {
            background: #2a2a2a;
            border-radius: 12px;
            box-shadow: 0 10px 25px rgba(255,255,255,0.1);
            padding: 3rem 4rem 4rem 4rem;
        }
        h1, h2, h3, h4 {
            color: #8ab4f8;
            font-weight: 700;
            margin-bottom: 1rem;
        }
        .stButton > button {
            background-color: #3a66ff;
            color: white;
            border-radius: 6px;
            padding: 0.5rem 1.2rem;
            font-weight: 600;
            transition: background-color 0.3s ease;
        }
        .stButton > button:hover {
            background-color: #2555dd;
            cursor: pointer;
        }
        .stAlert > div {
            font-weight: 600;
            font-size: 1rem;
        }
        .success {
            color: #6ade6a !important;
        }
        .warning {
            color: #ffa500 !important;
        }
        .info {
            color: #1e90ff !important;
        }
        .title {
            font-size: 3rem !important;
            font-weight: 800 !important;
            margin-bottom: 0.4rem !important;
        }
        .subtitle {
            font-size: 1.3rem !important;
            color: #a0b9f5 !important;
            margin-bottom: 1.5rem !important;
            font-weight: 500 !important;
        }
        #chat-container {
            max-height: 400px;
            overflow-y: auto;
            padding-right: 1rem;
            border: 1px solid #444;
            border-radius: 8px;
            background: #212121;
            margin-bottom: 1rem;
        }
    </style>
    """, unsafe_allow_html=True)
else:
    st.markdown("""
    <link href="https://fonts.googleapis.com/css2?family=Montserrat:wght@400;700&display=swap" rel="stylesheet">
    <style>
        body, .block-container {
            background-color: #f4f7fa;
            color: #0d3b66;
            font-family: 'Montserrat', sans-serif;
            padding: 2rem 4rem;
            max-width: 1100px;
            margin: auto;
        }
        .block-container {
            background: white;
            border-radius: 12px;
            box-shadow: 0 10px 25px rgba(0,0,0,0.1);
            padding: 3rem 4rem 4rem 4rem;
        }
        h1, h2, h3, h4 {
            color: #0d3b66;
            font-weight: 700;
            margin-bottom: 1rem;
        }
        .stButton > button {
            background-color: #0d3b66;
            color: white;
            border-radius: 6px;
            padding: 0.5rem 1.2rem;
            font-weight: 600;
            transition: background-color 0.3s ease;
        }
        .stButton > button:hover {
            background-color: #074060;
            cursor: pointer;
        }
        .stAlert > div {
            font-weight: 600;
            font-size: 1rem;
        }
        .success {
            color: #2d7a2d !important;
        }
        .warning {
            color: #e65c00 !important;
        }
        .info {
            color: #1e90ff !important;
        }
        .title {
            font-size: 3rem !important;
            font-weight: 800 !important;
            margin-bottom: 0.4rem !important;
        }
        .subtitle {
            font-size: 1.3rem !important;
            color: #3e5c76 !important;
            margin-bottom: 1.5rem !important;
            font-weight: 500 !important;
        }
        #chat-container {
            max-height: 400px;
            overflow-y: auto;
            padding-right: 1rem;
            border: 1px solid #ccc;
            border-radius: 8px;
            background: #f9fbfd;
            margin-bottom: 1rem;
        }
    </style>
    """, unsafe_allow_html=True)


# --------------------------
# Página de bienvenida / landing (se muestra solo la primera vez)
# --------------------------
if "landing_visited" not in st.session_state:
    st.session_state["landing_visited"] = True
    st.title("🧠 Bienvenido a Epidemiología 101")
    st.markdown(
        """
        Esta plataforma te ayudará a dominar conceptos clave y cálculos epidemiológicos.
        Navega por las pestañas para comenzar a aprender de forma práctica y didáctica.

        > Creada y diseñada por Yolanda Muvdi, Enfermera Epidemióloga.

        ---
        """
    )
    st.stop()


# --------------------------
# Títulos principales de la app
# --------------------------
st.markdown('<div class="title">🧠 Epidemiología 101 - Asistente educativo</div>', unsafe_allow_html=True)
st.markdown('<div class="subtitle">Plataforma para aprender epidemiología, creada por Yolanda Muvdi.</div>', unsafe_allow_html=True)


# --------------------------
# Configuración Gemini para chat
# --------------------------
GEMINI_KEY = st.secrets.get("GEMINI_API_KEY") or os.environ.get("GEMINI_API_KEY")

if not GENAI_AVAILABLE:
    st.warning("La librería `google-generativeai` no está instalada. El chat no estará disponible.")
elif not GEMINI_KEY:
    st.warning("❌ No se encontró GEMINI_API_KEY en secrets o variables de entorno. El chat no funcionará.")
else:
    try:
        genai.configure(api_key=GEMINI_KEY)
    except Exception as e:
        st.warning(f"Advertencia al configurar Gemini: {e}")


def chat_with_gemini_messages(messages):
    """Comunicación con modelo Gemini para chat."""
    if not GENAI_AVAILABLE:
        return "⚠ La librería google-generativeai no está disponible en este entorno."
    if not GEMINI_KEY:
        return "⚠ No hay GEMINI_API_KEY configurada."

    convo = []
    for m in messages:
        role = m.get("role", "user")
        content = m.get("content", "")
        convo.append(f"[{role.upper()}]\n{content}")
    prompt = "\n\n".join(convo) + "\n\n[ASSISTANT]\nResponde de forma clara y concisa, con tono didáctico."

    try:
        model = genai.GenerativeModel("gemini-2.5-pro")
        response = model.generate_content(prompt)
        text = getattr(response, "text", None)
        if not text:
            if hasattr(response, "candidates") and response.candidates:
                parts = getattr(response.candidates[0], "content", None)
                if hasattr(parts, "parts"):
                    text = "".join([p.text for p in parts.parts if hasattr(p, "text")])
                else:
                    text = str(parts)
            else:
                text = str(response)
        return text
    except Exception as e:
        return f"⚠ Error en la conexión con Gemini: {e}"


# --------------------------
# Funciones para análisis epidemiológico con validaciones, ICs, p-valores y alertas
# --------------------------

def validar_inputs(a, b, c, d):
    """Valida que inputs sean enteros positivos y filas no tengan suma cero."""
    mensajes = []
    valid = True

    for val, nombre in zip([a,b,c,d], ["a", "b", "c", "d"]):
        if not isinstance(val, (int, np.integer)) or val < 0:
            mensajes.append(f"El valor '{nombre}' debe ser un entero positivo.")
            valid = False

    if (a + b) == 0:
        mensajes.append("La suma de Casos (a+b) no puede ser cero.")
        valid = False
    if (c + d) == 0:
        mensajes.append("La suma de Controles (c+d) no puede ser cero.")
        valid = False
    return valid, mensajes


def aplicar_correccion_haldane_anscombe(a, b, c, d):
    """Corrige ceros en tabla 2x2 con 0.5 para evitar división por cero en cálculos."""
    if 0 in [a,b,c,d]:
        return a+0.5, b+0.5, c+0.5, d+0.5, True
    else:
        return a, b, c, d, False


def calcular_medidas(a, b, c, d):
    """Calcula RR, OR, RD, IC95%, p-valor, alertas y análisis de sensibilidad."""

    # Corrección
    a_adj, b_adj, c_adj, d_adj, corr = aplicar_correccion_haldane_anscombe(a, b, c, d)

    # Incidencias
    inc_exp = a_adj / (a_adj + b_adj)
    inc_noexp = c_adj / (c_adj + d_adj)

    # Riesgo Relativo y CI95%
    rr = inc_exp / inc_noexp if inc_noexp > 0 else np.nan
    se_log_rr = math.sqrt((1/a_adj - 1/(a_adj+b_adj)) + (1/c_adj - 1/(c_adj+d_adj)))
    if rr > 0 and not np.isnan(se_log_rr):
        ci95_rr = (math.exp(math.log(rr) - 1.96 * se_log_rr), math.exp(math.log(rr) + 1.96 * se_log_rr))
    else:
        ci95_rr = (np.nan, np.nan)

    # Odds Ratio y CI95%
    orr = (a_adj * d_adj) / (b_adj * c_adj) if (b_adj * c_adj) > 0 else np.nan
    se_log_or = math.sqrt(1/a_adj + 1/b_adj + 1/c_adj + 1/d_adj)
    if orr > 0 and not np.isnan(se_log_or):
        ci95_or = (math.exp(math.log(orr) - 1.96 * se_log_or), math.exp(math.log(orr) + 1.96 * se_log_or))
    else:
        ci95_or = (np.nan, np.nan)

    # Riesgo Atribuible y fracciones
    rd = inc_exp - inc_noexp
    se_rd = math.sqrt((inc_exp*(1-inc_exp)/(a_adj+b_adj)) + (inc_noexp*(1-inc_noexp)/(c_adj+d_adj)))
    ci95_rd = (rd - 1.96*se_rd, rd + 1.96*se_rd)
    rae = rd
    fae = (rae/inc_exp) if inc_exp != 0 else None
    pexp = (a + b) / (a + b + c + d)
    rap = pexp * rd
    ipop = (a + c) / (a + b + c + d) if (a + b + c + d) > 0 else None
    fap = (rap / ipop) if (ipop is not None and ipop != 0) else None
    nnt = None if rd == 0 else 1 / abs(rd)

    # Pruebas estadísticas si scipy disponible
    chi2, p_chi2, p_fisher = None, None, None
    if SCIPY_AVAILABLE:
        tabla_chi = np.array([[a, b], [c, d]])
        try:
            chi2, p_chi2, _, ex = chi2_contingency(tabla_chi)
            if (ex < 5).any():
                _, p_fisher = fisher_exact(tabla_chi)
        except Exception:
            pass

    # Alertas y recomendaciones
    total = a+b+c+d
    alertas = []
    if total < 30:
        alertas.append("⚠️ Tamaño muestral total pequeño (<30). Resultados con poca potencia estadística.")
    if corr:
        alertas.append("⚠️ Se aplicó corrección Haldane-Anscombe por ceros en la tabla.")
    if p_chi2 is not None and p_chi2 < 0.05:
        alertas.append(f"✅ Asociación estadísticamente significativa (Chi2 p={p_chi2:.3f}).")
    if p_fisher is not None and p_fisher < 0.05:
        alertas.append(f"✅ Asociación significativa (Test exacto de Fisher p={p_fisher:.3f}).")

    return {
        "rr": rr, "ci95_rr": ci95_rr,
        "or": orr, "ci95_or": ci95_or,
        "rd": rd, "ci95_rd": ci95_rd,
        "rae": rae, "fae": fae,
        "rap": rap, "fap": fap,
        "nnt": nnt,
        "chi2": chi2, "p_chi2": p_chi2,
        "p_fisher": p_fisher,
        "alertas": alertas
    }


def interpretar_medidas(medidas):
    """Texto explicativo para interpretación básica de medidas epidemiológicas."""
    textos = []
    rr = medidas["rr"]
    orr = medidas["or"]
    rd = medidas["rd"]

    if rr > 1:
        textos.append(f"El Riesgo Relativo (RR) > 1 indica mayor riesgo en expuestos (RR={rr:.2f}).")
    elif rr < 1:
        textos.append(f"El RR < 1 sugiere un efecto protector (RR={rr:.2f}).")
    else:
        textos.append("RR = 1 no indica asociación.")

    if orr > 1:
        textos.append(f"La Odds Ratio (OR) > 1 indica asociación positiva (OR={orr:.2f}).")
    elif orr < 1:
        textos.append(f"La OR < 1 indica posible efecto protector (OR={orr:.2f}).")
    else:
        textos.append("OR = 1 no indica asociación.")

    textos.append(f"La Diferencia de Riesgos (RD) es {rd:.3f}, que representa la diferencia absoluta en riesgo entre grupos.")
    return "\n".join(textos)


# --------------------------
# Función para plot forest plot RR y OR con IC95%
# --------------------------

def plot_forest_rr_or(rr, ci_rr, orr, ci_or):
    """Grafica forest plot simple para RR y OR con IC95% y colores semánticos."""

    fig, ax = plt.subplots(figsize=(6, 3))

    medidas = ["Riesgo Relativo (RR)", "Odds Ratio (OR)"]
    valores = [rr, orr]
    ci_inf = [ci_rr[0], ci_or[0]]
    ci_sup = [ci_rr[1], ci_or[1]]
    colores = ['green' if v < 1 else 'red' for v in valores]

    y_pos = np.arange(len(medidas))

    ax.errorbar(valores, y_pos, xerr=[np.array(valores) - np.array(ci_inf), np.array(ci_sup) - np.array(valores)],
                fmt='o', color='black', ecolor=colores, elinewidth=3, capsize=5)
    ax.axvline(x=1, color='grey', linestyle='--')

    ax.set_yticks(y_pos)
    ax.set_yticklabels(medidas)
    ax.set_xlabel("Medida (con IC 95%)")
    ax.set_title("Forest plot para RR y OR")
    ax.invert_yaxis()
    plt.tight_layout()
    return fig


# --------------------------
# Guardar y cargar estado para tabla 2x2 y resultados
# --------------------------

if "tabla_2x2" not in st.session_state:
    st.session_state["tabla_2x2"] = {"a": 15, "b": 25, "c": 5, "d": 30}  # valores ejemplo
if "resultados" not in st.session_state:
    st.session_state["resultados"] = None


# --------------------------
# Layout principal con pestañas
# --------------------------

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

# ---- TAB 0: Conceptos Básicos ----
with tabs[0]:
    st.header("📌 Conceptos Básicos")
    contenido = safe_execute(cargar_md, "contenido/conceptosbasicos.md")
    if contenido is None or contenido.startswith("Error cargando"):
        st.write("Aquí iría el contenido de 'Conceptos Básicos'. Coloca el archivo 'contenido/conceptosbasicos.md'.")
    else:
        st.markdown(contenido)

# ---- TAB 1: Medidas de Asociación ----
with tabs[1]:
    st.header("📈 Medidas de Asociación")
    contenido = safe_execute(cargar_md, "contenido/medidas_completas.md")
    if contenido is None or contenido.startswith("Error cargando"):
        st.write("Aquí iría el contenido de 'Medidas de Asociación'. Coloca el archivo 'contenido/medidas_completas.md'.")
    else:
        st.markdown(contenido)

# ---- TAB 2: Diseños de Estudio ----
with tabs[2]:
    st.header("📊 Diseños de Estudio")
    contenido = safe_execute(cargar_md, "contenido/disenos_completos.md")
    if contenido is None or contenido.startswith("Error cargando"):
        st.write("Aquí iría el contenido de 'Diseños de Estudio'. Coloca el archivo 'contenido/disenos_completos.md'.")
    else:
        st.markdown(contenido)

# ---- TAB 3: Sesgos y Errores ----
with tabs[3]:
    st.header("⚠️ Sesgos y Errores")
    contenido = safe_execute(cargar_md, "contenido/sesgos_completos.md")
    if contenido is None or contenido.startswith("Error cargando"):
        st.write("Aquí iría el contenido de 'Sesgos y Errores'. Coloca el archivo 'contenido/sesgos_completos.md'.")
    else:
        st.markdown(contenido)

    st.markdown("### 💡 Tips sobre sesgos en medidas epidemiológicas")
    st.markdown(
        """
        - El Riesgo Relativo (RR) puede estar afectado por sesgos de selección si los grupos no son comparables.
        - La Odds Ratio (OR) tiende a sobreestimar el riesgo relativo en enfermedades comunes.
        - Siempre validar diseño de estudio y controles para minimizar sesgos.
        """
    )

# ---- TAB 4: Glosario Interactivo ----
with tabs[4]:
    st.header("📚 Glosario Interactivo")
    contenido = safe_execute(cargar_md, "contenido/glosario.md")
    if contenido is None or contenido.startswith("Error cargando"):
        st.write("Aquí iría el contenido del Glosario. Coloca el archivo 'contenido/glosario.md'.")
    else:
        st.markdown(contenido)

# ---- TAB 5: Ejercicios Prácticos ----
with tabs[5]:
    st.header("📝 Ejercicios Prácticos")
    contenido = safe_execute(cargar_md, "contenido/ejercicios.md")
    if contenido is None or contenido.startswith("Error cargando"):
        st.write("Aquí irían ejercicios prácticos. Coloca el archivo 'contenido/ejercicios.md'.")
    else:
        st.markdown(contenido)

# ---- TAB 6: Tablas 2x2 y cálculos ----
with tabs[6]:
    st.header("🔢 Tablas 2x2 y Cálculos Epidemiológicos")

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Introduce los datos de la tabla 2x2")
        a = st.number_input("Casos con exposición (a)", min_value=0, value=st.session_state["tabla_2x2"]["a"], step=1)
        b = st.number_input("Casos sin exposición (b)", min_value=0, value=st.session_state["tabla_2x2"]["b"], step=1)
        c = st.number_input("Controles con exposición (c)", min_value=0, value=st.session_state["tabla_2x2"]["c"], step=1)
        d = st.number_input("Controles sin exposición (d)", min_value=0, value=st.session_state["tabla_2x2"]["d"], step=1)

        # Validar inputs
        valid, mensajes = validar_inputs(a, b, c, d)

        if mensajes:
            for msg in mensajes:
                st.warning(msg)

        # Guardar tabla actualizada
        st.session_state["tabla_2x2"] = {"a": a, "b": b, "c": c, "d": d}

    with col2:
        if valid:
            if st.button("Calcular medidas"):
                st.session_state["resultados"] = calcular_medidas(a, b, c, d)
        else:
            st.button("Calcular medidas", disabled=True)

        if st.session_state["resultados"]:
            res = st.session_state["resultados"]
            st.markdown("### Resultados:")

            st.write(f"• Riesgo Relativo (RR): {res['rr']:.3f} (IC95%: {res['ci95_rr'][0]:.3f} - {res['ci95_rr'][1]:.3f})")
            st.write(f"• Odds Ratio (OR): {res['or']:.3f} (IC95%: {res['ci95_or'][0]:.3f} - {res['ci95_or'][1]:.3f})")
            st.write(f"• Diferencia de Riesgos (RD): {res['rd']:.3f} (IC95%: {res['ci95_rd'][0]:.3f} - {res['ci95_rd'][1]:.3f})")

            if res["fae"] is not None:
                st.write(f"• Fracción atribuible en expuestos (FAE): {res['fae']:.3f}")
            if res["fap"] is not None:
                st.write(f"• Fracción atribuible en población (FAP): {res['fap']:.3f}")
            if res["nnt"] is not None:
                st.write(f"• Número necesario a tratar (NNT): {abs(res['nnt']):.1f}")

            # Alertas
            for alert in res["alertas"]:
                st.info(alert)

            # Interpretación breve
            st.markdown("#### Interpretación básica")
            st.write(interpretar_medidas(res))

            # Mostrar gráfico forest plot
            fig = plot_forest_rr_or(res["rr"], res["ci95_rr"], res["or"], res["ci95_or"])
            st.pyplot(fig)

# ---- TAB 7: Visualización de Datos ----
with tabs[7]:
    st.header("📊 Visualización de Datos")
    contenido = safe_execute(cargar_md, "contenido/visualizacion.md")
    if contenido is None or contenido.startswith("Error cargando"):
        st.write("Aquí iría el contenido para Visualización de Datos. Coloca el archivo 'contenido/visualizacion.md'.")
    else:
        st.markdown(contenido)

# ---- TAB 8: Chat IA ----
with tabs[8]:
    st.header("💬 Chat de Epidemiología 101")

    if "chat_history" not in st.session_state:
        st.session_state["chat_history"] = [
            {"role": "assistant", "content": "¡Hola! Soy tu asistente de Epidemiología 101. ¿En qué te puedo ayudar hoy?"}
        ]

    chat_container = st.container()

    with chat_container:
        for chat_msg in st.session_state["chat_history"]:
            role = chat_msg["role"]
            content = chat_msg["content"]
            if role == "user":
                st.markdown(f"<p style='color:#0d3b66;font-weight:bold;'>Tú:</p><p>{content}</p>", unsafe_allow_html=True)
            else:
                st.markdown(f"<p style='color:#3a66ff;font-weight:bold;'>Asistente:</p><p>{content}</p>", unsafe_allow_html=True)

    user_input = st.text_area("Escribe tu pregunta aquí...", key="chat_input")

    if st.button("Enviar", disabled=(not user_input.strip())):
        st.session_state["chat_history"].append({"role": "user", "content": user_input})
        with st.spinner("Consultando al asistente..."):
            respuesta = chat_with_gemini_messages(st.session_state["chat_history"])
        st.session_state["chat_history"].append({"role": "assistant", "content": respuesta})
        st.experimental_rerun()  # Para actualizar chat instantáneamente

# --------------------------
# Referencias fijas (pueden ir en sidebar o footer)
# --------------------------
st.sidebar.markdown("## 📚 Referencias y bibliografía")
st.sidebar.markdown(
    """
    - Rothman KJ, Greenland S, Lash TL. Modern Epidemiology, 3rd Ed.
    - Kleinbaum DG et al. Epidemiologic Research: Principles and Quantitative Methods.
    - Altman DG et al. Statistics with Confidence.
    """
)

