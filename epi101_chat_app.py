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
    fae = (rae/inc_exp) if inc_exp > 0 else np.nan
    fap = (rae / inc_noexp) if inc_noexp > 0 else np.nan

    # Cálculo p-valor de asociación (Chi2 o Fisher)
    if SCIPY_AVAILABLE:
        tabla = np.array([[a, b], [c, d]])
        try:
            chi2, p_val, _, _ = chi2_contingency(tabla)
        except Exception:
            p_val = np.nan
        # Fisher exact solo si tabla pequeña o Chi2 inválido
        if p_val > 0.05 and np.min(tabla) < 5:
            try:
                _, p_val_fisher = fisher_exact(tabla)
                p_val = p_val_fisher
            except Exception:
                pass
    else:
        p_val = np.nan

    # Alertas y recomendaciones
    alertas = []
    if corr:
        alertas.append("Se aplicó corrección de Haldane-Anscombe por valores cero en tabla.")
    if np.isnan(rr) or np.isnan(orr):
        alertas.append("Algunas medidas no pudieron calcularse correctamente por división por cero o datos insuficientes.")
    if p_val < 0.05:
        alertas.append("La asociación es estadísticamente significativa (p < 0.05).")
    else:
        alertas.append("La asociación no es estadísticamente significativa (p >= 0.05).")

    # Análisis de sensibilidad (opcional)
    analisis_sensibilidad = {}
    if not corr:
        analisis_sensibilidad["sin_correccion"] = True
    else:
        analisis_sensibilidad["sin_correccion"] = False

    # Empaquetar resultados
    resultados = {
        "RR": rr,
        "IC95_RR": ci95_rr,
        "OR": orr,
        "IC95_OR": ci95_or,
        "RD": rd,
        "IC95_RD": ci95_rd,
        "RAE": rae,
        "FAE": fae,
        "FAP": fap,
        "p_val": p_val,
        "alertas": alertas,
        "analisis_sensibilidad": analisis_sensibilidad
    }
    return resultados


def mostrar_resultados_tabla(resultados):
    """Muestra tabla resumen y alertas."""
    if resultados is None:
        st.warning("No hay resultados para mostrar.")
        return

    st.subheader("Resultados principales")
    data = {
        "Medida": ["Riesgo Relativo (RR)", "Odds Ratio (OR)", "Diferencia de Riesgos (RD)", "Riesgo atribuible en expuesto (RAE)",
                   "Fracción atribuible en expuesto (FAE)", "Fracción atribuible en población (FAP)", "Valor p (Asociación)"],
        "Valor": [resultados["RR"], resultados["OR"], resultados["RD"], resultados["RAE"], resultados["FAE"], resultados["FAP"], resultados["p_val"]],
        "IC 95%": [f"{resultados['IC95_RR'][0]:.3f} - {resultados['IC95_RR'][1]:.3f}" if all(np.isfinite(resultados['IC95_RR'])) else "N/A",
                   f"{resultados['IC95_OR'][0]:.3f} - {resultados['IC95_OR'][1]:.3f}" if all(np.isfinite(resultados['IC95_OR'])) else "N/A",
                   f"{resultados['IC95_RD'][0]:.3f} - {resultados['IC95_RD'][1]:.3f}" if all(np.isfinite(resultados['IC95_RD'])) else "N/A",
                   "-", "-", "-", "-"]
    }
    df = pd.DataFrame(data)
    st.table(df)

    st.subheader("Alertas y recomendaciones")
    for alerta in resultados.get("alertas", []):
        st.info(alerta)


def plot_forest_rr_or(rr, ci_rr, orr, ci_or):
    """Gráfico forest plot para RR y OR con sus IC 95%."""
    fig, ax = plt.subplots(figsize=(6, 3))
    medidas = ["Riesgo Relativo (RR)", "Odds Ratio (OR)"]
    valores = [rr, orr]
    ci_inf = [ci_rr[0], ci_or[0]]
    ci_sup = [ci_rr[1], ci_or[1]]
    y_pos = np.arange(len(medidas))

    for i, (val, inf, sup) in enumerate(zip(valores, ci_inf, ci_sup)):
        # Validar valores finitos para evitar error matplotlib
        if not (np.isfinite(val) and np.isfinite(inf) and np.isfinite(sup)):
            continue  # Ignorar si algún valor no es válido

        color = 'green' if val < 1 else 'red'
        ax.errorbar(val, y_pos[i], xerr=[[val - inf], [sup - val]],
                    fmt='o', color='black', ecolor=color, elinewidth=3, capsize=5)

    ax.axvline(x=1, color='grey', linestyle='--')
    ax.set_yticks(y_pos)
    ax.set_yticklabels(medidas)
    ax.set_xlabel("Medida (con IC 95%)")
    ax.set_title("Forest plot para RR y OR")
    ax.invert_yaxis()
    plt.tight_layout()
    return fig


# --------------------------
# Pestañas en Streamlit
# --------------------------

tab1, tab2, tab3 = st.tabs(["Conceptos básicos", "Tablas 2x2 y Cálculos", "Gráficos y visualización"])

with tab1:
    md_intro = cargar_md("assets/conceptos_basicos.md")
    st.markdown(md_intro)

with tab2:
    st.header("Ingreso de datos para tabla 2x2")
    col1, col2 = st.columns(2)

    with col1:
        a = st.number_input("a: Casos expuestos", min_value=0, value=10, step=1)
        b = st.number_input("b: No casos expuestos", min_value=0, value=20, step=1)

    with col2:
        c = st.number_input("c: Casos no expuestos", min_value=0, value=5, step=1)
        d = st.number_input("d: No casos no expuestos", min_value=0, value=40, step=1)

    valido, mensajes = validar_inputs(a, b, c, d)
    if not valido:
        for m in mensajes:
            st.warning(m)

    if valido:
        if st.button("Calcular medidas"):
            resultados = safe_execute(calcular_medidas, a, b, c, d)
            mostrar_resultados_tabla(resultados)
    else:
        st.info("Por favor ingresa valores válidos para calcular.")

with tab3:
    st.header("Visualización gráfica de RR y OR")
    # Necesitamos valores para graficar (pueden venir del tab2 o inputs manuales)
    rr_input = st.number_input("RR (Riesgo Relativo)", min_value=0.0, value=1.5, step=0.01, format="%.3f")
    rr_inf_input = st.number_input("Límite inferior IC RR", min_value=0.0, value=1.1, step=0.01, format="%.3f")
    rr_sup_input = st.number_input("Límite superior IC RR", min_value=0.0, value=2.0, step=0.01, format="%.3f")

    or_input = st.number_input("OR (Odds Ratio)", min_value=0.0, value=2.0, step=0.01, format="%.3f")
    or_inf_input = st.number_input("Límite inferior IC OR", min_value=0.0, value=1.2, step=0.01, format="%.3f")
    or_sup_input = st.number_input("Límite superior IC OR", min_value=0.0, value=3.3, step=0.01, format="%.3f")

    if st.button("Mostrar gráfico"):
        fig = safe_execute(plot_forest_rr_or,
                           rr_input, (rr_inf_input, rr_sup_input),
                           or_input, (or_inf_input, or_sup_input))
        if fig:
            st.pyplot(fig)


# --------------------------
# Chat Gemini - conversación (simple)
# --------------------------

if "chat_messages" not in st.session_state:
    st.session_state.chat_messages = []

st.header("💬 Chat con Epidemiólogo Virtual")

input_text = st.text_area("Escribe tu pregunta o duda:", height=100)

if st.button("Enviar"):
    if input_text.strip() != "":
        st.session_state.chat_messages.append({"role": "user", "content": input_text.strip()})
        respuesta = chat_with_gemini_messages(st.session_state.chat_messages)
        st.session_state.chat_messages.append({"role": "assistant", "content": respuesta})

if st.session_state.chat_messages:
    chat_container = st.container()
    with chat_container:
        for msg in st.session_state.chat_messages:
            if msg["role"] == "user":
                st.markdown(f"**Tú:** {msg['content']}")
            else:
                st.markdown(f"**Epidemiólogo:** {msg['content']}")


# --------------------------
# Footer
# --------------------------
st.markdown("---")
st.markdown("© 2025 Yolanda Muvdi - Todos los derechos reservados.")



