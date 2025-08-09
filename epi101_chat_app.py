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

# SciPy para pruebas estadísticas
try:
    from scipy.stats import chi2_contingency, fisher_exact
    SCIPY_AVAILABLE = True
except Exception:
    SCIPY_AVAILABLE = False


# --------- Manejo de errores centralizado ---------
def safe_execute(func, *args, **kwargs):
    try:
        return func(*args, **kwargs)
    except Exception as e:
        st.error(f"Error inesperado: {e}")
        return None


# --------- Carga cacheada de archivos ---------
@st.cache_data(show_spinner=False)
def cargar_md(ruta):
    try:
        with open(ruta, encoding="utf-8") as f:
            return f.read()
    except Exception as e:
        return f"Error cargando archivo: {e}"


@st.cache_data(show_spinner=False)
def cargar_py_variable(path_py, var_name):
    namespace = {}
    try:
        with open(path_py, "r", encoding="utf-8") as f:
            code = f.read()
        exec(code, namespace)
        return namespace.get(var_name, None)
    except Exception:
        return None


# --------- Configuración básica Streamlit ---------
st.set_page_config(page_title="Epidemiología 101", page_icon="🧪", layout="wide")

# --------- Branding: favicon (sube tu logo a assets/logo.ico) ---------
st.markdown("""
    <link rel="icon" href="assets/logo.ico" type="image/x-icon" />
""", unsafe_allow_html=True)


# --------- Modo oscuro toggle ---------
modo_oscuro = st.sidebar.checkbox("Modo oscuro")

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
    # Estilo claro (default)
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


# --------- Página de inicio / landing (solo la primera vez) ---------
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


# --------- Títulos principales ---------
st.markdown('<div class="title">🧠 Epidemiología 101 - Asistente educativo</div>', unsafe_allow_html=True)
st.markdown('<div class="subtitle">Plataforma para aprender epidemiología, creada por Yolanda Muvdi.</div>', unsafe_allow_html=True)


# --------- Configuración Gemini (segura) ---------
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


# --------- Función chat con Gemini ---------
def chat_with_gemini_messages(messages):
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


# --------- Pestañas principales ---------
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


# --- TAB 0: Conceptos Básicos (lazy load + safe)
with tabs[0]:
    st.header("📌 Conceptos Básicos de Epidemiología")
    contenido = safe_execute(cargar_md, "contenido/conceptosbasicos.md")
    if contenido is None or contenido.startswith("Error cargando"):
        st.write("Aquí iría el contenido de 'Conceptos Básicos'. Agrega 'contenido/conceptosbasicos.md' para mostrarlo.")
    else:
        st.markdown(contenido)


# --- TAB 1: Medidas de Asociación
with tabs[1]:
    st.header("📈 Medidas de Asociación")
    contenido = safe_execute(cargar_md, "contenido/medidas_completas.md")
    if contenido is None or contenido.startswith("Error cargando"):
        st.write("Aquí iría 'Medidas de Asociación'. Agrega 'contenido/medidas_completas.md'.")
    else:
        st.markdown(contenido)


# --- TAB 2: Diseños de Estudio
with tabs[2]:
    st.header("📊 Diseños de Estudio Epidemiológico")
    contenido = safe_execute(cargar_md, "contenido/disenos_completos.md")
    if contenido is None or contenido.startswith("Error cargando"):
        st.write("Aquí iría 'Diseños de Estudio'. Agrega 'contenido/disenos_completos.md'.")
    else:
        st.markdown(contenido)


# --- TAB 3: Sesgos y Errores
with tabs[3]:
    st.header("⚠️ Sesgos y Errores")
    contenido = safe_execute(cargar_md, "contenido/sesgos_completos.md")
    if contenido is None or contenido.startswith("Error cargando"):
        st.write("Aquí iría 'Sesgos y Errores'. Agrega 'contenido/sesgos_completos.md'.")
    else:
        st.markdown(contenido)


# --- TAB 4: Glosario Interactivo
with tabs[4]:
    st.header("📚 Glosario Interactivo A–Z")
    glosario = safe_execute(cargar_py_variable, "contenido/glosario_completo.py", "glosario")
    if glosario is None:
        st.write("No se pudo cargar el glosario. Agrega 'contenido/glosario_completo.py' con la variable `glosario`.")
    else:
        for termino, definicion in glosario.items():
            with st.expander(termino):
                st.write(definicion)


# --- TAB 5: Ejercicios Prácticos ---
with tabs[5]:
    st.header("🧪 Ejercicios Prácticos")
    preguntas = safe_execute(cargar_py_variable, "contenido/ejercicios_completos.py", "preguntas")
    if preguntas is None:
        st.write("No se pudieron cargar los ejercicios. Agrega 'contenido/ejercicios_completos.py' con la variable `preguntas`.")
    else:
        for i, q in enumerate(preguntas):
            st.subheader(f"Pregunta {i+1}")
            respuesta = st.radio(q['pregunta'], q['opciones'], key=f"pregunta_{i}")
            if st.button(f"Verificar {i+1}", key=f"verificar_{i}"):
                if respuesta == q['respuesta_correcta']:
                    st.success("✅ Correcto")
                else:
                    st.error(f"❌ Incorrecto. Respuesta correcta: {q['respuesta_correcta']}")


# --- TAB 6: Tablas 2x2 y cálculos ---
with tabs[6]:
    st.header("📊 Tablas 2x2 y Cálculos Epidemiológicos")

    if st.button("📌 Cargar ejemplo"):
        st.session_state["a_val"] = 30
        st.session_state["b_val"] = 70
        st.session_state["c_val"] = 10
        st.session_state["d_val"] = 90

    a_default = st.session_state.get("a_val", 0)
    b_default = st.session_state.get("b_val", 0)
    c_default = st.session_state.get("c_val", 0)
    d_default = st.session_state.get("d_val", 0)

    col1, col2 = st.columns(2)
    with col1:
        a = st.number_input("Casos con exposición (a)", min_value=0, step=1, value=a_default)
        b = st.number_input("Casos sin exposición (b)", min_value=0, step=1, value=b_default)
    with col2:
        c = st.number_input("Controles con exposición (c)", min_value=0, step=1, value=c_default)
        d = st.number_input("Controles sin exposición (d)", min_value=0, step=1, value=d_default)

    # Validaciones robustas antes de cálculo
    if a <= 0 or c <= 0:
        st.warning("⚠️ 'Casos con exposición (a)' y 'Controles con exposición (c)' deben ser mayores que 0.")
    elif (a + b) == 0 or (c + d) == 0:
        st.warning("⚠️ Las filas de Casos o Controles no pueden tener suma cero.")
    else:
        if st.button("Calcular medidas"):
            with st.spinner("Calculando medidas..."):
                try:
                    a0, b0, c0, d0 = int(a), int(b), int(c), int(d)

                    # Corrección Haldane-Anscombe
                    a_adj, b_adj, c_adj, d_adj = a0, b0, c0, d0
                    if 0 in [a0, b0, c0, d0]:
                        a_adj += 0.5
                        b_adj += 0.5
                        c_adj += 0.5
                        d_adj += 0.5
                        st.info("⚠️ Se aplicó corrección de Haldane-Anscombe por ceros.")

                    inc_exp = a_adj / (a_adj + b_adj)
                    inc_noexp = c_adj / (c_adj + d_adj)

                    rr = inc_exp / inc_noexp if inc_noexp > 0 else np.nan
                    se_log_rr = math.sqrt((1 / a_adj - 1 / (a_adj + b_adj)) + (1 / c_adj - 1 / (c_adj + d_adj)))
                    if rr > 0 and not np.isnan(se_log_rr):
                        ci95_rr = (math.exp(math.log(rr) - 1.96 * se_log_rr), math.exp(math.log(rr) + 1.96 * se_log_rr))
                    else:
                        ci95_rr = (np.nan, np.nan)

                    orr = (a_adj * d_adj) / (b_adj * c_adj) if (b_adj * c_adj) > 0 else np.nan
                    se_log_or = math.sqrt(1 / a_adj + 1 / b_adj + 1 / c_adj + 1 / d_adj)
                    if orr > 0 and not np.isnan(se_log_or):
                        ci95_or = (math.exp(math.log(orr) - 1.96 * se_log_or), math.exp(math.log(orr) + 1.96 * se_log_or))
                    else:
                        ci95_or = (np.nan, np.nan)

                    rd = inc_exp - inc_noexp
                    se_rd = math.sqrt((inc_exp * (1 - inc_exp) / (a_adj + b_adj)) + (inc_noexp * (1 - inc_noexp) / (c_adj + d_adj)))
                    ci95_rd = (rd - 1.96 * se_rd, rd + 1.96 * se_rd)

                    rae = rd
                    fae = (rae / inc_exp) if inc_exp != 0 else None
                    pexp = (a0 + b0) / (a0 + b0 + c0 + d0)
                    rap = pexp * rd
                    ipop = (a0 + c0) / (a0 + b0 + c0 + d0) if (a0 + b0 + c0 + d0) > 0 else None
                    fap = (rap / ipop) if (ipop is not None and ipop != 0) else None
                    nnt = None if rd == 0 else 1 / abs(rd)

                    st.markdown("**Tabla 2x2:**")
                    tabla = pd.DataFrame({
                        "Expuestos": [a0, c0],
                        "No Expuestos": [b0, d0]
                    }, index=["Casos", "Controles"])
                    st.dataframe(tabla)

                    st.markdown("### Resultados:")
                    st.write(f"• Riesgo relativo (RR): {rr:.3f} (IC 95%: {ci95_rr[0]:.3f} - {ci95_rr[1]:.3f})")
                    st.write(f"• Odds ratio (OR): {orr:.3f} (IC 95%: {ci95_or[0]:.3f} - {ci95_or[1]:.3f})")
                    st.write(f"• Riesgo atribuible (RAE): {rae:.3f}")
                    st.write(f"• Fracción atribuible en expuestos (FAE): {fae:.3f}" if fae is not None else "No calculable FAE")
                    st.write(f"• Riesgo atribuible en población (RAP): {rap:.3f}")
                    st.write(f"• Fracción atribuible en población (FAP): {fap:.3f}" if fap is not None else "No calculable FAP")
                    st.write(f"• Número necesario a tratar (NNT): {nnt:.1f}" if nnt is not None else "No calculable NNT")
                    st.write(f"• Diferencia de riesgos (RD): {rd:.3f} (IC 95%: {ci95_rd[0]:.3f} - {ci95_rd[1]:.3f})")

                    if SCIPY_AVAILABLE:
                        # Chi2 o Fisher según valores esperados
                        tabla_chi = np.array([[a0, b0], [c0, d0]])
                        chi2, p, dof, ex = chi2_contingency(tabla_chi)
                        st.write(f"• Chi2: {chi2:.3f}, p-valor: {p:.4f}")
                        if (ex < 5).any():
                            # Valores esperados bajos, usar Fisher
                            _, p_fisher = fisher_exact(tabla_chi)
                            st.write(f"• Test exacto de Fisher p-valor: {p_fisher:.4f}")
                except Exception as e:
                    st.error(f"Error calculando medidas: {e}")


# --- TAB 7: Visualización de Datos ---
with tabs[7]:
    st.header("📉 Visualización de Datos")
    st.info("Aquí podrías agregar gráficas dinámicas con matplotlib, seaborn o plotly según datos ingresados.")
    # Ejemplo simple de gráfico con matplotlib
    df = pd.DataFrame({
        "Categoría": ["Expuestos", "No Expuestos"],
        "Casos": [st.session_state.get("a_val", 0), st.session_state.get("c_val", 0)],
        "Controles": [st.session_state.get("b_val", 0), st.session_state.get("d_val", 0)]
    })
    fig, ax = plt.subplots()
    df.set_index("Categoría")[["Casos", "Controles"]].plot(kind="bar", ax=ax)
    ax.set_ylabel("Número de personas")
    ax.set_title("Distribución Casos y Controles")
    st.pyplot(fig)


# --- TAB 8: Chat con Gemini ---
with tabs[8]:
    st.header("💬 Chat Epidemiológico con Gemini")

    if "messages" not in st.session_state:
        st.session_state["messages"] = [
            {"role": "assistant", "content": "¡Hola! Soy tu asistente de Epidemiología 101. ¿En qué te puedo ayudar hoy?"}
        ]

    chat_container = st.container()

    with chat_container:
        st.markdown('<div id="chat-container">', unsafe_allow_html=True)
        for msg in st.session_state["messages"]:
            role = msg["role"]
            content = msg["content"]
            with st.chat_message(role):
                st.markdown(content)
        st.markdown('</div>', unsafe_allow_html=True)

    prompt = st.chat_input("Escribe tu pregunta aquí...")
    if prompt:
        st.session_state["messages"].append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        with st.chat_message("assistant"):
            with st.spinner("Procesando respuesta..."):
                reply = chat_with_gemini_messages(st.session_state["messages"])
                st.markdown(reply)
                st.session_state["messages"].append({"role": "assistant", "content": reply})
