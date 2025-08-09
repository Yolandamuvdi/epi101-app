import streamlit as st
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import math
import os

# Intento importar Gemini y SciPy
try:
    import google.generativeai as genai
    GENAI_AVAILABLE = True
except Exception:
    GENAI_AVAILABLE = False

try:
    from scipy.stats import chi2_contingency, fisher_exact
    SCIPY_AVAILABLE = True
except Exception:
    SCIPY_AVAILABLE = False

# Funciones utilitarias
def safe_execute(func, *args, **kwargs):
    try:
        return func(*args, **kwargs)
    except Exception as e:
        st.error(f"Error inesperado: {e}")
        return None

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
    except Exception as e:
        return None

# Configuración Streamlit
st.set_page_config(page_title="Epidemiología 101", page_icon="🧠", layout="wide")

# Estilos básicos (modo claro/oscuro simple)
modo_oscuro = st.sidebar.checkbox("Modo oscuro / Dark mode")
if modo_oscuro:
    st.markdown("""
    <style>
        body, .block-container {background-color:#121212;color:#e0e0e0;padding:2rem 4rem;max-width:1100px;margin:auto;}
        .block-container {background:#1e1e1e;border-radius:12px;padding:3rem 4rem;}
        h1,h2,h3,h4 {color:#90caf9;}
        .stButton>button {background:#42a5f5;color:#fff;border-radius:6px;padding:0.5rem 1.2rem;}
        .stButton>button:hover {background:#1e88e5;cursor:pointer;}
    </style>
    """, unsafe_allow_html=True)
else:
    st.markdown("""
    <style>
        body, .block-container {background-color:#f5f5f5;color:#222;padding:2rem 4rem;max-width:1100px;margin:auto;}
        .block-container {background:#fff;border-radius:12px;padding:3rem 4rem;}
        h1,h2,h3,h4 {color:#0d47a1;}
        .stButton>button {background:#1565c0;color:#fff;border-radius:6px;padding:0.5rem 1.2rem;}
        .stButton>button:hover {background:#0d3570;cursor:pointer;}
    </style>
    """, unsafe_allow_html=True)

# Título
st.markdown('<h1 style="text-align:center;">🧠 Epidemiología 101 - Asistente educativo</h1>', unsafe_allow_html=True)
st.markdown('<p style="text-align:center;">Plataforma para aprender epidemiología, creada por Yolanda Muvdi.</p>', unsafe_allow_html=True)

# Definición de pestañas
tabs = st.tabs([
    "Conceptos básicos", 
    "Medidas de asociación", 
    "Tablas 2x2 y cálculos", 
    "Gráficos y visualización",
    "Sesgos",
    "Glosario",
    "Ejercicios prácticos",
    "Chat Epidemiólogo Virtual"
])

# Pestaña 0: Conceptos básicos
with tabs[0]:
    md = cargar_md("contenido/conceptosbasicos.md")
    st.markdown(md, unsafe_allow_html=True)

# Pestaña 1: Medidas de asociación
with tabs[1]:
    md = cargar_md("contenido/medidas_completas.md")
    st.markdown(md, unsafe_allow_html=True)

# Pestaña 2: Tablas 2x2 y cálculos
with tabs[2]:
    st.header("Ingreso de datos para tabla 2x2")
    col1, col2 = st.columns(2)
    with col1:
        a = st.number_input("a: Casos expuestos", min_value=0, value=10, step=1)
        b = st.number_input("b: No casos expuestos", min_value=0, value=20, step=1)
    with col2:
        c = st.number_input("c: Casos no expuestos", min_value=0, value=5, step=1)
        d = st.number_input("d: No casos no expuestos", min_value=0, value=40, step=1)

    def validar_inputs(a, b, c, d):
        mensajes = []
        valido = True
        for val, nom in zip([a,b,c,d], ["a","b","c","d"]):
            if not isinstance(val, int) or val < 0:
                mensajes.append(f"'{nom}' debe ser entero positivo.")
                valido = False
        if (a+b)==0:
            mensajes.append("Suma casos expuestos no puede ser 0")
            valido = False
        if (c+d)==0:
            mensajes.append("Suma casos no expuestos no puede ser 0")
            valido = False
        return valido, mensajes

    valido, mensajes = validar_inputs(a,b,c,d)
    if not valido:
        for msg in mensajes:
            st.warning(msg)

    def aplicar_correccion(a,b,c,d):
        if 0 in [a,b,c,d]:
            return a+0.5,b+0.5,c+0.5,d+0.5, True
        return a,b,c,d, False

    def calcular_medidas(a,b,c,d):
        a_adj,b_adj,c_adj,d_adj, corr = aplicar_correccion(a,b,c,d)
        inc_exp = a_adj / (a_adj+b_adj)
        inc_noexp = c_adj / (c_adj+d_adj)
        rr = inc_exp / inc_noexp if inc_noexp>0 else np.nan
        se_log_rr = math.sqrt(1/a_adj - 1/(a_adj+b_adj) + 1/c_adj - 1/(c_adj+d_adj))
        ci_rr = (math.exp(math.log(rr) - 1.96*se_log_rr), math.exp(math.log(rr) + 1.96*se_log_rr)) if rr>0 else (np.nan,np.nan)
        orr = (a_adj*d_adj)/(b_adj*c_adj) if (b_adj*c_adj)>0 else np.nan
        se_log_or = math.sqrt(1/a_adj + 1/b_adj + 1/c_adj + 1/d_adj)
        ci_or = (math.exp(math.log(orr) - 1.96*se_log_or), math.exp(math.log(orr) + 1.96*se_log_or)) if orr>0 else (np.nan,np.nan)
        rd = inc_exp - inc_noexp
        se_rd = math.sqrt((inc_exp*(1-inc_exp)/(a_adj+b_adj)) + (inc_noexp*(1-inc_noexp)/(c_adj+d_adj)))
        ci_rd = (rd - 1.96*se_rd, rd + 1.96*se_rd)
        p_val = np.nan
        if SCIPY_AVAILABLE:
            tabla = np.array([[a,b],[c,d]])
            try:
                chi2, p_val, _, _ = chi2_contingency(tabla)
            except:
                p_val = np.nan
            if p_val>0.05 and np.min(tabla)<5:
                try:
                    _, p_val = fisher_exact(tabla)
                except:
                    pass
        alertas = []
        if corr:
            alertas.append("Se aplicó corrección Haldane-Anscombe por ceros.")
        if p_val<0.05:
            alertas.append("Asociación estadísticamente significativa (p < 0.05).")
        else:
            alertas.append("No significativa (p ≥ 0.05).")
        return dict(RR=rr, IC95_RR=ci_rr, OR=orr, IC95_OR=ci_or, RD=rd, IC95_RD=ci_rd, p_val=p_val, alertas=alertas)

    if valido:
        if st.button("Calcular medidas"):
            resultados = safe_execute(calcular_medidas, a,b,c,d)
            if resultados:
                st.write("### Resultados:")
                st.write(f"Riesgo Relativo (RR): {resultados['RR']:.3f} (IC95% {resultados['IC95_RR'][0]:.3f} - {resultados['IC95_RR'][1]:.3f})")
                st.write(f"Odds Ratio (OR): {resultados['OR']:.3f} (IC95% {resultados['IC95_OR'][0]:.3f} - {resultados['IC95_OR'][1]:.3f})")
                st.write(f"Diferencia de riesgos (RD): {resultados['RD']:.3f} (IC95% {resultados['IC95_RD'][0]:.3f} - {resultados['IC95_RD'][1]:.3f})")
                st.write(f"Valor p: {resultados['p_val']:.4f}")
                for alerta in resultados['alertas']:
                    st.info(alerta)
    else:
        st.info("Corrige errores para calcular.")

# Pestaña 3: Gráficos y visualización
with tabs[3]:
    md = cargar_md("contenido/visualizacion.md")
    st.markdown(md, unsafe_allow_html=True)

# Pestaña 4: Sesgos
with tabs[4]:
    md = cargar_md("contenido/sesgos_completos.md")
    st.markdown(md, unsafe_allow_html=True)

# Pestaña 5: Glosario
with tabs[5]:
    glosario = cargar_py_variable("contenido/glosario_completo.py", "glosario")
    if glosario:
        st.write(glosario)
    else:
        st.warning("No se pudo cargar el glosario.")

# Pestaña 6: Ejercicios prácticos
with tabs[6]:
    ejercicios = cargar_py_variable("contenido/ejercicios_completos.py", "ejercicios")
    if ejercicios:
        st.write(ejercicios)
    else:
        st.warning("No se pudo cargar los ejercicios.")

# Pestaña 7: Chat con Epidemiólogo Virtual
with tabs[7]:
    st.header("💬 Chat con Epidemiólogo Virtual")

    if "chat_messages" not in st.session_state:
        st.session_state.chat_messages = []

    input_text = st.text_area("Escribe tu pregunta o duda:", height=100)

    GEMINI_KEY = st.secrets.get("GEMINI_API_KEY") or os.environ.get("GEMINI_API_KEY")

    if not GENAI_AVAILABLE:
        st.warning("La librería google-generativeai no está instalada. El chat no estará disponible.")
    elif not GEMINI_KEY:
        st.warning("❌ No se encontró GEMINI_API_KEY en secrets o variables de entorno. El chat no funcionará.")
    else:
        try:
            genai.configure(api_key=GEMINI_KEY)
        except Exception as e:
            st.warning(f"Advertencia al configurar Gemini: {e}")

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

    if st.button("Enviar"):
        if input_text.strip() != "":
            st.session_state.chat_messages.append({"role": "user", "content": input_text.strip()})
            respuesta = chat_with_gemini_messages(st.session_state.chat_messages)
            st.session_state.chat_messages.append({"role": "assistant", "content": respuesta})

    for msg in st.session_state.chat_messages:
        if msg["role"] == "user":
            st.markdown(f"**Tú:** {msg['content']}")
        else:
            st.markdown(f"**Epidemiólogo:** {msg['content']}")

# Footer
st.markdown("---")
st.markdown("© 2025 Yolanda Muvdi - Todos los derechos reservados.")

