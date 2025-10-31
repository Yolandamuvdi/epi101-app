# main.py — Epidemiología 101 (PRO + Brotes + UX/UI + Auth robusto)
import streamlit as st
import os
import math
import io
import random
import requests
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import chi2_contingency, fisher_exact, norm
from streamlit_extras.let_it_rain import rain

# Optional: Gemini (si no está, fallback seguro)
try:
    import google.generativeai as genai
    GENAI_AVAILABLE = True
except ImportError:
    genai = None
    GENAI_AVAILABLE = False

# Import simulación adaptativa
try:
    from contenido.simulacion_adaptativa import simulacion_adaptativa as sim_adapt
except Exception:
    def sim_adapt(prev):
        return {"pregunta":"Demo: ¿Qué es incidencia?","opciones":["A","B","C"],"respuesta_correcta":"A","nivel":"Básico"}, "Demo"

# Brotes PRO
try:
    import simulacion_brotes as brotes_mod
    BROTES_AVAILABLE = True
except Exception:
    brotes_mod = None
    BROTES_AVAILABLE = False

# Configure Gemini si key en secrets
if GENAI_AVAILABLE:
    try:
        if "GEMINI_API_KEY" in st.secrets:
            genai.configure(api_key=st.secrets["GEMINI_API_KEY"])
    except Exception:
        pass

# Optional extras
try:
    import feedparser
    FEEDPARSER_AVAILABLE = True
except ImportError:
    FEEDPARSER_AVAILABLE = False

try:
    import folium
    from folium.plugins import HeatMap
    FOLIUM_AVAILABLE = True
except ImportError:
    FOLIUM_AVAILABLE = False

try:
    from streamlit_folium import st_folium
    STREAMLIT_FOLIUM_AVAILABLE = True
except ImportError:
    STREAMLIT_FOLIUM_AVAILABLE = False

try:
    import plotly.express as px
    PLOTLY_AVAILABLE = True
except ImportError:
    PLOTLY_AVAILABLE = False

try:
    from reportlab.lib.pagesizes import letter
    from reportlab.pdfgen import canvas
    from reportlab.lib.utils import ImageReader
    from PIL import Image
    REPORTLAB_AVAILABLE = True
except ImportError:
    REPORTLAB_AVAILABLE = False

try:
    import streamlit_authenticator as stauth
    AUTH_AVAILABLE = True
except ImportError:
    AUTH_AVAILABLE = False

# Streamlit extras: badges y timeline
try:
    from streamlit_extras.badges import badge
except ImportError:
    badge = None

try:
    from streamlit_extras.timeline import timeline
except ImportError:
    timeline = None

# ---------------------------
# Página + CSS
# ---------------------------
st.set_page_config(page_title="🧠 Epidemiología 101", page_icon="🧪", layout="wide")
st.markdown("""
<style>
:root { --oms-blue: #0d3b66; --health-green: #00a86b; }
body, .block-container { background: #f7fbfc; color: var(--oms-blue); font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif; }
.block-container { max-width: 1200px; margin: 1.5rem auto 3rem auto; padding: 1.5rem 2rem; background: #ffffff; border-radius: 12px; box-shadow: 0 10px 30px rgba(13,59,102,0.06); }
h1,h2,h3,h4 { color: var(--oms-blue); font-weight:700; }
a { color: var(--oms-blue); }
.stButton>button { background: linear-gradient(90deg,var(--oms-blue), #09466b); color: white; border-radius: 8px; padding: 8px 14px; font-weight:700; }
.stButton>button:hover { background: linear-gradient(90deg,var(--health-green), #2fcf8f); }
.small-muted { color: #6b7a86; font-size:0.9rem; }
.badge-green { background: linear-gradient(90deg,#00b36b,#2fcf8f); color: white; padding: 6px 10px; border-radius: 8px; display:inline-block; }
</style>
""", unsafe_allow_html=True)

# ---------------------------
# Funciones originales (sin cambios)
# ---------------------------
@st.cache_data(show_spinner=False)
def cargar_md(ruta):
    try:
        with open(ruta, encoding="utf-8") as f:
            return f.read()
    except:
        return None

@st.cache_data(show_spinner=False)
def cargar_py_variable(ruta_py, var_name):
    ns = {}
    try:
        with open(ruta_py, encoding="utf-8") as f:
            exec(f.read(), ns)
        return ns.get(var_name)
    except:
        return None

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

def interpretar_resultados(rr, rr_l, rr_u, or_, or_l, or_u, rd, rd_l, rd_u, p_val, test_name):
    texto = f"""
*Resultados Epidemiológicos:*
• Riesgo Relativo (RR): {rr:.3f} (IC95% {rr_l:.3f} - {rr_u:.3f})
• Odds Ratio (OR): {or_:.3f} (IC95% {or_l:.3f} - {or_u:.3f})
• Diferencia de Riesgos (RD): {rd:.3f} (IC95% {rd_l:.3f} - {rd_u:.3f})
• Valor p ({test_name}): {p_val:.4f}
"""
    if p_val < 0.05:
        texto += "🎯 Asociación estadísticamente significativa (p < 0.05)."
    else:
        texto += "⚠️ Asociación no estadísticamente significativa (p ≥ 0.05)."
    return texto

def plot_forest(rr, rr_l, rr_u, or_, or_l, or_u):
    fig, ax = plt.subplots(figsize=(6,3))
    ax.errorbar(x=[rr, or_], y=[2,1],
                 xerr=[[rr-rr_l, or_-or_l], [rr_u-rr, or_u-or_]],
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

def mostrar_confeti():
    rain(emoji="🎉", font_size=54, falling_speed=5, animation_length=3)

# ---------------------------
# Header + Badge (corregido)
# ---------------------------
st.title("🧠 Epidemiología 101")
st.markdown("Aprende epidemiología de manera interactiva y visual.")

# Fix badge
if badge is not None:
    st.markdown('<div class="badge-green">🧪 Versión PRO</div>', unsafe_allow_html=True)

st.markdown("---")

# ---------------------------
# Menú de secciones
# ---------------------------
menu = st.sidebar.selectbox("Navegación", ["Inicio","Simulador 2x2","Brotes","Simulación Adaptativa","Acerca"])

# ---------------------------
# Inicio
# ---------------------------
if menu == "Inicio":
    st.subheader("Bienvenida")
    st.markdown("Explora la epidemiología de manera visual e interactiva. Usa la barra lateral para navegar entre secciones.")
    mostrar_confeti()

# ---------------------------
# Simulador 2x2
# ---------------------------
elif menu == "Simulador 2x2":
    st.subheader("Simulador de Tabla 2x2")
    col1,col2 = st.columns(2)
    with col1:
        a = st.number_input("Casos expuestos (a)", min_value=0, value=10)
        b = st.number_input("No casos expuestos (b)", min_value=0, value=20)
    with col2:
        c = st.number_input("Casos no expuestos (c)", min_value=0, value=5)
        d = st.number_input("No casos no expuestos (d)", min_value=0, value=25)

    a,b,c,d,_ = corregir_ceros(a,b,c,d)

    rr, rr_l, rr_u = ic_riesgo_relativo(a,b,c,d)
    or_, or_l, or_u = ic_odds_ratio(a,b,c,d)
    rd, rd_l, rd_u = diferencia_riesgos(a,b,c,d)
    p_val, test_name = calcular_p_valor(a,b,c,d)
    st.markdown(interpretar_resultados(rr, rr_l, rr_u, or_, or_l, or_u, rd, rd_l, rd_u, p_val, test_name))
    plot_forest(rr, rr_l, rr_u, or_, or_l, or_u)
    plot_barras_expuestos(a,b,c,d)

# ---------------------------
# Brotes
# ---------------------------
elif menu == "Brotes":
    st.subheader("Simulación de Brotes")
    if BROTES_AVAILABLE:
        brotes_mod.run_app()
    else:
        st.info("Brotes PRO no disponible en este entorno.")

# ---------------------------
# Simulación Adaptativa
# ---------------------------
elif menu == "Simulación Adaptativa":
    st.subheader("Simulación Adaptativa")
    pregunta, meta = sim_adapt(None)
    st.markdown(f"**Pregunta:** {pregunta['pregunta']}")
    for opcion in pregunta['opciones']:
        st.radio("Opciones", pregunta['opciones'], index=0)

# ---------------------------
# Acerca
# ---------------------------
elif menu == "Acerca":
    st.subheader("Acerca de Epidemiología 101")
    st.markdown("Aplicación educativa PRO desarrollada por Yolanda Muvdi, enfermera epidemióloga.")

st.markdown("<br><br>", unsafe_allow_html=True)
