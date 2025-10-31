# main.py — Epidemiología 101 (integración PRO + Brotes + UX/UI + Auth)
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

# Optional: Gemini (si no está, quedamos en modo sin chat)
try:
    import google.generativeai as genai  # ✅ Importar Gemini
    GENAI_AVAILABLE = True
except Exception:
    genai = None
    GENAI_AVAILABLE = False

# Importar función de simulación adaptativa (mantén tu archivo)
try:
    from contenido.simulacion_adaptativa import simulacion_adaptativa as sim_adapt
except Exception:
    def sim_adapt(prev):
        return {"pregunta":"Demo: ¿Qué es incidencia?","opciones":["A","B","C"],"respuesta_correcta":"A","nivel":"Básico"}, "Demo"

# Importar módulo PRO de brotes (ahora en la raíz, renombrado a simulacion_brotes.py)
try:
    import simulacion_brotes as brotes_mod
    BROTES_AVAILABLE = True
except Exception:
    brotes_mod = None
    BROTES_AVAILABLE = False

# Configurar Gemini si está y key en secrets
if GENAI_AVAILABLE:
    try:
        if "GEMINI_API_KEY" in st.secrets:
            genai.configure(api_key=st.secrets["GEMINI_API_KEY"])
    except Exception:
        pass

# ---------------------------
# Optional extras with fallbacks
# ---------------------------
try:
    import feedparser
    FEEDPARSER_AVAILABLE = True
except Exception:
    FEEDPARSER_AVAILABLE = False

try:
    import folium
    from folium.plugins import HeatMap
    FOLIUM_AVAILABLE = True
except Exception:
    FOLIUM_AVAILABLE = False

try:
    from streamlit_folium import st_folium
    STREAMLIT_FOLIUM_AVAILABLE = True
except Exception:
    STREAMLIT_FOLIUM_AVAILABLE = False

try:
    import plotly.express as px
    PLOTLY_AVAILABLE = True
except Exception:
    PLOTLY_AVAILABLE = False

try:
    from reportlab.lib.pagesizes import letter
    from reportlab.pdfgen import canvas
    from reportlab.lib.utils import ImageReader
    from PIL import Image
    REPORTLAB_AVAILABLE = True
except Exception:
    REPORTLAB_AVAILABLE = False

try:
    import streamlit_authenticator as stauth
    AUTH_AVAILABLE = True
except Exception:
    AUTH_AVAILABLE = False

try:
    from streamlit_extras.badges import badge
except Exception:
    badge = None
try:
    from streamlit_extras.timeline import timeline
except Exception:
    timeline = None

# ---------------------------
# Page config + CSS (profesional UX/UI)
# ---------------------------
st.set_page_config(page_title="🧠 Epidemiología 101", page_icon="🧪", layout="wide")

st.markdown("""
<style>
:root {
  --blue-primary: #0d3b66;
  --green-accent: #00a86b;
  --light-bg: #f7fbfc;
  --card-bg: #ffffff;
}
body, .block-container {
    background: var(--light-bg);
    color: var(--blue-primary);
    font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
}
.block-container {
    max-width: 1200px;
    margin: 1.5rem auto 3rem auto;
    padding: 1.5rem 2rem;
    background: var(--card-bg);
    border-radius: 12px;
    box-shadow: 0 10px 30px rgba(13,59,102,0.06);
}
h1,h2,h3,h4 { color: var(--blue-primary); font-weight:700; }
a { color: var(--blue-primary); text-decoration:none; }
.stButton>button {
    background: linear-gradient(90deg,var(--blue-primary), #09466b);
    color: white;
    border-radius: 8px;
    padding: 8px 14px;
    font-weight:700;
}
.stButton>button:hover { background: linear-gradient(90deg,var(--green-accent), #2fcf8f); }
.small-muted { color: #6b7a86; font-size:0.9rem; }
.card {
    background: #ffffff;
    padding: 20px;
    border-radius: 12px;
    box-shadow: 0 6px 20px rgba(0,0,0,0.08);
    margin-bottom: 20px;
}
.card h3 { color: var(--blue-primary); }
</style>
""", unsafe_allow_html=True)

# ---------------------------
# Keep original functions (no changes)
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

# Epidemiological functions
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
# Utilities para export y brotes
# ---------------------------
def fig_to_bytes(fig, fmt="png"):
    buf = io.BytesIO()
    fig.savefig(buf, format=fmt, bbox_inches="tight")
    buf.seek(0)
    return buf.getvalue()

def crear_pdf_2x2(a,b,c,d, rr, rr_l, rr_u, or_, or_l, or_u, rd, rd_l, rd_u, p_val, test_name):
    if not REPORTLAB_AVAILABLE:
        return None
    pdf_buf = io.BytesIO()
    c = canvas.Canvas(pdf_buf, pagesize=letter)
    c.setFont("Helvetica-Bold", 14)
    c.drawString(50, 760, "Reporte 2x2 - Epidemiología 101")
    c.setFont("Helvetica", 10)
    c.drawString(50, 740, f"Tabla: a={a}, b={b}, c={c}, d={d}")
    c.drawString(50, 720, f"RR: {rr:.3f} (IC95% {rr_l:.3f}-{rr_u:.3f})")
    c.drawString(50, 704, f"OR: {or_:.3f} (IC95% {or_l:.3f}-{or_u:.3f})")
    c.drawString(50, 688, f"RD: {rd:.3f} (IC95% {rd_l:.3f}-{rd_u:.3f})")
    c.drawString(50, 672, f"p-value ({test_name}): {p_val:.4f}")
    try:
        fig = make_forest_fig(rr, rr_l, rr_u, or_, or_l, or_u)
        img = ImageReader(io.BytesIO(fig_to_bytes(fig, fmt="png")))
        c.drawImage(img, 50, 380, width=500, height=250)
    except Exception:
        pass
    c.showPage()
    c.save()
    pdf_buf.seek(0)
    return pdf_buf.getvalue()

def make_forest_fig(rr, rr_l, rr_u, or_, or_l, or_u):
    fig, ax = plt.subplots(figsize=(6,3))
    ax.errorbar(x=[rr, or_], y=[2,1],
                 xerr=[[rr-rr_l, or_-or_l], [rr_u-rr, or_u-or_]],
                 fmt='o', color='#0d3b66', capsize=5, markersize=10)
    ax.set_yticks([1,2])
    ax.set_yticklabels(["OR","RR"])
    ax.axvline(1, color='gray', linestyle='--')
    return fig

# ---------------------------
# HOME UI — limpio y profesional
# ---------------------------
with st.container():
    st.title("🧠 Epidemiología 101")
    st.markdown("""
Bienvenido a **Epidemiología 101**, tu plataforma interactiva para aprender epidemiología clínica y análisis de datos de manera visual, práctica y gamificada.  
Explora **conceptos básicos**, realiza **ejercicios interactivos**, analiza **tablas 2x2**, y visualiza datos reales con gráficos y mapas.
""")

    col1, col2, col3 = st.columns([2,3,2])
    with col1:
        st.markdown("""
        <div class="card">
        <h3>📚 Conceptos Básicos</h3>
        Aprende teoría clave, definiciones y fórmulas epidemiológicas.
        </div>
        """, unsafe_allow_html=True)
    with col2:
        st.markdown("""
        <div class="card">
        <h3>📊 Ejercicios & Práctica</h3>
        Genera tablas 2x2, calcula RR, OR, RD, IC95% y valor p.
        </div>
        """, unsafe_allow_html=True)
    with col3:
        st.markdown("""
        <div class="card">
        <h3>🌎 Visualización de Datos</h3>
        Mapas, gráficos interactivos y alertas de brotes de enfermedades.
        </div>
        """, unsafe_allow_html=True)

# ---------------------------
# Sidebar (profesional)
# ---------------------------
with st.sidebar:
    st.header("📌 Navegación")
    st.markdown("""
- [Inicio](#)
- [Conceptos Básicos](#)
- [Ejercicios](#)
- [2x2 & Medidas de Asociación](#)
- [Gráficos & Mapas](#)
- [Alertas & Brotes](#)
- [Chat IA](#)
""")

    st.markdown("---")
    st.markdown("🎯 **Tips rápidos**:")
    st.markdown("""
- Calcula RR, OR, RD al instante.  
- Visualiza IC95% con gráficos de bosque.  
- Gamifica tu aprendizaje con simulaciones y retos.  
- Accede a brotes y alertas WHO (si tienes conexión).  
""")

    if badge:
        badge(label="Versión PRO", color="green", icon="🧪")

# ---------------------------
# Optional: Lluvia de confeti al iniciar (gamificación)
# ---------------------------
mostrar_confeti()

    # end sections

# run
if __name__ == "__main__":
    main()
