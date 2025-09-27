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
    # fallback minimal (shouldn't happen if archivo existe)
    def sim_adapt(prev):
        return {"pregunta":"Demo: ¿Qué es incidencia?","opciones":["A","B","C"],"respuesta_correcta":"A","nivel":"Básico"}, "Demo"

# Importar módulo PRO de brotes (debe existir en contenido/simulacion_pro_brotes.py)
try:
    import contenido.simulacion_pro_brotes as brotes_mod
    BROTES_AVAILABLE = True
except Exception:
    brotes_mod = None
    BROTES_AVAILABLE = False

# Intentar configurar Gemini si está y si la key está en secrets (no obligatorio)
if GENAI_AVAILABLE:
    try:
        if "GEMINI_API_KEY" in st.secrets:
            genai.configure(api_key=st.secrets["GEMINI_API_KEY"])
    except Exception:
        pass

# ---------------------------
# Optional extras with fallbacks
# ---------------------------
# feedparser (WHO DONs)
try:
    import feedparser
    FEEDPARSER_AVAILABLE = True
except Exception:
    FEEDPARSER_AVAILABLE = False

# folium + streamlit_folium
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

# plotly
try:
    import plotly.express as px
    PLOTLY_AVAILABLE = True
except Exception:
    PLOTLY_AVAILABLE = False

# reportlab + Pillow for PDF export
try:
    from reportlab.lib.pagesizes import letter
    from reportlab.pdfgen import canvas
    from reportlab.lib.utils import ImageReader
    from PIL import Image
    REPORTLAB_AVAILABLE = True
except Exception:
    REPORTLAB_AVAILABLE = False

# streamlit-authenticator (login/roles)
try:
    import streamlit_authenticator as stauth
    AUTH_AVAILABLE = True
except Exception:
    AUTH_AVAILABLE = False

# streamlit_extras badges/timeline (optional)
try:
    from streamlit_extras.badges import badge
except Exception:
    badge = None
try:
    from streamlit_extras.timeline import timeline
except Exception:
    timeline = None

# ---------------------------
# Page config + CSS (azul OMS + acentos verdes)
# ---------------------------
st.set_page_config(page_title="🧠 Epidemiología 101", page_icon="🧪", layout="wide")

st.markdown("""
<style>
:root {
  --oms-blue: #0d3b66;
  --health-green: #00a86b;
}
body, .block-container {
    background: #f7fbfc;
    color: var(--oms-blue);
    font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
}
.block-container {
    max-width: 1200px;
    margin: 1.5rem auto 3rem auto;
    padding: 1.5rem 2rem;
    background: #ffffff;
    border-radius: 12px;
    box-shadow: 0 10px 30px rgba(13,59,102,0.06);
}
h1,h2,h3,h4 { color: var(--oms-blue); font-weight:700; }
a { color: var(--oms-blue); }
.stButton>button {
    background: linear-gradient(90deg,var(--oms-blue), #09466b);
    color: white;
    border-radius: 8px;
    padding: 8px 14px;
    font-weight:700;
}
.stButton>button:hover { background: linear-gradient(90deg,var(--health-green), #2fcf8f); }
.small-muted { color: #6b7a86; font-size:0.9rem; }
.badge-green { background: linear-gradient(90deg,#00b36b,#2fcf8f); color: white; padding: 6px 10px; border-radius: 8px; display:inline-block; }
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

# your epidemiological functions (unchanged)
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

# Gamification confetti (original)
def mostrar_confeti():
    rain(emoji="🎉", font_size=54, falling_speed=5, animation_length=3)

# ---------------------------
# Utilities para export y brotes (no tocan tus funciones originales)
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
    # Add forest plot image
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
    ax.set_yticklabels(["Odds Ratio (OR)", "Riesgo Relativo (RR)"])
    ax.axvline(1, color='gray', linestyle='--')
    ax.set_xlabel("Medidas de Asociación")
    ax.set_title("Intervalos de Confianza 95%")
    plt.tight_layout()
    return fig

# ---------------------------
# Auth setup (streamlit-authenticator or demo fallback)
# ---------------------------
def setup_auth():
    user = {"name": None, "username": None, "auth_status": False, "role": "Demo"}
    if AUTH_AVAILABLE:
        try:
            # demo credentials; in prod put in secrets
            credentials = {
                "usernames": {
                    "pro": {"name":"Paola", "password":"pro123"},
                    "estudiante": {"name":"Estudiante", "password":"est123"},
                    "demo": {"name":"Demo", "password":"demo"}
                }
            }
            authenticator = stauth.Authenticate(credentials, "epi101_cookie", "epi101_sig", cookie_expiry_days=1)
            name, auth_status, username = authenticator.login("Inicia sesión", "sidebar")
            if auth_status:
                authenticator.logout("Cerrar sesión", "sidebar")
            user["name"] = name
            user["username"] = username
            user["auth_status"] = auth_status
            if auth_status:
                if username == "pro":
                    user["role"] = "Pro"
                else:
                    user["role"] = "Estudiante"
            else:
                user["role"] = "Demo"
            return user
        except Exception:
            st.sidebar.warning("streamlit-authenticator falló. Entrando modo demo.")
    # fallback: demo mode
    if "demo_user" not in st.session_state:
        st.session_state["demo_user"] = "Demo User"
    user["name"] = st.session_state["demo_user"]
    user["username"] = "demo"
    user["auth_status"] = True
    user["role"] = "Demo"
    return user

# ---------------------------
# PAGE NAV + Main
# ---------------------------
def pagina_inicio():
    st.title("🧠 Epidemiología 101")
    st.markdown("Bienvenido/a a Epidemiología 101 — selecciona una sección desde la barra lateral.")
    st.markdown("**Recuerda instalar dependencias** (si quieres exportar PDF / ver mapas / alertas WHO):")
    st.code("pip install reportlab pillow streamlit-authenticator folium streamlit_folium feedparser openpyxl")

def barra_lateral(seleccion_actual, user_info):
    st.sidebar.title("🧪 Epidemiología 101")
    st.sidebar.markdown(f"👩‍⚕️ Creado por **Yolanda Muvdi**  \n**Usuario:** {user_info.get('name','-')}  \n**Rol:** {user_info.get('role','Demo')}")
    st.sidebar.markdown("---")
    opciones = [
        "📚 Academia", "🛠️ Toolkit", "📈 Medidas de Asociación", "📊 Diseños de Estudio",
        "⚠️ Sesgos y Errores", "📚 Glosario Interactivo", "🧪 Ejercicios Prácticos",
        "📊 Tablas 2x2 y Cálculos", "📊 Visualización de Datos", "🎥 Multimedia YouTube",
        "🤖 Chat Epidemiológico", "🎯 Gamificación", "📢 Brotes (PRO)"
    ]
    # if non-pro hide brotes and toolkit?
    if user_info.get("role") not in ["Pro","Estudiante","Demo"]:
        pass
    seleccion_sidebar = st.sidebar.radio("Ir a sección:", opciones, index=opciones.index(seleccion_actual) if seleccion_actual in opciones else 0)
    if seleccion_sidebar != seleccion_actual:
        st.session_state.seccion = seleccion_sidebar
    # timeline mini
    st.sidebar.markdown("---")
    st.sidebar.markdown("**Roadmap**")
    if timeline:
        try:
            items = [
                {"label":"M1","title":"Conceptos","description":"Bases"},
                {"label":"M2","title":"Medidas","description":"RR/OR/RD"},
                {"label":"M3","title":"Toolkit","description":"Export y APIs"},
                {"label":"M4","title":"Brotes","description":"Simulaciones PRO"}
            ]
            timeline(items)
        except Exception:
            st.sidebar.markdown("- M1 Conceptos\n- M2 Medidas\n- M3 Toolkit\n- M4 Brotes")
    else:
        st.sidebar.markdown("- M1 Conceptos\n- M2 Medidas\n- M3 Toolkit\n- M4 Brotes")
    return st.session_state.seccion

def main():
    # auth
    user_info = setup_auth()

    # init session state
    if "seccion" not in st.session_state:
        st.session_state.seccion = None
        st.session_state.nivel_gamificacion = None
        st.session_state.index_pregunta = 0
        st.session_state.respuestas_correctas = 0
        st.session_state.respuestas_usuario = {}
        st.session_state.applied_interventions = []
        st.session_state.progress = 20
        st.session_state.badges = []

    # show home if no section
    if st.session_state.seccion is None:
        pagina_inicio()
        # quick start buttons
        if st.sidebar.button("Abrir Academia"):
            st.session_state.seccion = "📚 Academia"
        return

    barra_lateral(st.session_state.seccion, user_info)
    seleccion = st.session_state.seccion

    # ---------- SECCIONES (manteniendo tu estructura) ----------
    if seleccion == "📚 Academia":
        st.header("📚 Academia")
        contenido = cargar_md("contenido/conceptosbasicos.md")
        if contenido:
            st.markdown(contenido)
        else:
            st.info("Archivo 'contenido/conceptosbasicos.md' no encontrado.")
        # videos quick
        st.markdown("### Videos recomendados")
        videos = {
            "Introducción a Epidemiología": "https://www.youtube.com/watch?v=qVFP-IkyWgQ",
            "Medidas de Asociación": "https://www.youtube.com/watch?v=d61E24xvRfI"
        }
        cols = st.columns(2)
        i = 0
        for t,u in videos.items():
            with cols[i%2]:
                st.markdown(f"**{t}**")
                st.video(u)
            i += 1

    elif seleccion == "🛠️ Toolkit":
        st.header("🛠️ Toolkit")
        # restrict to Pro or Estudiante (Pro has full)
        if user_info.get("role") == "Pro" or user_info.get("role") == "Estudiante" or user_info.get("role") == "Demo":
            st.markdown("Herramientas: exportación, APIs reales (OWID, INS cuando posible), cálculos.")
            # Example: load OWID (may take a bit)
            try:
                owid_url = "https://covid.ourworldindata.org/data/owid-covid-data.csv"
                if st.button("Cargar muestra OWID (puede tardar)"):
                    df_owid = pd.read_csv(owid_url, parse_dates=["date"])
                    st.success("OWID cargado")
                    st.dataframe(df_owid.head())
            except Exception:
                st.info("No fue posible descargar OWID en este entorno.")
        else:
            st.warning("Toolkit disponible solo para usuarios Pro — activa el modo Pro.")

    elif seleccion == "📈 Medidas de Asociación":
        st.header(seleccion)
        contenido = cargar_md("contenido/medidas_completas.md")
        if contenido: st.markdown(contenido)
        else: st.info("Archivo 'contenido/medidas_completas.md' no encontrado.")

    elif seleccion == "📊 Diseños de Estudio":
        st.header(seleccion)
        contenido = cargar_md("contenido/disenos_completos.md")
        if contenido: st.markdown(contenido)
        else: st.info("Archivo 'contenido/disenos_completos.md' no encontrado.")

    elif seleccion == "⚠️ Sesgos y Errores":
        st.header(seleccion)
        contenido = cargar_md("contenido/sesgos_completos.md")
        if contenido: st.markdown(contenido)
        else: st.info("Archivo 'contenido/sesgos_completos.md' no encontrado.")

    elif seleccion == "📚 Glosario Interactivo":
        st.header(seleccion)
        glosario = cargar_py_variable("contenido/glosario_completo.py", "glosario")
        if glosario:
            for termino, definicion in glosario.items():
                with st.expander(termino):
                    st.write(definicion)
        else:
            st.info("Archivo 'contenido/glosario_completo.py' no encontrado o variable 'glosario' no definida.")

    elif seleccion == "🧪 Ejercicios Prácticos":
        st.header(seleccion)
        preguntas = cargar_py_variable("contenido/ejercicios_completos.py", "preguntas")
        if preguntas:
            for i, p in enumerate(preguntas):
                st.subheader(f"Pregunta {i+1}")
                respuesta = st.radio(p["pregunta"], p["opciones"], key=f"ej_{i}")
                if st.button(f"Verificar respuesta {i+1}", key=f"btn_{i}"):
                    if respuesta == p["respuesta_correcta"]:
                        st.success("✅ Correcto")
                    else:
                        st.error(f"❌ Incorrecto. Respuesta correcta: {p['respuesta_correcta']}")
        else:
            st.info("Archivo 'contenido/ejercicios_completos.py' no encontrado o variable 'preguntas' no definida.")

    elif seleccion == "📊 Tablas 2x2 y Cálculos":
        st.header(seleccion)
        # defaults
        if "a" not in st.session_state: st.session_state.a = 10
        if "b" not in st.session_state: st.session_state.b = 20
        if "c" not in st.session_state: st.session_state.c = 5
        if "d" not in st.session_state: st.session_state.d = 40

        col1, col2 = st.columns(2)
        with col1:
            st.session_state.a = st.number_input("Casos expuestos (a)", min_value=0, value=st.session_state.a, step=1)
            st.session_state.b = st.number_input("No casos expuestos (b)", min_value=0, value=st.session_state.b, step=1)
        with col2:
            st.session_state.c = st.number_input("Casos no expuestos (c)", min_value=0, value=st.session_state.c, step=1)
            st.session_state.d = st.number_input("No casos no expuestos (d)", min_value=0, value=st.session_state.d, step=1)

        if st.button("Calcular"):
            a, b, c, d = st.session_state.a, st.session_state.b, st.session_state.c, st.session_state.d
            total = a+b+c+d
            if total == 0:
                st.error("Por favor ingresa valores mayores que cero.")
            else:
                a_, b_, c_, d_, corregido = corregir_ceros(a,b,c,d)
                rr, rr_l, rr_u = ic_riesgo_relativo(a_,b_,c_,d_)
                or_, or_l, or_u = ic_odds_ratio(a_,b_,c_,d_)
                rd, rd_l, rd_u = diferencia_riesgos(a_,b,c,d)
                p_val, test_name = calcular_p_valor(int(a_), int(b_), int(c_), int(d_))
                st.markdown(interpretar_resultados(rr, rr_l, rr_u, or_, or_l, or_u,
                                                  rd, rd_l, rd_u, p_val, test_name))
                if corregido:
                    st.warning("Se aplicó corrección de 0.5 en celdas con valor 0 para cálculos.")
                # show plots
                fig_f = make_forest_fig(rr, rr_l, rr_u, or_, or_l, or_u)
                st.pyplot(fig_f, use_container_width=True)
                plot_barras_expuestos(a,b,c,d)

                # export options
                df_res = pd.DataFrame([{"a":a,"b":b,"c":c,"d":d,"RR":rr,"RR_l":rr_l,"RR_u":rr_u,"OR":or_,"RD":rd,"p":p_val}])
                buf_xl = io.BytesIO(); df_res.to_excel(buf_xl,index=False); buf_xl.seek(0)
                st.download_button("⬇️ Descargar resultados (Excel)", data=buf_xl, file_name="resultados_2x2.xlsx", mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")

                # PDF
                if REPORTLAB_AVAILABLE:
                    pdf_bytes = crear_pdf_2x2(a,b,c,d, rr, rr_l, rr_u, or_, or_l, or_u, rd, rd_l, rd_u, p_val, test_name)
                    if pdf_bytes:
                        st.download_button("⬇️ Descargar reporte (PDF)", data=pdf_bytes, file_name="reporte_2x2.pdf", mime="application/pdf")
                else:
                    st.info("Para exportar PDF instala: reportlab pillow")

    elif seleccion == "📊 Visualización de Datos":
        st.header(seleccion)
        uploaded_file = st.file_uploader("Carga un archivo CSV para gráficos exploratorios", type=["csv"])
        if uploaded_file:
            try:
                df = pd.read_csv(uploaded_file)
                st.dataframe(df.head())
                num_cols = df.select_dtypes(include=np.number).columns.tolist()
                if num_cols:
                    col = st.selectbox("Selecciona columna numérica para graficar", num_cols)
                    fig, ax = plt.subplots()
                    df[col].value_counts().plot(kind='bar', ax=ax, color='#0d3b66')
                    ax.set_title(f"Distribución de {col}")
                    st.pyplot(fig, use_container_width=True)
                    png = fig_to_bytes(fig, fmt="png")
                    st.download_button("⬇️ Descargar gráfico (PNG)", data=png, file_name="grafico.png", mime="image/png")
                    excel_buf = io.BytesIO(); df.to_excel(excel_buf, index=False); excel_buf.seek(0)
                    st.download_button("⬇️ Descargar datos (Excel)", data=excel_buf, file_name="datos.xlsx")
                else:
                    st.info("No se detectaron columnas numéricas para graficar.")
            except Exception as e:
                st.error(f"Error leyendo CSV: {e}")
        else:
            st.info("Carga un archivo CSV para ver gráficos dinámicos.")

    elif seleccion == "🎥 Multimedia YouTube":
        st.header(seleccion)
        videos = {
            "Introducción a Epidemiología": "https://www.youtube.com/watch?v=qVFP-IkyWgQ",
            "Medidas de Asociación": "https://www.youtube.com/watch?v=d61E24xvRfI",
            "Diseños de Estudio": "https://www.youtube.com/watch?v=y6odn6E8yRs",
            "Sesgos en Epidemiología": "https://www.youtube.com/watch?v=1kyFIyG37qc"
        }
        for titulo, url in videos.items():
            st.markdown(f"**{titulo}**")
            st.video(url)
        st.markdown("Puedes pausar el video y aplicar las preguntas del módulo de brotes o de la simulación adaptativa para practicar decisiones reales.")

    elif seleccion == "🤖 Chat Epidemiológico":
        st.header(seleccion)
        pregunta = st.text_input("Escribe tu pregunta epidemiológica:")
        if st.button("Enviar") and pregunta:
            if not GENAI_AVAILABLE:
                st.warning("Gemini (google.generativeai) no está disponible en este entorno. Instala la librería y agrega GEMINI_API_KEY en secrets.")
            else:
                api_key = st.secrets.get("GEMINI_API_KEY")
                if not api_key:
                    st.error("No se encontró GEMINI_API_KEY en secrets.")
                else:
                    try:
                        model = genai.GenerativeModel("gemini-1.5-flash")
                        respuesta = model.generate_content(pregunta)
                        st.write(respuesta.text if hasattr(respuesta, "text") else str(respuesta))
                    except Exception as e:
                        st.error(f"Error consultando Gemini: {e}")

    elif seleccion == "🎯 Gamificación":
        st.header(seleccion)
        if st.session_state.nivel_gamificacion is None:
            st.subheader("Antes de comenzar, ¿en qué nivel sientes que estás en Epidemiología?")
            nivel = st.radio("Selecciona tu nivel:", ["Principiante","Intermedio","Avanzado"], index=0, key="nivel_inicial")
            if st.button("Comenzar"):
                st.session_state.nivel_gamificacion = nivel
                st.session_state.index_pregunta = 0
                st.session_state.respuestas_correctas = 0
                st.session_state.respuestas_usuario = {}
        else:
            if "pregunta_actual" not in st.session_state:
                st.session_state.pregunta_actual, st.session_state.mensaje = sim_adapt({})
            pregunta_actual = st.session_state.pregunta_actual
            mensaje = st.session_state.mensaje
            st.subheader(f"Pregunta {st.session_state.index_pregunta + 1}")
            st.write(pregunta_actual["pregunta"])
            st.info(mensaje)
            respuesta = st.radio("Selecciona tu respuesta:", pregunta_actual["opciones"], key=f"resp_temp_{st.session_state.index_pregunta}")
            if st.button("Enviar respuesta", key=f"btn_{st.session_state.index_pregunta}"):
                correcta = pregunta_actual["respuesta_correcta"]
                st.session_state.respuestas_usuario[st.session_state.index_pregunta] = {
                    "pregunta": pregunta_actual["pregunta"],
                    "nivel": pregunta_actual["nivel"],
                    "correcto": respuesta == correcta,
                    "seleccion": respuesta
                }
                if respuesta == correcta:
                    st.success("✅ Correcto")
                    mostrar_confeti()
                    st.session_state.respuestas_correctas += 1
                    st.session_state.progress = min(100, st.session_state.get("progress",20) + 8)
                else:
                    st.error(f"❌ Incorrecto. Respuesta correcta: {correcta}")
                st.session_state.index_pregunta += 1
                st.session_state.pregunta_actual, st.session_state.mensaje = sim_adapt(st.session_state.respuestas_usuario)
            # show progress + badges
            st.markdown("### Progreso")
            st.progress(st.session_state.get("progress",20))
            if st.session_state.get("respuestas_correctas",0) >= 5 and "Nivel Pro" not in st.session_state.badges:
                st.session_state.badges.append("Nivel Pro")
            st.markdown("### Badges")
            for b in st.session_state.badges:
                st.markdown(f"- {b}")
            st.markdown("### Ranking (local)")
            ranking_df = pd.DataFrame.from_dict({"Usuario":[user_info.get("name","Demo")], "Puntos":[st.session_state.get("respuestas_correctas",0)]})
            st.table(ranking_df)

    elif seleccion == "📢 Brotes (PRO)":
        st.header("📢 Simulación de Brotes (Módulo PRO)")
        if user_info.get("role") != "Pro":
            st.warning("Acceso restringido: este módulo está disponible para usuarios PRO. Usa modo demo o adquiere acceso Pro.")
            # still offer demo view of brotes summary if brotes_mod available
            if BROTES_AVAILABLE:
                st.info("Vista demo del módulo de brotes disponible.")
                if st.button("Abrir demo Brotes"):
                    try:
                        brotes_mod.app()
                    except Exception as e:
                        st.error(f"Error arrancando módulo brotes: {e}")
            else:
                st.info("Módulo BROTES no encontrado en contenido/simulacion_pro_brotes.py")
        else:
            # full PRO access
            if BROTES_AVAILABLE:
                try:
                    brotes_mod.app()
                except Exception as e:
                    st.error(f"Error arrancando módulo brotes: {e}")
            else:
                st.error("Módulo 'contenido/simulacion_pro_brotes.py' no encontrado. Añade el archivo y reintenta.")

    # end sections

# run
if __name__ == "__main__":
    main()
