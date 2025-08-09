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

# ---------- ESTILOS GENERALES ----------
st.set_page_config(
    page_title="🧠 Epidemiología 101 - Masterclass",
    page_icon="🧪",
    layout="wide",
    initial_sidebar_state="collapsed",
)

st.markdown("""
<style>
/* Background & typography */
body, .block-container {
    background: #f9fafb;
    color: #0d3b66;
    font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
    line-height: 1.5;
}
/* Container */
.block-container {
    max-width: 1100px !important;
    padding: 2rem 3rem !important;
    margin: 2rem auto 4rem auto !important;
    background: white;
    border-radius: 14px;
    box-shadow: 0 16px 35px rgba(13,59,102,0.12);
}

/* Botones */
.stButton>button {
    background-color: #0d3b66;
    color: white;
    border-radius: 8px;
    padding: 12px 25px;
    font-weight: 700;
    font-size: 1.1rem;
    transition: background-color 0.3s ease;
}
.stButton>button:hover {
    background-color: #09466b;
    cursor: pointer;
}

/* Links */
a {
    color: #f4d35e;
    text-decoration: none;
    font-weight: 600;
}
a:hover {
    text-decoration: underline;
}

/* Headers */
h1, h2, h3, h4 {
    color: #0d3b66;
    font-weight: 700;
}

/* Scroll para preguntas largas */
.stRadio > div {
    max-height: 140px;
    overflow-y: auto;
}

/* Footer */
footer {
    text-align: center;
    padding: 1rem 0;
    font-size: 0.9rem;
    color: #666;
    border-top: 1px solid #ddd;
    margin-top: 3rem;
}

/* Responsive botones */
@media (max-width: 768px) {
    .stButton>button {
        width: 100% !important;
        font-size: 1.3rem !important;
    }
}
</style>
""", unsafe_allow_html=True)

# ---------- FUNCIONES DE CÁLCULOS EPIDEMIOLÓGICOS ----------

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

def plot_forest(rr, rr_l, rr_u, or_, or_l, or_u):
    fig, ax = plt.subplots(figsize=(6,3))
    ax.errorbar(x=[rr, or_], y=[2,1], 
                xerr=[ [rr-rr_l, or_-or_l], [rr_u-rr, or_u-or_] ], 
                fmt='o', color='#0d3b66', capsize=5, markersize=12)
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
    fig, ax = plt.subplots(figsize=(8,4))
    ax.bar(labels, valores, color=colores, edgecolor='#09466b')
    ax.set_ylabel("Conteo")
    ax.set_title("Distribución de exposición y casos")
    plt.xticks(rotation=25)
    plt.grid(axis='y', linestyle='--', alpha=0.5)
    st.pyplot(fig, use_container_width=True)

# ---------- GAMIFICACIÓN ----------
niveles = {
    "Básico": [
        {
            "pregunta": "¿Qué es una tasa en epidemiología?",
            "opciones": ["Un número sin unidades", "Número de eventos por unidad de tiempo", "Una medida de asociación", "Ninguna"],
            "respuesta_correcta": "Número de eventos por unidad de tiempo"
        },
        {
            "pregunta": "¿Qué significa IR para un estudio?",
            "opciones": ["Incidencia", "Incidencia acumulada", "Riesgo relativo", "Odds ratio"],
            "respuesta_correcta": "Incidencia"
        }
    ],
    "Intermedio": [
        {
            "pregunta": "¿Cuál es la fórmula para Riesgo Relativo?",
            "opciones": ["a/(a+b)", "c/(c+d)", "(a/(a+b)) / (c/(c+d))", "a/c"],
            "respuesta_correcta": "(a/(a+b)) / (c/(c+d))"
        },
        {
            "pregunta": "¿Qué test estadístico se usa para tablas 2x2 con celdas pequeñas?",
            "opciones": ["Chi-cuadrado", "Fisher exacto", "T de Student", "ANOVA"],
            "respuesta_correcta": "Fisher exacto"
        }
    ],
    "Avanzado": [
        {
            "pregunta": "¿Cuál es la interpretación correcta de un OR > 1?",
            "opciones": [
                "Asociación negativa",
                "Asociación positiva",
                "No hay asociación",
                "Error en el estudio"
            ],
            "respuesta_correcta": "Asociación positiva"
        },
        {
            "pregunta": "¿Qué asunción es clave para el uso del Riesgo Relativo?",
            "opciones": [
                "Diseño transversal",
                "Estudio de cohorte",
                "Estudio de casos y controles",
                "Estudio ecológico"
            ],
            "respuesta_correcta": "Estudio de cohorte"
        }
    ]
}

def mostrar_confeti():
    st.balloons()
    st.success("🎉 ¡Excelente, eres un especialista en Epidemiología! 🎉")

# ---------- MENÚ Y SECCIONES ----------
secciones = [
    "Inicio",
    "Conceptos Básicos",
    "Medidas de Asociación",
    "Diseños de Estudio",
    "Sesgos y Errores",
    "Glosario Interactivo",
    "Ejercicios Prácticos",
    "Tablas 2x2 y Cálculos",
    "Visualización de Datos",
    "Multimedia YouTube",
    "Gamificación",
    "Chat Epidemiológico"
]

# ---------- CONTENIDO DE VIDEOS ----------
videos_youtube = {
    "Introducción a Epidemiología": "https://www.youtube.com/watch?v=Z4JDr11G0N4",
    "Medidas de Asociación": "https://www.youtube.com/watch?v=pD9Oa88kqQI",
    "Diseños de Estudios Epidemiológicos": "https://www.youtube.com/watch?v=sP4kzERW6wo",
    "Sesgos y Errores Comunes": "https://www.youtube.com/watch?v=RrKhv8OdLmY"
}

# ---------- CARGA DE ARCHIVOS MD Y PY ----------
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

# ---------- INTERFAZ PRINCIPAL ----------
st.title("🧪 Epidemiología 101")

# Bienvenida con diseño pro
st.markdown("""
    <div style="
        background: linear-gradient(90deg, #0d3b66 0%, #3e5c76 100%);
        padding: 2rem;
        border-radius: 15px;
        color: white;
        font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
        box-shadow: 0 8px 20px rgba(13,59,102,0.3);
        max-width: 900px;
        margin: 1rem auto 2rem auto;
        text-align: center;
    ">
        <h1 style="font-weight: 800; font-size: 2.8rem; margin-bottom: 0.2rem;">🧪 Bienvenido/a a <span style="color:#f4d35e;">Epidemiología 101</span></h1>
        <p style="font-size: 1.25rem; margin: 0.3rem 0 1rem 0;">
            Creado por <strong>Yolanda Muvdi</strong>, Enfermera con <em>MSc en Epidemiología</em>
        </p>
        <p style="font-size: 1.1rem; margin: 0.5rem 0;">
            📧 <a href="mailto:ymuvdi@gmail.com" style="color:#f4d35e; font-weight: 600; text-decoration: none;">
                ymuvdi<span style="font-family: monospace;">&#64;</span>gmail.com
            </a>
        </p>
        <p style="font-size: 1.1rem; margin: 0.5rem 0 1.5rem 0;">
            🔗 <a href="https://www.linkedin.com/in/yolanda-paola-muvdi-muvdi-778b73152/" 
                 target="_blank" style="color:#f4d35e; font-weight: 600; text-decoration: none;">
                LinkedIn
            </a>
        </p>
        <h3 style="font-weight: 700; margin-bottom: 0.3rem;">¿Qué sección quieres usar hoy?</h3>
        <p style="font-size: 1rem; font-style: italic; opacity: 0.8;">
            Explora conceptos, ejercicios, tablas 2x2, visualización y mucho más.
        </p>
    </div>
""", unsafe_allow_html=True)

# Dropdown para elegir sección
seccion = st.selectbox("Selecciona la sección:", secciones)

# ---------- SECCIONES ----------

if seccion == "Inicio":
    st.markdown("""
    ### ¡Hola! Este es un espacio interactivo para aprender epidemiología de forma práctica y entretenida.  
    Usa el menú desplegable para explorar conceptos, ejercicios, tablas 2x2, visualización, gamificación y más.
    """)

elif seccion == "Conceptos Básicos":
    st.header("📌 Conceptos Básicos")
    contenido = cargar_md("contenido/conceptosbasicos.md")
    if contenido:
        st.markdown(contenido)
    else:
        st.info("Agrega el archivo 'contenido/conceptosbasicos.md' para mostrar el contenido.")

elif seccion == "Medidas de Asociación":
    st.header("📈 Medidas de Asociación")
    contenido = cargar_md("contenido/medidas_completas.md")
    if contenido:
        st.markdown(contenido)
    else:
        st.info("Agrega el archivo 'contenido/medidas_completas.md' para mostrar el contenido.")

elif seccion == "Diseños de Estudio":
    st.header("📊 Diseños de Estudio")
    contenido = cargar_md("contenido/disenos_completos.md")
    if contenido:
        st.markdown(contenido)
    else:
        st.info("Agrega el archivo 'contenido/disenos_completos.md' para mostrar el contenido.")

elif seccion == "Sesgos y Errores":
    st.header("⚠️ Sesgos y Errores")
    contenido = cargar_md("contenido/sesgos_completos.md")
    if contenido:
        st.markdown(contenido)
    else:
        st.info("Agrega el archivo 'contenido/sesgos_completos.md' para mostrar el contenido.")

elif seccion == "Glosario Interactivo":
    st.header("📚 Glosario Interactivo")
    glosario = cargar_py_variable("contenido/glosario_completo.py", "glosario")
    if glosario:
        for termino, definicion in glosario.items():
            with st.expander(termino):
                st.write(definicion)
    else:
        st.info("Agrega 'contenido/glosario_completo.py' con variable glosario.")

elif seccion == "Ejercicios Prácticos":
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
                else:
                    st.error(f"❌ Incorrecto. La respuesta correcta es: {q['respuesta_correcta']}")
        if respuestas_correctas == len(preguntas) and len(preguntas) > 0:
            st.success("🎉 ¡Has completado todos los ejercicios!")
    else:
        st.info("Agrega 'contenido/ejercicios_completos.py' con variable preguntas.")

elif seccion == "Tablas 2x2 y Cálculos":
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
        st.number_input("Casos Expuestos (a)", min_value=0, value=st.session_state.a, key="a")
        st.number_input("No Casos Expuestos (b)", min_value=0, value=st.session_state.b, key="b")
    with col2:
        st.number_input("Casos No Expuestos (c)", min_value=0, value=st.session_state.c, key="c")
        st.number_input("No Casos No Expuestos (d)", min_value=0, value=st.session_state.d, key="d")

    a, b, c, d = st.session_state.a, st.session_state.b, st.session_state.c, st.session_state.d

    # Ajustar ceros si hay
    a_corr, b_corr, c_corr, d_corr, ajustado = corregir_ceros(a,b,c,d)
    if ajustado:
        st.warning("Se han ajustado los valores agregando 0.5 para evitar ceros en la tabla 2x2.")

    # Calcular medidas
    rr, rr_l, rr_u = ic_riesgo_relativo(a_corr,b_corr,c_corr,d_corr)
    or_, or_l, or_u = ic_odds_ratio(a_corr,b_corr,c_corr,d_corr)
    rd, rd_l, rd_u = diferencia_riesgos(a_corr,b_corr,c_corr,d_corr)
    p_val, test_name = calcular_p_valor(a,b,c,d)

    st.markdown(interpretar_resultados(rr, rr_l, rr_u, or_, or_l, or_u, rd, rd_l, rd_u, p_val, test_name))

    plot_forest(rr, rr_l, rr_u, or_, or_l, or_u)
    plot_barras_expuestos(a,b,c,d)

elif seccion == "Visualización de Datos":
    st.header("📊 Visualización de Datos")

    st.info("Carga aquí un archivo CSV con datos epidemiológicos para crear gráficos personalizados.")
    archivo = st.file_uploader("Sube tu archivo CSV", type=["csv"])
    if archivo:
        df = pd.read_csv(archivo)
        st.write("Vista previa de datos:")
        st.dataframe(df.head())

        columnas = df.columns.tolist()
        x_axis = st.selectbox("Selecciona columna para eje X", columnas)
        y_axis = st.selectbox("Selecciona columna para eje Y", columnas)

        if st.button("Generar gráfico de barras"):
            fig, ax = plt.subplots()
            df.plot(kind='bar', x=x_axis, y=y_axis, ax=ax, color='#0d3b66')
            st.pyplot(fig, use_container_width=True)

elif seccion == "Multimedia YouTube":
    st.header("📺 Videos recomendados de YouTube")
    for titulo, url in videos_youtube.items():
        st.markdown(f"**{titulo}**")
        st.video(url)

elif seccion == "Gamificación":
    st.header("🎮 Gamificación: Test de Conocimientos")

    nivel = st.selectbox("Elige tu nivel", list(niveles.keys()))
    preguntas_nivel = niveles[nivel]
    aciertos = 0

    for i, pregunta in enumerate(preguntas_nivel):
        st.subheader(f"Pregunta {i+1}")
        respuesta = st.radio(pregunta["pregunta"], pregunta["opciones"], key=f"gam_{nivel}_{i}")
        if st.button(f"Verificar respuesta {i+1}", key=f"btn_gam_{nivel}_{i}"):
            if respuesta == pregunta["respuesta_correcta"]:
                st.success("✅ Correcto")
                aciertos += 1
                if aciertos == len(preguntas_nivel):
                    mostrar_confeti()
            else:
                st.error(f"❌ Incorrecto. Respuesta correcta: {pregunta['respuesta_correcta']}")

elif seccion == "Chat Epidemiológico":
    st.header("💬 Chat Epidemiológico")
    if GENAI_AVAILABLE:
        st.info("Funcionalidad en construcción: próximamente podrás preguntar dudas epidemiológicas aquí.")
    else:
        st.warning("El módulo de IA no está disponible. Asegúrate de tener la librería google.generativeai instalada.")

# ---------- FOOTER ----------
st.markdown("""
<footer>
    <hr>
    Creado con ❤️ por <strong>Yolanda Muvdi</strong> - MSc Epidemiología |  
    📧 <a href="mailto:ymuvdi@gmail.com">ymuvdi&#64;gmail.com</a> |  
    🔗 <a href="https://www.linkedin.com/in/yolanda-paola-muvdi-muvdi-778b73152/" target="_blank">LinkedIn</a>
</footer>
""", unsafe_allow_html=True)




