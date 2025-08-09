import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import math
import os

# Import Google Generative AI (Gemini)
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

# Variables globales menú
SECCIONES = [
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

# Inicializar estado sesión
if "seccion" not in st.session_state:
    st.session_state.seccion = None

if "puntaje_gamificacion" not in st.session_state:
    st.session_state.puntaje_gamificacion = 0

if "chat_messages" not in st.session_state:
    st.session_state.chat_messages = []

# --- Funciones para cargar contenido (Markdown y Python) ---
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

# --- Función para mostrar footer fijo con info de Yolanda ---
def mostrar_footer():
    st.markdown("""
    <style>
    .footer {
        position: fixed;
        left: 0;
        bottom: 0;
        width: 100%;
        background-color: #0b2e58;
        color: white;
        text-align: right;
        padding: 0.5rem 1rem;
        font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
        font-size: 0.9rem;
        z-index: 9999;
    }
    .footer a {
        color: #a6c8ff;
        text-decoration: none;
        margin-left: 1rem;
    }
    .footer a:hover {
        text-decoration: underline;
    }
    </style>
    <div class="footer">
        Creado por <b>Yolanda Muvdi</b> | Email: ymuvdi&#64;gmail.com
        <a href="https://www.linkedin.com/in/yolanda-paola-muvdi-muvdi-778b73152/" target="_blank">LinkedIn</a>
    </div>
    """, unsafe_allow_html=True)

# --- Splash inicial simple ---
def mostrar_splash():
    st.markdown("""
    <style>
    .splash {
        display: flex;
        flex-direction: column;
        align-items: center;
        justify-content: center;
        height: 80vh;
        font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
        color: #0d3b66;
    }
    .splash h1 {
        font-size: 3.5rem;
        margin-bottom: 0.2rem;
    }
    .splash p {
        font-size: 1.3rem;
        margin-bottom: 2rem;
        font-weight: 500;
    }
    select {
        font-size: 1.1rem;
        padding: 0.5rem 0.7rem;
        border-radius: 6px;
        border: 2px solid #0d3b66;
        width: 300px;
    }
    </style>
    """, unsafe_allow_html=True)

    st.markdown('<div class="splash">', unsafe_allow_html=True)
    st.markdown("🧪 <h1>Bienvenido/a a Epidemiología 101</h1>", unsafe_allow_html=True)
    st.markdown("<p>¿Qué quieres aprender hoy?</p>", unsafe_allow_html=True)

    opcion = st.selectbox("Selecciona la sección", [""] + SECCIONES)

    st.markdown('</div>', unsafe_allow_html=True)

    if opcion and opcion != "":
        st.session_state.seccion = opcion
        st.experimental_rerun()

# --- Función para menú lateral ---
def mostrar_sidebar():
    with st.sidebar:
        st.markdown("## Menú de Secciones")
        seccion = st.radio("Navega por la app:", SECCIONES, index=SECCIONES.index(st.session_state.seccion) if st.session_state.seccion in SECCIONES else 0)
        if seccion != st.session_state.seccion:
            st.session_state.seccion = seccion
            st.experimental_rerun()

        st.markdown("---")
        st.markdown("""
        <small style="color:#0d3b66;">
        Creado por <b>Yolanda Muvdi</b><br>
        ymuvdi&#64;gmail.com<br>
        <a href="https://www.linkedin.com/in/yolanda-paola-muvdi-muvdi-778b73152/" target="_blank" style="color:#0d3b66;">LinkedIn</a>
        </small>
        """, unsafe_allow_html=True)

# --- Funciones para cada sección ---
def mostrar_inicio():
    st.title("🧪 Inicio")
    st.write("¡Bienvenido/a a Epidemiología 101! Explora las secciones usando el menú lateral.")

def mostrar_conceptos_basicos():
    st.title("📌 Conceptos Básicos")
    contenido = cargar_md("contenido/conceptosbasicos.md")
    if contenido:
        st.markdown(contenido)
    else:
        st.info("Agrega el archivo 'contenido/conceptosbasicos.md' para mostrar el contenido.")

def mostrar_medidas_asociacion():
    st.title("📈 Medidas de Asociación")
    contenido = cargar_md("contenido/medidas_completas.md")
    if contenido:
        st.markdown(contenido)
    else:
        st.info("Agrega el archivo 'contenido/medidas_completas.md' para mostrar el contenido.")

def mostrar_disenos_estudio():
    st.title("📊 Diseños de Estudio")
    contenido = cargar_md("contenido/disenos_completos.md")
    if contenido:
        st.markdown(contenido)
    else:
        st.info("Agrega el archivo 'contenido/disenos_completos.md' para mostrar el contenido.")

def mostrar_sesgos_errores():
    st.title("⚠️ Sesgos y Errores")
    contenido = cargar_md("contenido/sesgos_completos.md")
    if contenido:
        st.markdown(contenido)
    else:
        st.info("Agrega el archivo 'contenido/sesgos_completos.md' para mostrar el contenido.")

def mostrar_glosario_interactivo():
    st.title("📚 Glosario Interactivo")
    glosario = cargar_py_variable("contenido/glosario_completo.py", "glosario")
    if glosario:
        for termino, definicion in glosario.items():
            with st.expander(termino):
                st.write(definicion)
    else:
        st.info("Agrega 'contenido/glosario_completo.py' con variable glosario.")

def mostrar_ejercicios_practicos():
    st.title("🧪 Ejercicios Prácticos")
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
            st.success("🌟 ¡Felicidades! Completaste todos los ejercicios.")
    else:
        st.info("Agrega 'contenido/ejercicios_completos.py' con variable preguntas.")

def mostrar_tablas_2x2():
    st.title("📊 Tablas 2x2 y Cálculos Epidemiológicos")

    # Valores por defecto
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

    if st.button("Calcular medidas"):
        a, b, c, d = st.session_state.a, st.session_state.b, st.session_state.c, st.session_state.d
        total = a+b+c+d
        if total == 0:
            st.error("Por favor ingresa valores mayores que cero.")
        else:
            # Corrección para ceros
            a_, b_, c_, d_ = a, b, c, d
            corregido = False
            if 0 in [a,b,c,d]:
                a_ += 0.5
                b_ += 0.5
                c_ += 0.5
                d_ += 0.5
                corregido = True

            risk1 = a_/(a_+b_)
            risk2 = c_/(c_+d_)
            rr = risk1/risk2
            se_log_rr = math.sqrt(1/a_ - 1/(a_+b_) + 1/c_ - 1/(c_+d_))
            z = norm.ppf(0.975)
            rr_l = math.exp(math.log(rr) - z*se_log_rr)
            rr_u = math.exp(math.log(rr) + z*se_log_rr)

            or_ = (a_*d_)/(b_*c_)
            se_log_or = math.sqrt(1/a_ + 1/b_ + 1/c_ + 1/d_)
            or_l = math.exp(math.log(or_) - z*se_log_or)
            or_u = math.exp(math.log(or_) + z*se_log_or)

            rd = risk1 - risk2
            se_rd = math.sqrt((risk1*(1-risk1))/(a_+b_) + (risk2*(1-risk2))/(c_+d_))
            rd_l = rd - z*se_rd
            rd_u = rd + z*se_rd

            # P valor
            if SCIPY_AVAILABLE:
                table = np.array([[a,b],[c,d]])
                chi2, p_val, dof, expected = chi2_contingency(table, correction=False)
                test_name = "Chi-cuadrado sin corrección"
                if (expected < 5).any():
                    _, p_val = fisher_exact(table)
                    test_name = "Fisher exacto"
            else:
                p_val = None
                test_name = "scipy no disponible"

            st.markdown(f"""
            **Resultados Epidemiológicos:**

            - Riesgo Relativo (RR): {rr:.3f} (IC95% {rr_l:.3f} - {rr_u:.3f})  
            - Odds Ratio (OR): {or_:.3f} (IC95% {or_l:.3f} - {or_u:.3f})  
            - Diferencia de Riesgos (RD): {rd:.3f} (IC95% {rd_l:.3f} - {rd_u:.3f})  
            - Valor p ({test_name}): {p_val if p_val else 'N/A'}  
            """)

            if p_val and p_val < 0.05:
                st.success("🎯 Asociación estadísticamente significativa (p < 0.05).")
            elif p_val:
                st.warning("⚠️ No significativa estadísticamente (p ≥ 0.05).")
            else:
                st.info("⚠️ No se pudo calcular valor p.")

            if corregido:
                st.info("Se aplicó corrección de 0.5 por ceros en tabla.")

def mostrar_visualizacion_datos():
    st.title("📊 Visualización de Datos")
    uploaded_file = st.file_uploader("Carga un archivo CSV para gráficos exploratorios", type=["csv"])
    if uploaded_file:
        try:
            df = pd.read_csv(uploaded_file)
            st.write("Vista previa:")
            st.dataframe(df.head())
            columnas = df.columns.tolist()
            columna = st.selectbox("Selecciona columna para gráfico", columnas)
            tipo_grafico = st.radio("Tipo de gráfico", ("Boxplot", "Histograma"))
            if tipo_grafico == "Boxplot":
                fig, ax = plt.subplots()
                ax.boxplot(df[columna].dropna())
                ax.set_title(f"Boxplot de {columna}")
                st.pyplot(fig)
            else:
                fig, ax = plt.subplots()
                ax.hist(df[columna].dropna(), bins=20, color='#0d3b66', alpha=0.7)
                ax.set_title(f"Histograma de {columna}")
                st.pyplot(fig)
        except Exception as e:
            st.error(f"Error leyendo archivo CSV: {e}")

def mostrar_multimedia_youtube():
    st.title("📺 Multimedia - Videos Epidemiología")
    videos = {
        "Introducción a Epidemiología": "https://www.youtube.com/watch?v=Z4JDr11G0N4",
        "Medidas de Asociación": "https://www.youtube.com/watch?v=pD9Oa88kqQI",
        "Diseños de Estudios Epidemiológicos": "https://www.youtube.com/watch?v=sP4kzERW6wo",
        "Sesgos y Errores Comunes": "https://www.youtube.com/watch?v=RrKhv8OdLmY"
    }
    for nombre, url in videos.items():
        st.markdown(f"▶️ [{nombre}]({url})")

def mostrar_gamificacion():
    st.title("🎮 Gamificación")
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
                "opciones": ["Chi-cuadrado", "T-test", "Fisher exacto", "ANOVA"],
                "respuesta_correcta": "Fisher exacto"
            }
        ],
        "Avanzado": [
            {
                "pregunta": "¿Qué es el sesgo de información?",
                "opciones": ["Error sistemático en medición", "Error aleatorio", "Confusión", "Sesgo de selección"],
                "respuesta_correcta": "Error sistemático en medición"
            },
            {
                "pregunta": "¿Qué significa un IC 95% que incluye 1 para un OR?",
                "opciones": ["Significativo", "No significativo", "Mayor riesgo", "Menor riesgo"],
                "respuesta_correcta": "No significativo"
            }
        ]
    }

    # Detectar nivel según puntaje (ejemplo simple)
    if st.session_state.puntaje_gamificacion < 2:
        nivel_actual = "Básico"
    elif st.session_state.puntaje_gamificacion < 4:
        nivel_actual = "Intermedio"
    else:
        nivel_actual = "Avanzado"

    st.markdown(f"**Nivel actual:** {nivel_actual}")

    preguntas = niveles[nivel_actual]
    puntaje = 0

    for i, q in enumerate(preguntas):
        st.write(f"**Pregunta {i+1}:** {q['pregunta']}")
        respuesta = st.radio("Selecciona una opción:", q['opciones'], key=f"gam_{nivel_actual}_{i}")
        if st.button(f"Verificar respuesta {i+1}", key=f"btn_gam_{nivel_actual}_{i}"):
            if respuesta == q['respuesta_correcta']:
                st.success("¡Correcto!")
                puntaje += 1
                st.session_state.puntaje_gamificacion += 1
                st.balloons()
            else:
                st.error(f"Incorrecto. La respuesta correcta es: {q['respuesta_correcta']}")

    st.write(f"**Puntaje en esta sesión:** {st.session_state.puntaje_gamificacion}")

def mostrar_chat_epidemiologico():
    st.title("💬 Chat Epidemiológico")

    st.write("Funcionalidad simple de chat con Gemini AI (si disponible).")

    user_input = st.text_input("Escribe tu pregunta aquí:")

    if user_input:
        st.session_state.chat_messages.append({"role": "user", "content": user_input})

        if GENAI_AVAILABLE:
            try:
                response = genai.chat.create(
                    model="models/chat-bison-001",
                    messages=st.session_state.chat_messages,
                )
                respuesta = response.last.message.content
            except Exception as e:
                respuesta = f"Error al generar respuesta: {e}"
        else:
            respuesta = "Gemini AI no disponible."

        st.session_state.chat_messages.append({"role": "assistant", "content": respuesta})

    for msg in st.session_state.chat_messages:
        if msg['role'] == "user":
            st.markdown(f"**Tú:** {msg['content']}")
        else:
            st.markdown(f"**Bot:** {msg['content']}")

# --- MAIN ---
def main():
    st.set_page_config(
        page_title="Epidemiología 101",
        layout="wide",
        initial_sidebar_state="expanded"
    )

    # Si no hay sección elegida, mostrar splash
    if st.session_state.seccion is None:
        mostrar_splash()
        mostrar_footer()
    else:
        mostrar_sidebar()

        # Contenedor principal ancho
        st.markdown(
            """
            <style>
            .main-content {
                padding: 1rem 3rem;
                max-width: 900px;
                margin-left: auto;
                margin-right: auto;
                font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            }
            </style>
            """,
            unsafe_allow_html=True
        )

        with st.container():
            if st.session_state.seccion == "Inicio":
                mostrar_inicio()
            elif st.session_state.seccion == "Conceptos Básicos":
                mostrar_conceptos_basicos()
            elif st.session_state.seccion == "Medidas de Asociación":
                mostrar_medidas_asociacion()
            elif st.session_state.seccion == "Diseños de Estudio":
                mostrar_disenos_estudio()
            elif st.session_state.seccion == "Sesgos y Errores":
                mostrar_sesgos_errores()
            elif st.session_state.seccion == "Glosario Interactivo":
                mostrar_glosario_interactivo()
            elif st.session_state.seccion == "Ejercicios Prácticos":
                mostrar_ejercicios_practicos()
            elif st.session_state.seccion == "Tablas 2x2 y Cálculos":
                mostrar_tablas_2x2()
            elif st.session_state.seccion == "Visualización de Datos":
                mostrar_visualizacion_datos()
            elif st.session_state.seccion == "Multimedia YouTube":
                mostrar_multimedia_youtube()
            elif st.session_state.seccion == "Gamificación":
                mostrar_gamificacion()
        elif st.session_state.seccion == "Chat Epidemiológico":
    mostrar_chat_epidemiologico()
        mostrar_footer()


if __name__ == "__main__":
    main()



