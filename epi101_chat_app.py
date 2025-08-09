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

# -------------------- CONFIGURACIÓN GENERAL --------------------
st.set_page_config(page_title="🧠 Epidemiología 101 - Masterclass", page_icon="🧪", layout="wide", initial_sidebar_state="expanded")

# Estilos CSS (limpio, accesible, responsive)
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
</style>
""", unsafe_allow_html=True)

# -------------------- SIDEBAR CON NAVEGACIÓN --------------------
st.sidebar.title("🧪 Epidemiología 101")
st.sidebar.markdown("""
👩‍⚕️ Creado por Yolanda Muvdi, Enfermera Epidemióloga  
📧 [ymuvdi@gmail.com](mailto:ymuvdi@gmail.com)  
🔗 [LinkedIn](https://www.linkedin.com/in/yolanda-paola-muvdi-muvdi-778b73152/)
""")

menu = st.sidebar.radio("Ir a sección:", [
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
])

# -------------------- FUNCIONES DE CARGA --------------------
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

# -------------------- CÁLCULOS EPIDEMIOLÓGICOS --------------------
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

# -------------------- INTERPRETACIÓN --------------------
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

# -------------------- GRÁFICOS --------------------
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

# -------------------- GAMIFICACIÓN --------------------
def mostrar_insignia(tipo):
    insignias = {
        "inicio": "🎓 Bienvenida a Epidemiología 101. ¡Empecemos la aventura científica! 🧬",
        "ejercicio_correcto": "🏅 ¡Genial! Has desbloqueado una insignia por responder correctamente. Sigue así 🔥",
        "completo": "🌟 ¡Felicidades! Has completado todos los ejercicios y desbloqueado el certificado digital. 📜"
    }
    msg = insignias.get(tipo, "🎉 ¡Bien hecho!")
    st.toast(msg, icon="🎉")

# -------------------- CHAT GEMINI --------------------
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

# -------------------- CONTENIDO PRINCIPAL --------------------

if menu == "Inicio":
    st.title("🧪 Bienvenido/a a Epidemiología 101")
    st.markdown("""
    Bienvenido/a a la masterclass interactiva de Epidemiología creada por **Yolanda Muvdi**, Enfermera Epidemióloga.  
    Explora las secciones del menú para aprender conceptos, hacer ejercicios, usar calculadoras epidemiológicas y más.
    """)

elif menu == "Conceptos Básicos":
    st.header("📌 Conceptos Básicos")
    contenido = cargar_md("contenido/conceptosbasicos.md")
    if contenido:
        st.markdown(contenido)
    else:
        st.info("Agrega el archivo 'contenido/conceptosbasicos.md' para mostrar el contenido.")

elif menu == "Medidas de Asociación":
    st.header("📈 Medidas de Asociación")
    contenido = cargar_md("contenido/medidas_completas.md")
    if contenido:
        st.markdown(contenido)
    else:
        st.info("Agrega el archivo 'contenido/medidas_completas.md' para mostrar el contenido.")

elif menu == "Diseños de Estudio":
    st.header("📊 Diseños de Estudio")
    contenido = cargar_md("contenido/disenos_completos.md")
    if contenido:
        st.markdown(contenido)
    else:
        st.info("Agrega el archivo 'contenido/disenos_completos.md' para mostrar el contenido.")

elif menu == "Sesgos y Errores":
    st.header("⚠️ Sesgos y Errores")
    contenido = cargar_md("contenido/sesgos_completos.md")
    if contenido:
        st.markdown(contenido)
    else:
        st.info("Agrega el archivo 'contenido/sesgos_completos.md' para mostrar el contenido.")

elif menu == "Glosario Interactivo":
    st.header("📚 Glosario Interactivo")
    glosario = cargar_py_variable("contenido/glosario_completo.py", "glosario")
    if glosario:
        for termino, definicion in glosario.items():
            with st.expander(termino):
                st.write(definicion)
    else:
        st.info("Agrega 'contenido/glosario_completo.py' con variable `glosario`.")

elif menu == "Ejercicios Prácticos":
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
                    mostrar_insignia("ejercicio_correcto")
                else:
                    st.error(f"❌ Incorrecto. La respuesta correcta es: {q['respuesta_correcta']}")
        if respuestas_correctas == len(preguntas) and len(preguntas) > 0:
            mostrar_insignia("completo")
    else:
        st.info("Agrega 'contenido/ejercicios_completos.py' con variable `preguntas`.")

elif menu == "Tablas 2x2 y Cálculos":
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
        st.session_state.a = st.number_input("Casos expuestos (a)", min_value=0, value=st.session_state.a, step=1, key="input_a")
        st.session_state.b = st.number_input("No casos expuestos (b)", min_value=0, value=st.session_state.b, step=1, key="input_b")
    with col2:
        st.session_state.c = st.number_input("Casos no expuestos (c)", min_value=0, value=st.session_state.c, step=1, key="input_c")
        st.session_state.d = st.number_input("No casos no expuestos (d)", min_value=0, value=st.session_state.d, step=1, key="input_d")

    if st.button("Calcular medidas y mostrar gráficos"):
        a, b, c, d = st.session_state.a, st.session_state.b, st.session_state.c, st.session_state.d
        total = a+b+c+d
        if total == 0:
            st.error("Por favor ingresa valores mayores que cero.")
        else:
            a_, b_, c_, d_, corregido = corregir_ceros(a,b,c,d)
            rr, rr_l, rr_u = ic_riesgo_relativo(a_,b_,c_,d_)
            or_, or_l, or_u = ic_odds_ratio(a_,b_,c_,d_)
            rd, rd_l, rd_u = diferencia_riesgos(a_,b_,c_,d_)
            p_val, test_name = calcular_p_valor(int(a_), int(b_), int(c_), int(d_))

            st.markdown(interpretar_resultados(rr, rr_l, rr_u, or_, or_l, or_u, rd, rd_l, rd_u, p_val, test_name))

            if corregido:
                st.warning("Se aplicó corrección de 0.5 en celdas con valor 0 para cálculos.")

            plot_forest(rr, rr_l, rr_u, or_, or_l, or_u)
            plot_barras_expuestos(a,b,c,d)

elif menu == "Visualización de Datos":
    st.header("📊 Visualización de Datos")

    uploaded_file = st.file_uploader("Carga un archivo CSV para gráficos exploratorios", type=["csv"])

    if uploaded_file:
        try:
            df = pd.read_csv(uploaded_file)
            st.write("Vista previa de los datos cargados:")
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

elif menu == "Multimedia YouTube":
    st.header("📺 Multimedia - Videos Epidemiología")

    videos = {
        "Introducción a Epidemiología": "https://www.youtube.com/watch?v=Z4JDr11G0N4",
        "Medidas de Asociación": "https://www.youtube.com/watch?v=pD9Oa88kqQI",
        "Diseños de Estudios Epidemiológicos": "https://www.youtube.com/watch?v=sP4kzERW6wo",
        "Sesgos y Errores Comunes": "https://www.youtube.com/watch?v=RrKhv8OdLmY"
    }

    for nombre, url in videos.items():
        st.write(f"▶️ [{nombre}]({url})")

elif menu == "Gamificación":
    st.header("🎮 Gamificación - Elige tu nivel epidemiológico")

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

    # Inicializar puntaje en sesión
    if "puntaje_gamificacion" not in st.session_state:
        st.session_state.puntaje_gamificacion = 0

    nivel_elegido = st.selectbox("Selecciona el nivel:", options=list(niveles.keys()))
    preguntas = niveles[nivel_elegido]

    st.write(f"### Nivel seleccionado: {nivel_elegido}")

    for i, pregunta in enumerate(preguntas):
        st.subheader(f"Pregunta {i+1}")
        seleccion = st.radio(pregunta["pregunta"], pregunta["opciones"], key=f"preg_{nivel_elegido}_{i}")

        if st.button(f"Verificar respuesta {i+1}", key=f"btn_{nivel_elegido}_{i}"):
            if seleccion == pregunta["respuesta_correcta"]:
                st.success("✅ Correcto!")
                st.session_state.puntaje_gamificacion += 1
            else:
                st.error(f"❌ Incorrecto. Respuesta correcta: {pregunta['respuesta_correcta']}")

    st.markdown(f"### Puntaje acumulado: {st.session_state.puntaje_gamificacion} / {len(preguntas)}")

    if st.session_state.puntaje_gamificacion == len(preguntas) and len(preguntas) > 0:
        st.balloons()
        st.success(f"🎉 ¡Nivel {nivel_elegido} completado! Ahora desbloqueas un mini-juego.")

        if nivel_elegido == "Básico":
            st.write("**Mini-juego:** Adivina el término epidemiológico a partir de la definición.")
            definiciones = {
                "Incidencia": "Número de casos nuevos de una enfermedad en un periodo de tiempo.",
                "Prevalencia": "Número total de casos existentes en un momento dado.",
                "Cohorte": "Grupo de personas seguidas en el tiempo para estudiar la incidencia."
            }
            termino_correcto = np.random.choice(list(definiciones.keys()))
            definicion = definiciones[termino_correcto]
            respuesta_usuario = st.text_input("¿Cuál es el término?", key="input_mini_basico")

            if st.button("Verificar término"):
                if respuesta_usuario.strip().lower() == termino_correcto.lower():
                    st.success("¡Correcto! Eres un crack básico.")
                else:
                    st.error(f"Incorrecto, la respuesta era: {termino_correcto}")

        elif nivel_elegido == "Intermedio":
            st.write("**Mini-juego:** Calcula el Riesgo Relativo con datos dados.")
            st.write("Datos: a=30, b=70, c=15, d=85")
            rr = (30/(30+70)) / (15/(15+85))
            rr_user = st.number_input("Ingresa el RR calculado (3 decimales):", format="%.3f", key="input_mini_intermedio")

            if st.button("Verificar cálculo RR"):
                if abs(rr_user - rr) < 0.001:
                    st.success(f"¡Correcto! RR = {rr:.3f}")
                else:
                    st.error(f"Incorrecto. El RR correcto es {rr:.3f}")

        elif nivel_elegido == "Avanzado":
            st.write("**Mini-juego:** Interpreta el siguiente resultado.")
            st.write("OR = 1.8 (IC95% 1.2-2.7), p=0.03")
            interpretacion = st.radio("¿Qué significa?", options=[
                "Asociación estadísticamente significativa y positiva",
                "No hay asociación",
                "Error en el análisis",
                "Asociación negativa"
            ], key="input_mini_avanzado")

            if st.button("Verificar interpretación"):
                if interpretacion == "Asociación estadísticamente significativa y positiva":
                    st.success("¡Correcto! Bien interpretado.")
                else:
                    st.error("Incorrecto, esa no es la interpretación correcta.")

elif menu == "Chat Epidemiológico":
    st.header("💬 Chat Epidemiológico - Pregunta lo que quieras")

    if "chat_messages" not in st.session_state:
        st.session_state.chat_messages = []

    user_input = st.text_area("Escribe tu pregunta o tema", height=100)

    if st.button("Enviar"):
        if user_input.strip():
            st.session_state.chat_messages.append({"role": "user", "content": user_input.strip()})

            respuesta = chat_with_gemini(st.session_state.chat_messages)
            st.session_state.chat_messages.append({"role": "assistant", "content": respuesta})

    for i, msg in enumerate(st.session_state.chat_messages):
        if msg["role"] == "user":
            st.markdown(f"**Tú:** {msg['content']}")
        else:
            st.markdown(f"**Epidemiología AI:** {msg['content']}")

else:
    st.info("Selecciona una sección en el menú lateral para comenzar.")





