import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import math
import os

# Estadística y tests
try:
    from scipy.stats import chi2_contingency, fisher_exact, norm
    SCIPY_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False

# Gemini AI
try:
    import google.generativeai as genai
    GENAI_AVAILABLE = True
except ImportError:
    GENAI_AVAILABLE = False

# Configurar API Gemini si está disponible
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

# -------------------- CONFIGURACIÓN GENERAL --------------------
st.set_page_config(
    page_title="🧠 Epidemiología 101 - Masterclass",
    page_icon="🧪",
    layout="wide",
    initial_sidebar_state="expanded"
)

# CSS diseño profesional
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
    /* Sidebar */
    .sidebar .sidebar-content {
        padding-top: 1rem;
    }
    /* Botones grandes móvil */
    @media (max-width: 768px) {
        .stButton>button {
            width: 100% !important;
            font-size: 1.2rem !important;
        }
    }
    /* Footer fijo */
    footer {
        display: none;
    }
    .custom-footer {
        position: fixed;
        bottom: 0;
        width: 100%;
        background-color: #0d3b66;
        color: white;
        text-align: center;
        padding: 0.5rem 1rem;
        font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
        font-size: 0.9rem;
        z-index: 9999;
    }
    .custom-footer a {
        color: #a6c8ff;
        text-decoration: none;
        margin-left: 0.6rem;
    }
    .custom-footer a:hover {
        text-decoration: underline;
    }
</style>
""", unsafe_allow_html=True)

# -------------------- FUNCIONES --------------------

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
*Resultados Epidemiológicos:*

•⁠  ⁠Riesgo Relativo (RR): {rr:.3f} (IC95% {rr_l:.3f} - {rr_u:.3f})  
•⁠  ⁠Odds Ratio (OR): {or_:.3f} (IC95% {or_l:.3f} - {or_u:.3f})  
•⁠  ⁠Diferencia de Riesgos (RD): {rd:.3f} (IC95% {rd_l:.3f} - {rd_u:.3f})  
•⁠  ⁠Valor p ({test_name}): {p_val:.4f}  

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

# -------------------- GAMIFICACIÓN EXTENDIDA --------------------

def mostrar_insignia(tipo):
    insignias = {
        "inicio": "🎓 Bienvenida a Epidemiología 101. ¡Empecemos la aventura científica! 🧬",
        "ejercicio_correcto": "🏅 ¡Genial! Has desbloqueado una insignia por responder correctamente. Sigue así 🔥",
        "completo": "🌟 ¡Felicidades! Has completado todos los ejercicios y desbloqueado el certificado digital. 📜"
    }
    msg = insignias.get(tipo, "🎉 ¡Bien hecho!")
    st.toast(msg, icon="🎉")

def obtener_nivel(puntaje, total):
    porcentaje = puntaje / total if total > 0 else 0
    if porcentaje == 1:
        return "Pro Epidemiología 🏆", "¡Eres un/a maestro/a de la epidemiología! 👑"
    elif porcentaje >= 0.75:
        return "Avanzado", "¡Muy buen trabajo! Estás dominando la materia."
    elif porcentaje >= 0.5:
        return "Intermedio", "Vas por buen camino, sigue estudiando."
    else:
        return "Novato", "Sigue esforzándote, ¡puedes mejorar!"

def mostrar_gamificacion(preguntas):
    total_preguntas = len(preguntas)
    if "respuestas_correctas" not in st.session_state:
        st.session_state.respuestas_correctas = 0
    if "respondidas" not in st.session_state:
        st.session_state.respondidas = [False] * total_preguntas

    puntaje = st.session_state.respuestas_correctas

    st.markdown(f"### 🎯 Gamificación y Progreso")
    st.markdown(f"**Preguntas correctas:** {puntaje} de {total_preguntas}")

    nivel, mensaje = obtener_nivel(puntaje, total_preguntas)
    st.markdown(f"**Nivel:** {nivel} - _{mensaje}_")

    if puntaje == total_preguntas and total_preguntas > 0:
        st.balloons()
        mostrar_insignia("completo")
        st.success("🎉 ¡Has completado todos los ejercicios! Puedes descargar tu certificado digital.")
        if st.download_button("📜 Descargar certificado digital", data=f"Certificado Epidemiología 101 - Nivel: {nivel}\nUsuario: Yolanda Muvdi\nPuntaje: {puntaje}/{total_preguntas}\nFecha: {pd.Timestamp.now().date()}", file_name="certificado_epidemiologia_101.txt"):
            st.info("Certificado descargado. ¡Felicidades!")

# -------------------- SIDEBAR --------------------

st.sidebar.title("🧪 Epidemiología 101")
st.sidebar.markdown("""
👩‍⚕️ Creado por Yolanda Muvdi, Enfermera Epidemióloga  
📧 [ymuvdi@gmail.com](mailto:ymuvdi@gmail.com)  
🔗 [LinkedIn](https://www.linkedin.com/in/yolanda-paola-muvdi-muvdi-778b73152/)
""")

menu_items = {
    "📌 Conceptos Básicos": "conceptos_basicos",
    "📈 Medidas de Asociación": "medidas_asociacion",
    "📊 Diseños de Estudio": "disenos_estudio",
    "⚠️ Sesgos y Errores": "sesgos_errores",
    "📚 Glosario Interactivo": "glosario",
    "🧪 Ejercicios Prácticos": "ejercicios",
    "📊 Tablas 2x2 y Cálculos": "tablas_2x2",
    "📊 Visualización de Datos": "visualizacion_datos",
    "🎥 Multimedia YouTube": "multimedia",
    "🤖 Chat Epidemiológico": "chat_epidemiologico"
}

seleccion_sidebar = st.sidebar.radio("Ir a sección:", list(menu_items.keys()))
menu = menu_items[seleccion_sidebar]

# Dropdown principal (opcional, para UX)
seleccion_dropdown = st.selectbox(
    "Selecciona sección:",
    options=list(menu_items.keys()),
    index=list(menu_items.keys()).index(seleccion_sidebar),
    format_func=lambda x: x
)
if menu_items[seleccion_dropdown] != menu:
    menu = menu_items[seleccion_dropdown]

# -------------------- SECCIONES --------------------

if menu == "conceptos_basicos":
    st.header("📌 Conceptos Básicos")
    contenido = cargar_md("contenido/conceptosbasicos.md")
    if contenido:
        st.markdown(contenido)
    else:
        st.info("Agrega el archivo 'contenido/conceptosbasicos.md' para mostrar el contenido.")

elif menu == "medidas_asociacion":
    st.header("📈 Medidas de Asociación")
    contenido = cargar_md("contenido/medidas_completas.md")
    if contenido:
        st.markdown(contenido)
    else:
        st.info("Agrega el archivo 'contenido/medidas_completas.md' para mostrar el contenido.")

elif menu == "disenos_estudio":
    st.header("📊 Diseños de Estudio")
    contenido = cargar_md("contenido/disenos_completos.md")
    if contenido:
        st.markdown(contenido)
    else:
        st.info("Agrega el archivo 'contenido/disenos_completos.md' para mostrar el contenido.")

elif menu == "sesgos_errores":
    st.header("⚠️ Sesgos y Errores")
    contenido = cargar_md("contenido/sesgos_completos.md")
    if contenido:
        st.markdown(contenido)
    else:
        st.info("Agrega el archivo 'contenido/sesgos_completos.md' para mostrar el contenido.")

elif menu == "glosario":
    st.header("📚 Glosario Interactivo")
    glosario = cargar_py_variable("contenido/glosario_completo.py", "glosario")
    if glosario:
        for termino, definicion in glosario.items():
            with st.expander(termino):
                st.write(definicion)
    else:
        st.info("Agrega 'contenido/glosario_completo.py' con variable ⁠ glosario ⁠.")

elif menu == "ejercicios":
    st.header("🧪 Ejercicios Prácticos")
    preguntas = cargar_py_variable("contenido/ejercicios_completos.py", "preguntas")
    if preguntas:
        if "respuestas_correctas" not in st.session_state:
            st.session_state.respuestas_correctas = 0
        if "respondidas" not in st.session_state:
            st.session_state.respondidas = [False] * len(preguntas)

        for i, q in enumerate(preguntas):
            st.subheader(f"Pregunta {i+1}")
            respuesta = st.radio(q['pregunta'], q['opciones'], key=f"q{i}")
            if st.button(f"Verificar respuesta {i+1}", key=f"btn_{i}"):
                if respuesta == q['respuesta_correcta']:
                    if not st.session_state.respondidas[i]:
                        st.session_state.respuestas_correctas += 1
                        st.session_state.respondidas[i] = True
                        mostrar_insignia("ejercicio_correcto")
                    st.success("✅ Correcto")
                else:
                    st.error(f"❌ Incorrecto. La respuesta correcta es: {q['respuesta_correcta']}")
        mostrar_gamificacion(preguntas)
    else:
        st.info("Agrega 'contenido/ejercicios_completos.py' con variable ⁠ preguntas ⁠.")

elif menu == "tablas_2x2":
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

elif menu == "visualizacion_datos":
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
            st.error(f"Error al procesar archivo: {e}")
    else:
        st.info("Carga un archivo CSV para empezar.")

elif menu == "multimedia":
    st.header("🎥 Videos Educativos de Epidemiología")
    st.info("Aquí puedes encontrar recursos audiovisuales para complementar tu aprendizaje.")

    videos = {
        "Introducción a la Epidemiología": "https://www.youtube.com/embed/5uj0pPU-68E",
        "Medidas de Asociación": "https://www.youtube.com/embed/0nZPxtYQDrQ",
        "Diseños de Estudios Epidemiológicos": "https://www.youtube.com/embed/NMRJ9iJZynA",
        "Sesgos en Epidemiología": "https://www.youtube.com/embed/o7hvLv_lcVk"
    }

    for title, url in videos.items():
        st.subheader(title)
        st.video(url)

elif menu == "chat_epidemiologico":
    st.header("🤖 Chat Epidemiológico con Gemini AI")
    st.info("Pregunta cualquier duda epidemiológica. Gemini responde clara, precisa y con humor inteligente.")

    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []

    pregunta = st.text_input("Escribe tu pregunta aquí:", key="chat_input")

    if st.button("Enviar"):
        if pregunta.strip():
            st.session_state.chat_history.append({"role":"user", "content": pregunta})
            with st.spinner("Gemini está respondiendo..."):
                respuesta = chat_with_gemini(st.session_state.chat_history)
            st.session_state.chat_history.append({"role":"assistant", "content": respuesta})

    for msg in st.session_state.chat_history:
        if msg["role"] == "user":
            st.markdown(f"**Tú:** {msg['content']}")
        else:
            st.markdown(f"**Gemini:** {msg['content']}")

# -------------------- FOOTER --------------------
st.markdown("""
<div class="custom-footer">
© 2025 Yolanda Muvdi - Enfermera Epidemióloga | Proyecto Epidemiología 101  
Diseñado con ❤️ para profesionales de la salud  
</div>
""", unsafe_allow_html=True)
