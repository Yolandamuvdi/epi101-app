import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import math
import os

# Intento importar Google Gemini AI client
try:
    import google.generativeai as genai
    GENAI_AVAILABLE = True
except ImportError:
    GENAI_AVAILABLE = False

# Intento importar scipy para pruebas estadísticas
try:
    from scipy.stats import chi2_contingency, fisher_exact, norm
    SCIPY_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False

# Configuración página
st.set_page_config(page_title="Epidemiología 101", layout="wide", initial_sidebar_state="collapsed")

# CSS para splash, footer y UI
st.markdown("""
<style>
/* Splash */
.splash-container {
    background: linear-gradient(135deg, #0d3b66, #144d80);
    color: white;
    height: 90vh;
    display: flex;
    flex-direction: column;
    justify-content: center;
    align-items: center;
    font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
    text-align: center;
    padding: 20px;
    border-radius: 12px;
    box-shadow: 0 12px 30px rgba(0,0,0,0.3);
    margin-bottom: 1rem;
}
.splash-container h1 {
    font-size: 3.5rem;
    margin-bottom: 0.3rem;
    font-weight: 900;
}
.splash-container p {
    font-size: 1.3rem;
    max-width: 600px;
    margin: 0.3rem auto 1rem auto;
    font-weight: 600;
}
.splash-footer {
    margin-top: auto;
    font-size: 0.9rem;
    color: #a8c0ffcc;
    width: 100%;
    text-align: right;
    font-weight: 600;
    border-top: 1px solid #0b2e55;
    padding-top: 12px;
}

/* Sidebar */
[data-testid="stSidebar"] {
    background: #f0f4f8;
    padding-top: 1rem;
    min-width: 230px;
}
.stSidebar .css-1d391kg {
    padding-top: 1rem;
}

/* Main container */
.main-content {
    padding: 2rem 3rem;
    max-width: 1100px;
    margin: auto;
    font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
}

/* Footer general */
.footer {
    font-size: 0.85rem;
    color: #0d3b66;
    text-align: right;
    margin-top: 3rem;
    font-weight: 600;
}

/* Links */
a {
    color: #0d3b66;
    font-weight: 700;
    text-decoration: none;
}
a:hover {
    text-decoration: underline;
}

/* YouTube iframe responsive */
.youtube-container {
    position: relative;
    padding-bottom: 56.25%;
    height: 0;
    overflow: hidden;
    max-width: 100%;
}
.youtube-container iframe {
    position: absolute;
    top: 0;
    left: 0;
    width: 100%;
    height: 100%;
    border: 0;
}
</style>
""", unsafe_allow_html=True)

# Variables de estado
if "seccion" not in st.session_state:
    st.session_state.seccion = None

if "puntaje_gamificacion" not in st.session_state:
    st.session_state.puntaje_gamificacion = 0

# Funciones epidemiológicas
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

# Función para chat Gemini
GEMINI_KEY = st.secrets.get("GEMINI_API_KEY") or os.environ.get("GEMINI_API_KEY")
if GENAI_AVAILABLE and GEMINI_KEY:
    try:
        genai.configure(api_key=GEMINI_KEY)
    except Exception as e:
        st.warning(f"Error configurando Gemini: {e}")

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

# ------------------ SECCIONES ------------------

def seccion_inicio():
    st.markdown("""
    <div style="text-align:center; padding: 3rem 1rem;">
        <h1 style="color:#0d3b66; font-weight: 900;">🧪 Bienvenido/a a Epidemiología 101</h1>
        <p style="font-size:1.3rem; font-weight:600; max-width: 650px; margin:auto;">
        ¿Qué sección quieres usar hoy? Explora conceptos, ejercicios, tablas 2x2, visualización y mucho más.
        </p>
    </div>
    """, unsafe_allow_html=True)

def seccion_conceptos():
    st.header("📌 Conceptos Básicos")
    contenido = cargar_md("contenido/conceptosbasicos.md")
    if contenido:
        st.markdown(contenido)
    else:
        st.info("Agrega el archivo 'contenido/conceptosbasicos.md' para mostrar el contenido.")

def seccion_medidas():
    st.header("📈 Medidas de Asociación")
    contenido = cargar_md("contenido/medidas_completas.md")
    if contenido:
        st.markdown(contenido)
    else:
        st.info("Agrega el archivo 'contenido/medidas_completas.md' para mostrar el contenido.")

def seccion_disenos():
    st.header("📊 Diseños de Estudio")
    contenido = cargar_md("contenido/disenos_completos.md")
    if contenido:
        st.markdown(contenido)
    else:
        st.info("Agrega el archivo 'contenido/disenos_completos.md' para mostrar el contenido.")

def seccion_sesgos():
    st.header("⚠️ Sesgos y Errores")
    contenido = cargar_md("contenido/sesgos_completos.md")
    if contenido:
        st.markdown(contenido)
    else:
        st.info("Agrega el archivo 'contenido/sesgos_completos.md' para mostrar el contenido.")

def seccion_glosario():
    st.header("📚 Glosario Interactivo")
    glosario = cargar_py_variable("contenido/glosario_completo.py", "glosario")
    if glosario:
        for termino, definicion in glosario.items():
            with st.expander(termino):
                st.write(definicion)
    else:
        st.info("Agrega 'contenido/glosario_completo.py' con variable glosario.")

def seccion_ejercicios():
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
            st.balloons()
            st.success("🎉 ¡Has completado todos los ejercicios!")
    else:
        st.info("Agrega 'contenido/ejercicios_completos.py' con variable preguntas.")

def seccion_tablas2x2():
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
            # corregir ceros
            a_, b_, c_, d_, corr = corregir_ceros(a,b,c,d)

            rr, rr_l, rr_u = ic_riesgo_relativo(a_, b_, c_, d_)
            or_, or_l, or_u = ic_odds_ratio(a_, b_, c_, d_)
            rd, rd_l, rd_u = diferencia_riesgos(a_, b_, c_, d_)
            p_val, test_name = calcular_p_valor(a,b,c,d)

            st.markdown(interpretar_resultados(rr, rr_l, rr_u, or_, or_l, or_u, rd, rd_l, rd_u, p_val, test_name))
            plot_forest(rr, rr_l, rr_u, or_, or_l, or_u)
            plot_barras_expuestos(a,b,c,d)

def seccion_visualizacion():
    st.header("📉 Visualización de Datos Epidemiológicos")
    # Aquí puedes agregar más visualizaciones
    df = pd.DataFrame({
        "Categoría": ["Expuestos con caso", "Expuestos sin caso", "No expuestos con caso", "No expuestos sin caso"],
        "Conteo": [30, 70, 15, 85]
    })
    st.write("Ejemplo de gráfico de barras con datos simulados:")
    fig, ax = plt.subplots()
    ax.bar(df["Categoría"], df["Conteo"], color=["#0d3b66", "#3e5c76", "#82a0bc", "#b0c4de"])
    plt.xticks(rotation=20)
    st.pyplot(fig, use_container_width=True)

def seccion_videos():
    st.header("🎥 Videos de Epidemiología")

    videos = [
        {"title": "Introducción a Epidemiología", "youtube_id": "QmMT9PS7ghk"},
        {"title": "Medidas de Asociación Epidemiológica", "youtube_id": "p8f3P0lg4mA"},
        {"title": "Sesgos y Confusión en Estudios", "youtube_id": "2l1wBhEwQMc"},
    ]
    for v in videos:
        st.subheader(v["title"])
        st.markdown(f"""
        <div class="youtube-container">
        <iframe src="https://www.youtube.com/embed/{v['youtube_id']}" frameborder="0" allowfullscreen allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture"></iframe>
        </div>
        """, unsafe_allow_html=True)

def seccion_gamificacion():
    st.header("🎮 Gamificación")

    # Ejemplo básico de niveles y preguntas
    niveles = {
        1: {
            "pregunta": "¿Qué medida indica el riesgo relativo entre expuestos y no expuestos?",
            "opciones": ["Odds Ratio", "Riesgo Relativo", "Diferencia de Riesgos", "Prevalencia"],
            "respuesta_correcta": "Riesgo Relativo"
        },
        2: {
            "pregunta": "¿Cuál es el sesgo que ocurre cuando los participantes no son representativos?",
            "opciones": ["Sesgo de Selección", "Sesgo de Información", "Confusión", "Sesgo de Publicación"],
            "respuesta_correcta": "Sesgo de Selección"
        }
    }
    puntaje = st.session_state.puntaje_gamificacion
    nivel_actual = puntaje + 1 if puntaje < len(niveles) else len(niveles)
    st.subheader(f"Nivel {nivel_actual}")
    pregunta = niveles[nivel_actual]
    respuesta = st.radio(pregunta["pregunta"], pregunta["opciones"], key="gamif_radio")

    if st.button("Responder pregunta"):
        if respuesta == pregunta["respuesta_correcta"]:
            st.success("¡Respuesta correcta! 🎉")
            if puntaje < len(niveles):
                st.session_state.puntaje_gamificacion += 1
            if st.session_state.puntaje_gamificacion == len(niveles):
                st.balloons()
                st.success("🎊 ¡Has completado todos los niveles de gamificación! 🎊")
        else:
            st.error(f"Respuesta incorrecta. La respuesta correcta es: {pregunta['respuesta_correcta']}")

def seccion_chat():
    st.header("💬 Chat Epidemiológico")

    if not GENAI_AVAILABLE or not GEMINI_KEY:
        st.warning("Para usar este chat, necesitas configurar la API KEY de Gemini y tener instalada la librería google-generativeai.")
        return

    if "chat_hist" not in st.session_state:
        st.session_state.chat_hist = [{"role":"system","content":"Eres un asistente experto en epidemiología. Responde de forma clara y precisa."}]

    with st.form("chat_form", clear_on_submit=True):
        user_msg = st.text_area("Escribe tu pregunta epidemiológica aquí:", height=100)
        enviar = st.form_submit_button("Enviar")

    if enviar and user_msg.strip() != "":
        st.session_state.chat_hist.append({"role":"user", "content":user_msg})
        with st.spinner("Procesando respuesta..."):
            respuesta = chat_with_gemini(st.session_state.chat_hist)
        st.session_state.chat_hist.append({"role":"assistant", "content":respuesta})

    # Mostrar historial chat
    for m in st.session_state.chat_hist[1:]:
        if m["role"]=="user":
            st.markdown(f"**Tú:** {m['content']}")
        else:
            st.markdown(f"**Asistente:** {m['content']}")

# Función para cargar archivos markdown (puedes modificar para usar tus archivos)
def cargar_md(ruta):
    try:
        with open(ruta, "r", encoding="utf-8") as f:
            return f.read()
    except:
        return None

# Función para cargar variables desde scripts py (glosario, ejercicios, etc)
def cargar_py_variable(ruta, variable):
    try:
        import importlib.util
        spec = importlib.util.spec_from_file_location("modulo_temp", ruta)
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        return getattr(mod, variable, None)
    except Exception as e:
        return None

# --------- MAIN ---------
def main():
    # Splash inicial si no hay sección seleccionada
    if st.session_state.seccion is None:
        mostrar_splash()
        opciones = [
            "Inicio",
            "Conceptos Básicos",
            "Medidas de Asociación",
            "Diseños de Estudio",
            "Sesgos y Errores",
            "Glosario Interactivo",
            "Ejercicios Prácticos",
            "Tablas 2x2",
            "Visualización",
            "Videos",
            "Gamificación",
            "Chat Epidemiológico"
        ]
        seleccion = st.selectbox("Selecciona la sección:", opciones)
        if seleccion != "Inicio":
            st.session_state.seccion = seleccion
            st.experimental_rerun()
    else:
        # Barra lateral para navegar entre secciones
        with st.sidebar:
            st.title("Navegación")
            opciones_sidebar = [
                "Inicio",
                "Conceptos Básicos",
                "Medidas de Asociación",
                "Diseños de Estudio",
                "Sesgos y Errores",
                "Glosario Interactivo",
                "Ejercicios Prácticos",
                "Tablas 2x2",
                "Visualización",
                "Videos",
                "Gamificación",
                "Chat Epidemiológico"
            ]
            opcion = st.radio("Secciones", opciones_sidebar, index=opciones_sidebar.index(st.session_state.seccion))
            if opcion != st.session_state.seccion:
                st.session_state.seccion = opcion
                st.experimental_rerun()
            st.markdown("---")
            if st.button("Volver al Inicio"):
                st.session_state.seccion = None
                st.experimental_rerun()

        # Contenedor principal de contenido
        st.markdown('<div class="main-content">', unsafe_allow_html=True)

        if st.session_state.seccion == "Inicio":
            seccion_inicio()
        elif st.session_state.seccion == "Conceptos Básicos":
            seccion_conceptos()
        elif st.session_state.seccion == "Medidas de Asociación":
            seccion_medidas()
        elif st.session_state.seccion == "Diseños de Estudio":
            seccion_disenos()
        elif st.session_state.seccion == "Sesgos y Errores":
            seccion_sesgos()
        elif st.session_state.seccion == "Glosario Interactivo":
            seccion_glosario()
        elif st.session_state.seccion == "Ejercicios Prácticos":
            seccion_ejercicios()
        elif st.session_state.seccion == "Tablas 2x2":
            seccion_tablas2x2()
        elif st.session_state.seccion == "Visualización":
            seccion_visualizacion()
        elif st.session_state.seccion == "Videos":
            seccion_videos()
        elif st.session_state.seccion == "Gamificación":
            seccion_gamificacion()
        elif st.session_state.seccion == "Chat Epidemiológico":
            seccion_chat()

        st.markdown('</div>', unsafe_allow_html=True)

        # Footer fijo abajo derecha
        st.markdown("""
        <div class="footer">
            Creado por <strong>Yolanda Muvdi</strong> | 📧 ymuvdi@gmail.com | 🔗 <a href="https://www.linkedin.com/in/yolanda-paola-muvdi-muvdi-778b73152/" target="_blank">LinkedIn</a>
        </div>
        """, unsafe_allow_html=True)

# Función para mostrar splash (separada para main)
def mostrar_splash():
    st.markdown("""
    <div class="splash-container">
        <h1>🧪 Bienvenido/a a Epidemiología 101</h1>
        <p>Selecciona la sección que deseas explorar para aprender epidemiología de forma práctica, clara y divertida.</p>
        <p>Usa el menú que aparecerá al seleccionar la sección.</p>
        <div class="splash-footer">
            Creado por <strong>Yolanda Muvdi</strong> | 📧 ymuvdi@gmail.com | 🔗 <a href="https://www.linkedin.com/in/yolanda-paola-muvdi-muvdi-778b73152/" target="_blank" style="color:#a8c0ff;">LinkedIn</a>
        </div>
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()





