import streamlit as st
import pandas as pd
import numpy as np
import math
import matplotlib.pyplot as plt
from scipy.stats import chi2_contingency, fisher_exact, norm
import os

# ------------- CONFIG -------------
st.set_page_config(
    page_title="🧠 Epidemiología 101 - Masterclass",
    page_icon="🧪",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ----------- ESTILOS -----------
st.markdown("""
<style>
    .stApp {
        background-color: #fefefe;
        color: #0d3b66;
        font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
    }
    h1, h2, h3 {
        color: #0d3b66;
        font-weight: 700;
    }
    .block-container {
        max-width: 900px;
        padding: 2rem 3rem;
        margin: 2rem auto 4rem auto;
        background: white;
        border-radius: 10px;
        box-shadow: 0 8px 20px rgba(13,59,102,0.1);
    }
    .stButton > button {
        background-color: #0d3b66;
        color: white;
        border-radius: 7px;
        padding: 10px 20px;
        font-weight: 700;
        font-size: 1.1rem;
        transition: background-color 0.3s ease;
    }
    .stButton > button:hover {
        background-color: #09466b;
        cursor: pointer;
    }
</style>
""", unsafe_allow_html=True)

# -------- FUNCIONES DE CÁLCULO EPIDEMIOLÓGICO -----------
def corregir_ceros(a,b,c,d):
    if 0 in [a,b,c,d]:
        return a+0.5, b+0.5, c+0.5, d+0.5, True
    return a,b,c,d, False

def ic_riesgo_relativo(a,b,c,d, alpha=0.05):
    risk1 = a/(a+b)
    risk2 = c/(c+d)
    rr = risk1/risk2
    se_log_rr = math.sqrt(1/a - 1/(a+b) + 1/c - 1/(c+d))
    z = norm.ppf(1 - alpha/2)
    lower = math.exp(math.log(rr) - z * se_log_rr)
    upper = math.exp(math.log(rr) + z * se_log_rr)
    return rr, lower, upper

def ic_odds_ratio(a,b,c,d, alpha=0.05):
    or_ = (a*d)/(b*c)
    se_log_or = math.sqrt(1/a + 1/b + 1/c + 1/d)
    z = norm.ppf(1 - alpha/2)
    lower = math.exp(math.log(or_) - z*se_log_or)
    upper = math.exp(math.log(or_) + z*se_log_or)
    return or_, lower, upper

def diferencia_riesgos(a,b,c,d, alpha=0.05):
    risk1 = a/(a+b)
    risk2 = c/(c+d)
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

# --------- FUNCIONES DE PLOTEO ----------
def plot_forest(rr, rr_l, rr_u, or_, or_l, or_u):
    fig, ax = plt.subplots(figsize=(6,3))
    ax.errorbar(x=[rr, or_], y=[2,1], 
                xerr=[[rr - rr_l, or_ - or_l], [rr_u - rr, or_u - or_]],
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

# ------------- GAMIFICACIÓN --------------
def gamificacion():
    st.header("🎮 Gamificación: ¿Qué nivel crees tener en Epidemiología?")

    niveles = {
        "Básico": [
            {"pregunta": "¿Qué es la epidemiología?", 
             "opciones": ["Estudio de bacterias", "Estudio de enfermedades en poblaciones", "Estudio de genética", "Estudio del clima"],
             "respuesta_correcta": "Estudio de enfermedades en poblaciones"},
            {"pregunta": "¿Qué mide la incidencia?", 
             "opciones": ["Nuevos casos en un tiempo", "Total de casos acumulados", "Número de muertes", "Población total"],
             "respuesta_correcta": "Nuevos casos en un tiempo"}
        ],
        "Intermedio": [
            {"pregunta": "¿Qué es un diseño de cohorte?", 
             "opciones": ["Estudio transversal", "Estudio prospectivo de seguimiento", "Ensayo clínico", "Revisión sistemática"],
             "respuesta_correcta": "Estudio prospectivo de seguimiento"},
            {"pregunta": "¿Qué es un Odds Ratio?", 
             "opciones": ["Medida de asociación en casos y controles", "Medida de incidencia", "Medida de prevalencia", "Medida de mortalidad"],
             "respuesta_correcta": "Medida de asociación en casos y controles"}
        ],
        "Avanzado": [
            {"pregunta": "¿Qué test estadístico se usa si las frecuencias esperadas son bajas?", 
             "opciones": ["Chi-cuadrado", "Fisher exacto", "t-test", "ANOVA"],
             "respuesta_correcta": "Fisher exacto"},
            {"pregunta": "¿Qué es un sesgo de selección?", 
             "opciones": ["Error aleatorio", "Error sistemático en selección de participantes", "Confusión", "Error de medición"],
             "respuesta_correcta": "Error sistemático en selección de participantes"}
        ],
        "Experto / Messi": [
            {"pregunta": "¿Cuál es el propósito principal del análisis multivariado?", 
             "opciones": ["Describir datos", "Controlar confusión y evaluar asociaciones ajustadas", "Calcular prevalencia", "Realizar muestreos"],
             "respuesta_correcta": "Controlar confusión y evaluar asociaciones ajustadas"},
            {"pregunta": "¿Qué es el error tipo I?", 
             "opciones": ["No rechazar H0 cuando es falsa", "Rechazar H0 cuando es verdadera", "Error en muestreo", "Error en medición"],
             "respuesta_correcta": "Rechazar H0 cuando es verdadera"}
        ]
    }

    # Estado para gamificación
    if "nivel" not in st.session_state:
        st.session_state.nivel = None
    if "indice_pregunta" not in st.session_state:
        st.session_state.indice_pregunta = 0
    if "puntaje" not in st.session_state:
        st.session_state.puntaje = 0
    if "mostrar_resultado" not in st.session_state:
        st.session_state.mostrar_resultado = False

    if st.session_state.nivel is None:
        nivel_elegido = st.selectbox("Selecciona tu nivel:", list(niveles.keys()))
        if st.button("Iniciar Quiz"):
            st.session_state.nivel = nivel_elegido
            st.session_state.indice_pregunta = 0
            st.session_state.puntaje = 0
            st.session_state.mostrar_resultado = False
    else:
        preguntas = niveles[st.session_state.nivel]
        if st.session_state.indice_pregunta < len(preguntas):
            pregunta_actual = preguntas[st.session_state.indice_pregunta]
            st.subheader(f"Pregunta {st.session_state.indice_pregunta+1} de {len(preguntas)}")
            respuesta = st.radio(pregunta_actual["pregunta"], pregunta_actual["opciones"], key="resp_gam")

            if st.button("Responder", key="boton_responder"):
                if respuesta == pregunta_actual["respuesta_correcta"]:
                    st.success("✅ Correcto!")
                    st.session_state.puntaje += 1
                else:
                    st.error(f"❌ Incorrecto. La respuesta correcta es: {pregunta_actual['respuesta_correcta']}")
                st.session_state.indice_pregunta += 1
        else:
            st.write(f"Tu puntaje final es: {st.session_state.puntaje} de {len(preguntas)}")
            if st.session_state.puntaje == len(preguntas):
                st.balloons()
                st.success("🎉 ¡Eres un experto en Epidemiología! ¡Felicitaciones!")
            elif st.session_state.puntaje >= len(preguntas)*0.7:
                st.success("👍 Muy bien, casi experto. Sigue practicando.")
            else:
                st.info("📚 Sigue estudiando, ¡vas por buen camino!")

            if st.button("Volver a elegir nivel"):
                st.session_state.nivel = None
                st.session_state.indice_pregunta = 0
                st.session_state.puntaje = 0
                st.session_state.mostrar_resultado = False

# --------- CHAT GEMINI ------------
try:
    import google.generativeai as genai
    GENAI_AVAILABLE = True
except ImportError:
    GENAI_AVAILABLE = False

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
        st.info("⚠️ No configurada GEMINI_API_KEY.")

def chat_gemini_interface():
    st.header("🤖 Chat Epidemiológico con Gemini")
    if "chat_historial" not in st.session_state:
        st.session_state.chat_historial = [{"role":"system","content":"Eres un asistente experto en epidemiología, amable y didáctico."}]
    user_input = st.text_area("Pregunta algo sobre epidemiología:", key="chat_input")
    if st.button("Enviar pregunta"):
        if user_input.strip():
            st.session_state.chat_historial.append({"role":"user","content":user_input})
            respuesta = "⚠ Gemini no está disponible."
            if GENAI_AVAILABLE and GEMINI_KEY:
                try:
                    model = genai.GenerativeModel("gemini-2.5-pro")
                    prompt = "\n\n".join([f"[{m['role'].upper()}]\n{m['content']}" for m in st.session_state.chat_historial]) + "\n\n[ASSISTANT]\n"
                    response = model.generate_content(prompt)
                    text = getattr(response, "text", None)
                    if not text and hasattr(response, "candidates") and response.candidates:
                        text = getattr(response.candidates[0], "content", None)
                    respuesta = text or "⚠ No se recibió respuesta válida."
                except Exception as e:
                    respuesta = f"⚠ Error en Gemini: {e}"
            st.session_state.chat_historial.append({"role":"assistant","content":respuesta})
    for mensaje in st.session_state.chat_historial:
        role = mensaje["role"]
        contenido = mensaje["content"]
        if role == "user":
            st.markdown(f"**Tú:** {contenido}")
        elif role == "assistant":
            st.markdown(f"**Asistente:** {contenido}")

# --------- SECCIONES CON CONTENIDO Y VIDEOS --------------
def seccion_contenido(titulo, texto, video_url=None):
    st.header(titulo)
    st.markdown(texto)
    if video_url:
        st.video(video_url)

def seccion_ejercicios():
    st.header("🧪 Ejercicios Prácticos de Epidemiología")

    preguntas = [
        {
            "pregunta": "¿Cuál es la medida de frecuencia que calcula nuevos casos en un período de tiempo?",
            "opciones": ["Prevalencia", "Incidencia", "Mortalidad", "Tasa bruta"],
            "correcta": "Incidencia"
        },
        {
            "pregunta": "¿Qué tipo de estudio es el que sigue a un grupo de individuos a lo largo del tiempo?",
            "opciones": ["Estudio transversal", "Cohorte", "Caso-control", "Ensayo clínico"],
            "correcta": "Cohorte"
        },
        {
            "pregunta": "¿Qué significa un Odds Ratio mayor a 1?",
            "opciones": ["No hay asociación", "Asociación positiva", "Asociación negativa", "Error en datos"],
            "correcta": "Asociación positiva"
        }
    ]

    if "ejercicio_indice" not in st.session_state:
        st.session_state.ejercicio_indice = 0
    if "ejercicio_puntaje" not in st.session_state:
        st.session_state.ejercicio_puntaje = 0

    if st.session_state.ejercicio_indice < len(preguntas):
        p = preguntas[st.session_state.ejercicio_indice]
        st.subheader(f"Pregunta {st.session_state.ejercicio_indice+1} de {len(preguntas)}")
        respuesta = st.radio(p["pregunta"], p["opciones"], key="respuesta_ejercicio")

        if st.button("Responder ejercicio"):
            if respuesta == p["correcta"]:
                st.success("¡Correcto!")
                st.session_state.ejercicio_puntaje += 1
            else:
                st.error(f"Incorrecto. La respuesta correcta es: {p['correcta']}")
            st.session_state.ejercicio_indice += 1
    else:
        st.write(f"Terminaste todos los ejercicios con {st.session_state.ejercicio_puntaje} respuestas correctas de {len(preguntas)}")
        if st.session_state.ejercicio_puntaje == len(preguntas):
            st.balloons()
            st.success("🎉 Excelente, dominaste estos ejercicios.")
        elif st.session_state.ejercicio_puntaje >= len(preguntas)*0.7:
            st.success("¡Muy bien! Sigue practicando para mejorar aún más.")
        else:
            st.info("Sigue estudiando, lo importante es avanzar.")
        if st.button("Reiniciar ejercicios"):
            st.session_state.ejercicio_indice = 0
            st.session_state.ejercicio_puntaje = 0

# --------- TABLAS 2x2 Y CÁLCULOS -------------
def seccion_tablas_calculos():
    st.header("📊 Tablas 2x2 y Cálculos Epidemiológicos")
    st.markdown("Introduce los valores de la tabla 2x2 para calcular medidas de asociación:")

    a = st.number_input("Casos expuestos (a)", min_value=0, value=10, step=1)
    b = st.number_input("No casos expuestos (b)", min_value=0, value=15, step=1)
    c = st.number_input("Casos no expuestos (c)", min_value=0, value=5, step=1)
    d = st.number_input("No casos no expuestos (d)", min_value=0, value=20, step=1)

    if st.button("Calcular medidas"):
        A, B, C, D, corregido = corregir_ceros(a, b, c, d)
        rr, rr_l, rr_u = ic_riesgo_relativo(A,B,C,D)
        or_, or_l, or_u = ic_odds_ratio(A,B,C,D)
        rd, rd_l, rd_u = diferencia_riesgos(A,B,C,D)
        p_val, test_name = calcular_p_valor(A,B,C,D)

        st.markdown(f"""
        **Resultados:**

        - Riesgo Relativo (RR): {rr:.3f} (IC95% {rr_l:.3f} - {rr_u:.3f})  
        - Odds Ratio (OR): {or_:.3f} (IC95% {or_l:.3f} - {or_u:.3f})  
        - Diferencia de Riesgos (RD): {rd:.3f} (IC95% {rd_l:.3f} - {rd_u:.3f})  
        - Valor p ({test_name}): {p_val:.4f}  
        """)

        if p_val < 0.05:
            st.success("🎯 Asociación estadísticamente significativa (p < 0.05).")
        else:
            st.warning("⚠ No hay asociación estadísticamente significativa (p ≥ 0.05).")

        plot_forest(rr, rr_l, rr_u, or_, or_l, or_u)
        plot_barras_expuestos(a,b,c,d)

# -------- MULTIMEDIA YOUTUBE --------
def seccion_multimedia():
    st.header("🎥 Videos para Aprender Epidemiología")
    videos = {
        "Introducción a Epidemiología": "https://www.youtube.com/watch?v=6T_oIjzx6e8",
        "Medidas de Asociación": "https://www.youtube.com/watch?v=3EL8Dwb-OIk",
        "Sesgos en Estudios Epidemiológicos": "https://www.youtube.com/watch?v=VJlx6z2u94w"
    }
    for titulo, url in videos.items():
        st.subheader(titulo)
        st.video(url)

# --------- MENÚ LATERAL Y NAVEGACIÓN ---------
def main():
    st.sidebar.title("🧪 Epidemiología 101")
    opciones = [
        "Inicio",
        "Conceptos Básicos",
        "Medidas Completas",
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

    # Dropdown que oculta título al seleccionar otro
    seleccion = st.sidebar.selectbox("Seleccione sección", opciones)

    # Título principal solo si está en inicio
    if seleccion == "Inicio":
        st.title("🧠 Bienvenida a Epidemiología 101")
        st.markdown("""
        Este espacio está creado para ti, para dominar la epidemiología con contenido claro, ejercicios prácticos, visualizaciones y hasta gamificación para hacerlo divertido.  
        Selecciona una sección en la barra lateral para comenzar.
        """)
        return

    # Secciones
    if seleccion == "Conceptos Básicos":
        texto = """
        ### ¿Qué es Epidemiología?

        La epidemiología es la ciencia que estudia la frecuencia, distribución y determinantes de enfermedades o eventos relacionados con la salud en poblaciones humanas.  
        Permite identificar factores de riesgo, evaluar intervenciones y guiar políticas de salud pública.

        ---
        ### Medidas de Frecuencia

        - Prevalencia: proporción de individuos con la enfermedad en un momento o periodo.  
        - Incidencia: número de casos nuevos en un tiempo definido.  
        - Mortalidad: número de muertes en una población.

        """
        seccion_contenido("Conceptos Básicos", texto)

    elif seleccion == "Medidas Completas":
        texto = """
        ### Medidas de Asociación

        - Riesgo Relativo (RR): razón de incidencia en expuestos vs no expuestos.  
        - Odds Ratio (OR): odds de exposición en casos vs controles.  
        - Diferencia de Riesgos (RD): diferencia absoluta en incidencia entre grupos.

        ---
        ### Intervalos de Confianza e Importancia Estadística

        El IC 95% indica el rango plausible para la medida en la población.  
        Un valor p < 0.05 indica asociación estadísticamente significativa.

        """
        seccion_contenido("Medidas Completas", texto)

    elif seleccion == "Diseños de Estudio":
        texto = """
        ### Principales Diseños de Estudio

        - Estudio Transversal: análisis puntual, prevalencia.  
        - Estudio de Cohorte: seguimiento a lo largo del tiempo, incidencia.  
        - Estudio Caso-Control: comparativo retrospectivo para factores de riesgo.  
        - Ensayo Clínico: intervención asignada aleatoriamente para evaluar tratamientos.

        """
        seccion_contenido("Diseños de Estudio", texto)

    elif seleccion == "Sesgos y Errores":
        texto = """
        ### Sesgos Comunes en Epidemiología

        - Sesgo de Selección: error en la inclusión de participantes.  
        - Sesgo de Información: errores en medición o clasificación.  
        - Confusión: factor externo asociado a exposición y resultado que distorsiona la asociación.

        ---
        ### Cómo Minimizar Sesgos

        Diseño riguroso, uso de controles, cegamiento y análisis estadísticos ajustados.

        """
        seccion_contenido("Sesgos y Errores", texto)

    elif seleccion == "Glosario Interactivo":
        st.header("📚 Glosario Interactivo")
        glosario = {
            "Epidemiología": "Ciencia que estudia la distribución y determinantes de eventos de salud.",
            "Incidencia": "Número de casos nuevos de enfermedad en un tiempo determinado.",
            "Prevalencia": "Proporción de individuos con la enfermedad en un momento específico.",
            "Riesgo Relativo": "Medida que compara la incidencia entre expuestos y no expuestos.",
            "Odds Ratio": "Medida de asociación usada en estudios caso-control.",
            "Sesgo": "Error sistemático que distorsiona resultados de estudios.",
        }
        termino = st.selectbox("Selecciona un término para ver su definición:", sorted(glosario.keys()))
        st.write(f"**{termino}:** {glosario[termino]}")

    elif seleccion == "Ejercicios Prácticos":
        seccion_ejercicios()

    elif seleccion == "Tablas 2x2 y Cálculos":
        seccion_tablas_calculos()

    elif seleccion == "Visualización de Datos":
        st.header("📉 Visualización de Datos")
        st.info("En construcción: próximamente podrás cargar tus datos y graficarlos interactivo.")

    elif seleccion == "Multimedia YouTube":
        seccion_multimedia()

    elif seleccion == "Gamificación":
        gamificacion()

    elif seleccion == "Chat Epidemiológico":
        chat_gemini_interface()

    else:
        st.warning("Sección no encontrada. Intenta otra opción.")

if __name__ == "__main__":
    main()


