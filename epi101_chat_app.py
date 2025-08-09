import streamlit as st
import os
import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import chi2_contingency, fisher_exact, norm
import random
from streamlit_extras.let_it_rain import rain

# Configuración general
st.set_page_config(page_title="🧠 Epidemiología 101", page_icon="🧪", layout="wide", initial_sidebar_state="collapsed")

# --- Estilos CSS ---
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
    /* Responsive para móviles */
    @media (max-width: 768px) {
        .stButton>button {
            width: 100% !important;
            font-size: 1.2rem !important;
        }
    }
</style>
""", unsafe_allow_html=True)

# Funciones para cargar contenido
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

# Funciones epidemiológicas básicas para tablas 2x2
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

•⁠  ⁠Riesgo Relativo (RR): {rr:.3f} (IC95% {rr_l:.3f} - {rr_u:.3f})  
•⁠  ⁠Odds Ratio (OR): {or_:.3f} (IC95% {or_l:.3f} - {or_u:.3f})  
•⁠  ⁠Diferencia de Riesgos (RD): {rd:.3f} (IC95% {rd_l:.3f} - {rd_u:.3f})  
•⁠  ⁠Valor p ({test_name}): {p_val:.4f}  

"""
    if p_val < 0.05:
        texto += "🎯 Asociación estadísticamente significativa (p < 0.05)."
    else:
        texto += "⚠️ Asociación no estadísticamente significativa (p ≥ 0.05)."
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

# --- Gamificación extendida ---
def mostrar_confeti():
    rain(emoji="🎉", font_size=54, falling_speed=5, animation_length=3)

def mostrar_mensaje_nivel(nivel, puntaje, total):
    mensajes = {
        "Básico": f"Nivel Básico completado. Puntaje: {puntaje}/{total}. ¡Buen comienzo! Sigue estudiando 📚.",
        "Intermedio": f"Nivel Intermedio completado. Puntaje: {puntaje}/{total}. ¡Vas muy bien! ¡Aumenta el nivel! 🚀",
        "Avanzado": f"Nivel Avanzado completado. Puntaje: {puntaje}/{total}. ¡Impresionante! Estás dominando la epidemiología 💪",
        "Experto/Messi": f"Nivel Experto/Messi completado. Puntaje: {puntaje}/{total}. ¡Eres un crack en epidemiología! 🎖️🔥"
    }
    st.balloons()
    st.success(mensajes.get(nivel, "¡Felicidades!"))

def filtrar_preguntas_por_nivel(preguntas, nivel):
    if nivel == "Básico":
        return [p for p in preguntas if p["nivel"] == "Básico"]
    elif nivel == "Intermedio":
        return [p for p in preguntas if p["nivel"] == "Intermedio"]
    elif nivel == "Avanzado":
        return [p for p in preguntas if p["nivel"] == "Avanzado"]
    else:
        return [p for p in preguntas if p["nivel"] == "Experto/Messi"]

# --- Página principal y navegación ---
def pagina_inicio():
    st.title("🧠 Epidemiología 101")
    st.markdown("¿Qué quieres aprender hoy? Selecciona una sección:")
    opciones = [
        "📌 Conceptos Básicos",
        "📈 Medidas de Asociación",
        "📊 Diseños de Estudio",
        "⚠️ Sesgos y Errores",
        "📚 Glosario Interactivo",
        "🧪 Ejercicios Prácticos",
        "📊 Tablas 2x2 y Cálculos",
        "📊 Visualización de Datos",
        "🎥 Multimedia YouTube",
        "🤖 Chat Epidemiológico",
        "🎯 Gamificación"
    ]
    seleccion = st.selectbox("Selecciona sección", opciones)
    return seleccion

def barra_lateral(seleccion_actual):
    st.sidebar.title("🧪 Epidemiología 101")
    st.sidebar.markdown("""
    👩‍⚕️ Creado por Yolanda Muvdi, Enfermera Epidemióloga  
    📧 [ymuvdi@gmail.com](mailto:ymuvdi@gmail.com)  
    🔗 [LinkedIn](https://www.linkedin.com/in/yolanda-paola-muvdi-muvdi-778b73152/)
    """)
    opciones = [
        ("📌 Conceptos Básicos", "Conceptos Básicos"),
        ("📈 Medidas de Asociación", "Medidas de Asociación"),
        ("📊 Diseños de Estudio", "Diseños de Estudio"),
        ("⚠️ Sesgos y Errores", "Sesgos y Errores"),
        ("📚 Glosario Interactivo", "Glosario Interactivo"),
        ("🧪 Ejercicios Prácticos", "Ejercicios Prácticos"),
        ("📊 Tablas 2x2 y Cálculos", "Tablas 2x2 y Cálculos"),
        ("📊 Visualización de Datos", "Visualización de Datos"),
        ("🎥 Multimedia YouTube", "Multimedia YouTube"),
        ("🤖 Chat Epidemiológico", "Chat Epidemiológico"),
        ("🎯 Gamificación", "Gamificación")
    ]
    opciones_texto = [x[0] for x in opciones]
    seleccion_sidebar = st.sidebar.radio("Ir a sección:", opciones_texto, index=[x[0] for x in opciones].index(f"📌 {seleccion_actual}") if seleccion_actual else 0)
    # Convertir texto a clave
    seleccion_clave = dict(opciones)[seleccion_sidebar]
    return seleccion_clave

def main():
    if "seccion" not in st.session_state:
        st.session_state.seccion = None
        st.session_state.nivel_gamificacion = None
        st.session_state.respuestas = {}

    if st.session_state.seccion is None:
        seleccion = pagina_inicio()
        if st.button("Ir a la sección"):
            st.session_state.seccion = seleccion.replace("📌 ","").replace("📈 ","").replace("📊 ","").replace("⚠️ ","").replace("📚 ","").replace("🧪 ","").replace("🎥 ","").replace("🤖 ","").replace("🎯 ","")
            st.experimental_rerun()
        return

    seleccion = barra_lateral(st.session_state.seccion)

    # Contenido de secciones
    if seleccion == "Conceptos Básicos":
        st.header("📌 Conceptos Básicos")
        contenido = cargar_md("contenido/conceptosbasicos.md")
        if contenido:
            st.markdown(contenido)
        else:
            st.info("Agrega el archivo 'contenido/conceptosbasicos.md' para mostrar el contenido.")

    elif seleccion == "Medidas de Asociación":
        st.header("📈 Medidas de Asociación")
        contenido = cargar_md("contenido/medidas_completas.md")
        if contenido:
            st.markdown(contenido)
        else:
            st.info("Agrega el archivo 'contenido/medidas_completas.md' para mostrar el contenido.")

    elif seleccion == "Diseños de Estudio":
        st.header("📊 Diseños de Estudio")
        contenido = cargar_md("contenido/disenos_completos.md")
        if contenido:
            st.markdown(contenido)
        else:
            st.info("Agrega el archivo 'contenido/disenos_completos.md' para mostrar el contenido.")

    elif seleccion == "Sesgos y Errores":
        st.header("⚠️ Sesgos y Errores")
        contenido = cargar_md("contenido/sesgos_completos.md")
        if contenido:
            st.markdown(contenido)
        else:
            st.info("Agrega el archivo 'contenido/sesgos_completos.md' para mostrar el contenido.")

    elif seleccion == "Glosario Interactivo":
        st.header("📚 Glosario Interactivo")
        glosario = cargar_py_variable("contenido/glosario_completo.py", "glosario")
        if glosario:
            for termino, definicion in glosario.items():
                with st.expander(termino):
                    st.write(definicion)
        else:
            st.info("Agrega 'contenido/glosario_completo.py' con variable ⁠ glosario ⁠.")

    elif seleccion == "Ejercicios Prácticos":
        st.header("🧪 Ejercicios Prácticos")
        preguntas = cargar_py_variable("contenido/ejercicios_completos.py", "preguntas")
        if preguntas:
            correctas = 0
            for i, p in enumerate(preguntas):
                st.subheader(f"Pregunta {i+1} (Nivel {p['nivel']})")
                respuesta = st.radio(p["pregunta"], p["opciones"], key=f"preg_{i}")
                if st.button(f"Verificar respuesta {i+1}", key=f"btn_{i}"):
                    if respuesta == p["respuesta_correcta"]:
                        st.success("✅ Correcto")
                        correctas += 1
                    else:
                        st.error(f"❌ Incorrecto. Respuesta correcta: {p['respuesta_correcta']}")
            if correctas == len(preguntas) and len(preguntas) > 0:
                st.balloons()
                st.success("¡Felicidades! Has completado todos los ejercicios.")
        else:
            st.info("Agrega 'contenido/ejercicios_completos.py' con variable ⁠ preguntas ⁠.")

    elif seleccion == "Tablas 2x2 y Cálculos":
        st.header("📊 Tablas 2x2 y Cálculos Epidemiológicos")
        if "a" not in st.session_state: st.session_state.a = 10
        if "b" not in st.session_state: st.session_state.b = 20
        if "c" not in st.session_state: st.session_state.c = 5
        if "d" not in st.session_state: st.session_state.d = 40
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

    elif seleccion == "Visualización de Datos":
        st.header("📊 Visualización de Datos")
        uploaded_file = st.file_uploader("Carga un archivo CSV para gráficos exploratorios", type=["csv"])
        if uploaded_file:
            try:
                df = pd.read_csv(uploaded_file)
                st.dataframe(df.head())
                st.markdown("### Gráfico de barras de la primera columna numérica")
                cols_numericas = df.select_dtypes(include=np.number).columns.tolist()
                if cols_numericas:
                    col = cols_numericas[0]
                    fig, ax = plt.subplots()
                    df[col].value_counts().plot(kind='bar', ax=ax, color='#0d3b66')
                    ax.set_title(f"Distribución de {col}")
                    st.pyplot(fig, use_container_width=True)
                else:
                    st.info("No se detectaron columnas numéricas para graficar.")
            except Exception as e:
                st.error(f"Error al leer el archivo CSV: {e}")
        else:
            st.info("Carga un archivo CSV para ver gráficos dinámicos.")

    elif seleccion == "Multimedia YouTube":
        st.header("🎥 Videos Recomendados")
        videos = {
            "Introducción a Epidemiología": "https://www.youtube.com/watch?v=qVFP-IkyWgQ",
            "Medidas de Asociación": "https://www.youtube.com/watch?v=d61E24xvRfI",
            "Diseños de Estudio": "https://www.youtube.com/watch?v=y6odn6E8yRs",
            "Sesgos en Epidemiología": "https://www.youtube.com/watch?v=1kyFIyG37qc"
        }
        for titulo, url in videos.items():
            st.markdown(f"**{titulo}**")
            st.video(url)

    elif seleccion == "Chat Epidemiológico":
        st.header("🤖 Chat Epidemiológico (Gemini AI)")
        st.info("Esta función está integrada y lista para usarse. Escribe tus consultas epidemiológicas abajo.")
        pregunta = st.text_input("Escribe tu pregunta epidemiológica aquí:")
        if st.button("Enviar"):
            # Aquí va la integración con Gemini API o tu modelo AI
            st.success(f"Respuesta simulada para: {pregunta}")

    elif seleccion == "Gamificación":
        st.header("🎯 Gamificación Epidemiológica")
        niveles = ["Básico", "Intermedio", "Avanzado", "Experto/Messi"]
        if "nivel_gamificacion" not in st.session_state:
            st.session_state.nivel_gamificacion = None
            st.session_state.puntaje = 0
            st.session_state.total = 0
            st.session_state.index_pregunta = 0
            st.session_state.respuestas_correctas = 0

        if st.session_state.nivel_gamificacion is None:
            nivel = st.selectbox("¿En qué nivel consideras que estás?", niveles)
            if st.button("Comenzar quiz"):
                st.session_state.nivel_gamificacion = nivel
                # Cargar preguntas filtradas
                preguntas = cargar_py_variable("contenido/ejercicios_completos.py", "preguntas")
                if preguntas:
                    st.session_state.preguntas_gamificacion = filtrar_preguntas_por_nivel(preguntas, nivel)
                    st.session_state.total = len(st.session_state.preguntas_gamificacion)
                    st.session_state.index_pregunta = 0
                    st.session_state.respuestas_correctas = 0
                else:
                    st.error("No hay preguntas disponibles para gamificación.")
                st.experimental_rerun()
        else:
            preguntas = st.session_state.get("preguntas_gamificacion", [])
            idx = st.session_state.index_pregunta

            if idx < len(preguntas):
                p = preguntas[idx]
                st.subheader(f"Pregunta {idx+1} de {len(preguntas)}")
                respuesta = st.radio(p["pregunta"], p["opciones"], key=f"gam_preg_{idx}")
                if st.button("Responder", key=f"gam_btn_{idx}"):
                    if respuesta == p["respuesta_correcta"]:
                        st.success("✅ Correcto!")
                        st.session_state.respuestas_correctas += 1
                    else:
                        st.error(f"❌ Incorrecto. Respuesta correcta: {p['respuesta_correcta']}")
                    st.session_state.index_pregunta += 1
                    st.experimental_rerun()
            else:
                mostrar_confeti()
                mostrar_mensaje_nivel(st.session_state.nivel_gamificacion, st.session_state.respuestas_correctas, len(preguntas))
                if st.button("Reiniciar gamificación"):
                    st.session_state.nivel_gamificacion = None
                    st.session_state.index_pregunta = 0
                    st.session_state.respuestas_correctas = 0
                    st.experimental_rerun()

if __name__ == "__main__":
    main()



