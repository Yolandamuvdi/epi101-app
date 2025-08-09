import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import math
import os
from scipy.stats import chi2_contingency, fisher_exact, norm

# -------------------- CONFIGURACIÓN GENERAL --------------------
st.set_page_config(page_title="🧠 Epidemiología 101 - Masterclass", page_icon="🧪", layout="wide", initial_sidebar_state="expanded")

# Estilos CSS
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
menu_items = {
    "📌 Conceptos Básicos": "conceptos_basicos",
    "📈 Medidas de Asociación": "medidas_asociacion",
    "📊 Diseños de Estudio": "disenos_estudio",
    "⚠️ Sesgos y Errores": "sesgos_errores",
    "📚 Glosario Interactivo": "glosario_interactivo",
    "🧪 Ejercicios Prácticos": "ejercicios_practicos",
    "📊 Tablas 2x2 y Cálculos": "tablas_2x2",
    "📉 Visualización de Datos": "visualizacion_datos",
    "🎥 Multimedia YouTube": "multimedia_youtube",
    "🤖 Chat Epidemiológico": "chat_gemini",
    "🏆 Gamificación": "gamificacion"
}

st.sidebar.title("🧪 Epidemiología 101")
st.sidebar.markdown("""
👩‍⚕️ Creado por Yolanda Muvdi, Enfermera Epidemióloga  
📧 [ymuvdi@gmail.com](mailto:ymuvdi@gmail.com)  
🔗 [LinkedIn](https://www.linkedin.com/in/yolanda-paola-muvdi-muvdi-778b73152/)
""")

menu = st.sidebar.radio("Selecciona una sección:", list(menu_items.keys()))

# -------------------- FUNCIONES DE UTILIDAD --------------------
def cargar_md(ruta):
    try:
        with open(ruta, encoding="utf-8") as f:
            return f.read()
    except Exception:
        return None

def cargar_py_variable(ruta_py, var_name):
    ns = {}
    try:
        with open(ruta_py, encoding="utf-8") as f:
            exec(f.read(), ns)
        return ns.get(var_name)
    except Exception:
        return None

# Corrección 0.5 para celdas con cero
def corregir_ceros(a,b,c,d):
    if 0 in [a,b,c,d]:
        return a+0.5, b+0.5, c+0.5, d+0.5, True
    return a,b,c,d, False

# Cálculo de RR e IC95
def ic_riesgo_relativo(a,b,c,d, alpha=0.05):
    risk1 = a / (a + b)
    risk2 = c / (c + d)
    rr = risk1 / risk2
    se_log_rr = math.sqrt(1/a - 1/(a+b) + 1/c - 1/(c+d))
    z = norm.ppf(1 - alpha/2)
    lower = math.exp(math.log(rr) - z * se_log_rr)
    upper = math.exp(math.log(rr) + z * se_log_rr)
    return rr, lower, upper

# Cálculo de OR e IC95
def ic_odds_ratio(a,b,c,d, alpha=0.05):
    or_ = (a*d)/(b*c)
    se_log_or = math.sqrt(1/a + 1/b + 1/c + 1/d)
    z = norm.ppf(1 - alpha/2)
    lower = math.exp(math.log(or_) - z * se_log_or)
    upper = math.exp(math.log(or_) + z * se_log_or)
    return or_, lower, upper

# Diferencia de riesgos
def diferencia_riesgos(a,b,c,d, alpha=0.05):
    risk1 = a / (a + b)
    risk2 = c / (c + d)
    rd = risk1 - risk2
    se_rd = math.sqrt((risk1*(1-risk1))/(a+b) + (risk2*(1-risk2))/(c+d))
    z = norm.ppf(1 - alpha/2)
    lower = rd - z*se_rd
    upper = rd + z*se_rd
    return rd, lower, upper

# Cálculo de valor p
def calcular_p_valor(a,b,c,d):
    table = np.array([[a,b],[c,d]])
    chi2, p, dof, expected = chi2_contingency(table, correction=False)
    if (expected < 5).any():
        _, p = fisher_exact(table)
        test_used = "Fisher exact test"
    else:
        test_used = "Chi-cuadrado sin corrección"
    return p, test_used

# Interpretar resultados
def interpretar_resultados(rr, rr_l, rr_u, or_, or_l, or_u, rd, rd_l, rd_u, p_val, test_name):
    texto = f"""
**Resultados Epidemiológicos:**

•⁠  ⁠Riesgo Relativo (RR): {rr:.3f} (IC95% {rr_l:.3f} - {rr_u:.3f})  
•⁠  ⁠Odds Ratio (OR): {or_:.3f} (IC95% {or_l:.3f} - {or_u:.3f})  
•⁠  ⁠Diferencia de Riesgos (RD): {rd:.3f} (IC95% {rd_l:.3f} - {rd_u:.3f})  
•⁠  ⁠Valor p ({test_name}): {p_val:.4f}  

"""
    if p_val < 0.05:
        texto += "🎯 La asociación es estadísticamente significativa (p < 0.05)."
    else:
        texto += "⚠️ No se encontró asociación estadísticamente significativa (p ≥ 0.05)."
    return texto

# Gráficos para forest plot
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

# Barras exposición y casos
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
    mensajes = {
        "inicio": "🎓 Bienvenida a Epidemiología 101. ¡Comencemos la aventura científica! 🧬",
        "ejercicio_correcto": "🏅 ¡Correcto! Has ganado una insignia. Sigue así 🔥",
        "completo": "🌟 ¡Felicitaciones! Completaste todos los ejercicios y eres un PRO en Epidemiología. 📜",
        "nivel_basico": "📘 Nivel Básico alcanzado. ¡A seguir aprendiendo!",
        "nivel_intermedio": "📗 Nivel Intermedio alcanzado. Vas muy bien!",
        "nivel_avanzado": "📙 Nivel Avanzado alcanzado. ¡Impresionante!",
        "nivel_experto": "🏆 Nivel Experto/Messi alcanzado. ¡Eres un crack total! 🚀"
    }
    msg = mensajes.get(tipo, "🎉 ¡Bien hecho!")
    st.toast(msg, icon="🎉")

def gamificacion():
    st.header("🏆 Gamificación")

    niveles = ["Básico", "Intermedio", "Avanzado", "Experto/Messi"]
    nivel_usuario = st.selectbox("¿En qué nivel crees que estás?", niveles)

    preguntas_nivel = {
        "Básico": [
            {"pregunta": "¿Qué estudia la Epidemiología?", "opciones": ["Las enfermedades en poblaciones", "Las células individuales", "Los animales"], "respuesta_correcta": "Las enfermedades en poblaciones"},
            {"pregunta": "¿Qué es un factor de riesgo?", "opciones": ["Un evento que aumenta la probabilidad de enfermedad", "Una cura", "Un síntoma"], "respuesta_correcta": "Un evento que aumenta la probabilidad de enfermedad"}
        ],
        "Intermedio": [
            {"pregunta": "¿Qué es un estudio de cohorte?", "opciones": ["Estudio observacional prospectivo", "Estudio experimental", "Un tipo de vacuna"], "respuesta_correcta": "Estudio observacional prospectivo"},
            {"pregunta": "¿Qué mide el Riesgo Relativo?", "opciones": ["La asociación entre exposición y enfermedad", "La prevalencia", "La mortalidad"], "respuesta_correcta": "La asociación entre exposición y enfermedad"}
        ],
        "Avanzado": [
            {"pregunta": "¿Cuál es la fórmula del Odds Ratio?", "opciones": ["(a*d)/(b*c)", "(a+b)/(c+d)", "a/c"], "respuesta_correcta": "(a*d)/(b*c)"},
            {"pregunta": "¿Qué es el sesgo de selección?", "opciones": ["Error en la selección de participantes", "Error de medición", "Confusión"], "respuesta_correcta": "Error en la selección de participantes"}
        ],
        "Experto/Messi": [
            {"pregunta": "¿Qué prueba estadística se usa cuando hay celdas con valores menores a 5?", "opciones": ["Test de Fisher", "Chi-cuadrado", "t de Student"], "respuesta_correcta": "Test de Fisher"},
            {"pregunta": "¿Qué significa un valor p < 0.05?", "opciones": ["Asociación estadísticamente significativa", "No hay asociación", "Datos insuficientes"], "respuesta_correcta": "Asociación estadísticamente significativa"}
        ]
    }

    preguntas = preguntas_nivel.get(nivel_usuario, [])
    correctas = 0

    for i, q in enumerate(preguntas):
        respuesta = st.radio(q["pregunta"], q["opciones"], key=f"gamif_{nivel_usuario}_{i}")
        if st.button(f"Verificar respuesta {i+1}", key=f"btn_gamif_{nivel_usuario}_{i}"):
            if respuesta == q["respuesta_correcta"]:
                st.success("✅ Correcto")
                correctas += 1
                mostrar_insignia("ejercicio_correcto")
            else:
                st.error(f"❌ Incorrecto. La respuesta correcta es: {q['respuesta_correcta']}")

    if correctas == len(preguntas) and len(preguntas) > 0:
        mostrar_insignia(f"nivel_{nivel_usuario.lower()}")

# -------------------- CHAT GEMINI --------------------
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

# -------------------- CONTENIDO --------------------
def main():
    if menu == "📌 Conceptos Básicos":
        st.header("📌 Conceptos Básicos")
        contenido = cargar_md("contenido/conceptosbasicos.md")
        if contenido:
            st.markdown(contenido)
        else:
            st.info("Agrega el archivo 'contenido/conceptosbasicos.md' para mostrar el contenido.")

    elif menu == "📈 Medidas de Asociación":
        st.header("📈 Medidas de Asociación")
        contenido = cargar_md("contenido/medidas_completas.md")
        if contenido:
            st.markdown(contenido)
        else:
            st.info("Agrega el archivo 'contenido/medidas_completas.md' para mostrar el contenido.")

    elif menu == "📊 Diseños de Estudio":
        st.header("📊 Diseños de Estudio")
        contenido = cargar_md("contenido/disenos_completos.md")
        if contenido:
            st.markdown(contenido)
        else:
            st.info("Agrega el archivo 'contenido/disenos_completos.md' para mostrar el contenido.")

    elif menu == "⚠️ Sesgos y Errores":
        st.header("⚠️ Sesgos y Errores")
        contenido = cargar_md("contenido/sesgos_completos.md")
        if contenido:
            st.markdown(contenido)
        else:
            st.info("Agrega el archivo 'contenido/sesgos_completos.md' para mostrar el contenido.")

    elif menu == "📚 Glosario Interactivo":
        st.header("📚 Glosario Interactivo")
        glosario = cargar_py_variable("contenido/glosario_completo.py", "glosario")
        if glosario:
            for termino, definicion in glosario.items():
                with st.expander(termino):
                    st.write(definicion)
        else:
            st.info("Agrega 'contenido/glosario_completo.py' con variable ⁠ glosario ⁠.")

    elif menu == "🧪 Ejercicios Prácticos":
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
            st.info("Agrega 'contenido/ejercicios_completos.py' con variable ⁠ preguntas ⁠.")

    elif menu == "📊 Tablas 2x2 y Cálculos":
        st.header("📊 Tablas 2x2 y Cálculos Epidemiológicos")

        if "a" not in st.session_state:
            st.session_state.a = 10
        if "b" not in st.session_state:
            st.session_state.b = 20
        if "c" not in st.session_state:
            st.session_state.c = 15
        if "d" not in st.session_state:
            st.session_state.d = 30

        col1, col2 = st.columns(2)
        with col1:
            st.number_input("a (Casos expuestos)", min_value=0, value=st.session_state.a, key="a")
            st.number_input("b (No casos expuestos)", min_value=0, value=st.session_state.b, key="b")
        with col2:
            st.number_input("c (Casos no expuestos)", min_value=0, value=st.session_state.c, key="c")
            st.number_input("d (No casos no expuestos)", min_value=0, value=st.session_state.d, key="d")

        a = st.session_state.a
        b = st.session_state.b
        c = st.session_state.c
        d = st.session_state.d

        a_corr, b_corr, c_corr, d_corr, corrigio = corregir_ceros(a,b,c,d)

        if st.button("Calcular medidas"):
            if min(a,b,c,d) < 0:
                st.error("Los valores no pueden ser negativos.")
                return

            rr, rr_l, rr_u = ic_riesgo_relativo(a_corr, b_corr, c_corr, d_corr)
            or_, or_l, or_u = ic_odds_ratio(a_corr, b_corr, c_corr, d_corr)
            rd, rd_l, rd_u = diferencia_riesgos(a_corr, b_corr, c_corr, d_corr)
            p_val, test_name = calcular_p_valor(a,b,c,d)

            st.markdown(interpretar_resultados(rr, rr_l, rr_u, or_, or_l, or_u, rd, rd_l, rd_u, p_val, test_name))
            plot_forest(rr, rr_l, rr_u, or_, or_l, or_u)
            plot_barras_expuestos(a,b,c,d)
            if corrigio:
                st.info("Se corrigieron los ceros sumando 0.5 para evitar errores en cálculos estadísticos.")

    elif menu == "📉 Visualización de Datos":
        st.header("📉 Visualización de Datos")

        uploaded_file = st.file_uploader("Carga un archivo CSV para gráficos exploratorios", type=["csv"])

        if uploaded_file:
            try:
                df = pd.read_csv(uploaded_file)
                st.write("Vista previa de los datos cargados:")
                st.dataframe(df.head())

                columnas = df.columns.tolist()
                columna = st.selectbox("Selecciona la columna para graficar", columnas)

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
                st.error(f"Error al procesar el archivo: {e}")
        else:
            st.info("Carga un archivo CSV para comenzar a crear gráficos.")

    elif menu == "🎥 Multimedia YouTube":
        st.header("🎥 Videos Educativos")

        videos = {
            "Introducción a la Epidemiología": "https://www.youtube.com/embed/qVFP-IkyWgQ",
            "Medidas de Asociación": "https://www.youtube.com/embed/d61E24xvRfI",
            "Diseños de Estudio": "https://www.youtube.com/embed/y6odn6E8yRs",
            "Sesgos y Errores": "https://www.youtube.com/embed/1kyFIyG37qc"
        }

        for titulo, link in videos.items():
            st.subheader(titulo)
            st.markdown(f'<iframe width="560" height="315" src="{link}" frameborder="0" allowfullscreen></iframe>', unsafe_allow_html=True)

    elif menu == "🤖 Chat Epidemiológico":
        st.header("🤖 Chat con Gemini AI")

        if not (GENAI_AVAILABLE and GEMINI_KEY):
            st.warning("⚠️ Gemini no está configurado o la librería no está disponible.")
            return

        if "chat_historial" not in st.session_state:
            st.session_state.chat_historial = [{"role": "system", "content": "Eres un experto en epidemiología que responde de forma clara, amable y profesional."}]

        with st.form(key="form_gemini", clear_on_submit=True):
            entrada_usuario = st.text_input("Escribe tu pregunta de epidemiología:", "")
            submit = st.form_submit_button("Enviar")

        if submit and entrada_usuario.strip() != "":
            st.session_state.chat_historial.append({"role": "user", "content": entrada_usuario})
            respuesta = chat_with_gemini(st.session_state.chat_historial)
            st.session_state.chat_historial.append({"role": "assistant", "content": respuesta})

        for msg in st.session_state.chat_historial:
            if msg["role"] == "user":
                st.markdown(f"**Tú:** {msg['content']}")
            elif msg["role"] == "assistant":
                st.markdown(f"**Epidemiología 101:** {msg['content']}")

    elif menu == "🏆 Gamificación":
        gamificacion()

    else:
        st.info("Seleccione una sección desde el menú lateral para comenzar.")

# -------------------- EJECUCIÓN --------------------
if __name__ == "__main__":
    main()
