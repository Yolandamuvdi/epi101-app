import streamlit as st
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import math
import os

# Import Gemini client (google generative ai)
try:
    import google.generativeai as genai
    GENAI_AVAILABLE = True
except Exception:
    GENAI_AVAILABLE = False

# SciPy para Chi2 / Fisher
try:
    from scipy.stats import chi2_contingency, fisher_exact, norm
    SCIPY_AVAILABLE = True
except Exception:
    SCIPY_AVAILABLE = False

# 1. Configuración general Streamlit
st.set_page_config(page_title="Epidemiología 101", page_icon="🧪", layout="wide")

# Estilos limpios, sin modo oscuro
st.markdown("""
<style>
    body, .block-container {
        background: #fefefe;
        color: #0d3b66;
        font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
    }
    .block-container {
        max-width: 1100px;
        margin: 2rem auto;
        padding: 2rem 3rem;
        background: white;
        border-radius: 10px;
        box-shadow: 0 6px 18px rgb(0 0 0 / 0.1);
    }
    h1, h2, h3 {
        color: #0d3b66;
    }
    .stButton>button {
        background-color: #0d3b66;
        color: white;
        border-radius: 5px;
        padding: 8px 15px;
        font-weight: 600;
    }
    a {
        color: #0d3b66;
        text-decoration: none;
    }
    a:hover {
        text-decoration: underline;
    }
</style>
""", unsafe_allow_html=True)

st.title("🧠 Epidemiología 101 - Asistente educativo")
st.caption("Creado por Yolanda Muvdi")

# 2. Configurar Gemini (Google Generative AI)
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

# 3. Funciones auxiliares

def cargar_md(ruta):
    try:
        with open(ruta, encoding="utf-8") as f:
            return f.read()
    except:
        return None

def cargar_py_variable(path_py, var_name):
    ns = {}
    try:
        with open(path_py, encoding="utf-8") as f:
            exec(f.read(), ns)
        return ns.get(var_name)
    except:
        return None

# 4. Cálculos epidemiológicos: RR, OR, RD, IC 95% y p-valor

def corregir_ceros(a,b,c,d):
    # Si hay ceros, aplica corrección 0.5 para celdas
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

# 5. Interpretación automática

def interpretar_resultados(rr, rr_l, rr_u, or_, or_l, or_u, rd, rd_l, rd_u, p_val, test_name):
    texto = f"""
**Resultados:**

- Riesgo Relativo (RR): {rr:.3f} (IC95% {rr_l:.3f} - {rr_u:.3f})  
- Odds Ratio (OR): {or_:.3f} (IC95% {or_l:.3f} - {or_u:.3f})  
- Diferencia de Riesgos (RD): {rd:.3f} (IC95% {rd_l:.3f} - {rd_u:.3f})  
- Valor p ({test_name}): {p_val:.4f}  

"""
    if p_val < 0.05:
        texto += "La asociación es estadísticamente significativa (p < 0.05)."
    else:
        texto += "No se encontró asociación estadísticamente significativa (p ≥ 0.05)."
    return texto

# 6. Gráficos

def plot_forest(rr, rr_l, rr_u, or_, or_l, or_u):
    fig, ax = plt.subplots(figsize=(6,3))
    ax.errorbar(x=[rr, or_], y=[2,1], xerr=[ [rr-rr_l, or_-or_l], [rr_u-rr, or_u-or_] ], fmt='o', color='#0d3b66', capsize=5)
    ax.set_yticks([1,2])
    ax.set_yticklabels(["Odds Ratio (OR)", "Riesgo Relativo (RR)"])
    ax.axvline(1, color='gray', linestyle='--')
    ax.set_xlabel("Medidas de Asociación")
    ax.set_title("Intervalos de Confianza 95%")
    st.pyplot(fig)

def plot_barras_expuestos(a,b,c,d):
    labels = ["Casos expuestos", "No casos expuestos", "Casos no expuestos", "No casos no expuestos"]
    valores = [a,b,c,d]
    colores = ['#0d3b66', '#3e5c76', '#82a0bc', '#b0c4de']
    fig, ax = plt.subplots()
    ax.bar(labels, valores, color=colores)
    ax.set_ylabel("Conteo")
    ax.set_title("Distribución de exposición y casos")
    plt.xticks(rotation=15)
    st.pyplot(fig)

# 7. Contenido y pestañas

tabs = st.tabs([
    "Conceptos Básicos",
    "Medidas de Asociación",
    "Diseños de Estudio",
    "Sesgos y Errores",
    "Glosario Interactivo",
    "Ejercicios Prácticos",
    "Tablas 2x2 y Cálculos",
    "Visualización de Datos",
    "Chat"
])

# TAB Conceptos Básicos
with tabs[0]:
    st.header("📌 Conceptos Básicos")
    contenido = cargar_md("contenido/conceptosbasicos.md")
    if contenido:
        st.markdown(contenido)
    else:
        st.info("Agrega el archivo 'contenido/conceptosbasicos.md' para mostrar el contenido.")

# TAB Medidas de Asociación
with tabs[1]:
    st.header("📈 Medidas de Asociación")
    contenido = cargar_md("contenido/medidas_completas.md")
    if contenido:
        st.markdown(contenido)
    else:
        st.info("Agrega el archivo 'contenido/medidas_completas.md' para mostrar el contenido.")

# TAB Diseños de Estudio
with tabs[2]:
    st.header("📊 Diseños de Estudio")
    contenido = cargar_md("contenido/disenos_completos.md")
    if contenido:
        st.markdown(contenido)
    else:
        st.info("Agrega el archivo 'contenido/disenos_completos.md' para mostrar el contenido.")

# TAB Sesgos y Errores
with tabs[3]:
    st.header("⚠️ Sesgos y Errores")
    contenido = cargar_md("contenido/sesgos_completos.md")
    if contenido:
        st.markdown(contenido)
    else:
        st.info("Agrega el archivo 'contenido/sesgos_completos.md' para mostrar el contenido.")

# TAB Glosario Interactivo
with tabs[4]:
    st.header("📚 Glosario Interactivo")
    glosario = cargar_py_variable("contenido/glosario_completo.py", "glosario")
    if glosario:
        for termino, definicion in glosario.items():
            with st.expander(termino):
                st.write(definicion)
    else:
        st.info("Agrega 'contenido/glosario_completo.py' con variable `glosario`.")

# TAB Ejercicios Prácticos
with tabs[5]:
    st.header("🧪 Ejercicios Prácticos")
    preguntas = cargar_py_variable("contenido/ejercicios_completos.py", "preguntas")
    if preguntas:
        for i, q in enumerate(preguntas):
            st.subheader(f"Pregunta {i+1}")
            respuesta = st.radio(q['pregunta'], q['opciones'], key=f"q{i}")
            if st.button(f"Verificar respuesta {i+1}", key=f"btn_{i}"):
                if respuesta == q['respuesta_correcta']:
                    st.success("✅ Correcto")
                else:
                    st.error(f"❌ Incorrecto. La respuesta correcta es: {q['respuesta_correcta']}")
    else:
        st.info("Agrega 'contenido/ejercicios_completos.py' con variable `preguntas`.")

# TAB Tablas 2x2 y Cálculos
with tabs[6]:
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

    if st.button("Calcular medidas y mostrar gráfico"):
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

# TAB Visualización de Datos
with tabs[7]:
    st.header("📊 Visualización de Datos")
    st.info("Carga un archivo CSV para generar gráficos exploratorios.")

    uploaded_file = st.file_uploader("Carga archivo CSV", type=["csv"])

    if uploaded_file:
        try:
            df = pd.read_csv(uploaded_file)
            st.write("Vista previa de los datos cargados:")
            st.dataframe(df.head())

            columnas = df.columns.tolist()
            columna = st.selectbox("Selecciona columna para gráfico boxplot o histograma", columnas)

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

# TAB Chat Gemini
with tabs[8]:
    st.header("🤖 Chat Epidemiológico con Gemini AI")
    st.info("Pregunta cualquier duda epidemiológica. Gemini responde en español y con tono didáctico.")

    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []

    pregunta = st.text_input("Escribe tu pregunta aquí:", key="chat_input")

    if st.button("Enviar"):
        if pregunta.strip():
            st.session_state.chat_history.append({"role":"user", "content": pregunta})
            with st.spinner("Gemini está escribiendo..."):
                respuesta = chat_with_gemini(st.session_state.chat_history)
            st.session_state.chat_history.append({"role":"assistant", "content": respuesta})

    for i, msg in enumerate(st.session_state.chat_history):
        if msg["role"] == "user":
            st.markdown(f"**Tú:** {msg['content']}")
        else:
            st.markdown(f"**Gemini:** {msg['content']}")

# --- Footer con datos de contacto ---
st.markdown("---")
st.markdown(
    """
    <div style="font-size:0.85rem; color: #555; text-align: center; margin-top: 2rem;">
        Creado por <strong>Yolanda Muvdi</strong>, Enfermera Epidemióloga.<br>
        Contacto: <a href="mailto:ymuvdi@gmail.com">ymuvdi@gmail.com</a> | 
        <a href="https://www.linkedin.com/in/yolanda-paola-muvdi-muvdi-778b73152/" target="_blank">LinkedIn</a>
    </div>
    """,
    unsafe_allow_html=True,
)

