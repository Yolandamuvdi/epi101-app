# epi101_chat_app.py
import streamlit as st
import openai
import matplotlib.pyplot as plt
import pandas as pd
import importlib.util
import sys
import os
import numpy as np
import math
import google.generativeai as genai
# =========================================
# CONFIGURACIÓN DE GEMINI
# =========================================
genai.configure(api_key=st.secrets["GEMINI_API_KEY"])

def chat_with_gemini(prompt):
    model = genai.GenerativeModel("gemini-pro")
    response = model.generate_content(prompt)
    return response.text
# Intentar importar SciPy para pruebas estadísticas
try:
    from scipy.stats import chi2_contingency, fisher_exact
    SCIPY_AVAILABLE = True
except Exception:
    SCIPY_AVAILABLE = False

st.set_page_config(page_title="Epidemiología 101", page_icon="🧪", layout="wide")

# Estilo visual personalizado
st.markdown("""
    <style>
    body, .block-container {
        background: linear-gradient(135deg, #f0f4f8 0%, #d9e2ec 100%);
        color: #1b2838;
        font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
    }
    .block-container {
        padding: 2rem 4rem 3rem 4rem;
        max-width: 1100px;
        margin: auto;
        background: white;
        border-radius: 12px;
        box-shadow: 0 10px 25px rgba(0,0,0,0.1);
    }
    .title {
        font-size: 3rem;
        font-weight: 900;
        color: #0d3b66;
        margin-bottom: 0.2rem;
    }
    .subtitle {
        font-size: 1.3rem;
        color: #3e5c76;
        margin-bottom: 2rem;
        font-weight: 500;
    }
    </style>
""", unsafe_allow_html=True)

st.markdown('<div class="title">🧠 Epidemiología 101 - Asistente educativo</div>', unsafe_allow_html=True)
st.markdown('<div class="subtitle">Plataforma integral para el aprendizaje de conceptos clave de epidemiología, salud pública y análisis de datos, creada por Yolanda Muvdi.</div>', unsafe_allow_html=True)

# API KEY check (Streamlit Secrets)
if "OPENAI_API_KEY" in st.secrets:
    openai.api_key = st.secrets["OPENAI_API_KEY"]
    st.success("✅ Clave API detectada correctamente.")
else:
    st.error("❌ No se encontró OPENAI_API_KEY. Ve al panel de Secrets en Streamlit Cloud y agrégala.")
    # no st.stop() — dejamos la app correr sin chat si no hay clave
    # st.stop()

def cargar_md(ruta):
    try:
        with open(ruta, encoding="utf-8") as f:
            return f.read()
    except Exception as e:
        return f"Error cargando archivo: {e}"

def cargar_py_variable(path_py, var_name):
    """
    Carga una variable de un archivo .py sin eval, usando exec en un dict seguro.
    """
    namespace = {}
    try:
        with open(path_py, "r", encoding="utf-8") as f:
            code = f.read()
        exec(code, namespace)
        return namespace.get(var_name, None)
    except Exception:
        return None

# Pestañas de navegación
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

# --- TAB 0: Conceptos Básicos
with tabs[0]:
    st.header("📌 Conceptos Básicos de Epidemiología")
    # si tienes el .md, lo carga; si no, muestra mensaje breve
    contenido = cargar_md("contenido/conceptosbasicos.md")
    if contenido.startswith("Error cargando"):
        st.write("Aquí iría el contenido de 'Conceptos Básicos'. Coloca 'contenido/conceptosbasicos.md' para mostrarlo.")
    else:
        st.markdown(contenido)

# --- TAB 1: Medidas de Asociación
with tabs[1]:
    st.header("📈 Medidas de Asociación")
    contenido = cargar_md("contenido/medidas_completas.md")
    if contenido.startswith("Error cargando"):
        st.write("Aquí iría el contenido de 'Medidas de Asociación'. Coloca 'contenido/medidas_completas.md' para mostrarlo.")
    else:
        st.markdown(contenido)

# --- TAB 2: Diseños de Estudio
with tabs[2]:
    st.header("📊 Diseños de Estudio Epidemiológico")
    contenido = cargar_md("contenido/disenos_completos.md")
    if contenido.startswith("Error cargando"):
        st.write("Aquí iría el contenido de 'Diseños de Estudio'. Coloca 'contenido/disenos_completos.md' para mostrarlo.")
    else:
        st.markdown(contenido)

# --- TAB 3: Sesgos y Errores
with tabs[3]:
    st.header("⚠️ Sesgos y Errores")
    contenido = cargar_md("contenido/sesgos_completos.md")
    if contenido.startswith("Error cargando"):
        st.write("Aquí iría el contenido de 'Sesgos y Errores'. Coloca 'contenido/sesgos_completos.md' para mostrarlo.")
    else:
        st.markdown(contenido)

# --- TAB 4: Glosario Interactivo
with tabs[4]:
    st.header("📚 Glosario Interactivo A–Z")
    glosario = cargar_py_variable("contenido/glosario_completo.py", "glosario")
    if glosario is None:
        st.write("No se pudo cargar el glosario. Agrega 'contenido/glosario_completo.py' con la variable `glosario`.")
    else:
        for termino, definicion in glosario.items():
            with st.expander(termino):
                st.write(definicion)

# --- TAB 5: Ejercicios Prácticos
with tabs[5]:
    st.header("🧪 Ejercicios Prácticos")
    preguntas = cargar_py_variable("contenido/ejercicios_completos.py", "preguntas")
    if preguntas is None:
        st.write("No se pudieron cargar los ejercicios. Agrega 'contenido/ejercicios_completos.py' con la variable `preguntas`.")
    else:
        for i, q in enumerate(preguntas):
            st.subheader(f"Pregunta {i+1}")
            respuesta = st.radio(q['pregunta'], q['opciones'], key=f"q{i}")
            if st.button(f"Verificar {i+1}", key=f"btn_{i}"):
                if respuesta == q['respuesta_correcta']:
                    st.success("✅ Correcto")
                else:
                    st.error(f"❌ Incorrecto. Respuesta correcta: {q['respuesta_correcta']}")

# --- TAB 6: TABLAS 2x2 Y CÁLCULOS (MEJORADA) ---
with tabs[6]:
    st.header("📊 Tablas 2x2 y Cálculos Epidemiológicos")

    # Botón para cargar ejemplo (guarda en session_state)
    if st.button("📌 Cargar ejemplo"):
        st.session_state["a_val"] = 30
        st.session_state["b_val"] = 70
        st.session_state["c_val"] = 10
        st.session_state["d_val"] = 90

    # Valores por defecto (para mantener entre interacciones)
    a_default = st.session_state.get("a_val", 0)
    b_default = st.session_state.get("b_val", 0)
    c_default = st.session_state.get("c_val", 0)
    d_default = st.session_state.get("d_val", 0)

    col1, col2 = st.columns(2)
    with col1:
        a = st.number_input("Casos con exposición (a)", min_value=0, step=1, value=a_default)
        b = st.number_input("Casos sin exposición (b)", min_value=0, step=1, value=b_default)
    with col2:
        c = st.number_input("Controles con exposición (c)", min_value=0, step=1, value=c_default)
        d = st.number_input("Controles sin exposición (d)", min_value=0, step=1, value=d_default)

    if st.button("Calcular medidas"):
        try:
            # Guardar originales (enteros)
            a0, b0, c0, d0 = int(a), int(b), int(c), int(d)

            # Validación de filas vacías
            if (a0 + b0) == 0 or (c0 + d0) == 0:
                st.error("Las filas de 'Casos' o 'Controles' no pueden ser todas cero. Ingresa valores válidos.")
            else:
                # Aplicar corrección Haldane-Anscombe si hay ceros
                a_adj, b_adj, c_adj, d_adj = a0, b0, c0, d0
                if 0 in [a0, b0, c0, d0]:
                    a_adj = a0 + 0.5
                    b_adj = b0 + 0.5
                    c_adj = c0 + 0.5
                    d_adj = d0 + 0.5
                    st.info("⚠️ Se aplicó corrección de Haldane-Anscombe por presencia de ceros.")

                # Incidencias (proporciones)
                inc_exp = a_adj / (a_adj + b_adj)
                inc_noexp = c_adj / (c_adj + d_adj)

                # RR y su IC (usando aproximación log)
                rr = inc_exp / inc_noexp if inc_noexp > 0 else np.nan
                # evitar dividir por cero en SE: usamos a_adj etc.
                try:
                    se_log_rr = math.sqrt((1 / a_adj - 1 / (a_adj + b_adj)) + (1 / c_adj - 1 / (c_adj + d_adj)))
                except Exception:
                    se_log_rr = np.nan
                if rr > 0 and not np.isnan(se_log_rr):
                    ci95_rr = (math.exp(math.log(rr) - 1.96 * se_log_rr), math.exp(math.log(rr) + 1.96 * se_log_rr))
                else:
                    ci95_rr = (np.nan, np.nan)

                # OR y su IC
                orr = (a_adj * d_adj) / (b_adj * c_adj) if (b_adj * c_adj) > 0 else np.nan
                try:
                    se_log_or = math.sqrt(1 / a_adj + 1 / b_adj + 1 / c_adj + 1 / d_adj)
                except Exception:
                    se_log_or = np.nan
                if orr > 0 and not np.isnan(se_log_or):
                    ci95_or = (math.exp(math.log(orr) - 1.96 * se_log_or), math.exp(math.log(orr) + 1.96 * se_log_or))
                else:
                    ci95_or = (np.nan, np.nan)

                # RD (Diferencia de riesgos) y su IC
                rd = inc_exp - inc_noexp
                try:
                    se_rd = math.sqrt((inc_exp * (1 - inc_exp) / (a_adj + b_adj)) + (inc_noexp * (1 - inc_noexp) / (c_adj + d_adj)))
                except Exception:
                    se_rd = np.nan
                if not np.isnan(se_rd):
                    ci95_rd = (rd - 1.96 * se_rd, rd + 1.96 * se_rd)
                else:
                    ci95_rd = (np.nan, np.nan)

                # RAE, FAE, RAP, FAP, NNT
                rae = rd  # Riesgo atribuible en expuestos (absoluto)
                fae = (rae / inc_exp) if inc_exp != 0 else None  # Fracción atribuible en expuestos
                pexp = (a0 + b0) / (a0 + b0 + c0 + d0)
                rap = pexp * rd  # Riesgo atribuible poblacional (absoluto)
                # riesgo poblacional (Ipop) = (a+c) / N
                ipop = (a0 + c0) / (a0 + b0 + c0 + d0) if (a0 + b0 + c0 + d0) > 0 else None
                fap = (rap / ipop) if (ipop is not None and ipop != 0) else None
                nnt = None if rd == 0 else 1 / abs(rd)

                # Mostrar tabla original en formato 2x2
                st.markdown("**Tabla 2x2 (original)**")
                df_tabla = pd.DataFrame({
                    "Expuestos": [a0, c0],
                    "No expuestos": [b0, d0]
                }, index=["Casos", "Controles"])
                st.table(df_tabla)

                # Mostrar resultados
                st.subheader("📈 Resultados")
                st.write(f"Incidencia — Expuestos: {inc_exp:.4f}")
                st.write(f"Incidencia — No expuestos: {inc_noexp:.4f}")

                if not np.isnan(rr):
                    st.success(f"RR = {rr:.3f} (IC95%: {ci95_rr[0]:.3f} – {ci95_rr[1]:.3f})")
                else:
                    st.warning("RR: no calculable con los datos proporcionados.")

                if not np.isnan(orr):
                    st.success(f"OR = {orr:.3f} (IC95%: {ci95_or[0]:.3f} – {ci95_or[1]:.3f})")
                else:
                    st.warning("OR: no calculable con los datos proporcionados.")

                st.info(f"RD = {rd:.4f} (IC95%: {ci95_rd[0]:.4f} – {ci95_rd[1]:.4f})")

                st.write(f"Riesgo atribuible en expuestos (RAE): {rae:.4f}")
                if fae is not None:
                    st.write(f"Fracción atribuible en expuestos (FAE): {fae:.2%}")
                else:
                    st.write("FAE: No calculable")

                st.write(f"Riesgo atribuible poblacional (RAP): {rap:.4f}")
                if fap is not None:
                    st.write(f"Fracción atribuible poblacional (FAP): {fap:.2%}")
                else:
                    st.write("FAP: No calculable")

                if nnt is not None:
                    st.write(f"NNT (Número Necesario a Tratar/Evitar): {nnt:.2f}")
                else:
                    st.write("NNT: No calculable (RD = 0)")

                # Interpretación automática (breve)
                st.subheader("🧾 Interpretación rápida")
                if not np.isnan(rr):
                    if rr > 1:
                        st.warning(f"RR={rr:.2f}: asociación positiva — los expuestos tienen aproximadamente {(rr-1)*100:.0f}% más riesgo relativo que los no expuestos.")
                    elif rr < 1:
                        st.success(f"RR={rr:.2f}: posible efecto protector — los expuestos tienen aproximadamente {(1-rr)*100:.0f}% menos riesgo relativo.")
                    else:
                        st.info("RR≈1: no se observan diferencias en riesgo relativo entre los grupos.")
                else:
                    st.info("No se puede interpretar RR con los datos actuales.")

                # Pruebas estadísticas (usando datos originales sin corrección para Chi2/Fisher)
                if SCIPY_AVAILABLE:
                    try:
                        tabla_orig = np.array([[a0, b0], [c0, d0]])
                        chi2, p_chi, _, _ = chi2_contingency(tabla_orig, correction=False)
                        _, p_fisher = fisher_exact(tabla_orig)
                        st.write("**Pruebas de asociación**")
                        st.write(f"Chi² = {chi2:.3f}, p = {p_chi:.4f}")
                        st.write(f"Test exacto de Fisher (dos colas): p = {p_fisher:.4f}")
                    except Exception as e:
                        st.info(f"No se pudo calcular Chi²/Fisher: {e}")
                else:
                    st.info("Instala SciPy (`pip install scipy`) para obtener Chi² y Fisher.")

                # Gráfico de incidencias (con % sobre barras)
                fig, ax = plt.subplots(figsize=(6, 3))
                bars = ax.bar(["Expuestos", "No expuestos"], [inc_exp, inc_noexp],
                              color=["#ff9999", "#99ccff"], edgecolor="black")
                ax.set_ylim(0, 1)
                ax.set_ylabel("Incidencia (proporción)")
                ax.set_title("Comparación de incidencias")
                for bar, val in zip(bars, [inc_exp, inc_noexp]):
                    ax.text(bar.get_x() + bar.get_width() / 2, val + 0.02, f"{val:.1%}", ha="center", va="bottom", fontsize=9)
                st.pyplot(fig)

                # Preparar dataframe de resultados y botón descarga CSV
                df_resultados = pd.DataFrame({
                    "Medida": ["RR", "OR", "RD", "RAE", "FAE", "RAP", "FAP", "NNT"],
                    "Valor": [rr, orr, rd, rae, fae, rap, fap, nnt],
                    "IC95_inf": [ci95_rr[0], ci95_or[0], ci95_rd[0], None, None, None, None, None],
                    "IC95_sup": [ci95_rr[1], ci95_or[1], ci95_rd[1], None, None, None, None, None]
                })

                csv = df_resultados.to_csv(index=False).encode("utf-8")
                st.download_button("📥 Descargar resultados CSV", csv, "epi2x2_resultados.csv", "text/csv")

        except Exception as e:
            st.error(f"Error en los cálculos: {e}")

# --- TAB 7: Visualización de Datos ---
with tabs[7]:
    st.header("📈 Visualización de Datos")
    st.markdown("Carga tus datos en formato CSV para graficar.")
    uploaded_file = st.file_uploader("Sube un archivo CSV", type=["csv"])
    if uploaded_file is not None:
        try:
            df = pd.read_csv(uploaded_file)
            st.dataframe(df)
            col_x = st.selectbox("Selecciona variable para eje X", df.columns)
            col_y = st.selectbox("Selecciona variable para eje Y", df.columns)
            tipo = st.selectbox("Tipo de gráfico", ["Barras", "Líneas", "Dispersión"])

            fig, ax = plt.subplots()
            if tipo == "Barras":
                ax.bar(df[col_x], df[col_y])
            elif tipo == "Líneas":
                ax.plot(df[col_x], df[col_y])
            elif tipo == "Dispersión":
                ax.scatter(df[col_x], df[col_y])

            ax.set_xlabel(col_x)
            ax.set_ylabel(col_y)
            ax.set_title(f"{tipo} entre {col_x} y {col_y}")
            st.pyplot(fig)
        except Exception as e:
            st.error(f"Error al leer el CSV: {e}")

# --- TAB 8: Chat ---
with tabs[8]:
    st.header("💬 Chat con Epidemiología 101")
    if "messages" not in st.session_state:
        st.session_state["messages"] = [{
            "role": "system",
            "content": "Eres un docente experto en epidemiología. Explica conceptos y resuelve preguntas con claridad y evidencia."
        }]
    # Mostrar mensajes anteriores
    for msg in st.session_state["messages"]:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])
    # Entrada de usuario
    if prompt := st.chat_input("Haz tu pregunta de epidemiología..."):
        st.session_state["messages"].append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)
        with st.chat_message("assistant"):
            try:
                # Si no tienes API key en secrets, este bloque fallará; capturamos y mostramos error amigable
                if not hasattr(openai, "api_key") or openai.api_key is None:
                    st.error("No hay API key de OpenAI configurada en Streamlit Secrets. Agrega OPENAI_API_KEY si quieres usar el chat.")
                else:
                    response = openai.chat.completions.create(
                        model="gpt-3.5-turbo",
                        messages=st.session_state["messages"]
                    )
                    reply = response.choices[0].message.content
                    st.markdown(reply)
                    st.session_state["messages"].append({"role": "assistant", "content": reply})
            except Exception as e:
                st.error(f"Error al comunicarse con OpenAI: {e}")

