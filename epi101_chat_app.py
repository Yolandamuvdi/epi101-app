import streamlit as st
import openai
import matplotlib.pyplot as plt
import pandas as pd
import importlib.util
import sys
import os
import numpy as np
import math

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

if "OPENAI_API_KEY" in st.secrets:
    openai.api_key = st.secrets["OPENAI_API_KEY"]
    st.success("✅ Clave API detectada correctamente.")
else:
    st.error("❌ No se encontró OPENAI_API_KEY. Ve al panel de Secrets en Streamlit Cloud y agrégala.")
    st.stop()

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
    except Exception as e:
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

with tabs[0]:
    st.header("📌 Conceptos Básicos de Epidemiología")
    st.markdown(cargar_md("contenido/conceptosbasicos.md"))

with tabs[1]:
    st.header("📈 Medidas de Asociación")
    st.markdown(cargar_md("contenido/medidas_completas.md"))

with tabs[2]:
    st.header("📊 Diseños de Estudio Epidemiológico")
    st.markdown(cargar_md("contenido/disenos_completos.md"))

with tabs[3]:
    st.header("⚠️ Sesgos y Errores")
    st.markdown(cargar_md("contenido/sesgos_completos.md"))

with tabs[4]:
    st.header("📚 Glosario Interactivo A–Z")
    glosario = cargar_py_variable("contenido/glosario_completo.py", "glosario")
    if glosario is None:
        st.error("No se pudo cargar el glosario.")
    else:
        for termino, definicion in glosario.items():
            with st.expander(termino):
                st.write(definicion)

with tabs[5]:
    st.header("🧪 Ejercicios Prácticos")
    preguntas = cargar_py_variable("contenido/ejercicios_completos.py", "preguntas")
    if preguntas is None:
        st.error("No se pudieron cargar los ejercicios.")
    else:
        for i, q in enumerate(preguntas):
            st.subheader(f"Pregunta {i+1}")
            respuesta = st.radio(q['pregunta'], q['opciones'], key=f"q{i}")
            if st.button(f"Verificar {i+1}", key=f"btn_{i}"):
                if respuesta == q['respuesta_correcta']:
                    st.success("✅ Correcto")
                else:
                    st.error(f"❌ Incorrecto. Respuesta correcta: {q['respuesta_correcta']}")

# -----------------------------
# PESTAÑA MEJORADA: tabs[6]
# -----------------------------
with tabs[6]:
    st.header("📊 Tablas 2x2 y Cálculos Epidemiológicos")

    # Botón de ejemplo: carga valores de ejemplo en session_state
    if st.button("📌 Cargar ejemplo"):
        st.session_state["a_val"] = 30
        st.session_state["b_val"] = 70
        st.session_state["c_val"] = 10
        st.session_state["d_val"] = 90

    # obtener valores previos si existen (para mantener entre interacciones)
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
            # Guardar originales para pruebas
            a0, b0, c0, d0 = int(a), int(b), int(c), int(d)

            # Validación mínima
            if (a0 + b0) == 0 or (c0 + d0) == 0:
                st.error("Las filas de Casos o Controles no pueden ser todas cero. Ingresa valores válidos.")
            else:
                # Aplicar corrección de Haldane-Anscombe si hay ceros en alguna celda
                a_adj, b_adj, c_adj, d_adj = a0, b0, c0, d0
                applied_correction = False
                if 0 in [a0, b0, c0, d0]:
                    a_adj = a0 + 0.5
                    b_adj = b0 + 0.5
                    c_adj = c0 + 0.5
                    d_adj = d0 + 0.5
                    applied_correction = True
                    st.info("⚠️ Corrección Haldane-Anscombe aplicada por presencia de cero(s).")

                # Incidencias
                inc_exp = a_adj / (a_adj + b_adj)
                inc_noexp = c_adj / (c_adj + d_adj)

                # Risk Ratio (RR) y su IC logarítmico
                rr = inc_exp / inc_noexp if inc_noexp > 0 else np.nan
                se_log_rr = math.sqrt((1 / a_adj - 1 / (a_adj + b_adj)) + (1 / c_adj - 1 / (c_adj + d_adj)))
                ci95_rr = (math.exp(math.log(rr) - 1.96 * se_log_rr), math.exp(math.log(rr) + 1.96 * se_log_rr)) if rr > 0 else (np.nan, np.nan)

                # Odds Ratio (OR) y su IC
                orr = (a_adj * d_adj) / (b_adj * c_adj) if (b_adj * c_adj) > 0 else np.nan
                se_log_or = math.sqrt(1 / a_adj + 1 / b_adj + 1 / c_adj + 1 / d_adj)
                ci95_or = (math.exp(math.log(orr) - 1.96 * se_log_or), math.exp(math.log(orr) + 1.96 * se_log_or)) if orr > 0 else (np.nan, np.nan)

                # Diferencia de riesgos (RD) y su IC
                rd = inc_exp - inc_noexp
                se_rd = math.sqrt((inc_exp * (1 - inc_exp) / (a_adj + b_adj)) + (inc_noexp * (1 - inc_noexp) / (c_adj + d_adj)))
                ci95_rd = (rd - 1.96 * se_rd, rd + 1.96 * se_rd)

                # Medidas adicionales
                rae = rd  # Riesgo atribuible en expuestos
                fae = (rae / inc_exp) if inc_exp != 0 else None  # Fracción atribuible en expuestos
                pexp = (a0 + b0) / (a0 + b0 + c0 + d0)  # proporción expuestos en población
                rap = pexp * rd  # Riesgo atribuible poblacional
                # Fracción atribuible poblacional (FAP) = RAP / riesgo poblacional (Ipop)
                ipop = (a0 + c0) / (a0 + b0 + c0 + d0)
                fap = (rap / ipop) if ipop != 0 else None
                nnt = None if rd == 0 else 1 / abs(rd)

                # Mostrar tabla 2x2 original
                st.markdown("**Tabla 2x2 (original)**")
                df_table = pd.DataFrame({
                    "Expuestos": [a0, c0],
                    "No expuestos": [b0, d0]
                }, index=["Casos", "Controles"])
                st.table(df_table)

                # Resultados
                st.subheader("📈 Resultados")
                st.write(f"Incidencia — Expuestos: {inc_exp:.4f}")
                st.write(f"Incidencia — No expuestos: {inc_noexp:.4f}")

                if not np.isnan(rr):
                    st.success(f"RR = {rr:.3f} (IC95%: {ci95_rr[0]:.3f} – {ci95_rr[1]:.3f})")
                else:
                    st.warning("RR no calculable con los datos proporcionados.")

                if not np.isnan(orr):
                    st.success(f"OR = {orr:.3f} (IC95%: {ci95_or[0]:.3f} – {ci95_or[1]:.3f})")
                else:
                    st.warning("OR no calculable con los datos proporcionados.")

                st.info(f"RD = {rd:.4f} (IC95%: {ci95_rd[0]:.4f} – {ci95_rd[1]:.4f})")
                st.write(f"Riesgo atribuible en expuestos (RAE): {rae:.4f}")
                st.write(f"Fracción atribuible en expuestos (FAE): {fae:.2%}" if fae is not None else

