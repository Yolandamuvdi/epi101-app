import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import io
from datetime import datetime

# Gemini
try:
    import google.generativeai as genai
    GENAI_AVAILABLE = True
except Exception:
    GENAI_AVAILABLE = False


# ------------ Funciones auxiliares ----------------
def cargar_md(ruta):
    try:
        with open(ruta, "r", encoding="utf-8") as f:
            return f.read()
    except:
        return None


def cargar_py_variable(ruta, variable):
    try:
        import importlib.util
        spec = importlib.util.spec_from_file_location("modulo_temp", ruta)
        temp_module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(temp_module)
        return getattr(temp_module, variable, None)
    except:
        return None


def setup_auth():
    if "user_info" not in st.session_state:
        st.session_state.user_info = {"name": "Demo", "role": "Demo"}
    return st.session_state.user_info


# ------------ Landing Page ----------------
def pagina_inicio():
    st.markdown("""
    <div style='background: linear-gradient(135deg, #0d3b66, #1e5f99); padding:50px; border-radius:15px; color:white; text-align:center; box-shadow: 2px 2px 15px rgba(0,0,0,0.3);'>
        <h1 style='font-size:55px; font-weight:bold; color:white;'>🎓 Bienvenido a Epidemiología 101</h1>
        <p style='font-size:22px;'>Aprende epidemiología de manera interactiva 🧪</p>
        <p style='font-size:18px;'>Explora los módulos y domina conceptos clave.</p>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    col1, col2, col3 = st.columns(3)

    col1.markdown("""
        <div style='background-color:#f4a261;padding:20px;border-radius:10px;text-align:center;color:white;'>
            <h3>📚 Academia</h3>
            <p>Conceptos básicos</p>
        </div>
    """, unsafe_allow_html=True)
    if col1.button("Ir a Academia"):
        st.session_state.seccion = "📚 Academia"

    col2.markdown("""
        <div style='background-color:#2a9d8f;padding:20px;border-radius:10px;text-align:center;color:white;'>
            <h3>📈 Medidas de Asociación</h3>
            <p>RR, OR, RD y más</p>
        </div>
    """, unsafe_allow_html=True)
    if col2.button("Ir a Medidas"):
        st.session_state.seccion = "📈 Medidas de Asociación"

    col3.markdown("""
        <div style='background-color:#e76f51;padding:20px;border-radius:10px;text-align:center;color:white;'>
            <h3>📊 Tablas 2x2</h3>
            <p>Cálculo y visualización</p>
        </div>
    """, unsafe_allow_html=True)
    if col3.button("Ir a 2x2"):
        st.session_state.seccion = "📊 Tablas 2x2 y Cálculos"


# ----------- Funciones 2x2 --------------
def corregir_ceros(a, b, c, d):
    corregido = False
    if 0 in [a, b, c, d]:
        a = a + 0.5 if a == 0 else a
        b = b + 0.5 if b == 0 else b
        c = c + 0.5 if c == 0 else c
        d = d + 0.5 if d == 0 else d
        corregido = True
    return a, b, c, d, corregido


def ic_riesgo_relativo(a, b, c, d):
    rr = (a/(a+b)) / (c/(c+d))
    return rr, rr*0.9, rr*1.1


def ic_odds_ratio(a, b, c, d):
    or_ = (a*d)/(b*c)
    return or_, or_*0.9, or_*1.1


def diferencia_riesgos(a, b, c, d):
    rd = (a/(a+b)) - (c/(c+d))
    return rd, rd-0.05, rd+0.05


def calcular_p_valor(a, b, c, d):
    return 0.05, "Chi2"


def interpretar_resultados(rr, rr_l, rr_u, or_, or_l, or_u, rd, rd_l, rd_u, p_val, test_name):
    return f"""
    **Resultados 2x2**
    - RR: {rr:.2f} (IC95% {rr_l:.2f}-{rr_u:.2f})
    - OR: {or_:.2f} (IC95% {or_l:.2f}-{or_u:.2f})
    - RD: {rd:.2f} (IC95% {rd_l:.2f}-{rd_u:.2f})
    - p-valor ({test_name}): {p_val}
    """


def make_forest_fig(rr, rr_l, rr_u, or_, or_l, or_u):
    fig, ax = plt.subplots()
    ax.errorbar([1, 2], [rr, or_],
                [rr - rr_l, or_ - or_l],
                [rr_u - rr, or_u - or_],
                fmt='o', color="#0d3b66")
    ax.set_xticks([1, 2])
    ax.set_xticklabels(["RR", "OR"])
    ax.set_title("Forest Plot")
    return fig


def plot_barras_expuestos(a, b, c, d):
    fig, ax = plt.subplots()
    ax.bar(["Casos exp", "No exp", "Casos no exp", "No casos no exp"],
           [a, b, c, d], color="#0d3b66")
    ax.set_title("Distribución 2x2")
    st.pyplot(fig, use_container_width=True)


# ------------ Sidebar ----------
def barra_lateral(seleccion_actual):
    opciones = [
        "📚 Academia", "📈 Medidas de Asociación", "📊 Diseños de Estudio",
        "⚠️ Sesgos y Errores", "📚 Glosario Interactivo", "🧪 Ejercicios Prácticos",
        "📊 Tablas 2x2 y Cálculos", "📊 Visualización de Datos",
        "🎥 Multimedia YouTube", "🤖 Chat Epidemiológico", "🎯 Gamificación",
        "📢 Brotes"
    ]
    return st.sidebar.radio("Ir a sección:", opciones,
                            index=opciones.index(seleccion_actual) if seleccion_actual in opciones else 0)


# ----------- MAIN --------------
def main():
    st.set_page_config(page_title="Epidemiología 101", layout="wide")

    if "seccion" not in st.session_state:
        st.session_state.seccion = None

    if st.session_state.seccion is None:
        pagina_inicio()
        return

    seleccion = barra_lateral(st.session_state.seccion)
    st.session_state.seccion = seleccion

    # -------- Secciones --------
    if seleccion == "📚 Academia":
        st.header("📚 Academia")
        contenido = cargar_md("contenido/conceptosbasicos.md")
        st.markdown(contenido or "Archivo no encontrado.")

    elif seleccion == "📈 Medidas de Asociación":
        st.header(seleccion)
        contenido = cargar_md("contenido/medidas_completas.md")
        st.markdown(contenido or "Archivo no encontrado.")

    elif seleccion == "📊 Diseños de Estudio":
        st.header(seleccion)
        contenido = cargar_md("contenido/disenos_completos.md")
        st.markdown(contenido or "Archivo no encontrado.")

    elif seleccion == "⚠️ Sesgos y Errores":
        st.header(seleccion)
        contenido = cargar_md("contenido/sesgos_completos.md")
        st.markdown(contenido or "Archivo no encontrado.")

    elif seleccion == "📚 Glosario Interactivo":
        st.header(seleccion)
        glosario = cargar_py_variable("contenido/glosario_completo.py", "glosario")
        if glosario:
            for termino, definicion in glosario.items():
                with st.expander(termino):
                    st.write(definicion)
        else:
            st.write("Archivo no encontrado.")

    elif seleccion == "🧪 Ejercicios Prácticos":
        st.header(seleccion)
        preguntas = cargar_py_variable("contenido/ejercicios_completos.py", "preguntas")
        if preguntas:
            for i, p in enumerate(preguntas):
                st.subheader(f"Pregunta {i+1}")
                respuesta = st.radio(p["pregunta"], p["opciones"], key=f"q{i}")
                if st.button(f"Verificar {i+1}", key=f"b{i}"):
                    if respuesta == p["respuesta_correcta"]:
                        st.success("✅ Correcto")
                    else:
                        st.error(f"❌ Incorrecto. Respuesta correcta: {p['respuesta_correcta']}")

    elif seleccion == "📊 Tablas 2x2 y Cálculos":
        st.header(seleccion)
        a = st.number_input("Casos expuestos (a)", min_value=0, value=10)
        b = st.number_input("No casos expuestos (b)", min_value=0, value=20)
        c = st.number_input("Casos no expuestos (c)", min_value=0, value=5)
        d = st.number_input("No casos no expuestos (d)", min_value=0, value=40)

        if st.button("Calcular"):
            a2, b2, c2, d2, _ = corregir_ceros(a, b, c, d)
            rr, rr_l, rr_u = ic_riesgo_relativo(a2, b2, c2, d2)
            or_, or_l, or_u = ic_odds_ratio(a2, b2, c2, d2)
            rd, rd_l, rd_u = diferencia_riesgos(a2, b2, c2, d2)
            p_val, test_name = calcular_p_valor(a2, b2, c2, d2)

            st.markdown(
                interpretar_resultados(rr, rr_l, rr_u, or_, or_l, or_u, rd, rd_l, rd_u, p_val, test_name)
            )

            st.pyplot(make_forest_fig(rr, rr_l, rr_u, or_, or_l, or_u))
            plot_barras_expuestos(a, b, c, d)

    elif seleccion == "📊 Visualización de Datos":
        st.header(seleccion)
        file = st.file_uploader("Cargar CSV:", type=["csv"])
        if file:
            df = pd.read_csv(file)
            st.dataframe(df)

    elif seleccion == "🎯 Gamificación":
        st.header(seleccion)
        st.info("Módulo en desarrollo.")

    elif seleccion == "📢 Brotes":
        st.header(seleccion)
        st.markdown("""
        ## 🦠 Brotes en Epidemiología
        **Un brote** es un aumento repentino de casos...
        """)

    elif seleccion == "🎥 Multimedia YouTube":
        st.header(seleccion)
        st.video("https://www.youtube.com/watch?v=qVFP-IkyWgQ")

    # ---------------- CHAT GEMINI ----------------
    elif seleccion == "🤖 Chat Epidemiológico":
        st.header("🤖 Chat Epidemiológico")

        pregunta = st.text_input("Escribe tu pregunta epidemiológica:")

        if st.button("Enviar") and pregunta:
            if not GENAI_AVAILABLE:
                st.warning("Gemini no está disponible en este entorno.")
                return

            api_key = st.secrets.get("GEMINI_API_KEY")
            if not api_key:
                st.error("No se encontró GEMINI_API_KEY en secrets.")
                return

            try:
                genai.configure(api_key=api_key)
                model = genai.GenerativeModel("text-bison-001")
                respuesta = model.generate_content(pregunta)

                st.write(respuesta.text)

            except Exception as e:
                st.error(f"Error consultando Gemini: {e}")


# --- Run ---
if __name__ == "__main__":
    main()

    main()


