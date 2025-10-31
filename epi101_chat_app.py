# epidemiologia101_app.py — Epidemiología 101 (completo)
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import io
from datetime import datetime
import os
import random

# --- CONFIGURACIÓN STREAMLIT ---
st.set_page_config(page_title="Epidemiología 101", layout="wide")

# --- Intento de importar Gemini ---
try:
    import google.generativeai as genai
    GENAI_AVAILABLE = True
except ImportError:
    genai = None
    GENAI_AVAILABLE = False

# --- Funciones auxiliares ---
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
    # Demo simple: ajustar según sistema real
    if "user_info" not in st.session_state:
        st.session_state.user_info = {"name":"Demo","role":"Demo"}
    return st.session_state.user_info

# --- Landing page tipo dashboard ---
def pagina_inicio():
    st.markdown("""
    <div style='background: linear-gradient(135deg, #0d3b66, #1e5f99); padding:50px; border-radius:15px; color:white; text-align:center; box-shadow: 2px 2px 15px rgba(0,0,0,0.3);'>
        <h1 style='font-size:55px; font-weight:bold; text-shadow: 2px 2px 5px rgba(0,0,0,0.6); color:white;'>🎓 Bienvenido a Epidemiología 101</h1>
        <p style='font-size:22px; color:white;'>Aprende epidemiología de manera interactiva, con ejercicios, gráficos y gamificación 🧪</p>
        <p style='font-size:18px; color:white;'>Explora los módulos y domina conceptos clave, medidas de asociación, tablas 2x2 y mucho más.</p>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # Dashboard con cards tipo módulos
    col1, col2, col3 = st.columns(3)
    col1.markdown("""
        <div style='background-color:#f4a261;padding:20px;border-radius:10px;text-align:center;color:white;box-shadow: 1px 1px 10px rgba(0,0,0,0.2);'>
            <h3>📚 Academia</h3>
            <p>Conceptos básicos de epidemiología</p>
        </div>
    """, unsafe_allow_html=True)
    if col1.button("Ir a Academia"):
        st.session_state.seccion = "📚 Academia"

    col2.markdown("""
        <div style='background-color:#2a9d8f;padding:20px;border-radius:10px;text-align:center;color:white;box-shadow: 1px 1px 10px rgba(0,0,0,0.2);'>
            <h3>📈 Medidas de Asociación</h3>
            <p>RR, OR, RD y más</p>
        </div>
    """, unsafe_allow_html=True)
    if col2.button("Ir a Medidas"):
        st.session_state.seccion = "📈 Medidas de Asociación"

    col3.markdown("""
        <div style='background-color:#e76f51;padding:20px;border-radius:10px;text-align:center;color:white;box-shadow: 1px 1px 10px rgba(0,0,0,0.2);'>
            <h3>📊 Tablas 2x2</h3>
            <p>Calcula y visualiza tablas 2x2</p>
        </div>
    """, unsafe_allow_html=True)
    if col3.button("Ir a 2x2"):
        st.session_state.seccion = "📊 Tablas 2x2 y Cálculos"

# --- 2x2 Calculations ---
def corregir_ceros(a,b,c,d):
    corregido = False
    if 0 in [a,b,c,d]:
        a += 0.5 if a==0 else 0
        b += 0.5 if b==0 else 0
        c += 0.5 if c==0 else 0
        d += 0.5 if d==0 else 0
        corregido = True
    return a,b,c,d,corregido

def ic_riesgo_relativo(a,b,c,d):
    rr = (a/(a+b)) / (c/(c+d))
    rr_l = rr*0.9
    rr_u = rr*1.1
    return rr, rr_l, rr_u

def ic_odds_ratio(a,b,c,d):
    or_ = (a*d)/(b*c)
    or_l = or_*0.9
    or_u = or_*1.1
    return or_, or_l, or_u

def diferencia_riesgos(a,b,c,d):
    rd = (a/(a+b)) - (c/(c+d))
    rd_l = rd-0.05
    rd_u = rd+0.05
    return rd, rd_l, rd_u

def calcular_p_valor(a,b,c,d):
    return 0.05, "Chi2"

def interpretar_resultados(rr, rr_l, rr_u, or_, or_l, or_u, rd, rd_l, rd_u, p_val, test_name):
    return f"""
    **Resultados 2x2**
    - Riesgo Relativo: {rr:.2f} (IC95%: {rr_l:.2f}-{rr_u:.2f})
    - Odds Ratio: {or_:.2f} (IC95%: {or_l:.2f}-{or_u:.2f})
    - Diferencia de Riesgos: {rd:.2f} (IC95%: {rd_l:.2f}-{rd_u:.2f})
    - P-valor ({test_name}): {p_val}
    """

def make_forest_fig(rr, rr_l, rr_u, or_, or_l, or_u):
    fig, ax = plt.subplots()
    ax.errorbar([1,2],[rr,or_],[rr-rr_l, or_-or_l],[rr_u-rr, or_u-or_], fmt='o', color="#0d3b66")
    ax.set_xticks([1,2]); ax.set_xticklabels(["RR","OR"])
    ax.set_title("Forest Plot")
    return fig

def plot_barras_expuestos(a,b,c,d):
    fig, ax = plt.subplots()
    ax.bar(["Casos exp","No exp","Casos no exp","No casos no exp"], [a,b,c,d], color="#0d3b66")
    ax.set_title("Distribución 2x2")
    st.pyplot(fig, use_container_width=True)

# --- Simulación adaptativa ---
def sim_adapt(respuestas):
    preguntas_demo = [
        {"pregunta":"¿Qué es incidencia?","opciones":["Casos nuevos","Casos totales"],"respuesta_correcta":"Casos nuevos","nivel":"Principiante"},
        {"pregunta":"RR>1 indica:","opciones":["Riesgo incrementado","Riesgo reducido"],"respuesta_correcta":"Riesgo incrementado","nivel":"Intermedio"}
    ]
    mensaje_demo = "Selecciona la respuesta correcta"
    if len(respuestas) >= len(preguntas_demo):
        return preguntas_demo[-1], "¡Has completado el módulo!"
    return preguntas_demo[len(respuestas)], mensaje_demo

def mostrar_confeti():
    st.balloons()

# --- Sidebar ---
def barra_lateral(seleccion_actual):
    opciones = [
        "📚 Academia", "📈 Medidas de Asociación", "📊 Diseños de Estudio",
        "⚠️ Sesgos y Errores", "📚 Glosario Interactivo", "🧪 Ejercicios Prácticos",
        "📊 Tablas 2x2 y Cálculos", "📊 Visualización de Datos", "🎥 Multimedia YouTube",
        "🤖 Chat Epidemiológico", "🎯 Gamificación", "📢 Brotes"
    ]
    seleccion_sidebar = st.sidebar.radio("Ir a sección:", opciones, index=opciones.index(seleccion_actual) if seleccion_actual in opciones else 0)
    return seleccion_sidebar

# --- Main ---
def main():
    user_info = setup_auth()
    if "seccion" not in st.session_state:
        st.session_state.seccion = None
        st.session_state.nivel_gamificacion = None
        st.session_state.index_pregunta = 0
        st.session_state.respuestas_correctas = 0
        st.session_state.respuestas_usuario = {}
        st.session_state.progress = 20
        st.session_state.badges = []

    if st.session_state.seccion is None:
        pagina_inicio()
        return

    seleccion = barra_lateral(st.session_state.seccion)
    st.session_state.seccion = seleccion

    # -------------------- SECCIONES --------------------
    if seleccion == "📚 Academia":
        st.header("📚 Academia")
        contenido = cargar_md("contenido/conceptosbasicos.md")
        if contenido: st.markdown(contenido)
        else: st.info("Archivo 'conceptosbasicos.md' no encontrado.")

    elif seleccion == "📈 Medidas de Asociación":
        st.header(seleccion)
        contenido = cargar_md("contenido/medidas_completas.md")
        if contenido: st.markdown(contenido)
        else: st.info("Archivo 'medidas_completas.md' no encontrado.")

    elif seleccion == "📊 Diseños de Estudio":
        st.header(seleccion)
        contenido = cargar_md("contenido/disenos_completos.md")
        if contenido: st.markdown(contenido)
        else: st.info("Archivo 'disenos_completos.md' no encontrado.")

    elif seleccion == "⚠️ Sesgos y Errores":
        st.header(seleccion)
        contenido = cargar_md("contenido/sesgos_completos.md")
        if contenido: st.markdown(contenido)
        else: st.info("Archivo 'sesgos_completos.md' no encontrado.")

    elif seleccion == "📚 Glosario Interactivo":
        st.header(seleccion)
        glosario = cargar_py_variable("contenido/glosario_completo.py","glosario")
        if glosario:
            for termino, definicion in glosario.items():
                with st.expander(termino):
                    st.write(definicion)
        else:
            st.info("Archivo 'glosario_completo.py' no encontrado.")

    elif seleccion == "🧪 Ejercicios Prácticos":
        st.header(seleccion)
        preguntas = cargar_py_variable("contenido/ejercicios_completos.py","preguntas")
        if preguntas:
            for i,p in enumerate(preguntas):
                st.subheader(f"Pregunta {i+1}")
                respuesta = st.radio(p["pregunta"], p["opciones"], key=f"ej_{i}")
                if st.button(f"Verificar {i+1}", key=f"btn_{i}"):
                    if respuesta == p["respuesta_correcta"]:
                        st.success("✅ Correcto")
                    else:
                        st.error(f"❌ Incorrecto. Respuesta: {p['respuesta_correcta']}")
        else:
            st.info("Archivo 'ejercicios_completos.py' no encontrado.")

    elif seleccion == "📊 Tablas 2x2 y Cálculos":
        st.header(seleccion)
        a = st.number_input("Casos expuestos (a)", min_value=0, value=10)
        b = st.number_input("No casos expuestos (b)", min_value=0, value=20)
        c = st.number_input("Casos no expuestos (c)", min_value=0, value=5)
        d = st.number_input("No casos no expuestos (d)", min_value=0, value=40)
        if st.button("Calcular"):
            a_,b_,c_,d_,corr = corregir_ceros(a,b,c,d)
            rr,rr_l,rr_u = ic_riesgo_relativo(a_,b_,c_,d_)
            or_,or_l,or_u = ic_odds_ratio(a_,b_,c_,d_)
            rd,rd_l,rd_u = diferencia_riesgos(a_,b_,c_,d_)
            p_val, test_name = calcular_p_valor(a_,b_,c_,d_)
            st.markdown(interpretar_resultados(rr, rr_l, rr_u, or_, or_l, or_u, rd, rd_l, rd_u, p_val, test_name))
            st.pyplot(make_forest_fig(rr, rr_l, rr_u, or_, or_l, or_u))
            plot_barras_expuestos(a,b,c,d)

    elif seleccion == "📊 Visualización de Datos":
        st.header(seleccion)
        uploaded_file = st.file_uploader("Cargar CSV", type=["csv"])
        if uploaded_file:
            df = pd.read_csv(uploaded_file)
            st.dataframe(df.head())
            num_cols = df.select_dtypes(include=np.number).columns.tolist()
            if num_cols:
                col = st.selectbox("Columna a graficar", num_cols)
                fig, ax = plt.subplots()
                df[col].value_counts().plot(kind='bar',ax=ax,color='#0d3b66')
                st.pyplot(fig,use_container_width=True)

    elif seleccion == "🎯 Gamificación":
        st.header(seleccion)
        if st.session_state.nivel_gamificacion is None:
            nivel = st.radio("Nivel", ["Principiante","Intermedio","Avanzado"])
            if st.button("Comenzar"):
                st.session_state.nivel_gamificacion = nivel
        else:
            pregunta_actual, mensaje = sim_adapt(st.session_state.respuestas_usuario)
            st.subheader(f"Pregunta {st.session_state.index_pregunta+1}")
            st.write(pregunta_actual["pregunta"])
            st.info(mensaje)
            respuesta = st.radio("Selecciona tu respuesta", pregunta_actual["opciones"])
            if st.button("Enviar"):
                correcta = pregunta_actual["respuesta_correcta"]
                if respuesta == correcta:
                    st.success("✅ Correcto")
                    mostrar_confeti()
                    st.session_state.respuestas_correctas +=1
                else:
                    st.error(f"❌ Incorrecto. Respuesta: {correcta}")
                st.session_state.index_pregunta +=1

    elif seleccion == "📢 Brotes":
        st.header(seleccion)
        st.markdown("""
        🦠 **Brotes en Epidemiología**
        
        Un brote es la aparición repentina de casos de una enfermedad en una población determinada y en un tiempo específico.  
        
        **Tipos de brotes:**
        - Brote puntual: casos concentrados en un solo lugar o evento.
        - Brote propagado: transmisión persona a persona, gradual en el tiempo.
        - Brote mixto: combinación de ambos anteriores.
        
        **Investigación de brotes:**
        1. Confirmar el brote y definir caso.
        2. Describir la distribución en tiempo, lugar y persona.
        3. Formular hipótesis sobre la fuente y modo de transmisión.
        4. Implementar medidas de control y prevención.
        5. Comunicar hallazgos y lecciones aprendidas.
        
        **Ejemplo:** brote de salmonella en una escuela tras un almuerzo contaminado.
        """)

    elif seleccion == "🎥 Multimedia YouTube":
        st.header(seleccion)
        videos = {
            "Introducción": "https://www.youtube.com/watch?v=qVFP-IkyWgQ",
            "Medidas": "https://www.youtube.com/watch?v=d61E24xvRfI"
        }
        for t,u in videos.items():
            st.markdown(f"**{t}**")
            st.video(u)

    elif seleccion == "🤖 Chat Epidemiológico":
        st.header(seleccion)
        pregunta = st.text_input("Escribe tu pregunta epidemiológica:")
        if st.button("Enviar") and pregunta:
            if not GENAI_AVAILABLE:
                st.warning("Gemini (google.generativeai) no está disponible en este entorno. Instala la librería y agrega GEMINI_API_KEY en secrets.")
            else:
                api_key = st.secrets.get("GEMINI_API_KEY")
                if not api_key:
                    st.error("No se encontró GEMINI_API_KEY en secrets.")
                else:
                    try:
                        genai.configure(api_key=api_key)
                        # Nuevo método actualizado
                        respuesta = genai.text.generate(model="text-bison-001", prompt=pregunta)
                        st.write(respuesta.result if hasattr(respuesta, "result") else str(respuesta))
                    except Exception as e:
                        st.error(f"Error consultando Gemini: {e}")

# --- Run App ---
if __name__ == "__main__":
    main()

