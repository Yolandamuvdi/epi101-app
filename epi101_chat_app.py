# epidemiologia101_app.py
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import io
from datetime import datetime

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

def pagina_inicio():
    st.markdown("# Bienvenido a Epidemiología 101")
    st.markdown("Una solución creada para aprender epidemiología de manera interactiva 🎯")
    st.info("Selecciona una sección en la barra lateral para comenzar.")

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
    # Simple CI aproximado
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
    # placeholder, usar chi2 test o Fisher exact en real
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
    ax.errorbar([1,2],[rr,or_],[rr-rr_l, or_-or_l],[rr_u-rr, or_u-or_], fmt='o')
    ax.set_xticks([1,2]); ax.set_xticklabels(["RR","OR"])
    ax.set_title("Forest Plot")
    return fig

def plot_barras_expuestos(a,b,c,d):
    fig, ax = plt.subplots()
    ax.bar(["Casos exp","No exp","Casos no exp","No casos no exp"], [a,b,c,d], color="#0d3b66")
    ax.set_title("Distribución 2x2")
    st.pyplot(fig, use_container_width=True)

# --- Simulación adaptativa para gamificación ---
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
def barra_lateral(seleccion_actual, user_info):
    opciones = [
        "📚 Academia", "🛠️ Toolkit", "📈 Medidas de Asociación", "📊 Diseños de Estudio",
        "⚠️ Sesgos y Errores", "📚 Glosario Interactivo", "🧪 Ejercicios Prácticos",
        "📊 Tablas 2x2 y Cálculos", "📊 Visualización de Datos", "🎥 Multimedia YouTube",
        "🤖 Chat Epidemiológico", "🎯 Gamificación", "📢 Brotes"
    ]
    seleccion_sidebar = st.sidebar.radio("Ir a sección:", opciones, index=opciones.index(seleccion_actual) if seleccion_actual in opciones else 0)
    st.sidebar.markdown("---")
    st.sidebar.markdown(f"**Usuario:** {user_info.get('name','Demo')} | Rol: {user_info.get('role','Demo')}")
    st.sidebar.markdown("### Roadmap")
    st.sidebar.markdown("- M1 Conceptos\n- M2 Medidas\n- M3 Toolkit\n- M4 Brotes")
    return seleccion_sidebar

# --- Main ---
def main():
    st.set_page_config(page_title="Epidemiología 101", layout="wide")
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
        if st.sidebar.button("Abrir Academia"):
            st.session_state.seccion = "📚 Academia"
        return

    seleccion = barra_lateral(st.session_state.seccion, user_info)
    st.session_state.seccion = seleccion

    # -------------------- SECCIONES --------------------
    if seleccion == "📚 Academia":
        st.header("📚 Academia")
        contenido = cargar_md("contenido/conceptosbasicos.md")
        if contenido: st.markdown(contenido)
        else: st.info("Archivo 'contenido/conceptosbasicos.md' no encontrado.")
    
    elif seleccion == "🛠️ Toolkit":
        st.header("🛠️ Toolkit")
        if user_info.get("role") in ["Pro","Estudiante","Demo"]:
            st.markdown("Herramientas de análisis y APIs externas")
        else:
            st.warning("Toolkit solo disponible para usuarios Pro o Estudiantes.")

    elif seleccion == "📈 Medidas de Asociación":
        st.header(seleccion)
        contenido = cargar_md("contenido/medidas_completas.md")
        if contenido: st.markdown(contenido)
        else: st.info("Archivo 'contenido/medidas_completas.md' no encontrado.")

    elif seleccion == "📊 Diseños de Estudio":
        st.header(seleccion)
        contenido = cargar_md("contenido/disenos_completos.md")
        if contenido: st.markdown(contenido)
        else: st.info("Archivo 'contenido/disenos_completos.md' no encontrado.")

    elif seleccion == "⚠️ Sesgos y Errores":
        st.header(seleccion)
        contenido = cargar_md("contenido/sesgos_completos.md")
        if contenido: st.markdown(contenido)
        else: st.info("Archivo 'contenido/sesgos_completos.md' no encontrado.")

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
        st.info("Aquí iría la simulación de brotes (módulo interactivo).")

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
        pregunta = st.text_input("Pregunta epidemiológica")
        if st.button("Enviar") and pregunta:
            st.info("Aquí iría la respuesta generada por IA (Gemini).")

# --- Run App ---
if __name__ == "__main__":
    main()
