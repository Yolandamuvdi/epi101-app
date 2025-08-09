import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import chi2_contingency, fisher_exact
from streamlit_extras.let_it_rain import rain

# Constantes de secciones con emojis
SECCIONES = [
    ("🧪", "Inicio"),
    ("📌", "Conceptos Básicos"),
    ("📈", "Medidas de Asociación"),
    ("📊", "Diseños de Estudio"),
    ("⚠️", "Sesgos y Errores"),
    ("📚", "Glosario Interactivo"),
    ("🧪", "Ejercicios Prácticos"),
    ("📋", "Tablas 2x2 y Cálculos"),
    ("📊", "Visualización de Datos"),
    ("📺", "Multimedia YouTube"),
    ("🎮", "Gamificación"),
    ("💬", "Chat Epidemiológico"),
]

# Estado inicial
if "seccion" not in st.session_state:
    st.session_state.seccion = None  # Aquí estará la sección seleccionada o None para dropdown inicial

# --- FUNCIONES ---

def mostrar_footer():
    st.markdown("""
    <style>
    .footer {
        position: fixed;
        left: 0; bottom: 0; width: 100%;
        background-color: #0b2e58;
        color: white; text-align: center;
        padding: 0.5rem 1rem;
        font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
        font-size: 0.9rem; z-index: 9999;
    }
    .footer a {
        color: #a6c8ff; text-decoration: none; margin-left: 0.6rem;
    }
    .footer a:hover {
        text-decoration: underline;
    }
    </style>
    <div class="footer">
        Creado por <b>Yolanda Muvdi</b>, Enfermera MSc Epidemiología |
        <a href="mailto:ymuvdi@gmail.com">ymuvdi@gmail.com</a> |
        <a href="https://www.linkedin.com/in/yolanda-paola-muvdi-muvdi-778b73152/" target="_blank">LinkedIn</a>
    </div>
    """, unsafe_allow_html=True)

def mostrar_titulo(emoji, texto):
    st.markdown(f"""
    <h1 style='display:flex; align-items:center; gap:0.5rem; font-family: "Segoe UI", Tahoma, Geneva, Verdana, sans-serif; color:#0d3b66;'>
        <span style='font-size:2.8rem;'>{emoji}</span> {texto}
    </h1>
    """, unsafe_allow_html=True)

def mostrar_dropdown_inicio():
    st.markdown("""
    <style>
    .selectbox-container {
        max-width: 600px;
        margin: 3rem auto;
        font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
        text-align: center;
    }
    .selectbox-title {
        font-size: 3rem;
        font-weight: 800;
        color: #274c77;
        margin-bottom: 1rem;
    }
    .selectbox-subtitle {
        font-size: 1.3rem;
        color: #415a77;
        margin-bottom: 2rem;
    }
    </style>
    """, unsafe_allow_html=True)

    st.markdown('<div class="selectbox-container">', unsafe_allow_html=True)
    st.markdown('<div class="selectbox-title">🧪 Epidemiología 101</div>', unsafe_allow_html=True)
    st.markdown('<div class="selectbox-subtitle">¿Qué quieres aprender hoy? Selecciona una sección:</div>', unsafe_allow_html=True)

    opciones = [f"{icon}  {texto}" for icon, texto in SECCIONES]
    seleccion = st.selectbox("", [""] + opciones, key="dropdown_inicio")

    st.markdown('</div>', unsafe_allow_html=True)

    # Actualizar sección si seleccionó
    if seleccion and seleccion != "":
        # Sacar solo el texto sin emoji para simplificar
        texto_sin_emoji = seleccion.split("  ")[1]
        st.session_state.seccion = texto_sin_emoji

def mostrar_sidebar():
    with st.sidebar:
        st.markdown("## Menú de Secciones")
        opciones = [f"{icon}  {texto}" for icon, texto in SECCIONES]
        # Obtener índice actual
        try:
            index = [texto for _, texto in SECCIONES].index(st.session_state.seccion)
        except ValueError:
            index = 0
        seleccion = st.radio("Navega:", opciones, index=index)
        texto_sin_emoji = seleccion.split("  ")[1]

        if texto_sin_emoji != st.session_state.seccion:
            st.session_state.seccion = texto_sin_emoji
            # No usar rerun para evitar loops molestos

        st.markdown("---")
        st.markdown("""
        <small style="color:#0d3b66;">
        Creado por <b>Yolanda Muvdi</b><br>
        ymuvdi&#64;gmail.com<br>
        <a href="https://www.linkedin.com/in/yolanda-paola-muvdi-muvdi-778b73152/" target="_blank" style="color:#0d3b66;">LinkedIn</a>
        </small>
        """, unsafe_allow_html=True)

# --- Secciones básicas (ejemplo con título y texto para que armes contenido) ---
def mostrar_inicio():
    mostrar_titulo("🧪", "Inicio")
    st.markdown("""
    ¡Bienvenido/a a Epidemiología 101!  

    Esta app es tu compañera ideal para dominar conceptos, practicar ejercicios, aprender con gamificación y analizar datos epidemiológicos.

    Usa el menú lateral para navegar las secciones.

    ¡A aprender y disfrutar! 🩺📊
    """)

def mostrar_conceptos_basicos():
    mostrar_titulo("📌", "Conceptos Básicos")
    st.write("Aquí va el contenido de conceptos básicos...")

def mostrar_medidas_asociacion():
    mostrar_titulo("📈", "Medidas de Asociación")
    st.write("Contenido para medidas de asociación...")

def mostrar_disenos_estudio():
    mostrar_titulo("📊", "Diseños de Estudio")
    st.write("Contenido para diseños de estudio...")

def mostrar_sesgos_errores():
    mostrar_titulo("⚠️", "Sesgos y Errores")
    st.write("Contenido para sesgos y errores...")

def mostrar_glosario_interactivo():
    mostrar_titulo("📚", "Glosario Interactivo")
    st.write("Contenido del glosario interactivo...")

def mostrar_ejercicios_practicos():
    mostrar_titulo("🧪", "Ejercicios Prácticos")
    st.write("Ejercicios prácticos aquí...")

def mostrar_tablas_2x2():
    mostrar_titulo("📋", "Tablas 2x2 y Cálculos")
    st.write("Funcionalidad de tablas 2x2 aquí...")

def mostrar_visualizacion_datos():
    mostrar_titulo("📊", "Visualización de Datos")
    st.write("Gráficos y visualización de datos aquí...")

def mostrar_multimedia_youtube():
    mostrar_titulo("📺", "Multimedia YouTube")
    st.write("Videos integrados aquí...")

def mostrar_gamificacion():
    mostrar_titulo("🎮", "Gamificación")
    st.write("Gamificación con preguntas, niveles y confeti aquí...")

def mostrar_chat_epidemiologico():
    mostrar_titulo("💬", "Chat Epidemiológico")
    st.write("Chat con IA epidemiológica próximamente...")

# --- MAIN ---
def main():
    st.set_page_config(page_title="Epidemiología 101", page_icon="🧪", layout="wide")

    if st.session_state.seccion is None:
        mostrar_dropdown_inicio()
    else:
        mostrar_sidebar()
        st.write("")  # espacio arriba

        # Muestra contenido según sección
        sec = st.session_state.seccion
        if sec == "Inicio":
            mostrar_inicio()
        elif sec == "Conceptos Básicos":
            mostrar_conceptos_basicos()
        elif sec == "Medidas de Asociación":
            mostrar_medidas_asociacion()
        elif sec == "Diseños de Estudio":
            mostrar_disenos_estudio()
        elif sec == "Sesgos y Errores":
            mostrar_sesgos_errores()
        elif sec == "Glosario Interactivo":
            mostrar_glosario_interactivo()
        elif sec == "Ejercicios Prácticos":
            mostrar_ejercicios_practicos()
        elif sec == "Tablas 2x2 y Cálculos":
            mostrar_tablas_2x2()
        elif sec == "Visualización de Datos":
            mostrar_visualizacion_datos()
        elif sec == "Multimedia YouTube":
            mostrar_multimedia_youtube()
        elif sec == "Gamificación":
            mostrar_gamificacion()
        elif sec == "Chat Epidemiológico":
            mostrar_chat_epidemiologico()
        else:
            st.error("Sección no encontrada.")

    mostrar_footer()

if __name__ == "__main__":
    main()
