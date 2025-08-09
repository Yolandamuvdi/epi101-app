import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import math
import random

# Para confeti
from streamlit_extras.let_it_rain import rain

# Google Gemini AI (opcional)
try:
    import google.generativeai as genai
    GENAI_AVAILABLE = True
except Exception:
    GENAI_AVAILABLE = False

# SciPy para tests estadísticos
try:
    from scipy.stats import chi2_contingency, fisher_exact, norm
    SCIPY_AVAILABLE = True
except Exception:
    SCIPY_AVAILABLE = False

# Secciones con emojis
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
    st.session_state.seccion = None
if "puntaje_gamificacion" not in st.session_state:
    st.session_state.puntaje_gamificacion = 0

# -------- FUNCIONES --------

def mostrar_footer():
    st.markdown("""
    <style>
    .footer {
        position: fixed;
        left: 0;
        bottom: 0;
        width: 100%;
        background-color: #0b2e58;
        color: white;
        text-align: center;
        padding: 0.5rem 1rem;
        font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
        font-size: 0.9rem;
        z-index: 9999;
    }
    .footer a {
        color: #a6c8ff;
        text-decoration: none;
        margin-left: 0.6rem;
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

def mostrar_splash():
    st.markdown("""
    <style>
    .container {
        max-width: 540px;
        margin: 4rem auto 3rem auto;
        font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
        text-align: center;
        color: #1b263b;
        user-select: none;
    }
    .title {
        font-size: 3.8rem;
        font-weight: 900;
        margin-bottom: 0.3rem;
        display: flex;
        justify-content: center;
        align-items: center;
        gap: 1rem;
        color: #274c77;
    }
    .title .icon {
        font-size: 4.8rem;
        animation: pulse 2.5s infinite ease-in-out;
        color: #0d3b66;
    }
    @keyframes pulse {
        0%, 100% { transform: scale(1); }
        50% { transform: scale(1.12); }
    }
    .question {
        font-size: 1.9rem;
        font-weight: 600;
        margin-bottom: 2rem;
        color: #415a77;
        letter-spacing: 0.6px;
    }
    .custom-selectbox select {
        width: 100%;
        padding: 1.1rem 1.7rem;
        border-radius: 14px;
        border: 2.5px solid #274c77;
        font-size: 1.3rem;
        font-weight: 700;
        color: #1b263b;
        background-color: #f7f9fc;
        cursor: pointer;
        appearance: none;
        -webkit-appearance: none;
        -moz-appearance: none;
        box-shadow: 0 6px 12px rgba(0,0,0,0.07);
        background-image: url("data:image/svg+xml;utf8,<svg fill='%23274c77' height='28' viewBox='0 0 24 24' width='28' xmlns='http://www.w3.org/2000/svg'><path d='M7 10l5 5 5-5z'/></svg>");
        background-repeat: no-repeat;
        background-position: right 1.3rem center;
        background-size: 1.2rem;
        transition: border-color 0.3s ease;
    }
    .custom-selectbox select:hover {
        border-color: #274c77cc;
    }
    .custom-selectbox select:focus {
        outline: none;
        border-color: #0d3b66;
        box-shadow: 0 0 16px #0d3b66aa;
    }
    </style>
    """, unsafe_allow_html=True)

    st.markdown('<div class="container">', unsafe_allow_html=True)
    st.markdown('<div class="title"><span class="icon">🧪</span> Epidemiología 101</div>', unsafe_allow_html=True)
    st.markdown('<div class="question">¿Qué quieres aprender hoy?</div>', unsafe_allow_html=True)

    opciones = [f"{icon}  {texto}" for icon, texto in SECCIONES]
    opcion = st.selectbox("", [""] + opciones, key="splash_select", label_visibility="collapsed")

    st.markdown('</div>', unsafe_allow_html=True)

    if opcion and opcion != "":
        texto_sin_emoji = opcion.split(" ", 1)[1] if " " in opcion else opcion
        st.session_state.seccion = texto_sin_emoji
        st.experimental_rerun()  # SOLO aquí, no más rerun

def mostrar_titulo_con_emoji(seccion_texto):
    emoji = next((icon for icon, texto in SECCIONES if texto == seccion_texto), "")
    st.markdown(f"<h1 style='display:flex; align-items:center; gap:0.6rem; font-family:Segoe UI, Tahoma, Geneva, Verdana, sans-serif; color:#0d3b66;'>"
                f"<span style='font-size:3rem;'>{emoji}</span>{seccion_texto}</h1>", unsafe_allow_html=True)

def mostrar_inicio():
    mostrar_titulo_con_emoji("Inicio")
    st.markdown("""
    ### ¡Bienvenido/a a Epidemiología 101!

    Aquí tienes un espacio limpio, profesional y funcional para explorar todo lo relacionado con Epidemiología, con un toque de alegría y emojis para hacerlo más amigable.

    - Usa el menú lateral o el dropdown para navegar entre las secciones.
    - Disfruta la gamificación con niveles y premios 🎉.
    - Todo lo que necesitas, al alcance de un clic.

    ---
    """)
    st.image("https://images.unsplash.com/photo-1581092334311-eaff36f51c3a?auto=format&fit=crop&w=800&q=80", caption="Epidemiología: Ciencia que salva vidas", use_column_width=True)
    st.write("")
    st.markdown("Si tienes dudas o sugerencias, no dudes en escribirme al correo en el footer. ¡Gracias por usar esta app!")

def mostrar_gamificacion():
    mostrar_titulo_con_emoji("Gamificación")

    niveles = {
        1: [  # Básico
            {
                "pregunta": "¿Qué es una tasa en epidemiología?",
                "opciones": ["Un número sin unidades", "Número de eventos por unidad de tiempo", "Una medida de asociación", "Ninguna"],
                "respuesta_correcta": "Número de eventos por unidad de tiempo"
            },
            {
                "pregunta": "¿Qué significa IR para un estudio?",
                "opciones": ["Incidencia", "Incidencia acumulada", "Riesgo relativo", "Odds ratio"],
                "respuesta_correcta": "Incidencia"
            }
        ],
        2: [  # Intermedio
            {
                "pregunta": "¿Cuál es la fórmula para Riesgo Relativo?",
                "opciones": ["a/(a+b)", "c/(c+d)", "(a/(a+b)) / (c/(c+d))", "a/c"],
                "respuesta_correcta": "(a/(a+b)) / (c/(c+d))"
            },
            {
                "pregunta": "¿Qué test estadístico se usa para tablas 2x2 con celdas pequeñas?",
                "opciones": ["Chi-cuadrado", "T-test", "Fisher exacto", "ANOVA"],
                "respuesta_correcta": "Fisher exacto"
            }
        ],
        3: [  # Avanzado
            {
                "pregunta": "¿Qué es el sesgo de información?",
                "opciones": ["Error sistemático en medición", "Error aleatorio", "Confusión", "Sesgo de selección"],
                "respuesta_correcta": "Error sistemático en medición"
            },
            {
                "pregunta": "¿Qué significa un IC 95% que incluye 1 para un OR?",
                "opciones": ["Significativo", "No significativo", "Mayor riesgo", "Menor riesgo"],
                "respuesta_correcta": "No significativo"
            }
        ]
    }

    puntaje = st.session_state.get("puntaje_gamificacion", 0)
    nivel = 1
    if puntaje >= 4:
        nivel = 3
    elif puntaje >= 2:
        nivel = 2

    preguntas = niveles[nivel]

    st.markdown(f"### Nivel actual: {'Básico' if nivel==1 else 'Intermedio' if nivel==2 else 'Avanzado'}")

    for i, q in enumerate(preguntas):
        st.write(f"**Pregunta {i+1}:** {q['pregunta']}")
        key_radio = f"gam_{nivel}_{i}"
        respuesta = st.radio("Selecciona una opción:", q['opciones'], key=key_radio)
        btn_key = f"btn_gam_{nivel}_{i}"
        if st.button("Verificar respuesta", key=btn_key):
            if respuesta == q['respuesta_correcta']:
                st.success("¡Correcto! 🎉")
                st.session_state.puntaje_gamificacion += 1
                rain(
                    emoji="🎉",
                    font_size=30,
                    falling_speed=5,
                    animation_length=5,
                )
            else:
                st.error(f"Incorrecto. La respuesta correcta es: {q['respuesta_correcta']}")

    st.markdown(f"**Puntaje total:** {st.session_state.puntaje_gamificacion}")

# Placeholder para otras secciones
def mostrar_conceptos_basicos():
    mostrar_titulo_con_emoji("Conceptos Básicos")
    st.write("Contenido de conceptos básicos aquí...")

def mostrar_medidas_asociacion():
    mostrar_titulo_con_emoji("Medidas de Asociación")
    st.write("Contenido de medidas de asociación aquí...")

def mostrar_disenos_estudio():
    mostrar_titulo_con_emoji("Diseños de Estudio")
    st.write("Contenido de diseños de estudio aquí...")

def mostrar_sesgos_errores():
    mostrar_titulo_con_emoji("Sesgos y Errores")
    st.write("Contenido de sesgos y errores aquí...")

def mostrar_glosario_interactivo():
    mostrar_titulo_con_emoji("Glosario Interactivo")
    st.write("Contenido del glosario interactivo aquí...")

def mostrar_ejercicios_practicos():
    mostrar_titulo_con_emoji("Ejercicios Prácticos")
    st.write("Contenido de ejercicios prácticos aquí...")

def mostrar_tablas_2x2():
    mostrar_titulo_con_emoji("Tablas 2x2 y Cálculos")
    st.write("Contenido de tablas 2x2 y cálculos epidemiológicos aquí...")

def mostrar_visualizacion_datos():
    mostrar_titulo_con_emoji("Visualización de Datos")
    st.write("Contenido de visualización de datos aquí...")

def mostrar_multimedia_youtube():
    mostrar_titulo_con_emoji("Multimedia YouTube")
    st.write("Contenido multimedia con videos aquí...")

def mostrar_chat_epidemiologico():
    mostrar_titulo_con_emoji("Chat Epidemiológico")
    st.write("Chat con IA aquí...")

# Sidebar con emojis
def mostrar_sidebar():
    with st.sidebar:
        st.markdown("## Menú de Secciones")
        seccion_actual = st.session_state.get("seccion", None)
        # El radio muestra icono + texto
        opciones_radio = [f"{icon}  {texto}" for icon, texto in SECCIONES]
        index = 0
        if seccion_actual:
            for i, (_, texto) in enumerate(SECCIONES):
                if texto == seccion_actual:
                    index = i
                    break

        seleccion = st.radio("Navega por la app:", opciones_radio, index=index)
        texto_seleccion = seleccion.split(" ", 1)[1]
        if texto_seleccion != seccion_actual:
            st.session_state.seccion = texto_seleccion
            st.experimental_rerun()

        st.markdown("---")
        st.markdown("""
        <small style="color:#0d3b66;">
        Creado por <b>Yolanda Muvdi</b><br>
        ymuvdi&#64;gmail.com<br>
        <a href="https://www.linkedin.com/in/yolanda-paola-muvdi-muvdi-778b73152/" target="_blank" style="color:#0d3b66;">LinkedIn</a>
        </small>
        """, unsafe_allow_html=True)

def main():
    st.set_page_config(
        page_title="Epidemiología 101",
        page_icon="🧪",
        layout="wide",
        initial_sidebar_state="collapsed",
    )

    if st.session_state.seccion is None:
        mostrar_splash()
    else:
        mostrar_sidebar()
        st.write("")  # pequeño espacio arriba

        if st.session_state.seccion == "Inicio":
            mostrar_inicio()
        elif st.session_state.seccion == "Conceptos Básicos":
            mostrar_conceptos_basicos()
        elif st.session_state.seccion == "Medidas de Asociación":
            mostrar_medidas_asociacion()
        elif st.session_state.seccion == "Diseños de Estudio":
            mostrar_disenos_estudio()
        elif st.session_state.seccion == "Sesgos y Errores":
            mostrar_sesgos_errores()
        elif st.session_state.seccion == "Glosario Interactivo":
            mostrar_glosario_interactivo()
        elif st.session_state.seccion == "Ejercicios Prácticos":
            mostrar_ejercicios_practicos()
        elif st.session_state.seccion == "Tablas 2x2 y Cálculos":
            mostrar_tablas_2x2()
        elif st.session_state.seccion == "Visualización de Datos":
            mostrar_visualizacion_datos()
        elif st.session_state.seccion == "Multimedia YouTube":
            mostrar_multimedia_youtube()
        elif st.session_state.seccion == "Gamificación":
            mostrar_gamificacion()
        elif st.session_state.seccion == "Chat Epidemiológico":
            mostrar_chat_epidemiologico()

    mostrar_footer()

if __name__ == "__main__":
    main()


