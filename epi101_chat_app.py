import streamlit as st
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import math
import os

# Confetti library para animación visual (streamlit-extras)
try:
    from streamlit_extras.let_it_rain import rain
    CONFETTI_AVAILABLE = True
except ImportError:
    CONFETTI_AVAILABLE = False

# SciPy para tests estadísticos
try:
    from scipy.stats import chi2_contingency, fisher_exact, norm
    SCIPY_AVAILABLE = True
except Exception:
    SCIPY_AVAILABLE = False

# -------------------- CONFIGURACIÓN GENERAL --------------------
st.set_page_config(
    page_title="🧠 Epidemiología 101 - Masterclass",
    page_icon="🧪",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Estilos CSS limpios y accesibles
st.markdown("""
<style>
    body, .block-container {
        background: #fefefe;
        color: #0d3b66;
        font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
        line-height: 1.5;
    }
    .block-container {
        max-width: 900px;
        margin: 2rem auto 4rem auto;
        padding: 2rem 2rem;
        background: white;
        border-radius: 12px;
        box-shadow: 0 12px 30px rgba(13,59,102,0.1);
    }
    h1, h2, h3 {
        color: #0d3b66;
        font-weight: 700;
    }
    a {
        color: #0d3b66;
        text-decoration: none;
        font-weight: 600;
    }
    a:hover {
        text-decoration: underline;
    }
    footer {
        text-align: center;
        padding: 1rem;
        font-size: 0.9rem;
        color: #0d3b66;
        background-color: #f0f5fa;
        border-top: 1px solid #ddd;
        position: fixed;
        bottom: 0;
        width: 100%;
    }
    /* Botones grandes para móvil */
    @media (max-width: 768px) {
        button {
            width: 100% !important;
            font-size: 1.2rem !important;
        }
    }
</style>
""", unsafe_allow_html=True)

# -------------------- INFORMACIÓN FIJA --------------------
CREADOR = "Yolanda Muvdi"
TITULO = "Enfermera con MSc en Epidemiología"
EMAIL = "ymuvdiarrobagmail.com"
LINKEDIN = "https://www.linkedin.com/in/yolanda-paola-muvdi-muvdi-778b73152/"

# -------------------- CONTENIDO --------------------

secciones = [
    "Inicio",
    "Conceptos Básicos",
    "Medidas de Asociación",
    "Diseños de Estudio",
    "Sesgos y Errores",
    "Glosario Interactivo",
    "Ejercicios Prácticos",
    "Tablas 2x2 y Cálculos",
    "Visualización de Datos",
    "Multimedia YouTube",
    "Gamificación",
    "Chat Epidemiológico"
]

# -- Funciones auxiliares (simplificadas para foco en gamificación) --

def mostrar_videos():
    videos = {
        "Introducción a Epidemiología": "https://www.youtube.com/watch?v=Z4JDr11G0N4",
        "Medidas de Asociación": "https://www.youtube.com/watch?v=pD9Oa88kqQI",
        "Diseños de Estudios Epidemiológicos": "https://www.youtube.com/watch?v=sP4kzERW6wo",
        "Sesgos y Errores Comunes": "https://www.youtube.com/watch?v=RrKhv8OdLmY"
    }
    st.markdown("### 📺 Videos recomendados")
    for nombre, url in videos.items():
        st.markdown(f"▶️ [{nombre}]({url})")

def mostrar_gamificacion():
    st.header("🎮 Gamificación - Elige tu nivel epidemiológico")

    niveles = {
        "Básico": [
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
        "Intermedio": [
            {
                "pregunta": "¿Cuál es la fórmula para Riesgo Relativo?",
                "opciones": ["a/(a+b)", "c/(c+d)", "(a/(a+b)) / (c/(c+d))", "a/c"],
                "respuesta_correcta": "(a/(a+b)) / (c/(c+d))"
            },
            {
                "pregunta": "¿Qué test estadístico se usa para tablas 2x2 con celdas pequeñas?",
                "opciones": ["Chi-cuadrado", "Fisher exacto", "T de Student", "ANOVA"],
                "respuesta_correcta": "Fisher exacto"
            }
        ],
        "Avanzado": [
            {
                "pregunta": "¿Cuál es la interpretación correcta de un OR > 1?",
                "opciones": [
                    "Asociación negativa",
                    "Asociación positiva",
                    "No hay asociación",
                    "Error en el estudio"
                ],
                "respuesta_correcta": "Asociación positiva"
            },
            {
                "pregunta": "¿Qué asunción es clave para el uso del Riesgo Relativo?",
                "opciones": [
                    "Diseño transversal",
                    "Estudio de cohorte",
                    "Estudio de casos y controles",
                    "Estudio ecológico"
                ],
                "respuesta_correcta": "Estudio de cohorte"
            }
        ]
    }

    # Inicializar puntaje en sesión (por nivel)
    if "puntajes" not in st.session_state:
        st.session_state.puntajes = {"Básico":0, "Intermedio":0, "Avanzado":0}
    if "respuestas_hechas" not in st.session_state:
        st.session_state.respuestas_hechas = {"Básico":[], "Intermedio":[], "Avanzado":[]}

    nivel = st.selectbox("¿En qué nivel crees estar?", list(niveles.keys()))

    preguntas = niveles[nivel]

    st.markdown(f"### Nivel seleccionado: {nivel}")

    preguntas_hechas = st.session_state.respuestas_hechas[nivel]

    for i, pregunta in enumerate(preguntas):
        if i not in preguntas_hechas:
            st.subheader(f"Pregunta {i+1}")
            seleccion = st.radio(pregunta["pregunta"], pregunta["opciones"], key=f"{nivel}_preg_{i}")
            if st.button(f"Verificar respuesta {i+1}", key=f"{nivel}_btn_{i}"):
                if seleccion == pregunta["respuesta_correcta"]:
                    st.success("✅ Correcto!")
                    st.session_state.puntajes[nivel] += 1
                else:
                    st.error(f"❌ Incorrecto. Respuesta correcta: {pregunta['respuesta_correcta']}")
                preguntas_hechas.append(i)
                st.experimental_rerun()
        else:
            st.markdown(f"Pregunta {i+1} respondida.")

    puntaje = st.session_state.puntajes[nivel]
    total = len(preguntas)

    st.markdown(f"### Puntaje en nivel {nivel}: {puntaje} / {total}")

    if puntaje == total and total > 0:
        st.success(f"🎉 ¡Excelente, eres un especialista en Epidemiología nivel {nivel}!")
        if CONFETTI_AVAILABLE:
            rain(emoji="🎉", font_size=40, falling_speed=5, animation_length=3)
        else:
            st.balloons()
        st.markdown("""
        **Podemos mejorar siempre, sigue explorando otras secciones y juega de nuevo para perfeccionar tus conocimientos!**
        """)

# -------------------- APP PRINCIPAL --------------------

def main():
    st.title("🧪 Bienvenido/a a Epidemiología 101")
    st.markdown(f"""
    Creado por **{CREADOR}**, {TITULO}  
    📧 {EMAIL}  
    🔗 [LinkedIn]({LINKEDIN})
    """)
    st.markdown("---")

    # Dropdown para seleccionar sección
    seccion = st.selectbox("Selecciona una sección para comenzar:", secciones)

    st.markdown("---")

    if seccion == "Inicio":
        st.markdown("""
        ¡Hola! Este es un espacio interactivo para aprender epidemiología de forma práctica y entretenida.  
        Usa el menú desplegable para explorar conceptos, ejercicios, tablas 2x2, visualización y mucho más.
        """)

    elif seccion == "Conceptos Básicos":
        st.header("📌 Conceptos Básicos")
        st.info("Carga aquí el contenido de conceptos básicos, por ejemplo desde un archivo markdown.")

    elif seccion == "Medidas de Asociación":
        st.header("📈 Medidas de Asociación")
        st.info("Carga aquí el contenido de medidas de asociación.")

    elif seccion == "Diseños de Estudio":
        st.header("📊 Diseños de Estudio")
        st.info("Carga aquí el contenido de diseños de estudio.")

    elif seccion == "Sesgos y Errores":
        st.header("⚠️ Sesgos y Errores")
        st.info("Carga aquí el contenido de sesgos y errores comunes.")

    elif seccion == "Glosario Interactivo":
        st.header("📚 Glosario Interactivo")
        st.info("Carga aquí tu glosario interactivo.")

    elif seccion == "Ejercicios Prácticos":
        st.header("🧪 Ejercicios Prácticos")
        st.info("Carga aquí ejercicios prácticos con verificación.")

    elif seccion == "Tablas 2x2 y Cálculos":
        st.header("📊 Tablas 2x2 y Cálculos Epidemiológicos")
        st.info("Aquí puedes implementar la calculadora de tablas 2x2 con RR, OR, RD.")

    elif seccion == "Visualización de Datos":
        st.header("📊 Visualización de Datos")
        st.info("Carga aquí datos y genera gráficos exploratorios.")

    elif seccion == "Multimedia YouTube":
        mostrar_videos()

    elif seccion == "Gamificación":
        mostrar_gamificacion()

    elif seccion == "Chat Epidemiológico":
        st.header("💬 Chat Epidemiológico")
        st.info("Implementa aquí el chat con Gemini o IA para responder preguntas.")

    # Footer fijo
    st.markdown(f"""
    <footer>
        Creado por {CREADOR} | {TITULO} | 📧 {EMAIL} | 🔗 <a href="{LINKEDIN}" target="_blank">LinkedIn</a>
    </footer>
    """, unsafe_allow_html=True)


if __name__ == "__main__":
    main()





