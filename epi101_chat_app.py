import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import chi2_contingency, fisher_exact, norm
import random
from streamlit_extras.let_it_rain import rain

# --- CONSTANTES ---
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

# --- ESTADO INICIAL ---
if "seccion" not in st.session_state:
    st.session_state.seccion = "Inicio"

# Para gamificación:
if "puntaje" not in st.session_state:
    st.session_state.puntaje = 0
if "nivel" not in st.session_state:
    st.session_state.nivel = 1
if "pregunta_idx" not in st.session_state:
    st.session_state.pregunta_idx = 0

# --- FUNCIONES UTILES ---
def mostrar_footer():
    st.markdown("""
    <style>
    .footer {
        position: fixed;
        left: 0; bottom: 0;
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

def mostrar_titulo(emoji, texto):
    st.markdown(f"""
    <h1 style='display:flex; align-items:center; gap:0.5rem; font-family: "Segoe UI", Tahoma, Geneva, Verdana, sans-serif; color:#0d3b66;'>
        <span style='font-size:2.8rem;'>{emoji}</span> {texto}
    </h1>
    """, unsafe_allow_html=True)

# --- SECCIONES ---

def mostrar_inicio():
    mostrar_titulo("🧪", "Inicio")
    st.markdown("""
    ### Bienvenido/a a Epidemiología 101

    Esta app es tu compañera ideal para dominar conceptos, practicar ejercicios, aprender con gamificación y analizar datos epidemiológicos.

    Usa el menú lateral o el dropdown para navegar.

    ¡A aprender y disfrutar! 🩺📊
    """)

def mostrar_conceptos_basicos():
    mostrar_titulo("📌", "Conceptos Básicos")
    conceptos = {
        "Incidencia": "Número de casos nuevos de una enfermedad en un periodo determinado.",
        "Prevalencia": "Número total de casos de una enfermedad en un momento o periodo específico.",
        "Riesgo Relativo": "Medida que compara la incidencia entre expuestos y no expuestos.",
        "Odds Ratio": "Medida que compara las probabilidades entre grupos.",
        "Sesgo": "Error sistemático que afecta la validez de un estudio."
    }
    st.write("Lista de conceptos clave:")
    for k,v in conceptos.items():
        st.markdown(f"**{k}:** {v}")

def mostrar_medidas_asociacion():
    mostrar_titulo("📈", "Medidas de Asociación")
    st.markdown("""
    Las medidas más comunes son:

    - **Riesgo Relativo (RR):** Incidencia en expuestos / incidencia en no expuestos.
    - **Odds Ratio (OR):** Odds de enfermedad en expuestos / odds en no expuestos.
    - **Diferencia de Riesgo:** Incidencia en expuestos - incidencia en no expuestos.
    """)

def mostrar_disenos_estudio():
    mostrar_titulo("📊", "Diseños de Estudio")
    st.markdown("""
    - Estudio de cohorte
    - Estudio de casos y controles
    - Estudio transversal
    - Ensayo clínico
    """)

def mostrar_sesgos_errores():
    mostrar_titulo("⚠️", "Sesgos y Errores")
    st.markdown("""
    - Sesgo de selección
    - Sesgo de información
    - Confusión
    - Error aleatorio
    """)

def mostrar_glosario_interactivo():
    mostrar_titulo("📚", "Glosario Interactivo")
    glosario = {
        "Incidencia": "Número de casos nuevos durante un periodo.",
        "Prevalencia": "Número total de casos en un punto o periodo.",
        "Sensibilidad": "Probabilidad de detectar un caso verdadero.",
        "Especificidad": "Probabilidad de detectar un no caso.",
        "Confusión": "Variables que distorsionan asociación."
    }
    busqueda = st.text_input("Buscar término:")
    if busqueda:
        resultado = {k:v for k,v in glosario.items() if busqueda.lower() in k.lower()}
        if resultado:
            for k,v in resultado.items():
                st.markdown(f"**{k}:** {v}")
        else:
            st.write("No se encontró el término.")
    else:
        st.write("Ingresa un término arriba para buscar.")

def mostrar_ejercicios_practicos():
    mostrar_titulo("🧪", "Ejercicios Prácticos")
    st.markdown("""
    Ejercicio: Calcula la incidencia si 50 casos nuevos ocurrieron en una población de 2000 durante un año.
    """)

def mostrar_tablas_2x2():
    mostrar_titulo("📋", "Tablas 2x2 y Cálculos")
    st.markdown("Ingresa los datos de la tabla 2x2:")

    a = st.number_input("a (casos expuestos)", min_value=0, value=10, step=1)
    b = st.number_input("b (no casos expuestos)", min_value=0, value=20, step=1)
    c = st.number_input("c (casos no expuestos)", min_value=0, value=5, step=1)
    d = st.number_input("d (no casos no expuestos)", min_value=0, value=25, step=1)

    st.markdown(f"""
    |        | Enfermos | No enfermos |
    |--------|----------|-------------|
    | Expuestos | {a}      | {b}         |
    | No expuestos | {c}      | {d}         |
    """)

    # Calculos
    try:
        rr = (a / (a + b)) / (c / (c + d)) if (a + b) > 0 and (c + d) > 0 else np.nan
        or_ = (a * d) / (b * c) if b * c > 0 else np.nan

        # IC 95% para OR (log)
        se_log_or = np.sqrt(1/a + 1/b + 1/c + 1/d) if all(x>0 for x in [a,b,c,d]) else np.nan
        log_or = np.log(or_) if or_ > 0 else np.nan
        ci_lower_or = np.exp(log_or - 1.96 * se_log_or) if not np.isnan(log_or) else np.nan
        ci_upper_or = np.exp(log_or + 1.96 * se_log_or) if not np.isnan(log_or) else np.nan

        # Test Chi2 o Fisher
        tabla = np.array([[a,b],[c,d]])
        if SCIPY_AVAILABLE:
            chi2, p_val, _, _ = chi2_contingency(tabla)
            _, p_fisher = fisher_exact(tabla)
        else:
            p_val = p_fisher = np.nan

        st.markdown(f"""
        **Riesgo Relativo (RR):** {rr:.3f}  
        **Odds Ratio (OR):** {or_:.3f}  
        **IC 95% OR:** [{ci_lower_or:.3f}, {ci_upper_or:.3f}]  
        **p-valor Chi2:** {p_val:.4f}  
        **p-valor Fisher:** {p_fisher:.4f}  
        """)

    except Exception as e:
        st.error(f"Error en cálculos: {e}")

def mostrar_visualizacion_datos():
    mostrar_titulo("📊", "Visualización de Datos")
    st.markdown("Genera un gráfico de barras simple:")

    categorias = st.text_area("Categorías separadas por coma", "A,B,C,D")
    valores = st.text_area("Valores separados por coma", "10,20,15,5")

    if st.button("Generar gráfico"):
        try:
            cats = [x.strip() for x in categorias.split(",")]
            vals = [float(x) for x in valores.split(",")]
            if len(cats) != len(vals):
                st.error("Número de categorías y valores debe coincidir.")
                return
            fig, ax = plt.subplots()
            ax.bar(cats, vals, color="#274c77")
            ax.set_title("Gráfico de barras")
            st.pyplot(fig)
        except Exception as e:
            st.error(f"Error: {e}")

def mostrar_multimedia_youtube():
    mostrar_titulo("📺", "Multimedia YouTube")
    video_id = st.text_input("Ingresa el ID del video de YouTube", "dQw4w9WgXcQ")
    st.markdown(f"""
    <iframe width="560" height="315" src="https://www.youtube.com/embed/{video_id}" frameborder="0" allowfullscreen></iframe>
    """, unsafe_allow_html=True)

# --- GAMIFICACIÓN ---
PREGUNTAS = {
    1: [
        {"pregunta": "¿Qué es una tasa en epidemiología?", "opciones": ["Número sin unidades", "Número de eventos por unidad de tiempo", "Medida de asociación", "Ninguna"], "respuesta": "Número de eventos por unidad de tiempo"},
        {"pregunta": "¿Qué significa IR para un estudio?", "opciones": ["Incidencia", "Incidencia acumulada", "Riesgo relativo", "Odds ratio"], "respuesta": "Incidencia"},
    ],
    2: [
        {"pregunta": "Fórmula para Riesgo Relativo", "opciones": ["a/(a+b)", "c/(c+d)", "(a/(a+b)) / (c/(c+d))", "a/c"], "respuesta": "(a/(a+b)) / (c/(c+d))"},
        {"pregunta": "Test estadístico para tablas 2x2 con celdas pequeñas", "opciones": ["Chi-cuadrado", "T-test", "Fisher exacto", "ANOVA"], "respuesta": "Fisher exacto"},
    ],
    3: [
        {"pregunta": "¿Qué es sesgo de información?", "opciones": ["Error sistemático en medición", "Error aleatorio", "Confusión", "Sesgo de selección"], "respuesta": "Error sistemático en medición"},
        {"pregunta": "¿Qué significa un IC 95% que incluye 1 para un OR?", "opciones": ["Significativo", "No significativo", "Mayor riesgo", "Menor riesgo"], "respuesta": "No significativo"},
    ],
}

def mostrar_gamificacion():
    mostrar_titulo("🎮", "Gamificación")
    nivel = st.session_state.nivel
    idx = st.session_state.pregunta_idx
    puntaje = st.session_state.puntaje

    preguntas_nivel = PREGUNTAS[nivel]
    pregunta_actual = preguntas_nivel[idx]

    st.markdown(f"**Nivel:** {nivel} - Pregunta {idx+1} de {len(preguntas_nivel)}")
    st.write(pregunta_actual["pregunta"])

    respuesta_usuario = st.radio("Elige una respuesta:", pregunta_actual["opciones"], key=f"preg_{nivel}_{idx}")

    if st.button("Verificar respuesta"):
        if respuesta_usuario == pregunta_actual["respuesta"]:
            st.success("¡Correcto! 🎉")
            st.session_state.puntaje += 1
            rain(emoji="🎉", font_size=30, falling_speed=5, animation_length=5)
        else:
            st.error(f"Incorrecto. La respuesta correcta es: {pregunta_actual['respuesta']}")

        # Avanzar pregunta o nivel
        if idx + 1 < len(preguntas_nivel):
            st.session_state.pregunta_idx += 1
        else:
            st.session_state.pregunta_idx = 0
            if nivel < 3:
                st.session_state.nivel += 1
                st.success(f"¡Has avanzado al nivel {st.session_state.nivel}! 🎊")
            else:
                st.balloons()
                st.success("¡Felicitaciones, completaste todos los niveles! 🏆")

def mostrar_chat_epidemiologico():
    mostrar_titulo("💬", "Chat Epidemiológico")
    st.info("Esta sección está en desarrollo. Próximamente podrás interactuar con una IA especializada en epidemiología.")

# --- BARRA LATERAL ---
def mostrar_sidebar():
    with st.sidebar:
        st.markdown("## Menú de Secciones")
        opciones = [f"{icon}  {texto}" for icon, texto in SECCIONES]
        seleccionado = st.radio("Navega:", opciones, index=[t[1] for t in SECCIONES].index(st.session_state.seccion))
        # Extraer texto sin emoji
        texto_sin_emoji = seleccionado.split("  ")[1]
        if texto_sin_emoji != st.session_state.seccion:
            st.session_state.seccion = texto_sin_emoji
            st.experimental_rerun()

        st.markdown("---")
        st.markdown("""
        <small style="color:#0d3b66;">
        Creado por <b>Yolanda Muvdi</b><br>
        ymuvdi&#64;gmail.com<br>
        <a href="https://www.linkedin.com/in/yolanda-paola-muvdi-muvdi-778b73152/" target="_blank" style="color:#0d3b66;">LinkedIn</a>
        </small>
        """, unsafe_allow_html=True)

# --- MAIN ---
def main():
    st.set_page_config(page_title="Epidemiología 101", page_icon="🧪", layout="wide")

    mostrar_sidebar()

    st.write("")  # espacio

    seccion = st.session_state.seccion

    if seccion == "Inicio":
        mostrar_inicio()
    elif seccion == "Conceptos Básicos":
        mostrar_conceptos_basicos()
    elif seccion == "Medidas de Asociación":
        mostrar_medidas_asociacion()
    elif seccion == "Diseños de Estudio":
        mostrar_disenos_estudio()
    elif seccion == "Sesgos y Errores":
        mostrar_sesgos_errores()
    elif seccion == "Glosario Interactivo":
        mostrar_glosario_interactivo()
    elif seccion == "Ejercicios Prácticos":
        mostrar_ejercicios_practicos()
    elif seccion == "Tablas 2x2 y Cálculos":
        mostrar_tablas_2x2()
    elif seccion == "Visualización de Datos":
        mostrar_visualizacion_datos()
    elif seccion == "Multimedia YouTube":
        mostrar_multimedia_youtube()
    elif seccion == "Gamificación":
        mostrar_gamificacion()
    elif seccion == "Chat Epidemiológico":
        mostrar_chat_epidemiologico()

    mostrar_footer()

if __name__ == "__main__":
    main()
