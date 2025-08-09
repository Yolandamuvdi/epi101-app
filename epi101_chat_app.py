import streamlit as st
import pandas as pd
import numpy as np
import math
import matplotlib.pyplot as plt

# Intento integrar Gemini (Google Generative AI) si disponible
try:
    import google.generativeai as genai
    GENAI_AVAILABLE = True
    genai.configure(api_key="TU_API_KEY_AQUI")  # Cambia esto por tu API Key real
except Exception:
    GENAI_AVAILABLE = False

# --- Datos para pestañas ---
glosario = {
    "Años de vida perdidos (AVP)": "Medida del impacto de una enfermedad al estimar los años no vividos debido a una muerte prematura.",
    "Azar": "Factor impredecible que puede afectar los resultados de un estudio; se controla con tamaño de muestra y aleatorización.",
    "Bias (Sesgo)": "Error sistemático en el diseño o ejecución del estudio que afecta la validez de los resultados.",
    "Zoonosis": "Enfermedades que se transmiten de animales a humanos."
    # Agrega todos los términos que quieras
}

conceptos_basicos = """
# Conceptos Básicos de Epidemiología

- Incidencia: Número de casos nuevos de una enfermedad en un período determinado.
- Prevalencia: Proporción de individuos que tienen la enfermedad en un momento dado.
- Riesgo Relativo (RR): Razón del riesgo en expuestos vs no expuestos.
- Odds Ratio (OR): Razón de odds en casos vs controles.
"""

sesgos_texto = """
# Sesgos en Epidemiología

- Sesgo de selección: cuando la muestra no representa a la población.
- Sesgo de información: errores en medición de variables.
- Confusión: efecto de variables externas no controladas.
"""

medidas_asociacion_texto = """
# Medidas de Asociación

- Riesgo Relativo (RR)
- Odds Ratio (OR)
- Diferencia de Riesgos (RD)
- Razón de Tasas
"""

ejercicios = [
    {"pregunta": "¿Qué es la incidencia?", "opciones": ["Casos nuevos", "Casos totales", "Muertes", "Ninguna"], "respuesta": 0},
    {"pregunta": "¿Qué mide el RR?", "opciones": ["Diferencia de medias", "Riesgo relativo entre expuestos y no expuestos", "Número de casos", "Ninguna"], "respuesta": 1},
    # Agrega más preguntas hasta 44 si quieres
]

# --- Funciones Epidemiológicas ---

def aplicar_correccion_haldane_anscombe(a,b,c,d):
    if 0 in [a,b,c,d]:
        return a+0.5, b+0.5, c+0.5, d+0.5, True
    else:
        return a,b,c,d,False

def calcular_medidas(a,b,c,d):
    a_,b_,c_,d_, corregido = aplicar_correccion_haldane_anscombe(a,b,c,d)
    inc_exp = a_/(a_+b_)
    inc_noexp = c_/(c_+d_)

    rr = inc_exp/inc_noexp if inc_noexp>0 else np.nan
    se_log_rr = math.sqrt( (1/a_) - (1/(a_+b_)) + (1/c_) - (1/(c_+d_)) )
    ci_rr = (math.exp(math.log(rr)-1.96*se_log_rr), math.exp(math.log(rr)+1.96*se_log_rr)) if rr>0 else (np.nan,np.nan)

    orr = (a_*d_)/(b_*c_) if (b_*c_)>0 else np.nan
    se_log_or = math.sqrt(1/a_ + 1/b_ + 1/c_ + 1/d_)
    ci_or = (math.exp(math.log(orr)-1.96*se_log_or), math.exp(math.log(orr)+1.96*se_log_or)) if orr>0 else (np.nan,np.nan)

    rd = inc_exp - inc_noexp
    se_rd = math.sqrt( (inc_exp*(1-inc_exp)/(a_+b_)) + (inc_noexp*(1-inc_noexp)/(c_+d_)) )
    ci_rd = (rd - 1.96*se_rd, rd + 1.96*se_rd)

    rae = rd
    fae = rae/inc_exp if inc_exp>0 else np.nan
    fap = rae/inc_noexp if inc_noexp>0 else np.nan

    # p-valor simple chi2 con scipy si disponible
    p_val = np.nan
    try:
        from scipy.stats import chi2_contingency, fisher_exact
        tabla = np.array([[a,b],[c,d]])
        chi2, p_val_chi, _, _ = chi2_contingency(tabla)
        p_val = p_val_chi
        # Si tabla pequeña, usar Fisher
        if p_val>0.05 and np.min(tabla)<5:
            _, p_val_fisher = fisher_exact(tabla)
            p_val = p_val_fisher
    except:
        p_val = np.nan

    return {
        "RR": rr, "CI_RR": ci_rr,
        "OR": orr, "CI_OR": ci_or,
        "RD": rd, "CI_RD": ci_rd,
        "RAE": rae, "FAE": fae, "FAP": fap,
        "p_val": p_val,
        "corregido": corregido
    }

# --- Visualizaciones ---

def plot_forest(rr, ci_rr, orr, ci_or):
    fig, ax = plt.subplots(figsize=(6,3))
    medidas = ['Riesgo Relativo (RR)', 'Odds Ratio (OR)']
    valores = [rr, orr]
    ci_inf = [ci_rr[0], ci_or[0]]
    ci_sup = [ci_rr[1], ci_or[1]]
    y_pos = np.arange(len(medidas))

    for i, (val, inf, sup) in enumerate(zip(valores, ci_inf, ci_sup)):
        if not (np.isfinite(val) and np.isfinite(inf) and np.isfinite(sup)):
            continue
        ax.errorbar(val, y_pos[i], xerr=[[val-inf],[sup-val]], fmt='o', color='black', ecolor='blue', elinewidth=2, capsize=4)
    ax.axvline(x=1, color='grey', linestyle='--')
    ax.set_yticks(y_pos)
    ax.set_yticklabels(medidas)
    ax.set_xlabel("Medida (IC 95%)")
    ax.set_title("Forest plot RR y OR")
    ax.invert_yaxis()
    st.pyplot(fig)

def plot_barras(a,b,c,d):
    labels = ['Casos Expuestos', 'No Casos Expuestos', 'Casos No Expuestos', 'No Casos No Expuestos']
    valores = [a,b,c,d]
    colores = ['#1f77b4','#aec7e8','#ff7f0e','#ffbb78']

    fig, ax = plt.subplots()
    bars = ax.bar(labels, valores, color=colores)
    ax.set_ylabel('Número')
    ax.set_title('Distribución de casos y no casos')
    for bar in bars:
        height = bar.get_height()
        ax.annotate(f'{int(height)}', xy=(bar.get_x() + bar.get_width()/2, height),
                    xytext=(0,3), textcoords="offset points", ha='center', va='bottom')
    plt.xticks(rotation=20)
    st.pyplot(fig)

# --- App Streamlit ---

st.title("🧠 Epidemiología 101 - Asistente educativo")

# Sidebar menú
pestanas = ["Conceptos Básicos", "Glosario", "Tablas 2x2", "Medidas de Asociación", "Sesgos", "Visualización de Datos", "Ejercicios"]

opcion = st.sidebar.selectbox("Seleccione la pestaña:", pestanas)

if opcion == "Conceptos Básicos":
    st.markdown(conceptos_basicos)

elif opcion == "Glosario":
    st.header("📚 Glosario interactivo")
    for termino, definicion in glosario.items():
        with st.expander(termino):
            st.write(definicion)

elif opcion == "Tablas 2x2":
    st.header("Calculadora de Tabla 2x2 con medidas epidemiológicas")

    a = st.number_input("a: Casos expuestos", min_value=0, step=1, value=10)
    b = st.number_input("b: No casos expuestos", min_value=0, step=1, value=20)
    c = st.number_input("c: Casos no expuestos", min_value=0, step=1, value=5)
    d = st.number_input("d: No casos no expuestos", min_value=0, step=1, value=40)

    if st.button("Calcular medidas"):
        resultados = calcular_medidas(a,b,c,d)

        st.subheader("Resultados")
        st.write(f"Riesgo Relativo (RR): {resultados['RR']:.3f} (IC95%: {resultados['CI_RR'][0]:.3f} - {resultados['CI_RR'][1]:.3f})")
        st.write(f"Odds Ratio (OR): {resultados['OR']:.3f} (IC95%: {resultados['CI_OR'][0]:.3f} - {resultados['CI_OR'][1]:.3f})")
        st.write(f"Diferencia de Riesgos (RD): {resultados['RD']:.3f} (IC95%: {resultados['CI_RD'][0]:.3f} - {resultados['CI_RD'][1]:.3f})")
        st.write(f"Riesgo atribuible en expuestos (RAE): {resultados['RAE']:.3f}")
        st.write(f"Fracción atribuible en expuestos (FAE): {resultados['FAE']:.3f}")
        st.write(f"Fracción atribuible en población (FAP): {resultados['FAP']:.3f}")
        st.write(f"P-valor: {resultados['p_val']:.4f}")

        if resultados['corregido']:
            st.warning("Se aplicó corrección de Haldane-Anscombe debido a ceros en la tabla.")

        if resultados['p_val'] < 0.05:
            st.success("La asociación es estadísticamente significativa (p < 0.05).")
        else:
            st.info("La asociación NO es estadísticamente significativa (p ≥ 0.05).")

        st.subheader("Interpretación")
        if resultados['RR'] > 1:
            st.write("El riesgo es mayor en el grupo expuesto.")
        elif resultados['RR'] < 1:
            st.write("El riesgo es menor en el grupo expuesto.")
        else:
            st.write("No hay diferencia en el riesgo entre grupos.")

        st.subheader("Gráficos")
        plot_forest(resultados['RR'], resultados['CI_RR'], resultados['OR'], resultados['CI_OR'])
        plot_barras(a,b,c,d)

elif opcion == "Medidas de Asociación":
    st.markdown(medidas_asociacion_texto)

elif opcion == "Sesgos":
    st.markdown(sesgos_texto)

elif opcion == "Visualización de Datos":
    st.header("Carga y visualización de datos")

    archivo = st.file_uploader("Carga un archivo CSV con columnas numéricas", type=["csv"])

    if archivo:
        df = pd.read_csv(archivo)
        st.write(df.head())

        st.subheader("Gráficos disponibles")

        columnas_num = df.select_dtypes(include=np.number).columns.tolist()
        columna_graf = st.selectbox("Selecciona columna para gráfico Boxplot", columnas_num)

        if columna_graf:
            fig, ax = plt.subplots()
            ax.boxplot(df[columna_graf].dropna())
            ax.set_title(f"Boxplot de {columna_graf}")
            st.pyplot(fig)

elif opcion == "Ejercicios":
    st.header("Ejercicios prácticos")

    for i, ejercicio in enumerate(ejercicios):
        st.write(f"**{i+1}. {ejercicio['pregunta']}**")
        opciones = ejercicio['opciones']
        respuesta_usuario = st.radio(f"Selecciona una opción para la pregunta {i+1}:", opciones, key=f"ej_{i}")

        if respuesta_usuario == opciones[ejercicio['respuesta']]:
            st.success("Respuesta correcta 👍")
        else:
            st.error("Respuesta incorrecta ❌")

# --- Gemini ejemplo básico ---
if GENAI_AVAILABLE:
    st.sidebar.markdown("---")
    st.sidebar.header("Gemini AI (demo)")
    prompt = st.sidebar.text_input("Pregunta para Gemini:")
    if prompt:
        try:
            response = genai.generate_text(model="models/text-bison-001", prompt=prompt, max_output_tokens=128)
            st.sidebar.write(response.text)
        except Exception as e:
            st.sidebar.error(f"Error Gemini: {e}")

