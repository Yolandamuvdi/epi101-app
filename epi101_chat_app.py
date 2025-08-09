import streamlit as st
import pandas as pd
import numpy as np
import math
import matplotlib.pyplot as plt

# Intentar importar SciPy
try:
    from scipy.stats import chi2_contingency, fisher_exact
    scipy_ok = True
except ImportError:
    scipy_ok = False

st.set_page_config(page_title="Epi 101", layout="wide")

st.title("📚 Epi 101 - Herramientas Epidemiológicas")

tabs = st.tabs([
    "🏠 Inicio",
    "📖 Conceptos",
    "📚 Glosario",
    "🧮 Ejercicios",
    "📏 Medidas",
    "📊 Gráficos",
    "📊 Tablas 2x2"
])

with tabs[0]:
    st.header("🏠 Bienvenido a Epi 101")
    st.write("Plataforma interactiva para aprender y practicar conceptos básicos de Epidemiología.")

with tabs[1]:
    st.header("📖 Conceptos")
    st.write("Aquí irían definiciones clave de epidemiología...")

with tabs[2]:
    st.header("📚 Glosario")
    st.write("Glosario de términos epidemiológicos...")

with tabs[3]:
    st.header("🧮 Ejercicios")
    st.write("Ejercicios prácticos de cálculo epidemiológico...")

with tabs[4]:
    st.header("📏 Medidas")
    st.write("Explicación y fórmulas de medidas como RR, OR, RD...")

with tabs[5]:
    st.header("📊 Gráficos")
    st.write("Ejemplos de visualización de datos epidemiológicos...")

with tabs[6]:
    st.header("📊 Tablas 2x2 y Cálculos Epidemiológicos")

    col1, col2 = st.columns(2)
    with col1:
        a = st.number_input("Casos con exposición (a)", min_value=0, step=1)
        b = st.number_input("Casos sin exposición (b)", min_value=0, step=1)
    with col2:
        c = st.number_input("Controles con exposición (c)", min_value=0, step=1)
        d = st.number_input("Controles sin exposición (d)", min_value=0, step=1)

    if st.button("Calcular medidas"):
        try:
            # Guardar originales para tabla
            a0, b0, c0, d0 = a, b, c, d

            # Corrección de Haldane-Anscombe
            if 0 in [a, b, c, d]:
                a += 0.5
                b += 0.5
                c += 0.5
                d += 0.5
                st.info("⚠️ Se aplicó corrección de Haldane-Anscombe por presencia de ceros.")

            # Incidencias
            inc_exp = a / (a + b)
            inc_noexp = c / (c + d)

            # Riesgo Relativo (RR)
            rr = inc_exp / inc_noexp
            se_log_rr = math.sqrt((1/a - 1/(a+b)) + (1/c - 1/(c+d)))
            ci95_rr = (
                math.exp(math.log(rr) - 1.96*se_log_rr),
                math.exp(math.log(rr) + 1.96*se_log_rr)
            )

            # Odds Ratio (OR)
            orr = (a*d) / (b*c)
            se_log_or = math.sqrt(1/a + 1/b + 1/c + 1/d)
            ci95_or = (
                math.exp(math.log(orr) - 1.96*se_log_or),
                math.exp(math.log(orr) + 1.96*se_log_or)
            )

            # Diferencia de Riesgos (RD)
            rd = inc_exp - inc_noexp
            se_rd = math.sqrt((inc_exp*(1-inc_exp)/(a+b)) + (inc_noexp*(1-inc_noexp)/(c+d)))
            ci95_rd = (rd - 1.96*se_rd, rd + 1.96*se_rd)

            # Fracción Atribuible en Expuestos (FAE)
            fae = (rr - 1) / rr if rr != 0 else None

            # Número Necesario a Tratar / Exponer (NNT o NNH)
            nnt = 1 / rd if rd != 0 else None

            # Resultados
            st.subheader("📈 Resultados")
            st.write(f"Incidencia expuestos: {inc_exp:.3f}")
            st.write(f"Incidencia no expuestos: {inc_noexp:.3f}")
            st.success(f"RR = {rr:.2f} (IC95%: {ci95_rr[0]:.2f} – {ci95_rr[1]:.2f})")
            st.success(f"OR = {orr:.2f} (IC95%: {ci95_or[0]:.2f} – {ci95_or[1]:.2f})")
            st.info(f"RD = {rd:.3f} (IC95%: {ci95_rd[0]:.3f} – {ci95_rd[1]:.3f})")
            if fae is not None:
                st.write(f"Fracción atribuible en expuestos (FAE): {fae:.2%}")
            else:
                st.write("FAE no calculable")
            if nnt is not None:
                st.write(f"NNT/NNH: {nnt:.2f}")
            else:
                st.write("NNT/NNH no calculable")

            # Interpretación rápida
            if rr > 1:
                st.warning("El RR sugiere una asociación positiva (posible factor de riesgo).")
            elif rr < 1:
                st.success("El RR sugiere un posible efecto protector.")
            else:
                st.info("No hay evidencia de asociación.")

            # Estadísticas adicionales
            if scipy_ok:
                tabla = [[a0, b0], [c0, d0]]
                chi2, p_chi, _, _ = chi2_contingency(tabla)
                _, p_fisher = fisher_exact(tabla)
                st.write(f"Chi² = {chi2:.3f}, p = {p_chi:.4f}")
                st.write(f"Test exacto de Fisher, p = {p_fisher:.4f}")
            else:
                st.info("Instala SciPy para pruebas Chi² y Fisher.")

            # Gráfico de barras
            fig, ax = plt.subplots()
            ax.bar(["Expuestos", "No expuestos"], [inc_exp, inc_noexp], color=["#ff9999","#99ccff"])
            ax.set_ylabel("Incidencia")
            ax.set_ylim(0, 1)
            st.pyplot(fig)

            # Descargar resultados
            df_resultados = pd.DataFrame({
                "Medida": ["RR", "OR", "RD", "FAE", "NNT/NNH"],
                "Valor": [rr, orr, rd, fae, nnt],
                "IC95_inf": [ci95_rr[0], ci95_or[0], ci95_rd[0], None, None],
                "IC95_sup": [ci95_rr[1], ci95_or[1], ci95_rd[1], None, None]
            })
            csv = df_resultados.to_csv(index=False)
            st.download_button("📥 Descargar resultados CSV", csv, "resultados.csv", "text/csv")

        except Exception as e:
            st.error(f"Error en los cálculos: {e}")
