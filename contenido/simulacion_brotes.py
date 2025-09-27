# contenido/simulacion_pro_brotes.py
"""
Módulo PRO: Simulación de Brotes (Integrado)
- WHO DONs (RSS)
- Mapas interactivos (folium / plotly)
- SEIR simplificado + intervención (escenario comparador)
- Casos ramificados, roles y decisiones
- Videos interactivos + preguntas
- Export PDF / Excel
- Alertas: nuevos DONs hoy
"""

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import requests
import io
import datetime
import json
import math
from io import BytesIO

# Optional dependencies with safe fallbacks
try:
    import feedparser
    FEEDPARSER_AVAILABLE = True
except Exception:
    FEEDPARSER_AVAILABLE = False

try:
    import folium
    from folium.plugins import HeatMap
    FOLIUM_AVAILABLE = True
except Exception:
    FOLIUM_AVAILABLE = False

try:
    from streamlit_folium import st_folium
    STREAMLIT_FOLIUM_AVAILABLE = True
except Exception:
    STREAMLIT_FOLIUM_AVAILABLE = False

try:
    import plotly.express as px
    PLOTLY_AVAILABLE = True
except Exception:
    PLOTLY_AVAILABLE = False

# PDF support
try:
    from reportlab.lib.pagesizes import letter
    from reportlab.pdfgen import canvas
    from reportlab.lib.utils import ImageReader
    from PIL import Image
    REPORTLAB_AVAILABLE = True
except Exception:
    REPORTLAB_AVAILABLE = False

# --------------------------
# CONSTANTES / CONFIG
# --------------------------
WHO_DON_RSS = "https://www.who.int/emergencies/disease-outbreak-news/rss/en/"
OWID_CSV = "https://covid.ourworldindata.org/data/owid-covid-data.csv"  # ejemplo
DEFAULT_SERIAL_INTERVAL_DAYS = 4.0

# --------------------------
# UTILIDADES
# --------------------------
@st.cache_data(show_spinner=False)
def fetch_who_dons():
    """Descarga WHO DONs via RSS (fallback si feedparser no está)."""
    if not FEEDPARSER_AVAILABLE:
        return None, "feedparser no instalado"
    d = feedparser.parse(WHO_DON_RSS)
    entries = []
    for e in d.entries:
        entries.append({
            "title": e.get("title"),
            "link": e.get("link"),
            "published": e.get("published"),
            "summary": e.get("summary")
        })
    return entries, None

@st.cache_data(show_spinner=False)
def fetch_owid_sample(nrows=2000):
    """Descarga una porción OWID como ejemplo (puede tardar)."""
    try:
        df = pd.read_csv(OWID_CSV, parse_dates=["date"])
        return df.head(nrows)
    except Exception as e:
        return None

def seir_simulate(N, I0, E0, R0_value, days, sigma=1/5.2, gamma=1/7, fatality=0.01, interventions=None):
    """
    Simulador SEIR determinista con posibilidad de intervención (reduce beta).
    - N: population
    - I0: initial infectives
    - E0: initial exposed
    - R0_value: reproduction number (R0)
    - days: number of days to simulate
    - sigma: 1/incubation period
    - gamma: 1/infectious period
    - fatality: IFR (proportion)
    - interventions: list of tuples (day_start, reduction_factor) e.g. (10, 0.5) reduces beta by 50% from day 10
    Returns DataFrame with S,E,I,R,new_infections,deaths
    """
    dt = 1.0
    beta = R0_value * gamma  # initial transmission rate
    S = N - I0 - E0
    E = E0
    I = I0
    R = 0.0
    results = []
    for t in range(days):
        # apply interventions
        effective_beta = beta
        if interventions:
            for (start_day, reduction) in interventions:
                if t >= start_day:
                    effective_beta = effective_beta * (1 - reduction)
        # SEIR equations
        new_exposed = (effective_beta * I * S / N) * dt
        new_infectious = sigma * E * dt
        new_recovered = gamma * I * dt
        # deaths as proportion of new_recovered * fatality (approx)
        new_deaths = new_recovered * fatality

        S = max(0, S - new_exposed)
        E = max(0, E + new_exposed - new_infectious)
        I = max(0, I + new_infectious - new_recovered)
        R = max(0, R + new_recovered - new_deaths)

        results.append({
            "day": t,
            "S": S,
            "E": E,
            "I": I,
            "R": R,
            "new_infections": new_exposed,
            "new_recovered": new_recovered,
            "new_deaths": new_deaths,
            "beta": effective_beta
        })
    df = pd.DataFrame(results)
    df["date"] = pd.Timestamp.today().normalize() + pd.to_timedelta(df["day"], unit="D")
    return df

def fig_to_bytes(fig, fmt="png"):
    buf = BytesIO()
    fig.savefig(buf, format=fmt, bbox_inches="tight")
    buf.seek(0)
    return buf.getvalue()

def create_pdf_report(title, subtitle, text_lines, fig_bytes_list):
    """Crea PDF con texto y figuras (usa reportlab)."""
    if not REPORTLAB_AVAILABLE:
        return None
    buf = BytesIO()
    c = canvas.Canvas(buf, pagesize=letter)
    c.setFont("Helvetica-Bold", 14)
    c.drawString(40, 760, title)
    c.setFont("Helvetica", 10)
    c.drawString(40, 745, subtitle)
    y = 720
    for ln in text_lines:
        c.drawString(40, y, ln)
        y -= 12
        if y < 120:
            c.showPage()
            y = 760
    for fb in fig_bytes_list:
        try:
            img = ImageReader(BytesIO(fb))
            c.drawImage(img, 40, 320, width=520, height=320)
            c.showPage()
        except Exception:
            pass
    c.save()
    buf.seek(0)
    return buf.getvalue()

# --------------------------
# BANCO DE CASOS (históricos + ficticios)
# --------------------------
# Cases include: id, title, description, video (optional), timeline (list of events), initial conditions for SEIR
CASES = [
    {
        "id":"cholera_haiti",
        "title":"Cólera - Haití (2010) - Caso histórico",
        "description":"Brote de cólera post-huracán con rápida transmisión por agua contaminada. Lecciones: acceso a agua segura y tratamiento de deshidratación.",
        "video":"https://www.youtube.com/watch?v=tVcmFSF9N6E",  # ejemplo
        "population":100000,
        "init": {"I0":10, "E0":0, "R0":2.0, "fatality":0.01},
        "timeline":[
            {"date":"2010-10-01","event":"Primeros casos reportados"},
            {"date":"2010-10-05","event":"Aumento semanal del 300%"}
        ],
        "tags":["cholera","waterborne","historical"]
    },
    {
        "id":"ebola_west_africa",
        "title":"Ébola - África Occidental (2014) - Caso histórico",
        "description":"Brote de Ébola con alta letalidad y transmisión por contacto. Lecciones: aislamiento, rastreo de contactos, confianza comunitaria.",
        "video":"https://www.youtube.com/watch?v=a8dIqXYxYLA",
        "population":500000,
        "init": {"I0":5, "E0":2, "R0":1.8, "fatality":0.4},
        "timeline":[],
        "tags":["ebola","hemorrhagic","historical"]
    },
    {
        "id":"dengue_municipio",
        "title":"Dengue - Municipio costero (Simulado)",
        "description":"Aumento de casos febriles con signos sugestivos de dengue. Enfoque en vigilancia entomológica y control vectorial.",
        "video":"https://www.youtube.com/watch?v=UDni_A0cLpM",
        "population":50000,
        "init": {"I0":8, "E0":5, "R0":2.5, "fatality":0.005},
        "timeline":[],
        "tags":["dengue","vectorborne","simulation"]
    },
    {
        "id":"covid_university",
        "title":"COVID-19 - Brote en Universidad (Simulado)",
        "description":"Surtido de casos en campus universitario. Rápida toma de decisiones sobre aislamiento y clases.",
        "video":"https://www.youtube.com/watch?v=RGhn-fW2424",
        "population":20000,
        "init": {"I0":15, "E0":10, "R0":2.1, "fatality":0.005},
        "timeline":[],
        "tags":["covid","institution","simulation"]
    }
]

# --------------------------
# ROLE-BASED DECISION TREE (simple)
# --------------------------
DECISION_TREES = {
    # Each node: text, options => next node id or effect (function)
    "dengue_response": {
        "start":{
            "text":"Se detectan 20 casos febriles. ¿Qué haces primero?",
            "options":[
                {"label":"Notificar INS y buscar casos", "effect":{"score":10,"intervention":(3,0.4)}},
                {"label":"Iniciar fumigación masiva sin investigar", "effect":{"score":-5,"intervention":(7,0.1)}},
                {"label":"Esperar resultados de laboratorio", "effect":{"score":-10,"intervention":None}}
            ]
        }
    }
}

# --------------------------
# APP UI
# --------------------------
def app():
    st.title("🧠 Simulación PRO de Brotes — Epi101 (Módulo avanzado)")
    st.markdown("Panel PRO: datos en tiempo real, mapas, SEIR, roles, casos históricos y decisiones ramificadas.")

    # Sidebar: difficulty, role, alerts
    st.sidebar.header("Configuración PRO")
    difficulty = st.sidebar.selectbox("Nivel", ["Estudiante","Profesional","Experto"], index=0)
    role = st.sidebar.selectbox("Rol", ["Epidemiólogo de campo","Autoridad sanitaria local","Vocero de comunicación"], index=0)
    st.sidebar.markdown("---")
    st.sidebar.markdown("🔔 Alertas WHO DONs")
    # Fetch WHO DONs
    entries, err = (None, None)
    if FEEDPARSER_AVAILABLE:
        entries, err = fetch_who_dons()
    else:
        st.sidebar.info("Instala feedparser para alertas WHO DONs (feedparser).")

    # Show headlines and mark new today
    if entries:
        today = datetime.date.today()
        new_today = []
        for e in entries:
            try:
                pub = pd.to_datetime(e["published"])
                if pub.date() == today:
                    new_today.append(e)
            except Exception:
                pass
        if new_today:
            st.sidebar.success(f"{len(new_today)} DON(s) publicados hoy")
            for e in new_today[:5]:
                st.sidebar.markdown(f"- [{e['title']}]({e['link']})")
        else:
            st.sidebar.write("No hay DONs nuevos hoy (según feed).")
    else:
        if err:
            st.sidebar.warning(f"WHO feed: {err}")
        else:
            st.sidebar.info("No hay acceso a WHO DONs (feedparser faltante o conexión).")

    # Main content: tabs
    tab_data, tab_map, tab_sim, tab_cases, tab_history = st.tabs([
        "📡 Datos en tiempo real",
        "🌍 Mapas & Heatmap",
        "🎲 Simulación SEIR (comparador)",
        "🧭 Casos y decisiones (roles)",
        "📚 Biblioteca histórica"
    ])

    # --------------------------
    # TAB 1: Datos en tiempo real (WHO DONs + OWID sample)
    # --------------------------
    with tab_data:
        st.header("📡 Datos en tiempo real")
        st.markdown("Conexión WHO Disease Outbreak News (DONs) y dataset OWID (ejemplo).")
        # WHO DONs list
        if entries:
            st.subheader("WHO Disease Outbreak News (últimos)")
            for e in entries[:10]:
                st.markdown(f"**[{e['title']}]({e['link']})**  \n_{e.get('published','')}_")
                st.write(e.get("summary","")[:300] + "...")
        else:
            st.info("No se pudieron obtener DONs. Instala feedparser o revisa conexión.")
        # OWID sample
        st.subheader("Our World in Data (muestra)")
        df_owid = fetch_owid_sample()
        if df_owid is not None:
            # interactive preview and country selector
            st.write("Preview OWID (COVID example).")
            st.dataframe(df_owid.iloc[:100][["location","date","new_cases"]].head(100))
            if PLOTLY_AVAILABLE:
                country_choices = sorted(df_owid["location"].unique().tolist())
                country = st.selectbox("Selecciona país (OWID sample)", country_choices, index=country_choices.index("Colombia") if "Colombia" in country_choices else 0)
                df_ctry = df_owid[df_owid["location"]==country].copy()
                fig = px.line(df_ctry, x="date", y="new_cases", title=f"Serie new_cases - {country}")
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.line_chart(df_owid.groupby("date")["new_cases"].sum().fillna(0))
        else:
            st.info("OWID no disponible (conexión fallida). Puedes subir CSV.")

    # --------------------------
    # TAB 2: Mapas & Heatmap
    # --------------------------
    with tab_map:
        st.header("🌍 Mapas Interactivos")
        st.markdown("Mapa global demo + heatmap simulado según casos y R0.")
        # allow user to pick a case population center set
        loc_options = {
            "Simulated cluster - Coastal city": {"lat":10.4,"lon":-75.5, "cases":[(10.4,-75.5,30),(10.42,-75.52,20),(10.38,-75.48,10)]},
            "Simulated cluster - Capital": {"lat":4.6,"lon":-74.07, "cases":[(4.6,-74.07,80),(4.61,-74.05,40),(4.59,-74.09,20)]}
        }
        chosen = st.selectbox("Seleccione cluster de demo:", list(loc_options.keys()))
        cluster = loc_options[chosen]
        if FOLIUM_AVAILABLE and STREAMLIT_FOLIUM_AVAILABLE:
            m = folium.Map(location=[cluster["lat"], cluster["lon"]], zoom_start=12)
            heat_data = [[lat, lon, count] for lat,lon,count in [(c[0],c[1],c[2]) for c in cluster["cases"]]]
            HeatMap([[c[0],c[1], c[2]] for c in cluster["cases"]], radius=25).add_to(m)
            for lat,lon,count in cluster["cases"]:
                folium.Circle(location=[lat,lon], radius=50+count*5, popup=f"Cases:{count}", color="crimson", fill=True).add_to(m)
            st_folium(m, width=700, height=450)
        else:
            st.info("Instala folium + streamlit_folium para ver mapas interactivos (pip install folium streamlit_folium).")
            st.write(cluster)

        # Simulate propagation heatmap based on R0 & mobility slider
        st.subheader("Simulación rápida: heatmap por R0 + movilidad")
        R0_user = st.slider("R0 (transmisibilidad)", 0.5, 4.0, 1.8, 0.1)
        mobility = st.slider("Movilidad (0 bajo - 1 alto)", 0.0, 1.0, 0.5, 0.05)
        # create grid of points around cluster center and compute risk = base * exp(R0*mob)
        lats = np.linspace(cluster["lat"]-0.03, cluster["lat"]+0.03, 25)
        lons = np.linspace(cluster["lon"]-0.03, cluster["lon"]+0.03, 25)
        risk = np.zeros((25,25))
        for i,lat in enumerate(lats):
            for j,lon in enumerate(lons):
                # base risk decays with distance from center
                dist = math.hypot(lat - cluster["lat"], lon - cluster["lon"])
                base = math.exp(-dist*50)
                risk[i,j] = base * math.exp(R0_user*mobility/2)
        # show heatmap via matplotlib
        fig, ax = plt.subplots(figsize=(6,4))
        im = ax.imshow(risk, cmap="hot", origin="lower")
        ax.set_title("Mapa de riesgo (simulación)")
        plt.colorbar(im, ax=ax, label="Relative risk")
        st.pyplot(fig)

    # --------------------------
    # TAB 3: SEIR Simulation (comparador de intervenciones)
    # --------------------------
    with tab_sim:
        st.header("🎲 Simulación SEIR avanzada")
        st.markdown("Configura R0, letalidad e intervenciones. Compara escenarios lado a lado.")
        # baseline inputs
        population = st.number_input("Población (N)", min_value=1000, value=100000)
        I0 = st.number_input("Infectados iniciales (I0)", min_value=1, value=10)
        E0 = st.number_input("Expuestos iniciales (E0)", min_value=0, value=5)
        R0_val = st.slider("R0 (baseline)", 0.5, 5.0, 2.2, 0.1)
        fatality = st.slider("Fracción fatalidad (IFR)", 0.0, 0.5, 0.01, 0.001)
        days = st.slider("Días a simular", 30, 365, 120)

        # interventions definition UI
        st.markdown("Define intervenciones (reducción relativa de transmisión a partir del día X).")
        interventions = []
        col1, col2, col3 = st.columns([1,1,1])
        day1 = col1.number_input("Día inicio interv 1", min_value=0, max_value=days, value=10)
        red1 = col2.slider("Reducción (%) interv 1", 0, 100, 40) / 100.0
        add_int = col3.button("Agregar intervención")
        if add_int:
            interventions.append((int(day1), float(red1)))
            st.session_state.setdefault("user_interventions", []).append((int(day1), float(red1)))
        # merge session interventions
        session_int = st.session_state.get("user_interventions", [])
        if session_int:
            st.write("Intervenciones guardadas:", session_int)

        # compare scenarios: baseline vs interventions
        if st.button("Simular escenarios"):
            # baseline
            df_baseline = seir_simulate(population, I0, E0, R0_val, days, fatality=fatality, interventions=None)
            # with interventions from session
            df_int = seir_simulate(population, I0, E0, R0_val, days, fatality=fatality, interventions=session_int)
            # plot comparison
            fig, ax = plt.subplots(figsize=(9,4))
            ax.plot(df_baseline["date"], df_baseline["I"], label="I - baseline")
            ax.plot(df_int["date"], df_int["I"], label="I - con intervenciones")
            ax.set_ylabel("Número infectados (I)")
            ax.set_xlabel("Fecha")
            ax.legend()
            st.pyplot(fig)
            # metrics: peak I and day
            peak_b = df_baseline["I"].max(); day_b = df_baseline.loc[df_baseline["I"].idxmax(), "date"]
            peak_i = df_int["I"].max(); day_i = df_int.loc[df_int["I"].idxmax(), "date"]
            st.metric("Peak baseline (I)", f"{int(peak_b)} on {pd.to_datetime(day_b).date()}")
            st.metric("Peak with interventions (I)", f"{int(peak_i)} on {pd.to_datetime(day_i).date()}")
            # enable export
            buf_xl = BytesIO()
            with pd.ExcelWriter(buf_xl, engine="openpyxl") as writer:
                df_baseline.to_excel(writer, sheet_name="baseline", index=False)
                df_int.to_excel(writer, sheet_name="interventions", index=False)
            buf_xl.seek(0)
            st.download_button("⬇️ Descargar datos (Excel)", data=buf_xl, file_name="seir_compare.xlsx", mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")
            # pdf with figs
            fig_bytes = fig_to_bytes(fig)
            pdf_bytes = create_pdf_report(f"SEIR simulation - {datetime.date.today()}", f"R0={R0_val}, fatality={fatality}", ["Comparación baseline vs intervenciones"], [fig_bytes])
            if pdf_bytes:
                st.download_button("⬇️ Descargar reporte PDF", data=pdf_bytes, file_name="seir_report.pdf", mime="application/pdf")
            else:
                st.info("Instala reportlab + pillow para exportar PDF con figuras.")

    # --------------------------
    # TAB 4: Casos & Decisiones (roles + branching)
    # --------------------------
    with tab_cases:
        st.header("🧭 Casos interactivos y decisiones (roles)")
        st.markdown("Selecciona un caso y toma decisiones según tu rol. Las decisiones afectan la simulación y el puntaje.")
        case_ids = [c["id"] for c in CASES]
        sel_id = st.selectbox("Selecciona un caso", case_ids, format_func=lambda x: next((c['title'] for c in CASES if c['id']==x), x))
        case = next(c for c in CASES if c["id"]==sel_id)
        st.subheader(case["title"])
        st.write(case["description"])
        if case.get("video"):
            st.markdown("**Video del caso** (ver antes de decidir):")
            try:
                st.video(case["video"])
            except Exception:
                st.write("Enlace de video:", case["video"])

        # role-based decisions
        st.markdown(f"**Tu rol:** {role}")
        # Simple branching: provide 2 decision points with consequences that modify interventions
        decisions_record = st.session_state.setdefault("decisions_record", {})
        score = st.session_state.setdefault("decisions_score", 0)

        # Decision 1 (triage)
        d1 = st.radio("Decisión 1: ¿Cuál es tu primera acción?", [
            "Notificar a autoridad y activar vigilancia", 
            "Esperar confirmación laboratorio", 
            "Implementar control inmediato (mass interventions)"
        ], key=f"d1_{sel_id}")
        if st.button("Enviar decisión 1", key=f"btn_d1_{sel_id}"):
            if d1 == "Notificar a autoridad y activar vigilancia":
                st.success("✅ Correcto — activa vigilancia y rastreo.")
                score += 15
                st.session_state["decisions_score"] = score
                st.session_state.setdefault("applied_interventions", []).append((3,0.5))
            elif d1 == "Esperar confirmación laboratorio":
                st.warning("⚠️ La espera puede costar tiempo.")
                score -= 5
                st.session_state["decisions_score"] = score
            else:
                st.info("Intervención agresiva registrada — evaluar costo/beneficio.")
                score += 5
                st.session_state["decisions_score"] = score
                st.session_state.setdefault("applied_interventions", []).append((1,0.6))

        # Decision 2 (risk communication) - depends on role
        st.markdown("Decisión 2: Comunicación y medidas comunitarias")
        if role == "Vocero de comunicación":
            opt = ["Transparencia y mensajes claros", "Minimizar riesgo públicamente", "Difundir rumores"]
            correct = "Transparencia y mensajes claros"
        elif role == "Autoridad sanitaria local":
            opt = ["Movilizar recursos y coordinar", "Cerrar todo sin plan", "No informar"]
            correct = "Movilizar recursos y coordinar"
        else:  # field epi
            opt = ["Búsqueda activa y toma de muestras", "Solo atención hospitalaria", "Nada"]
            correct = "Búsqueda activa y toma de muestras"

        d2 = st.radio("Elige acción:", opt, key=f"d2_{sel_id}")
        if st.button("Enviar decisión 2", key=f"btn_d2_{sel_id}"):
            if d2 == correct:
                st.success("✅ Buena decisión según rol; impacto positivo en control.")
                score += 10
                st.session_state["decisions_score"] = score
            else:
                st.error("❌ Mala decisión: aumenta riesgo de propagación.")
                score -= 10
                st.session_state["decisions_score"] = score

        st.markdown("### Estado actual")
        st.write(f"Puntaje acumulado: {st.session_state.get('decisions_score',0)}")
        st.write("Intervenciones aplicadas (día, reducción):", st.session_state.get("applied_interventions", []))

        # Run a quick SEIR with current interventions for scenario preview
        if st.button("Simular con intervenciones aplicadas"):
            interventions = st.session_state.get("applied_interventions", [])
            init = case["init"]
            df_sim = seir_simulate(case["population"], init["I0"], init["E0"], init["R0"], days=120, fatality=init.get("fatality",0.01), interventions=interventions)
            fig, ax = plt.subplots(figsize=(8,3))
            ax.plot(df_sim["date"], df_sim["I"], label="Infectados")
            ax.plot(df_sim["date"], df_sim["new_deaths"].cumsum(), label="Muertes acumuladas")
            ax.set_title(f"Simulación - {case['title']}")
            ax.legend()
            st.pyplot(fig)
            # allow export
            excel_buf = BytesIO()
            with pd.ExcelWriter(excel_buf, engine="openpyxl") as writer:
                df_sim.to_excel(writer, index=False, sheet_name="simulation")
            excel_buf.seek(0)
            st.download_button("⬇️ Descargar simulación (Excel)", data=excel_buf, file_name=f"sim_{case['id']}.xlsx")

            # PDF report
            fig_bytes = fig_to_bytes(fig)
            pdf_bytes = create_pdf_report(f"Reporte caso - {case['title']}", f"Rol: {role}", [f"Puntaje: {st.session_state.get('decisions_score',0)}"], [fig_bytes])
            if pdf_bytes:
                st.download_button("⬇️ Descargar PDF del caso", data=pdf_bytes, file_name=f"report_{case['id']}.pdf", mime="application/pdf")
            else:
                st.info("Instala reportlab y pillow para exportar PDF con figuras.")

    # --------------------------
    # TAB 5: Biblioteca histórica
    # --------------------------
    with tab_history:
        st.header("📚 Biblioteca de brotes históricos")
        st.markdown("Bases de casos históricos y lecciones. Selecciona para ver detalles y cronología.")
        for c in CASES:
            with st.expander(c["title"]):
                st.write(c["description"])
                if c.get("timeline"):
                    st.markdown("**Cronología (ejemplo):**")
                    for ev in c["timeline"]:
                        st.markdown(f"- {ev['date']}: {ev['event']}")
                if c.get("video"):
                    st.markdown("**Video relacionado**")
                    try:
                        st.video(c["video"])
                    except Exception:
                        st.write(c["video"])
                st.markdown("**Tags:** " + ", ".join(c.get("tags", [])))
                st.markdown("---")

    # --------------------------
    # End: summary / reset
    # --------------------------
    st.sidebar.markdown("---")
    if st.sidebar.button("Reset decisiones & sesiones (PRO)"):
        keys = ["applied_interventions","decisions_score","decisions_record","user_interventions","brotes_state","decisions_record"]
        for k in keys:
            if k in st.session_state: del st.session_state[k]
        st.sidebar.success("Estado reseteado.")

# Run app if called directly
if __name__ == "__main__":
    app()

