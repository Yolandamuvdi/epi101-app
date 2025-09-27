import random
import datetime
from .ejercicios_completos import preguntas
import pandas as pd
from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas
from io import BytesIO
import base64

def simulacion_adaptativa(respuestas_usuario, max_preguntas=10, puntaje=0):
    """
    Simulación adaptativa para Epidemiología 101 con motivación, progreso y badges.
    """

    usadas = [r["pregunta"] for r in respuestas_usuario.values()]

    # Limite máximo de preguntas
    if len(respuestas_usuario) >= max_preguntas:
        badge = asignar_badge(puntaje)
        return None, f"🎉 ¡Simulación completada! Respondiste {len(respuestas_usuario)} preguntas. {badge}", puntaje

    if not respuestas_usuario:
        # Primera pregunta siempre nivel Básico
        disponibles = [q for q in preguntas if q["nivel"] == "Básico" and q["pregunta"] not in usadas]
        if not disponibles:
            return None, "No hay preguntas disponibles en nivel Básico.", puntaje
        pregunta = random.choice(disponibles)
        return pregunta, "🌟 Primera pregunta, nivel Básico. ¡Tú puedes!", puntaje

    ultima = list(respuestas_usuario.values())[-1]
    ultimo_nivel = ultima["nivel"]
    acierto = ultima["correcto"]
    mensaje = ""

    # Ajustar puntaje
    if acierto:
        puntaje += 10
    else:
        puntaje = max(0, puntaje - 5)

    # Lógica adaptativa
    if ultimo_nivel == "Básico":
        if acierto:
            nivel_siguiente = "Intermedio"
            mensaje = "🌟 ¡Bien hecho! Has subido al nivel Intermedio. Sigue así 🚀"
        else:
            nivel_siguiente = "Básico"
            mensaje = "💪 No te rindas, intenta otra pregunta en nivel Básico."

    elif ultimo_nivel == "Intermedio":
        if acierto:
            nivel_siguiente = "Avanzado"
            mensaje = "🔥 ¡Excelente! Ahora estás en nivel Avanzado. ¡Tú puedes!"
        else:
            nivel_siguiente = "Básico"
            mensaje = "👍 Respira y vuelve al nivel Básico para afianzar conceptos."

    elif ultimo_nivel == "Avanzado":
        if acierto:
            badge = asignar_badge(puntaje)
            return None, f"🏆 ¡Felicidades! Has completado la simulación adaptativa y alcanzaste el nivel Avanzado 🎉 {badge}", puntaje
        nivel_siguiente = "Intermedio"
        mensaje = "⚡ Casi llegas al final. Regresas a nivel Intermedio para reforzar conocimientos."

    # Buscar pregunta disponible
    disponibles = [q for q in preguntas if q["nivel"] == nivel_siguiente and q["pregunta"] not in usadas]

    if not disponibles:
        return None, "No hay más preguntas disponibles. Simulación finalizada.", puntaje

    pregunta = random.choice(disponibles)
    return pregunta, mensaje, puntaje


def asignar_badge(puntaje):
    """Asigna badges según puntaje alcanzado."""
    if puntaje >= 80:
        return "🏅 Badge Oro: ¡Epidemiólogo Maestro!"
    elif puntaje >= 50:
        return "🥈 Badge Plata: ¡Buen trabajo!"
    elif puntaje >= 20:
        return "🥉 Badge Bronce: ¡Vas por buen camino!"
    else:
        return "🎯 Badge Inicial: Lo importante es comenzar."


def exportar_resultados_pdf(respuestas_usuario, puntaje):
    """Genera un PDF con el historial de respuestas y puntaje."""
    buffer = BytesIO()
    c = canvas.Canvas(buffer, pagesize=letter)
    c.setFont("Helvetica", 12)

    c.drawString(50, 750, "Reporte Simulación Adaptativa - Epidemiología 101")
    c.drawString(50, 730, f"Fecha: {datetime.date.today()}")
    c.drawString(50, 710, f"Puntaje final: {puntaje}")

    y = 680
    for idx, datos in respuestas_usuario.items():
        texto = f"{idx}. {datos['pregunta']} | Nivel: {datos['nivel']} | Correcto: {datos['correcto']}"
        c.drawString(50, y, texto)
        y -= 20

    c.save()
    pdf = buffer.getvalue()
    buffer.close()
    return pdf


def exportar_resultados_excel(respuestas_usuario, puntaje):
    """Exporta historial de respuestas a Excel (DataFrame)."""
    data = [
        {"Pregunta": v["pregunta"], "Nivel": v["nivel"], "Correcto": v["correcto"]}
        for v in respuestas_usuario.values()
    ]
    df = pd.DataFrame(data)
    df["Puntaje final"] = puntaje
    return df
