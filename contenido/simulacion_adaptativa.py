import random
from ejercicios_completos import preguntas

def simulacion_adaptativa(respuestas_usuario, max_preguntas=10):
    """
    Simulación adaptativa para Epidemiología 101 con mensajes motivadores.

    respuestas_usuario = diccionario con historial de respuestas
    Ejemplo:
    {
        1: {"pregunta": "¿Qué significa incidencia?", "nivel": "Básico", "correcto": True},
        2: {"pregunta": "¿Cuál es un estudio analítico?", "nivel": "Intermedio", "correcto": False}
    }

    max_preguntas = número máximo de preguntas por simulación
    """

    usadas = [r["pregunta"] for r in respuestas_usuario.values()]

    # Limite máximo de preguntas
    if len(respuestas_usuario) >= max_preguntas:
        return {"mensaje": f"🎉 ¡Simulación completada! Respondiste {len(respuestas_usuario)} preguntas. Excelente trabajo."}

    if not respuestas_usuario:
        # Primera pregunta siempre nivel Básico
        disponibles = [q for q in preguntas if q["nivel"] == "Básico" and q["pregunta"] not in usadas]
        return random.choice(disponibles) if disponibles else None

    ultima = list(respuestas_usuario.values())[-1]
    ultimo_nivel = ultima["nivel"]
    acierto = ultima["correcto"]
    mensaje = ""

    # Lógica adaptativa con motivación
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
            return {"mensaje": "🏆 ¡Felicidades! Has completado la simulación adaptativa y alcanzaste el nivel Avanzado 🎉"}
        nivel_siguiente = "Intermedio"
        mensaje = "⚡ Casi llegas al final. Regresas a nivel Intermedio para reforzar conocimientos."

    # Buscar pregunta disponible en el nivel correspondiente
    disponibles = [q for q in preguntas if q["nivel"] == nivel_siguiente and q["pregunta"] not in usadas]

    if not disponibles:
        # No hay más preguntas en el nivel, intentar subir/bajar nivel
        niveles_orden = ["Básico", "Intermedio", "Avanzado"]
        idx = niveles_orden.index(nivel_siguiente)
        for i in range(len(niveles_orden)):
            siguiente_idx = (idx + i) % len(niveles_orden)
            posibles = [q for q in preguntas if q["nivel"] == niveles_orden[siguiente_idx] and q["pregunta"] not in usadas]
            if posibles:
                return {"pregunta": random.choice(posibles), "mensaje": mensaje}

        return {"mensaje": "No hay más preguntas disponibles. Simulación finalizada."}

    return {"pregunta": random.choice(disponibles), "mensaje": mensaje}
