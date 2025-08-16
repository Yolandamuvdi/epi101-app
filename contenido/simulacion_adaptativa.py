import random
from ejercicios_completos import preguntas

def simulacion_adaptativa(respuestas_usuario):
    """
    respuestas_usuario = diccionario con historial de respuestas
    Ejemplo:
    {
        1: {"pregunta": "¿Qué significa incidencia?", "nivel": "Básico", "correcto": True},
        2: {"pregunta": "¿Cuál es un estudio analítico?", "nivel": "Intermedio", "correcto": False}
    }
    """

    # Lista de preguntas ya usadas
    usadas = [r["pregunta"] for r in respuestas_usuario.values()]

    if not respuestas_usuario:
        # Primera pregunta siempre nivel Básico
        disponibles = [q for q in preguntas if q["nivel"] == "Básico" and q["pregunta"] not in usadas]
        return random.choice(disponibles) if disponibles else None

    ultima = list(respuestas_usuario.values())[-1]
    ultimo_nivel = ultima["nivel"]
    acierto = ultima["correcto"]

    # Lógica adaptativa
    if ultimo_nivel == "Básico":
        if acierto:
            nivel_siguiente = "Intermedio"
        else:
            nivel_siguiente = "Básico"

    elif ultimo_nivel == "Intermedio":
        if acierto:
            nivel_siguiente = "Avanzado"
        else:
            nivel_siguiente = "Básico"

    elif ultimo_nivel == "Avanzado":
        if acierto:
            return None  # Simulación terminada
        else:
            nivel_siguiente = "Intermedio"

    # Buscar pregunta disponible en el nivel correspondiente
    disponibles = [q for q in preguntas if q["nivel"] == nivel_siguiente and q["pregunta"] not in usadas]
    return random.choice(disponibles) if disponibles else None
