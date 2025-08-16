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

    if not respuestas_usuario:
        # Primera pregunta siempre en nivel Básico
        return [q for q in preguntas if q["nivel"] == "Básico"][0]

    ultima = list(respuestas_usuario.values())[-1]
    ultimo_nivel = ultima["nivel"]
    acierto = ultima["correcto"]

    if ultimo_nivel == "Básico":
        if acierto:
            return [q for q in preguntas if q["nivel"] == "Intermedio"][0]
        else:
            return [q for q in preguntas if q["nivel"] == "Básico"][1]

    elif ultimo_nivel == "Intermedio":
        if acierto:
            return [q for q in preguntas if q["nivel"] == "Avanzado"][0]
        else:
            return [q for q in preguntas if q["nivel"] == "Básico"][2]

    elif ultimo_nivel == "Avanzado":
        if acierto:
            return None  # Simulación terminada
        else:
            return [q for q in preguntas if q["nivel"] == "Intermedio"][1]
