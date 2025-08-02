MEDIDAS DE ASOCIACIÓN EN EPIDEMIOLOGÍA
1. ¿Qué son?
Las medidas de asociación permiten cuantificar la relación entre una exposición (como fumar) y un desenlace (como el cáncer de pulmón).

Nos ayudan a responder: ¿la exposición está relacionada con el evento? ¿Cuánto más probable es?

2. Tipos principales
a. Riesgo Relativo (RR)
¿Qué tan probable es que los expuestos desarrollen el evento comparado con los no expuestos?

Fórmula:
RR = Incidencia en expuestos / Incidencia en no expuestos

# Interpretación del Riesgo Relativo (RR)

| Valor del RR       | Interpretación                                | ¿Por qué?                                                                                       |
|--------------------|-----------------------------------------------|------------------------------------------------------------------------------------------------|
| RR = 1             | No hay asociación entre exposición y evento   | El riesgo de enfermar es igual en expuestos y no expuestos.                                   |
| RR > 1             | La exposición aumenta el riesgo del evento    | El evento (por ejemplo, una enfermedad) ocurre más en el grupo expuesto → posible factor de riesgo. |
| RR < 1             | La exposición es protectora                   | El evento ocurre menos en el grupo expuesto → posible factor protector.                       |


b. Odds Ratio (OR)
Se usa especialmente en estudios de casos y controles.

Fórmula:
OR = (a/c) / (b/d) = (a×d)/(b×c)

# Interpretación del Odds Ratio (OR)

| Valor del OR       | Interpretación                                                  | ¿Por qué?                                                                                     |
|--------------------|----------------------------------------------------------------|----------------------------------------------------------------------------------------------|
| OR = 1             | No hay asociación entre la exposición y el evento              | La probabilidad del evento es igual en expuestos y no expuestos.                             |
| OR > 1             | La exposición se asocia a mayor probabilidad del evento        | La exposición podría ser un factor de riesgo. El evento ocurre más en el grupo expuesto.     |
| OR < 1             | La exposición se asocia a menor probabilidad del evento        | La exposición podría ser un factor protector. El evento ocurre menos en el grupo expuesto.   |


c. Razón de Prevalencias (RP)
Se utiliza en estudios transversales para comparar la prevalencia del desenlace entre expuestos y no expuestos.

Fórmula:
RP = Prevalencia en expuestos / Prevalencia en no expuestos

# Interpretación de la Razón de Prevalencias (RP)

| Valor de la RP     | Interpretación                                 | ¿Por qué?                                                                                          |
|--------------------|------------------------------------------------|---------------------------------------------------------------------------------------------------|
| RP = 1             | No hay asociación entre exposición y desenlace | La prevalencia del desenlace es igual en los grupos expuesto y no expuesto.                      |
| RP > 1             | Mayor prevalencia del desenlace con exposición | El desenlace (enfermedad, condición, etc.) es más común en los expuestos → posible asociación positiva. |
| RP < 1             | Menor prevalencia del desenlace con exposición | El desenlace es menos común en los expuestos → la exposición podría ser protectora.              |

📊 Tabla 2x2 para análisis epidemiológico|                
|                  | Evento (sí) | Evento (no) | Total       |
| ---------------- | ----------- | ----------- | ----------- |
| **Expuestos**    | a           | b           | a + b       |
| **No expuestos** | c           | d           | c + d       |
| **Total**        | a + c       | b + d       | N (a+b+c+d) |

# ¿Cómo elegir la medida adecuada?

| Diseño del estudio | Medida recomendada         | ¿Por qué?                                                                                   |
|--------------------|----------------------------|--------------------------------------------------------------------------------------------|
| Cohorte            | Riesgo Relativo (RR)       | Permite comparar el riesgo de enfermedad entre expuestos y no expuestos en un seguimiento. |
| Casos y controles  | Odds Ratio (OR)            | Útil cuando no se puede calcular incidencia directamente, compara probabilidades relativas. |
| Transversal        | Razón de Prevalencias (RP) | Evalúa la proporción de casos existentes en diferentes grupos en un solo momento.          |


🔍 4. Comparación entre RR, OR y RP
| Característica       | RR (Riesgo Relativo)    | OR (Odds Ratio)                   | RP (Razón de Prevalencias) |
| -------------------- | ----------------------- | --------------------------------- | -------------------------- |
| ¿En qué estudios?    | Cohorte                 | Casos y controles                 | Transversales              |
| ¿Usa incidencia?     | Sí                      | No (usa odds)                     | No (usa prevalencia)       |
| ¿Mide riesgo real?   | Sí                      | No directamente                   | No directamente            |
| Interpretación fácil | Sí                      | Menos intuitiva si OR > 2 o < 0.5 | Similar al RR              |
| ¿Estima causalidad?  | Con diseño adecuado, sí | No directamente                   | No directamente            |

6. Limitaciones de las medidas
RR: Solo se puede calcular si se conoce el tiempo de seguimiento y el total en riesgo.

OR: Puede sobreestimar la fuerza de asociación si el evento es muy frecuente.

RP: No indica causalidad ni permite inferencias temporales.

🎯 7. Cómo reportarlas (en publicaciones científicas)
"Se encontró que el riesgo de enfermedad fue 2 veces mayor en los expuestos comparado con los no expuestos (RR=2.0, IC95%: 1.2–3.3, p<0.01)."

TIP: Siempre reporta con IC95% y valor p para interpretación precisa.

