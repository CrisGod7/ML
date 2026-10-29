# Plan del proyecto: análisis probabilístico de partidos de fútbol

> Plan de ejecución. Complementa a [`MODELOS_ML_FUTBOL.md`](MODELOS_ML_FUTBOL.md) (catálogo técnico de modelos y validación).
> Estado: **planificación** — no se ha escrito código del proyecto todavía.

---

## 1. Principios de diseño

| # | Principio | Qué implica |
|---|---|---|
| 1 | **Estadística primero** | El modelo se construye con datos de partidos, jugadores, entrenadores y contexto. **Las cuotas no son una entrada ni una referencia obligatoria** |
| 2 | **Probabilidades, no certezas** | Cada resultado se entrega como un porcentaje calibrado ("Ambos anotan: 67 %"), nunca como "pick seguro" |
| 3 | **Conservador por diseño** | La cifra que ve el usuario lleva un **margen de seguridad** calculado (sección 4), no un número optimista |
| 4 | **Muchos parámetros, seleccionados con evidencia** | Se parte de un catálogo amplio (sección 5) y cada grupo de variables debe demostrar su aporte fuera de muestra |
| 5 | **Varios tipos de ML combinados** | Estadístico generativo + gradient boosting + redes + modelos de conteo + meta-modelo (sección 6) |
| 6 | **Validación sin trampas** | Walk-forward, test bloqueado, pruebas de fuga de datos, calibración por liga (sección 7) |
| 7 | **Transparencia** | La interfaz muestra también cuánto ha acertado el sistema históricamente por rango de probabilidad (sección 8) |

### Sobre las casas de apuestas

Tienes razón en que las cuotas incluyen un margen (el *overround*: las probabilidades implícitas suman más de 100 %), así que leídas "tal cual" sobreestiman cada resultado. Ese margen se puede quitar, pero aun así decidimos **no depender de ellas**: el modelo principal se entrena y se evalúa solo con estadística. Quedan únicamente como **auditoría opcional offline** (comparar después, sin margen, para detectar fallos graves); si no se usan, nada del proyecto se rompe.

---

## 2. Lo que el producto debe entregar

### 2.1 Mercados (por partido)

| Grupo | Mercados | Modelo principal que lo alimenta |
|---|---|---|
| Resultado | 1X2, doble oportunidad, empate no apuesta, gana a cero | Marcador (Dixon-Coles / bayesiano) + GBM |
| Goles | **Ambos anotan (BTTS)**, over/under 0.5 · 1.5 · 2.5 · 3.5, goles por equipo, marcador exacto (top 5) | Marcador + GBM binario |
| Tiempos | Gol en 1.ª parte, resultado al descanso, over 0.5 en cada mitad | Marcador por tiempos |
| Córners | Over/under córners totales y por equipo | Binomial Negativa + GBM Poisson |
| Disciplina | Tarjetas totales y por equipo, penalti | Binomial Negativa (+ árbitro) |
| Tiros | Tiros y tiros a puerta por equipo | Binomial Negativa + GBM |
| **Jugadores** | P(tiros a puerta ≥ 1, ≥ 2), P(marca gol), P(asistencia), P(tarjeta) | Minutos esperados × tasa jerárquica |

### 2.2 Salida por mercado

Para cada mercado el sistema devuelve:

| Campo | Significado |
|---|---|
| `p_calibrated` | Probabilidad calibrada del modelo (ej. 80 %) |
| `p_shown` | **Probabilidad conservadora** que se muestra (ej. 75 %) |
| `interval` | Rango plausible (ej. 71–84 %) |
| `confidence` | Nivel: **Alta / Media / Baja** (según incertidumbre y calidad de datos) |
| `flags` | Avisos: alineación no confirmada, entrenador nuevo, pocos datos, desacuerdo entre modelos… |
| `drivers` | Los 3 factores que más empujan la predicción (explicación legible) |

---

## 3. Fuentes de datos (sin depender de cuotas)

Resumen; el detalle de endpoints está en el otro documento.

| Necesidad | Fuente principal | Alternativa / complemento |
|---|---|---|
| Calendario, resultados, tiros, córners, tarjetas | API-Football, football-data.co.uk (CSV) | football-data.org |
| Alineaciones y formaciones | API-Football `/fixtures/lineups` | Predicción propia del XI cuando aún no hay oficial |
| Estadísticas por jugador y partido | API-Football `/fixtures/players` | StatsBomb Open Data (competiciones concretas) |
| Lesiones y sanciones | API-Football `/injuries` | Transfermarkt (datasets derivados) |
| xG y calidad de ocasiones | Understat | Datos de eventos (StatsBomb, de pago: Opta/Wyscout) |
| Elo | ClubElo (y Elo propio calculado) | — |
| Entrenadores | API-Football `/coachs` | Transfermarkt |
| Clima | Open-Meteo | — |
| Árbitro | football-data.co.uk, API-Football | — |

**Decisión pendiente (tuya):** plan de API-Football y presupuesto, porque limita cuántas ligas y cuánta historia puedes bajar (ver sección 11).

---

## 4. Margen de seguridad en las probabilidades

Tu idea: *si el modelo dice 80 %, mostrar 75 %*. La implementamos, pero de forma que tenga sentido estadístico.

### 4.1 Qué hace y qué no hace el margen

- **Sí:** hace que el número mostrado sea **más fiable**. Si se muestra 75 %, en la práctica debería acertar **75 % o más**.
- **No:** restar puntos no aumenta por sí solo los aciertos. Lo que aumenta la tasa de acierto es **filtrar**: mostrar como "destacados" solo los mercados con probabilidad conservadora alta y confianza alta (4.4).
- Un descuento fijo (siempre −5) es insuficiente: para un partido con alineación incierta, −5 es poco; para uno muy predecible, es demasiado. Por eso el margen será **variable**, con un **mínimo fijo**.

### 4.2 Fórmula

```
margen = clip( z · σ_total + penalización_contexto ,  margen_mínimo ,  margen_máximo )
p_shown = p_calibrated − margen
```

| Componente | Cómo se obtiene |
|---|---|
| `σ_total` | Combina: (a) **desacuerdo entre modelos** del ensamble, (b) variabilidad por **re-muestreo** (*bootstrap* / semillas), (c) error de estimación del **calibrador** en ese tramo de probabilidad |
| `z` | 1.0 por defecto (≈ límite inferior de un intervalo de ~84 %); se sube a 1.28–1.65 para ser más estricto |
| `penalización_contexto` | Puntos extra por: alineación no confirmada, entrenador nuevo, pocos partidos previos, liga con poca historia, lesión de pieza clave de última hora |
| `margen_mínimo` | Piso fijo (por defecto **3 pp**; tu ejemplo de 5 pp sería `margen_mínimo = 0.05`) |
| `margen_máximo` | Tope (p. ej. 15 pp) para que la cifra no se vuelva inútil |

Ejemplo: `p_calibrated = 80 %`, `σ_total = 3 pp`, `penalización = 2 pp` ⇒ margen 5 pp ⇒ **se muestra 75 %**.

```python
import numpy as np


def ensemble_sigma(p_models: np.ndarray, calib_bin_n: np.ndarray, p_cal: np.ndarray) -> np.ndarray:
    """Incertidumbre combinada por predicción.

    p_models:    (n, k) probabilidades calibradas de k modelos/semillas para un mismo mercado.
    calib_bin_n: (n,) nº de partidos de validación en el tramo del calibrador (más datos -> menos error).
    p_cal:       (n,) probabilidad final calibrada.
    """
    s_disagree = p_models.std(axis=1)
    s_calib = np.sqrt(p_cal * (1 - p_cal) / np.maximum(calib_bin_n, 1))
    return np.sqrt(s_disagree**2 + s_calib**2)


def conservative_prob(
    p_cal: np.ndarray,
    sigma: np.ndarray,
    ctx_penalty: np.ndarray | float = 0.0,
    z: float = 1.0,
    min_margin: float = 0.03,
    max_margin: float = 0.15,
    floor: float = 0.01,
) -> np.ndarray:
    """Probabilidad mostrada = calibrada - margen variable (con piso y tope)."""
    margin = np.clip(z * sigma + ctx_penalty, min_margin, max_margin)
    return np.clip(p_cal - margin, floor, 1.0)
```

### 4.3 Cómo comprobar que el margen funciona (en el backtest)

Un margen conservador **se valida**, no se asume. Con las predicciones fuera de muestra:

1. Agrupar por tramos de `p_shown` (50–55, 55–60, …).
2. En cada tramo, la **frecuencia real de acierto debe ser ≥ `p_shown`**.
3. Criterio de aceptación: se cumple en ≥ 90 % de los tramos con suficiente muestra (≥ 30 casos), **por liga y por temporada**.
4. Si en algún tramo la frecuencia real queda por debajo de `p_shown`, el margen es insuficiente ⇒ subir `z` o `margen_mínimo`.

```python
import numpy as np


def conservativeness_report(p_shown: np.ndarray, y: np.ndarray, bins: int = 10, min_n: int = 30):
    """Por tramo: (inicio, fin, n, tasa_real, media_mostrada, ok). ok = la tasa real >= la mostrada."""
    edges = np.linspace(0, 1, bins + 1)
    idx = np.digitize(p_shown, edges[1:-1])
    rows = []
    for b in range(bins):
        m = idx == b
        if m.sum() >= min_n:
            real, shown = float(y[m].mean()), float(p_shown[m].mean())
            rows.append((edges[b], edges[b + 1], int(m.sum()), real, shown, real >= shown))
    return rows
```

### 4.4 Selección de "destacados" (donde sí sube la tasa de acierto)

Solo se marca como **destacado** un mercado si cumple todo:

- `p_shown ≥ umbral` (por defecto 70 %; configurable por mercado),
- `confidence = Alta`,
- sin avisos graves (alineación confirmada, datos suficientes).

Se publica la **curva riesgo–cobertura**: al subir el umbral, la tasa de acierto sube pero hay **menos** destacados. Es un compromiso: el sistema dirá "hoy no hay destacados" en muchos partidos, y eso es correcto.

> **Expectativa realista:** mercados como BTTS u Over 2.5 tienen una frecuencia base cercana al 50 % y mucho azar. Es de esperar una precisión global modesta (orden del 60 % o algo más) y destacados poco frecuentes. Si la validación no demuestra mejor, el sistema no debe prometerlo.

### 4.5 Combinadas

La probabilidad de varios eventos **no es el producto** de las individuales (están correlacionados: p. ej. over 2.5 y ambos anotan). Las combinadas se calculan sobre la **distribución conjunta del marcador** (simulación), y el margen se aplica al resultado conjunto.

---

## 5. Catálogo de parámetros (variables)

Todas deben respetar la regla *point-in-time*: solo información disponible **antes** del momento de la predicción. Cada grupo (A–L) se evalúa por **ablación** (con/sin el grupo).

### A. Fuerza base del equipo
| Variable | Notas |
|---|---|
| Elo propio y global (conectado entre ligas) | Diferencia local–visitante |
| Ataque/defensa Dixon-Coles | Con decaimiento temporal |
| xG a favor / en contra (EWM, varias ventanas: 3, 5, 10, 20) | Menos ruido que los goles |
| xGOT (calidad de tiros a puerta) | Si el proveedor lo ofrece |
| Goles esperados vs. reales (sobre/infra-rendimiento) | Tiende a revertir |
| Valor de mercado y edad de la plantilla | Proxy de calidad |
| Posición y puntos por partido en la tabla | Contexto de la liga |

### B. Forma reciente ("estado anímico")
| Variable | Notas |
|---|---|
| Puntos EWM (3/5/10 partidos) | Forma clásica |
| Diferencia de xG EWM | Forma sin azar |
| Racha (victorias/derrotas/sin perder/sin ganar) | Con *shrinkage* |
| Sorpresa vs. Elo (residual) | Rendir por encima/debajo de lo esperado |
| Derrota/victoria abultada reciente | Impacto emocional |
| Goles en los minutos finales (encajados/marcados) | Resiliencia |
| Eliminación o título reciente (copa/Europa) | Efecto de moral/carga |
| Cambio de entrenador (partidos desde el cambio) | "Efecto rebote" |

### C. Local / visitante
| Variable | Notas |
|---|---|
| Ventaja de campo específica del equipo | Jerárquica |
| xG, goles, tiros y córners como local / como visitante | Ventanas separadas |
| Rendimiento ante rivales de nivel alto/medio/bajo | Segmentado por Elo |
| Distancia de viaje, altitud, zona horaria | Visitante |
| Campo neutral, aficionados, estadio con césped sintético | Contexto |

### D. Enfrentamientos directos (H2H)
| Variable | Notas |
|---|---|
| Resultados, goles, BTTS, over 2.5 y córners en los últimos N H2H | Peso por antigüedad |
| Misma sede vs. cualquier sede | Ambas |
| Ajuste (*shrinkage*) hacia la expectativa del modelo | Pocas muestras |
| Misma plantilla/entrenador que en el H2H | Mide relevancia del dato |

### E. Entrenador y estilo de juego
| Variable | Notas |
|---|---|
| PPDA, intensidad de presión, *field tilt* | Estilo defensivo |
| Posesión, pases progresivos, longitud de pase | Estilo ofensivo |
| % de centros, juego por bandas, córners generados | Para mercado de córners |
| Ritmo (tiros por minuto de posesión, transiciones) | Total de goles |
| Formación predominante y variación táctica | Flexibilidad |
| Rotación (cambios de XI), uso de suplentes (minuto de cambios) | Plantillas largas |
| Arquetipo de estilo (clúster) y **choque** entre arquetipos | Interacción |
| Historial del entrenador en clubes anteriores | Prior si es nuevo |
| Rendimiento del entrenador en partidos clave / como visitante / tras derrota | Perfil |

### F. Alineación y jugadores
| Variable | Notas |
|---|---|
| XI confirmado vs. probable (`lineup_confirmed`) | Bandera de confianza |
| Calidad agregada del XI (rating, xG+xA/90, minutos) | Suma/promedio por línea (defensa, medio, ataque) |
| Ausencias clave (delantero, creador, central, portero) | Diferencia vs. XI ideal |
| Impacto individual (*on/off*: rendimiento del equipo con y sin el jugador) | Con *shrinkage* |
| Portero: PSxG − goles encajados | Calidad real del portero |
| Tiros, tiros a puerta, xG y minutos por 90 de cada jugador | Base para props |
| Rol y posición (delantero, extremo, mediapunta) y tiros de balón parado | Para props de jugador |
| Penaltis y faltas: quién los ejecuta | Goles y tiros a puerta |
| Fatiga: minutos acumulados, partidos en 7/14 días | Rotación probable |
| Sanciones y apercibidos de tarjeta | Disponibilidad |
| Regreso reciente de lesión (minutos limitados) | Minutos esperados |

### G. Balón parado y particularidades
| Variable | Notas |
|---|---|
| Córners a favor/en contra (EWM, local/visitante) | Mercado de córners |
| Goles a balón parado a favor/en contra | Defensa aérea |
| Altura media, duelos aéreos ganados | Estilo físico |
| Faltas laterales, penaltis a favor/en contra | Dependencia de pelota parada |

### H. Estructura y ritmo del partido
| Variable | Notas |
|---|---|
| Distribución de goles por tramo (1.ª/2.ª parte, minutos 75+) | Mercados por tiempos |
| Rendimiento según marcador (yendo ganando/perdiendo/empatando) | *Game state* |
| % de partidos con BTTS, over 1.5/2.5/3.5, *clean sheet*, *failed to score* | Directo para mercados de goles |
| Tiros totales y a puerta por partido, % de conversión | Ritmo y eficiencia |
| Faltas, interrupciones, tiempo efectivo | Tarjetas y ritmo |

### I. Contexto del partido
| Variable | Notas |
|---|---|
| Motivación (descenso, título, Europa, nada en juego) | Calculada por la tabla y partidos restantes |
| Días de descanso y congestión de calendario | Fatiga |
| Derbi / rivalidad | Más tarjetas y tensión |
| Jornada de la temporada (inicio, final) | Cambio de dinámica |
| Parón de selecciones, viajes de internacionales | Regreso de jugadores |
| Clima (lluvia, viento, temperatura) y hora del partido | Efecto en goles y córners |
| Estado del césped y campo | Opcional |

### J. Árbitro y disciplina
| Variable | Notas |
|---|---|
| Tarjetas, faltas y penaltis por partido del árbitro | Tarjetas y penaltis |
| Tendencia por sede (favorece al local) | Con *shrinkage* |
| Tarjetas del equipo (propias y recibidas) | Perfil de dureza |
| Uso del VAR en la liga | Cambio de régimen |

### K. Mercado (opcional, solo auditoría)
Cuotas sin margen para **comparar después** el resultado del modelo. No entran como variables.

### L. Metadatos de calidad de datos
| Variable | Uso |
|---|---|
| Nº de partidos previos del equipo/jugador/entrenador | Alimenta `σ_total` y la penalización de contexto |
| Completitud del partido (¿hay xG? ¿hay datos de jugadores?) | Bandera de confianza |
| Antigüedad del último dato | Alerta de datos viejos |

### Selección de variables

Se parte de todo el catálogo y se reduce con evidencia, nunca "a ojo":

1. **Ablación por grupo** (A–J) con walk-forward: un grupo se queda si mejora el log-loss/RPS/Brier fuera de muestra.
2. **Importancia por permutación** y valores **SHAP** (estabilidad entre pliegues y ligas).
3. Eliminar redundancia (correlación > 0.9; mantener la más estable).
4. **Regularización** (L2/L1, `min_child_samples` alto) para que más parámetros no signifiquen más sobreajuste.
5. Registro de cada decisión (variable, grupo, motivo) en el experimento.

---

## 6. Arquitectura de modelos combinados

La idea es **no apostar todo a un solo tipo de modelo**: cada capa corrige las debilidades de la otra, y todas se reconcilian en un meta-modelo.

```text
                    ┌──────────────────────────────────────────────┐
                    │  FEATURES (grupos A–L, point-in-time)        │
                    └──┬───────────┬───────────┬───────────┬───────┘
                       ▼           ▼           ▼           ▼
           ┌───────────────┐ ┌───────────┐ ┌──────────┐ ┌─────────────────┐
           │ CAPA 1        │ │ CAPA 2    │ │ CAPA 3   │ │ CAPA 4          │
           │ Generativa    │ │ Boosting  │ │ Redes    │ │ Especialistas   │
           │ Dixon-Coles / │ │ LightGBM/ │ │ GRU /    │ │ Córners, tarjetas│
           │ bayesiano     │ │ CatBoost  │ │ embeddings│ │ jugadores (NB)  │
           └──────┬────────┘ └─────┬─────┘ └────┬─────┘ └────────┬────────┘
                  └────────────────┴────┬───────┴────────────────┘
                                        ▼
                   ┌────────────────────────────────────────┐
                   │ CAPA 5: Meta-modelo (stacking)          │
                   │ pesos aprendidos con OOF temporales     │
                   └────────────────────┬───────────────────┘
                                        ▼
                   Calibración  →  Incertidumbre (σ)  →  Margen  →  p_shown
                                        ▼
                         Coherencia entre mercados + explicación
```

### 6.1 Rol de cada capa

| Capa | Modelo | Aporta | Debilidad que cubre otra capa |
|---|---|---|---|
| 1. Generativa | **Dixon-Coles / Poisson bivariante** con covariables; versión **bayesiana jerárquica** (PyMC/bambi) | Matriz completa de marcadores ⇒ 1X2, BTTS, over/under, marcador exacto **coherentes entre sí**; poca muestra bien manejada | Poco flexible con muchas variables |
| 2. Boosting | **LightGBM / CatBoost** por mercado (binario, multiclase, Poisson) | Interacciones no lineales y muchas variables | Puede sobreajustar; no garantiza coherencia entre mercados |
| 3. Redes | **GRU/Transformer** sobre la secuencia de últimos partidos + **embeddings** de equipo, jugador y entrenador | Captura dinámicas temporales y similitud de estilos | Necesita mucho dato; inestable |
| 4. Especialistas | **Binomial Negativa** (GLM / jerárquica) para córners, tarjetas, tiros; **minutos × tasa** para jugadores | Distribuciones de conteo correctas, jugadores con pocos datos | Específicos de un mercado |
| 5. Meta-modelo | **Regresión logística regularizada** con pesos no negativos (stacking) | Combina las capas; reduce varianza | Requiere predicciones fuera de muestra |

### 6.2 Qué modelos alimentan cada mercado

| Mercado | Capas que participan |
|---|---|
| 1X2, doble oportunidad | 1 + 2 + 3 |
| BTTS, over/under goles | 1 + 2 + 3 |
| Marcador exacto, tiempos | 1 (principal) |
| Córners, tarjetas, tiros | 4 + 2 (+ 1 como ritmo esperado) |
| Props de jugador | 4 (minutos × tasa) + 2 |

### 6.3 Coherencia entre mercados

Las probabilidades finales no pueden contradecirse (p. ej. `BTTS ≤ over 1.5`, `over 3.5 ≤ over 2.5`, `P(1)+P(X)+P(2)=1`). Estrategia: la **matriz de marcadores** de la capa 1 es el ancla; las capas 2–3 la **corrigen** a través del meta-modelo y después se verifica la coherencia automáticamente (con prueba unitaria). Ejemplo de cómo salen BTTS y over/under de una sola matriz:

```python
import numpy as np
from scipy.stats import poisson


def scoreline_matrix(mu_home: float, mu_away: float, rho: float = 0.0, max_goals: int = 10) -> np.ndarray:
    """P[i, j] = probabilidad de marcador i-j (Poisson + corrección Dixon-Coles de marcadores bajos)."""
    g = np.arange(max_goals + 1)
    m = np.outer(poisson.pmf(g, mu_home), poisson.pmf(g, mu_away))
    m[0, 0] *= 1 - mu_home * mu_away * rho
    m[0, 1] *= 1 + mu_home * rho
    m[1, 0] *= 1 + mu_away * rho
    m[1, 1] *= 1 - rho
    return m / m.sum()


def derive_markets(m: np.ndarray) -> dict[str, float]:
    i, j = np.indices(m.shape)
    total = i + j
    return {
        "home": float(m[i > j].sum()),
        "draw": float(m[i == j].sum()),
        "away": float(m[i < j].sum()),
        "btts": float(m[(i > 0) & (j > 0)].sum()),
        "over_1_5": float(m[total >= 2].sum()),
        "over_2_5": float(m[total >= 3].sum()),
        "over_3_5": float(m[total >= 4].sum()),
    }
```

### 6.4 Entrenamiento del meta-modelo y la calibración

1. Entrenar las capas 1–4 con walk-forward y guardar sus predicciones **fuera de muestra**.
2. Entrenar el meta-modelo **solo** con esas predicciones.
3. Calibrar (temperatura / isotónica / *beta calibration*) en un tramo temporal posterior, por mercado y, si hay datos, por liga.
4. Estimar `σ_total` (desacuerdo entre capas, semillas, error del calibrador).
5. Aplicar el margen de la sección 4.

---

## 7. Validación (enfoque estadístico, sin cuotas)

### 7.1 Líneas base a superar
1. **Frecuencia histórica por liga** (y por local/visitante).
2. **Elo ordinal**.
3. **Dixon-Coles** con decaimiento temporal.

### 7.2 Protocolo
- **Walk-forward** cronológico, con embargo de unos días entre entrenamiento y validación.
- **Test bloqueado:** última(s) temporada(s) completa(s), evaluada(s) una sola vez al final.
- **Hiperparámetros** con *nested* temporal (Optuna), nunca ajustados con el test.
- **Pruebas de fuga de datos:** *shuffle test*, *future-feature canary*, revisión de importancia sospechosa.
- Métricas **por mercado, por liga y por temporada**.

### 7.3 Métricas y criterios de aceptación (antes de publicar un mercado)

| Mercado | Métricas | Aceptación |
|---|---|---|
| 1X2 | RPS, log-loss, ECE | Mejora a las 3 bases con IC 95 % (bootstrap) que excluye 0 |
| Binarios (BTTS, over/under, SoT ≥ 1) | Brier, log-loss, ECE | Igual |
| Conteos (córners, tarjetas, tiros) | CRPS, deviance NB, MAE | Mejor que la media móvil y la media de liga |
| **Todos** | Informe de conservadurismo (4.3) | `frecuencia real ≥ p_shown` en ≥ 90 % de los tramos suficientes, por liga |
| **Todos** | Estabilidad temporal | Sin deterioro sistemático en las últimas temporadas |
| **Todos** | Curva riesgo–cobertura | Publicada y estable |

Un mercado que no cumple **no se muestra** en la interfaz (o se muestra como "informativo, baja confianza").

### 7.4 Pruebas de estrés
- ¿Qué pasa si la alineación no está confirmada? (`σ` y margen deben crecer).
- ¿Y con entrenador nuevo, ascendidos o liga con poca historia?
- Resultados por mes (inicio de temporada, final de temporada, parones internacionales).

### 7.5 Auditoría externa opcional
Cuotas históricas sin margen (normalización proporcional/Shin) como **comparación posterior**. Si el modelo queda muy por debajo, es una alerta de datos/*features*, no un criterio de diseño.

---

## 8. Interfaz estilo "Draftea"

Objetivo: lectura rápida, tarjetas claras, porcentajes grandes, mobile-first, español.

### 8.1 Pantallas

**1) Partidos del día** — lista por liga, con filtros (liga, fecha, mercado) y los destacados arriba.

```text
┌────────────────────────────────────────────────┐
│  PREMIER LEAGUE · Hoy 14:30                    │
│  Arsenal  vs  Brighton                         │
│                                                │
│  Ambos anotan        ███████░░░  67 %   ● Alta │
│  Más de 2.5 goles    ██████░░░░  58 %   ● Media│
│  Gana Arsenal        ██████░░░░  61 %   ● Media│
│                                                │
│  ⚠ Alineación no confirmada                    │
└────────────────────────────────────────────────┘
```

**2) Detalle del partido** — todos los mercados agrupados (goles, resultado, córners, tarjetas, tiros), con:

- barra de probabilidad **mostrada** (`p_shown`) y el rango (`interval`) en pequeño,
- semáforo de confianza (Alta / Media / Baja),
- **"¿Por qué?"**: 3 factores clave en lenguaje simple ("Arsenal promedia 2.1 xG como local", "Brighton encaja en 8 de sus últimos 10 visitantes"),
- alineaciones (probables o confirmadas) y bajas relevantes.

**3) Jugadores** — tarjetas por jugador: P(tiros a puerta ≥ 1), P(≥ 2), P(marca), P(titular), minutos esperados.

```text
┌──────────────────────────────────────┐
│ B. Saka          Arsenal · Extremo   │
│ Tiros a puerta ≥ 1   ████████░░ 71 % │
│ Tiros a puerta ≥ 2   ███░░░░░░░ 28 % │
│ Probabilidad de titular     94 %     │
└──────────────────────────────────────┘
```

**4) Transparencia / historial** — "¿Cuánto ha acertado el sistema?": gráfico de calibración (probabilidad mostrada vs. frecuencia real), aciertos por mercado y liga, y destacados de las últimas semanas. Es lo que construye confianza (y lo que obliga a ser honestos).

**5) Combinadas** — el usuario selecciona mercados y ve la **probabilidad conjunta real** (no el producto), con aviso de correlación.

### 8.2 Reglas de presentación
- Mostrar siempre `p_shown` como número principal; el rango y los avisos, secundarios.
- **Nunca** presentar "seguro" ni "garantizado". Texto fijo: *"Probabilidad estimada por un modelo estadístico. No garantiza resultados."*
- Mercados sin validación suficiente: etiqueta *"Informativo"*.
- Alineación sin confirmar ⇒ icono de aviso y margen mayor.
- Accesibilidad: no depender solo del color para la confianza (texto + icono).

### 8.3 Tecnología propuesta (pensada para trabajar desde terminal en Linux)

| Fase | Opción | Por qué |
|---|---|---|
| **MVP** | **Streamlit** (o NiceGUI) sobre Python | Rápido, todo en Python, ideal para validar la idea |
| **v2** | **FastAPI** (backend) + **React / Next.js** (frontend, mobile-first) | Interfaz tipo app, tarjetas, filtros, publicable |
| Datos | **PostgreSQL** local (+ Parquet/DuckDB para features) | Control total del entorno local |
| Entorno | `uv` o `venv`; contenedores con **Podman** (nativo en Fedora) | Reproducible |
| Planificador | `systemd` timers o `cron` | Ingesta y predicciones programadas, sin infraestructura pesada |
| Experimentos | **MLflow** | Registro de modelos y métricas |

Contrato de API (el frontend consume esto):

```json
{
  "match_id": 123456,
  "league": "Premier League",
  "kickoff": "2026-11-01T14:30:00Z",
  "home": "Arsenal",
  "away": "Brighton",
  "lineup_confirmed": false,
  "markets": [
    {
      "market": "btts",
      "p_calibrated": 0.72,
      "p_shown": 0.67,
      "interval": [0.62, 0.77],
      "confidence": "alta",
      "flags": ["lineup_unconfirmed"],
      "drivers": [
        "Arsenal: 2.1 xG de media como local (últimos 10)",
        "Brighton encaja en 8 de sus últimos 10 partidos como visitante",
        "Brighton sin su portero titular"
      ],
      "validated": true
    }
  ]
}
```

---

## 9. Esquema de datos (resumen)

| Tabla | Contenido | Clave |
|---|---|---|
| `matches` | Calendario, resultado, liga, temporada, árbitro, estadio | `match_id` |
| `team_match_stats` | Tiros, tiros a puerta, córners, xG, posesión, faltas, tarjetas | `match_id, team_id` |
| `player_match_stats` | Minutos, tiros, tiros a puerta, goles, asistencias, tarjetas | `match_id, player_id` |
| `lineups` | XI, suplentes, formación, entrenador y **`available_at`** | `match_id, team_id` |
| `injuries` | Bajas/sanciones con fecha de alta y de baja | `player_id, from, to` |
| `coaches` | Entrenador por equipo y rango de fechas | `team_id, from` |
| `features_snapshot` | Variables congeladas por (partido, momento de predicción) | `match_id, as_of` |
| `predictions` | **Log inmutable**: modelo, versión, `p_calibrated`, `p_shown`, `interval`, `flags`, resultado posterior | `match_id, market, model_version` |

Regla: lo que se guarda en `predictions` **nunca se sobrescribe**; así el historial de transparencia es auditable.

---

## 10. Hoja de ruta por fases (con puertas de calidad)

| Fase | Entregable | Puerta de salida (no se avanza sin cumplirla) |
|---|---|---|
| **0. Decisiones** | Ligas objetivo, plan de API, mercados prioritarios, umbrales (sección 11) | Decisiones tomadas y documentadas |
| **1. Datos** | Ingesta y esquema (sección 9), 3 temporadas completas de 2–3 ligas, controles de calidad | Datos sin duplicados ni huecos críticos; pruebas `pandera` en verde |
| **2. Bases** | Frecuencias, Elo, Dixon-Coles + marcador conjunto + mercados derivados (código de 6.3) | RPS/log-loss de las bases reproducibles; coherencia entre mercados |
| **3. Features** | Grupos A–J *point-in-time* + pruebas de fuga | Todas las pruebas de fuga pasan |
| **4. Boosting + calibración** | LightGBM por mercado, calibradores, informe de conservadurismo | Mejora a las bases; conservadurismo validado (4.3) |
| **5. Especialistas** | Córners, tarjetas, tiros (NB) y props de jugador (minutos × tasa) | Calibración por tramos en los mercados de conteo y de jugador |
| **6. Redes + meta-modelo** | GRU/embeddings, stacking, `σ_total`, margen final | El ensamble mejora a la mejor capa individual; `σ` correlaciona con el error real |
| **7. Interfaz MVP** | Streamlit con las pantallas 1–4 sobre el log de predicciones | Se entiende sin explicación; la pantalla de transparencia funciona |
| **8. Producción** | Programación diaria, reentrenamiento, monitoreo de deriva, FastAPI + React | Alertas de deriva activas; predicciones registradas antes de cada partido |
| **9. Expansión** | Más ligas, combinadas, notificaciones | Cada liga nueva pasa los criterios de la sección 7 |

---

## 11. Decisiones que necesito de ti antes de empezar

| # | Decisión | Mi propuesta por defecto |
|---|---|---|
| 1 | **Ligas iniciales** | Premier League, LaLiga y Liga MX (o las que prefieras) para empezar; luego ampliar |
| 2 | **Profundidad histórica** | 5 temporadas por liga (menos para xG si la fuente no cubre) |
| 3 | **Plan de API-Football** | Revisar el plan vigente y su límite diario; si es el gratuito, empezar con 1–2 ligas y apoyarse en football-data.co.uk para el histórico |
| 4 | **Mercados del MVP** | BTTS, over/under goles, 1X2, córners y tiros a puerta de jugador |
| 5 | **Margen mínimo** | 3 pp (o 5 pp si prefieres tu ejemplo 80 → 75) |
| 6 | **Umbral de "destacado"** | `p_shown ≥ 70 %` y confianza Alta |
| 7 | **Dónde correrá** | Local en tu Fedora al inicio; después se decide si se publica |
| 8 | **Interfaz** | Streamlit para el MVP y luego FastAPI + React |

---

## 12. Riesgos y mitigaciones

| Riesgo | Mitigación |
|---|---|
| Fuga de datos ⇒ resultados irreales en el backtest | Pruebas *canary*, `available_at`, test bloqueado |
| Sobreajuste con muchos parámetros | Regularización, ablación por grupo, validación multi-liga/temporada |
| Datos de alineaciones tardíos o ausentes | Modelo "pre-alineación", bandera `lineup_confirmed`, margen mayor |
| Datos de jugador de baja calidad o inconsistentes | Validaciones de rango, mínimo de minutos, *shrinkage* |
| Cambios de régimen (reglas, entrenadores, mercado de fichajes) | Peso por recencia, reentrenamiento periódico, monitoreo de deriva |
| Falsa sensación de seguridad con probabilidades altas | Margen, intervalos, avisos fijos, pantalla de transparencia |
| Dependencia de una sola API | Caché de respuestas crudas, fuentes alternativas, esquema propio |
| Términos de uso de APIs / scraping | Revisarlos antes de cada fuente; preferir APIs oficiales |
| Uso inadecuado (apuestas irresponsables) | Aviso visible; el sistema informa probabilidades, no recomienda apostar |

---

## 13. Definición de "terminado" del MVP

- [ ] 2–3 ligas con datos limpios y reproducibles.
- [ ] Mercados del MVP validados con los criterios de la sección 7.
- [ ] `p_shown` conservadora validada: tasa real ≥ mostrada en ≥ 90 % de los tramos suficientes.
- [ ] Interfaz con pantallas 1–4 y log de predicciones inmutable.
- [ ] Pruebas automáticas de fuga, coherencia entre mercados y calidad de datos.
- [ ] Documentación de cómo reentrenar y ejecutar (comandos de terminal).
