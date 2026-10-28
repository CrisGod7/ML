# Modelos de Machine Learning para analizar partidos de fútbol (multi-liga)

> Guía de diseño: qué predecir, con qué datos, con qué modelos y, sobre todo, **cómo validar con rigor** para reducir errores y no engañarnos a nosotros mismos.

---

## 0. Resumen ejecutivo

| Tema | Decisión recomendada |
|---|---|
| Qué predecir | Varios *targets* por separado: 1X2, goles (marcador), over/under, ambos marcan, **córners de equipo**, **tiros a puerta por jugador** |
| Modelo base | Elo + Dixon-Coles (Poisson bivariante con decaimiento temporal). Es el "listón" a superar |
| Modelo principal (tabular) | Gradient Boosting (LightGBM / CatBoost) con *features* temporales sin fuga de datos |
| Conteos (córners, tiros) | Regresión Poisson / Binomial Negativa (GLM o jerárquica bayesiana) + GBM con objetivo Poisson/Tweedie |
| Multi-liga | Efectos por liga con *partial pooling* (jerárquico) + Elo global conectado por competiciones europeas |
| Validación | **Walk-forward** (expanding window) por fecha, nunca *split* aleatorio; métricas probabilísticas (RPS, log-loss, Brier) |
| Referencia dura | Las **cuotas de cierre** de casas de apuestas (Pinnacle, por ejemplo). Si no las igualas/superas en log-loss, el modelo no aporta |
| Calibración | Obligatoria (isotónica / temperatura) y medida (ECE, curvas de fiabilidad) |
| Datos externos | API de alineaciones, lesiones, estadísticas por jugador, cuotas, clima |

**Expectativa honesta:** el fútbol tiene mucho azar y el mercado de apuestas es muy eficiente. Un buen modelo en las 5 grandes ligas suele acertar el 1X2 en el orden del **50–55 %**, muy cerca de lo que implican las cuotas. "Evitar errores" no significa "no fallar", significa **no sobreestimar la confianza**: que cuando el modelo dice 70 %, ocurra ~70 % de las veces. Por eso la calibración y la validación estricta pesan más que la complejidad del modelo.

---

## 1. Problemas a modelar (targets)

Conviene **no** meter todo en un único modelo. Cada pregunta tiene su propia distribución y su propia métrica.

| # | Target | Tipo | Distribución / modelo natural | Métrica principal |
|---|---|---|---|---|
| T1 | Resultado 1X2 | Clasificación (3 clases ordenadas) | Multinomial ordinal, o derivado de goles | RPS, log-loss |
| T2 | Goles local / visitante | Conteo | Poisson / Dixon-Coles / Poisson bivariante | Log-verosimilitud, RPS sobre marcador |
| T3 | Over/Under goles, BTTS | Binaria | Derivadas de T2 + GBM directo | Brier, log-loss |
| T4 | Córners por equipo y totales | Conteo (sobredispersión) | Binomial Negativa | Deviance NB, CRPS, MAE |
| T5 | Tiros a puerta por jugador | Conteo con muchos ceros y exposición (minutos) | Poisson/NB con *offset* de minutos, Zero-Inflated | Log-verosimilitud, CRPS |
| T6 | Tiros a puerta / tiros totales por equipo | Conteo | NB + GBM Poisson | Deviance, MAE |

> Los T4–T6 son los más aprovechables porque los mercados de córners y de jugador suelen ser menos eficientes que el 1X2, pero también los más sensibles a los datos de alineaciones (¿juega o no?) y a los minutos esperados.

---

## 2. Fuentes de datos

> Verifica siempre los **límites, planes y términos de uso vigentes** de cada proveedor antes de construir sobre ellos. Algunos prohíben el *scraping* o la redistribución.

### 2.1 APIs y datasets

| Fuente | Qué aporta | Notas |
|---|---|---|
| **API-Football** (api-sports.io) | Fixtures, **alineaciones** (`/fixtures/lineups`), estadísticas de equipo (`/fixtures/statistics`: tiros, tiros a puerta, córners, posesión), **estadísticas por jugador y partido** (`/fixtures/players`), H2H (`/fixtures/headtohead`), lesiones (`/injuries`), entrenadores (`/coachs`), cuotas (`/odds`) | Cubre muchas ligas; la profundidad histórica y los límites dependen del plan |
| **football-data.org** | Calendarios, resultados, tablas, plantillas | Plan gratuito limitado a pocas competiciones; poco detalle de estadísticas |
| **football-data.co.uk** | CSV históricos gratuitos con resultados, **tiros (HS/AS), tiros a puerta (HST/AST), córners (HC/AC)**, tarjetas, árbitro y **cuotas** (apertura y cierre en muchas temporadas) | Excelente para el histórico y para la línea base de mercado |
| **Understat** | xG por partido y por tiro (6 ligas) | Útil para forma "limpia" de azar |
| **StatsBomb Open Data** | Datos de eventos completos de competiciones seleccionadas | Gratis, ideal para extraer *estilo de juego* (PPDA, pases progresivos...) |
| **ClubElo** (clubelo.com) | Elo histórico de clubes europeos | Útil como *feature* y como baseline |
| **FBref** | Estadísticas por equipo/jugador | Su oferta de datos avanzados y sus términos de scraping han cambiado; revisa disponibilidad actual |
| **Transfermarkt** (datasets derivados) | Valor de mercado, plantillas, entrenadores, historial de bajas | Respeta sus términos; hay datasets públicos ya procesados |
| **Open-Meteo** | Clima histórico y previsión (gratis) | Lluvia/viento/temperatura en el estadio |
| **The Odds API / Pinnacle / Betfair** | Cuotas multi-casa | Para baseline, *closing line value* y *features* de mercado |
| **Sportmonks / Opta / Wyscout** (de pago) | Datos de eventos y tracking de alta calidad | Solo si el proyecto lo justifica |

### 2.2 Ejemplo mínimo: alineaciones y estadísticas desde API-Football

```python
import os
import requests

BASE = "https://v3.football.api-sports.io"
HEADERS = {"x-apisports-key": os.environ["API_FOOTBALL_KEY"]}


def api_get(endpoint: str, **params) -> list[dict]:
    r = requests.get(f"{BASE}/{endpoint}", headers=HEADERS, params=params, timeout=30)
    r.raise_for_status()
    payload = r.json()
    if payload.get("errors"):
        raise RuntimeError(payload["errors"])
    return payload["response"]


def lineups(fixture_id: int) -> list[dict]:
    """XI inicial + suplentes + formación + entrenador, por equipo."""
    return api_get("fixtures/lineups", fixture=fixture_id)


def player_stats(fixture_id: int) -> list[dict]:
    """Minutos, tiros, tiros a puerta, pases, etc. por jugador en un partido."""
    return api_get("fixtures/players", fixture=fixture_id)


def team_stats(fixture_id: int) -> list[dict]:
    """Tiros, tiros a puerta, córners, posesión... por equipo."""
    return api_get("fixtures/statistics", fixture=fixture_id)
```

Buenas prácticas de ingesta:

1. **Guardar la respuesta cruda** (JSON) con marca de tiempo antes de transformarla (`data/raw/`). Permite reproducir y depurar.
2. Cachear en disco y respetar *rate limits* (reintentos con *backoff*).
3. Guardar siempre **cuándo** se obtuvo cada dato (ver la regla *point-in-time* en la sección 6).
4. Normalizar IDs de equipos/jugadores entre proveedores en una tabla de equivalencias (`team_map`, `player_map`).

---

## 3. Ingeniería de características según los parámetros propuestos

Regla de oro: **toda *feature* de un partido en la fecha *t* solo puede usar información disponible antes de *t*** (y, para alineaciones, antes de la hora a la que realmente se conoce).

### 3.1 Enfrentamientos directos (H2H)

- Resultado, goles y córners de los últimos *N* H2H (por ejemplo 5 años), con **peso por antigüedad**.
- Mismo local/visitante vs. cualquier orden (guarda ambas versiones).
- **Cuidado:** muestras muy pequeñas (2–4 partidos) y plantillas/entrenadores que ya no existen. Úsalo con **encogimiento (*shrinkage*) bayesiano** hacia la expectativa que dan Elo/Dixon-Coles:

  `h2h_ajustado = (k · h2h_observado + m · prior_modelo) / (k + m)`, con *k* = nº de H2H y *m* = fuerza del prior (p. ej. 5–10).

- Valida con una **ablación**: si quitar H2H no empeora el log-loss fuera de muestra, no lo uses. En muchos estudios aporta poco una vez que se controla la fuerza de los equipos.

### 3.2 Estado anímico / forma reciente

El "ánimo" no se mide directamente; se aproxima con indicadores objetivos:

| Proxy | Cómo calcularlo |
|---|---|
| Forma de puntos | Puntos de los últimos 5/10 partidos con media móvil exponencial (EWM) |
| Forma "limpia de azar" | Diferencia xG a favor – xG en contra (EWM), **mejor predictor que los puntos** |
| Sorpresa vs. expectativa | Residual: resultado real – resultado esperado según Elo/mercado (rachas de sobrerrendimiento tienden a revertir) |
| Impacto emocional | Derrota abultada reciente, eliminación en copa/Europa, remontada/goles en el minuto 90+ |
| Carga física | Días de descanso, partidos en últimos 14 días, viajes europeos |
| Cambios institucionales | Entrenador nuevo (matches desde el cambio), racha de lesionados clave |
| Sentimiento en noticias (opcional) | NLP sobre prensa. Señal débil y ruidosa; solo en fases avanzadas |

```python
import pandas as pd

# df: una fila por (equipo, partido), ordenada por fecha. 'pts', 'xg_for', 'xg_against' ya calculados.
df = df.sort_values(["team", "date"])
g = df.groupby("team")

# shift(1): el partido actual NO entra en su propia feature (evita fuga de datos)
df["form_pts_ewm"] = g["pts"].transform(lambda s: s.shift(1).ewm(halflife=4, min_periods=3).mean())
df["form_xgd_ewm"] = (df["xg_for"] - df["xg_against"]).groupby(df["team"]).transform(
    lambda s: s.shift(1).ewm(halflife=5, min_periods=3).mean()
)
df["days_rest"] = g["date"].diff().dt.days
```

> Aviso: el efecto psicológico "puro" (*momentum*) es débil. Gran parte de lo que parece "estado anímico" es simplemente calidad subyacente del equipo, ya capturada por xG y Elo. Mide siempre su aporte incremental.

### 3.3 Local vs. visitante

- Ventaja de campo **específica por equipo** (algunos rinden mucho más en casa): modelarla jerárquicamente (parámetro de equipo con *shrinkage* hacia la media de la liga).
- Estadísticas separadas: xG a favor/en contra, tiros, córners **como local** y **como visitante** (ventanas móviles).
- Contexto: distancia de viaje, altitud, público (cuando existe), campo neutral (finales), partidos sin aficionados (periodo COVID: marcarlo como variable o excluirlo).

### 3.4 Estilo de juego y entrenador

El estilo se **infiere** de métricas de equipo asociadas a cada entrenador (no al club), con ventana por entrenador-temporada:

| Dimensión | Métricas |
|---|---|
| Presión / altura de línea | PPDA, recuperaciones en campo rival, *field tilt* |
| Dominio / juego de posesión | % posesión, pases por posesión, pases progresivos |
| Verticalidad | Longitud media de pase, % de balones largos, velocidad de ataque |
| Juego de banda | % de centros, córners generados por partido |
| Disciplina / dureza | Faltas, tarjetas |
| Táctica | Formación predominante, nº de cambios de XI entre partidos (rotación) |

Pasos:

1. Construir un vector de estilo por (entrenador, ventana) y **estandarizar por liga** (z-score dentro de la liga y la temporada).
2. Reducir dimensión y agrupar (PCA + *k-means* / GMM) para obtener "arquetipos" (p. ej. *posesión alta*, *contraataque*, *bloque bajo*, *presión alta*).
3. Features de **choque de estilos**: interacción entre el vector del local y el del visitante (p. ej. presión alta vs. salida de balón pobre → más pérdidas y más córners).
4. **Entrenador nuevo:** pocos partidos ⇒ usar *prior* con su estilo en clubes anteriores (encogimiento) y aumentar la incertidumbre.

### 3.5 Alineaciones (API)

- **Cuándo se conoce:** los onces oficiales suelen publicarse en torno a una hora antes del partido (comprueba lo que entrega tu API). Antes de eso solo hay **alineación probable**.
- Construir dos versiones del modelo (o un indicador `lineup_confirmed`):
  - **Pre-alineación:** XI probable = titulares más frecuentes de los últimos partidos, ajustados por lesiones/sanciones (`/injuries`).
  - **Post-alineación:** features con los jugadores reales.
- *Features* derivadas del XI:
  - Calidad agregada del once (suma/promedio de rating de jugador, valor de mercado, xG+xA por 90 de los titulares).
  - **Ausencias clave:** delantero titular, portero, centrales (diferencia vs. XI "ideal" del equipo).
  - Continuidad (cuántos titulares repiten) y formación.
  - Edad media y minutos acumulados (fatiga).

### 3.6 Tiros a puerta por jugador

Modelo en dos capas (más estable que predecir el conteo "a pelo"):

1. **Minutos esperados:** probabilidad de ser titular × minutos esperados (según rol, rotación, marcador esperado, lesiones recientes).
2. **Tasa de tiros a puerta por 90 minutos**, ajustada por rival y contexto:

   `SoT_esperados = tasa_90(jugador) × (minutos_esperados / 90) × factor_rival × factor_partido`

- `tasa_90` con *shrinkage* hacia la media de su posición (con pocos minutos, la tasa observada es ruido).
- `factor_rival`: tiros a puerta concedidos al rol/posición por la defensa rival.
- `factor_partido`: ritmo y dominio esperado (total de tiros esperados del equipo según el modelo de equipo).
- Distribución: Poisson/NB con `offset = log(minutos/90)`, o inflación en cero para jugadores con poca participación.
- Salida útil: **P(SoT ≥ 1), P(SoT ≥ 2)** y no solo la media.

### 3.7 Córners de equipo

- Córners a favor del equipo A = f(ataque de A, defensa de B, estilo de juego, ventaja de campo, marcador esperado).
- *Features*: córners a favor/en contra (EWM, local/visitante por separado), % de centros, tiros totales, posesión, estilo de ambos entrenadores, **diferencia de nivel esperada** (el equipo que va perdiendo suele generar más córners), árbitro y viento.
- Modelar **cada equipo por separado** (a favor de A, a favor de B) y sumar para el total; los córners tienen sobredispersión ⇒ **Binomial Negativa**.

### 3.8 Variables adicionales recomendadas

| Variable | Por qué |
|---|---|
| Elo de club (global, conectado entre ligas) | Fuerza base muy estable |
| xG / xGA, tiros, tiros a puerta (EWM) | Menos ruido que los goles |
| **Cuotas de apertura y cierre** | Contienen información de lesiones, dinero profesional, etc. (ver 5.3) |
| Motivación / tabla | Se juega algo (descenso, título, Europa) o no; partidos "sin nada en juego" |
| Árbitro | Faltas, penaltis y tarjetas por partido; afecta a córners y goles |
| Clima y campo | Lluvia, viento, césped sintético |
| Calendario | Derbi, fin de semana europeo, selecciones (parón internacional) |
| Valor de plantilla | Proxy de calidad para equipos con pocos datos |

---

## 4. Catálogo de modelos

### 4.1 Línea base (siempre primero)

1. **Frecuencias históricas** (local/empate/visitante por liga): el "modelo tonto". Hay que superarlo siempre.
2. **Elo** con ventaja de campo y diferencia de goles → probabilidad 1X2 (con modelo ordinal para el empate).
3. **Dixon-Coles**: goles ~ Poisson con ataque/defensa por equipo, ventaja de campo y corrección para marcadores bajos (0-0, 1-0, 0-1, 1-1), con **peso temporal exponencial** (partidos recientes pesan más).

```python
import numpy as np


def dc_tau(x: int, y: int, mu: float, lam: float, rho: float) -> float:
    """Corrección de Dixon-Coles para marcadores bajos (x = goles local, y = goles visitante)."""
    if x == 0 and y == 0:
        return 1 - mu * lam * rho
    if x == 0 and y == 1:
        return 1 + mu * rho
    if x == 1 and y == 0:
        return 1 + lam * rho
    if x == 1 and y == 1:
        return 1 - rho
    return 1.0


def time_weight(days_ago: np.ndarray, xi: float = 0.0019) -> np.ndarray:
    """Peso exponencial por antigüedad; xi se ajusta por validación (p. ej. media vida ~1 año)."""
    return np.exp(-xi * days_ago)
```

### 4.2 Modelos estadísticos interpretables

| Modelo | Uso |
|---|---|
| Poisson bivariante / Skellam | Marcadores y diferencia de goles |
| Regresión ordinal (logit ordenado) | 1X2 respetando el orden Visitante < Empate < Local |
| **GLM Poisson / Binomial Negativa** | Córners, tiros, tiros a puerta |
| **Modelos jerárquicos bayesianos** (PyMC, Stan, `bambi`) | Multi-liga con *partial pooling*, jugadores con pocos datos, incertidumbre completa (distribución posterior) |

```python
import statsmodels.formula.api as smf

# Córners a favor de un equipo en un partido (una fila por equipo-partido). 'df' ya con features sin fuga.
nb_corners = smf.negativebinomial(
    "corners_for ~ is_home + att_corners_ewm + opp_def_corners_ewm"
    " + style_press_diff + elo_diff_scaled + C(league)",
    data=df_train,
).fit(disp=False)
```

### 4.3 Gradient Boosting (modelo principal tabular)

- **LightGBM / CatBoost / XGBoost** con objetivo `multiclass` (1X2) o `poisson` / `tweedie` (conteos).
- Ventajas: manejan no linealidades, interacciones (estilo × estilo) y valores faltantes.
- Riesgos: sobreajuste con pocas filas por liga; muy sensibles a la fuga de datos. Controlar con regularización fuerte (`min_child_samples` alto, `num_leaves` pequeño, `feature_fraction`, parada temprana en validación temporal), y con **restricciones monótonas** donde tengan sentido (más Elo propio ⇒ más probabilidad de ganar).

```python
import lightgbm as lgb

params = dict(
    objective="multiclass", num_class=3, learning_rate=0.03,
    num_leaves=15, min_child_samples=80, feature_fraction=0.7,
    bagging_fraction=0.8, bagging_freq=1, lambda_l2=10.0, verbose=-1,
)
model = lgb.train(
    params,
    lgb.Dataset(X_tr, y_tr),
    num_boost_round=2000,
    valid_sets=[lgb.Dataset(X_va, y_va)],
    callbacks=[lgb.early_stopping(100, verbose=False)],
)
p_va = model.predict(X_va, num_iteration=model.best_iteration)  # (n, 3): visitante/empate/local según codificación de y
```

### 4.4 Redes neuronales (fase avanzada)

- **Embeddings de equipo/jugador/entrenador** entrenados con el resultado: capturan "calidad latente" y similitud de estilos.
- **Secuenciales (GRU/LSTM/Transformer)** sobre la secuencia de últimos partidos del equipo, en lugar de medias móviles manuales.
- **Redes de jugadores** (*set/attention* sobre el XI) para predecir tiros del equipo a partir de quién juega.
- Con datasets del tamaño de unas pocas ligas (decenas de miles de partidos), **rara vez superan a un GBM bien afinado**; úsalas como miembro del ensamble, no como sustituto.

### 4.5 Ensamble y *stacking*

- Promedio ponderado en el espacio de probabilidades (o de logits) de: Dixon-Coles + Elo-ordinal + LightGBM + (opcional) red neuronal.
- Meta-modelo (regresión logística regularizada) entrenado **solo con predicciones fuera de muestra** (*out-of-fold* temporales).
- Los ensambles heterogéneos mejoran la calibración y reducen la varianza; es la vía más barata de mejora.

### 4.6 Multi-liga

- **Efectos de liga** (intercepto de goles, ventaja de campo, ritmo) con *partial pooling*.
- **Fuerza comparable entre ligas:** Elo global alimentado por partidos de competiciones europeas (Champions, Europa League, Conference), que conectan los grafos de las ligas.
- Normalizar *features* **dentro de cada liga-temporada** (z-score) antes de mezclar.
- Ligas con pocos datos: hereditan los parámetros de ligas parecidas (jerárquico) y empiezan con mayor incertidumbre; marcar en la salida.
- Reportar métricas **por liga**: un buen promedio global puede esconder ligas donde el modelo es peor que las cuotas.

---

## 5. Entrenamiento y validación exigentes (cómo evitar errores)

Esta es la sección clave. La mayoría de los modelos de predicción deportiva "funcionan" en el cuaderno y fracasan en producción por **fuga de datos** y **mala validación**, no por falta de potencia.

### 5.1 Prevención de fuga de datos (*leakage*)

| Riesgo | Solución |
|---|---|
| Usar estadísticas del propio partido (tiros, posesión, xG del partido) como *feature* | Solo usar agregados **de partidos anteriores** (`shift(1)` antes de cualquier ventana móvil) |
| Normalizar/escalar con todo el dataset | Ajustar *scalers*, encoders e imputadores **solo con el conjunto de entrenamiento** de cada pliegue (usar `Pipeline`) |
| *Split* aleatorio | **Prohibido.** Usar partición cronológica |
| Alineación real usada para predecir antes de que se publique | Guardar la **hora de disponibilidad** de cada dato (*point-in-time*) y filtrar por `available_at <= prediction_time` |
| Cuotas de cierre como *feature* para un modelo que "predice" antes del cierre | Decidir el momento de uso: cuotas de apertura como *feature* o cuotas de cierre solo como **benchmark** |
| Información posterior en fichajes/entrenadores | Usar la plantilla y el entrenador vigentes **a la fecha del partido** |
| Rankings/Elo calculados con datos futuros | Recalcular Elo de forma incremental, partido a partido |
| Búsqueda de hiperparámetros con el conjunto de test | *Nested* temporal: ajuste interno ≠ evaluación externa |

**Prueba de cordura obligatoria (*canary tests*):**
- *Shuffle test:* barajar las etiquetas ⇒ el modelo debe caer al nivel del azar. Si no, hay fuga.
- *Future-feature test:* añadir a propósito una variable que "mira al futuro" y comprobar que el pipeline la detecta/rechaza.
- Revisar la importancia de variables: si una *feature* sospechosamente domina, investigar.

### 5.2 Esquema de validación: *walk-forward* (expanding window)

```text
Temporadas:   T1   T2   T3   T4   T5   T6   T7   T8
Pliegue 1:   [train ][val]
Pliegue 2:   [train      ][val]
Pliegue 3:   [train           ][val]
...
TEST FINAL:  [train (todo lo anterior)         ][ TEST BLOQUEADO ]   <- se toca una sola vez
```

```python
import numpy as np
import pandas as pd


def walk_forward(df: pd.DataFrame, first_val_start: str, step_days: int = 30, embargo_days: int = 2):
    """Genera (idx_train, idx_val) cronológicos. El embargo evita solapes entre entrenamiento y validación."""
    df = df.sort_values("date")
    start = pd.Timestamp(first_val_start)
    end = df["date"].max()
    while start <= end:
        stop = start + pd.Timedelta(days=step_days)
        train_idx = df.index[df["date"] < start - pd.Timedelta(days=embargo_days)]
        val_idx = df.index[(df["date"] >= start) & (df["date"] < stop)]
        if len(val_idx):
            yield train_idx, val_idx
        start = stop
```

- **Test final bloqueado:** la(s) última(s) temporada(s) completa(s) se reservan y se evalúan **una única vez**, al final. Si se vuelve a mirar y se retoca el modelo, deja de ser test.
- Evaluar en **varias temporadas y varias ligas** para comprobar estabilidad, no un solo periodo afortunado.
- Reentrenar de forma periódica (semanal/mensual) como en producción, no una sola vez.

### 5.3 Métricas

| Métrica | Para qué | Notas |
|---|---|---|
| **RPS** (*Ranked Probability Score*) | 1X2 ordenado | La métrica estándar en predicción de fútbol; menor es mejor |
| **Log-loss** | Probabilidades | Penaliza fuerte la sobreconfianza |
| **Brier score** | Binarias (BTTS, over/under, SoT ≥ 1) | Descomponible en calibración + resolución |
| **ECE** y curvas de fiabilidad | Calibración | Medir por liga y por rango de probabilidad |
| **CRPS / deviance NB** | Conteos (córners, tiros) | Evalúa toda la distribución, no solo la media |
| MAE / RMSE | Referencia secundaria | Solo informativas |
| Accuracy | Solo informativa | No sirve sola: ignora la confianza |
| **CLV** (*closing line value*) | Si algún día se usa para apostar | Comparar con la cuota de cierre; es el mejor indicador de ventaja real |

```python
import numpy as np


def rps(probs: np.ndarray, outcome: np.ndarray) -> float:
    """RPS medio. probs: (n, 3) en orden [local, empate, visitante]; outcome: índices 0/1/2."""
    cum_p = np.cumsum(probs, axis=1)
    cum_o = np.cumsum(np.eye(3)[outcome], axis=1)
    return float(np.mean(np.sum((cum_p - cum_o) ** 2, axis=1) / 2))


def log_loss_multi(probs: np.ndarray, outcome: np.ndarray, eps: float = 1e-12) -> float:
    p = np.clip(probs[np.arange(len(outcome)), outcome], eps, 1.0)
    return float(-np.mean(np.log(p)))
```

**Referencia de mercado:** convertir las cuotas de cierre en probabilidades (quitando el margen de la casa, p. ej. por normalización proporcional o método de Shin) y calcular su RPS/log-loss en **el mismo conjunto de partidos**. Ese es el techo realista.

### 5.4 Criterios de aceptación ("ser exigentes")

Un modelo **no se publica** hasta cumplir, en el **test bloqueado**:

1. Mejora estadísticamente a las tres bases: frecuencias, Elo y Dixon-Coles (intervalo de confianza *bootstrap* al 95 % sobre la diferencia de log-loss/RPS por partido que **no incluya 0**; opcionalmente test de Diebold-Mariano).
2. Se mantiene **dentro de un margen pequeño de las cuotas de cierre** (o las supera); si queda claramente por debajo, sirve como herramienta de análisis, no como fuente de decisiones.
3. **Calibración:** ECE bajo (p. ej. < 2–3 %) y curva de fiabilidad cercana a la diagonal, **en cada liga relevante** y por tramos de probabilidad.
4. **Estabilidad temporal:** no depender de una sola temporada; la métrica no empeora sistemáticamente en las últimas.
5. Pasa los *canary tests* de fuga de datos (5.1).
6. Cada nueva *feature* demuestra su valor con una **ablación** (con/sin) fuera de muestra.
7. Registro de experimentos reproducible: semilla, versión de datos, *commit*, parámetros, métricas (MLflow o similar).

### 5.5 Calibración

```python
import numpy as np
from sklearn.isotonic import IsotonicRegression


class MulticlassIsotonic:
    """Calibración isotónica uno-contra-todos + renormalización. Ajustar con predicciones fuera de muestra."""

    def fit(self, probs: np.ndarray, y: np.ndarray):
        self.models_ = [
            IsotonicRegression(out_of_bounds="clip").fit(probs[:, k], (y == k).astype(float))
            for k in range(probs.shape[1])
        ]
        return self

    def transform(self, probs: np.ndarray) -> np.ndarray:
        cal = np.column_stack([m.predict(probs[:, k]) for k, m in enumerate(self.models_)])
        cal = np.clip(cal, 1e-6, None)
        return cal / cal.sum(axis=1, keepdims=True)
```

- Ajustar el calibrador en un tramo temporal **posterior** al del entrenamiento del modelo y **anterior** al test.
- Con pocos datos preferir **escalado de temperatura** (un solo parámetro) o Platt, que la isotónica sobreajusta con muestras pequeñas.
- Recalibrar periódicamente y vigilar la deriva (*drift*).

### 5.6 Incertidumbre y abstención

- Mostrar siempre **probabilidades**, no solo la clase ganadora.
- **Conformal prediction:** conjuntos de predicción con cobertura garantizada (p. ej. "{Local, Empate}" al 80 %); si el conjunto contiene las tres clases, el partido es **no predecible** con confianza.
- **Política de abstención:** no emitir recomendaciones cuando la alineación no está confirmada, el entrenador es nuevo, faltan datos de la liga o la discrepancia con el mercado es enorme sin explicación (suele ser un dato malo, no una ventaja).
- Para jugadores: informar P(titular) además de P(SoT ≥ 1).

### 5.7 Otros cuidados

- **Datos mínimos:** exigir un número mínimo de partidos/minutos previos para emitir una predicción (equipo, jugador, entrenador).
- **Comparaciones múltiples:** si pruebas 50 combinaciones de *features*/modelos, alguna ganará por azar. Contarlas y corregir (o usar el test bloqueado solo para el finalista).
- **Cambios de régimen:** mercado de fichajes, nuevo entrenador, reglas (VAR, 5 cambios, tiempo añadido) ⇒ reponderar por recencia.
- **Calidad de datos:** validar rangos (p. ej. tiros a puerta ≤ tiros), duplicados, partidos aplazados, equipos renombrados; pruebas automáticas (Great Expectations o `pandera`).
- **Monitorización en producción:** registrar cada predicción **antes** del partido y su resultado después; panel con RPS/ECE móviles y alarmas de deriva.

---

## 6. Arquitectura del pipeline

```text
                       ┌─────────────┐
 APIs (partidos,       │  INGESTA    │  data/raw (JSON, con timestamp)
 alineaciones, cuotas, ─▶  + caché   │
 lesiones, clima)      └──────┬──────┘
                              ▼
                       ┌─────────────┐
                       │  LIMPIEZA   │  data/interim (IDs unificados, validaciones)
                       └──────┬──────┘
                              ▼
                       ┌─────────────┐
                       │  FEATURES   │  store *point-in-time* (available_at)
                       │ (sin fuga)  │  Elo, forma, estilo, XI, H2H, mercado
                       └──────┬──────┘
              ┌───────────────┼────────────────┐
              ▼               ▼                ▼
        Modelo 1X2 /     Modelo córners   Modelo jugador
        goles (DC+GBM)   (NB + GBM)       (minutos × tasa)
              └───────────────┼────────────────┘
                              ▼
                  Ensamble + calibración + conformal
                              ▼
              Evaluación walk-forward + comparación con mercado
                              ▼
                 Predicciones registradas (log inmutable)
```

Estructura de carpetas sugerida:

```text
ml/
├── configs/            # YAML: ligas, ventanas, hiperparámetros
├── data/
│   ├── raw/            # respuestas crudas de la API (no se editan)
│   ├── interim/
│   └── processed/
├── src/
│   ├── ingest/         # clientes de API + caché
│   ├── features/       # elo.py, form.py, style.py, lineup.py, h2h.py, market.py
│   ├── models/         # dixon_coles.py, gbm.py, count_models.py, player_sot.py, ensemble.py
│   ├── eval/           # walk_forward.py, metrics.py, calibration.py, leakage_tests.py
│   └── serve/          # predicción previa al partido + registro
├── notebooks/
├── tests/              # tests unitarios, incluidos canary de fuga
└── MODELOS_ML_FUTBOL.md
```

Herramientas recomendadas: `pandas`/`polars`, `scikit-learn`, `lightgbm`/`catboost`, `statsmodels`, `PyMC`/`bambi`, `optuna` (con validación temporal), `mlflow`, `pandera`, `pytest`.

---

## 7. Hoja de ruta por fases

| Fase | Entregable | Criterio de salida |
|---|---|---|
| **1. Datos y base** | Ingesta de 2–3 ligas + football-data.co.uk, Elo y Dixon-Coles, evaluación walk-forward, baseline de mercado | RPS/log-loss de la base y del mercado reproducibles |
| **2. Features de equipo** | Forma (xG), local/visitante, descanso, H2H con *shrinkage*, mercado de apertura | Mejora vs. Dixon-Coles fuera de muestra (con ablaciones) |
| **3. GBM + calibración** | LightGBM 1X2 + calibrador + métricas por liga | Cumple los criterios de la sección 5.4 en el test bloqueado |
| **4. Córners y tiros de equipo** | Modelo NB/GBM Poisson con estilo de entrenador | Mejor CRPS que la media móvil simple y que la media de liga |
| **5. Alineaciones y jugadores** | Integración de `/fixtures/lineups`, `/injuries`; modelo minutos × tasa para SoT | Calibración de P(SoT ≥ 1) por rango de probabilidad |
| **6. Multi-liga y ensamble** | Jerárquico, Elo global, *stacking*, conformal | Estabilidad en todas las ligas objetivo |
| **7. Producción** | Registro inmutable de predicciones, panel de monitorización, reentrenamiento programado | Alarmas de deriva funcionando |

---

## 8. Advertencias finales

- Ningún modelo elimina el azar; el objetivo es **probabilidades bien calibradas**, no certezas.
- Si el uso final son apuestas, el único indicador fiable de ventaja real es superar de forma sostenida a la **cuota de cierre** (CLV) con una muestra grande; y conviene practicar juego responsable y gestionar el riesgo (p. ej. Kelly fraccional) y la legalidad local.
- Respetar los **términos de servicio** de cada API/sitio (límites, redistribución, *scraping*).
- Documentar cada experimento: lo que no funcionó es tan valioso como lo que sí.

---

## 9. Referencias para profundizar

- Dixon, M. & Coles, S. (1997). *Modelling Association Football Scores and Inefficiencies in the Football Betting Market*.
- Constantinou, A. & Fenton, N. (2012). *Solving the problem of inadequate scoring rules for assessing probabilistic football forecast models* (RPS).
- Hvattum, L. & Arntzen, H. (2010). *Using ELO ratings for match result prediction in association football*.
- Baio, G. & Blangiardo, M. (2010). *Bayesian hierarchical model for the prediction of football results*.
- Dubitzky, W. et al. (2019). *Soccer Prediction Challenge* (Machine Learning, número especial).
- Angelopoulos, A. & Bates, S. (2021). *A Gentle Introduction to Conformal Prediction*.
