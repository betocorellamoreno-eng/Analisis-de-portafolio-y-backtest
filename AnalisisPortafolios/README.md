# Análisis y backtesting de portafolios de inversión

**Autor: Alberto José Corella Moreno**

Este proyecto estudia cómo cambia el comportamiento de una cartera diversificada cuando se modifica su regla de rebalanceo. Compara crecimiento, caídas, costos y rendimiento ajustado por riesgo, manteniendo constantes los activos y los pesos objetivo. El propósito es entender las consecuencias de cada regla y la estabilidad de sus resultados históricos.

El análisis completo, las tablas, las gráficas y el motor de simulación están en [Análisis y backtesting de portafolios.ipynb](<Análisis y backtesting de portafolios.ipynb>). Las ocho investigaciones y sus conclusiones están terminadas. Las cifras de este documento corresponden a los datos y parámetros guardados; cambiar la muestra, los costos o las reglas requiere recalcularlas.

## Datos y alcance

La muestra abarca del **4 de enero de 2010 al 31 de diciembre de 2025**: 4,024 observaciones de precios ajustados y 4,023 rendimientos diarios por activo. Todo se expresa en USD nominales.

| Instrumento | Exposición representada | Peso objetivo |
|---|---|---:|
| SPY | Acciones estadounidenses del S&P 500 | 25% |
| IEF | Bonos del Tesoro estadounidense de 7 a 10 años | 25% |
| GLD | Oro | 25% |
| VNQ | Renta variable inmobiliaria estadounidense, incluidos REIT | 25% |

Los precios proceden de Yahoo Finance, descargados mediante [yfinance](https://github.com/ranaroussi/yfinance). Se utiliza `Adj Close`, con los ajustes del proveedor por distribuciones y eventos corporativos; no se agregan dividendos por separado. Los archivos locales conservan la muestra utilizada. La carga del notebook verifica fechas ordenadas y únicas, valores válidos y coincidencia entre rendimientos guardados y calculados.

La referencia libre de riesgo es la columna diaria **RF** de la [biblioteca de Kenneth R. French](https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/data_library.html). Su [documentación metodológica](https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/Data_Library/f-f_factors.html) identifica la referencia de un mes de Ibbotson hasta mayo de 2024 y el índice ICE BofA US 1-Month Treasury Bill desde junio de 2024. La copia archivada corresponde a la versión CRSP 202608. Se emplea para evaluar resultados históricos y no interviene en las señales de inversión.

## Diseño del estudio

| Sección | Experimento y finalidad |
|---|---|
| 1. Análisis exploratorio | Rendimientos, volatilidad, drawdowns y correlaciones de los cuatro activos. |
| 2. Portafolio mensual | Construcción del motor, financiación de costos y comparación con comprar y mantener SPY. |
| 3. Frecuencias de rebalanceo | Comparación entre no rebalancear y operar mensual, trimestral o anualmente. |
| 4. Subperiodos y costos | Repetición en cuatro bloques de cuatro años y bajo distintas tasas de costo. |
| 5. Ventanas móviles | Sensibilidad a la fecha de entrada: 145 ventanas nominales de 48 meses, desplazadas un mes. |
| 6. Bandas | Rebalanceo por desviaciones de ±2.5, ±5 y ±10 puntos porcentuales; actividad, deriva y comparación con la regla mensual. |
| 7. Ejecución diferida | Separación de la señal al cierre y la operación al cierre de la siguiente sesión. |
| 8. Riesgo ajustado por RF | Sharpe y Sortino con referencia histórica diaria, en la muestra completa y por subperiodo. |

Cada bloque y ventana reinicia la cartera con el mismo capital y pesos. Los bloques son 2010–2013, 2014–2017, 2018–2021 y 2022–2025. Las ventanas se solapan: sus resultados describen sensibilidad al punto de entrada, pero no constituyen 145 observaciones independientes ni una validación fuera de muestra.

## Supuestos de simulación

- **Capital inicial:** USD 10,000, antes del costo de compra. Sin aportaciones, retiros, apalancamiento ni posiciones cortas.
- **Asignación:** 25% por activo al inicio y después de cada rebalanceo. Se permiten participaciones fraccionarias y se invierte el capital disponible después de costos.
- **Costo activo:** 10 puntos base, equivalentes a 0.10%, sobre el importe de compras más ventas. Incluye la compra inicial y se paga con recursos de la cartera. No es un cargo fijo sobre todo el patrimonio en cada fecha.
- **Calendario:** las reglas periódicas identifican el último cierre disponible de cada periodo. No se rebalancea en la fecha inicial ni en la final, ni se aplica una liquidación terminal.
- **Bandas:** se activa una orden cuando al menos un peso supera el umbral absoluto respecto de su objetivo. Por ejemplo, ±5 puntos porcentuales corresponde al intervalo de 20% a 30% alrededor de un objetivo de 25%. La ejecución restablece todos los pesos objetivo.
- **Referencia:** SPY se compra y mantiene, con el mismo capital y costo de entrada. Su exposición es distinta de la cartera diversificada.

Las secciones 2 a 6 utilizan **mismo cierre**: la regla observa y rebalancea con el cierre de la fecha de señal. La sección 7 contrasta ese supuesto con **siguiente cierre**: se conserva la cartera anterior durante todo el intervalo entre cierres y se ejecuta en la próxima sesión disponible. La compra inicial no se retrasa. Una orden por bandas pendiente se conserva aunque los pesos vuelvan al intervalo; una señal en la penúltima sesión se registra como omitida porque no se opera en la fecha final.

El escenario de siguiente cierre es la referencia principal para la síntesis final. Distingue señal y operación, pero continúa usando precios ajustados y posiciones fraccionarias: no reproduce cotizaciones ni órdenes reales.

## Resultados principales

**Muestra completa, ejecución al siguiente cierre y costo de 10 puntos base.** CAGR y máxima caída incluyen costos. Sharpe y Sortino están anualizados y usan RF histórica; son ratios sin unidades.

| Estrategia | CAGR | Máxima caída | Sharpe | Sortino |
|---|---:|---:|---:|---:|
| Sin rebalanceo | 9.42% | −25.95% | 0.712 | 0.991 |
| Mensual | 8.98% | −21.03% | 0.778 | 1.094 |
| Trimestral | 9.03% | −21.19% | 0.784 | 1.104 |
| Anual | 8.96% | −20.96% | 0.785 | 1.102 |
| Banda ±2.5 pp | 9.02% | −21.22% | 0.775 | 1.091 |
| Banda ±5 pp | 9.21% | −21.36% | 0.790 | 1.116 |
| Banda ±10 pp | 9.18% | −21.19% | 0.783 | 1.096 |
| SPY | 13.90% | −33.72% | 0.764 | 1.073 |

SPY registra el mayor crecimiento, acompañado de una caída más profunda. Entre las carteras con cuatro activos, no rebalancear obtiene el mayor CAGR, pero deja que los pesos se alejen del objetivo y presenta peores ratios que las reglas de rebalanceo. Por tanto, el rendimiento adicional debe interpretarse junto con la exposición asumida.

La banda de ±5 puntos porcentuales alcanza el mayor Sharpe y Sortino de esta muestra, con **12 rebalanceos frente a 191** del calendario mensual. Su CAGR supera al mensual, aunque su máxima caída es ligeramente más profunda. La ventaja en los ratios es pequeña: +0.012 en Sharpe y +0.022 en Sortino. Además, supera al mensual en ambos ratios solamente en **2 de los 4 subperiodos**. Estos resultados respaldan su interés como regla para controlar pesos con menor actividad, pero no demuestran superioridad estadística ni una ventaja persistente.

Retrasar la ejecución modifica también los resultados: el CAGR mensual pasa de 8.88% a 8.98% y el de la banda de ±5 pp de 9.15% a 9.21%. Entre reglas y subperiodos, el efecto sobre CAGR varía de −0.152 a +0.300 puntos porcentuales anuales. El retraso puede favorecer o perjudicar según la trayectoria de los precios; no equivale a un costo constante.

La conclusión del proyecto es que rebalancear permite administrar la deriva de las exposiciones y cambia el equilibrio entre crecimiento, caídas y actividad. Ninguna regla domina todos los criterios y periodos. La elección de una política requiere definir primero qué exposición y nivel de riesgo se pretende mantener; el mayor rendimiento o ratio histórico, por sí solo, no resuelve esa decisión.

## Convenciones de las métricas

El CAGR usa la duración efectiva de la inversión en días divididos entre 365.25. La volatilidad anualiza la desviación estándar muestral con la raíz de 252. El drawdown se calcula respecto del máximo de patrimonio alcanzado, incluyendo el capital previo al costo de entrada como referencia inicial.

Sharpe utiliza la media del exceso diario sobre RF dividida entre su desviación estándar muestral. Sortino utiliza el mismo exceso medio y la raíz de la media de los cuadrados de sus valores negativos, contando todos los intervalos en el denominador. Ambos se multiplican por la raíz de 252. Las referencias metodológicas son [Sharpe (1994)](https://web.stanford.edu/~wfsharpe/art/sr/sr.htm) y [Rollinger y Hoffman, documento alojado por CME](https://www.cmegroup.com/content/dam/cmegroup/education/files/sortino-a-sharper-ratio.pdf).

RF se publica en porcentaje diario por sesión: se divide entre 100 una sola vez, sin volver a dividir entre 252 ni agregar interés de fin de semana. Se exige coincidencia exacta de fechas. En Sharpe y Sortino, el costo inicial se incorpora al primer intervalo real de mercado; el producto de los rendimientos netos reproduce el patrimonio final respecto del capital inicial. La volatilidad de las secciones previas excluye ese costo puntual de entrada. Los ratios con denominador nulo se muestran como no definidos.

## Estructura del proyecto

```text
AnalisisPortafolios/
├── Análisis y backtesting de portafolios.ipynb
├── README.md
├── data/
│   ├── historico_descargado.csv
│   ├── precios_ajustados.csv
│   ├── rendimientos_diarios.csv
│   ├── F-F_Research_Data_Factors_daily_CSV.zip
│   ├── rf_diario_usd.csv
│   └── rf_diario_usd_metadata.json
├── scripts/
│   └── prepare_risk_free.py
└── tests/
    ├── test_backtest.py
    ├── test_bands.py
    ├── test_execution_delay.py
    ├── test_risk_metrics.py
    ├── test_robustness.py
    └── test_rolling.py
```

El notebook contiene el código del motor. Las pruebas cargan directamente sus funciones para comprobar la implementación utilizada en el análisis. Los metadatos de RF registran fuente, versión, unidades, cobertura y huellas SHA-256 del ZIP y del CSV.

## Reproducción

Abra PowerShell en la raíz del proyecto. Si ya dispone del entorno utilizado en el análisis:

```powershell
conda activate portfolio-analysis
python -m jupyterlab
```

Para crear un entorno separado con las versiones principales inspeccionadas en el proyecto:

```powershell
conda create -n portfolio-analysis-repro python=3.12.14 pip
conda activate portfolio-analysis-repro
python -m pip install numpy==2.5.2 pandas==3.0.6 matplotlib==3.11.2 ipython==9.17.1 jinja2==3.1.6 jupyterlab==4.6.4 ipykernel==7.3.0 yfinance==1.7.0
python -m ipykernel install --user --name portfolio-analysis-repro --display-name "Python (portfolio-analysis-repro)"
python -m jupyterlab
```

Estas son las versiones observadas al documentar el proyecto. La instalación propuesta en un entorno nuevo no se ha ejecutado durante esta revisión; no se incluye un archivo de bloqueo de todas las dependencias transitivas.

1. Abra el notebook y seleccione el kernel del entorno correspondiente.
2. Para conservar la muestra guardada, **omita la primera celda de código**, que descarga precios y sobrescribe los tres CSV de activos.
3. Ejecute desde la celda de carga y validación de CSV de la sección 1, identificable por `carpeta = Path("data")` y las llamadas a `pd.read_csv`.
4. Continúe con todas las celdas posteriores en orden hasta el cierre. Esta ruta utiliza datos locales y no requiere descargar precios ni RF.

Los parámetros principales están en las secciones 2, 4, 5 y 6: `CAPITAL_INICIAL`, `COSTO_BPS`, `PESOS_OBJETIVO`, `SUBPERIODOS`, `COSTOS_PRUEBA_BPS`, `MESES_VENTANA`, `PASO_MESES`, `BANDAS_PRUEBA_PP` y `BANDA_PRINCIPAL_PP`. Después de modificarlos, ejecute las secciones dependientes en orden. Las cifras narradas en Markdown y en este README describen la configuración guardada y requieren revisión si se cambian esos parámetros.

### Reconstrucción de RF

El CSV de RF ya está preparado. Para reconstruirlo desde el ZIP archivado y regenerar sus metadatos:

```powershell
python scripts/prepare_risk_free.py
```

Si el ZIP existe, el script lo reutiliza. Si falta, lo descarga. La opción siguiente fuerza una nueva descarga y sustituye la copia local:

```powershell
python scripts/prepare_risk_free.py --refresh
```

Una actualización puede cambiar la versión histórica del proveedor y los resultados. Para reproducir las cifras documentadas, conserve los archivos actuales. El notebook valida la huella del CSV contra sus metadatos antes de calcular los ratios.

### Pruebas

Desde el entorno activado y la raíz del proyecto, ejecute los seis archivos de prueba individualmente:

```powershell
$pruebas = Get-ChildItem -LiteralPath ".\tests" -Filter "test_*.py" | Sort-Object Name
foreach ($prueba in $pruebas) {
    python $prueba.FullName
    if ($LASTEXITCODE -ne 0) { throw "Falló la prueba: $($prueba.Name)" }
}
```

La suite contiene **51 pruebas** de contabilidad, costos, calendarios, ventanas, bandas, causalidad de la ejecución diferida, alineación de RF y fórmulas de riesgo. No necesita conexión a internet. Se ejecuta cada archivo directamente porque sus cargadores reservan el primer argumento para una ruta alternativa al notebook.

## Límites de interpretación

- El universo de cuatro instrumentos es fijo y fue definido para este estudio. No se investiga un universo histórico completo ni se corrigen sesgos de selección de activos.
- Todo el historial se utiliza para describir y comparar reglas. Los subperiodos y ventanas no son una evaluación fuera de muestra; no se estiman significancias, intervalos de confianza ni correcciones por comparaciones múltiples.
- Los precios ajustados y RF son copias retrospectivas que pueden incorporar revisiones del proveedor. No se dispone de una reconstrucción de la información tal como se conocía en cada fecha.
- La simulación omite spreads, deslizamiento, liquidez, impacto de mercado, lotes enteros y restricciones operativas. El costo proporcional es una simplificación y el cierre ajustado no es una cotización de ejecución.
- Los resultados no incluyen impuestos, inflación ni conversión a MXN. No representan el rendimiento neto específico de una persona residente en México.
- RF es una referencia teórica sin costos de implementación y con redondeo diario. La anualización de ratios es convencional y su interpretación depende de la distribución y dependencia temporal de los rendimientos.

El proyecto queda cerrado como estudio histórico reproducible. Sus resultados permiten comparar políticas de asignación bajo supuestos explícitos; no constituyen una previsión de rendimientos ni una recomendación personalizada de inversión.
