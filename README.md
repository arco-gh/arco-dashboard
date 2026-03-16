# Dashboard Predictivo ARCO — Módulo 3

Sistema de predicción de afluencia y eficiencia operativa para Plaza San Isidro, Culiacán, Sinaloa.

**Maestría en Analítica e Inteligencia de Negocios · UDEM · 2025**
Autor: Emanuel Delgadillo Hernández | Director: Dr. Rafael Cruz Reyes

---

## Archivos necesarios

```
modulo3_dashboard/
├── app.py                                    ← aplicación principal
├── requirements.txt                          ← dependencias
├── README.md                                 ← este archivo
└── Informacion-PSI-23-24-25-detallado.xlsx   ← datos (agregar manualmente)
```

> ⚠️ El archivo Excel de datos NO se incluye en el repositorio por privacidad.
> Debes subirlo manualmente al repositorio antes de desplegar.

---

## Ejecución local

```bash
# 1. Instalar dependencias
pip install -r requirements.txt

# 2. Colocar el archivo Excel en la misma carpeta que app.py

# 3. Ejecutar
streamlit run app.py
```

El dashboard se abre automáticamente en http://localhost:8501

---

## Despliegue en Streamlit Community Cloud (gratuito)

### Paso 1 — Crear repositorio en GitHub
1. Ve a github.com y crea una cuenta si no tienes
2. Crea un repositorio nuevo (puede ser privado): `arco-dashboard`
3. Sube estos archivos: `app.py`, `requirements.txt`, el archivo `.xlsx`

### Paso 2 — Conectar con Streamlit Community Cloud
1. Ve a share.streamlit.io
2. Inicia sesión con tu cuenta de GitHub
3. Haz clic en "New app"
4. Selecciona tu repositorio `arco-dashboard`
5. En "Main file path" escribe: `app.py`
6. Haz clic en "Deploy"

### Paso 3 — Obtener la URL pública
Streamlit genera una URL del tipo:
`https://[tu-usuario]-arco-dashboard-app-[hash].streamlit.app`

Esa URL es la que compartes con ARCO y presentas en la defensa de tesis.

---

## Vistas del dashboard

### 🏢 Panel Operativo (para Operaciones ARCO)
- Selecciona el mes y año en el menú lateral
- Obtén en tiempo real:
  - Afluencia esperada de visitantes
  - Gasto operativo proyectado vs promedio histórico y presupuesto
  - Ratio gasto/ingresos vs meta del 10%
  - Ahorro estimado vs promedio histórico
  - Consumo eléctrico (kWh) y agua (m³) proyectados
  - Gráficas de tendencia histórica 2023–2025

### 🎓 Panel Académico (para defensa de tesis)
- Validación formal de H1 y H2
- Desempeño de los 4 modelos (MAPE, precisión, R²)
- Gráficas de análisis: ratio por año, afluencia real vs predicha,
  gasto real vs predicho, diagrama de identidad
- Hallazgos clave y aportaciones del estudio
- Referencias bibliográficas

---

## Fundamento metodológico

| Modelo | Variable | Algoritmo | Precisión |
|--------|----------|-----------|-----------|
| A | Afluencia mensual | Regresión lineal + estacionalidad | ~89% |
| B | Gasto operativo | Ridge Regression (L2) | ~90% |
| C | Consumo eléctrico | Regresión lineal múltiple | Referencial |
| D | Consumo de agua | Regresión lineal múltiple | Referencial |

Referencias: Amangeldy et al. (2025) · Gu et al. (2023) · Ozdemir et al. (2022) ·
Taylor & Letham (2018) · Tibshirani (1996)
