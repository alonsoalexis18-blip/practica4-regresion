import streamlit as st
import numpy as np
import pandas as pd
import duckdb
import statsmodels.api as sm
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_squared_error
import scipy.stats as stats
import matplotlib.pyplot as plt
import seaborn as sns

# Configuración de la página web
st.set_page_config(page_title="Práctica 4.2 - Regresión Lineal", layout="wide")

st.title("💧 Práctica 4.2: Regresión Lineal y Correlación")
st.markdown("**Asignatura:** Análisis y Exploración de Datos | **Caso de Estudio:** Consumo Per Cápita de Agua Potable")

# ==========================================
# 1. SIMULACIÓN DE DATOS (DuckDB con semilla 0.42)
# ==========================================
@st.cache_data
def cargar_datos():
    con = duckdb.connect(database=':memory:')
    con.execute("SELECT setseed(0.42);")
    con.execute("""
    CREATE TABLE df_agua AS 
    SELECT 
        range AS id_residencia,
        ROUND(15 + (RANDOM() * 20), 2) AS temperatura_max,
        ROUND(30 + (RANDOM() * 50), 2) AS humedad_max,
        ROUND(1 + (RANDOM() * 6), 1) AS personas_familia,
        ROUND(0.5 + (RANDOM() * 2), 2) AS consumo_bidon_pc,
        ROUND(
            90.0 
            + (2.5 * (15 + (RANDOM() * 20))) 
            - (0.8 * (30 + (RANDOM() * 50))) 
            - (5.0 * (1 + (RANDOM() * 6))) 
            + (14.5 * (0.5 + (RANDOM() * 2)))
            + (RANDOM() * 3), 2
        ) AS cpc_consumo
    FROM range(1, 151);
    """)
    return con.execute("SELECT * FROM df_agua;").fetchdf()

df = cargar_datos()

# Mostrar datos en la web
st.subheader("1. Primeros registros de la base simulada (DuckDB)")
st.dataframe(df.head())

# ==========================================
# 2. MATRIZ DE CORRELACIÓN DE PEARSON
# ==========================================
df_analisis = df.drop(columns=['id_residencia'])
matriz_correlacion = df_analisis.corr(method='pearson')

st.subheader("2. Matriz de Correlación de Pearson")
st.write(matriz_correlacion)

# ==========================================
# 3. MODELO OLS Y MÉTRICAS
# ==========================================
X = df[['temperatura_max', 'humedad_max', 'personas_familia', 'consumo_bidon_pc']]
y = df['cpc_consumo']
X_sm = sm.add_constant(X)
modelo_ols = sm.OLS(y, X_sm).fit()

reg_sk = LinearRegression()
reg_sk.fit(X, y)
y_pred_sk = reg_sk.predict(X)
r2 = r2_score(y, y_pred_sk)
rmse = np.sqrt(mean_squared_error(y, y_pred_sk))

col1, col2 = st.columns(2)
col1.metric("Coeficiente de Determinación (R²)", f"{r2*100:.2f}%")
col2.metric("Raíz del Error Cuadrático Medio (RMSE)", f"{rmse:.4f} lt/hab-día")

# ==========================================
# 4. PANEL DE DIAGNÓSTICO VISUAL
# ==========================================
st.subheader("3. Panel de Gráficas de Diagnóstico")

fig, axes = plt.subplots(2, 3, figsize=(16, 9))
fig.suptitle('Panel de Diagnóstico Visual: Regresión Lineal Múltiple', fontsize=14, fontweight='bold')

# 1. Mapa de Calor
sns.heatmap(matriz_correlacion, annot=True, fmt=".2f", cmap="coolwarm", cbar=True, ax=axes[0, 0])
axes[0, 0].set_title('1. Matriz de Correlación')

# 2. Scatter Plot
sns.regplot(data=df, x='humedad_max', y='cpc_consumo', color='b', ax=axes[0, 1], line_kws={'color': 'red'})
axes[0, 1].set_title('2. Humedad Máx vs Consumo')

# 3. Valores Reales vs Predichos
axes[0, 2].scatter(y_pred_sk, y, alpha=0.7, color='teal')
axes[0, 2].plot([y.min(), y.max()], [y.min(), y.max()], 'r--', lw=2)
axes[0, 2].set_title(f'3. Reales vs Predichos (R² = {r2*100:.2f}%)')

# 4. Residuos vs Ajustados
residuos = modelo_ols.resid
y_pred_ols = modelo_ols.fittedvalues
axes[1, 0].scatter(y_pred_ols, residuos, color='purple', alpha=0.6)
axes[1, 0].axhline(y=0, color='r', linestyle='--', lw=2)
axes[1, 0].set_title('4. Residuos vs Homocedasticidad')

# 5. Gráfico Q-Q Normal
sm.qqplot(residuos, line='45', fit=True, ax=axes[1, 1])
axes[1, 1].set_title('5. Gráfico Q-Q Normal')

axes[1, 2].axis('off')
plt.tight_layout()

st.pyplot(fig)