#!/usr/bin/env python
# coding: utf-8

# ## 7. Ajuste del Modelo de Regresión Lineal

# In[1]:


# Tratamiento de datos
# ==============================================================================
import pandas as pd
import numpy as np

# Gráficos
# ==============================================================================
import matplotlib.pyplot as plt
from matplotlib import style
import seaborn as sns

# Preprocesado y modelado
# ==============================================================================
from scipy.stats import pearsonr
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
from sklearn.metrics import mean_squared_error
import statsmodels.api as sm
import statsmodels.formula.api as smf
from statsmodels.stats.anova import anova_lm
from scipy import stats

# Configuración matplotlib
# ==============================================================================
plt.rcParams['image.cmap'] = "bwr"
#plt.rcParams['figure.dpi'] = "100"
plt.rcParams['savefig.bbox'] = "tight"
style.use('ggplot') or plt.style.use('ggplot')

# Configuración warnings
# ==============================================================================
import warnings
warnings.filterwarnings('ignore')


# En primer lugar separamos los datos de training de los datos de test

# In[2]:


# Funcion que separa el juego de datos entre datos de entrenamiento y datos de trainning
def split_model (Dataset_features,Dataset_target):
# realizamos la división entre test y trainning
    df_feat_train,df_feat_test,df_targ_train,df_targ_test = train_test_split(
                                        Dataset_features,
                                        Dataset_target.values.reshape(-1,1),
                                        train_size   = 0.7,
                                        random_state = 1234,
                                        shuffle      = True
                                    )
    return df_feat_train,df_feat_test,df_targ_train,df_targ_test


# In[3]:


# Creamos el modelo de manera manual
# Param in : 
# df_features --> dataframe con la features para entrenae el modelo
def create_manual_model (df_features_tr,df_target_tr):
    df_in_model=pd.DataFrame(
                     np.hstack((df_features_tr, df_target_tr)),
                     columns=['Distance_NEW', 'Rooms', 'Bathroom', 'Price']
               )
    modelo = smf.ols(formula = 'Price ~ Distance_NEW + Rooms + Bathroom', data = df_in_model)
    modelo = modelo.fit()
    print(modelo.summary())


# In[4]:


# Creación del modelo utilizando matrices como en scikitlearn
# A la matriz de predictores se le tiene que añadir una columna de 1s para el intercept del modelo
def create_model_scikitlearn(df_features_tr,df_target_tr):
    df_features = sm.add_constant(df_features_tr, prepend=True)
    modelo = sm.OLS(endog=df_target_tr, exog=df_features_tr,)
    modelo = modelo.fit()
    print(modelo.summary())


# In[5]:


# funcion para calcular los intervalos de confianza
def calcular_intervalos_de_Confianza(modelo,al):
    intervalos_ci = modelo.conf_int(alpha=0.05)
    intervalos_ci.columns = ['2.5%', '97.5%']
    print (intervalos_ci)
    return intervalos_ci


# In[6]:


def calcular_residuos(df_features_tr,df_target_tr):
    df_target = y_train.flatten()
    prediccion_train = modelo.predict(exog = df_features_tr)
    residuos_train   = prediccion_train - df_target_tr
    print("Residuos",residuos_train)
    return residuos_train


# In[7]:


def inspeccion_visual_modelo(df_target_tr,prediccion_train,residuos_train):
    # Gráficos
    # ==============================================================================
    fig, axes = plt.subplots(nrows=3, ncols=2, figsize=(9, 8))

    axes[0, 0].scatter(df_target_tr, prediccion_train, edgecolors=(0, 0, 0), alpha = 0.4)
    axes[0, 0].plot([df_target_tr.min(), df_target_tr.max()], [y_trdf_target_train.min(), df_target_tr.max()],
                'k--', color = 'black', lw=2)
    axes[0, 0].set_title('Valor predicho vs valor real', fontsize = 10, fontweight = "bold")
    axes[0, 0].set_xlabel('Real')
    axes[0, 0].set_ylabel('Predicción')
    axes[0, 0].tick_params(labelsize = 7)

    axes[0, 1].scatter(list(range(len(df_target_tr))), residuos_train,
                   edgecolors=(0, 0, 0), alpha = 0.4)
    axes[0, 1].axhline(y = 0, linestyle = '--', color = 'black', lw=2)
    axes[0, 1].set_title('Residuos del modelo', fontsize = 10, fontweight = "bold")
    axes[0, 1].set_xlabel('id')
    axes[0, 1].set_ylabel('Residuo')
    axes[0, 1].tick_params(labelsize = 7)

    sns.histplot(
    data    = residuos_train,
    stat    = "density",
    kde     = True,
    line_kws= {'linewidth': 1},
    color   = "firebrick",
    alpha   = 0.3,
    ax      = axes[1, 0]
    )

    axes[1, 0].set_title('Distribución residuos del modelo', fontsize = 10,
                     fontweight = "bold")
    axes[1, 0].set_xlabel("Residuo")
    axes[1, 0].tick_params(labelsize = 7)


    sm.qqplot(
    residuos_train,
    fit   = True,
    line  = 'q',
    ax    = axes[1, 1], 
    color = 'firebrick',
    alpha = 0.4,
    lw    = 2
    )
    axes[1, 1].set_title('Q-Q residuos del modelo', fontsize = 10, fontweight = "bold")
    axes[1, 1].tick_params(labelsize = 7)

    axes[2, 0].scatter(prediccion_train, residuos_train,
                   edgecolors=(0, 0, 0), alpha = 0.4)
    axes[2, 0].axhline(y = 0, linestyle = '--', color = 'black', lw=2)
    axes[2, 0].set_title('Residuos del modelo vs predicción', fontsize = 10, fontweight = "bold")
    axes[2, 0].set_xlabel('Predicción')
    axes[2, 0].set_ylabel('Residuo')
    axes[2, 0].tick_params(labelsize = 7)

    # Se eliminan los axes vacíos
    fig.delaxes(axes[2,1])

    fig.tight_layout()
    plt.subplots_adjust(top=0.9)
    fig.suptitle('Diagnóstico residuos', fontsize = 12, fontweight = "bold");


# In[8]:


def test_normalidad_shapiro (residuos_train):
    shapiro_test = stats.shapiro(residuos_train)
    print (shapiro_test)
    return shapiro_test


# In[9]:


def test_normalidad_Dagostino():
    k2, p_value = stats.normaltest(residuos_train)
    print(f"Estadítico= {k2}, p-value = {p_value}")
    return k2,p_value


# In[10]:


def predecir(modelo,intervalo_confianza):
    predicciones = modelo.get_prediction(exog = X_train).summary_frame(alpha=intervalo_confianza)
    print(predicciones.head(4))
    return predicciones


# In[11]:


def calcular_error_RMSE(df_features_test,df_target_test):
# Error de test del modelo 
# ==============================================================================
    X_test = sm.add_constant(df_features_test, prepend=True)
    predicciones = modelo.predict(exog = df_features_test)
    rmse = mean_squared_error(
        y_true  = df_target_test,
        y_pred  = predicciones,
        squared = False
       )
    print("")
    print(f"El error (rmse) de test es: {rmse}")


# In[12]:


def comparar_modelos_anova(model1,model2):
   anova_lm(modelo, modelo_interacion)


# In[ ]:





# 
# ```{toctree}
# :hidden:
# :titlesonly:
# 
# 
# markdown
# notebooks
# notebooks
# ```
# 
