#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd 
import numpy as np
import folium
import math
import time
from scipy import stats
pd.options.mode.chained_assignment = None  # default='warn'
from plotnine import ggplot, aes, geom_line, geom_point, geom_bar, geom_boxplot
dataframe = pd.read_csv('/home/ruben/tmp/HH_intro_git_ds/precios_casas_sinduplicados.csv')
dataframe_old = pd.read_csv('/home/ruben/tmp/HH_intro_git_ds/Melbourne_housing_FULL.csv')
import scipy.stats as ss
import matplotlib.pyplot as plot
import seaborn as sb
from seaborn import kdeplot
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

def quartile_skew(x):
  q = x.quantile([.25, .50, .75]) 
  return ((q[0.75] - q[0.5]) - (q[0.5] - q[0.25])) / (q[0.75] - q[0.25])
dataframe['Price']=dataframe_old['Price']

duplicateRowsDF=pd.DataFrame()
duplicateRowsDF = dataframe[dataframe.duplicated(['Suburb', 'Address','Postcode','CouncilArea',],keep=False)]
duplicateRowsDF=duplicateRowsDF.drop_duplicates(subset=['Address','Price','Date'])
duplicateRowsDF=duplicateRowsDF.dropna(subset=['Price'])
dataframe=dataframe.drop_duplicates(subset=['Address','Suburb'], keep=False, ignore_index=True)
dataframe.append(duplicateRowsDF)
dataframe=dataframe.dropna(subset=['Price'])
dataframe=dataframe.reset_index(drop=True)



#dataframe['Price']=dataframe['Price'].replace(0, 1)
print(dataframe)
print(dataframe_old.dtypes)
dataframe['Rooms']=np.sqrt(dataframe['Rooms'])
dataframe['Distance']=np.sqrt(dataframe['Distance'])


dataframe['Longtitude']=dataframe['Longtitude'].replace(0, 1)
dataframe['Distance']=dataframe['Distance'].replace(0, 1)
dataframe['Landsize']=dataframe['Landsize'].replace(0, 1)
dataframe=dataframe[dataframe['Landsize']>0]
dataframe['Longtitude']=dataframe['Longtitude'].replace(0, 1)

dataframe_filtered=pd.DataFrame(dataframe[dataframe["Distance"].notnull()])
print(dataframe_filtered)
dataframe_filtered=dataframe_filtered[dataframe_filtered["Landsize"].notnull()]
dataframe_filtered=dataframe_filtered[dataframe_filtered["Price"].notnull()]
dataframe_filtered=dataframe_filtered[dataframe_filtered['Price']!=np.nan]
dataframe_filtered['Price']=np.sqrt(np.log10(dataframe_filtered['Price'].astype(np.int64)))    
dataframe_filtered=dataframe_filtered[dataframe_filtered['Distance']<40]
dataframe_filtered=dataframe_filtered[dataframe_filtered['Rooms']<10]
dataframe_filtered=dataframe_filtered[dataframe_filtered['Bathroom']<5]
dataframe_filtered['Landsize']=np.log(dataframe_filtered['Landsize'])
dataframe_filtered['Longtitude']=np.log(dataframe_filtered['Longtitude'])
dataframe_filtered['Propertycount']=np.log(dataframe_filtered['Propertycount'])
dataframe_filtered['Propertycount']=np.log(dataframe_filtered['Propertycount'])
dataframe_filtered=dataframe_filtered[dataframe_filtered['Distance']!=np.nan]
dataframe_filtered=dataframe_filtered[dataframe_filtered['Landsize']!=np.nan]
dataframe_filtered=dataframe_filtered[dataframe_filtered['Longtitude']!=np.nan]
dataframe_filtered['BathAndRooms']=(dataframe_filtered["Rooms"]+dataframe_filtered["Bathroom"])/np.sqrt(dataframe_filtered['Distance'])
dataframe_filtered.describe()


# In[2]:


# División de los datos en train y test
# ==============================================================================
X = dataframe_filtered[['BathAndRooms','Rooms', 'Bathroom', 'Distance','Landsize','Longtitude']]
y = dataframe_filtered['Price']

X_train, X_test, y_train, y_test = train_test_split(
                                        X,
                                        y.values.reshape(-1,1),
                                        train_size   = 0.8,
                                        random_state = 1234,
                                        shuffle      = True
                                    )

# Creación del modelo utilizando el modo fórmula (similar a R)
# ==============================================================================
'''datos_train = pd.DataFrame(
                     np.hstack((X_train, y_train)),
                     columns=['Rooms', 'Bathroom', 'Distance', 'Price']
               )
modelo = smf.ols(formula = 'Price ~ BathAndRooms', data = datos_train)
modelo = modelo.fit()
print(modelo.summary())
'''

# Creación del modelo utilizando matrices como en scikitlearn
# ==============================================================================
# A la matriz de predictores se le tiene que añadir una columna de 1s para el intercept del modelo

X_train = sm.add_constant(X_train, prepend=True)
modelo = sm.OLS(endog=y_train, exog=X_train,)
modelo = modelo.fit()
print(modelo.summary())


# Intervalos de confianza para los coeficientes del modelo
# ==============================================================================
intervalos_ci = modelo.conf_int(alpha=0.05)
intervalos_ci.columns = ['2.5%', '97.5%']
intervalos_ci




# Diagnóstico errores (residuos) de las predicciones de entrenamiento
# ==============================================================================
y_train = y_train.flatten()
prediccion_train = modelo.predict(exog = X_train)
residuos_train   = prediccion_train - y_train

# Gráficos
# ==============================================================================
fig, axes = plt.subplots(nrows=3, ncols=2, figsize=(9, 8))

axes[0, 0].scatter(y_train, prediccion_train, edgecolors=(0, 0, 0), alpha = 0.4)
axes[0, 0].plot([y_train.min(), y_train.max()], [y_train.min(), y_train.max()],
                'k--', color = 'black', lw=2)
axes[0, 0].set_title('Valor predicho vs valor real', fontsize = 10, fontweight = "bold")
axes[0, 0].set_xlabel('Real')
axes[0, 0].set_ylabel('Predicción')
axes[0, 0].tick_params(labelsize = 7)

axes[0, 1].scatter(list(range(len(y_train))), residuos_train,
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


# In[ ]:




