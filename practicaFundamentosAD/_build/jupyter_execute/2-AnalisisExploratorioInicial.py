#!/usr/bin/env python
# coding: utf-8

# # 2. Análisis Exploratorio Inicial (EDA)

# Trás definir el objetivo de la práctica procedemos a hacer el analisis exploratorio de los datos. 
# 
# En primer luga cargamos todas las librerias que necesitaremos y hacemos una visualización general de dataframe, de su composición y del tipo de datos que contiene cada columna.

# In[1]:


import pandas as pd 
from pandas_profiling import ProfileReport
data = pd.read_csv('Melbourne_housing_FULL.csv')
get_ipython().run_line_magic('run', '-i fundamentos_datos_variables.py')


# In[2]:


data.head(20)


# In[3]:


print("Cantidad de Filas y de Columnas",data.shape)


# In[4]:


data.info()


# Durante estas primeras observaciones ya se puede identificar que el dataset  es bastante grande con 34857 y 21 vcolumnas, pero también contiene datos faltantes en bastantes variables incluyendo la varibable objetivo "Price". Además, cuenta a priori con varibles cualitativas y cuantitivas.
# 
# Por lo tanto, para hacer un análisis en profundidad usamos la función "profiling" de pandas. 
# 

# In[5]:


profile = ProfileReport(data, title="Pandas Profiling Report")


# In[6]:


profile


# De esta función obtenemos informacion muy relevante con respecto al dataset:
# 
# + El dataset cuenta con 13 variables numéricas y 8 categóricas 
# 
# + Existe un pequeño porcentaje de datos duplicados, tomaremos esto en cuenta durante el análisis de variables para poder identificarlos y tratarlos. 
# 
# + El pocetaje de datos faltantes corresponde a un 13,8%  y todo ellos se encuentran en las variables numéricas. (es alto es bajo). Siendo "YeatBuild" , "BuildingArea", y "Landsize".
# 
# + Las variables con mayor correlación son: "rooms" y "Bedrrom2". Tambien podemos  observar que tanto ambas variables tienen una alta correlacion entre ellas y podrían producir colinealidad. 
# 
# Para poder clasificar los tipo de varibles vamos a realizar una función que las identifique que tipo de varaibles es cada una en función del tipo de datos que contiene. 

# In[7]:



clasificar_variables(data)
   


# Esto nos será de mucha utilidad en el análisis de las que realizaremos a continuación y su posterior transformación. 

# In[ ]:




