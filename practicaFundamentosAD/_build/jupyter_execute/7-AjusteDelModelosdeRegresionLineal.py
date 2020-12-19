#!/usr/bin/env python
# coding: utf-8

# #  7.Ajuste del Modelo
# 
# Ajustamos el modelo para predecir el precio de las casas de Melbourne (Australia) en función de las características de las casas.
# Se ha comprobado que 

# In[1]:


import warnings
warnings.filterwarnings('ignore')
get_ipython().run_line_magic('run', '-i 7-AjusteDelModelosdeRegresionLineal.py')


# In[2]:


# División de los datos en train y test
# ==============================================================================

x=dataframe_old[['Distance','Rooms','Landsize','Lattitude','Bathroom']]
y=dataframe_old['Price']

entrena_y_muestra_modelo(x,y)


# In[3]:



x = dataframe_filtered[['Distancia_NEW','Rooms','Lattitude','Landsize','Bathroom']]
y = dataframe_filtered['Price']

entrena_y_muestra_modelo(x,y)


# El modelo con todas las variables introducidas como predictores tiene un R2 Ajustado de aceptable (0.642), es capaz de explicar el 64.2% de los precios de venta de las casas en Melbourne. 
# 
# El omnibus es de 569 cuanto mas se acerque a 0 mas normalidad tendrán nuestros residuos, se ha mejorado mucho haciendo las transformaciones indicadas.Al comienzo del ajuste del modelo era mayor a 8000 por lo que la mejora es significativa.
# 
# La prueba Durbin-Watson devuelve un valor de 2.003 por lo que no se pude afirmar que la distribucion de los residuos sea normal

# In[ ]:




