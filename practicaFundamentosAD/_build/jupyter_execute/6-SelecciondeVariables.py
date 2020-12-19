#!/usr/bin/env python
# coding: utf-8

# # 6.  Variables seleccionadas

# Después, de tratar los datos y de haber analizado las variables por separado las hemos selecciondo en función de la correlación con el precio. 
# 
# Las variables con las que comenzaremos a realizar el modelo de regresión lineal son: 
# 
# Cualitativas: Regionname, Method y Type. 
# Cuantitativas: Todas las variables son susceptibles de ser usasadas en el modelo excepto las que fueron eliminadas en la imputación de missings. "YearBuilt", "BuildingArea" y "Bedroom2".
# 
# Aplicamos distintos métodos para reducir variables:
# 
# ## 6.1  Lasso 
# 
# Hemos utilzado una función para encontrar el alpha optimo= 0.0002  y de eta forma encontramos el alpha optimo para aplicar Lasso y eliminar las variables correspondientes. 
# 
# aplicamos el algorimo Lasso y vmaos recorriendo todos los coeficientes y cuando los coeficientes son 0 imprimimos por pantalla que esa variable no debe ser seleccionada.
# 

# In[1]:


import pandas as pd 
dataframe = pd.read_csv('Variables_Cuantitativas_v2.csv')

get_ipython().run_line_magic('run', '-i Seleccion_Variables.py')

lista_parametros=['Rooms','Distancia_NEW','Distance', 'Bathroom', 'Car', 'Lattitude', 'Longtitude', 'Landsize', 'Propertycount']

X=dataframe[lista_parametros]
y=dataframe['Price']

X=np.nan_to_num(X)
y=np.nan_to_num(y)
lasso_prueba(X,y,lista_parametros)


# 
# 
# ## 6.2 Stepwise
# 
# Hemos utilisado un algoritmo que crea modelos en funcion al numero de columnas. 
# Cuando el r cuadrado entre el modelo anterior y el modelo actual es menor a 0.0001  pare de incluir varibales en el modelo. 
# 
# 

# In[ ]:



Best_stepwise_selection(dataframe,X)


# Las variables que se obtienen son " Distania_NEW", "Room" " Latitude", "Landsize" y " Bathrooms".

# In[ ]:




