#!/usr/bin/env python
# coding: utf-8

# # 5. Limpieza de los datos
# 
# Tras es análisis de la variables hemos detectado tanto duplicidades en los datos como presencia de datos faltantes. Procedemos a la limpieza de los datos para la posterior construcción de un modelo mas exacto. 
# 
# En primer lugar, cargamos todas las librerias necesarias y el archivo de código. 
# 

# ## 5.1 Eliminacion de duplicados
# 
# En el análisis de las variables cualitativas obsevamos que algunos datos tenian la misma dirección y precio.
# Basándonos en este razonamiento, buscamos las  filas que tengan las variables 'Suburb', 'Address','Postcode'y 'CouncilArea' iguales y también en las que las variables "Adress", " Prices" y "Date" coincidan. 
# Con le objetivo de no eliminar datos que no estuvieran duplicados mantuvimos los que se encontraban solo en uno de los dataframes. 
# 
# 
# ((Esto hay que mirarlo bien por que no se si es así.)) 

# In[1]:


import pandas as pd 
data = pd.read_csv('/home/inma/HH_intro_git_ds/Melbourne_housing_FULL.csv')
get_ipython().run_line_magic('run', '-i fundamentos_datos_variables.py')
data_duplicados=eliminar_duplicados(data)
print("El número de datos duplicados eliminados es", ((len(data))-(len(data_duplicados))))


# ## 5.2 Detección e Imputación de Datos Faltantes
# 
# Una vez eliminiados los datos duplicados, para finalizar la preparación de los datos es necesario eliminar los datos faltantes. 
# 
# Para ellos en primer lugar representamos gráficamente los datos faltantes en cada una de las variables realizando  una función de visualización. 
# 

# In[37]:


get_ipython().run_line_magic('run', '-i fundamentos_datos_missings.py')
visualizacion_missings(data_duplicados)


# Se detecta que hay variables donde no existen datos faltantes mientras que en otras el número es muy alto, como vimos en el EDA inicial.
# 
# En este punto es importante seleccionar de forma adecuada la imputación de los mismo. Directamente descartamos las variables "YearBuilt", "BuildingArea" y "Bedroom2" ya que presentan muchos datos faltantes, además de una baja correlación y/o muchas similitud con otras variables como en el caso de "Bedrooms2" y "rooms". 
# 
# 
# Por otro, lado procedemos a imputar los datos restantes, para ellos realizaremos una función que complete los datos fataltantes usando un modelo de regresiñon lineal o random forest, en función de cual de los dos es más adecuado. 
# 
# En primer lugar, usamos la función "Kfold()" para separar nuestros datos en 5 partes, usaremos 4 como training y una como test.  Cada una de las variables tendrá por lo tanto  4 grupos diferentes para realizar la regresión lineal y el random forest, usaremos estos modelos para cálcular la media de los errores cuadráticos del resultado de ambos modelos con el grupo "test". 
# 
# Finalmente, la función selecciona el mejor modelo y con el completa los datos faltantes en el dataframe de cada una de las variables. Para mejorar la exactitud del modelo cada vez que una variable es completada se incluye dentro de los modelos para ser usada en el cálculo de las siguientes variables.  

# In[40]:


data_no_missings = pd.read_csv('/home/inma/HH_intro_git_ds/precios_casas_sinduplicados_indexTRUE.csv')

data_no_missings["Price"]=data["Price"]

#Eliminamos los missings restantes 
dataframe=data_no_missings.dropna(subset=['Distance']) 
dataframe=dataframe.dropna(subset=['Regionname']) 
dataframe=dataframe.dropna(subset=['Regionname']) 
dataframe=dataframe.dropna(subset=['CouncilArea']) 
dataframe=dataframe.dropna(subset=['Propertycount']) 
dataframe=dataframe.dropna(subset=['Postcode'])

visualizacion_missings(dataframe)

