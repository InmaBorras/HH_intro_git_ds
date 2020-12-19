#!/usr/bin/env python
# coding: utf-8

# # 3. Variables Cuantitativas
# 
# 
# Procedemos a analizar las variables cuantitativas.

# In[1]:


import pandas as pd 
import numpy as np
import folium
import math
import time
from scipy import stats
pd.options.mode.chained_assignment = None  # default='warn'
from plotnine import ggplot, aes, geom_line, geom_point, geom_bar, geom_boxplot
import scipy.stats as ss
import matplotlib.pyplot as plot
import seaborn as sb
from seaborn import kdeplot
dataframe = pd.read_csv('/home/inma/Master_Data_Science _Contenido/Fundamentos_de_Analisis _de_Datos/Practica/Datos/Melbourne_housing_FULL.csv')
get_ipython().run_line_magic('run', '-i Variablescuantitativas.py')


# En primer lugar vamos a realizar una inspeccion ocular del dataset:
# 

# In[2]:



dataframe2016=dataframe[dataframe["Date"].str[-4:] =='2016']
dataframe2016.describe()
print("precio medio 2016 ",dataframe2016["Price"].mean())
dataframe2017=dataframe[dataframe["Date"].str[-4:] =='2017']
print("precio medio 2017 ",dataframe2017["Price"].mean())
dataframe2017.describe()
dataframe2018=dataframe[dataframe["Date"].str[-4:] =='2018']
print("precio medio 2018 ",dataframe2018["Price"].mean())
dataframe2017.describe()
dataframe.head()


# Vemos que tenemos en total 21 variables algunas con aspecto de ser cualitativas y otras cuantitativas. En los proximos puntos iremos analizando las caracteristitcas de las mismas.
# Vemos en primer lugar el tipo de las variables:

# In[3]:


dataframe.dtypes
len(dataframe["Suburb"].unique())


# In[4]:


aux=pd.DataFrame({"Suburb":pd.value_counts(dataframe['Suburb']),"Method":pd.value_counts(dataframe['Method']),                  "Regionname":pd.value_counts(dataframe['Regionname']),"SellerG":pd.value_counts(dataframe['SellerG']),                 "Method":pd.value_counts(dataframe['Method']),"CouncilArea":pd.value_counts(dataframe['CouncilArea'])})
aux


# In[5]:


print(dataframe.describe())
aux=pd.DataFrame({"Suburb":dataframe["Suburb"].describe(),"CouncilArea":dataframe["CouncilArea"].describe(),                  "Regionname":dataframe["Regionname"].describe(),"SellerG":dataframe["SellerG"].describe(),                 "Method":dataframe["Method"].describe()})
aux


# ## 3.1 Análisis  y transfromación de Variables Cualitativas
# 
# Para cada una de las variables cualitativas del dataframe comprobaremos sus medidas de centralidad y veremos cuales de ellas tiene sentido analizar.
# 
# 

# In[6]:


dataframe.describe()


# 
# Antes de empezar el analisis de las variables indicar que el coeficiente de asimetria empleado es el de SKEWNESS por tanto para analizar los resultados podemos usar la sigueiente funcion.  
# - Si el coeficiente de asimetría es menor que **-1 o mayor que 1**, la distribución de la variable es **extremadamente sesgada**.  
# - Si el coeficiente de asimetría se encuentra entre **-1 y -0,5 o entre 0,5 y 1**, la distribución de la variable es **moderadamente sesgada**.  
# - Si el coeficiente de asimetría se encuentra entre **-0,5 y 0,5**, la distribución de la variable es **aproximadamente sesgada**.  

# ### 3.1.1 Analisis de la variable Rooms

# Esta variable contiene el número de habitaciones de cada propiedad que hay en la muesta. Como se puede ver en la tabla 3.3.1 la variable tiene valor en 34857 toma valores discretos en el rango 1 a 16 dormitorios, que es el máximo encontrado. La mitad de la muestra tiene tres habitaciones o menos y el 75% de pisos tienen entre 1 y 4 habitaciones.
# A continuacion vamos a ver las frecuencias de la variable

# In[7]:


pd.value_counts(dataframe['Rooms'])/dataframe["Rooms"].count()


# Como se puede ver mas de un 40% de los pisos vendidos tienen 3 dormitorios que es mas del doble del porcentaje del numero de pisos que tiene 2 dormitorios (un 23%) o 4 dormitorios(un 22,8%).
# 
# Pasamos a hacer el cálculo de las medidas de dispersión y simetria calcularemos el coficiente de variación, rango, IQR y simtetria:

# In[8]:


mostrar_analisis_var_cuantitativas(dataframe["Rooms"])


# Como se puede var en la tabla anterio la variable tiene un coeficiente de variacion del 32% con un coeficiente de asimetria igual a 0 por que la variable es simetrica.
# 
# 

# In[9]:


mostrar_graf_variables_discretas(dataframe,"Rooms")
dataframe["Rooms_TR"]=dataframe["Rooms"].apply(np.sqrt)
sb.scatterplot(data=dataframe, x="Rooms", y="Price")
#mostrar_graf_variables_continuas(dataframe_filtered,"Distance_SQR")
plot.show()
sb.regplot(x="Rooms", y="Price", data=dataframe);
plot.show()


# Como ya se habia comprobado numéricamente la variable es muy simetrica aunque se aprecia unos outliers, viviendas de mas de 7 dormitorios que posteriormente veremos que efecto tienen en los modelos.  
# A priori no parece necesario tratar de ningún modo esta variable por ser simétrica y los outliers no parecen ser error de muestra.

# ### 3.1.2 Análisis de la variable Bedroom2
# 
# Esta variable contiene el número de dormitorios de cada propiedad. Como se puede ver en la tabla anterior la variable tiene valor en 26640 elementos de la muestra.Toma valores discretos en el rango 0 a 30 dormitorios, que es el máximo encontrado. La mitad de la muestra tiene tres dormitorios o menos y el 75% de pisos tienen entre 0 y 4 dormitorios. Estas cifras llaman la atencion ya que **el máximo número de habitaciones de la variable rooms tenia como maximo 16**, por lo que esos datos puede que estén mal imputados.

# In[10]:


pd.value_counts(dataframe['Bedroom2'])/dataframe["Bedroom2"].count()


# Los resultados son muy parecidos a los obtenidos con la variable Rooms por lo que no es probable que la variable rooms haga referencia también a número de dormitorios y no solo de habitaciones. Como ya ocurría con la variable rooms casi el 89% de las propiedades tiene entre 2 y 4 dormitorios.
# Vamos a verificar las medidas de dispersión y asimetría de la variable

# In[11]:


mostrar_analisis_var_cuantitativas(dataframe["Bedroom2"])


# El coeficiente de variacion es muy similar a la variable rooms y con el mismo rango intercuantilico, aunque en este caso el coficiente de asimetría es 0 lo que parece indica que se trata de una distribucion simétrica de los datos. Lo verificaremos con los siguientes diagramas:

# In[12]:


mostrar_graf_variables_discretas(dataframe,"Bedroom2")


# Como ya se había comprobado numéricamente la variable es muy simétrica aunque se aprecia unos outliers, viviendas de más de 7 dormitorios que se deberían eliminar. También habrá que seleccionar que variable es de mejor calidad para llevar a cabo el modelo.
# 
# En este caso la variable tiene un coeficiente de asimetría 0 y algún outlier que veremos posteriormente como afecta al modelo 

# ### 3.1.3 Analisis de la Variable Distance
# 
# La variable Distance indica la distancia al centro del inmbueble y toma como unidad las millas. Como se puede ver en la tabla 3.3.1 esta variable tiene valores en 34856 elementos de la muestra. La variable toma valores entre 0 y 48 millas , el 50% de los pisos de la muestra se encuentran a menos de 10,3 millas del centro de la ciudad y la desviacion tipica es 6,78. 
# Calculamos medidas de dispersión y simetría ya que las de centralidad las hemos visto en la tabla inicial con la descripción del dataframe.
# 

# In[13]:


mostrar_analisis_var_cuantitativas(dataframe["Distance"])


# Según el coeficiente la variable parece bastante simetrica, vemaos el boxplot para cerciorarnos.
# Vemos la distribucion de la variable y posibles outliers

# In[14]:


mostrar_graf_variables_continuas(dataframe,"Distance")


# 

# Como indica el coeficiente de asimetría, la variable es ligeramente asimétrica, vamos a intentar que se ajuste mejor a una distribución normal.

# In[15]:


dataframe["BathsAndRooms"]=(dataframe["Rooms"]+dataframe["Bathroom"])/dataframe["Distance"].apply(np.sqrt)
dataframe_filtered=dataframe[dataframe["Distance"] >0]
dataframe_filtered.reset_index(drop=True,inplace=True)
sb.scatterplot(data=dataframe, x="Distance", y="Price")
plot.show()

dataframe["Distance_TRA"]=dataframe["Distance"].apply(np.sqrt)

mostrar_graf_variables_continuas(dataframe_filtered,"Distance")
sb.regplot(x="Distance_TRA", y="Price", data=dataframe);
plot.show()
sb.regplot(x="BathsAndRooms", y="Price", data=dataframe);
plot.show()

dataframe.corr()


# ### 4.3.4 Analisis de la variable Bathrooms
# 
# Esta variable indica el número de baños del inmuble.Como se puede ver en la tabla inicial la variable tiene valor en 26631 elementos de la muestra. Toma valores discretos en el rango 0 a 12 dormitorios. El 75% de pisos tienen entre 0 y 2 baños . A continuación, vamos a ver las frecuencias de la variable:

# In[16]:


pd.value_counts(dataframe['Bathroom'])/dataframe["Bathroom"].count()


# Como se puede ver en la tabla anterior el 89% de la muestra tiene igual o menos de dos dormitorios con una media de 1,6 baños. En la tabla inicial respecto a las medidas de dispersión podemos ver que la desviación típica es de 0,72 y en la siguiente tabla vemos más variables de dispersión y asimetría:

# In[17]:


mostrar_analisis_var_cuantitativas(dataframe["Bathroom"])


# El coeficiente de variación es del 44% y el coeficiente de asimetria es -1 lo que indica que la variable es moderadamente sesgada, tiene cola a la derecha. comprobaremos esto mismo de manera visual pintando su histograma y boxplot

# In[18]:


mostrar_graf_variables_discretas(dataframe,"Bathroom")

sb.regplot(x="Bathroom", y="Price", data=dataframe);
plot.show()


# In[19]:


sb.scatterplot(data=dataframe, x="Bathroom", y="Price")
#mostrar_graf_variables_continuas(dataframe_filtered,"Distance_SQR")
plot.show()


# In[20]:


dataframe_filtered=dataframe[(dataframe["Bathroom"]>0) & (dataframe["Bathroom"] is not None)]
dataframe["Bathroom_TRA"]=dataframe["Bathroom"].apply(np.sqrt)
dataframe["Price_TRA"]=dataframe["Price"].apply(np.sqrt)
dataframe["Price_TRA"]=np.log2(dataframe["Price"])
#dataframe["Distance_SQR"]=dataframe["Distance"].apply(np.sqrt)
sb.scatterplot(data=dataframe, x="Bathroom_TRA", y="Price")
mostrar_graf_variables_continuas(dataframe,"Bathroom_TRA")
plot.show()

sb.displot(data=dataframe, x="Bathroom_TRA",y="Price_TRA",kind="kde")
plot.show()
#dataframe["Distance_SQR"]=dataframe["Distance"].apply(np.sqrt)
#sb.scatterplot(data=dataframe, x="Distance", y="Price")
#plot.show()
mostrar_graf_variables_continuas(dataframe,"Bathroom_TRA")
sb.regplot(x="Bathroom_TRA", y="Price_TRA", data=dataframe);
plot.show()
#hola


# ### 3.1.5 Análisis de la variable Car
# 
# Esta variable contiene el número de plazas de aparcamiento que tiene asociadas la vivienda. Como se puede ver en la tabla inicial la variable tiene valor en 26129 de la muestra y toma valores discretos en el rango 0 a 26 plazas de aparcamiento. El 75% de pisos tienen entre 0 y 2 plazas de aparcamiento . A continuacion vamos a ver las frecuencias de la variable:
# 

# In[21]:


pd.value_counts(dataframe['Car'])/dataframe["Car"].count()


# Como se puede ver casi el 47% de las casas de la muestra tienen dos plaza de aparcamiento y el 81% entre 1 y 2 plazas de aparcamiento.Respecto a las medidas de dispersión, en la tabla inicial podemos ver que la desviación típica es de 1.01. En la siguiente tabla vemos más variables de dispersión y asimetría:

# In[22]:


mostrar_analisis_var_cuantitativas(dataframe["Car"])


# Como se puede ver hay un coeficiente de variación alto y asimetría por la izquierda que verificaremos mejor haciendo algunos diagramas

# In[23]:


mostrar_graf_variables_discretas(dataframe,"Car")


# In[24]:


sb.scatterplot(data=dataframe, x="Car", y="Price")
mostrar_graf_variables_continuas(dataframe,"Car")
plot.show()


# ### 3.1.6 Analisis de la variable Landsize
# 
# Esta variable contiene el tamaño del terreno asociado a la vivienda, excluye los metros de la vivienda y está calculada en metros cuadrados. La media de terreno asociado al inmbueble es de  593.598993  El 75% de pisos tienen menos de 670 metros cuadrados de parcela. A continuacion vamos a ver las frecuencias de la variable:

# In[25]:


mostrar_analisis_var_cuantitativas(dataframe["Landsize"])


# La variable es aproximadamente sesgada a la izquierda y con un rango de valores muy amplio.Pintamos algunos gráficos para entender mejor la asimietría y la dispersión del a variablable

# In[26]:


mostrar_graf_variables_continuas(dataframe,"Landsize")


# Vamos a eliminar el registro superior que está desvirtuando el gráfico para poder analizarlo con mas detalle

# In[27]:


#Eliminamos outliers superiores para poder seguir analizando 
df_filtered=dataframe[dataframe["Landsize"]<20000]
print("registros filtrado = ",(dataframe["Landsize"].count()-df_filtered["Landsize"].count()))
mostrar_graf_variables_continuas(df_filtered,"Landsize")



sb.scatterplot(data=df_filtered, x="Landsize", y="Price")
#mostrar_graf_variables_continuas(dataframe_filtered,"Distance_SQR")
plot.show()


# Continua habiendo outliers que hacen que la variabe sea muy dispersa por lo que la variable es clara condidata a ser transformada

# In[28]:


#dataframe=dataframe[dataframe["Distance"].notna()] 

dataframe_filtered=dataframe[(dataframe["Landsize"]>0) ]

#dataframe_filtered.reset_index(drop=True,inplace=True)

#dataframe_filtered["Landsize_TRA"]=stats.boxcox(dataframe_filtered["Distance"])[0]

#dataframe_filtered["Landsize_TRA"]=dataframe_filtered["Landsize"].apply(np.sqrt)
dataframe_filtered["Landsize_TRA"]=np.log(dataframe_filtered["Landsize"])



sb.scatterplot(data=dataframe_filtered, x="Landsize_TRA", y="Price")
#mostrar_graf_variables_continuas(dataframe_filtered,"Distance_SQR")
plot.show()
#dataframe_filtered['Landsize_TRA'] = np.where(dataframe['Landsize'] >= 40000, 40000, (dataframe['Landsize']//100)*100)
#dataframe_filtered['Landsize_TRA'] = np.where(dataframe_filtered['Landsize_TRA'] <= 70, 70, dataframe_filtered['Landsize_TRA'])


sb.displot(data=dataframe_filtered, x="Landsize_TRA",y="Price")
plot.show()
#dataframe["Distance_SQR"]=dataframe["Distance"].apply(np.sqrt)
#sb.scatterplot(data=dataframe, x="Distance", y="Price")
#plot.show()
mostrar_graf_variables_continuas(dataframe_filtered,"Landsize_TRA")


# ### 3.1.7 Análisis de la variable BulldingArea
# 
# Esta variable contiene el el tamaño del terreno asociado a la vivienda en metros cuadrados. Como se puede ver en la tabla inicial la variable tiene valor en 13742 elementos de la muestra.La media del tamaño de los inmubles de 160.25640 y toma valores continuos en el rango 0 a 44515 . El 75% de pisos tienen menos de 188.00000 metros cuadrado. A continuacion, vamos a ver las medidas de dispersión de la varibale:

# In[29]:


mostrar_analisis_var_cuantitativas(dataframe["BuildingArea"])


# Como se puede ver la variable tiene asimetria a la derecha, pero en la tabla inicial vimos que la desviación estándar era muy alta para el valor que tomaba la media. Pasamos a comprobar visualmente la simetria.

# In[30]:


mostrar_graf_variables_continuas(dataframe,"BuildingArea")


# Como se puede ver hay claramente algún outlier que desvirtua la muestra (ya lo pudimos ver tambien en la tabla inicial donde la media era de 522 y el máximo era mas de 44000 metros cuadrados de parcela).
# Eliminamos el el máximo en cuestión y repintamos los gráficos

# In[31]:


#Eliminamos el maximo y vemos que ocurre
df_filtered=dataframe[dataframe["BuildingArea"]<1000]
print("registros filtrado = ",(dataframe["BuildingArea"].count()-df_filtered["BuildingArea"].count()))
mostrar_graf_variables_continuas(df_filtered,"BuildingArea")
#df_filtered["BuildingArea_TRA"]=np.log(dataframe_filtered["BuildingArea"])
df_filtered["BuildingArea_TRA"]=df_filtered["Distance"].apply(np.sqrt)
sb.scatterplot(data=df_filtered, x="BuildingArea_TRA", y="Price")
#mostrar_graf_variables_continuas(dataframe_filtered,"Distance_SQR")
plot.show()



df_filtered=dataframe[dataframe["YearBuilt"]>1800]

sb.scatterplot(data=df_filtered, x="YearBuilt", y="Price")
#mostrar_graf_variables_continuas(dataframe_filtered,"Distance_SQR")
plot.show()


# Los datos sin ese registro tienen mejor aspecto y parecen muy concentrados.

# ### 3.1.8 Análisis de la variable PropertyCountss
# 
# Esta variable contiene el numero de viviendas existentes en el barrio. Como se puede ver en la tabla inicial la variable tiene valor en 34854 y toma valores discretos en el rango 83 a 21650(que es el máximo de viviendas en un barrio). El 75% de pisos están en barrios con 10412 viviendas o menos.
# A continuación vamos a ver las frecuencias de la variable:

# In[32]:


mostrar_analisis_var_cuantitativas(dataframe["Propertycount"])


# La variable es aproximadamente sesgada a la derecha , lo comprobaremos visalmente:

# In[33]:


mostrar_graf_variables_continuas(dataframe,"Propertycount")


# ### 3.1.9 Análisis de la variable Latitud 
# 
# Esta variable contiene la coordenada geográfica correspondiente a la latitud y longitud del inmueble. Como se puede ver en las tablas de abajo.

# In[34]:


mostrar_analisis_var_cuantitativas(dataframe["Lattitude"])


# In[35]:


mostrar_analisis_var_cuantitativas(dataframe["Longtitude"])


# Pintamos la distribución de los precios por localización tomando la latitud y longitud de las casas. 
# 
# Se dividió por intervalos de precio y se asignó un color a cada  uno de ellos. 
#    
#     -Color Azul -->['Price'] < float('500000.0')
#     -color Verde-->['Price'] >= float('500000.0')) & (v['Price'] < float('800000.0'))
#     -Color Amarillo-->['Price'] >= float('800000.0')) & (v['Price'] < float('1200000.0'))
#     -Color cian-->['Price'] >= float('1200000.0'))& (v['Price'] < float('1500000.0'))
#     -Color Rojo-->['Price'] >= float('1500000.0'))& (v['Price'] < float('1800000.0'))
#     -Color Naranja-->['Price'] >= float('1800000.0'))& (v['Price'] < float('2100000.0'))
#     -Color Morado-->['Price'] >= float('2100000.0'))
#        
#         

# In[36]:



mapa  


# Utilizamos la información obtenida para localizar el suburbio más caro, el cual asumimos que es el centro de la  ciudad que corresponde con el suburbio "Koyong".
# 
# Finalmente localizamos el centro del suburbio y calculamos las nuevas distancias entre la localización de las viviendas y en centro. 
# 
# Llamaremos a la nueva variable "Distancia_NEW".

# In[37]:


data_filtered.describe()


# ## 3.2 Correlación  de Variables
# 
# Calculamos la correlación entre las variables entre ellas, prestando especial atención a la relación con el precio, para poder seleccionar las variables más adecuadas para incluir en el modelo. 

# In[38]:


data_filtered.describe()
data_filtered.corr()


# In[39]:


sb.heatmap(data_filtered.corr(), annot=True, fmt='.2f')


# En primer lugar, se puede observar que la nueva variable "Distancia_NEW" tiene una correlación bastante más alta -0.411055	que la variable "Distance" -0.238312, con el precio.
# 
# Además de las variable "Distancia_NEW"  hay otras dos variables que presentan un coeficiente de correlación alto "Bathrooms" 0.387676 y "rooms" 0.416335 que es la más alta. 
# Es importante tener en cuenta que ambas variables presentan también alta correlación entre ellas y esto podría crear problemas en las prediciones. 
# 
# 

# In[41]:


data_filtered.to_csv("Variables_Cuantitativas_v2.csv")


# In[ ]:




