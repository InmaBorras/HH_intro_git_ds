#!/usr/bin/env python
# coding: utf-8

# In[ ]:





# # 3.Variables Cualitativas 
# 
# La forma mas sencilla de  resumir las varibales cualitativas es hacer una tabla de contigencia que resuma las distribuciones de frecuencia. 

# In[1]:


import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn import preprocessing
from sklearn.preprocessing import LabelEncoder

data= pd.read_csv('/home/inma/Master_Data_Science _Contenido/Fundamentos_de_Analisis _de_Datos/Practica/Datos/Melbourne_housing_FULL.csv')


# In[2]:


data.head()


# In[3]:


data.info() #añadimos post code como variable cualitativa


# Podemos observar que 8 de las variables  son cualitativas, pero añadiremos "Postcode" dentro del analisis de las variables cualitativas ya que apesar de estar compuesto por un valor numerico  son datos independientes que debe ser tratados como categorias. 
# 
#  # 3.1 Resumen Numérico de Variables Cualitativas
# 
# ### Variable "Suburb" 
# 
# Esta variable  hace referencia al barrio donde se encuentra la casa. Procedemos a hacer un analisis de la distribución de densidad y podemos concluir que hay 351 suburbios diferentes. 
#  ( podemos agruparlos por zonas pero no tine sentido si hacemos referencia a la localizacion mejor que a esto)

# In[4]:


Variables_cualitativas=data[["Suburb","Address","Type","Method","SellerG","Date","CouncilArea","Regionname","Postcode"]]
pd.value_counts(Variables_cualitativas['Suburb'])


# ### Variable "Address"
# 
# La variable Address  indica la dirección donde se encuentran las casas. Al realizar el analisis  de distribución por densidad, podemos comprobar que hay direcciones repetidas, esto puede hacer referencia a la dirección de un edificio,  la venta de la misma casa en difrentes fechas o a duplicados. Para ello procedemos a analizar dichas categorias en profundidad. 

# In[5]:


Address=pd.value_counts(Variables_cualitativas['Address'])
Address[Address!=1]


# In[6]:


#Analizamos individualmente  las primeras direcciones para ver si si efectivamente son una casa. 
data_aux=data[data.Address == '118 Westgarth St']
print(data_aux)


# In[7]:


data[data.Address=="118 Westgarth St"]
data_aux.keys()
data_aux["Bedroom2"]

En este caso, parece ser  un casa que ha sido  traspasa , remodelada y puesta a la venta. 
# In[8]:


data[data.Address=="5 Charles St"]


# En este caso parece ser que todas son casas diferentes. 

# In[9]:


data[data.Address=="25 William St"]


# In[10]:


data[data.Address=="28 Blair St"]#En este caso, las dos ultimas filas,  son un duplicado.


# In[11]:


data[data.Address=="36 Aberfeldie St"]# en este caso tambien tenemos datos duplicados y efectivamente es una casa 


# Al existir la sospecha de que algunos de los datos estan duplicados, procedemos a hacer una eliminacion de duplicados en nuestro data set por longitud, latitud , precio y numero de habitaciones. 
# 
# ### Variable "PostCode"
# 
# La variable Postcode, como comentamos anteriormente, se incluye dentro del analisis de las variables cualitativas. 

# In[12]:


Variables_cualitativas["Postcode"]=Variables_cualitativas["Postcode"].astype("object")
pd.value_counts(Variables_cualitativas['Postcode'])


# La variables postcode cuenta con muchas categorias  y lo que dificultaria el modelo. 

# ### Variable "Type"
# 
# Esta variable cualitativa  es polítoma y presenta mas de dos valor no numéricos y corresponde con el tipo de vivienda. 
# La descripción determinada por la base de datos es la siguiente: 
# br - bedroom(s); 
# h - house,cottage,villa, semi,terrace; 
# u - unit, duplex;
# t - townhouse; 
# dev site - development site; 
# o res - other residential
# 
# Sin embargo, procederemos al renombramiento de las categorias para su mejor comprensión. 
# 
# Dormitorio- bedroom(s); 
# Casa - house,cottage,villa, semi,terrace; 
# Piso  - unit, duplex;
# Adosado- townhouse; 
# dev site - development site; 
# o res - other residential

# In[13]:


#data.apply(pd.Series.replace, to_replace='br', value='Dormitorio')# hacer una lista  para todos 
Variables_cualitativas["Type"].replace({"br":"Dormitorio","h":"Casa","u":"Piso","t":"Adosado"}, inplace=True)
pd.value_counts(data['Type'])
100*Variables_cualitativas["Type"].value_counts() /len(Variables_cualitativas["Type"])


# In[14]:


#data.apply(pd.Series.replace, to_replace='br', value='Dormitorio') #&(to_replace='h', value='Casa'))


# Esto hace referencia a la cultura del pais, donde la mayoria de la viviendas son corresponden a edificios unifamiliares. 

# ### Variable "Method"
# 
# Esta variable cualitativa  es polítoma y presenta mas de dos valor no numéricos y corresponde a  tipo de venta por la que adquirio el precio que refleja el dataset de cada una de las propiedades. 
# La descripción determinada por la base de datos es la siguiente: 
# - S - property sold( Vendida)--> vend ;
# - SP - property sold prior(vendida anteriormente)--> vend_ant;
# - PI - property passed in(propiedad traspasada) -->traspasada; 
# - PN - sold prior not disclosed (Venta anterior no revelada)-->Vent_ant_x; 
# - SN - sold not disclosed( Venta no revelada)-->vent_x; 
# - NB - no bid ( Sin oferta)-->sin_oferta; 
# - VB - vendor bid (oferta del vendedor)-->oferta_vendedor; 
# - W - withdrawn prior to auction (Retirada antes de la subasta)-->retirada_sub; 
# - SA - sold after auction( Vendida antes de la subasta)-->pre_sub; 
# - SS - sold after auction price not disclosed (Vendido despues de la subasta, precio no revelado)-->pre_sub_x. 
# - N/A - price or highest bid not available(Precio u oferta mas alta no disponible).
# 
# Cambiamos el nombre de las variables, para hacerlas mas entendibles. 

# In[15]:


Variables_cualitativas["Method"].replace({"S":"vend","SP":"vend_ant","PI":"traspasada","PN":"vent_ant_x","SN":"vent_x","NB":"sin_oferta","VB":"oferta_vendedor","W":"retirada_sub","SA":"pre_sub","SS":"pre_sub_x"}, inplace=True)
pd.value_counts(Variables_cualitativas['Method'])
100*Variables_cualitativas["Method"].value_counts() /len(Variables_cualitativas["Method"])


# Analizamos la relacion entre las variables que indican falta de precio  y la columna " Price" para comprobar la relacion con de estas categorias y los datos faltantes en precio. 
# 
# - vent_ant_x - sold prior not disclosed (Venta anterior no revelada); 
# - vent_x - sold not disclosed( Venta no revelada); 
# - sin_oferta - no bid ( Sin oferta); 
# - pre_sub_x - sold after auction price not disclosed (Vendido despues de la subasta, precio no revelado). 
# - N/A - price or highest bid not available(Precio u oferta mas alta no disponible).
# 

# In[16]:


Price_Method=data[["Price", "Method"]]
Price_Method=Price_Method[(data.Method=="vent_ant_x")|(data.Method=="vent_x")|(data.Method=="sin_oferta")|(data.Method=="pre_sub_x")|(data.Method=="N/A")]
#pd.crosstab(index=Price_Method["Price"],columns=Price_Method["Method"],margins=True)

pd.value_counts(Price_Method['Price'])#ordenamos para poder ver los precios mas altos.
Price_Method.head(10)


# Con este análisis podemos  observar categorías  mencionadas anteriormente presentan datos faltantes en la variables objetivo. 
# Esto es interesante tenerlo en cuenta para el tratamiento de missings
# La Variable "Method" contiene un  alto grado de categorías, por lo que después de haberla analizado, procedemos a agruparla según 3 criterios: 
# 
# - Categorias mas comunes: las 4 primera categorias acumulan más del 93% de todos los datos, que las dejaremos intactas. 
# 
#         vend            56.642855
#         vend_ant        14.616863
#         trasp           13.913991
#         oferta_vend      8.916430
# 
# - Categiras sin precio(sin_precio):  son todas aquellas que por su propia descripción y nuestra posterior comprobación no estan asociadas a un precio, nuestra variable target.
# 
#         vent_ant_x - sold prior not disclosed (Venta anterior no revelada);
#         vent_x - sold not disclosed( Venta no revelada); 
#         sin_oferta - no bid ( Sin oferta); 
#         pre_sub_x - sold after auction price not disclosed (Vendido despues de la subasta, precio no revelado).
#         N/A - price or highest bid not available(Precio u oferta mas alta no disponible).
# 
# - Otros(otro): son el resto de categorias con poca incidencia.  
# 
#         pre_sub          0.648363
#         retirada_sub     0.496314

# In[17]:


Variables_cualitativas= Variables_cualitativas.replace({"vent_ant_x":"sin_precio","vent_x":"sin_precio","sin_oferta":"sin_precio","pre_sub_x":"sin_precio","N/A":"sin_precio","pre_sub":"otro","retirada_sub":"otro"})
print(Variables_cualitativas)


# ### Variable "SellerG"
# 
# 
# La variable SellerG indica la fecha de venta o de recogida del precio de dicho apartamento. En este caso podemos ver que la fechas se encuentran entre Enero de 2016 y Octubre de 2018.
# 

# In[18]:


pd.value_counts(Variables_cualitativas['SellerG'])


# ###  Variable "Date" 
# 
# La variables fecha es muy importante para localizar en el tiempo el dataset. Para ellos determiandos la fecha maxima y minima. 

# In[19]:


# convertimos el campo fecha  datetime de pandas
from datetime import datetime
data['Date'] = pd.to_datetime(data['Date'])
print('La fecha mínima del data set es',data['Date'].min())
print('La fecha máxima del data set es',data['Date'].max())
pd.value_counts(Variables_cualitativas['Date'])


# ### Variable "CouncilArea'"
# 
# 
# La variabe Counsil Area corresponde con al area municipal donde se encuentran cada una de las casas. Esta variable aporta una informacion parecida a la que aporta postcode o regionname, por lo tanto podria ser considerada una de las variables a descartar. 

# In[20]:


pd.value_counts(Variables_cualitativas['CouncilArea'])


# ### Variable "Regionname"
# 
# 
# Corresponde a la regiones de Melburne, principalmente  podemos obsevar que la mayoria de las viviendas se encuentran alrededor de la zona metropolitana, que tiene relacion con las zonas con mayor densidad de poblacion. 

# In[21]:


pd.value_counts(Variables_cualitativas['Regionname'])
100*Variables_cualitativas["Regionname"].value_counts() /len(Variables_cualitativas["Regionname"])


# ## 3.2 Relación entre las variables cualitativas
# 
# Por otro lado, una vez analizado cada una de las variables podemos ver que es interesante  observar la relacion entre algunas de las variables. 
# Por ejemplo haciendo una relacion entre el tipo de casa y al region en la que se encuentra. 

# In[22]:


pd.crosstab(index=Variables_cualitativas["Type"],columns=Variables_cualitativas["Regionname"],margins=True)#representacion de la distribucion de tipo de casas por region 


# In[23]:


plot = pd.crosstab(index=Variables_cualitativas['Regionname'],
            columns=Variables_cualitativas['Type']).apply(lambda r: r/r.sum() *100,
                                              axis=1).plot(kind='bar')


# En el grafico anterior observamos que  casi todos los tipos de casas que corresponden con piso en encuentran en las zonas centrales de la ciudad.

# In[24]:


pd.crosstab(index=Variables_cualitativas["Regionname"],columns=Variables_cualitativas["Method"],margins=True)# ver tiene setido comprarlas variables.  hacer graficas por variable
#Mirar como son los barrios caros,etc. 
# ver las proporciones de ventas entre  S y SA ( por ejemplo) en diferenctre regiones. 


# En esta tabla, se observa que las regiones  con as ventas siempre corresponden a las de la zonas metropolitanas. Esto es logico debido a que son la areas con mayor densidad de población. 

# #  3.3 Selección de variables categóricas
# 
# #### Variables relacionadas con la localización
# 
# Tras analizar las variables cualitativas podemos, obsevar que muchas de ellas se refieren a la localizacion del alojamiento. 
# 
# Tanto como "Suburb","Address","CouncilArea","Postcode" cuentan con un numero muy amplio de categorías por lo que complicaría el desarrollo del modelo. La variable "Regionname" cuenta con un número(8 categorías) más adecuado y puede ser usasda para relacionar la localización con el precio. 
# 
# 
# 

# In[25]:



var = 'Regionname'
R = pd.concat([data['Price'], Variables_cualitativas[var]], axis=1)
f, ax = plt.subplots(figsize=(15, 8))
fig = sns.boxplot(x=var, y="Price", data=R)
fig.axis();


# Podemos observar que en las regiones metropolitanas el precio es ligeramente superior. Sobre todo en la "Southern Metropolitan".
# 
# 
# Para poder decidir cual de ellas es  mejor utilizar en nuestra regresion lineal,  usamos la funcion get_dummies para transformarlas en varibles factoriales 

# #### Variable "Type"
# 
# Analizamos la relacion entre la variables "Type" y la variables objetivo "Price"
# 

# In[26]:



p =pd.DataFrame(data[["Price","Type"]])
p["Type"].replace({"br":"Dormitorio","h":"Casa","u":"Piso","t":"Adosado"}, inplace=True)
#b=Variables_cualitativas[["Type"]]

sns.set_style("whitegrid")               
ax=sns.stripplot(x="Type", y="Price", data=p)


# Se observa que  la categoría "Casa"  no solo  prenseta  mayor  proponcion si no que además es mas  tiene unos precios mas altos. 

# #### Variable "SellerG"
# 
# Analizamos la relacion entre la variables "SellerG" y la variable objetivo "Price"

# In[27]:


G =data[["Price","SellerG"]]
sns.set_style("whitegrid")
ax=sns.stripplot(x="SellerG", y="Price", data=G)
ax.plot()


# La Agencia de Venta son también contiene un número muy alto de  categorías 388, a priori parece que algunos de los "SellerG", están mas relacionados con un precio  mas alto o mas bajo. 
#  Sin embargo, estas agencias estan localizadas por zonas geograficas por lo tanto descartamos esta variable como posible para la primera creación de nuestro modelo. 

# #### Variable "Method"

# In[28]:


M=pd.DataFrame(data["Method"].replace({"S":"vend","SP":"vend_ant","PI":"traspasada","PN":"vent_ant_x","SN":"vent_x","NB":"sin_oferta","VB":"oferta_vendedor","W":"retirada_sub","SA":"pre_sub","SS":"pre_sub_x"}, inplace=True))
M=M.replace({"vent_ant_x":"sin_precio","vent_x":"sin_precio","sin_oferta":"sin_precio","pre_sub_x":"sin_precio","N/A":"sin_precio","pre_sub":"otro","retirada_sub":"otro"})
M =data[["Price","Method"]]
#fig, (ax1) = plot.subplots(1,figsize=(12,6)
sns.set_style("whitegrid")
ax=sns.stripplot(x="Method", y="Price", data=M,)
ax.plot()


# #### Variable "Date"

# In[29]:


G =data[["Price","Date"]]
G=G.sort_values('Date',ascending=False)
with sns.axes_style("white"):
    sns.jointplot(x="Date",y="Price", data=G, kind="kde",height=7,fill=True)


# # 3.4 Correlación de variables categóricas 
# 
# Observamos la relación del cada una delas variables categóricas con el precio. 

# In[30]:


Variables_cualitativas["Price"]=data["Price"]
Variables_cualitativas_Corr=Variables_cualitativas[["Price","Method","Type","Date","Regionname"]]
Variables_cualitativas_Corr["Date"]=Variables_cualitativas["Date"].astype("str")
Variables_cualitativas_Corr["Regionname"]=Variables_cualitativas_Corr["Regionname"].astype("str")
encoder = LabelEncoder()
Variables_cualitativas_Corr["Type"]=encoder.fit_transform(Variables_cualitativas_Corr["Type"])
Variables_cualitativas_Corr["Method"]=encoder.fit_transform(Variables_cualitativas_Corr["Method"])
Variables_cualitativas_Corr["Regionname"]=encoder.fit_transform(Variables_cualitativas_Corr["Regionname"])
Variables_cualitativas_Corr["Date"]=encoder.fit_transform(Variables_cualitativas_Corr["Date"])

Variables_cualitativas_Corr.head(100)


# In[31]:


corr_matrix=Variables_cualitativas_Corr.corr(method='pearson')         
corr_matrix["Price"].sort_values(ascending=False)


# In[32]:


sns.heatmap(corr_matrix.corr(), annot=True, fmt='.2f')


# In[33]:


Variables_cualitativas["Price"]=data["Price"]
#Mantenemos todas la categorias de todas las variables para poder analizarlas por separado
Variables_cualitativas_T=pd.get_dummies(Variables_cualitativas,columns = ["Regionname"])
Variables_cualitativas_T=pd.get_dummies(Variables_cualitativas_T,columns = ["Type"])
Variables_cualitativas_T=pd.get_dummies(Variables_cualitativas_T,columns = ["Date"])
Variables_cualitativas_T=pd.get_dummies(Variables_cualitativas_T,columns = ["Method"])
Variables_cualitativas_T.head()


# Estudiamos la correlación ahora de cada una de las categóricas por Variable por separado.
# 
# #### Regionname 

# In[34]:



f=Variables_cualitativas_T[["Price","Regionname_Eastern Victoria"]]
corr_matrix=f.corr(method='pearson')         
corr_matrix["Price"].sort_values(ascending=False)


# In[35]:


f=Variables_cualitativas_T[["Price","Regionname_Northern Metropolitan"]]
corr_matrix=f.corr(method='pearson')         
corr_matrix["Price"].sort_values(ascending=False)


# In[36]:


f=Variables_cualitativas_T[["Price","Regionname_Northern Victoria"]]
corr_matrix=f.corr(method='pearson')         
corr_matrix["Price"].sort_values(ascending=False)


# In[37]:


f=Variables_cualitativas_T[["Price","Regionname_South-Eastern Metropolitan"]]
corr_matrix=f.corr(method='pearson')         
corr_matrix["Price"].sort_values(ascending=False)


# In[38]:


f=Variables_cualitativas_T[["Price","Regionname_Southern Metropolitan"]]
corr_matrix=f.corr(method='pearson')         
corr_matrix["Price"].sort_values(ascending=False)


# In[39]:


f=Variables_cualitativas_T[["Price","Regionname_Western Metropolitan"]]
corr_matrix=f.corr(method='pearson')         
corr_matrix["Price"].sort_values(ascending=False)


# In[40]:


f=Variables_cualitativas_T[["Price","Regionname_Western Victoria"]]
corr_matrix=f.corr(method='pearson')         
corr_matrix["Price"].sort_values(ascending=False)


# In[41]:


f=Variables_cualitativas_T[["Price","Regionname_Eastern Metropolitan"]]
corr_matrix=f.corr(method='pearson')         
corr_matrix["Price"].sort_values(ascending=False)


# #### Type
# 

# In[42]:


f=Variables_cualitativas_T[["Price","Type_Casa"]]
corr_matrix=f.corr(method='pearson')         
corr_matrix["Price"].sort_values(ascending=False)


# In[43]:


f=Variables_cualitativas_T[["Price","Type_Piso"]]
corr_matrix=f.corr(method='pearson')         
corr_matrix["Price"].sort_values(ascending=False)


# In[44]:


f=Variables_cualitativas_T[["Price","Type_Adosado"]]
corr_matrix=f.corr(method='pearson')         
corr_matrix["Price"].sort_values(ascending=False)


# #### Method 
# 

# In[45]:


f=Variables_cualitativas_T[["Price","Method_otro"]]
corr_matrix=f.corr(method='pearson')         
corr_matrix["Price"].sort_values(ascending=False)


# In[46]:


f=Variables_cualitativas_T[["Price","Method_sin_precio"]]
corr_matrix=f.corr(method='pearson')         
corr_matrix["Price"].sort_values(ascending=False)


# In[47]:


f=Variables_cualitativas_T[["Price","Method_traspasada"]]
corr_matrix=f.corr(method='pearson')         
corr_matrix["Price"].sort_values(ascending=False)


# In[48]:


f=Variables_cualitativas_T[["Price","Method_vend"]]
corr_matrix=f.corr(method='pearson')         
corr_matrix["Price"].sort_values(ascending=False)


# In[49]:


f=Variables_cualitativas_T[["Price","Method_vend_ant"]]
corr_matrix=f.corr(method='pearson')         
corr_matrix["Price"].sort_values(ascending=False)


# In[50]:


f=Variables_cualitativas_T[["Price","Method_oferta_vendedor"]]
corr_matrix=f.corr(method='pearson')         
corr_matrix["Price"].sort_values(ascending=False)


# 

# # 3.5 Transformación de variables categóricas 
# 
# Despues de la selección de variables vamos a proceder a transformarlas para su uso porterior en el modelo. 

# In[51]:


Variables_cualitativas_T=pd.get_dummies(Variables_cualitativas,columns = ["Regionname"],drop_first= True)
Variables_cualitativas_T=pd.get_dummies(Variables_cualitativas_T,columns = ["Type"],drop_first= True)
Variables_cualitativas_T=pd.get_dummies(Variables_cualitativas_T,columns = ["Date"],drop_first= True)
Variables_cualitativas_T=pd.get_dummies(Variables_cualitativas_T,columns = ["Method"],drop_first= True)


# In[52]:


Variables_cualitativas_T


# In[57]:


Variables_cualitativas_T.to_csv('precios_casas.csv', index=False)


# In[59]:


Variables_cualitativas_Corr.to_csv('precios_casas_labelencoder.csv', index=False)


# In[ ]:




