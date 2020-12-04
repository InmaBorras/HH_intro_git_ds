import pandas as pd 
import numpy as np
from plotnine import ggplot, aes, geom_line, geom_point, geom_bar, geom_boxplot
import matplotlib.pyplot as plt
from pandas.plotting import scatter_matrix
dataframe = pd.read_csv('precios_casas_full.csv')

def esDeProporcion(minimo):
    if(minimo<0):
        return(False)
    else:
        return(True)

def esAcotada(minimo,maximo):
    if( minimo==0 and maximo ==100):
        return(True)
    else:
        return(False)

def clasificar_variables(dataframe):
        
    tipos_columnas=dict()
    for index,i in enumerate(dataframe.keys()):
        if(isinstance(dataframe.iloc[0][index], ( np.int64))):
            #variable cuantitativa discreta
            if(len(dataframe[str(i)].value_counts())>2):
                #politomizada
                minimo=dataframe[str(i)].min()
                maximo=dataframe[str(i)].max()
                if(esDeProporcion(minimo)):
                    if(esAcotada(minimo,maximo)):
                        tipos_columnas[str(i)]='Cuantitativa Discreta de Proporcion Politomizada Acotada'
                    else:
                        tipos_columnas[str(i)]='Cuantitativa Discreta de Proporcion Politomizada No Acotada'
                else:
                    if(esAcotada(dataframe[str(i)])):
                        tipos_columnas[str(i)]='Cuantitativa Discreta de Intervalo Politomizada Acotada'
                    else:
                        tipos_columnas[str(i)]='Cuantitativa Discreta de Intervalo Politomizada No Acotada'

            elif(len(dataframe[str(i)].value_counts())==2):
                #dicotomizada
                tipos_columnas[str(i)]='Cuanitativa Dicotoma'
            else:
                print('deberias borrar la variable '+ str(i))
        elif(isinstance(dataframe.iloc[0][index], ( np.float64))):
            minimo=dataframe[str(i)].min()
            maximo=dataframe[str(i)].max()
            #variable cuantitativa continua
            if(esDeProporcion(minimo)):
                    if(esAcotada(minimo,maximo)):
                        tipos_columnas[str(i)]='Cuantitativa Continua de Proporcion Politomizada Acotada'
                    else:
                        tipos_columnas[str(i)]='Cuantitativa Continua de Proporcion Politomizada No Acotada'
            else:
                if(esAcotada(dataframe[str(i)])):
                        tipos_columnas[str(i)]='Cuantitativa Continua de Intervalo Politomizada Acotada'
                else:
                        tipos_columnas[str(i)]='Cuantitativa Continua de Intervalo Politomizada No Acotada'
        elif(isinstance(dataframe.iloc[0][index],bool)):
            #variable cualitativva dicotoma
            tipos_columnas[str(i)]='Cualitativa Dicotoma'
        elif(isinstance(dataframe.iloc[0][index],str)):
            if(len(dataframe[str(i)].value_counts())>2 and len(dataframe[str(i)].value_counts())<12):
                #politomizada y alomejor ordinal
                print(dataframe[str(i)].value_counts())
                ordinal=input('¿Es ordinal?(seleccionar si o no): ')
                if(ordinal=='si'):
                    tipos_columnas[str(i)]='Cualitativa Politoma Ordinal'
                elif(ordinal=='no'):
                    tipos_columnas[str(i)]='Cualitativa Politoma No ordinal'
                else:
                    tipos_columnas[str(i)]='Cualitativa Politoma Nan'
            elif(len(dataframe[str(i)].value_counts())>2):
                 tipos_columnas[str(i)]='Cualitativa Politoma'
            elif(len(dataframe[str(i)].value_counts())==2):
                #dicotomizada
                tipos_columnas[str(i)]='Cualitativa Dicotoma'
            else:
                print('deberias borrar la variable '+ str(i))
    return(tipos_columnas)
dataframe_bueno=pd.DataFrame()

'''
linea =dataframe.iloc[0]
linea
    for index, i in enumerate(dataframe['Date']):
    if( i=='1/07/2017'):
        import pdb;pdb.set_trace()
for index, i in enumerate(dataframe['Address']):
    if( i=='3A Church Rd'):
        import pdb;pdb.set_trace()
dataframe['Price'].notnull().sum()
#dataframe.drop_duplicates(subset=['Address','Regionname','Distance','CouncilArea','Rooms'])
#eliminas 6000
#670
#1152
 #2/06/2018
for i in linea: print(type(i))
dataframe.keys()
ggplot(dataframe) + aes(x='CouncilArea',y='Rooms')+geom_point()
boxplot = dataframe.boxplot(column=['Rooms', 'Distance'])
boxplot = dataframe.boxplot(column=['Price'])
import pdb; pdb.set_trace() 
#boxplot = df.boxplot(column=['Rooms', 'Price', 'Distance'],by=['type'])
ggplot(dataframe) + aes(x='Type',y='Rooms')+geom_point()
ggplot(dataframe)+ aes(x="Regionname", y="Price")+ geom_boxplot()
ggplot(dataframe) + aes(x="Rooms") + geom_bar()
#es asimetrico a la derecha las habitaciones y su distribucion
'''

'''
housing.plot(kind="scatter", x="longitude", y="latitude")


housing.plot(kind="scatter", x="longitude", y="latitude", alpha=0.4,
s=housing["population"]/100, label="population", figsize=(10,7),
c="median_house_value", cmap=plt.get_cmap("jet"), colorbar=True,
)
plt.legend()'''
'''import pdb;pdb.set_trace()
corr_matrix=dataframe.corr(method='pearson')
corr_matrix["Price"].sort_values(ascending=False)
#dataframe.hist(bins=50, figsize=(20,15))
#plt.show()'''
import pdb;pdb.set_trace()

#COGEMOS LOS QUE TENGAN MAS CORRELACION ENTRE ELLOS Y HACEMOS LO SIGUIENTE PARA VER COMO ES LA CORRELACION

corr_matrix=dataframe.corr(method='pearson')         
corr_matrix["Price"].sort_values(ascending=False)

attributes = ['Price',"Rooms", "Bedroom2", "Bathroom","Car",'Lattitude','YearBuilt']
scatter_matrix(dataframe[attributes], figsize=(12, 8))
plt.show()
import pdb;pdb.set_trace()


#DESPUES DE VERLAS TODAS NOS CENTRAMOS EN LA QUE TIENE MAS RELACION CON EL PRECIO Y HACEMOS LO SIGUIENTE 
#OBSERVAR SI HAY LINEAS HORIZONTALES EN LA GRAFICA Y ANALIZAR SI ES CONVENIENTE ELIMINAR CIERTOS DATOS DE AHI
plt.plot(dataframe['Rooms'],dataframe['Price'],"ro")
plt.ylabel('Price')
plt.xlabel('Romms')
plt.plot(dataframe['Distance'],dataframe['Price'],"bo")
plt.ylabel('Price')
plt.xlabel('Distance')
plt.plot(dataframe['YearBuilt'],dataframe['Price'],"bo")
plt.ylabel('Price')
plt.xlabel('YearBuilt')
dataframe=dataframe.assign(Bathroom_per_rooms=dataframe['Bathroom']/dataframe['Rooms'])
#despues experimentar combinando diferentes tipos de datos a ver cual es la correlacion con estos  Bathroom
#df_comunidad = df_comunidad.assign(num_casos_prueba_pcr=df_aux2['num_casos_prueba_pcr'])

for i in dataframe['Suburb']:
    dataframe_aux=dataframe[dataframe.Suburb ==i]
    mean_room=dataframe_aux['Rooms'].mean()*dataframe_aux['Propertycount']
    dataframe_aux=dataframe_aux.assign(rooms_per_Suburb=mean_room)
    dataframe_aux=dataframe_aux.assign(Total_rooms_Suburb=dataframe_aux['Rooms'].sum())
    dataframe_aux=dataframe_aux.assign(pr_rooms_Suburb=100*round(dataframe['Rooms']/dataframe_aux['Total_rooms_Suburb'],6))
    dataframe_bueno=pd.concat([dataframe_bueno, dataframe_aux], axis=1,join='inner')
import pdb;pdb.set_trace()

corr_matrix=dataframe.corr(method='pearson')         
corr_matrix["Price"].sort_values(ascending=False)
#tipos_columnas=clasificar_variables(dataframe)

import pdb;pdb.set_trace()
duplicateRowsDF = dataframe_bueno[dataframe_bueno.duplicated(['Suburb', 'Address','Postcode','CouncilArea'])]
duplicateRowsDF.value_counts().sum()
 #aqui vemos que hay varios casos en donde no se pudo vender la casa y otro vendedor si que pudo
dataframe_bueno=dataframe_bueno[dataframe_bueno.Address =='15 Incana Dr']

dataframe_aux=dataframe_bueno[dataframe_bueno.Address =='17 Talofa Av']
dataframe_aux=dataframe_bueno[dataframe_bueno.Address =='22 Yongala St']
dataframe_aux=dataframe_bueno[dataframe_bueno.Address =='26 Grace St']
#26 Grace St, Yarraville VIC, Australia  aqui se puede ver que es una casa y que ha habido diferentes ventas por el mismo precio en diferentes fechas por el mismo comprados, duplicados

#lo mismo paa con yongala
duplicateRowsDF = dataframe_bueno[dataframe_bueno.duplicated(['Suburb', 'Address','Postcode','CouncilArea',],keep=False)]
duplicateRowsDF=duplicateRowsDF.drop_duplicates(subset=['Address','SellerG','Price','Date'])
duplicateRowsDF=duplicateRowsDF.dropna(subset=['Price'])
dataframe_bueno=dataframe_bueno.drop_duplicates(subset=['Address','Suburb'], keep=False, ignore_index=True)
dataframe_bueno=pd.concat([dataframe_bueno, duplicateRowsDF], axis=1,join='inner')
#diferentes dias de ventas diferentes metodos de ventas diferentes numero de habitaciones diferentes precios misma casa 
dataframe_aux=dataframe_bueno[dataframe_bueno.Address =='118 Westgarth St']
dataframe_aux=duplicateRowsDF[duplicateRowsDF.Address =='11 Harrington Rd']
