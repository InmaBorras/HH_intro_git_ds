import pandas as pd 
import numpy as np
from plotnine import ggplot, aes, geom_line, geom_point, geom_bar, geom_boxplot
import matplotlib.pyplot as plt
from pandas.plotting import scatter_matrix
import missingno as msno
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import KFold
from sklearn.ensemble import RandomForestRegressor
from sklearn.feature_selection import SelectKBest
import statsmodels.api as sm

#import fancyimpute
'''
fs = SelectKBest(score_func=f_regression, k='all')
	# learn relationship from training data
	fs.fit(X_train, y_train)
	# transform train input data
	X_train_fs = fs.transform(X_train)
	# transform test input data
	X_test_fs = fs.transform(X_test)

    
'''
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

def error_cuadratico(y_true,y_pred):
    error_cuadratico_sub=0
    for index,y_sub_pred in enumerate(y_pred):
        sumatorio = 0  
        n = len(y_sub_pred) 
        y_true[index]=np.nan_to_num(y_true[index])
        y_sub_pred=np.nan_to_num(y_sub_pred)
        for i in range (0,n):  
            difference = y_true[index][i] - y_sub_pred[i]  
            diferencia_cuadratica = difference**2  
            sumatorio = sumatorio + diferencia_cuadratica 
        error_cuadratico_sub = error_cuadratico_sub+sumatorio/n
    import pdb;pdb.set_trace() 
    error_cuadratico=round(error_cuadratico_sub/len(y_pred),4)
    return(error_cuadratico)
'''import pdb;pdb.set_trace()
lista_true=[[5,7,9,11,13],[2,4,6,8,10]]
lista_pred=[[19/4,40/6,8,12,13],[5/2,7/2,13/2,17/2,9]]
ec_prueba=error_cuadratico(lista_true,lista_pred)'''
dataframe_bueno=pd.DataFrame()
#dataframe=dataframe[dataframe.BuildingArea.drop()]
dataframe_final=pd.DataFrame()
dataframe=dataframe.dropna(subset=['Price'])
for i in dataframe.keys():
    if(i!='YearBuilt' and i!='BuildingArea' and i!='Bedroom2'):
        dataframe_final[i]=dataframe[i]
dataframe=dataframe_final
lista_parametros=list()
lista_missings=list()
for index, i in enumerate(dataframe.keys()):
    if(isinstance(dataframe.iloc[0][index], ( np.int64))  or isinstance(dataframe.iloc[0][index],(np.float64)) ):
        if(dataframe[str(i)].isna().sum()>1000):
            lista_missings.append(i)
        else:
            lista_parametros.append(i)
#deter_data = pd.DataFrame(columns = ["Det" + name for name in missing_columns])
'''
#analisis bivariante explicativo para ver las interacciones entre las variables   standar scaler  restando la media dividiendo ppor la desviacion tipica
#categorizar  CATEGORIZAR
predictions= model.summary()      
predictions
dataframe_final =pd.Dataframe()
for index, i in dataframe.keys(): dataframe_final[i]=dataframe[]
dataframe =pd.Dataframe()
dataframe = dataframe.dropna(subset=["BuildingArea", "YearBuilt"],axis=1)
mod = sm.OLS(X_train.endog, y_train.exog)

res = mod.fit()
'''
def prediccion_varias_variables(dataframe,df_filter,feature_to_predict,features,df_missings):
    kf = KFold(n_splits=5, random_state = 42,shuffle=True)
    y_pred_lineal=[]
    y_true_l=[]
    y_true_r=[]
    y_pred_rf=[]
    #kfold para dividir el dataframe en 10 partes y utilizar 9 como training y la otra como test
    for train_index, test_index in kf.split(df_filter):
        df_test = df_filter.iloc[test_index]
        df_train = df_filter.iloc[train_index]
        #creamos datos entrenamiento y test lineales .reshape(-1, 1) 
        X_train = np.array(df_train[features])   
        y_train = np.array(df_train[str(feature_to_predict)])
        X_test = np.array(df_test[features])
        y_test_l = np.array(df_test[str(feature_to_predict)])
        X_train=np.nan_to_num(X_train)
        y_train=np.nan_to_num(y_train)
        X_test=np.nan_to_num(X_test)
        #modelo regresion lineal
        #en regresion lineal no funciona debido a que hay que ir variable a variable sino no encajan el numero de x_train e y_train, hacer for o algo
        model=LinearRegression()
        model.fit(X_train,y_train)
        predictions = model.predict(X_test)
        y_pred_lineal.append(model.predict(X_test))
        y_true_l.append(y_test_l)
        #creamos test  y training de random forest y vamos viendo lo que ocurre  X_test = np.squeeze(np.asarray(X_test))
        #modelo random forest
        model_rf = RandomForestRegressor(n_estimators = 1000, max_depth = 1000, random_state = 42)
        model_rf.fit(X_train, y_train)
        y_pred_rf.append(model_rf.predict(X_test))

    import pdb;pdb.set_trace()
    ec_lineal=error_cuadratico(y_true_l,y_pred_lineal)
    ec_forest=error_cuadratico(y_true_l,y_pred_rf)
    X_train = np.array(df_filter[features])   
    y_train = np.array(df_filter[str(feature_to_predict)])
    x_test_missing =  np.array(df_missings[features])
    X_train=np.nan_to_num(X_train)
    y_train=np.nan_to_num(y_train)
    x_test_missing=np.nan_to_num(x_test_missing)
    if(ec_lineal>ec_forest):
        # si el error cuadratico de random forest es mayor no hacemos el reshape para el modelo.SSSS  np.array(df_missings[features])
        model_rf = RandomForestRegressor(n_estimators = 1000, max_depth = 1000, random_state = 42)
        model_rf.fit(X_train, y_train)  
        y_pred_missing = model_rf.predict(x_test_missing)
    else:
        model=LinearRegression()
        model.fit(X_train,y_train)
        y_pred_missing = model.predict(x_test_missing)
    import pdb;pdb.set_trace()
    #df_missings[str(feature_to_predict) + '_imp']=y_pred_missing
    return(y_pred_missing)

for index, j in enumerate(lista_missings):
    dataframe[str(j) + '_imp'] = dataframe[str(j)]
    #deter_data["Det" + str(j)] = dataframe[str(j) + "_imp"]
    df_missings=dataframe[dataframe.isnull().any(axis=1)].copy()
    #df_missings=dataframe[dataframe[str(j)] == np.nan].copy()
    #df_missings=dataframe[dataframe[lista_parametros] != np.nan].copy()
    df_filter = dataframe[dataframe[str(j)] != np.nan].copy()
    corr_matrix=dataframe.corr(method='pearson')         
    max_corr=corr_matrix[str(j)].sort_values(ascending=False)
    import pdb;pdb.set_trace()
    prediccion=prediccion_varias_variables(dataframe,df_filter,str(j),lista_parametros,df_missings)
    lista_parametros.append(str(j))
    model = linear_model.LinearRegression()
    model.fit(X = df[parameters], y = df[feature + '_imp'])
    

import pdb;pdb.set_trace()

#COGEMOS LOS QUE TENGAN MAS CORRELACION ENTRE ELLOS Y HACEMOS LO SIGUIENTE PARA VER COMO ES LA CORRELACION

corr_matrix=dataframe.corr(method='pearson')         
corr_matrix["Price"].sort_values(ascending=False)

attributes = ['Price',"Rooms", "Bedroom2", "Bathroom","Car",'Lattitude','YearBuilt']
#scatter_matrix(dataframe[attributes], figsize=(12, 8))
#plt.show()
import pdb;pdb.set_trace()
#habria que borrar estas filas ya que se carga la visualizacion
#hacer una funcion que pase el nombre de la columna y el valor a partir del cual quiere borrar y eliminar todos los datos
dataframe_aux=dataframe[dataframe.YearBuilt <1800]
dataframe=dataframe.drop(dataframe_aux.index) 
#DESPUES DE VERLAS TODAS NOS CENTRAMOS EN LA QUE TIENE MAS RELACION CON EL PRECIO Y HACEMOS LO SIGUIENTE 
#OBSERVAR SI HAY LINEAS HORIZONTALES EN LA GRAFICA Y ANALIZAR SI ES CONVENIENTE ELIMINAR CIERTOS DATOS DE AHI


dataframe=dataframe.assign(Bathroom_per_rooms=dataframe['Bathroom']/dataframe['Rooms'])
#despues experimentar combinando diferentes tipos de datos a ver cual es la correlacion con estos  Bathroom
#df_comunidad = df_comunidad.assign(num_casos_prueba_pcr=df_aux2['num_casos_prueba_pcr'])

for i in dataframe['Suburb'].value_counts().index:
    dataframe_aux=dataframe[dataframe.Suburb ==i]
    mean_room=dataframe_aux['Rooms'].mean()*dataframe_aux['Propertycount']
    dataframe_aux=dataframe_aux.assign(rooms_per_Suburb=mean_room)
    dataframe_aux=dataframe_aux.assign(Total_rooms_Suburb=dataframe_aux['Rooms'].sum())
    dataframe_aux=dataframe_aux.assign(pr_rooms_Suburb=100*round(dataframe['Rooms']/dataframe_aux['Total_rooms_Suburb'],6))
    frames = [dataframe_bueno, dataframe_aux]
    dataframe_bueno = pd.concat(frames)
    dataframe_bueno=dataframe_bueno.reset_index(drop=True)
import pdb;pdb.set_trace()

corr_matrix=dataframe_bueno.corr(method='pearson')         
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
dataframe_aux=dataframe[dataframe.Address =='22 Yongala St']

#lo mismo paa con yongala
duplicateRowsDF = dataframe_bueno[dataframe_bueno.duplicated(['Suburb', 'Address','Postcode','CouncilArea',],keep=False)]
duplicateRowsDF=duplicateRowsDF.drop_duplicates(subset=['Address','SellerG','Price','Date'])
duplicateRowsDF=duplicateRowsDF.dropna(subset=['Price'])
dataframe_bueno=dataframe_bueno.drop_duplicates(subset=['Address','Suburb'], keep=False, ignore_index=True)
dataframe_bueno=pd.concat([dataframe_bueno, duplicateRowsDF], axis=1,join='inner')
dataframe_bueno=dataframe_bueno.dropna(subset=['Price'])
missings_number=dict()
for index, i in enumerate(dataframe.keys()):
    if(dataframe_bueno[str(i)].isnull().value_counts()[0]!=len(dataframe_bueno)):
        missings_number[str(i)]=100*round(dataframe_bueno[str(i)].isnull().value_counts()[1]/len(dataframe_bueno),5)

#dataframe.isnull().value_counts()
#diferentes dias de ventas diferentes metodos de ventas diferentes numero de habitaciones diferentes precios misma casa 
dataframe_aux=dataframe_bueno[dataframe_bueno.Address =='118 Westgarth St']
dataframe_aux=duplicateRowsDF[duplicateRowsDF.Address =='11 Harrington Rd']


import pdb;pdb.set_trace()

#ELIMINACION VISUALIZACION Y SUSTITUCION DE LOS MISSINGS
dataframe_bueno.isna().sum()
msno.matrix(dataframe_bueno)
msno.heatmap(dataframe_bueno)
# Missingno library also provides heatmaps that show if there is any correlation between missing values in different columns.
msno.bar(dataframe_bueno)


import pdb;pdb.set_trace()
#crear una funcion que te cree las columnas que he hecho combinando datos y 
#otra funcion que te elimine las columnas en funcion a la correlacion,  cuando sea menor de 0.1 en valor absoluto preguntar si si o no eliminarla

'''mirar las paginas pero creo que puede funcionar, crear lista con las columnas que no tienen missings y con ellas hacer un coeficiente de correlacion
con las columnas con missings, con ello, podemos hacer un modelo de regresion  lineal con ello'''

'''
feature_help=max_corr.keys()[1]
def prediccion_una_variables(dataframe,df_filter,feature_to_predict,feature_help=max_corr.keys()[1],deter_data,lista_parametros):
    kf = KFold(n_splits=10, random_state = 42)
    y_pred_lineal=[]
    y_true=[]
    y_pred_forest=[]
    #kfold para dividir el dataframe en 10 partes y utilizar 9 como training y la otra como test
    for train_index, test_index in kf.split(df_filter):
        df_test = df_filter.iloc[test_index]
        df_train = df_filter.iloc[train_index]
        X_train = np.array(df_train[str(feature_help)]).reshape(-1, 1)     
        y_train = np.array(df_train[str(feature_to_predict)]).reshape(-1, 1)
        X_test = np.array(df_test[str(feature_help)]).reshape(-1, 1)  
        y_test = np.array(df_test[str(feature_to_predict)]).reshape(-1, 1)
        #random forest y regresion lineal
        X_test_missing = np.array(dataframe[str(feature_help) + '_imp']).reshape(-1,1)
        y_test_missing = np.array(dataframe[str(feature_to_predict) + '_imp']).reshape(-1, 1)
        model = LinearRegression()
        model.fit(X_train, y_train)
        y_pred.append(model.predict(X_test)[0])
        y_true.append(y_test[0])

        X_test_rf = np.array(dataframe[lista_parametros])
        X_train_rf = np.array(df_filter[lista_parametros])
        y_train_rf = np.array(df_filter[feature_to_predict])
        model_rf = RandomForestRegressor(n_estimators = 1000, max_depth = 1000, random_state = 42)
        y_pred_forest.append(model_rf.fit(X_train_rf, y_train_rf))
df[df.isnull().any(axis=1)]
    return(y_pred,y_true)
plt.plot(dataframe['Rooms'],dataframe['Price'],"ro")
plt.ylabel('Price')
plt.xlabel('Romms')
plt.show()
plt.plot(dataframe['Distance'],dataframe['Price'],"bo")
plt.ylabel('Price')
plt.xlabel('Distance')
plt.show()
plt.plot(dataframe['YearBuilt'],dataframe['Price'],"bo")
plt.ylabel('Price')
plt.xlabel('YearBuilt')
plt.show()'''


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