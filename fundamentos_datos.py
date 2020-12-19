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
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import RepeatedKFold
from sklearn.linear_model import Lasso
from sklearn.metrics import mean_squared_error
from scipy import stats
import statsmodels.api as sm
import statsmodels.formula.api as smf
import pwlf
from sklearn.model_selection import train_test_split
from statsmodels.stats.outliers_influence import OLSInfluence
import itertools    
from geopy import distance
#import fancyimpute
#statsmodels.regression.linear_model.OLSResults.aic     statsmodels.tools.eval_measures.aic #categorizar las variables
#mirar f test en el modelo   anova_lv mirarlo a ver que podemos hacer al comparar los modelos

'''
name = ['Jarque-Bera', 'Chi^2 two-tail prob.', 'Skew', 'Kurtosis']
test = sms.jarque_bera(mod.fit().resid)
lzip(name, test)



fs = SelectKBest(score_func=f_regression, k='all')
	# learn relationship from training data
	fs.fit(X_train, y_train)
	# transform train input data
	X_train_fs = fs.transform(X_train)
	# transform test input data
	X_test_fs = fs.transform(X_test)

    
'''

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
                    if(esAcotada(minimo,maximo)):
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
                if(esAcotada(minimo,maximo)):
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
def eliminar_duplicados(dataframe_bueno):
    import pdb;pdb.set_trace()
    duplicateRowsDF=pd.DataFrame()
    duplicateRowsDF = dataframe_bueno[dataframe_bueno.duplicated(['Suburb', 'Address','Postcode','CouncilArea',],keep=False)]
    duplicateRowsDF=duplicateRowsDF.drop_duplicates(subset=['Address','Price','Date'])
    duplicateRowsDF=duplicateRowsDF.dropna(subset=['Price'])
    dataframe_bueno=dataframe_bueno.drop_duplicates(subset=['Address','Suburb'], keep=False, ignore_index=True)
    dataframe_bueno.append(duplicateRowsDF)
    dataframe_bueno=dataframe_bueno.dropna(subset=['Price'])
    dataframe_bueno=dataframe_bueno.reset_index(drop=True)
    return(dataframe_bueno)
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
    error_cuadratico=round(error_cuadratico_sub/len(y_pred),4)
    return(error_cuadratico)
def alfa_optima(X,y):
    modelo_lasso = Lasso()
    # definimos el metodo de evaluacion del modelo_lassoo
    cv = RepeatedKFold(n_splits=10, n_repeats=3, random_state=1)
    grid = dict()
    import pdb;pdb.set_trace() 
    grid['alpha'] = np.arange(0, 0.01, 0.0001)
    busqueda = GridSearchCV(modelo_lasso, grid, scoring='neg_mean_absolute_error', cv=cv, n_jobs=-1)
    results = busqueda.fit(X, y)
    import pdb;pdb.set_trace()
    return(results.best_params_['alpha'],results.best_score_)
    
def lasso_prueba(X,y,lista_parametros):
    import pdb;pdb.set_trace()
    alfa_op, alfa_score=alfa_optima(X,y)
    clf = Lasso(alpha=alfa_op)
    import pdb;pdb.set_trace()
    clf.fit(X,y)
    coeficientes=clf.coef_

    for index,i in enumerate(coeficientes):
        import pdb;pdb.set_trace()
        if(round(i,1)==0):
            print('hay que borrar el parametro: '+str(lista_parametros[index]) )
            lista_parametros.pop(index)
    print(clf.intercept_)
    return(lista_parametros)

def info_del_modelo(dataframe,feature_to_predict,X,y):
    import pdb;pdb.set_trace()
    mod = smf.ols(formula='Price ~ Bathroom_times_rooms +Car+ Distance +Lattitude+Longtitude', data=dataframe)
    res = mod.fit()
    #print(mod.fvalue, mod.f_pvalue)
    #probablemente no es normal asi que hay que recurrir a un test de permutaciones, seguir indagando.   best score -341640.2872449162  dataframe['Bathroom']
    print(res.summary())
def prediccion_por_intervalos(x,y):
    import pdb;pdb.set_trace()
    #si sabemos el numero especifico de veces que cambia el gradiente podemos utilizar fitfast https://jekel.me/piecewise_linear_fit_py/examples.html
    my_pwlf = pwlf.PiecewiseLinFit(x, y)
    breaks = my_pwlf.fit(2)
    print('el gradiente cambia en: '+str(breaks))
    x_hat = np.linspace(x.min(), x.max(), num=4)
    y_hat = my_pwlf.predict(x_hat)
    plt.figure()
    plt.plot(x, y, 'o')
    plt.plot(x_hat, y_hat, '-')
    plt.show()
def processSubset(X,y,feature_set):
    # Fit model on feature_set and calculate RSS
    model = sm.OLS(y,X[list(feature_set)])
    regr = model.fit()
    RSS = ((regr.predict(X[list(feature_set)]) - y) ** 2).sum()
    aic=regr.aic
    return {"model":regr, "RSS":RSS,'AIC':aic}
def processSubset_tr(y,X,tr):
    # Fit model on feature_set and calculate RSS
    model = sm.OLS(y,X)
    regr = model.fit()
    RSS = (regr.predict(X - y) ** 2).sum()
    aic=regr.aic
    r_squared=regr.r_squared
    return {"model":regr, "RSS":RSS,'AIC':abs(aic),'transformacion':tr}
def getBest(dataframe,y,k):
    results = []
    for combo in itertools.combinations(dataframe.columns, k):
        results.append(processSubset(dataframe,y,combo))
    # Wrap everything up in a nice dataframe
    models = pd.DataFrame(results)
    # Choose the model with the highest RSS
    best_model = models.loc[models['RSS'].argmin()]
    
    # Return the best model, along with some other useful information about the model  
    return best_model
def Best_stepwise_selection(dataframe,X):
    y=dataframe['Price']
    models_best = pd.DataFrame(columns=["RSS", "model"])
    import pdb;pdb.set_trace()
    for i in range(1,len(X.columns)):
        models_best.loc[i] = getBest(X,y,i)
        if(len(models_best)>1):
            if(round(models_best.loc[i, "model"].rsquared-models_best.loc[i-1, "model"].rsquared,4)<0.001):
                break
    import pdb;pdb.set_trace()
#lista_parametros=['Rooms', 'Distance', 'Bathroom', 'Car', 'Lattitude', 'Longtitude', 'Landsize', 'Propertycount'] X=dataframe[lista_parametros]
 #X=dataframe[lista_parametros]    y=np.sqrtdataframe['Price']
def modelo_lineal(dataframe,features):
    X=dataframe[features] 
    y=np.sqrt(dataframe['Price'])
    X=np.nan_to_num(X)
    y=np.nan_to_num(y)
    import pdb;pdb.set_trace()
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=50)
    features=lasso_prueba(X,y,features)
    import pdb;pdb.set_trace()
    #prediccion_por_intervalos(X,y)
    import pdb;pdb.set_trace()
    model=LinearRegression()
    model.fit(X_train,y_train)

    #predictions = model.predict(X_test)
    #RSS = mean_squared_error(y_test,predictions) * len(y_test)
    #R_squared = model.score(X,y)
    #ec_modelo=mean_squared_error(y_test,predictions)
    #pendiente, intercepto, r_value, p_value, std_err = stats.linregress(X_train,y_train)

    info_del_modelo(dataframe,'Price',X_train,y_train)
    import pdb;pdb.set_trace()
    features=lasso_prueba(X_train,y_train,features)
def distancia_optimizada(dataframe):
    #suburbio mas caro
    data_frameKoyong=dataframe[dataframe["Suburb"]=='Kooyong']
    Koyong_lat_mean=data_frameKoyong["Lattitude"].mean()
    Koyong_lon_mean=data_frameKoyong["Longtitude"].mean()
    
    #Calculamos la nueva distancia
    arr=[]
    for lat, lon in zip(dataframe['Lattitude'],dataframe['Longtitude']):
        arr.append(distance((lat,lon), (Koyong_lat_mean,Koyong_lon_mean)).km)
    return(arr)
def visualizacion_missings(dataframe):
    #ELIMINACION VISUALIZACION Y SUSTITUCION DE LOS MISSINGS
    print(dataframe.isnull().sum())
    msno.matrix(dataframe)
    plt.show()
    msno.heatmap(dataframe)
    plt.show()
    # Missingno library also provides heatmaps that show if there is any correlation between missing values in different columns.
    msno.bar(dataframe)
    plt.show()
'''import pdb;pdb.set_trace()
lista_true=[[5,7,9,11,13],[2,4,6,8,10]]
lista_pred=[[19/4,40/6,8,12,13],[5/2,7/2,13/2,17/2,9]]
ec_prueba=error_cuadratico(lista_true,lista_pred)'''   
dataframe_cualitativas = pd.read_csv('precios_casas_Cualitativas.csv')
dataframe_cualitativas2=pd.read_csv('precios_casas.csv')
dataframe3 = pd.read_csv('precios_casas_full.csv')
dataframe_bueno=pd.DataFrame()
#dataframe=dataframe[dataframe.BuildingArea.drop()]
dataframe_final=pd.DataFrame()
#dataframe=dataframe.dropna(subset=['Price'])
'''
for i in dataframe.keys():
    if(i!='YearBuilt' and i!='BuildingArea' and i!='Bedroom2'):
        dataframe_final[i]=dataframe[i]
dataframe=dataframe_final'''
lista_parametros=list()
lista_missings=list()
for index, i in enumerate(dataframe3.keys()):
    if(isinstance(dataframe3.iloc[0][index], ( np.int64))  or isinstance(dataframe3.iloc[0][index],(np.float64)) ):
        if(dataframe3[str(i)].isna().sum()>1000):
            lista_missings.append(i)
        else:
            lista_parametros.append(i)

'''
#deter_data = pd.DataFrame(columns = ["Det" + name for name in missing_columns])

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
def lista_transformaciones(dataframe,y,feature):
    tr=list()
    tr.append(processSubset_tr(y,dataframe[feature],'none'))
    tr.append(processSubset_tr(y,np.log(dataframe[feature]),'ln'))
    tr.append(processSubset_tr(y,np.log10(dataframe[feature]),'log10'))
    tr.append(processSubset_tr(y,1/dataframe[feature],'inversa'))
    tr.append(processSubset_tr(y,1/np.sqrt(dataframe[feature]),'inversa_sqrt'))
    tr.append(processSubset_tr(y,np.sqrt(dataframe[feature]),'sqrt'))
    return(tr)
def transformaciones_variables(dataframe):
    dict_tr=dict()
    for  i in dataframe.keys():
        if((isinstance(dataframe[str(i)].iloc[0], ( np.int64))  or isinstance(dataframe[str(i)].iloc[0],(np.float64))) and i!='Lattitude' and i!='Location_TRA' ):
            transformaciones=lista_transformaciones(dataframe,dataframe['Price'],str(i))
            models_tr = pd.DataFrame(transformaciones)
            best_model_aic = models_tr.loc[models_tr['AIC'].argmin()]
            best_model_RS = models_tr.loc[models_tr['RSS'].argmin()]
            dict_tr[str(i)+'_aic']=best_model_aic
            dict_tr[str(i)+'_rss']=best_model_RS
            import pdb;pdb.set_trace()

    import pdb;pdb.set_trace()

    return(dict_tr)
def prediccion_varias_variables(dataframe,df_filter,feature_to_predict,features,df_missings):
    kf = KFold(n_splits=5, random_state = 42,shuffle=True)
    y_pred_lineal=[]
    y_true_l=[]
    y_true_r=[]
    y_pred_rf=[]
    #kfold para dividir el dataframe en 5 partes y utilizar 4 como training y la otra como test
    for train_index, test_index in kf.split(df_filter):
        import pdb;pdb.set_trace()
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
#visualizacion_missings(dataframe3)

#visualizacion_missings(dataframe)
'''
import pdb;pdb.set_trace()
for index, j in enumerate(lista_missings):
    #deter_data["Det" + str(j)] = dataframe[str(j) + "_imp"]      
    df_missings=dataframe[dataframe[j].isnull()].copy()
    #df_missings=dataframe[dataframe[str(j)] == np.nan].copy()
    #df_missings=dataframe[dataframe[lista_parametros] != np.nan].copy()         prediccion_redondeada = [round(num, 5) for num in y_pred_missing]
    df_filter = dataframe[dataframe[str(j)] != np.nan].copy()
    corr_matrix=dataframe.corr(method='pearson')         
    max_corr=corr_matrix[str(j)].sort_values(ascending=False)
    import pdb;pdb.set_trace()
    prediccion=prediccion_varias_variables(dataframe,df_filter,str(j),lista_parametros,df_missings)
    prediccion_redondeada = [round(num, 1) for num in prediccion]
    prediccion_redondeada = [round(num, 0) for num in prediccion_redondeada]
    df_missings['prediccion']=prediccion_redondeada
    dataframe[str(j)].fillna(df_missings['prediccion'], inplace = True)
    import pdb;pdb.set_trace()
    #dataframe[dataframe[str(j)].isnull().any(axis=1)]   dataframe.to_csv('precios_casas.csv', index=False)
    lista_parametros.append(str(j))
    
dataframe=eliminar_duplicados(dataframe)
visualizacion_missings(dataframe)
'''




#COGEMOS LOS QUE TENGAN MAS CORRELACION ENTRE ELLOS Y HACEMOS LO SIGUIENTE PARA VER COMO ES LA CORRELACION

#corr_matrix=dataframe.corr(method='pearson')         
#corr_matrix["Price"].sort_values(ascending=False)

#attributes = ['Price',"Rooms", "Bedroom2", "Bathroom","Car",'Lattitude','YearBuilt']
#scatter_matrix(dataframe[attributes], figsize=(12, 8))
#plt.show()

dataframe= pd.read_csv('precios_casas_sinduplicados.csv')
#dataframe['Price']=dataframe3['Price']
dataframe=eliminar_duplicados(dataframe)
#dataframe["Distancia_NEW"]=distancia_optimizada(dataframe)
#dataframe["Location_TRA"]=dataframe.Longtitude/dataframe.Lattitude
#dataframe_less=dataframe[(dataframe["Location_TRA"]>=dataframe["Location_TRA"].mean()) ]
#
# dataframe_over=dataframe[(dataframe["Location_TRA"]<dataframe["Location_TRA"].mean()) ]

dataframe=dataframe.dropna(subset=['Price']) 
dataframe=dataframe.dropna(subset=['Distance']) 
dataframe=dataframe.dropna(subset=['Regionname']) 
dataframe=dataframe.dropna(subset=['Regionname']) 
dataframe=dataframe.dropna(subset=['CouncilArea']) 
dataframe=dataframe.dropna(subset=['Propertycount']) 
dataframe=dataframe.dropna(subset=['Postcode']) 


#habria que borrar estas filas ya que se carga la visualizacion
#hacer una funcion que pase el nombre de la columna y el valor a partir del cual quiere borrar y eliminar todos los datos
#dataframe_aux=dataframe[dataframe.YearBuilt <1800]
#dataframe=dataframe.drop(dataframe_aux.index) 
#DESPUES DE VERLAS TODAS NOS CENTRAMOS EN LA QUE TIENE MAS RELACION CON EL PRECIO Y HACEMOS LO SIGUIENTE    dataframe=dataframe[dataframe['Rooms']>10]
   
#OBSERVAR SI HAY LINEAS HORIZONTALES EN LA GRAFICA Y ANALIZAR SI ES CONVENIENTE ELIMINAR CIERTOS DATOS DE AHI dataframe_aux=dataframe[dataframe['Bathroom']>4]
import pdb;pdb.set_trace()
dataframe['Price']=np.log10(dataframe['Price'])

#dataframe['Price']=np.sqrt(np.log10(dataframe['Price']))
#ataframe['Price']=1/np.log(np.sqrt(dataframe['Price']))

dataframe=dataframe[dataframe['Distance']<40]
dataframe=dataframe[dataframe['Rooms']<10]
dataframe=dataframe[dataframe['Bathroom']<5]
#dataframe=dataframe[dataframe['Type']=='h']
#dataframe=dataframe[dataframe['Regionname']=='Southern Metropolitan']
#el aic se ha reducido con esto de 70000 a 13000

dataframe=dataframe.assign(Bathroom_times_rooms=dataframe['Bathroom']*dataframe['Rooms'])
dataframe=dataframe[dataframe['Landsize']>0] 
#dataframe['Bathroom']=dataframe['Bathroom'].replace(0, 1)
#dataframe['Bathroom_times_rooms']=dataframe['Bathroom_times_rooms'].replace(0, 1)
#dataframe['Lattitude']=dataframe['Lattitude'].replace(0, 1)
dataframe['Longtitude']=dataframe['Longtitude'].replace(0, 1)
#dataframe['Car']=dataframe['Car'].replace(0, 1)
#dataframe['Rooms']=np.sqrt(dataframe['Rooms'])
#dataframe['Distance']=np.sqrt(dataframe['Distance'])
dataframe['Landsize']=np.log(dataframe['Landsize'])
dataframe['Longtitude']=np.log(dataframe['Longtitude'])
dataframe['Propertycount']=np.log(dataframe['Propertycount'])
dataframe['Propertycount']=np.log(dataframe['Propertycount'])
#dataframe['Bathroom_times_rooms']=np.sqrt(dataframe['Bathroom_times_rooms'])
#despues experimentar combinando diferentes tipos de datos a ver cual es la correlacion con estos  Bathroom
#df_comunidad = df_comunidad.assign(num_casos_prueba_pcr=df_aux2['num_casos_prueba_pcr'])    "Regionname_Southern Metropolitan"   

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
lista_parametros=['Rooms', 'Distance','Bathroom_times_rooms', 'Bathroom', 'Car', 'Lattitude', 'Longtitude', 'Total_rooms_Suburb', 'Landsize', 'Propertycount','rooms_per_Suburb','pr_rooms_Suburb']
modelo_lineal(dataframe_bueno,lista_parametros)

#transformaciones_variables(dataframe)
import pdb;pdb.set_trace()
X=dataframe_bueno[lista_parametros]
#lista_parametros=['Rooms', 'Distance', 'Bathroom', 'Car', 'Lattitude', 'Longtitude', 'Total_rooms_Suburb', 'Landsize', 'Propertycount']
#Bathroom_times_rooms + Distance   Bathroom_times_rooms +Car+ Landsize +Propertycount    model = sm.OLS(y, sm.add_constant(X, prepend=False))
Best_stepwise_selection(dataframe_bueno,X)
import pdb;pdb.set_trace()

'''
dataframe_bueno['Rooms']=dataframe_bueno['Rooms'].replace(0, 1)
dataframe_bueno['Rooms']=np.log(dataframe_bueno['Rooms'])
dataframe_bueno['Distance']=dataframe_bueno['Distance'].replace(0, 1)
dataframe_bueno['Distance']=np.log(dataframe_bueno['Distance'])

dataframe_bueno['Landsize']=dataframe_bueno['Landsize'].replace(0, 1)
dataframe_bueno['Landsize']=np.log(dataframe_bueno['Landsize'])
dataframe_bueno['Longtitude']=dataframe_bueno['Longtitude'].replace(0, 1)
dataframe_bueno['Longtitude']=np.log(dataframe_bueno['Longtitude'])
'''
lista_parametros=['Rooms', 'Distance', 'Bathroom']

lista_parametros=['Rooms', 'Distance','Bathroom_times_rooms', 'Bathroom', 'Car', 'Lattitude', 'Longtitude', 'Total_rooms_Suburb', 'Landsize', 'Propertycount','rooms_per_Suburb','pr_rooms_Suburb']
X=dataframe_bueno[lista_parametros]
#lista_parametros=['Rooms', 'Distance', 'Bathroom', 'Car', 'Lattitude', 'Longtitude', 'Total_rooms_Suburb', 'Landsize', 'Propertycount']
#Bathroom_times_rooms + Distance   Bathroom_times_rooms +Car+ Landsize +Propertycount    model = sm.OLS(y, sm.add_constant(X, prepend=False))
Best_stepwise_selection(dataframe_bueno,X)
#lista_parametros=['Bathroom_times_rooms', 'Car','Landsize', 'Propertycount']
modelo_lineal(dataframe_bueno,lista_parametros)

import pdb;pdb.set_trace()

corr_matrix=dataframe_bueno.corr(method='pearson')         
corr_matrix["Price"].sort_values(ascending=False)
#tipos_columnas=clasificar_variables(dataframe)  ['Bathroom', 'Car', 'Lattitude', 'Longtitude', 'Bathroom_per_rooms', 'Total_rooms_Suburb', 'Landsize']

#Funcion que pinta los cálculos de dispersion y asimetria de una columna de un dataframe
def mostrar_analisis_var_cuantitativas(data):
    #calcular coeficiente de variacion
 datos_variable=pd.DataFrame([{"coeficiente de Variacion":(data.std()/data.mean())*100,\
                 "rango de la variable":data.max() - dataframe["Rooms"].min(),
                 "rango intercuartilico":data.quantile(0.75) - data.quantile(0.25),
                 "coeficiente de asimetria":ss.skew(data)}])
 return(datos_variable)

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
missings_number=dict()
for index, i in enumerate(dataframe.keys()):
    if(dataframe_bueno[str(i)].isnull().value_counts()[0]!=len(dataframe_bueno)):
        missings_number[str(i)]=100*round(dataframe_bueno[str(i)].isnull().value_counts()[1]/len(dataframe_bueno),5)

#dataframe.isnull().value_counts()
#diferentes dias de ventas diferentes metodos de ventas diferentes numero de habitaciones diferentes precios misma casa 
dataframe_aux=dataframe_bueno[dataframe_bueno.Address =='118 Westgarth St']
dataframe_aux=duplicateRowsDF[duplicateRowsDF.Address =='11 Harrington Rd']


import pdb;pdb.set_trace()



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
#plt.show()
# 
# 
# 
# Omnibus / Prob (Omnibus): una prueba de la asimetría y curtosis del residual (característica n. ° 2). Esperamos ver un valor cercano a cero que indicaría normalidad. El Prob (Omnibus) realiza una prueba estadística que indica la probabilidad de que los residuos se distribuyan normalmente. Esperamos ver algo cercano a 1 aquí. En este caso, Omnibus es relativamente bajo y el Prob (Omnibus) es relativamente alto, por lo que los datos son algo normales, pero no del todo ideales. Un enfoque de regresión lineal probablemente sería mejor que una conjetura aleatoria, pero probablemente no tan bueno como un enfoque no lineal.

Sesgo: una medida de simetría de datos. Queremos ver algo cercano a cero, lo que indica que la distribución residual es normal. Tenga en cuenta que este valor también impulsa al Omnibus. Este resultado tiene un sesgo pequeño y, por lo tanto, bueno.

Curtosis: una medida de "picos" o curvatura de los datos. Los picos más altos conducen a una mayor curtosis. La mayor curtosis se puede interpretar como una agrupación más estrecha de residuos alrededor de cero, lo que implica un mejor modelo con pocos valores atípicos.

Durbin-Watson: pruebas de homocedasticidad (característica n. ° 3). Esperamos tener un valor entre 1 y 2. En este caso, los datos están cerca, pero dentro de límites.

Jarque-Bera (JB) / Prob (JB): como la prueba Omnibus en el sentido de que prueba tanto la inclinación como la curtosis. Esperamos ver en esta prueba una confirmación de la prueba Omnibus. En este caso lo hacemos.

Número de condición: esta prueba mide la sensibilidad de la salida de una función en comparación con su entrada (característica n. ° 4). Cuando tenemos multicolinealidad, podemos esperar fluctuaciones mucho mayores a pequeños cambios en los datos, por lo tanto, esperamos ver un número relativamente pequeño, algo por debajo de 30. En este caso, estamos muy por debajo de 30, lo que esperaríamos dado nuestro modelo solamente tiene dos variables y una es una constante.
# 
# 
# 
# 
# 
# 
# 
# '''
