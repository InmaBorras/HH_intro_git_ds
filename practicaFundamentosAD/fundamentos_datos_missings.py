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

 
dataframe_cualitativas = pd.read_csv('/home/inma/HH_intro_git_ds/precios_casas_Cualitativas.csv')
dataframe_cualitativas2=pd.read_csv('/home/inma/HH_intro_git_ds/precios_casas.csv')
dataframe3 = pd.read_csv('/home/inma/HH_intro_git_ds/precios_casas_full.csv')
dataframe_bueno=pd.DataFrame()
#dataframe=dataframe[dataframe.BuildingArea.drop()]
dataframe_final=pd.DataFrame()
#dataframe=dataframe.dropna(subset=['Price'])


for i in dataframe.keys():
    if(i!='YearBuilt' and i!='BuildingArea' and i!='Bedroom2'):
        dataframe_final[i]=dataframe[i]
dataframe=dataframe_final
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



dataframe=dataframe.dropna(subset=['Distance']) 
dataframe=dataframe.dropna(subset=['Regionname']) 
dataframe=dataframe.dropna(subset=['Regionname']) 
dataframe=dataframe.dropna(subset=['CouncilArea']) 
dataframe=dataframe.dropna(subset=['Propertycount']) 
dataframe=dataframe.dropna(subset=['Postcode'])
#visualizacion_missings(dataframe3)

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

