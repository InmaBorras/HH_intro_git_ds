#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Dec 19 11:45:06 2020

@author: inma
"""

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
   #alfa_op, alfa_score=alfa_optima(X,y)
    clf = Lasso(alpha=0.0002)
    #import pdb;pdb.set_trace()
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
 #X=dataframe[lista_parametros]    y=np.sqrtdataframe['Price'
 
 
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
 