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
    #import pdb;pdb.set_trace()
    duplicateRowsDF=pd.DataFrame()
    duplicateRowsDF = dataframe_bueno[dataframe_bueno.duplicated(['Suburb', 'Address','Postcode','CouncilArea',],keep=False)]
    duplicateRowsDF=duplicateRowsDF.drop_duplicates(subset=['Address','Price','Date'])
    duplicateRowsDF=duplicateRowsDF.dropna(subset=['Price'])
    dataframe_bueno=dataframe_bueno.drop_duplicates(subset=['Address','Suburb'], keep=False, ignore_index=True)
    dataframe_bueno.append(duplicateRowsDF)
    dataframe_bueno=dataframe_bueno.dropna(subset=['Price'])
    dataframe_bueno=dataframe_bueno.reset_index(drop=True)
    return(dataframe_bueno)