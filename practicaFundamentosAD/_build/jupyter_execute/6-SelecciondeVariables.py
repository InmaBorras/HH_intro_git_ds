#!/usr/bin/env python
# coding: utf-8

# ## 6. Selección de Variables 

# Despues de tratar los datos  y de haber analizado las variables por separado procedemos a la selección de las mas adecuadas para la creación del modelo. 
# 
# En primer lugar procedemos a separar los datos en dos datasets, uno para hacer en entrenamiento del modelo y otro para hacer el test final. 

# In[1]:


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


# In[ ]:




