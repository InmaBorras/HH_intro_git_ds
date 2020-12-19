#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Dec 19 01:56:46 2020

@author: inma
"""

import pandas as pd 
import numpy as np
import folium
import math
import time
from scipy import stats
pd.options.mode.chained_assignment = None  # default='warn'
from plotnine import ggplot, aes, geom_line, geom_point, geom_bar, geom_boxplot
dataframe = pd.read_csv('/home/inma/Master_Data_Science _Contenido/Fundamentos_de_Analisis _de_Datos/Practica/Datos/Melbourne_housing_FULL.csv')
import scipy.stats as ss
import matplotlib.pyplot as plot
import seaborn as sb
from seaborn import kdeplot
def quartile_skew(x):
  q = x.quantile([.25, .50, .75]) 
  return ((q[0.75] - q[0.5]) - (q[0.5] - q[0.25])) / (q[0.75] - q[0.25])


def mostrar_analisis_var_cuantitativas(data):
    #calcular coeficiente de variacion
 datos_variable=pd.DataFrame([{"coeficiente de Variacion":(data.std()/data.mean())*100,\
                 "rango de la variable":data.max() - data.min(),
                 "rango intercuartilico":data.quantile(0.75) - data.quantile(0.25),
                 "coeficiente de asimetria":quartile_skew(data),
                 "Min":data.min(),
                 "Max":data.max(),
                 "Mean":data.mean()}])
 return(datos_variable)

def mostrar_graf_variables_continuas(df_data,column):
    sb.set_theme(style="whitegrid")
    fig, (ax1,ax2) = plot.subplots(1,2,figsize=(12,6))
    sb.histplot(data=df_data,x=column,ax=ax1)
    sb.boxplot(data=df_data,x=column,ax=ax2)
    sb.displot(data=df_data, x=column,kind="kde",rug=True)
    return plot.show()
def mostrar_graf_variables_discretas(df_data,column):
    sb.set_theme(style="whitegrid")
    fig, (ax1,ax2) = plot.subplots(1,2,figsize=(12,6))
    sb.boxplot(data=df_data,x=column,ax=ax1)
    sb.countplot(data=df_data,x=column,ax=ax2)
    plot.show()
    
    
 #visualizacion del mapa de precios
import geopy.distance 
import geopandas
import numpy as np
from shapely.geometry import Point
from geopy.distance import distance
import geopy
from geopy.geocoders import Nominatim
# creamos el objeto geolocator

'''
geolocator = Nominatim(user_agent="Modelo-Melbourne")

#normalizamos las direcciones
dataframe["Address_normalized"]=dataframe["Address"].replace("/","") + str(", City of Melbourne") + ", " + dataframe["Postcode"].astype(str).str.slice(stop=4) + str(", Australia")
print(dataframe["Address_normalized"])
Loc=geolocator.geocode("217 Langridge St, City of Melbourne, 3067, Australia")

print(Loc)

def update_latitude(row):
  if ((row["Lattitude"]>-36.0) | (row["Lattitude"]<-38.0)) | ((row["Longtitude"]>146.0) | (row["Longtitude"]<144.0)):
    time.sleep(1)
    Loc=geolocator.geocode(row["Address_normalized"],timeout=None)
    if Loc != None:
        print(row["Address_normalized"])
        row["Lattitude"]=Loc.latitude
        row["Longtitude"]=Loc.longitude
        print(Loc.latitude, Loc.longitude)
        
dataframe.apply(update_latitude, axis=1)


for i,v in  dataframe.iterrows():
    if ((v["Lattitude"]>-36.0) | (v["Lattitude"]<-38.0)) | ((v["Longtitude"]>146.0) | (v["Longtitude"]<144.0)):
        print (v["Address_normalized"])
        Loc=geolocator.geocode(v["Address_normalized"],timeout=None)
        print(dataframe.loc[i]["Lattitude"], dataframe.loc[i]["Longtitude"])
        if Loc != None:
            dataframe.loc[i]["Lattitude"]=Loc.latitude
            dataframe.loc[i]["Longtitude"]=Loc.longitude
            print (v["Address_normalized"])
            print(Loc)
            time.sleep(1)
'''
mapa = folium.Map(location=[-37.810634, 145.001851], zoom_start=11)
data_filtered=dataframe[dataframe['Lattitude'].notnull() & \
                        dataframe['Longtitude'].notnull() &\
                       dataframe['Rooms'].notnull() & \
                       dataframe['Bathroom'].notnull() & \
                       dataframe['Price'].notnull()]
coords_2=(-37.810634,145.001851)



locations=data_filtered[['Lattitude', 'Longtitude']]
locations.round(6)
locationlist=locations.values.tolist()
len(locationlist)
for i,v in  data_filtered.iterrows():
    popup = popup = """
    Bathrooms: <b>%s</b><br>
    Rooms: <b>%s</b><br>
    Suburb : <b>%s</b><br>
    Price : <b>%s</b><br>
    """ % (v['Bathroom'], v['Rooms'], v['Suburb'], v['Price'])
    #print(v['Price'])
    if v['Price'] < float('500000.0'):
        #Color Azul
        folium.CircleMarker(location=[v['Lattitude'], v['Longtitude']], tooltip=popup,
                            color='#0000FF',
                            fill_color='#0000FF',
                            fill=True, radius=1 ).add_to(mapa)
    elif  (v['Price'] >= float('500000.0')) & (v['Price'] < float('800000.0')):
        #color Verde
        folium.CircleMarker(location=[v['Lattitude'], v['Longtitude']], tooltip=popup,
                            color='#00FF40',
                            fill_color='#00FF40',
                            fill=True, radius=1 ).add_to(mapa)
    elif  (v['Price'] >= float('800000.0')) & (v['Price'] < float('1200000.0')):
        #Color Amarillo
        folium.CircleMarker(location=[v['Lattitude'], v['Longtitude']], tooltip=popup,
                            color='#FFFF00',
                            fill_color='#FFFF00',
                            fill=True, radius=1 ).add_to(mapa)
    elif  (v['Price'] >= float('1200000.0'))& (v['Price'] < float('1500000.0')):
        #Color cian
        folium.CircleMarker(location=[v['Lattitude'], v['Longtitude']], tooltip=popup,
                            color='#00fff7',
                            fill_color='#00fff7',
                            fill=True, radius=1 ).add_to(mapa)
    elif  (v['Price'] >= float('1500000.0'))& (v['Price'] < float('1800000.0')):
        #Color Rojo
        folium.CircleMarker(location=[v['Lattitude'], v['Longtitude']], tooltip=popup,
                            color='#ff2300',
                            fill_color='#ff2300',
                            fill=True, radius=1 ).add_to(mapa)
    elif  (v['Price'] >= float('1800000.0'))& (v['Price'] < float('2100000.0')):
        #Color Naranja
        folium.CircleMarker(location=[v['Lattitude'], v['Longtitude']], tooltip=popup,
                            color='#ffc900',
                            fill_color='#ffc900',
                            fill=True, radius=1 ).add_to(mapa)
    elif  (v['Price'] >= float('2100000.0')):
        #Color Morado
        folium.CircleMarker(location=[v['Lattitude'], v['Longtitude']], tooltip=popup,
                            color='#f700ff',
                            fill_color='#f700ff',
                            fill=True, radius=1 ).add_to(mapa)
folium.Marker(location=('37.84280999999999','145.03483'), tooltip=popup,
                            color='#f711ff',
                            fill_color='#ffffff',
                            fill=True, radius=20 ).add_to(mapa)





# Calculo de la nueva distancia al centro. 




data_filtered=dataframe[dataframe['Lattitude'].notnull() & \
                        dataframe['Longtitude'].notnull() &\
                       dataframe['Rooms'].notnull() & \
                       dataframe['Bathroom'].notnull() & \
                       dataframe['Price'].notnull()]




#Calculamos el precio medio por barrio
data_aux=data_filtered.groupby("Suburb")[["Price"]].mean().reset_index() 
#nos quedamos con el precio medio mas alta
data_aux[data_aux["Price"]==max(data_aux['Price'])]
#cogemos como centro para el calculo de distancia la latitud y longitud media del barrio mas caro
data_frameKoyong=data_filtered[(data_filtered["Suburb"]=='Kooyong')  &  (dataframe["Lattitude"]<-35.0) &
                             (dataframe["Lattitude"]>-39.0) &
                             (dataframe["Longtitude"]<147.0) &
                              (dataframe["Longtitude"]>143.0)]

Koyong_lat_mean=data_frameKoyong["Lattitude"].mean()
Koyong_lon_mean=data_frameKoyong["Longtitude"].mean()
#Calculamos la nueva distancia 
arr=[]
for lat, lon in zip(data_filtered['Lattitude'], data_filtered['Longtitude']):
    arr.append(distance((lat,lon), (Koyong_lat_mean,Koyong_lon_mean)).km)
    #print(Koyong_lat_mean,Koyong_lon_mean)
    #print(distance((lat,lon), (Koyong_lat_mean,Koyong_lon_mean)).km)

data_filtered["Distancia_NEW"]=arr
#Comprobamos si la correlaciñon mejora y verficamos que casi duplica 
data_filtered=data_filtered[data_filtered["Distancia_NEW"]<50]
#sb.scatterplot(data=data_filtered, x="Distancia_NEW", y="Price")
#mostrar_graf_variables_continuas(dataframe_filtered,"Distance_SQR")
#plot.show()

#sb.regplot(data=data_filtered, x="Distancia_NEW", y="Price")
#plot.show()
#data_filtered["BathsAndRooms"]=(data_filtered["Rooms"]+data_filtered["Bathroom"])/data_filtered["Distancia_NEW"].apply(np.log10)





