import pandas as pd
import numpy as np
import joblib
import os

os.chdir(r'C:\Users\amqj1\OneDrive\Escritorio\Codigos JD\Analitica III\Salud\mod_Salud')

#cargar predicciones 
predictions = joblib.load('salidas\\predictions.joblib')

#crear el DataFrame 
data_dic = {}

#numpy.ndarray to list
lista_aux = []
for i in range(len(predictions)):
    lista_aux.append(predictions[i][0])

#añadir la columna de predicciones
data_dic['Predicciones'] = lista_aux
data = pd.DataFrame(data_dic)

#clasificar según el valor de la predicccion
def clasificar_prioridad(valor):
    if 0 <= valor <= 0.25:
        return 'prioridad baja'
    elif 0.25 < valor <= 0.75:
        return 'prioridad media'
    else:
        return 'prioridad alta'

#añadir la columna de clasificación
data['Clasificación'] = data['Predicciones'].apply(clasificar_prioridad)

data['ID'] = [i for i in range(1, len(data)+1)]

#dataFrame resultante
data

#df a xlsx
data.to_excel('salidas//resultados_predicciones.xlsx', index=False)