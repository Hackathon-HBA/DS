from fastapi import FastAPI
from pydantic import BaseModel
import pickle
import pandas as pd

# Define el modelo de entrada para la solicitud de predicción de gastos
class PrediccionGastosSolicitud(BaseModel):
    cc_num: int

# Define el modelo de entrada para la solicitud de cálculo de ahorro posible
class AhorroPosibleSolicitud(BaseModel):
    nomina: float
    gastos: float

# Carga los modelos desde los archivos pickle
with open('modelo_arima_ahorro.pkl', 'rb') as f:
    loaded_model_ahorro = pickle.load(f)

with open('modelo_sarima.pkl', 'rb') as f:
    loaded_model_gastos = pickle.load(f)

# Crea una instancia de FastAPI
app = FastAPI()

# Define la ruta para hacer predicciones de gastos
@app.post("/prediccion_gastos/")
async def predecir_gastos(prediccion: PrediccionGastosSolicitud):
    # Aquí cargarías los datos históricos de gastos para el cc_num proporcionado
    
    # Supongamos que estamos haciendo una predicción simple de gastos basada en el último mes registrado
    gastos_predichos = loaded_model_gastos.predict([prediccion.cc_num])  # Suponiendo que cc_num es el único predictor
    
    # Devolver la predicción de gastos como respuesta
    return {"gastos_predichos": gastos_predichos[0]}  # Suponiendo que el modelo devuelve una lista con una sola predicción

# Define la ruta para calcular el ahorro posible
@app.post("/calculo_ahorro_posible/")
async def calcular_ahorro_posible(ahorro_solicitud: AhorroPosibleSolicitud):
    # Hacer predicción de ahorro usando el modelo cargado
    ahorro_predicho = loaded_model_ahorro.predict([[ahorro_solicitud.nomina, ahorro_solicitud.gastos]])
    
    # Devolver la predicción de ahorro como respuesta
    return {"ahorro_predicho": ahorro_predicho[0]}  # Suponiendo que el modelo devuelve una lista con una sola predicción
