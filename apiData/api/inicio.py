import nltk
nltk.download('punkt')
import pandas as pd
import time
import os
from apiData.api.procesamiento import cargar_data, quitar_columnas, elegir_columna, limpiar_tokenizar, eliminar_vacios
from apiData.api.LDA import lemmatize_text, creacion_LDA, mostrar_temas, predict_topic



def logs(mensaje):
    try:
        with open("logs.txt", "r") as file:
            pass
    except:
        with open("logs.txt", "w") as file:
            pass

    with open("logs.txt", "a") as file:
        file.write(time.strftime("%Y-%m-%d %H:%M:%S") + " - " + mensaje + "\n")

def inicio(data, columna, cluster=False):
    if data is None or data == "" or columna is None or columna == "":
        return "No se ha ingresado ningun dato"
    
    datos = cargar_data(data)
    if isinstance(datos, str):
        print(datos)
        return datos
    
    print(datos.head())  # Ejemplo de procesamiento del DataFrame
    
    if not isinstance(cluster, int) or not isinstance(columna, int):
        try:
            cluster = int(cluster)
            columna = int(columna)
        except ValueError:
            return "El dato en cluster no es un numero"
        
    if cluster < 3:
        return "El cluster debe ser mayor a 2"
    
    if cluster > len(datos):
        return "El cluster debe ser menor a la cantidad de datos"
    
    datos.dropna(inplace=True)

    if len(datos.columns) < columna:
        return "La columna no existe"
    
    columna_trabajar = elegir_columna(datos, columna)

    datos = quitar_columnas(datos)
    print(datos.head())  # Ejemplo de procesamiento del DataFrame

    if isinstance(columna_trabajar, pd.Series):
        columna_trabajar = columna_trabajar.to_frame()

    print(columna_trabajar.head())
    columna_trabajar['TEXTO_TOKEN'] = columna_trabajar.iloc[:, 0].fillna('').astype(str).apply(limpiar_tokenizar)
    print(columna_trabajar.head())
    
    columna_trabajar = eliminar_vacios(columna_trabajar)

    columna_trabajar['LEMMA'] = columna_trabajar['TEXTO_TOKEN'].apply(lemmatize_text)
    print(columna_trabajar.head())

    diccionario, lda_model = creacion_LDA(columna_trabajar, cluster)

    temas = mostrar_temas(lda_model)
    print(temas)

    datos = pd.concat([datos, columna_trabajar], axis=1)
    datos = datos.dropna(subset=['TEXTO_TOKEN'])

    datos_final = predict_topic(datos, diccionario, lda_model, temas)
    datos_final.to_csv("datos_procesados.csv", index=False, encoding="utf-8")
    print("Datos procesados guardados en 'datos_procesados.csv'")
    return datos_final

if __name__ == "__main__":
    T_inicio = time.time()
    print(time.strftime("%Y-%m-%d %H:%M:%S") + " - Inicio del programa")
    cosa = inicio("Opiniones.csv", 4, 8)
    print(cosa)
    print(time.strftime("%Y-%m-%d %H:%M:%S") + " - Fin del programa")
    print("Tiempo de ejecución:", time.time() - T_inicio)