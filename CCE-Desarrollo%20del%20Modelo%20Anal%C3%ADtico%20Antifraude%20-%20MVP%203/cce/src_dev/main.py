
import logging
import time
from datetime import datetime
import os
from Features.feature_engineering_experimento4 import feature_engineering_exp4
from Features.feature_engineering_ratios import feature_engineering_ratio
from Models.Train.train import trainmodel
from Models.Predict.predict import predict_model
import psutil


process = psutil.Process()


# Obtener la fecha y hora actual
fecha_hora_actual = datetime.now()

# Formatear la fecha y hora en un formato legible
formato = "%Y%m%d_%H%M%S_"
fecha_hora_formateada = fecha_hora_actual.strftime(formato)

# Construir la ruta del archivo de registro
nombre_archivo = fecha_hora_formateada + 'registro.log'
ruta_archivo = os.path.join("src_dev", "Logs", nombre_archivo)

# Configuración del sistema de logging
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')

def main():
    # try:
       
    logging.info("Iniciando el proceso............")

    # Paso 1
    start_time = time.time()
    logging.info("Comenzando el pipeline feature engineering 1.................")
    feature_engineering_exp4()
    print(f"Memory usage: {process.memory_info().rss / (1024 * 1024 * 1024):.2f} GB")

    end_time = time.time()
    elapsed_time = end_time - start_time
    logging.info("Paso 1 completado en %.2f segundos", elapsed_time)
    
    start_time = time.time()
    logging.info("Comenzando el pipeline feature engineering ratios..............")
    
    final_90_same, final_90_diff = feature_engineering_ratio()
    
    end_time = time.time()
    elapsed_time = end_time - start_time
    logging.info("Paso 2 completado en %.2f segundos", elapsed_time)
    

    #TRAIN
    start_time = time.time()
    logging.info("Comenzando el pipeline de entrenamiento......................")
    trainmodel()
    end_time = time.time()
    elapsed_time = end_time - start_time
    logging.info("Entrenaminento completado en %.2f segundos", elapsed_time)


    # #PREDICT
    start_time = time.time()
    logging.info("Comenzando el pipeline de predicción......................")
    predict_model(final_90_same)
    end_time = time.time()
    elapsed_time = end_time - start_time
    logging.info("Paso 3 completado en %.2f segundos", elapsed_time)

    
    print("Done")
    # except Exception as e:
    #     logging.error("Ocurrió un error: %s", str(e))
        
# %%TODO main

if __name__ == "__main__":
    main()    