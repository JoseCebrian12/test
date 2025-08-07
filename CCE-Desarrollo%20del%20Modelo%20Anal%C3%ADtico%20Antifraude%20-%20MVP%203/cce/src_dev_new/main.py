import logging
import time
from datetime import datetime
import os
from Features.feature_engineering_experimento4 import feature_engineering_exp4
from Features.feature_engineering_ratios import feature_engineering_ratio

# from Models.Train.train import trainmodel
from Models.Predict.predict import predict_model

# Obtener la fecha y hora actual
fecha_hora_actual = datetime.now()

# Formatear la fecha y hora en un formato legible
formato = "%Y%m%d_%H%M%S_"
fecha_hora_formateada = fecha_hora_actual.strftime(formato)

# Construir la ruta del archivo de registro
nombre_archivo = fecha_hora_formateada + "registro.log"
ruta_archivo = os.path.join("src_dev_new", "Logs", nombre_archivo)

# Configuración del sistema de logging
logging.basicConfig(
    level=logging.DEBUG, format="%(asctime)s - %(levelname)s - %(message)s"
)


def main():
    # try:

    logging.info("Iniciando el proceso............")

    # Paso 1
    start_time = time.time()
    logging.info("Comenzando el pipeline feature engineering 1.................")
    max_run_id = feature_engineering_exp4()
    end_time = time.time()
    elapsed_time = end_time - start_time
    logging.info("Paso 1 completado en %.2f segundos", elapsed_time)

    start_time = time.time()
    logging.info("Comenzando el pipeline feature engineering ratios..............")

    final_90_same, final_90_diff, final_90_black, final_90_clean = (
        feature_engineering_ratio(max_run_id)
    )

    end_time = time.time()
    elapsed_time = end_time - start_time
    logging.info("Paso 2 completado en %.2f segundos", elapsed_time)

    # TRAIN
    # start_time = time.time()
    # logging.info("Comenzando el pipeline de entrenamiento......................")
    # trainmodel()
    # end_time = time.time()
    # elapsed_time = end_time - start_time
    # logging.info("Entrenaminento completado en %.2f segundos", elapsed_time)

    # #PREDICT
    start_time = time.time()
    logging.info("Comenzando el pipeline de predicción......................")
    predict_model(max_run_id, final_90_same, final_90_black)
    end_time = time.time()
    elapsed_time = end_time - start_time
    logging.info("Paso 3 completado en %.2f segundos", elapsed_time)

    print("Done")
    # except Exception as e:
    #     logging.error("Ocurrió un error: %s", str(e))


# %%TODO main

if __name__ == "__main__":
    main()
