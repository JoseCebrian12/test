import logging
from logging.handlers import RotatingFileHandler
import time
import os
import sys
from Features.feature_engineering_experimento4 import feature_engineering_exp4
from Features.feature_engineering_ratios import feature_engineering_ratio
from Models.Train.train import trainmodel
from Models.Predict.predict import predict_model


# Crear un nivel personalizado para los mensajes de `print()`
PRINT_LEVEL = 25  # Nivel intermedio entre INFO (20) y WARNING (30)
logging.addLevelName(PRINT_LEVEL, "PRINT")


# Redirigir stdout y stderr al logger
class LoggerWriter:
    def __init__(self, logger):
        self.logger = logger

    def write(self, message):
        if message.strip():  # Evitar mensajes vacíos
            self.logger.log(PRINT_LEVEL, message.strip())

    def flush(self):
        pass  # Requerido para compatibilidad con sys.stdout


# Configuración de logging con rotación
def configurar_logger():
    ruta_archivo = "/home/cnvdba/cce/src/Logs/registro.log"  # Archivo fijo para mantener la rotación

    handler = RotatingFileHandler(
        ruta_archivo,
        maxBytes=100 * 1024 * 1024,
        backupCount=1,  # Máximo 100 MB y 0 respaldos
    )
    formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
    handler.setFormatter(formatter)

    logger = logging.getLogger()
    logger.setLevel(logging.DEBUG)  # Nivel de logging
    logger.addHandler(handler)

    # Añadir también salida a la consola
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    # Redirigir stdout y stderr al logger
    sys.stdout = LoggerWriter(logger)  # Captura stdout como PRINT
    sys.stderr = LoggerWriter(
        logger
    )  # Captura stderr como PRINT (puedes cambiarlo a ERROR si prefieres)

    return logger


def main():
    # Configurar logger
    configurar_logger()

    # Separador inicial
    logging.info("=" * 50)
    logging.info("Iniciando el proceso principal...")
    logging.info("=" * 50 + "\n")

    try:
        # Paso 1
        start_time = time.time()
        logging.info("Comenzando el pipeline feature engineering 1.................")
        max_run_id = feature_engineering_exp4()
        print(f"max_run_id generado: {max_run_id}")  # Capturado como PRINT

        end_time = time.time()
        elapsed_time = end_time - start_time
        logging.info("Paso 1 completado en %.2f segundos", elapsed_time)

        # Paso 2
        start_time = time.time()
        logging.info("Comenzando el pipeline feature engineering ratios..............")
        final_90_same, final_90_diff = feature_engineering_ratio(max_run_id)
        print(
            "Ratios calculados:", final_90_same, final_90_diff
        )  # Capturado como PRINT
        end_time = time.time()
        elapsed_time = end_time - start_time
        logging.info("Paso 2 completado en %.2f segundos", elapsed_time)

        # TRAIN (descomentado si es necesario)
        # start_time = time.time()
        # logging.info("Comenzando el pipeline de entrenamiento......................")
        # trainmodel()
        # end_time = time.time()
        # elapsed_time = end_time - start_time
        # logging.info("Entrenaminento completado en %.2f segundos", elapsed_time)

        # PREDICT
        start_time = time.time()
        logging.info("Comenzando el pipeline de predicción......................")
        predict_model(max_run_id, final_90_same)
        print("Predicción completada.")  # Capturado como PRINT
        end_time = time.time()
        elapsed_time = end_time - start_time
        logging.info("Paso 3 completado en %.2f segundos", elapsed_time)

        logging.info("Proceso completado exitosamente.\n")

    except Exception as e:
        logging.error("Ocurrió un error: %s", str(e))

    # Mensaje final
    print("Done")  # Capturado como PRINT


# %%TODO main
if __name__ == "__main__":
    main()
