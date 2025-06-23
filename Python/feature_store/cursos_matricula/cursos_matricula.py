import datetime
import os
import time

import numpy as np

from ..general.dataframe_funcs import formatting
from ..general.funcs import (
    check_columns,
    consolidate_parquet_files_from_folder,
    create_parquet_from_df,
    get_all_files,
    load_csv,
    load_parquet,
    refactor_folder_file,
)
from ..window_aggregations import WindowAggregation
from .bronze_to_silver import processing_bronze_file_raw_level
from .silver_to_gold import processing_file_feature_level


# fmt: off
class CursoMatricula:

    def __init__(self) -> None:

        self.DATASET_NAME = "01_cursos_matricula"
        
        # rutas de interes
        self.BRONZE_FILES_PATH = "00_data/00_bronze/01_cursos_matricula/"
        self.SILVER_FILES_PATH = "00_data/01_silver/01_cursos_matricula/"
        self.GOLD_FILES_PATH = "00_data/02_gold/01_cursos_matricula/"

        self.REPORT_OUTPUT_PATH = "00_data/05_reportes/00_eda"

        self.SILVER_FILES_PATH_RAW = "00_data/01_silver/01_cursos_matricula/00_RAW"
        self.SILVER_FILES_PATH_EOP = "00_data/01_silver/01_cursos_matricula/01_EOP"

        self.GOLD_FILES_PATH_EOP = "00_data/02_gold/01_cursos_matricula/01_EOP"
        self.GOLD_FILES_PATH_WINDOWN_AGG = "00_data/02_gold/01_cursos_matricula/03_WINDOWAGG"

        self.CALENDAR_EOP = "00_data/03_assets/calendar_eop.parquet"
        self.ALIAS_EOP = "cursos_matricula_eop"
        self.REFACTOR_PERIOD = None

        # generar parametros propios de la clase
        self._generate_parameters()

        pass

    def bronze_to_silver_raw(self) -> None:
        process_start_time = datetime.datetime.now()
        files = get_all_files(self.BRONZE_FILES_PATH, "*.csv")
        if files:
            for file in files:
                filename_no_extension = os.path.basename(file).split(".")[0]

                if self.REFACTOR_PERIOD is not None:
                    if self.REFACTOR_PERIOD != int(filename_no_extension):
                        continue
                    else:
                        refactor_folder_file(os.path.join(self.SILVER_FILES_PATH_RAW, f"{filename_no_extension}.parquet"), flag_folder=0)
                try:
                    start_time = time.time()
                    df = load_csv(file)
                    df.rename(self.BRONZE_FILE_FEATURES, axis=1, inplace=True)
                    check_columns_val = check_columns(df, set(self.BRONZE_FILE_FEATURES.values()))
                    if check_columns_val:
                        df = processing_bronze_file_raw_level(df, self.SILVER_FILE_FEATURES_RAW)
                        create_parquet_from_df(df, self.SILVER_FILES_PATH_RAW, f"{filename_no_extension}.parquet")
                        end_time = time.time()
                        print(f"\t{filename_no_extension}: {(end_time-start_time):.3f} segs")
                    else:
                        print(f"\t{filename_no_extension}:\tRevisar variables de la capa bronze")
                except Exception as e:
                    print(f"\t{filename_no_extension}:\tError de procesamiento\t{e}")
        process_end_time = datetime.datetime.now()
        print(f'Duracion: {process_end_time - process_start_time}')
        return None

    def silver_to_silver_eop(self) -> None:
        process_start_time = datetime.datetime.now()
        files = get_all_files(self.SILVER_FILES_PATH_RAW, "*.parquet")
        if files:
            if self.REFACTOR_PERIOD is not None:
                refactor_folder_file(os.path.join(self.SILVER_FILES_PATH_EOP, f"{self.ALIAS_EOP}_silver.parquet"), flag_folder=0)
            try:
                start_time = time.time()
                consolidate_parquet_files_from_folder(self.SILVER_FILES_PATH_RAW, "*.parquet", self.SILVER_FILES_PATH_EOP, f"{self.ALIAS_EOP}_silver.parquet", parts=None)
                end_time = time.time()
                print(f"\tConsolidar raw files: {(end_time-start_time):.3f} segs")
            except Exception as e:
                print(f"\tConsolidar raw files:\tError de procesamiento\t{e}")
        process_end_time = datetime.datetime.now()
        print(f'Duracion: {process_end_time - process_start_time}')
        return None

    def silver_to_gold_eop(self) -> None:
        process_start_time = datetime.datetime.now()
        files = get_all_files(self.SILVER_FILES_PATH_EOP, "*.parquet")
        if files:
            for file in files:
                if self.REFACTOR_PERIOD is not None:
                    refactor_folder_file(os.path.join(self.GOLD_FILES_PATH_EOP, f"{self.ALIAS_EOP}_features.parquet"), flag_folder=0,)

                try:
                    start_time = time.time()
                    df = load_parquet(file)
                    calendar = load_parquet(self.CALENDAR_EOP)
                    df = df.merge(calendar, how="left", on=["periodo"])
                    df = processing_file_feature_level(df, self.GOLD_FILE_FEATURES_DICT)
                    df = formatting(df, self.GOLD_FILE_FEATURES_DICT)
                    create_parquet_from_df(df, self.GOLD_FILES_PATH_EOP, f"{self.ALIAS_EOP}_features.parquet",)
                    end_time = time.time()
                    print(f"\tGold features: {(end_time-start_time):.3f} segs")
                except Exception as e:
                    print(f"\tGold features:\tError de procesamiento\t{e}")
            now_str = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            consolidate_parquet_files_from_folder(self.GOLD_FILES_PATH_EOP, "*_features.parquet", self.GOLD_FILES_PATH, f"{self.ALIAS_EOP}_{now_str}.parquet")
        process_end_time = datetime.datetime.now()
        print(f'Duracion: {process_end_time - process_start_time}')
        return None

    def gold_eop_to_window_aggregation(self) -> None:
        """
        Realiza el proceso de agregación de ventana en los datos de EOP (end of period) y guarda el resultado en un archivo parquet.

        Este método lleva a cabo las siguientes tareas:
        1. Inicializa la clase WindowAggregation con la ruta del archivo parquet y el diccionario de funciones de ventana.
        2. Crea nuevas columnas en el DataFrame basado en umbrales específicos.
        3. Genera todas las agregaciones de ventana especificadas y las añade al DataFrame original.
        4. Guarda el DataFrame resultante en un archivo parquet con un nombre que incluye la fecha y hora actual.

        Args:
            self: La instancia actual de la clase que contiene las rutas y configuraciones necesarias.
        """

        process_start_time = datetime.datetime.now()

        # Inicializa una instancia de WindowAggregation con la ruta del archivo EOP y el diccionario de funciones de ventana.
        windowAgg = WindowAggregation(eop_path=self.GOLD_FILES_PATH_EOP,list_window_agg=self.GOLD_FILE_AGG_FEATURES_LIST)

        # Crea nuevas columnas en el DataFrame basado en el número de cursos matriculados (n_curmatr) y umbrales específicos.
        windowAgg.create_column_threshold(col="n_curmatr", threshold=1, comparison_op_str=">=")
        windowAgg.create_column_threshold(col="n_curmatr", threshold=3, comparison_op_str=">=")
        windowAgg.create_column_threshold(col="n_curmatr", threshold=6, comparison_op_str=">=")

        # Crea nuevas columnas en el DataFrame basado en el número de cursos obligatorios matriculados (n_curmatr_oblig) y umbrales específicos.
        windowAgg.create_column_threshold(col="n_curmatr_oblig", threshold=1, comparison_op_str=">=")
        windowAgg.create_column_threshold(col="n_curmatr_oblig", threshold=3, comparison_op_str=">=")
        windowAgg.create_column_threshold(col="n_curmatr_oblig", threshold=6, comparison_op_str=">=")

        # Genera todas las agregaciones de ventana especificadas y añade las nuevas columnas al DataFrame original.
        df_windownAgg = windowAgg.get_aggregations()

        # Obtiene la fecha y hora actual como una cadena en el formato 'YYYYMMDD_HHMMSS'.
        now_str = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")

        # Guarda el DataFrame resultante en un archivo parquet en la ruta especificada, con un nombre que incluye la fecha y hora actual.
        create_parquet_from_df(df_windownAgg, self.GOLD_FILES_PATH_WINDOWN_AGG, f"cursos_matricula_eop_wa_{now_str}.parquet")

        process_end_time = datetime.datetime.now()
        print(f'Duracion: {process_end_time - process_start_time}')
        return None

    def _generate_parameters(self) -> None:

        # columnas necesarias para procesar los archivos capa bronze
        self.BRONZE_FILE_FEATURES = {
            "COD_PERIODO_MATRICULA": "periodo",
            "#COD_PERIODO_MATRICULA": "periodo",
            "COD_ALUMNO": "cod_alumno",
            "COD_CARRERA": "cod_carrera",
            "COD_CURSO": "cod_curso",
            "DES_CURSO": "descr_curso",
            "CANT_CREDITOS_CURSO": "cant_creditos",
            "TIPO_CURSO": "tipo_curso",
            "DES_CICLO": "ciclo_curso",
        }

        # columnas que se devolveran del proceso BRONZE -> SILVER RAW
        self.SILVER_FILE_FEATURES_RAW = [
            "periodo",
            "cod_alumno",
            "cod_carrera",
            "cod_curso",
            "descr_curso",
            "cant_creditos",
            "tipo_curso",
            "ciclo_curso",
        ]

        self.GOLD_FILE_FEATURES_DICT = {
            "periodo": "skip",
            "cod_alumno": "skip",
            "fecha_corte": "skip",
            "n_curmatr": 0.0,
            "n_curmatr_oblig": 0.0,
            "r_curmatr_oblg": 0.0,
            "n_curmatr_elect": 0.0,
            "r_curmatr_elect": 0.0,
            "n_curmatr_nocred": 0.0,
            "avg_ciclo": np.nan,
            "max_ciclo": np.nan,
            "min_ciclo": np.nan,
            "sum_cred_curmatr": 0.0,
            "sum_cred_oblig_curmatr": 0.0,
            "r_cred_curmatr_oblg": 0.0,
            "sum_cred_elect_curmatr": 0.0,
            "r_cred_curmatr_elect": 0.0,
            "cred_acum_hist": 0.0,
        }

        self.GOLD_FILE_AGG_FEATURES_LIST = [
            ("n_curmatr_mayor_igual_1", [2, 3, 4], "count"),
            ("n_curmatr_mayor_igual_3", [2, 3, 4], "count"),
            ("n_curmatr_mayor_igual_6", [2, 3, 4], "count"),
            ("n_curmatr_oblig_mayor_igual_1", [2, 3, 4], "count"),
            ("n_curmatr_oblig_mayor_igual_3", [2, 3, 4], "count"),
            ("n_curmatr_oblig_mayor_igual_6", [2, 3, 4], "count"),
            ("sum_cred_curmatr", [2, 3, 4], "sum"),
            ("sum_cred_oblig_curmatr", [2, 3, 4], "sum"),
            ("sum_cred_elect_curmatr", [2, 3, 4], "sum"),
            ("n_curmatr", [2, 3, 4], "sum"),
            ("n_curmatr_oblig", [2, 3, 4], "sum"),
            ("n_curmatr_elect", [2, 3, 4], "sum"),
            ("r_cred_curmatr_oblg", [2, 3, 4], "mean"),
            ("r_cred_curmatr_elect", [2, 3, 4], "mean"),
        ]

        return None

    def check_refactor(self) -> None:
        if self.REFACTOR_PERIOD is None:
            refactor_folder_file(self.SILVER_FILES_PATH)
            refactor_folder_file(self.GOLD_FILES_PATH)
