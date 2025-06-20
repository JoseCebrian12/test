import datetime
import os
import time
import traceback

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
class MatriculaClass:

    def __init__(self) -> None:

        # rutas de interes
        self.DATASET_NAME = "00_matricula"
        
        self.BRONZE_FILES_PATH = "00_data/00_bronze/00_matricula/"
        self.SILVER_FILES_PATH = "00_data/01_silver/00_matricula/"
        self.GOLD_FILES_PATH = "00_data/02_gold/00_matricula/"

        self.REPORT_OUTPUT_PATH = "00_data/05_reportes/00_eda"
        
        self.SILVER_FILES_PATH_RAW = "00_data/01_silver/00_matricula/00_RAW"
        self.SILVER_FILES_PATH_EOP = "00_data/01_silver/00_matricula/01_EOP"

        self.GOLD_FILES_PATH_EOP = "00_data/02_gold/00_matricula/01_EOP"
        self.GOLD_FILES_PATH_WINDOWN_AGG = '00_data/02_gold/00_matricula/03_WINDOWAGG'
        
        self.CALENDAR_EOP = "00_data/03_assets/calendar_eop.parquet"
        self.ALIAS_EOP = "matricula_eop"
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
                        df = processing_bronze_file_raw_level(df) # self.SILVER_FILE_FEATURES_RAW
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
                    refactor_folder_file(os.path.join(self.GOLD_FILES_PATH_EOP, f"{self.ALIAS_EOP}_features.parquet"), flag_folder=0)

                try:
                    start_time = time.time()
                    df = load_parquet(file)
                    calendar = load_parquet(self.CALENDAR_EOP)
                    df = df.merge(calendar, how="left", on=["periodo"])
                    df = processing_file_feature_level(df, self.GOLD_FILE_FEATURES_DICT) 
                    df = formatting(df, self.GOLD_FILE_FEATURES_DICT)
                    create_parquet_from_df(df, self.GOLD_FILES_PATH_EOP, f"{self.ALIAS_EOP}_features.parquet")                    
                    end_time = time.time()
                    print(f"\tGold features: {(end_time-start_time):.3f} segs")
                except Exception as e:
                    print(f"\tGold features:\tError de procesamiento\t{e}")
                    print(traceback.format_exc())
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
        windowAgg = WindowAggregation(eop_path=self.GOLD_FILES_PATH_EOP, list_window_agg=self.GOLD_FILE_AGG_FEATURES_LIST)

        # Crea nuevas columnas en el DataFrame basado en el número de periodos historicos matriculados (n_per_mtr_hist) y umbrales específicos.
        windowAgg.create_column_threshold(col="n_per_mtr_hist", threshold=1, comparison_op_str=">=")
        windowAgg.create_column_threshold(col="n_per_mtr_hist", threshold=7, comparison_op_str=">=")
        windowAgg.create_column_threshold(col="n_per_mtr_hist", threshold=12, comparison_op_str=">=")

        # Crea nuevas columnas en el DataFrame basado en el número de periodos historicos matriculados de periodos regulares (n_per_mtr_hist_reg) y umbrales específicos.
        windowAgg.create_column_threshold(col="n_per_mtr_hist_reg", threshold=1, comparison_op_str=">=")
        windowAgg.create_column_threshold(col="n_per_mtr_hist_reg", threshold=7, comparison_op_str=">=")
        windowAgg.create_column_threshold(col="n_per_mtr_hist_reg", threshold=12, comparison_op_str=">=")

        # Genera todas las agregaciones de ventana especificadas y añade las nuevas columnas al DataFrame original.
        df_windownAgg = windowAgg.get_aggregations()

        # Obtiene la fecha y hora actual como una cadena en el formato 'YYYYMMDD_HHMMSS'.
        now_str = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
        
        # Guarda el DataFrame resultante en un archivo parquet en la ruta especificada, con un nombre que incluye la fecha y hora actual.
        create_parquet_from_df(df_windownAgg, self.GOLD_FILES_PATH_WINDOWN_AGG, f"matricula_eop_wa_{now_str}.parquet")

        process_end_time = datetime.datetime.now()
        print(f'Duracion: {process_end_time - process_start_time}')
        return None

    def _generate_parameters(self) -> None:

        # columnas necesarias para procesar los archivos capa bronze
        self.BRONZE_FILE_FEATURES = {
            "COD_PERIODO_MATRICULA": "periodo",
            "#COD_PERIODO_MATRICULA": "periodo",
            "COD_ALUMNO": "cod_alumno",
            "DES_COLEGIO": "colegio",
            "DES_TIPO_GESTION_COLEGIO": "tipo_colegio",
            "MODALIDAD_ESTUDIO": "modalidad_estudio",
            "MODALIDAD_FORMATIVA": "modalidad_formativa",
            "ESTADO_INCIAL_DETALLE": "estado",
            "DES_TIPO_INGRESO": "tipo_ingreso",
            "COD_CATEGORIA_PAGO": "categoria_pago",
            "FEC_NACIMIENTO": "fecha_nacimiento",
            "CREDITOS_MATRICULADOS": "cant_creditos",
            "COD_CARRERA": "cod_carrera",
            "CARRERA": "carrera",
            "DES_ESTADO_MATRICULA": "estado_matricula",
            "PCT_BECA": "porcentaje_beca",
            "DES_BECA": "desc_beca",
            "FEC_INGRESO_ADMISION": "fecha_ingreso",
            "IND_CICLO_ALUMNO": "ciclo_aprox",
            "CANTIDAD_CURSOS_MATRICULADOS": "cant_cursos",
            "CANTIDAD_CURSOS_OBLIGATORIOS": "cant_cursos_obligatorios",
            "CANTIDAD_CRED_MATRICULADOS_OBLIGATORIOS": "cant_creditos_obligatorios",
            "COD_FACULTAD": "cod_facultad",
            "NRO_DOCUMENTO_IDENTIDAD": "nro_documento_identidad",
            "DES_DIRECCION": "des_direccion",
            "FACULTAD": "facultad",
            "FECHA_MATRICULA": "fecha_matricula",
            "EDAD": "edad",
            "DES_DEPARTAMENTO": "departamento",
            "DES_PROVINCIA": "provincia",
            "DES_DISTRITO": "distrito",
            "COD_UBIGEO": "ubigeo",
            "DES_CAMPUS": "campus",
            "TIP_MERITO_PRODUCTO": "tipo_merito_per_ant",
            "TERCIO_SUPERIOR_CICLO_ANTERIOR": "flag_tercio_per_ant",
            "QUINTO_SUPERIOR_CICLO_ANTERIOR": "flag_quinto_per_ant",
            "DECIMO_SUPERIOR_CICLO_ANTERIOR": "flag_decimo_per_ant",
            "NRO_PONDERADO_ACTUAL": "ponderado_actual",
            "NRO_PONDERADO_ACUMULADO": "ponderado_acumulado",
            "FLG_RIESGO": "flag_riesgo",
            "CTD_CICLOS_VERANO": "cant_ciclos_verano",
        }

        # columnas que se devolveran del proceso BRONZE -> SILVER
        self.SILVER_FILE_FEATURES_RAW = [
        ]

        self.GOLD_FILE_FEATURES_DICT = {
            "periodo": "skip", 
            "cod_alumno": "skip",
            "fecha_corte": "skip",
            
            "categoria_pago": "NO_DETERMINADO",
            "cod_carrera": "NO_DETERMINADO",
            "carrera": "NO_DETERMINADO",
            "carrera_desc": "NO_DETERMINADO",
            "cod_facultad": "NO_DETERMINADO",
            "facultad": "NO_DETERMINADO",
            "facultad_desc": "NO_DETERMINADO",
            "campus": "NO_DETERMINADO",
            "campus_desc": "NO_DETERMINADO",
            "edad": np.nan,
            "departamento": "NO_DETERMINADO",
            "provincia": "NO_DETERMINADO",
            "distrito": "NO_DETERMINADO",
            "flag_tercio_per_ant": 0.0,
            "flag_quinto_per_ant": 0.0,
            "flag_decimo_per_ant": 0.0,
            "fecha_ingreso": np.datetime64("NaT"),
            "unidad_negocio": "NO_DETERMINADO",
            "modalidad": "NO_DETERMINADO",
            "grupo_tipo_ingreso": "NO_DETERMINADO",
            "es_verano": 0.0,
            "flag_mtr_per_ant": 0.0,
            "max_mtr_continuas_reg": 0.0,
            "pct_beca": 0.0,
            "flag_beca": 0.0,
            "desc_beca": np.nan,
            "n_per_mtr_hist": 0.0,
            "n_per_mtr_hist_reg": 0.0,
            "n_per_mtr_hist_verano": 0.0,
            "flag_mtr_verano": 0.0,
            "estado_matricula_new": "NO_DETERMINADO",
        }

        self.GOLD_FILE_AGG_FEATURES_LIST = [
            ("n_per_mtr_hist_mayor_igual_1", [2, 3, 4], "count"),
            ("n_per_mtr_hist_mayor_igual_7", [2, 3, 4], "count"),
            ("n_per_mtr_hist_mayor_igual_12", [2, 3, 4], "count"),
            ("n_per_mtr_hist_reg_mayor_igual_1", [2, 3, 4], "count"),
            ("n_per_mtr_hist_reg_mayor_igual_7", [2, 3, 4], "count"),
            ("n_per_mtr_hist_reg_mayor_igual_12", [2, 3, 4], "count"),
            ("n_per_mtr_hist", [2, 3, 4], "sum"),
            ("n_per_mtr_hist_reg", [2, 3, 4], "sum"),
            ("flag_mtr_per_ant", [2, 3, 4], "sum"),
            ("flag_beca", [2, 3, 4], "sum"),
            ("pct_beca", [2, 3, 4], "mean"),
            ("max_mtr_continuas_reg", [2, 3, 4], "mean"),
        ]

        self.DATA_DTYPE = {
            "periodo": "int",
            "cod_alumno": "str",
            "colegio": "str",
            "tipo_colegio": "str",
            "modalidad_estudio": "str",
            "modalidad_formativa": "str",
            "estado": "str",
            "tipo_ingreso": "str",
            "categoria_pago": "str",
            "fecha_nacimiento": "datetime",
            "cant_creditos": "float",
            "cod_carrera": "str",
            "carrera": "str",
            "estado_matricula": "str",
            "porcentaje_beca": "float",
            "desc_beca": "str",
            "fecha_ingreso": "datetime",
            "ciclo_aprox": "float",
            "cant_cursos": "str",
            "cant_cursos_obligatorios": "str",
            "cant_creditos_obligatorios": "str",
            "cod_facultad": "int",
            "facultad": "str",
            "fecha_matricula": "datetime",
            "edad": "float",
            "departamento": "str",
            "provincia": "str",
            "distrito": "str",
            "ubigeo": "str",
            "nro_documento_identidad": "str",
            "des_direccion": "str",
            "campus": "str",
            "tipo_merito_per_ant": "str",
            "flag_tercio_per_ant": "int",
            "flag_quinto_per_ant": "int",
            "flag_decimo_per_ant": "int",
            "ponderado_actual": "float",
            "ponderado_acumulado": "float",
            "flag_riesgo": "int",
            "cant_ciclos_verano": "float",
            "n_nulls": "int",
        }

        return None
    
    def check_refactor(self) -> None:
        if self.REFACTOR_PERIOD is None:
            refactor_folder_file(self.SILVER_FILES_PATH)
            refactor_folder_file(self.GOLD_FILES_PATH)
