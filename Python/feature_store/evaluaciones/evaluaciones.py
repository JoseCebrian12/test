import datetime
import os
import time

import pandas as pd

from ..general.dataframe_funcs import formatting, merge_calendar
from ..general.funcs import (
    check_columns,
    consolidate_parquet_files_from_folder,
    create_parquet_from_df,
    get_all_files,
    get_all_folders,
    load_csv,
    load_parquet,
    refactor_folder_file,
)
from ..window_aggregations import WindowAggregation
from .bronze_to_silver import processing_bronze_file_raw_level, processing_silver_file
from .silver_to_gold import processing_file_ac_progress, processing_file_feature_level


# fmt: off
class EvalClass:

    def __init__(self) -> None:

        # rutas de interes
        self.DATASET_NAME = "02_evaluaciones"

        self.BRONZE_FILES_PATH = '00_data/00_bronze/02_evaluaciones/'
        self.SILVER_FILES_PATH = '00_data/01_silver/02_evaluaciones/'
        self.GOLD_FILES_PATH = '00_data/02_gold/02_evaluaciones/'
        
        self.REPORT_OUTPUT_PATH = "00_data/05_reportes/00_eda"
        
        self.SILVER_FILES_PATH_RAW = '00_data/01_silver/02_evaluaciones/00_RAW'
        self.SILVER_FILES_PATH_EOP = '00_data/01_silver/02_evaluaciones/01_EOP'
        self.SILVER_FILES_PATH_DATE = '00_data/01_silver/02_evaluaciones/02_DATE'

        self.GOLD_FILES_PATH_EOP = '00_data/02_gold/02_evaluaciones/01_EOP'
        self.GOLD_FILES_PATH_EOP_AC = '00_data/02_gold/02_evaluaciones/01_EOP_AC'
        self.GOLD_FILES_PATH_WINDOWN_AGG = '00_data/02_gold/02_evaluaciones/03_WINDOWAGG'
        self.GOLD_FILES_PATH_DATE = '00_data/02_gold/02_evaluaciones/02_DATE'
        self.GOLD_FILES_PATH_DATE_AC = '00_data/02_gold/02_evaluaciones/02_DATE_AC'

        self.CALENDAR_EOP = '00_data/03_assets/calendar_eop.parquet'
        self.CALENDAR_DATE = '00_data/03_assets/calendar_date.parquet'
        self.MATRICULADOS_BASE = '00_data/03_assets/matriculados_eop.parquet'
        
        self.ALIAS_EOP = 'evaluaciones_eop'
        self.ALIAS_EOP_AC = 'evaluaciones_eop_ac'
        self.ALIAS_DATE = 'evaluaciones_date'
        self.ALIAS_DATE_AC = 'evaluaciones_date_ac'

        self.REFACTOR_PERIOD = None

        # generar parametros propios de la clase
        self._generate_parameters()

        pass

    def bronze_to_silver_raw(self) -> None:
        process_start_time = datetime.datetime.now()
        files = get_all_files(self.BRONZE_FILES_PATH, '*.csv')
        if files:
            for file in files:
                filename_no_extension = os.path.basename(file).split('.')[0]

                if self.REFACTOR_PERIOD is not None:
                    
                    if self.REFACTOR_PERIOD != int(filename_no_extension):
                        continue
                    else:
                        refactor_folder_file(os.path.join(self.SILVER_FILES_PATH_RAW, f'{filename_no_extension}.parquet'), flag_folder=0)

                try:
                    start_time = time.time()
                    df = load_csv(file)
                    df.rename(self.BRONZE_FILE_FEATURES, axis=1, inplace=True)
                    check_columns_val = check_columns(df, set(self.BRONZE_FILE_FEATURES.values()))
                    if check_columns_val:
                        df = processing_bronze_file_raw_level(df, self.SILVER_FILE_FEATURES_RAW)
                        create_parquet_from_df(df, self.SILVER_FILES_PATH_RAW, f'{filename_no_extension}.parquet')
                        end_time = time.time()
                        print(f'\t{filename_no_extension}: {(end_time-start_time):.3f} segs')
                    else:
                        print(f'\t{filename_no_extension}:\tRevisar variables de la capa bronze')
                except Exception as e:
                    print(f'\t{filename_no_extension}:\tError de procesamiento\t{e}')
        process_end_time = datetime.datetime.now()
        print(f'Duracion: {process_end_time - process_start_time}')
        return None

    def silver_to_silver_eop(self) -> None:
        process_start_time = datetime.datetime.now()
        files = get_all_files(self.SILVER_FILES_PATH_RAW, '*.parquet')
        calendar = load_parquet(self.CALENDAR_EOP)
        if files:
            for file in files:
                filename_no_extension = os.path.basename(file).split('.')[0]

                if self.REFACTOR_PERIOD is not None:
                    if self.REFACTOR_PERIOD != int(filename_no_extension):
                        continue
                    else:
                        refactor_folder_file(os.path.join(self.SILVER_FILES_PATH_EOP, f'{filename_no_extension}.parquet'), flag_folder=0)

                try:
                    start_time = time.time()
                    df = load_parquet(file)
                    df = df.merge(calendar, how='left', on=['periodo'])
                    df = processing_silver_file(df)
                    create_parquet_from_df(df, self.SILVER_FILES_PATH_EOP, f'{filename_no_extension}.parquet')
                    end_time = time.time()
                    print(f'\t{filename_no_extension}: {(end_time-start_time):.3f} segs')
                except Exception as e:
                    print(f'\t{filename_no_extension}:\tError de procesamiento\t{e}')
        process_end_time = datetime.datetime.now()
        print(f'Duracion: {process_end_time - process_start_time}')
        return None
    
    def silver_to_silver_date(self) -> None:
        process_start_time = datetime.datetime.now()
        files = get_all_files(self.SILVER_FILES_PATH_RAW, '*.parquet')
        if files:
            for file in files:
                filename_no_extension = os.path.basename(file).split('.')[0]

                if self.REFACTOR_PERIOD is not None:
                    if self.REFACTOR_PERIOD != int(filename_no_extension):
                        continue
                    else:
                        refactor_folder_file(os.path.join(self.SILVER_FILES_PATH_DATE, f'{filename_no_extension}'), flag_folder=1)

                print(f'\t{filename_no_extension}:')
                try:
                    calendar = load_parquet(self.CALENDAR_DATE)
                    calendar = calendar.loc[calendar['periodo'] == int(filename_no_extension)]
                    for corte in calendar['fecha_corte'].astype(str).str[:10].unique():
                        start_time = time.time()
                        df = load_parquet(file)
                        df.loc[:, 'fecha_corte'] = corte
                        df['fecha_corte'] = pd.to_datetime(df['fecha_corte'], errors='coerce')
                        df = processing_silver_file(df)
                        create_parquet_from_df(df, os.path.join(self.SILVER_FILES_PATH_DATE, f'{filename_no_extension}'), f'{filename_no_extension}_{corte.replace("-","")}.parquet')
                        end_time = time.time()
                        print(f'\t{corte}: {(end_time-start_time):.3f} segs')
                except Exception as e:
                    print(f'\t{corte}:\tError de procesamiento\t{e}')
        process_end_time = datetime.datetime.now()
        print(f'Duracion: {process_end_time - process_start_time}')
        return None

    def silver_to_gold_eop(self) -> None:
        process_start_time = datetime.datetime.now()
        files = get_all_files(self.SILVER_FILES_PATH_EOP, '*.parquet')
        # print(files)
        if files:
            for file in files:
                filename_no_extension = os.path.basename(file).split('.')[0]

                if self.REFACTOR_PERIOD is not None:
                    if self.REFACTOR_PERIOD != int(filename_no_extension):
                        continue
                    # else:
                    #     refactor_folder_file(os.path.join(self.GOLD_FILES_PATH_EOP, f'{filename_no_extension}_features.parquet'), flag_folder=0)

                try:
                    start_time = time.time()
                    # print(file)
                    df = load_parquet(file)
                    df = processing_file_feature_level(df, self.GOLD_FILE_FEATURES_DICT)
                    df = formatting(df, self.GOLD_FILE_FEATURES_DICT)
                    create_parquet_from_df(df, self.GOLD_FILES_PATH_EOP, f'{filename_no_extension}_features.parquet')                    
                    end_time = time.time()
                    print(f'\t{filename_no_extension}: {(end_time-start_time):.3f} segs',)
                except Exception as e:
                    print(f'\t{filename_no_extension}:\tError de procesamiento\t{e}')
                
            now_str = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
            consolidate_parquet_files_from_folder(self.GOLD_FILES_PATH_EOP, '*_features.parquet', self.GOLD_FILES_PATH, f'{self.ALIAS_EOP}_{now_str}.parquet', parts=None)
        process_end_time = datetime.datetime.now()
        print(f'Duracion: {process_end_time - process_start_time}')
        return None
    
    def silver_to_gold_eop_ac(self) -> None:
        process_start_time = datetime.datetime.now()
        files = get_all_files(self.SILVER_FILES_PATH_EOP, '*.parquet')
        if files:
            for file in files:
                filename_no_extension = os.path.basename(file).split('.')[0]

                if self.REFACTOR_PERIOD is not None:
                    if self.REFACTOR_PERIOD != int(filename_no_extension):
                        continue
                    # else:
                    #     refactor_folder_file(os.path.join(self.GOLD_FILES_PATH_EOP, f'{filename_no_extension}_features.parquet'), flag_folder=0)

                try:
                    start_time = time.time()
                    df = load_parquet(file)
                    df = processing_file_ac_progress(df, self.SILVER_FILE_AC_COLUMNS)
                    create_parquet_from_df(df, self.GOLD_FILES_PATH_EOP_AC, f'{filename_no_extension}_features.parquet')                    
                    end_time = time.time()
                    print(f'\t{filename_no_extension}: {(end_time-start_time):.3f} segs')
                except Exception as e:
                    print(f'\t{filename_no_extension}:\tError de procesamiento\t{e}')
        
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

        # Crea nuevas columnas en el DataFrame basado en umbrales específicos para la columna 'avg_evaluaciones'.
        windowAgg.create_column_threshold(col="avg_evaluaciones", threshold=10, comparison_op_str=">=")
        windowAgg.create_column_threshold(col="avg_evaluaciones", threshold=15, comparison_op_str=">=")

        # Crea nuevas columnas en el DataFrame basado en umbrales específicos para la columna 'avg_practicas'.
        windowAgg.create_column_threshold(col="avg_practicas", threshold=10, comparison_op_str=">=")
        windowAgg.create_column_threshold(col="avg_practicas", threshold=15, comparison_op_str=">=")

        # Crea nuevas columnas en el DataFrame basado en umbrales específicos para la columna 'avg_examenes'.
        windowAgg.create_column_threshold(col="avg_examenes", threshold=10, comparison_op_str=">=")
        windowAgg.create_column_threshold(col="avg_examenes", threshold=15, comparison_op_str=">=")

        # Crea nuevas columnas en el DataFrame basado en umbrales específicos para la columna 'n_eval'.
        windowAgg.create_column_threshold(col="n_eval", threshold=5, comparison_op_str=">=")
        windowAgg.create_column_threshold(col="n_eval", threshold=15, comparison_op_str=">=")
        windowAgg.create_column_threshold(col="n_eval", threshold=25, comparison_op_str=">=")

        # Crea nuevas columnas en el DataFrame basado en umbrales específicos para la columna 'n_eval_practicas'.
        windowAgg.create_column_threshold(col="n_eval_practicas", threshold=5, comparison_op_str=">=")
        windowAgg.create_column_threshold(col="n_eval_practicas", threshold=15, comparison_op_str=">=")
        windowAgg.create_column_threshold(col="n_eval_practicas", threshold=25, comparison_op_str=">=")

        # Crea nuevas columnas en el DataFrame basado en umbrales específicos para la columna 'n_eval_aprob'.
        windowAgg.create_column_threshold(col="n_eval_aprob", threshold=5, comparison_op_str=">=")
        windowAgg.create_column_threshold(col="n_eval_aprob", threshold=15, comparison_op_str=">=")
        windowAgg.create_column_threshold(col="n_eval_aprob", threshold=25, comparison_op_str=">=")

        # Genera todas las agregaciones de ventana especificadas y añade las nuevas columnas al DataFrame original.
        df_windownAgg = windowAgg.get_aggregations()

        # Obtiene la fecha y hora actual como una cadena en el formato 'YYYYMMDD_HHMMSS'.
        now_str = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
        
        # Guarda el DataFrame resultante en un archivo parquet en la ruta especificada, con un nombre que incluye la fecha y hora actual.
        create_parquet_from_df(df_windownAgg, self.GOLD_FILES_PATH_WINDOWN_AGG, f"evaluaciones_eop_wa_{now_str}.parquet")

        process_end_time = datetime.datetime.now()
        print(f'Duracion: {process_end_time - process_start_time}')
        return None

    
    def silver_to_gold_date(self) -> None:
        process_start_time = datetime.datetime.now()
        folders = get_all_folders(self.SILVER_FILES_PATH_DATE)
        if folders:
            for folder in folders:
                files = get_all_files(os.path.join(self.SILVER_FILES_PATH_DATE, folder), '*.parquet')

                if self.REFACTOR_PERIOD is not None:
                    if self.REFACTOR_PERIOD != int(folder):
                        continue
                    else:
                        refactor_folder_file(os.path.join(self.GOLD_FILES_PATH_DATE, f'{folder}'), flag_folder=1)

                if files:
                    print(f'\t{folder}:')
                    for file in files:
                        filename_no_extension = os.path.basename(file).split('.')[0]
                        try:
                            start_time = time.time()
                            df = load_parquet(file)
                            df = processing_file_feature_level(df, self.GOLD_FILE_FEATURES_DICT)
                            # df = merge_matricula(df, self.MATRICULADOS_BASE, int(filename_no_extension.split('_')[0]))
                            df = formatting(df, self.GOLD_FILE_FEATURES_DICT)
                            df = merge_calendar(df, self.CALENDAR_DATE)
                            create_parquet_from_df(df, os.path.join(self.GOLD_FILES_PATH_DATE, folder), f'{filename_no_extension}_features.parquet')                    
                            end_time = time.time()
                            print(f'\t{filename_no_extension}: {(end_time-start_time):.3f} segs')
                        except Exception as e:
                            print(f'\t{filename_no_extension}:\tError de procesamiento\t{e}')

                consolidate_parquet_files_from_folder(os.path.join(self.GOLD_FILES_PATH_DATE, folder), '*_features.parquet', os.path.join(self.GOLD_FILES_PATH_DATE, 'CONSOLIDADO'), f'{folder}_features.parquet')
            
            now_str = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
            consolidate_parquet_files_from_folder(os.path.join(self.GOLD_FILES_PATH_DATE, 'CONSOLIDADO'), '*_features.parquet', self.GOLD_FILES_PATH, f'{self.ALIAS_DATE}_{now_str}.parquet', parts=4)
        process_end_time = datetime.datetime.now()
        print(f'Duracion: {process_end_time - process_start_time}')
        return None
    
    def silver_to_gold_date_ac(self) -> None:
        process_start_time = datetime.datetime.now()
        folders = get_all_folders(self.SILVER_FILES_PATH_DATE)
        if folders:
            for folder in folders:
                files = get_all_files(os.path.join(self.SILVER_FILES_PATH_DATE, folder), '*.parquet')

                if self.REFACTOR_PERIOD is not None:
                    if self.REFACTOR_PERIOD != int(folder):
                        continue
                    else:
                        refactor_folder_file(os.path.join(self.GOLD_FILES_PATH_DATE_AC, f'{folder}'), flag_folder=1)

                if files:
                    print(f'\t{folder}:')
                    for file in files:
                        filename_no_extension = os.path.basename(file).split('.')[0]
                        
                        try:
                            start_time = time.time()
                            df = load_parquet(file)
                            df = processing_file_ac_progress(df, self.SILVER_FILE_AC_COLUMNS)
                            df = merge_calendar(df, self.CALENDAR_DATE)
                            create_parquet_from_df(df, os.path.join(self.GOLD_FILES_PATH_DATE_AC, folder), f'{filename_no_extension}_features.parquet')                    
                            end_time = time.time()
                            print(f'\t{filename_no_extension}: {(end_time-start_time):.3f} segs')
                        except Exception as e:
                            print(f'\t{filename_no_extension}:\tError de procesamiento\t{e}')

                consolidate_parquet_files_from_folder(os.path.join(self.GOLD_FILES_PATH_DATE_AC, folder), '*_features.parquet', os.path.join(self.GOLD_FILES_PATH_DATE_AC, 'CONSOLIDADO'), f'{folder}_features.parquet')
        process_end_time = datetime.datetime.now()
        print(f'Duracion: {process_end_time - process_start_time}')
        return None

    def check_refactor(self) -> None:
        if self.REFACTOR_PERIOD is None:
            refactor_folder_file(self.SILVER_FILES_PATH)
            refactor_folder_file(self.GOLD_FILES_PATH)

    def _generate_parameters(self) -> None:

        self.BRONZE_FILE_FEATURES = {
            'COD_PERIODO': 'periodo',
            '#COD_PERIODO': 'periodo',
            'FECHA_CREACION': 'fecha_creacion',
            'COD_ALUMNO': 'cod_alumno',
            'COD_CURSO': 'cod_curso',
            'DES_CURSO': 'curso',
            'DES_FORMULA': 'formula',
            'COD_TIPO_PRUEBA': 'cod_tipo_prueba',
            'NUM_TIPO_PRUEBA': 'num_prueba',
            'NOTA': 'nota',
        }

        self.SILVER_FILE_FEATURES_RAW = [
            'periodo',
            'fecha_creacion',
            'cod_alumno',
            'cod_curso',
            'curso',
            'formula',
            'cod_tipo_prueba',
            'num_prueba',
            'nota',
        ]

        self.GOLD_FILE_FEATURES_DICT = {
            'periodo': 'skip', 
            'cod_alumno': 'skip',
            'fecha_corte': 'skip',

            'n_eval': 0.0,
            'n_eval_asist': 0.0,
            'r_eval_asist': 0.0,
            
            'n_eval_aprob': 0.0,   
            'r_eval_aprob': 0.0,

            'avg_evaluaciones': "skip",
            'avg_evaluaciones_aprob': "skip",

            'n_eval_practicas': 0.0,
            'avg_practicas': "skip",
            'avg_practicas_aprob': "skip",
            'n_eval_practicas_aprob': 0.0,
            'r_eval_practicas_aprob': 0.0,

            'n_eval_examenes': 0.0,
            'avg_examenes': "skip",
            'avg_examenes_aprob': "skip",
            'n_eval_examenes_aprob': 0.0,
            'r_eval_examenes_aprob': 0.0,
            
            'n_eval_trabajos': 0.0,
            'avg_trabajos': "skip",
            'avg_trabajos_aprob': "skip",
            'n_eval_trabajos_aprob': 0.0,
            'r_eval_trabajos_aprob': 0.0,
            
            'n_eval_participacion': 0.0,
            'avg_participacion': "skip",
            'avg_participacion_aprob': "skip",
            'n_eval_participacion_aprob': 0.0,
            'r_eval_participacion_aprob': 0.0,
        }

        self.GOLD_FILE_AGG_FEATURES_LIST = [
            ("avg_evaluaciones_mayor_igual_10", [2, 3, 4], "count"),
            ("avg_evaluaciones_mayor_igual_15", [2, 3, 4], "count"),
            ("avg_practicas_mayor_igual_10", [2, 3, 4], "count"),
            ("avg_practicas_mayor_igual_15", [2, 3, 4], "count"),
            ("avg_examenes_mayor_igual_10", [2, 3, 4], "count"),
            ("avg_examenes_mayor_igual_15", [2, 3, 4], "count"),
            ("n_eval_mayor_igual_5", [2, 3, 4], "count"),
            ("n_eval_mayor_igual_15", [2, 3, 4], "count"),
            ("n_eval_mayor_igual_25", [2, 3, 4], "count"),
            ("n_eval_practicas_mayor_igual_5", [2, 3, 4], "count"),
            ("n_eval_practicas_mayor_igual_15", [2, 3, 4], "count"),
            ("n_eval_practicas_mayor_igual_25", [2, 3, 4], "count"),
            ("n_eval_aprob_mayor_igual_5", [2, 3, 4], "count"),
            ("n_eval_aprob_mayor_igual_15", [2, 3, 4], "count"),
            ("n_eval_aprob_mayor_igual_25", [2, 3, 4], "count"),
            ("n_eval", [2, 3, 4], "sum"),
            ("n_eval_asist", [2, 3, 4], "sum"),
            ("n_eval_aprob", [2, 3, 4], "sum"),
            ("n_eval_practicas", [2, 3, 4], "sum"),
            ("n_eval_practicas_aprob", [2, 3, 4], "sum"),
            ("n_eval_examenes", [2, 3, 4], "sum"),
            ("n_eval_examenes_aprob", [2, 3, 4], "sum"),
            ("n_eval_trabajos", [2, 3, 4], "sum"),
            ("n_eval_trabajos_aprob", [2, 3, 4], "sum"),
            ("n_eval_participacion", [2, 3, 4], "sum"),
            ("n_eval_participacion_aprob", [2, 3, 4], "sum"),
            ("r_eval_asist", [2, 3, 4], "mean"),
            ("r_eval_aprob", [2, 3, 4], "mean"),
            ("avg_evaluaciones", [2, 3, 4], "mean"),
            ("avg_evaluaciones_aprob", [2, 3, 4], "mean"),
            ("avg_practicas", [2, 3, 4], "mean"),
            ("avg_practicas_aprob", [2, 3, 4], "mean"),
            ("r_eval_practicas_aprob", [2, 3, 4], "mean"),
            ("avg_examenes", [2, 3, 4], "mean"),
            ("avg_examenes_aprob", [2, 3, 4], "mean"),
            ("r_eval_examenes_aprob", [2, 3, 4], "mean"),
            ("avg_trabajos", [2, 3, 4], "mean"),
            ("avg_trabajos_aprob", [2, 3, 4], "mean"),
            ("r_eval_trabajos_aprob", [2, 3, 4], "mean"),
            ("avg_participacion", [2, 3, 4], "mean"),
            ("avg_participacion_aprob", [2, 3, 4], "mean"),
            ("r_eval_participacion_aprob", [2, 3, 4], "mean"),
        ]

        self.SILVER_FILE_AC_COLUMNS = [
            "periodo",
            "cod_alumno",
            "fecha_corte",
            "cod_curso",
            "pct_avance",
            "flag_curso_completado",
            "prom_final_actual",
            "prom_max_restante",
            "prom_final_max_posible",
            # "prom_ponderado_actual",
        ]

        
        return None
