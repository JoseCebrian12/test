import datetime
import os
import time
import traceback

from ..general.dataframe_funcs import merge_calendar
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
class RetiroCicloClass:

    def __init__(self) -> None:

        # rutas de interes
        self.DATASET_NAME = "03_retiro_ciclo"

        self.BRONZE_FILES_PATH = '00_data/00_bronze/03_retiro_ciclo/'
        self.SILVER_FILES_PATH = '00_data/01_silver/03_retiro_ciclo/'
        self.GOLD_FILES_PATH = '00_data/02_gold/03_retiro_ciclo/'

        self.REPORT_OUTPUT_PATH = "00_data/05_reportes/00_eda"
        
        self.SILVER_FILES_PATH_RAW = '00_data/01_silver/03_retiro_ciclo/00_RAW'
        self.SILVER_FILES_PATH_EOP = '00_data/01_silver/03_retiro_ciclo/01_EOP'
        self.SILVER_FILES_PATH_DATE = '00_data/01_silver/03_retiro_ciclo/02_DATE'

        self.GOLD_FILES_PATH_EOP = '00_data/02_gold/03_retiro_ciclo/01_EOP'
        self.GOLD_FILES_PATH_DATE = '00_data/02_gold/03_retiro_ciclo/02_DATE'
        self.GOLD_FILES_PATH_WINDOWN_AGG = '00_data/02_gold/03_retiro_ciclo/03_WINDOWAGG'

        self.CALENDAR_EOP = '00_data/03_assets/calendar_eop.parquet'
        self.CALENDAR_DATE = '00_data/03_assets/calendar_date.parquet'
        self.MATRICULADOS_BASE = '00_data/03_assets/matriculados_eop.parquet'
        
        self.ALIAS_EOP = 'retciclo_eop'
        self.ALIAS_DATE = 'retciclo_date'

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
                    # print(check_columns_val)
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
        if files:        
            if self.REFACTOR_PERIOD is not None:
                refactor_folder_file(os.path.join(self.SILVER_FILES_PATH_EOP, f'{self.ALIAS_EOP}_silver.parquet'), flag_folder=0)
            try:
                start_time = time.time()
                consolidate_parquet_files_from_folder(self.SILVER_FILES_PATH_RAW, '*.parquet', self.SILVER_FILES_PATH_EOP, f'{self.ALIAS_EOP}_silver.parquet', parts=None)
                end_time = time.time()
                print(f'\tConsolidar raw files: {(end_time-start_time):.3f} segs')
            except Exception as e:
                print(f'\tConsolidar raw files:\tError de procesamiento\t{e}')
        process_end_time = datetime.datetime.now()
        print(f'Duracion: {process_end_time - process_start_time}')
        return None
    
    def silver_to_gold_eop(self) -> None:
        process_start_time = datetime.datetime.now()
        files = get_all_files(self.SILVER_FILES_PATH_EOP, '*.parquet')
        if files:
            for file in files:
                df = load_parquet(file)
                calendar = load_parquet(self.CALENDAR_EOP)
                base = load_parquet(self.MATRICULADOS_BASE)

                for periodo in list(calendar['periodo'].unique()):
                    
                    if self.REFACTOR_PERIOD is not None:
                        if self.REFACTOR_PERIOD != int(periodo):
                            continue
                        else:
                            refactor_folder_file(os.path.join(self.GOLD_FILES_PATH_EOP, f'{periodo}_silver.parquet'), flag_folder=0)
                    try:
                        start_time = time.time()
                        df_ = df.copy().reset_index(drop=True)
                        base_ = base.copy().reset_index(drop=True)

                        df_ = df_[df_['periodo'] <= periodo]
                        base_ = base_[base_['periodo'] == periodo]
                        
                        result = processing_file_feature_level(df_, base_)
                        create_parquet_from_df(result, self.GOLD_FILES_PATH_EOP, f'{periodo}_features.parquet')
                        end_time = time.time()
                        print(f'\t{periodo}: {(end_time-start_time):.3f} segs')
                    except Exception as e:
                        print(f'\t{periodo}: \tError de procesamiento\t{e}')

            now_str = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
            consolidate_parquet_files_from_folder(self.GOLD_FILES_PATH_EOP, '*_features.parquet', self.GOLD_FILES_PATH, f'{self.ALIAS_EOP}_{now_str}.parquet')
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

        # Crea nuevas columnas en el DataFrame basado en el número de retiros de ciclos acumulados (retiro_ciclo_acum) y umbrales específicos.
        windowAgg.create_column_threshold(col="retiro_ciclo_acum", threshold=1, comparison_op_str=">=")
        windowAgg.create_column_threshold(col="retiro_ciclo_acum", threshold=3, comparison_op_str=">=")
        windowAgg.create_column_threshold(col="retiro_ciclo_acum", threshold=6, comparison_op_str=">=")

        # Genera todas las agregaciones de ventana especificadas y añade las nuevas columnas al DataFrame original.
        df_windownAgg = windowAgg.get_aggregations()

        # Obtiene la fecha y hora actual como una cadena en el formato 'YYYYMMDD_HHMMSS'.
        now_str = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
        
        # Guarda el DataFrame resultante en un archivo parquet en la ruta especificada, con un nombre que incluye la fecha y hora actual.
        create_parquet_from_df(df_windownAgg, self.GOLD_FILES_PATH_WINDOWN_AGG, f"retciclo_eop_wa_{now_str}.parquet")

        process_end_time = datetime.datetime.now()
        print(f'Duracion: {process_end_time - process_start_time}')
        return None
    
    def silver_to_gold_date(self) -> None:

        process_start_time = datetime.datetime.now()
        files = get_all_files(self.SILVER_FILES_PATH_EOP, '*.parquet')
        if files:
            for file in files:
                df = load_parquet(file)
                calendar = load_parquet(self.CALENDAR_DATE)
                base = load_parquet(self.MATRICULADOS_BASE)

                for periodo in list(calendar['periodo'].unique()):
                    calendar_sub = calendar[calendar['periodo'] == periodo].copy()
                    if self.REFACTOR_PERIOD is not None:
                        if self.REFACTOR_PERIOD != int(periodo):
                            continue
                        else:
                            refactor_folder_file(os.path.join(self.GOLD_FILES_PATH_DATE, f'{periodo}'), flag_folder=1)
                    
                    for fecha_corte in list(calendar_sub['fecha_corte'].unique()):
                        try:
                            start_time = time.time()
                            df_ = df.copy().reset_index(drop=True)
                            base_ = base.copy().reset_index(drop=True)

                            df_ = df_[df_['fec_retciclo'] <= fecha_corte]
                            base_ = base_[base_['periodo'] == periodo]
                            
                            result = processing_file_feature_level(df_, base_)
                            result['fecha_corte'] = fecha_corte
                            result = merge_calendar(result, self.CALENDAR_DATE)
                            create_parquet_from_df(result, os.path.join(self.GOLD_FILES_PATH_DATE, f'{periodo}'), f'{periodo}_{fecha_corte.strftime("%Y%m%d")}_features.parquet')
                            end_time = time.time()
                            print(f'\t{fecha_corte.strftime("%Y-%m-%d")}: {(end_time-start_time):.3f} segs')
                        except Exception as e:
                            print(traceback.print_exc())
                            print(f'\t{fecha_corte.strftime("%Y-%m-%d")}: Error de procesamiento\t{e}')
                
                    consolidate_parquet_files_from_folder(os.path.join(self.GOLD_FILES_PATH_DATE, f'{periodo}'), '*_features.parquet', os.path.join(self.GOLD_FILES_PATH_DATE, 'CONSOLIDADO'), f'{periodo}_features.parquet')
            
            now_str = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
            consolidate_parquet_files_from_folder(os.path.join(self.GOLD_FILES_PATH_DATE, 'CONSOLIDADO'), '*_features.parquet', self.GOLD_FILES_PATH, f'{self.ALIAS_DATE}_{now_str}.parquet', parts=None)
        process_end_time = datetime.datetime.now()
        print(f'Duracion: {process_end_time - process_start_time}')
        return None

    def check_refactor(self) -> None:
        if self.REFACTOR_PERIOD is None:
            refactor_folder_file(self.SILVER_FILES_PATH)
            refactor_folder_file(self.GOLD_FILES_PATH)

    def _generate_parameters(self) -> None:

        self.BRONZE_FILE_FEATURES = {
            'COD_PERIODO_ACADEMICO': 'periodo',
            'COD_ALUMNO': 'cod_alumno',
            'COD_TIPO_RETIRO': 'cod_tipo_retciclo',
            'FEC_SOLCITUD': 'fec_sol_retciclo',
            'FEC_RETIRO': 'fec_retciclo',
            'COD_MOTIVO_RETIRO': 'cod_motivo_retciclo',
        }

        self.SILVER_FILE_FEATURES_RAW = [
            'periodo',
            'cod_alumno',
            'fec_retciclo',
        ]

        self.GOLD_FILE_AGG_FEATURES_LIST = [
            ("retiro_ciclo_acum_mayor_igual_1", [2, 3, 4], "count"),
            ("retiro_ciclo_acum_mayor_igual_3", [2, 3, 4], "count"),
            ("retiro_ciclo_acum_mayor_igual_6", [2, 3, 4], "count"),
            ("flag_retiro_ciclo", [2, 3, 4], "sum"),
            ("retiro_ciclo_acum", [2, 3, 4], "sum"),
        ]

        return None
