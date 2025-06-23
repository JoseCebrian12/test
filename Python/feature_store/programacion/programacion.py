import datetime
import os
import time

import pandas as pd

from ..general.dataframe_funcs import merge_calendar
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

from .bronze_to_silver import processing_bronze_file_raw_level, processing_silver_file
from .silver_to_gold import processing_file_feature_level


# fmt: off
class ProgramacionClass:

    def __init__(self) -> None:

        # rutas de interes
        self.DATASET_NAME = "06_programacion"

        self.BRONZE_FILES_PATH = '00_data/00_bronze/06_programacion/'
        self.SILVER_FILES_PATH = '00_data/01_silver/06_programacion/'
        self.GOLD_FILES_PATH = '00_data/02_gold/06_programacion/'

        self.REPORT_OUTPUT_PATH = "00_data/05_reportes/00_eda"
        
        self.SILVER_FILES_PATH_RAW = '00_data/01_silver/06_programacion/00_RAW'
        self.SILVER_FILES_PATH_EOP = '00_data/01_silver/06_programacion/01_EOP'
        self.SILVER_FILES_PATH_DATE = '00_data/01_silver/06_programacion/02_DATE'

        self.GOLD_FILES_PATH_EOP = '00_data/02_gold/06_programacion/01_EOP'
        self.GOLD_FILES_PATH_DATE = '00_data/02_gold/06_programacion/02_DATE'
        self.GOLD_FILES_PATH_WINDOWN_AGG = '00_data/02_gold/06_programacion/03_WINDOWAGG'

        self.CALENDAR_EOP = '00_data/03_assets/calendar_eop.parquet'
        self.CALENDAR_DATE = '00_data/03_assets/calendar_date.parquet'
        self.MATRICULADOS_BASE = '00_data/03_assets/matriculados_eop.parquet'
        
        self.ALIAS_EOP = 'programacion_eop'
        self.ALIAS_DATE = 'programacion_date'

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
                        df = processing_bronze_file_raw_level(df)
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
                    # else:
                    #     refactor_folder_file(os.path.join(self.SILVER_FILES_PATH_DATE, f'{filename_no_extension}'), flag_folder=1)

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
        if files:
            for file in files:
                filename_no_extension = os.path.basename(file).split('.')[0]

                if self.REFACTOR_PERIOD is not None:
                    if self.REFACTOR_PERIOD != int(filename_no_extension):
                        continue
                    else:
                        refactor_folder_file(os.path.join(self.GOLD_FILES_PATH_EOP, f'{filename_no_extension}_features.parquet'), flag_folder=0)

                try:
                    start_time = time.time()
                    df = load_parquet(file)
                    df = processing_file_feature_level(df)
                    # df = merge_matricula(df, self.MATRICULADOS_BASE, int(filename_no_extension))
                    # df = formatting(df)
                    create_parquet_from_df(df, self.GOLD_FILES_PATH_EOP, f'{filename_no_extension}_features.parquet')                    
                    end_time = time.time()
                    print(f'\t{filename_no_extension}: {(end_time-start_time):.3f} segs')
                except Exception as e:
                    print(f'\t{filename_no_extension}:\tError de procesamiento\t{e}')
            now_str = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
            consolidate_parquet_files_from_folder(self.GOLD_FILES_PATH_EOP, '*_features.parquet', self.GOLD_FILES_PATH, f'{self.ALIAS_EOP}_{now_str}.parquet', parts=None)
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
                            df = processing_file_feature_level(df)
                            # df = merge_matricula(df, self.MATRICULADOS_BASE, int(filename_no_extension.split('_')[0]))
                            # df = formatting(df)
                            df = merge_calendar(df, self.CALENDAR_DATE)
                            create_parquet_from_df(df, os.path.join(self.GOLD_FILES_PATH_DATE, folder), f'{filename_no_extension}_features.parquet')                    
                            end_time = time.time()
                            print(f'\t{filename_no_extension}: {(end_time-start_time):.3f} segs')
                        except Exception as e:
                            print(f'\t{filename_no_extension}:\tError de procesamiento\t{e}')
                consolidate_parquet_files_from_folder(os.path.join(self.GOLD_FILES_PATH_DATE, folder), '*_features.parquet', os.path.join(self.GOLD_FILES_PATH_DATE, 'CONSOLIDADO'), f'{folder}_features.parquet')
            
            now_str = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
            consolidate_parquet_files_from_folder(os.path.join(self.GOLD_FILES_PATH_DATE, 'CONSOLIDADO'), '*_features.parquet', self.GOLD_FILES_PATH, f'{self.ALIAS_DATE}_{now_str}.parquet')
        process_end_time = datetime.datetime.now()
        print(f'Duracion: {process_end_time - process_start_time}')
        return None
    
    def check_refactor(self) -> None:
        if self.REFACTOR_PERIOD is None:
            refactor_folder_file(self.SILVER_FILES_PATH)
            refactor_folder_file(self.GOLD_FILES_PATH)
    
    def _generate_parameters(self) -> None:

        self.BRONZE_FILE_FEATURES = {
            "#COD_SESION": "cod_sesion",
            "COD_SESION": "cod_sesion",
            "COD_MODALIDAD_ESTUDIO": "modalidad_estudio",
            "COD_PERIODO_ACADEMICO": "periodo",
            "COD_TIPO_SESION": "cod_tipo_sesion",
            "DES_TIPO_SESION": "des_tipo_sesion",
            "FEC_INICIO_SESION": "fec_inicio_sesion",
            "COD_AULA": "cod_aula",
            "COD_SECCION": "cod_seccion",
            "COD_GRUPO": "cod_grupo",
            "COD_CURSO": "cod_curso",
            "DES_CURSO": "des_curso",
        }

        self.SILVER_FILE_FEATURES_RAW = [
            "periodo",
            "cod_alumno",
            "cod_curso",
            "fecha_sesion",
            "conteo_innprogramacion",
        ]

        return None
