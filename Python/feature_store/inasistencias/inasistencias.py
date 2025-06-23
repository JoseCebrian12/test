import datetime
import os
import time

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
class InasistenciaClass:

    def __init__(self) -> None:

        # rutas de interes
        self.DATASET_NAME = "07_inasistencias"

        self.BRONZE_FILES_PATH = '00_data/00_bronze/07_inasistencias/'
        self.SILVER_FILES_PATH = '00_data/01_silver/07_inasistencias/'
        self.GOLD_FILES_PATH = '00_data/02_gold/07_inasistencias/'

        self.REPORT_OUTPUT_PATH = "00_data/05_reportes/00_eda"
        
        self.SILVER_FILES_PATH_RAW = '00_data/01_silver/07_inasistencias/00_RAW'
        self.SILVER_FILES_PATH_EOP = '00_data/01_silver/07_inasistencias/01_EOP'
        self.SILVER_FILES_PATH_DATE = '00_data/01_silver/07_inasistencias/02_DATE'

        self.PROGRAMACION_SILVER_FILES_PATH_RAW = '00_data/01_silver/06_programacion/00_RAW'
        self.PROGRAMACION_SILVER_FILES_PATH_EOP = '00_data/01_silver/06_programacion/01_EOP'
        self.PROGRAMACION_SILVER_FILES_PATH_DATE = '00_data/01_silver/06_programacion/02_DATE'

        self.GOLD_FILES_PATH_EOP = '00_data/02_gold/07_inasistencias/01_EOP'
        self.GOLD_FILES_PATH_DATE = '00_data/02_gold/07_inasistencias/02_DATE'
        self.GOLD_FILES_PATH_WINDOWN_AGG = '00_data/02_gold/07_inasistencias/03_WINDOWAGG'

        self.PROGRAMACION_GOLD_FILES_PATH_EOP = '00_data/02_gold/06_programacion/01_EOP'
        self.PROGRAMACION_GOLD_FILES_PATH_DATE = '00_data/02_gold/06_programacion/02_DATE'

        self.CALENDAR_EOP = '00_data/03_assets/calendar_eop.parquet'
        self.CALENDAR_DATE = '00_data/03_assets/calendar_date.parquet'
        self.MATRICULADOS_BASE = '00_data/03_assets/matriculados_eop.parquet'
        
        self.ALIAS_EOP = 'inasistencias_eop'
        self.ALIAS_DATE = 'inasistencias_date'

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
        progr_files = get_all_files(self.PROGRAMACION_SILVER_FILES_PATH_EOP, '*.parquet')
        if files:
            for file, progr_file in zip(files, progr_files):
                if os.path.basename(file) == os.path.basename(progr_file):
                    filename_no_extension = os.path.basename(file).split('.')[0]

                    if self.REFACTOR_PERIOD is not None:
                        if self.REFACTOR_PERIOD != int(filename_no_extension):
                            continue
                        else:
                            refactor_folder_file(os.path.join(self.SILVER_FILES_PATH_EOP, f'{filename_no_extension}.parquet'), flag_folder=0)

                    try:
                        start_time = time.time()
                        df_inasistencia = load_parquet(file)
                        df_programacion = load_parquet(progr_file)
                        df = processing_silver_file(df_inasistencia, df_programacion)
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

                progr_files = get_all_files(os.path.join(self.PROGRAMACION_SILVER_FILES_PATH_DATE, filename_no_extension), "*.parquet")
                print(f'\t{filename_no_extension}:')
                for progr_file in progr_files:
                    try:
                        corte = datetime.datetime.strptime(os.path.basename(progr_file)[7:15], "%Y%m%d").strftime("%Y-%m-%d")
                        start_time = time.time()
                        df_inasistencia = load_parquet(file)
                        df_programacion = load_parquet(progr_file)
                        df = processing_silver_file(df_inasistencia, df_programacion)
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
        progr_files = get_all_files(self.PROGRAMACION_GOLD_FILES_PATH_EOP, '*.parquet')
        if files:
            for file, progr_file in zip(files, progr_files):
                if os.path.basename(file)[:4] == os.path.basename(progr_file)[:4]:
                    filename_no_extension = os.path.basename(file).split('.')[0]

                    if self.REFACTOR_PERIOD is not None:
                        if self.REFACTOR_PERIOD != int(filename_no_extension):
                            continue
                        else:
                            refactor_folder_file(os.path.join(self.GOLD_FILES_PATH_EOP, f'{filename_no_extension}_features.parquet'), flag_folder=0)

                    try:
                        start_time = time.time()
                        df_inasistencia = load_parquet(file)
                        df_programacion = load_parquet(progr_file)
                        df = processing_file_feature_level(df_inasistencia, df_programacion)
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
                progr_files = get_all_files(os.path.join(self.PROGRAMACION_GOLD_FILES_PATH_DATE, folder), '*.parquet')

                if self.REFACTOR_PERIOD is not None:
                    if self.REFACTOR_PERIOD != int(folder):
                        continue
                    else:
                        refactor_folder_file(os.path.join(self.GOLD_FILES_PATH_DATE, f'{folder}'), flag_folder=1)
                if files:
                    print(f'\t{folder}:')
                    for file, progr_file in zip(files, progr_files):
                        filename_no_extension = os.path.basename(file).split('.')[0]
                        if os.path.basename(file)[:15] == os.path.basename(progr_file)[:15]:
                            try:
                                start_time = time.time()
                                df_inasistencia = load_parquet(file)
                                df_programacion = load_parquet(progr_file)
                                df = processing_file_feature_level(df_inasistencia, df_programacion)
                                df = merge_calendar(df, self.CALENDAR_DATE)
                                create_parquet_from_df(df, os.path.join(self.GOLD_FILES_PATH_DATE, folder), f'{filename_no_extension}_features.parquet')                    
                                end_time = time.time()
                                print(f'\t{filename_no_extension}: {(end_time-start_time):.3f} segs')
                            except Exception as e:
                                print(f'\t{filename_no_extension}:\tError de procesamiento\t{e}')
                        else:
                            print(os.path.basename(file)[:16], "\t", os.path.basename(progr_file)[:16])
                    consolidate_parquet_files_from_folder(os.path.join(self.GOLD_FILES_PATH_DATE, folder), '*_features.parquet', os.path.join(self.GOLD_FILES_PATH_DATE, 'CONSOLIDADO'), f'{folder}_features.parquet')
            
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
            "#ID_SESION": "cod_sesion",
            "ID_SESION": "cod_sesion",
            "ID_MATRICULA": "id_matricula",
            "COD_PERIODO_MATRICULA": "periodo",
            "COD_MODALIDAD_ESTUDIO": "modalidad_estudio",
            "COD_PRODUCTO_MATRICULA": "producto_matricula",
            "COD_ALUMNO": "cod_alumno",
            "COD_CURSO": "cod_curso",
            "COD_SECCION": "cod_seccion",
            "COD_GRUPO": "cod_grupo",
        }

        self.SILVER_FILE_FEATURES_RAW = [
            "periodo",
            "cod_alumno",
            "cod_curso",
            "fecha_sesion",
            "conteo_innasistencias",
        ]

        return None
