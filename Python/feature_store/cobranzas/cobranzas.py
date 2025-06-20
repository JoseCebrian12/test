import datetime
import os
import time

import numpy as np
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
from .bronze_to_silver import (
    processing_bronze_file_raw_level,
    processing_file_doc_level,
)
from .silver_to_gold import (
    next_cuota_tuition,
    processing_file_feature_level,
    processing_file_perfil_morosidad_date,
    processing_file_perfil_morosidad_eop,
)


# fmt: off
class CobranzasClass:

    def __init__(self) -> None:

        # rutas de interes
        self.DATASET_NAME = "05_cobranzas"

        self.BRONZE_FILES_PATH = "00_data/00_bronze/05_cobranzas/"
        self.SILVER_FILES_PATH = "00_data/01_silver/05_cobranzas/"
        self.GOLD_FILES_PATH = "00_data/02_gold/05_cobranzas/"

        self.REPORT_OUTPUT_PATH = "00_data/05_reportes/00_eda"

        self.SILVER_FILES_PATH_RAW = "00_data/01_silver/05_cobranzas/00_RAW"
        self.SILVER_FILES_PATH_EOP = "00_data/01_silver/05_cobranzas/01_EOP"
        self.SILVER_FILES_PATH_DATE = "00_data/01_silver/05_cobranzas/02_DATE"

        self.GOLD_FILES_PATH_EOP = "00_data/02_gold/05_cobranzas/01_EOP"
        self.GOLD_FILES_PATH_DATE = "00_data/02_gold/05_cobranzas/02_DATE"
        self.GOLD_FILES_PATH_WINDOWN_AGG = "00_data/02_gold/05_cobranzas/03_WINDOWAGG"

        self.GOLD_FILES_PATH_EOP_PERFIL_MOROSIDAD = "00_data/02_gold/05_cobranzas/01_EOP_PM"
        self.GOLD_FILES_PATH_DATE_PERFIL_MOROSIDAD = "00_data/02_gold/05_cobranzas/02_DATE_PM"

        self.CALENDAR_EOP = "00_data/03_assets/calendar_eop.parquet"
        self.CALENDAR_DATE = "00_data/03_assets/calendar_date.parquet"
        self.MATRICULADOS_BASE = "00_data/03_assets/matriculados_eop.parquet"

        self.ASSET_PERFIL_MOROSIDAD_MORA = "00_data/03_assets/perfil_morosidad_mora.parquet"
        self.ASSET_PERFIL_MOROSIDAD_PAGO = "00_data/03_assets/perfil_morosidad_pago.parquet"

        self.ALIAS_EOP = "cobranzas_eop"
        self.ALIAS_DATE = "cobranzas_date"

        self.ALIAS_EOP_PERFIL_MOROSIDAD = "cobranzas_eop_pm"
        self.ALIAS_DATE_PERFIL_MOROSIDAD = "cobranzas_date_pm"

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
                        create_parquet_from_df(df,self.SILVER_FILES_PATH_RAW,f"{filename_no_extension}.parquet",)
                        end_time = time.time()
                        print(f"\t{filename_no_extension}: {(end_time-start_time):.3f} segs")
                    else:
                        print(f"\t{filename_no_extension}:\tRevisar variables de la capa bronze")
                except Exception as e:
                    print(f"\t{filename_no_extension}:\tError de procesamiento\t{e}")
        process_end_time = datetime.datetime.now()
        print(f'Duracion: {process_end_time - process_start_time}')
        return None

    def silver_to_silver_doc_eop(self) -> None:
        process_start_time = datetime.datetime.now()
        files = get_all_files(self.SILVER_FILES_PATH_RAW, "*.parquet")
        calendar = load_parquet(self.CALENDAR_EOP)
        if files:
            for file in files:
                filename_no_extension = os.path.basename(file).split(".")[0]

                if self.REFACTOR_PERIOD is not None:
                    if self.REFACTOR_PERIOD != int(filename_no_extension):
                        continue
                    else:
                        refactor_folder_file(os.path.join(self.SILVER_FILES_PATH_EOP, f"{filename_no_extension}_doc.parquet"), flag_folder=0)

                try:
                    start_time = time.time()
                    df = load_parquet(file)
                    df = df.merge(calendar, how="left", on=["periodo"])
                    df = processing_file_doc_level(df)
                    create_parquet_from_df(df, self.SILVER_FILES_PATH_EOP, f"{filename_no_extension}_doc.parquet")
                    end_time = time.time()
                    print(f"\t{filename_no_extension}: {(end_time-start_time):.3f} segs")
                except Exception as e:
                    print(f"\t{filename_no_extension}:\tError de procesamiento\t{e}")
        process_end_time = datetime.datetime.now()
        print(f'Duracion: {process_end_time - process_start_time}')
        return None

    def silver_to_silver_doc_date(self) -> None:
        process_start_time = datetime.datetime.now()
        files = get_all_files(self.SILVER_FILES_PATH_RAW, "*.parquet")
        if files:
            for file in files:
                filename_no_extension = os.path.basename(file).split(".")[0]

                if self.REFACTOR_PERIOD is not None:
                    if self.REFACTOR_PERIOD != int(filename_no_extension):
                        continue
                    else:
                        refactor_folder_file(os.path.join(self.SILVER_FILES_PATH_DATE, f"{filename_no_extension}"), flag_folder=1)

                print(f"\t{filename_no_extension}:")
                try:
                    calendar = load_parquet(self.CALENDAR_DATE)
                    calendar = calendar.loc[calendar["periodo"] == int(filename_no_extension)]
                    fecha_fin_periodo = (calendar["fecha_fin_periodo"].astype(str).str[:10].unique().flat[0])
                    for corte in calendar["fecha_corte"].astype(str).str[:10].unique():
                        start_time = time.time()
                        df = load_parquet(file)
                        df.loc[:, "fecha_corte"] = corte
                        df["fecha_corte"] = pd.to_datetime(df["fecha_corte"], errors="coerce")
                        df = processing_file_doc_level(df)
                        df.loc[:, "fecha_fin_periodo"] = fecha_fin_periodo
                        df["fecha_fin_periodo"] = pd.to_datetime(df["fecha_fin_periodo"], errors="coerce")
                        create_parquet_from_df(df, os.path.join(self.SILVER_FILES_PATH_DATE, f"{filename_no_extension}"), f'{filename_no_extension}_{corte.replace("-","")}_doc.parquet')
                        end_time = time.time()
                        print(f"\t{corte}: {(end_time-start_time):.3f} segs")
                except Exception as e:
                    print(f"\t{corte}:\tError de procesamiento\t{e}")
        process_end_time = datetime.datetime.now()
        print(f'Duracion: {process_end_time - process_start_time}')
        return None

    def silver_to_gold_eop(self) -> None:
        process_start_time = datetime.datetime.now()
        files = get_all_files(self.SILVER_FILES_PATH_EOP, "*_doc.parquet")
        if files:
            for file in files:
                filename_no_extension = os.path.basename(file).split("_")[0]

                if self.REFACTOR_PERIOD is not None:
                    if self.REFACTOR_PERIOD != int(filename_no_extension):
                        continue
                    else:
                        refactor_folder_file(os.path.join(self.GOLD_FILES_PATH_EOP, f"{filename_no_extension}_features.parquet"), flag_folder=0)

                try:
                    start_time = time.time()
                    df = load_parquet(file)
                    df = processing_file_feature_level(df, self.GOLD_FILE_FEATURES_DICT)
                    # df = merge_matricula(df, self.MATRICULADOS_BASE, int(filename_no_extension))
                    df = formatting(df, self.GOLD_FILE_FEATURES_DICT)
                    create_parquet_from_df(df, self.GOLD_FILES_PATH_EOP, f"{filename_no_extension}_features.parquet")
                    end_time = time.time()
                    print(f"\t{filename_no_extension}: {(end_time-start_time):.3f} segs")
                except Exception as e:
                    print(f"\t{filename_no_extension}:\tError de procesamiento\t{e}")
            
            now_str = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            consolidate_parquet_files_from_folder(self.GOLD_FILES_PATH_EOP, "*_features.parquet", self.GOLD_FILES_PATH, f"{self.ALIAS_EOP}_{now_str}.parquet")
        process_end_time = datetime.datetime.now()
        print(f'Duracion: {process_end_time - process_start_time}')
        return None

    def silver_to_gold_eop_pm(self) -> None:
        process_start_time = datetime.datetime.now()
        files = get_all_files(self.SILVER_FILES_PATH_EOP, "*_doc.parquet")
        files = [file for file in files if "00_doc" not in file]
        if files:
            df = load_parquet(files)
            for file in files:
                filename_no_extension = os.path.basename(file).split("_")[0]

                if self.REFACTOR_PERIOD is not None:
                    if self.REFACTOR_PERIOD != int(filename_no_extension):
                        continue
                    else:
                        refactor_folder_file(os.path.join(self.GOLD_FILES_PATH_EOP_PERFIL_MOROSIDAD, f"{filename_no_extension}_pm.parquet"), flag_folder=0)
                try:
                    start_time = time.time()
                    df_ = df.copy()
                    df_ = df_.loc[df_["periodo"] <= int(filename_no_extension)]
                    df_ = processing_file_perfil_morosidad_eop(df_, self.PESO_MORA, self.PESO_PAGO, self.PESO_TIPO_PAGO, int(filename_no_extension))
                    create_parquet_from_df(df_, self.GOLD_FILES_PATH_EOP_PERFIL_MOROSIDAD, f"{int(filename_no_extension)}_pm.parquet")
                    end_time = time.time()
                    print(f"\t{filename_no_extension}: {(end_time-start_time):.3f} segs")
                except Exception as e:
                    print(f"\tError de procesamiento\t{e}")
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

        # Crea nuevas columnas en el DataFrame basado en el número de documentos (n_doc) y umbrales específicos.
        windowAgg.create_column_threshold(col="n_doc", threshold=5, comparison_op_str=">=")
        windowAgg.create_column_threshold(col="n_doc", threshold=15, comparison_op_str=">=")
        windowAgg.create_column_threshold(col="n_doc", threshold=30, comparison_op_str=">=")

        # Crea nuevas columnas en el DataFrame basado en el número de documentos de pago (n_doc_pago) y umbrales específicos.
        windowAgg.create_column_threshold(col="n_doc_pago", threshold=5, comparison_op_str=">=")
        windowAgg.create_column_threshold(col="n_doc_pago", threshold=15, comparison_op_str=">=")
        windowAgg.create_column_threshold(col="n_doc_pago", threshold=30, comparison_op_str=">=")

        # Genera todas las agregaciones de ventana especificadas y añade las nuevas columnas al DataFrame original.
        df_windownAgg = windowAgg.get_aggregations()

        # Obtiene la fecha y hora actual como una cadena en el formato 'YYYYMMDD_HHMMSS'.
        now_str = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")

        # Guarda el DataFrame resultante en un archivo parquet en la ruta especificada, con un nombre que incluye la fecha y hora actual.
        create_parquet_from_df(df_windownAgg, self.GOLD_FILES_PATH_WINDOWN_AGG, f"cobranzas_eop_wa_{now_str}.parquet")

        process_end_time = datetime.datetime.now()
        print(f'Duracion: {process_end_time - process_start_time}')
        return None

    def silver_to_gold_date(self) -> None:
        process_start_time = datetime.datetime.now()
        folders = get_all_folders(self.SILVER_FILES_PATH_DATE)
        if folders:
            for folder in folders:
                files = get_all_files(os.path.join(self.SILVER_FILES_PATH_DATE, folder), "*_doc.parquet")

                if self.REFACTOR_PERIOD is not None:
                    if self.REFACTOR_PERIOD != int(folder):
                        continue
                    # else:
                    #     refactor_folder_file(os.path.join(self.GOLD_FILES_PATH_DATE, f"{folder}"), flag_folder=1)

                if files:
                    print(f"\t{folder}:")
                    for file in files:
                        filename_no_extension = os.path.basename(file).split(".")[0][:-4]
                        try:
                            start_time = time.time()
                            df = load_parquet(file)
                            df = processing_file_feature_level(df, self.GOLD_FILE_FEATURES_DICT)
                            df = formatting(df, self.GOLD_FILE_FEATURES_DICT)
                            df = next_cuota_tuition(df, self.SILVER_FILES_PATH_EOP, f'{filename_no_extension.split("_")[0]}_doc.parquet')
                            df = merge_calendar(df, self.CALENDAR_DATE)
                            create_parquet_from_df(df, os.path.join(self.GOLD_FILES_PATH_DATE, folder), f"{filename_no_extension}_features.parquet")
                            end_time = time.time()
                            print(f"\t{filename_no_extension}: {(end_time-start_time):.3f} segs")
                        except Exception as e:
                            print(f"\t{filename_no_extension}:\tError de procesamiento\t{e}")
                consolidate_parquet_files_from_folder(os.path.join(self.GOLD_FILES_PATH_DATE, folder), "*_features.parquet", os.path.join(self.GOLD_FILES_PATH_DATE, "CONSOLIDADO"), f"{folder}_features.parquet")

            now_str = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            consolidate_parquet_files_from_folder(os.path.join(self.GOLD_FILES_PATH_DATE, "CONSOLIDADO"), "*_features.parquet", self.GOLD_FILES_PATH, f"{self.ALIAS_DATE}_{now_str}.parquet", parts=4)
        process_end_time = datetime.datetime.now()
        print(f'Duracion: {process_end_time - process_start_time}')
        return None

    def silver_to_gold_date_pm(self) -> None:
        process_start_time = datetime.datetime.now()
        folders = get_all_folders(self.SILVER_FILES_PATH_DATE)
        folders = [folder for folder in folders if folder[-2:] != "00"]
        if folders:
            for folder in folders:
                files = get_all_files(os.path.join(self.SILVER_FILES_PATH_DATE, folder), '*_doc.parquet')
                folders_temp = sorted([folder_temp for folder_temp in folders if int(folder_temp) < int(folder)])[-3:]
                last_3_periodo_files = [max(get_all_files(os.path.join(self.SILVER_FILES_PATH_DATE, folder), '*_doc.parquet'), default=None) for folder in folders_temp]
                last_3_periodo_files = [file for file in last_3_periodo_files if isinstance(file, str)]

                if self.REFACTOR_PERIOD is not None:
                    if self.REFACTOR_PERIOD != int(folder):
                        continue
                    else:
                        refactor_folder_file(os.path.join(self.GOLD_FILES_PATH_DATE_PERFIL_MOROSIDAD, f'{folder}'), flag_folder=1)

                if files:
                    print(f'\t{folder}:')
                    for file in files:
                        filename_no_extension = os.path.basename(file).split('.')[0][:-4]
                        try:
                            start_time = time.time()
                            df = load_parquet(file)
                            df = processing_file_perfil_morosidad_date(df, last_3_periodo_files, self.PESO_MORA, self.PESO_PAGO, self.PESO_TIPO_PAGO, filename_no_extension[-8:])
                            df = merge_calendar(df, self.CALENDAR_DATE)
                            create_parquet_from_df(df, os.path.join(self.GOLD_FILES_PATH_DATE_PERFIL_MOROSIDAD, folder), f'{filename_no_extension}_pm.parquet')                    
                            end_time = time.time()
                            print(f'\t{filename_no_extension}: {(end_time-start_time):.3f} segs')
                        except Exception as e:
                            print(f'\t{filename_no_extension}:\tError de procesamiento\t{e}')
                consolidate_parquet_files_from_folder(os.path.join(self.GOLD_FILES_PATH_DATE_PERFIL_MOROSIDAD, folder), '*_pm.parquet', os.path.join(self.GOLD_FILES_PATH_DATE_PERFIL_MOROSIDAD, 'CONSOLIDADO'), f'{folder}_pm.parquet')
            
        process_end_time = datetime.datetime.now()
        print(f'Duracion: {process_end_time - process_start_time}')
        return None

    def check_refactor(self) -> None:
        if self.REFACTOR_PERIOD is None:
            refactor_folder_file(self.SILVER_FILES_PATH)
            refactor_folder_file(self.GOLD_FILES_PATH)

    def _generate_parameters(self) -> None:

        # columnas necesarias para procesar los archivos capa bronze
        self.BRONZE_FILE_FEATURES = {
            "COD_PERIODO": "periodo",
            "#COD_ALUMNO": "cod_alumno",
            "COD_ALUMNO": "cod_alumno",
            "COD_TIPODOCUMENTO": "tipo_documento",
            "BOLETAS_PROGRAMADAS": "cod_documento",
            "ESTADO_BOLETA": "estado_documento",
            "FEC_DOCUMENTO": "fecha_documento",
            "FEC_VENCIMIENTO": "fecha_vencimiento",
            "MONTO": "monto",
            "MTO_PAGADO": "monto_pagado",
            "COD_COBRANZA": "cod_cobranza",
            "EST_COBRANZADETALLE": "est_cobranza",
            "FECHA_PAGO": "fecha_pago",
            "MATRICULA": "matricula",
            "TUITION": "tuition",
            "BOLETA_ASOCIADA": "cod_documento_asociado",
            "NRO_CUOTA": "nro_cuota",
        }

        # columnas que se devolveran del proceso BRONZE -> SILVER RAW
        self.SILVER_FILE_FEATURES_RAW = [
            "periodo",
            "cod_alumno",
            "tipo_documento",
            "cod_documento",
            "estado_documento",
            "fecha_documento",
            "fecha_vencimiento",
            "fecha_pago",
            "monto",
            "monto_pagado",
            "cod_cobranza",
            "est_cobranza",
            "flag_matricula",
            "flag_tuition",
            "cod_documento_asociado",
            "nro_cuota",
        ]

        # columnas que se devolveran del proceso SILVER -> SILVER DOC
        self.SILVER_FILE_FEATURES_DICT = {}

        # columnas que se devolveran del proceso SILVER -> GOLD FEATURE
        self.GOLD_FILE_FEATURES_DICT = {
            "periodo": "skip",
            "cod_alumno": "skip",
            "fecha_corte": "skip",
            "n_doc": 0.0,
            "n_doc_pago": 0.0,
            "r_doc_pago": 0.0,
            "n_doc_matricula": 0.0,
            "n_doc_matricula_pago": 0.0,
            "n_doc_matricula_pago_7d": 0.0,
            "r_doc_matricula_pago": 0.0,
            "r_doc_matricula_pago_7d": 0.0,
            "n_doc_tuition": 0.0,
            "n_doc_tuition_pago": 0.0,
            "n_doc_tuition_pago_7d": 0.0,
            "r_doc_tuition_pago": 0.0,
            "r_doc_tuition_pago_7d": 0.0,
            "n_doc_BV": 0.0,
            "r_doc_BV": 0.0,
            "n_doc_BV_pago": 0.0,
            "n_doc_BV_pago_7d": 0.0,
            "r_doc_BV_pago": 0.0,
            "r_doc_BV_pago_7d": 0.0,
            "n_doc_FC": 0.0,
            "r_doc_FC": 0.0,
            "n_doc_FC_pago": 0.0,
            "n_doc_FC_pago_7d": 0.0,
            "r_doc_FC_pago": 0.0,
            "r_doc_FC_pago_7d": 0.0,
            "n_doc_LC": 0.0,
            "r_doc_LC": 0.0,
            "n_doc_NC": 0.0,
            "n_doc_anulados": 0.0,
            "monto_emit": 0.0,
            "monto_pago": 0.0,
            "monto_pago_7d": 0.0,
            "r_monto_pago": 0.0,
            "r_monto_pago_7d": 0.0,
            "monto_emit_matricula": 0.0,
            "monto_matricula_pago": 0.0,
            "monto_matricula_pago_7d": 0.0,
            "r_monto_matricula_pago": 0.0,
            "r_monto_matricula_pago_7d": 0.0,
            "monto_emit_tuition": 0.0,
            "monto_tuition_pago": 0.0,
            "monto_tuition_pago_7d": 0.0,
            "r_monto_tuition_pago": 0.0,
            "r_monto_tuition_pago_7d": 0.0,
            "monto_emit_BV": 0.0,
            "monto_emit_FC": 0.0,
            "monto_emit_LC": 0.0,
            "monto_emit_NC": 0.0,
            "flag_cuota_1": 0.0,
            "flag_cuota_2": 0.0,
            "flag_cuota_3": 0.0,
            "flag_cuota_4": 0.0,
            "flag_cuota_5": 0.0,
            "flag_cuota_6": 0.0,
            "fecha_pago_cuota_1": np.datetime64("NaT"),
            "fecha_pago_cuota_2": np.datetime64("NaT"),
            "fecha_pago_cuota_3": np.datetime64("NaT"),
            "fecha_pago_cuota_4": np.datetime64("NaT"),
            "fecha_pago_cuota_5": np.datetime64("NaT"),
            "fecha_pago_cuota_6": np.datetime64("NaT"),
            "dias_pago_cuota_1": np.nan,
            "dias_pago_cuota_2": np.nan,
            "dias_pago_cuota_3": np.nan,
            "dias_pago_cuota_4": np.nan,
            "dias_pago_cuota_5": np.nan,
            "dias_pago_cuota_6": np.nan,
            "dias_mora_tuition_acum": 0.0,
            "dias_mora_tuition_avg": 0.0,
            "avg_dias_pago_tuition": np.nan,
            "avg_dias_pago_matricula": np.nan,
        }

        self.GOLD_FILE_AGG_FEATURES_LIST = [
            ("n_doc_mayor_igual_5", [2, 3, 4], "count"),
            ("n_doc_mayor_igual_15", [2, 3, 4], "count"),
            ("n_doc_mayor_igual_30", [2, 3, 4], "count"),
            ("n_doc_pago_mayor_igual_5", [2, 3, 4], "count"),
            ("n_doc_pago_mayor_igual_15", [2, 3, 4], "count"),
            ("n_doc_pago_mayor_igual_30", [2, 3, 4], "count"),
            ("n_doc", [2, 3, 4], "sum"),
            ("n_doc_tuition", [2, 3, 4], "sum"),
            ("n_doc_matricula", [2, 3, 4], "sum"),
            ("n_doc_pago", [2, 3, 4], "sum"),
            ("n_doc_tuition_pago", [2, 3, 4], "sum"),
            ("n_doc_matricula_pago", [2, 3, 4], "sum"),
            ("r_doc_pago", [2, 3, 4], "mean"),
            ("r_doc_matricula_pago", [2, 3, 4], "mean"),
            ("r_doc_tuition_pago", [2, 3, 4], "mean"),
            ("r_monto_pago", [2, 3, 4], "mean"),
            ("r_monto_matricula_pago", [2, 3, 4], "mean"),
            ("r_monto_tuition_pago", [2, 3, 4], "mean"),
            ("monto_pago", [2, 3, 4], "mean"),
            ("monto_matricula_pago", [2, 3, 4], "mean"),
            ("monto_tuition_pago", [2, 3, 4], "mean"),
            ("monto_emit", [2, 3, 4], "mean"),
            ("monto_emit_matricula", [2, 3, 4], "mean"),
            ("monto_emit_tuition", [2, 3, 4], "mean"),
        ]

        self.PESO_MORA = {'NO_MORA': 0, 'MOROSO_1': 1, 'MOROSO_2': 2, 'MOROSO_3': 3, 'MOROSO_4': 4}
        self.PESO_PAGO = {'PAGO_0': 0, 'PAGO_1': 1, 'PAGO_2': 2, 'PAGO_3': 3, 'PAGO_4': 4}
        self.PESO_TIPO_PAGO = {'REG': 0, 'EXT': 1}

        return None
