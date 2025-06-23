import datetime
import os
import warnings

import numpy as np
import pandas as pd

from .general.funcs import (
    clean_number,
    clean_text,
    create_parquet_from_df,
    get_all_files,
    load_csv,
)

warnings.filterwarnings("ignore")

# fmt: off
class Assets:
    def __init__(self) -> None:
        """
        Clase para generar datasets assets a partir de archivos CSV.
        
        Atributos:
        BASE_FILES (str): Ruta de la carpeta donde se encuentran los archivos CSV de matrícula.
        EOP_CALENDAR_FILE (str): Ruta del archivo parquet para el calendario de corte de fin de período (EOP).
        DATE_CALENDAR_FILE (str): Ruta del archivo parquet para el calendario a fechas de corte.
        ASSETS_FOLDER (str): Ruta de la carpeta donde se guardarán los activos generados.
        """

        # rutas importantes
        self.BASE_FILES = "00_data/00_bronze/00_matricula" # el calendario se crea en funcion de los arhivos presentes en la capa bronze de matricula
        self.EOP_CALENDAR_FILE = f"00_data/03_assets/calendar_eop.parquet"
        self.DATE_CALENDAR_FILE = f"00_data/03_assets/calendar_date.parquet"
        self.ASSETS_FOLDER = f"00_data/03_assets/"

        self._generate_parameters()
        
        return None

    def calendar_eop(self) -> None:
        """
        Genera un calendario de corte de fin de período (EOP) a partir de los archivos CSV de matrícula.
        
        El calendario contiene las siguientes columnas:
        - periodo: Código del período académico
        - fecha_fin_periodo: Último día del período académico
        - fecha_corte: Fecha de corte, que es 15 días después del último día del período
        
        El calendario se guarda en un archivo parquet en la ruta especificada por ASSETS_FOLDER.
        """
        files = get_all_files(self.BASE_FILES, "*.csv")
        if files:
            file_name_list = [os.path.basename(f).split(".")[0] for f in files]
            calendar_eop_df = pd.DataFrame({"periodo": file_name_list})

            calendar_eop_df.loc[:, "year"] = calendar_eop_df["periodo"].str[:4]
            calendar_eop_df.loc[:, "semester"] = calendar_eop_df["periodo"].str[-2:]
            calendar_eop_df.loc[:, "month"] = calendar_eop_df["semester"].map(self.DICT_SEMESTER_LAST_MONTH)
            calendar_eop_df.loc[:, "fecha_fin_periodo"] = calendar_eop_df["year"] + calendar_eop_df["month"] + "31"

            # calendar_eop_df.loc[:, "fecha_fin_periodo"] = calendar_eop_df["fecha_fin_periodo"].apply(lambda x: datetime.datetime.strptime(x, "%Y%m%d"))
            calendar_eop_df.loc[:, "fecha_fin_periodo"] = pd.to_datetime(calendar_eop_df["fecha_fin_periodo"], errors="coerce", format="%Y%m%d")
            calendar_eop_df.loc[:, "fecha_corte"] = calendar_eop_df.loc[:, "fecha_fin_periodo"] + datetime.timedelta(days=15)

            calendar_eop_df = calendar_eop_df[["periodo", "fecha_fin_periodo", "fecha_corte"]]
            calendar_eop_df.loc[:, "periodo"] = calendar_eop_df["periodo"].astype(int)

            create_parquet_from_df(calendar_eop_df, self.ASSETS_FOLDER, "calendar_eop.parquet")

        return None
    
    def calendar_date(self) -> None:
        """
        Genera un calendario de fechas a partir de los archivos CSV de matrícula.
        
        El calendario contiene las siguientes columnas:
        - periodo: Código del período académico
        - fecha_corte: Fecha de corte
        - semana_semestre: Número de semana dentro del semestre académico
        
        El calendario se guarda en un archivo parquet en la ruta especificada por ASSETS_FOLDER.
        """
        files = get_all_files(self.BASE_FILES, "*.csv")
        if files:
            file_name_list = [os.path.basename(f).split(".")[0] for f in files]
            calendar_date_df = pd.DataFrame({"periodo": file_name_list})
            calendar_date_df.loc[:, "year"] = calendar_date_df["periodo"].str[:4]
            calendar_date_df.loc[:, "semester"] = calendar_date_df["periodo"].str[-2:]

            calendar_date_df.loc[:, "min_date"] = calendar_date_df["semester"].map(self.DICT_SEMESTER_MIN_DATE)
            calendar_date_df.loc[:, "min_date"] = calendar_date_df["year"] + calendar_date_df["min_date"]
            calendar_date_df["min_date"] = pd.to_datetime(calendar_date_df["min_date"], errors="coerce", format="%Y%m%d")

            calendar_date_df.loc[:, "max_date"] = calendar_date_df["semester"].map(self.DICT_SEMESTER_MAX_DATE)
            calendar_date_df.loc[:, "max_date"] = calendar_date_df["year"] + calendar_date_df["max_date"]
            calendar_date_df["max_date"] = pd.to_datetime(calendar_date_df["max_date"], errors="coerce", format="%Y%m%d")

            calendar_date_df["max_date"] = np.where(
                calendar_date_df["semester"] == "02",
                calendar_date_df["max_date"] + pd.DateOffset(years=1),
                calendar_date_df["max_date"],
            )

            calendar_date_df.loc[:, "min_monday"] = calendar_date_df["min_date"] + pd.DateOffset(weekday=0)

            calendar_date_df.loc[:, "month"] = calendar_date_df["semester"].map(self.DICT_SEMESTER_LAST_MONTH)
            calendar_date_df.loc[:, "fecha_fin_periodo"] = calendar_date_df["year"] + calendar_date_df["month"] + "31"
            calendar_date_df.loc[:, "fecha_fin_periodo"] = pd.to_datetime(calendar_date_df["fecha_fin_periodo"], errors="coerce", format="%Y%m%d")

            result_list = []
            for ix, row in calendar_date_df.iterrows():
                possible_dates = pd.date_range(row["min_monday"], row["max_date"], freq="7d")
                possible_dates = [date.strftime("%Y-%m-%d") for date in possible_dates]
                df_ = pd.DataFrame.from_dict({"fecha_corte": possible_dates})
                df_.loc[:, "periodo"] = row["periodo"]
                df_["periodo"] = df_["periodo"].astype(int)
                df_["fecha_fin_periodo"] = row["fecha_fin_periodo"]
                df_ = self._get_near_date(df_, 5)  
                df_ = self._get_near_date(df_, 10)
                df_ = self._get_near_date(df_, 15)
                df_ = self._get_near_date(df_, 20)
                df_ = self._get_near_date(df_, 23)
                df_ = self._get_near_date(df_, 25)
                df_ = self._get_near_date(df_, 30)
                df_ = self._semana_semestre(df_)
                result_list.append(df_)

            result_cal = pd.concat(result_list, axis=0).reset_index(drop=True)
            create_parquet_from_df(result_cal, self.ASSETS_FOLDER, "calendar_date.parquet")

        return None

    
    def _generate_parameters(self) -> None:
        """
        Genera los parámetros necesarios para calcular las fechas de corte y las semanas del semestre.
        
        Los parámetros incluyen:
        - DICT_SEMESTER_LAST_MONTH: Diccionario que mapea el código de semestre al último mes del semestre.
        - DICT_SEMESTER_MIN_DATE: Diccionario que mapea el código de semestre a la fecha mínima del semestre.
        - DICT_SEMESTER_MAX_DATE: Diccionario que mapea el código de semestre a la fecha máxima del semestre.
        - INICIO_FIN_SEMESTRE: Diccionario que mapea el código de semestre a un rango de fechas que representa el inicio y fin del semestre.
        """
        
        self.DICT_SEMESTER_LAST_MONTH = {
            "00": "03",
            "01": "07",
            "02": "12",
        }

        self.DICT_SEMESTER_MIN_DATE = {
            "00": "0101", # 1ero de enero
            "01": "0301", # 1ero de marzo
            "02": "0701", # 1ero de julio
        }

        self.DICT_SEMESTER_MAX_DATE = {
            "00": "0331", # 31 marzo
            "01": "0820", # 20 agosto
            "02": "0310", # 10 marzo
        }
        
        self.INICIO_FIN_SEMESTRE = {
            "00": "-01-07_-02-25", # la primera semana de enero - la semana del 25 de febrero
            "01": "-03-10_-07-07", 
            "02": "-08-10_-12-07",
        }

        return None

    def _get_near_date(self, df: pd.DataFrame, day_value: int) -> pd.DataFrame:
        """
        Función auxiliar para calcular la fecha más cercana a un día específico dentro de cada mes.
        
        Parámetros:
        df (pd.DataFrame): DataFrame que contiene las fechas de corte.
        day_value (int): Día del mes para el que se busca la fecha más cercana.
        
        Retorna:
        pd.DataFrame: DataFrame con una nueva columna que indica si la fecha de corte está cerca del día especificado.
        """
        df_ = df.copy().reset_index(drop=True)
        
        # Convertir fecha_corte a datetime si aún no lo es
        df_["fecha_corte"] = pd.to_datetime(df_["fecha_corte"], errors='coerce', yearfirst=True, format="%Y-%m-%d")
        
        # Extraer año y mes de fecha_corte
        df_["year_month"] = df_["fecha_corte"].dt.to_period('M')

        # Intentar crear la fecha con el day_value y ajustar a fin de mes si es necesario
        df_["calculated_day"] = pd.to_datetime(df_["year_month"].astype(str) + '-' + str(day_value), errors='coerce')
        # Rellenar NaT con el último día del mes
        df_["calculated_day"] = df_["calculated_day"].fillna(df_["year_month"].dt.end_time.dt.to_period('D').dt.to_timestamp())

        # Calcular la distancia en días
        df_["distancia_dias"] = np.abs((df_["calculated_day"] - df_["fecha_corte"]).dt.days)
        
        # Ordenar y asignar flags
        df_ = df_.sort_values(by=["fecha_corte"], ignore_index=True)
        df_[f"flag_d{day_value}"] = df_.groupby(["year_month"])["distancia_dias"].transform(lambda x: (x == x.min()).astype(int))

        # Limpieza de columnas innecesarias
        df_.drop(["year_month", "calculated_day", "distancia_dias"], axis=1, inplace=True)

        return df_
    
    def _semana_semestre(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Función auxiliar para calcular el número de semana dentro del semestre académico para cada fecha de corte.
        
        Parámetros:
        df (pd.DataFrame): DataFrame que contiene las fechas de corte.
        
        Retorna:
        pd.DataFrame: DataFrame con una nueva columna que indica el número de semana dentro del semestre.
        """
        df_ = df.copy().reset_index(drop=True)
        cols = list(df_.columns)

        df_.loc[:, "year"] = df_["periodo"].astype(str).str[:4]
        df_.loc[:, "semester"] = df_["periodo"].astype(str).str[-2:]

        df_.loc[:, "fecha_min"] = df_["semester"].map(self.INICIO_FIN_SEMESTRE).str.split("_").str[0]
        df_.loc[:, "fecha_min"] = df_["year"] + df_["fecha_min"]
        df_["fecha_min"] = pd.to_datetime(df_["fecha_min"], yearfirst=True, format="%Y-%m-%d")

        df_.loc[:, "fecha_max"] = df_["semester"].map(self.INICIO_FIN_SEMESTRE).str.split("_").str[1]
        df_.loc[:, "fecha_max"] = df_["year"] + df_["fecha_max"]
        df_["fecha_max"] = pd.to_datetime(df_["fecha_max"], yearfirst=True, format="%Y-%m-%d")

        df_.loc[:, "flag_between_fechas"] = np.where((df_["fecha_corte"] >= df_["fecha_min"]) & (df_["fecha_corte"] <= df_["fecha_max"]), 1, 0)
        df_.loc[:, "semana_semestre"] = np.ceil((df_["fecha_corte"] - df_["fecha_min"]).dt.days / 7)
        df_.loc[:, "semana_semestre"] = np.where(df_["flag_between_fechas"] == 0 , 0, df_["semana_semestre"])
        
        df_["semana_anio"] = df_["fecha_corte"].dt.isocalendar().week

        df_["semana_engagement"] = (df_["semana_semestre"] == 1).astype(int)
        df_["semana_engagement"] = df_.groupby(["periodo"])["semana_engagement"].cumsum()
        df_["semana_engagement"] = df_.groupby(["periodo"])["semana_engagement"].cumsum()

        return df_[cols + ["semana_semestre", "semana_anio", "semana_engagement"]]
    
    def generate_enrollment_base(self) -> None:
        """
        Genera una base de datos de alumnos matriculados a partir de los archivos CSV de matrícula.
        
        La base de datos contiene las siguientes columnas:
        - periodo: Código del período académico
        - cod_alumno: Código del alumno
        
        La base de datos se guarda en un archivo parquet en la ruta especificada por ASSETS_FOLDER.
        """
        files = get_all_files(self.BASE_FILES, "*.csv")
        if files:
            result = []
            for file in files:
                df = load_csv(file)
                df.rename({"#COD_PERIODO_MATRICULA": "periodo",
                           "COD_PERIODO_MATRICULA": "periodo",
                           "COD_ALUMNO": "cod_alumno"},
                           axis=1,
                           inplace=True)
                df = df[["periodo", "cod_alumno"]]

                df["periodo"] = df["periodo"].apply(clean_number, format="int")
                df["cod_alumno"] = df["cod_alumno"].apply(clean_text)

                df = df.drop_duplicates().dropna()
                result.append(df)

            df = pd.concat(result, axis=0)
            create_parquet_from_df(df, self.ASSETS_FOLDER, "matriculados_eop.parquet")
        
        return None
