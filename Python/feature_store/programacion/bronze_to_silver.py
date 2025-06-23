import warnings

import numpy as np
import pandas as pd

from ..general.funcs import clean_number, clean_text

warnings.filterwarnings("ignore")

# fmt: off
def processing_bronze_file_raw_level(
    df: pd.DataFrame,
    final_cols: list = None,
) -> pd.DataFrame:

    # Hacer una copia del DataFrame original para no modificar el original
    df_ = df.copy()

    # Aplicar la función clean_text a cada columna del DataFrame
    for col_name in df_.columns:
        df_[col_name] = df_[col_name].apply(clean_text)

    # Eliminando columnas innecesarias
    df_ = df_.drop(
        columns=[
            "cod_tipo_sesion",
            "des_tipo_sesion",
            "cod_aula",
        ]
    )

    # Reemplazar valores vacíos o espacios en blanco con NaN y eliminar duplicados
    df_ = df_.replace(to_replace=["", " "], value=np.nan)
    df_ = df_.drop_duplicates(ignore_index=True)

    # Limpiar y convertir la columna "periodo tipo int usando clean_number
    df_["periodo"] = df_["periodo"].apply(clean_number, format="int")
    df_["fec_inicio_sesion"] = pd.to_datetime(df_["fec_inicio_sesion"], errors="coerce").dt.normalize()

    # Si se especifican columnas finales, quedarse solo con esas columnas
    if final_cols is not None:
        df_ = df_[final_cols]

    return df_


def processing_silver_file(df: pd.DataFrame) -> pd.DataFrame:

    df_ = df.copy().reset_index(drop=True)

    # verificamos que no se pase la fecha de la evaluacion sobre la fecha de corte
    df_ = df_.loc[(df_["fec_inicio_sesion"] <= df_["fecha_corte"])]

    return df_
