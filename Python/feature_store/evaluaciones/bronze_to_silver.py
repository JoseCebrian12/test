import pandas as pd

from ..general.funcs import clean_number, clean_text

# fmt: off
def processing_bronze_file_raw_level(df: pd.DataFrame, final_cols: list = None) -> pd.DataFrame:
    df_ = df.copy()

    for col_name in df_.columns:
        df_[col_name] = df_[col_name].apply(clean_text)

    df_ = df_.loc[((df_["periodo"].notna()) & (df_["cod_alumno"].notna()))]

    # Limpieza espacios en blanco strings
    df_obj = df_.select_dtypes("object")
    df[df_obj.columns] = df_obj.apply(lambda x: x.str.strip())

    # limpieza de numeros
    df_["periodo"] = df_["periodo"].apply(clean_number, format="int")
    df_["num_prueba"] = df_["num_prueba"].apply(lambda x: clean_number(x, "int"))
    df_["nota"] = df_["nota"].apply(lambda x: clean_number(x, "float"))

    # limpieza de fechas
    df_["fecha_creacion"] = pd.to_datetime(df_["fecha_creacion"], errors="coerce")
    df_["fecha_creacion"] = df_["fecha_creacion"].dt.normalize()

    # eliminacion de potenciales duplicados
    df_ = df_.sort_values(by=["periodo", "cod_alumno", "cod_curso", "cod_tipo_prueba", "num_prueba", "nota"], ascending=[True, True, True, True, True, False])
    df_ = df_.drop_duplicates(subset=["periodo", "cod_alumno", "cod_curso", "cod_tipo_prueba", "num_prueba"], keep="first")
    
    # solo nos quedamos con las columnas necesarias
    if final_cols is not None:
        df_ = df_[final_cols]

    return df_

def processing_silver_file(df: pd.DataFrame) -> pd.DataFrame:

    df_ = df.copy().reset_index(drop=True)

    # verificamos que no se pase la fecha de la evaluacion sobre la fecha de corte
    df_ = df_.loc[(df_["fecha_creacion"] <= df_["fecha_corte"])]

    return df_
