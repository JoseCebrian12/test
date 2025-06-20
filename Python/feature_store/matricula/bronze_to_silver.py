import pandas as pd

from ..general.funcs import clean_number, clean_text

# fmt: off
def processing_bronze_file_raw_level(df: pd.DataFrame, final_cols: list = None) -> pd.DataFrame:
    df_ = df.copy()

    for col_name in df_.columns:
        df_.loc[:, col_name] = df_[col_name].apply(clean_text)

    df_ = df_.loc[((df_["periodo"].notna()) & (df_["cod_alumno"].notna()))]

    # eliminacion de duplicidad por periodo
    # son casos puntuales.
    df_.loc[:, "n_nulls"] = df_.isna().sum(axis=1)
    df_ = df_.sort_values(["periodo", "cod_alumno", "n_nulls"], ascending=[True, True, True]) # nos quedamos con el primer registro que tenga menos nulos
    df_ = df_.drop_duplicates(subset=["periodo", "cod_alumno"], ignore_index=True)

    # limpieza de numeros
    df_.loc[:,"periodo"] = df_["periodo"].apply(clean_number, format="int")
    df_.loc[:,"ciclo_aprox"] = df_["ciclo_aprox"].apply(clean_number, format="float")
    df_.loc[:,"cod_facultad"] = df_["cod_facultad"].apply(clean_number, format="int")
    df_.loc[:,"edad"] = df_["edad"].apply(clean_number, format="float")
    df_.loc[:,"porcentaje_beca"] = df_["porcentaje_beca"].apply(clean_number, format="float")
    df_.loc[:,"cant_creditos"] = df_["cant_creditos"].apply(clean_number, format="float")
    df_.loc[:,"ponderado_actual"] = df_["ponderado_actual"].apply(clean_number, format="float")
    df_.loc[:,"ponderado_acumulado"] = df_["ponderado_acumulado"].apply(clean_number, format="float")
    df_.loc[:,"cant_ciclos_verano"] = df_["cant_ciclos_verano"].apply(clean_number, format="float")

    # flags
    df_.loc[:,"flag_tercio_per_ant"] = df_["flag_tercio_per_ant"].str.upper().map({"SI": 1, "NO": 0})
    df_.loc[:,"flag_quinto_per_ant"] = df_["flag_quinto_per_ant"].str.upper().map({"SI": 1, "NO": 0})
    df_.loc[:,"flag_decimo_per_ant"] = df_["flag_decimo_per_ant"].str.upper().map({"SI": 1, "NO": 0})
    df_.loc[:,"flag_riesgo"] = df_["flag_riesgo"].str.upper().map({"SI": 1, "NO": 0})

    # fechas
    df_.loc[:,"fecha_nacimiento"] = pd.to_datetime(df_["fecha_nacimiento"], errors="coerce")
    df_.loc[:,"fecha_ingreso"] = pd.to_datetime(df_["fecha_ingreso"], errors="coerce")
    df_.loc[:,"fecha_matricula"] = pd.to_datetime(df_["fecha_matricula"], errors="coerce")

    # solo nos quedamos con las columnas necesarias
    if final_cols is not None:
        df_ = df_[final_cols]

    return df_
