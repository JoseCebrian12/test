import numpy as np
import pandas as pd

from ..general.funcs import clean_number, clean_text


# fmt: off
def processing_bronze_file_raw_level(df: pd.DataFrame, final_cols: list = None) -> pd.DataFrame:
    df_ = df.copy()

    for col_name in df_.columns:
        df_.loc[:, col_name] = df_[col_name].apply(clean_text)

    df_ = df_.loc[((df_['periodo'].notna()) & (df_['cod_alumno'].notna()))]

    # procesamos todos los campos como si fueran texto
    # esto para eliminar registros no deseados en los campos periodo y cod_alumno
    cols = [col for col in df_.columns]
    for col_name in cols:
        df_[col_name] = df_[col_name].apply(clean_text)

    # limpieza de numeros
    df_.loc[:, 'periodo'] = df_['periodo'].apply(clean_number, format='int')
    df_.loc[:, 'cant_creditos'] = df_['cant_creditos'].apply(clean_number, format='float')
    df_.loc[:, 'ciclo_curso'] = df_['ciclo_curso'].apply(clean_number, format='int')

    # Eliminando los registros que no tengan codigo de curso: cod_curso
    df_ = df_.dropna(subset=['cod_curso'])

    # cambiamos aquellos cursos que tienen creditos = 0
    b1 = df_['cant_creditos'] == 0
    df_.loc[:, 'tipo_curso'] = np.where(b1, 'NO_CRED', df_['tipo_curso'])

    # eliminacion de potenciales duplicados
    df_ = df_.sort_values(by=['periodo', 'cod_alumno', 'cod_curso'])
    df_ = df_.drop_duplicates(subset=['periodo', 'cod_alumno', 'cod_curso'], keep='first', ignore_index=True)

    # solo nos quedamos con las columnas necesarias
    if final_cols is not None:
        df_ = df_[final_cols]

    return df_
