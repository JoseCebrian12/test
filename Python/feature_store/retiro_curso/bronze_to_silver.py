import pandas as pd

from ..general.funcs import clean_number, clean_text

# fmt: off
def processing_bronze_file_raw_level(
    df: pd.DataFrame,
    final_cols: list = None,
) -> pd.DataFrame:

    df_ = df.copy().reset_index(drop=True)

    # Eliminamos las dos ultimas columnas de FEC_CARGA	DES_USR_CARGA
    # df_ = df_.drop(columns=["FEC_CARGA", "DES_USR_CARGA"])

    # Limpieza espacios en blanco strings
    df__obj = df_.select_dtypes("object")
    df_[df__obj.columns] = df__obj.apply(lambda x: x.str.strip())

    # Limpiar texto
    # fmt: off
    df_["cod_usr_creacion"] = df_["cod_usr_creacion"].apply(clean_text)
    df_["cod_usr_modificacion"] = df_["cod_usr_modificacion"].apply(clean_text)

    # Asignar tipo de dato Datetime
    df_["fec_solicitud"] = pd.to_datetime(df_["fec_solicitud"], errors="coerce").dt.normalize()
    df_["fec_retiro"] = pd.to_datetime(df_["fec_retiro"], errors="coerce").dt.normalize()
    df_["fec_creacion"] = pd.to_datetime(df_["fec_creacion"], errors="coerce").dt.normalize()
    df_["fec_modificacion"] = pd.to_datetime(df_["fec_modificacion"], errors="coerce").dt.normalize()

    # Limpiar y casteo valores numericos
    df_["periodo"] = df_["periodo"].apply(clean_number, format='int')

    # solo nos quedamos con las columnas necesarias
    if final_cols is not None:
        df_ = df_[final_cols]

    return df_
