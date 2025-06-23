import pandas as pd

def processing_file_feature_level(df: pd.DataFrame, features_dict: dict = None) -> pd.DataFrame:
    """
    Calculo de los features que se encuentran en la capa gold.
    Estos calculos se encuentran a nivel de PERIODO-COD_ALUMNO-FECHA_CORTE.
    """

    df_ = df.copy()

    df_.loc[:, "flag_obligatorio"] = (df_["tipo_curso"] == "OBLIGATORIO").astype(int)
    df_.loc[:, "flag_electivo"] = (df_["tipo_curso"] == "ELECTIVO").astype(int)
    df_.loc[:, "flag_nocred"] = (df_["tipo_curso"] == "NO_CRED").astype(int)

    df_.loc[:, "creds_obligatorio"] = df_["flag_obligatorio"] * df_["cant_creditos"]
    df_.loc[:, "creds_electivo"] = df_["flag_electivo"] * df_["cant_creditos"]
    df_.loc[:, "creds_nocred"] = df_["flag_nocred"] * df_["cant_creditos"]

    df_ = df_.groupby(["periodo", "cod_alumno", "fecha_corte"], as_index=False, dropna=False).agg(
        n_curmatr = ("cod_curso", "count"),

        n_curmatr_oblig = ("flag_obligatorio", "sum"),
        n_curmatr_elect = ("flag_electivo", "sum"),
        n_curmatr_nocred = ("flag_nocred", "sum"),

        avg_ciclo = ("ciclo_curso", "mean"),
        max_ciclo = ("ciclo_curso", "max"),
        min_ciclo = ("ciclo_curso", "min"),

        sum_cred_curmatr = ("cant_creditos", "sum"),
        sum_cred_oblig_curmatr = ("creds_obligatorio", "sum"),
        sum_cred_elect_curmatr = ("creds_electivo", "sum"),
        sum_cred_nocred_curmatr = ("creds_nocred", "sum"),
    )

    # creds acumulados
    df_ = df_.sort_values(by=["cod_alumno", "periodo", "fecha_corte"], ignore_index=True)
    df_.loc[:, "cred_acum_hist"] = df_.groupby(["cod_alumno"])["sum_cred_curmatr"].transform("cumsum")

    # ratios
    df_.loc[:, "r_curmatr_oblg"] = (df_["n_curmatr_oblig"] / df_["n_curmatr"])
    df_.loc[:, "r_curmatr_elect"] = (df_["n_curmatr_elect"] / df_["n_curmatr"])
    df_.loc[:, "r_curmatr_nocred"] = (df_["n_curmatr_nocred"] / df_["n_curmatr"])

    df_.loc[:, "r_cred_curmatr_oblg"] = (df_["sum_cred_oblig_curmatr"] / df_["sum_cred_curmatr"])
    df_.loc[:, "r_cred_curmatr_elect"] = (df_["sum_cred_elect_curmatr"] / df_["sum_cred_curmatr"])

    if features_dict is not None:
        df_ = df_[features_dict.keys()]

    return df_
