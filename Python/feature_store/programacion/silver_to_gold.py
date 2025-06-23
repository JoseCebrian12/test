import pandas as pd


def processing_file_feature_level(
    df: pd.DataFrame, features_dict: dict = None
) -> pd.DataFrame:
    """
    Calculo de los features que se encuentran en la capa gold.
    Estos calculos se encuentran a nivel de PERIODO-COD_ALUMNO-FECHA_CORTE.
    """

    # Hacer una copia del DataFrame original para no modificar el original
    df_ = df.copy()

    df_ = df_.sort_values(
        by=[
            "periodo",
            "cod_curso",
            "cod_grupo",
            "modalidad_estudio",
            "fec_inicio_sesion",
        ],
        ignore_index=True,
    )

    df_final = df_.groupby(
        ["periodo", "fecha_corte", "cod_curso", "cod_seccion"],
        as_index=False,
        dropna=False,
    ).agg(
        cant_sesiones=("fec_inicio_sesion", "nunique"),
    )

    # Si se especifican características finales, quedarse solo con esas características
    if features_dict is not None:
        df_final = df_final[features_dict.keys()]

    return df_final
