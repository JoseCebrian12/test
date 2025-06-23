import pandas as pd


# fmt: off
def processing_file_feature_level(
    df_inasistencia: pd.DataFrame,
    df_programacion: pd.DataFrame,
    features_dict: dict = None,
) -> pd.DataFrame:
    """
    Calculo de los features que se encuentran en la capa gold.
    Estos cálculos se encuentran a nivel de PERIODO-COD_ALUMNO-FECHA_CORTE.
    """

    # Ordena el DataFrame de inasistencias por las columnas especificadas para facilitar el procesamiento posterior.
    df_inasistencia = df_inasistencia.sort_values(
        by=["periodo", "cod_alumno", "cod_curso", "fec_inicio_sesion"],
        ascending=True,
        ignore_index=True,
    )

    # Calcula un conteo acumulativo de inasistencias para cada alumno y curso.
    columnas_conteo = ["periodo", "cod_alumno", "cod_curso", "fecha_corte"]
    df_inasistencia["conteo_inasistencias"] = (df_inasistencia.groupby(columnas_conteo + ["fec_inicio_sesion"]).cumcount() == 0).astype(int)
    df_inasistencia["conteo_inasistencias"] = df_inasistencia.groupby(columnas_conteo)["conteo_inasistencias"].cumsum()

    # Determina el máximo de inasistencias por alumno y período.
    max_inasistencias_per_curso = df_inasistencia.sort_values(by=["periodo", "cod_alumno", "conteo_inasistencias"], ascending=[True, True, True]).copy()
    max_inasistencias_per_curso = max_inasistencias_per_curso.drop_duplicates(subset=["periodo", "cod_alumno"], keep="last", ignore_index=True)
    max_inasistencias_per_curso = max_inasistencias_per_curso[["periodo", "cod_alumno", "conteo_inasistencias"]]
    max_inasistencias_per_curso = max_inasistencias_per_curso.rename(columns={"conteo_inasistencias": "max_inasistencias_per_curso"})

    # Determina la fecha más reciente de inasistencia por alumno y período.
    max_fecha_inasistencia = df_inasistencia.sort_values(by=["periodo", "cod_alumno", "fec_inicio_sesion"], ascending=[True, True, True]).copy()
    max_fecha_inasistencia = max_fecha_inasistencia.drop_duplicates(subset=["periodo", "cod_alumno"], keep="last", ignore_index=True)
    max_fecha_inasistencia = max_fecha_inasistencia[["periodo", "cod_alumno", "fec_inicio_sesion"]]
    max_fecha_inasistencia = max_fecha_inasistencia.rename(columns={"fec_inicio_sesion": "max_fecha_inasistencia"})

    # Agrupa por periodo, alumno y curso para contar las inasistencias únicas y calcular las sesiones totales.
    df_inasistencia_gb = df_inasistencia.groupby(
        ["periodo", "cod_alumno", "cod_curso", "cod_seccion"],
        as_index=False,
        dropna=False,
    ).agg(
        cant_inasistencias=("fec_inicio_sesion", "count"),
    )

    # Combina los datos de inasistencias con los de programación para completar la información de sesiones.
    df_ = pd.merge(
        df_inasistencia_gb,
        df_programacion,
        on=["periodo", "cod_curso", "cod_seccion"],
        how="left",
    )

    # Filtra y reagrupa por periodo, alumno y fecha de corte para sumar inasistencias y sesiones, calculando las asistencias.
    df_ = df_[["periodo", "cod_alumno", "fecha_corte", "cod_curso", "cant_inasistencias", "cant_sesiones"]]
    df_ = df_.groupby(
        ["periodo", "cod_alumno", "fecha_corte"], as_index=False
    ).agg(
        cant_inasistencias=("cant_inasistencias", "sum"),
        cant_sesiones=("cant_sesiones", "sum"),
    )

    # Calcula el número de asistencias y el ratio de inasistencias sobre el total de sesiones.
    df_["cant_asistencias"] = df_["cant_sesiones"] - df_["cant_inasistencias"]
    df_["r_inasistencias"] = (df_["cant_inasistencias"] / df_["cant_sesiones"]).round(2)

    # Combina con los DataFrames de máximo de inasistencias y fecha más reciente de inasistencia.
    df_ = pd.merge(
        df_,
        max_inasistencias_per_curso,
        on=["periodo", "cod_alumno"],
        how="left",
    )
    df_final = pd.merge(
        df_,
        max_fecha_inasistencia,
        on=["periodo", "cod_alumno"],
        how="left",
    )

    # Filtra las columnas finales según un diccionario de características, si se proporciona.
    if features_dict is not None:
        df_final = df_[features_dict.keys()]

    return df_final
