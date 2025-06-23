import pandas as pd
import numpy as np

import warnings

warnings.filterwarnings("ignore")


def processing_file_features_curs_matr_date(
    df_retiro_curso: pd.DataFrame,
    df_base_cursos_matricula: pd.DataFrame,
    df_calendar: pd.DataFrame,
) -> pd.DataFrame:
    """
    Calculo de los features que se encuentran en la capa gold.
    Estos calculos se encuentran a nivel de PERIODO-COD_ALUMNO-FECHA_CORTE.
    """

    # filtramos los retiros con motivo R
    # sin importar la fecha son los retiros que importan
    df_retiro_curso = df_retiro_curso[df_retiro_curso["cod_estado_retiro"] == "R"]

    df_retiro_curso["flg_retiro_curso"] = (
        df_retiro_curso["cod_estado_retiro"] == "R"
    ).astype(int)

    # validamos que un estudiante no se haya retirado de un mismo curso 2 veces
    df_retiro_curso = df_retiro_curso.sort_values(
        by=["periodo", "cod_alumno", "cod_curso", "fec_retiro"],
        ascending=[True, True, True, True],
    )
    df_retiro_curso = df_retiro_curso.drop_duplicates(
        subset=["periodo", "cod_alumno", "cod_curso"], keep="first"
    )

    # columnas que no interesan
    to_drop = [
        "cod_corporativo",
        "cod_linea_negocio",
        "cod_modalidad_estudio",
        "cod_producto",
        "cod_retiro_curso",
        "fec_solicitud",
        "cod_estado_retiro",
        "cod_usr_creacion",
        "fec_creacion",
        "fec_modificacion",
        "cod_usr_modificacion",
    ]

    df_retiro_curso = df_retiro_curso.drop(to_drop, axis=1)

    # solo nos importa la fecha
    # dt.date lo convierte en objeto
    df_retiro_curso["fec_retiro"] = pd.to_datetime(
        df_retiro_curso["fec_retiro"].dt.date, yearfirst=True, format="%Y-%m-%d"
    )

    # de cursos matricula solo nos interesa saber los datos del curso por lo que trabajaremos con un subset
    df_cur_matr_ = df_base_cursos_matricula.copy()
    df_cur_matr_ = df_cur_matr_[
        ["periodo", "cod_alumno", "cod_curso", "cant_creditos", "tipo_curso"]
    ].copy()

    # creamos columnas para poder determinar numero de cursos matriculados y si son o no obligatorios
    df_cur_matr_["n"] = 1
    df_cur_matr_["flag_oblig"] = np.where(
        df_cur_matr_["tipo_curso"] == "OBLIGATORIO", 1, 0
    )
    df_cur_matr_["flag_elect"] = np.where(
        df_cur_matr_["tipo_curso"] == "OBLIGATORIO", 0, 1
    )

    # volvemos a retiros y cruzamos los valores que corresponden a sus retiros
    df_retiro_curso_step_2 = df_retiro_curso.copy()
    df_retiro_curso_step_2 = df_retiro_curso_step_2.merge(
        df_cur_matr_, how="left", on=["periodo", "cod_alumno", "cod_curso"]
    )

    # calculamos unas columnas para las variables de cantidad de creditos electivos o obligatorios se ha retirado
    df_retiro_curso_step_2["n_cred_ret_oblig"] = (
        df_retiro_curso_step_2["flag_oblig"] * df_retiro_curso_step_2["cant_creditos"]
    )
    df_retiro_curso_step_2["n_cred_ret_elect"] = (
        df_retiro_curso_step_2["flag_elect"] * df_retiro_curso_step_2["cant_creditos"]
    )

    # ahora trabajamos a nivel de cursos_matricula
    # solo nos importa la data a nivel de periodo ya que esto no cambia
    df_cur_matr_step_2 = df_cur_matr_.copy()

    # creamos las columnas de creditos totales
    df_cur_matr_step_2["n_cred_oblig"] = (
        df_cur_matr_step_2["flag_oblig"] * df_cur_matr_step_2["cant_creditos"]
    )
    df_cur_matr_step_2["n_cred_elect"] = (
        df_cur_matr_step_2["flag_elect"] * df_cur_matr_step_2["cant_creditos"]
    )

    # agrupamos por periodo
    df_cur_matr_step_2 = df_cur_matr_step_2.groupby(
        ["periodo", "cod_alumno"], as_index=False, dropna=False
    ).agg(
        cant_curso_mat=("n", "sum"),
        sum_cred_mat=("cant_creditos", "sum"),
        n_cursos_oblig_matric=("flag_oblig", "sum"),
        n_cursos_elect_matric=("flag_elect", "sum"),
        n_cred_oblig_matric=("n_cred_oblig", "sum"),
        n_cred_elect_matric=("n_cred_elect", "sum"),
    )

    # cruzamos el calendario con los cursos matriculados para tener una foto por cada fecha de corte
    df_cur_matr_step_3 = df_cur_matr_step_2.copy()
    df_cur_matr_step_3 = df_cur_matr_step_3.merge(
        df_calendar, how="left", on=["periodo"]
    )

    return df_retiro_curso_step_2, df_cur_matr_step_3


def processing_file_feature_level(
    df_retiro_curso: pd.DataFrame,
    df_base_matricula: pd.DataFrame,
) -> pd.DataFrame:

    df_retiro_curso = df_retiro_curso.drop_duplicates(
        subset=["periodo", "cod_alumno", "cod_curso"],
        keep="first",
        ignore_index=True,
    )

    # Seleccionar columnas relevantes
    df_retiro_curso = df_retiro_curso[
        [
            "periodo",
            "cod_alumno",
            "cod_modalidad_estudio",
            "cod_producto",
            "cod_curso",
            "cod_retiro_curso",
            "fec_solicitud",
            "fec_retiro",
            "cod_estado_retiro",
            "fec_creacion",
            "fec_modificacion",
        ]
    ]

    # Combinar datos de cursos de matrícula con datos de retiro de cursos
    df = pd.merge(
        df_base_matricula,
        df_retiro_curso,
        how="left",
        on=["periodo", "cod_alumno", "cod_curso"],
    )

    # fmt: off
    # Seleccionar columnas relevantes después de la combinación
    df = df[["periodo", "cod_alumno", "cod_carrera", "cod_curso",
            "cant_creditos", "tipo_curso", "ciclo_curso", "cod_modalidad_estudio",
            "cod_producto", "cod_retiro_curso", "fec_solicitud", "fec_retiro",
            "cod_estado_retiro", "fec_creacion", "fec_modificacion"]]

    df["flg_retiro_curso"] = (df["cod_estado_retiro"] == "R").astype(int)

    # Agregacion de retiros obligatorios
    df_cur_ret_oblg = (
        df.query('tipo_curso == "OBLIGATORIO" & flg_retiro_curso == 1')
        .groupby(["periodo", "cod_alumno"], as_index=False)
        .agg(
            n_retiro_curso_oblig=("cod_retiro_curso", "count"),
            n_cred_ret_curso_oblig=("cant_creditos", "sum"),
        )
    )

    # Agregacion de retiros electivos
    df_cur_ret_elect = (
        df.query('tipo_curso == "ELECTIVO" & flg_retiro_curso == 1')
        .groupby(["periodo", "cod_alumno"], as_index=False)
        .agg(
            n_retiro_curso_elect=("cod_retiro_curso", "count"),
            n_cred_ret_curso_elect=("cant_creditos", "sum"),
        )
    )

    # Agregacion de retiros acumulados máximos
    df_cur_rept_acum_max = (
        df.query('cod_estado_retiro == "R"')
        .groupby(["periodo", "cod_alumno", "cod_curso"], as_index=False)
        .agg(cant_veces_repetido=("cod_curso", "count"))
        .assign(
            cant_veces_repetido_accum=lambda x: x.groupby(["cod_alumno", "cod_curso"])["cant_veces_repetido"].cumsum()
        )
        .groupby(["periodo", "cod_alumno"], as_index=False)
        .agg(max_retiro_curso=("cant_veces_repetido_accum", "max"))
    )

    # Agregacion general y merge
    df_ret_curso_final = (
        df.groupby(["periodo", "cod_alumno"])
        .agg(
            cant_curso_mat=("cod_curso", "count"),
            sum_cred_mat=("cant_creditos", "sum"),
            flg_retiro_curso=("flg_retiro_curso", "max"),
            n_retiro_curso=("cod_retiro_curso", "count"),
        )
        .reset_index()
        .merge(df_cur_ret_oblg, how="left", on=["periodo", "cod_alumno"])
        .merge(df_cur_ret_elect, how="left", on=["periodo", "cod_alumno"])
        .merge(df_cur_rept_acum_max, how="left", on=["periodo", "cod_alumno"])
        .fillna(0)
    )

    # Variables metricas Retiro Curso
    df_ret_curso_final = df_ret_curso_final.sort_values(by=['cod_alumno', 'periodo'], ignore_index=True)
    df_ret_curso_final["n_cred_ret_curso_totales"] = df_ret_curso_final["n_cred_ret_curso_oblig"] + df_ret_curso_final["n_cred_ret_curso_elect"]
    df_ret_curso_final["r_retiro_curso"] = round(df_ret_curso_final["n_retiro_curso"] / df_ret_curso_final["cant_curso_mat"], 3)
    df_ret_curso_final["r_retiro_curso_oblig"] = round(df_ret_curso_final["n_retiro_curso_oblig"] / df_ret_curso_final["cant_curso_mat"], 3)
    df_ret_curso_final["r_retiro_curso_elect"] = round(df_ret_curso_final["n_retiro_curso_elect"] / df_ret_curso_final["cant_curso_mat"], 3)

    df_ret_curso_final["es_verano"] = df_ret_curso_final["periodo"].astype(str).str.endswith("00")

    df_ret_curso_final["n_retiro_curso_hist"] = df_ret_curso_final.groupby("cod_alumno")["n_retiro_curso"].cumsum()

    #print(df_ret_curso_final.query("cod_alumno == '20171A454'")[['periodo', 'cod_alumno', 'flg_retiro_curso', 'n_retiro_curso', 'n_retiro_curso_hist']].to_string())

    df_ret_curso_final["r_retiro_curso_cred"] = round(df_ret_curso_final["n_cred_ret_curso_totales"] / df_ret_curso_final["sum_cred_mat"], 3)
    df_ret_curso_final["r_retiro_curso_cred_oblig"] = round(df_ret_curso_final["n_cred_ret_curso_oblig"] / df_ret_curso_final["sum_cred_mat"], 3)
    df_ret_curso_final["r_retiro_curso_cred_elect"] = round(df_ret_curso_final["n_cred_ret_curso_elect"] / df_ret_curso_final["sum_cred_mat"], 3)

    # Calcular la variable n_retiro_curso_hist_reg para periodos no verano
    df_ret_curso_final.loc[~df_ret_curso_final["es_verano"], "n_retiro_curso_hist_reg"] = (
        df_ret_curso_final[~df_ret_curso_final["es_verano"]]
        .groupby("cod_alumno")["n_retiro_curso"]
        .cumsum()
    )
    # Completar los valores faltantes en la columna n_retiro_curso_hist_reg
    df_ret_curso_final["n_retiro_curso_hist_reg"] = (
        df_ret_curso_final.groupby("cod_alumno")["n_retiro_curso_hist_reg"]
        .transform(lambda x: x.ffill())
        .fillna(0)
    )

    df_ret_curso_final = df_ret_curso_final.drop_duplicates(subset=["periodo", "cod_alumno"], ignore_index=True)

    # Seleccionar las columnas finales del DataFrame
    df_ret_curso_final = df_ret_curso_final[["periodo", "cod_alumno", "flg_retiro_curso",
                                                "n_retiro_curso", "n_retiro_curso_oblig", "n_retiro_curso_elect",
                                                "r_retiro_curso", "r_retiro_curso_oblig", "r_retiro_curso_elect",
                                                "max_retiro_curso", "n_retiro_curso_hist", "n_retiro_curso_hist_reg",
                                                "n_cred_ret_curso_totales", "n_cred_ret_curso_oblig", "n_cred_ret_curso_elect",
                                                "r_retiro_curso_cred", "r_retiro_curso_cred_oblig", "r_retiro_curso_cred_elect",
                                            ]]
    
    numeric_columns = df_ret_curso_final.select_dtypes(include=np.number).columns
    df_ret_curso_final[numeric_columns] = df_ret_curso_final[numeric_columns].round(3)
    
    return df_ret_curso_final.drop_duplicates(ignore_index=True)


def processing_file_feature_level_date(
    df_retiro_curso: pd.DataFrame,
    df_base_matricula: pd.DataFrame,
    features_list: list = None,
) -> pd.DataFrame:
    """
    Calculo de los features que se encuentran en la capa gold.
    Estos calculos se encuentran a nivel de PERIODO-COD_ALUMNO-FECHA_CORTE.
    """

    # Agrupamos el DataFrame de retiro de curso por periodo, código de alumno y fecha de corte
    df_retiro_curso = df_retiro_curso.groupby(
        ["periodo", "cod_alumno", "fecha_corte"], as_index=False, dropna=False
    ).agg(
        n_retiro_curso=("n", "sum"),
        flg_retiro_curso=("flg_retiro_curso", "max"),
        n_retiro_curso_oblig=("flag_oblig", "sum"),
        n_retiro_curso_elect=("flag_elect", "sum"),
        n_cred_ret_curso_oblig=("n_cred_ret_oblig", "sum"),
        n_cred_ret_curso_elect=("n_cred_ret_elect", "sum"),
    )

    # Ordenamos los datos por código de alumno y periodo
    df_retiro_curso = df_retiro_curso.sort_values(by=["cod_alumno", "periodo"])

    # Calculamos variables históricas para cada estudiante
    df_retiro_curso["n_retiros_historicos"] = df_retiro_curso.groupby("cod_alumno")[
        "n_retiro_curso"
    ].cumsum()

    # Calculamos el total de créditos retirados
    df_retiro_curso["n_cred_ret_curso_totales"] = (
        df_retiro_curso["n_cred_ret_curso_oblig"]
        + df_retiro_curso["n_cred_ret_curso_elect"]
    )

    # Calculamos el historial de retiros de curso
    df_retiro_curso["n_retiro_curso_hist"] = df_retiro_curso.groupby("cod_alumno")[
        "n_retiro_curso"
    ].cumsum()

    # Unimos los datos de retiro de curso con la base de matrícula
    df_retcurso_cursmatri = df_retiro_curso.merge(
        df_base_matricula,
        how="inner",
        on=["periodo", "cod_alumno", "fecha_corte"],
    )

    # Calculamos ratios de retiro de curso
    df_retcurso_cursmatri["r_retiro_curso"] = round(
        df_retcurso_cursmatri["n_retiro_curso"]
        / df_retcurso_cursmatri["cant_curso_mat"],
        3,
    )
    df_retcurso_cursmatri["r_retiro_curso_oblig"] = round(
        df_retcurso_cursmatri["n_retiro_curso_oblig"]
        / df_retcurso_cursmatri["cant_curso_mat"],
        3,
    )
    df_retcurso_cursmatri["r_retiro_curso_elect"] = round(
        df_retcurso_cursmatri["n_retiro_curso_elect"]
        / df_retcurso_cursmatri["cant_curso_mat"],
        3,
    )
    df_retcurso_cursmatri["r_retiro_curso_cred"] = round(
        df_retcurso_cursmatri["n_cred_ret_curso_totales"]
        / df_retcurso_cursmatri["sum_cred_mat"],
        3,
    )
    df_retcurso_cursmatri["r_retiro_curso_cred_oblig"] = round(
        df_retcurso_cursmatri["n_cred_ret_curso_oblig"]
        / df_retcurso_cursmatri["sum_cred_mat"],
        3,
    )
    df_retcurso_cursmatri["r_retiro_curso_cred_elect"] = round(
        df_retcurso_cursmatri["n_cred_ret_curso_elect"]
        / df_retcurso_cursmatri["sum_cred_mat"],
        3,
    )

    if features_list is not None:
        df_retcurso_cursmatri = df_retcurso_cursmatri[features_list]

    return df_retcurso_cursmatri
