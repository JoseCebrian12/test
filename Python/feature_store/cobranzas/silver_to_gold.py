import os

import numpy as np
import pandas as pd

from ..general.funcs import load_parquet


# fmt: off
def processing_file_feature_level(df: pd.DataFrame, features_dict: dict = None) -> pd.DataFrame:
    """
    Calculo de los features que se encuentran en la capa gold.
    Estos calculos se encuentran a nivel de PERIODO-COD_ALUMNO-FECHA_CORTE.
    """

    df_ = df.copy()

    df_.loc[:, "flag_bv"] = np.where(df_["cod_tipo_documento"] == 1, 1, 0)
    df_.loc[:, "flag_fc"] = np.where(df_["cod_tipo_documento"] == 5, 1, 0)
    df_.loc[:, "flag_lc"] = np.where(df_["cod_tipo_documento"] == 4, 1, 0)

    # documentos anulados
    df_.loc[:, "flag_anulado"] = np.where(df_["estado"] == "ANULADO", 1, 0)
    df_.loc[:, "monto_anulado"] = df_["flag_anulado"] * df_["monto"]

    # documentos no anulados
    df_.loc[:, "flag_doc_valido"] = np.where(df_["estado"].isin(["ANULADO", "NO_DETERMINADO"]), 0, 1)
    df_.loc[:, "monto_valido"] = df_["flag_doc_valido"] * df_["monto"]

    # flag conceptos de matricula y valido
    df_.loc[:, "flag_matricula"] = np.where(df_["matricula"] == 1, 1, 0)
    df_.loc[:, "monto_matricula"] = df_["flag_matricula"] * df_["monto"] * df_["flag_doc_valido"]

    # flag concepto tuition y valido
    df_.loc[:, "flag_tuition"] = np.where(df_["tuition"] == 1, 1, 0)
    df_.loc[:, "monto_tuition"] = df_["flag_tuition"] * df_["monto"] * df_["flag_doc_valido"]

    # montos emitidos BV, FC, LC y validos
    df_.loc[:, "monto_bv"] = df_["flag_bv"] * df_["monto"] * df_["flag_doc_valido"]
    df_.loc[:, "monto_fc"] = df_["flag_fc"] * df_["monto"] * df_["flag_doc_valido"]
    df_.loc[:, "monto_lc"] = df_["flag_lc"] * df_["monto"] * df_["flag_doc_valido"]

    # delta de dias de pago entre los documentos
    df_.loc[:, "delta_dias_pago"] = (df_["fecha_pago_filtro"].dt.normalize() - df_["fecha_vencimiento"].dt.normalize()).dt.days

    # calculo de dias de mora solo para cuotas tuition
    df_.loc[:, "delta_dias_pago_tuition"] = np.where(df_["nro_cuota"].between(1, 10), df_["delta_dias_pago"], np.nan)
    df_.loc[:, "delta_dias_pago_matricula"] = np.where(df_["flag_matricula"] == 1, df_["delta_dias_pago"], np.nan)
    
    # calculamos los dias de mora
    df_.loc[:, "dias_mora"] = np.where(df_["fecha_pago_filtro"].isna(), (df_["fecha_corte"].dt.normalize() - df_["fecha_vencimiento"].dt.normalize()).dt.days, df_["delta_dias_pago"])
    df_.loc[:, "dias_mora"] = np.where(df_["dias_mora"]<=6, 0, df_["dias_mora"] - 6)

    # dias acumulados de la diferencia entre la fecha de pago y la fecha de vencimiento
    df_.loc[:, "dias_mora_tuition"] = df_["flag_tuition"] * df_["dias_mora"]

    # flag morosidad
    df_.loc[:, "flag_mora"] = np.where(df_["dias_mora"]>0, 1, 0)

    # flag pago 7d
    df_.loc[:, "flag_pago_7d"] = np.where(df_["delta_dias_pago"] <= 6, 1, 0)

    # documentos pagados
    df_.loc[:, "flag_pago"] = np.where(df_["estado"].isin(["PAGO_REGULAR", "PAGO_NC"]), 1, 0)

    # montos pagados
    df_.loc[:, "monto_pagado"] = df_["flag_pago"] * df_["monto_pagado_filtro"]
    df_.loc[:, "monto_pagado_7d"] = df_.loc[:, "monto_pagado"] * df_["flag_pago_7d"]

    # documentos matricula
    df_.loc[:, "flag_pago_matricula"] = df_["flag_pago"] * df_["flag_matricula"]
    df_.loc[:, "flag_pago_matricula_7d"] = df_["flag_pago"] * df_["flag_matricula"] * df_["flag_pago_7d"]

    # monto pagados sobre matricula
    df_.loc[:, "monto_pagado_matricula"] = df_["flag_pago_matricula"] * df_["monto_pagado_filtro"]
    df_.loc[:, "monto_pagado_matricula_7d"] = df_["monto_pagado_matricula"] * df_["flag_pago_matricula_7d"]

    # documentos tuition
    df_.loc[:, "flag_pago_tuition"] = df_["flag_pago"] * df_["flag_tuition"]
    df_.loc[:, "flag_pago_tuition_7d"] = df_["flag_pago"] * df_["flag_tuition"] * df_["flag_pago_7d"]

    # monto pagados sobre tuition
    df_.loc[:, "monto_pagado_tuition"] = df_["flag_pago_tuition"] * df_["monto_pagado_filtro"]
    df_.loc[:, "monto_pagado_tuition_7d"] = df_["monto_pagado_tuition"] * df_["flag_pago_tuition_7d"]

    # documento boletas pagados
    df_.loc[:, "flag_pago_bv"] = df_["flag_bv"] * df_["flag_pago"]
    df_.loc[:, "flag_pago_bv_7d"] = df_["flag_pago_bv"] * df_["flag_pago_7d"]

    # documento facturas pagados
    df_.loc[:, "flag_pago_fc"] = df_["flag_fc"] * df_["flag_pago"]
    df_.loc[:, "flag_pago_fc_7d"] = df_["flag_pago_fc"] * df_["flag_pago_7d"]

    # Al final del periodo, días que demora en pagar la cuota 1.
    df_.loc[:, "flag_cuota1"] = (df_["nro_cuota"] == 1).astype(int)
    df_.loc[:, "dias_pago_cuota_1"] = np.where(df_["flag_cuota1"] == 1, df_["delta_dias_pago"], np.nan)
    df_.loc[:, "fecha_pago_cuota_1"] = np.where(df_["flag_cuota1"] == 1, df_["fecha_pago_filtro"], np.datetime64("NaT"))

    # Al final del periodo, días que demora en pagar la cuota 2.
    df_.loc[:, "flag_cuota2"] = (df_["nro_cuota"] == 2).astype(int)
    df_.loc[:, "dias_pago_cuota_2"] = np.where(df_["flag_cuota2"] == 1, df_["delta_dias_pago"], np.nan)
    df_.loc[:, "fecha_pago_cuota_2"] = np.where(df_["flag_cuota2"] == 1, df_["fecha_pago_filtro"], np.datetime64("NaT"))

    # Al final del periodo, días que demora en pagar la cuota 3.
    df_.loc[:, "flag_cuota3"] = (df_["nro_cuota"] == 3).astype(int)
    df_.loc[:, "dias_pago_cuota_3"] = np.where(df_["flag_cuota3"] == 1, df_["delta_dias_pago"], np.nan)
    df_.loc[:, "fecha_pago_cuota_3"] = np.where(df_["flag_cuota3"] == 1, df_["fecha_pago_filtro"], np.datetime64("NaT"))

    # Al final del periodo, días que demora en pagar la cuota 4.
    df_.loc[:, "flag_cuota4"] = (df_["nro_cuota"] == 4).astype(int)
    df_.loc[:, "dias_pago_cuota_4"] = np.where(df_["flag_cuota4"] == 1, df_["delta_dias_pago"], np.nan)
    df_.loc[:, "fecha_pago_cuota_4"] = np.where(df_["flag_cuota4"] == 1, df_["fecha_pago_filtro"], np.datetime64("NaT"))
    
    # Al final del periodo, días que demora en pagar la cuota 5
    df_.loc[:, "flag_cuota5"] = (df_["nro_cuota"] == 5).astype(int)
    df_.loc[:, "dias_pago_cuota_5"] = np.where(df_["flag_cuota5"] == 1, df_["delta_dias_pago"], np.nan)
    df_.loc[:, "fecha_pago_cuota_5"] = np.where(df_["flag_cuota5"] == 1, df_["fecha_pago_filtro"], np.datetime64("NaT"))

    # Al final del periodo, días que demora en pagar la cuota 6
    df_.loc[:, "flag_cuota6"] = (df_["nro_cuota"] == 6).astype(int)
    df_.loc[:, "dias_pago_cuota_6"] = np.where(df_["flag_cuota6"] == 1, df_["delta_dias_pago"], np.nan)
    df_.loc[:, "fecha_pago_cuota_6"] = np.where(df_["flag_cuota6"] == 1, df_["fecha_pago_filtro"], np.datetime64("NaT"))

    df_ = df_.groupby(["periodo", "cod_alumno", "fecha_corte"], as_index=False, dropna=False).agg(

        n_doc = ("cod_documento", "count"), # total de documentos que el estudiante tiene hasta la fecha de corte
        n_doc_matricula = ("matricula", "sum"), # total de documentos relacionados a matricula que el estudiante tiene hasta la fecha de corte
        n_doc_tuition = ("tuition", "sum"), # total de documentos relacionados a tuition que el estudiante tiene hasta la fecha de corte
        
        n_doc_BV = ("flag_bv", "sum"),  # total de documentos que son boletas
        n_doc_FC = ("flag_fc", "sum"), # total de documentos que son facturas
        n_doc_LC = ("flag_lc", "sum"), # total de documentos que son letras de canje
        n_doc_NC = ("n_nc", "sum"), # total de documentos que son notas de credito (este campo viene consolidado desde la capa anterior)
        
        n_doc_anulados = ("flag_anulado", "sum"), # total de documentos con estado anulado.

        monto_emit = ("monto_valido", "sum"), # total del monto que se ha emitido, no considera documetos con estados ANULADO o NO_DETERMINADO.
        monto_emit_matricula = ("monto_matricula", "sum"), # total del monto de matricula emitido, no considera documetos con estados ANULADO o NO_DETERMINADO.
        monto_emit_tuition = ("monto_tuition", "sum"), # total del monto de tuition emitido, no considera documetos con estados ANULADO o NO_DETERMINADO.

        monto_emit_BV = ("monto_bv", "sum"), # total del monto de boletas emitido, no considera documetos con estados ANULADO o NO_DETERMINADO.
        monto_emit_FC = ("monto_fc", "sum"), # total del monto de facturas emitido, no considera documetos con estados ANULADO o NO_DETERMINADO.
        monto_emit_LC = ("monto_lc", "sum"), # total del monto de letras de canje emitido, no considera documetos con estados ANULADO o NO_DETERMINADO.
        monto_emit_NC = ("monto_nc", "sum"), # total del monto de notas de credito emitido, no considera documetos con estados ANULADO o NO_DETERMINADO.
        
        n_doc_pago = ("flag_pago", "sum"), # total de documentos que se encuentran con el estado PAGADO o PAGADO_NC a la fecha de corte.

        n_doc_matricula_pago = ("flag_pago_matricula", "sum"), # total de documentos relacionados a matricula que han sido pagados.
        n_doc_matricula_pago_7d = ("flag_pago_matricula_7d", "sum"), # total de documentos relacionados a matricula que han sido pagados hasta 7d luego del vencimiento.
        
        n_doc_tuition_pago = ("flag_pago_tuition", "sum"), # total de documentos relacionados a tuition que han sido pagados.
        n_doc_tuition_pago_7d = ("flag_pago_tuition_7d", "sum"), # total de documentos relacionados a tuition que han sido pagados hasta 7d luego del vencimiento.
        
        n_doc_BV_pago = ("flag_pago_bv", "sum"), # total de documentos relacionados a boletas que han sido pagados.
        n_doc_BV_pago_7d = ("flag_pago_bv_7d", "sum"), # total de documentos relacionados a boletas que han sido pagados hasta 7d luego del vencimiento.

        n_doc_FC_pago = ("flag_pago_fc", "sum"), # total de documentos relacionados a facturas que han sido pagados.
        n_doc_FC_pago_7d = ("flag_pago_fc_7d", "sum"), # total de documentos relacionados a facturas que han sido pagados hasta 7d luego del vencimiento.

        monto_pago = ("monto_pagado", "sum"), # monto total pagado.
        monto_pago_7d =("monto_pagado_7d", "sum"), # monto total pagado hasta 7 dias luego de su vencimiento.
        
        monto_matricula_pago = ("monto_pagado_matricula", "sum"), # monto total de documentos relacionados a matricula pagado.
        monto_matricula_pago_7d = ("monto_pagado_matricula_7d", "sum"), # monto total de documentos relacionados a matricula pagados hasta 7 dias luego de su vencimiento.
        
        monto_tuition_pago = ("monto_pagado_tuition", "sum"), # monto total de documentos relacionados a tuition pagado.
        monto_tuition_pago_7d = ("monto_pagado_tuition_7d", "sum"), # monto total de documentos relacionados a tuition pagados hasta 7 dias luego de su vencimiento.

        flag_cuota_1 = ("flag_cuota1", "max"), # flag que indica la tenencia o no de la cuota 1
        flag_cuota_2 = ("flag_cuota2", "max"), # flag que indica la tenencia o no de la cuota 2
        flag_cuota_3 = ("flag_cuota3", "max"), # flag que indica la tenencia o no de la cuota 3
        flag_cuota_4 = ("flag_cuota4", "max"), # flag que indica la tenencia o no de la cuota 4
        flag_cuota_5 = ("flag_cuota5", "max"), # flag que indica la tenencia o no de la cuota 5
        flag_cuota_6 = ("flag_cuota6", "max"), # flag que indica la tenencia o no de la cuota 6

        dias_pago_cuota_1 = ("dias_pago_cuota_1", "max"), # dias que se demoro en pagar la cuota 1
        dias_pago_cuota_2 = ("dias_pago_cuota_2", "max"), # dias que se demoro en pagar la cuota 2
        dias_pago_cuota_3 = ("dias_pago_cuota_3", "max"), # dias que se demoro en pagar la cuota 3
        dias_pago_cuota_4 = ("dias_pago_cuota_4", "max"), # dias que se demoro en pagar la cuota 4
        dias_pago_cuota_5 = ("dias_pago_cuota_5", "max"), # dias que se demoro en pagar la cuota 5
        dias_pago_cuota_6 = ("dias_pago_cuota_6", "max"), # dias que se demoro en pagar la cuota 6

        fecha_pago_cuota_1 = ("fecha_pago_cuota_1", "max"), # fecha de pago de la cuota 1 
        fecha_pago_cuota_2 = ("fecha_pago_cuota_2", "max"), # fecha de pago de la cuota 2
        fecha_pago_cuota_3 = ("fecha_pago_cuota_3", "max"), # fecha de pago de la cuota 3
        fecha_pago_cuota_4 = ("fecha_pago_cuota_4", "max"), # fecha de pago de la cuota 4
        fecha_pago_cuota_5 = ("fecha_pago_cuota_5", "max"), # fecha de pago de la cuota 5
        fecha_pago_cuota_6 = ("fecha_pago_cuota_6", "max"), # fecha de pago de la cuota 6

        dias_mora_tuition_acum = ("dias_mora_tuition", "sum"), # total de dias de mora acumulado de los documentos relacionados a tuition.
        dias_mora_tuition_avg = ("dias_mora_tuition", "mean"), # total de dias de mora acumulado de los documentos relacionados a tuition.

        avg_dias_pago_tuition = ("delta_dias_pago_tuition", "mean"), # promedio de dias en los que paga los documentos de tuition.
        avg_dias_pago_matricula = ("delta_dias_pago_matricula", "mean"), # promedio de dias en los que paga los documentos de matricula.
    )

    df_ = df_.assign(
        r_doc_BV = df_["n_doc_BV"] / df_["n_doc"], # ratio de documentos que son boletas
        r_doc_FC = df_["n_doc_FC"] / df_["n_doc"], # ratio de documentos que son facturas
        r_doc_LC = df_["n_doc_LC"] / df_["n_doc"], # ratio de documentos que son letras de canje

        r_doc_pago = df_["n_doc_pago"] / df_["n_doc"], # ratio de documentos pagados
        r_doc_matricula_pago = df_["n_doc_matricula_pago"] / df_["n_doc_matricula"], # ratio de documentos matriculas pagados
        r_doc_tuition_pago = df_["n_doc_tuition_pago"] / df_["n_doc_tuition"], # ratio de documentos sobre tuition pagados

        r_doc_BV_pago = df_["n_doc_BV_pago"] / df_["n_doc_BV"], # ratio de documentos boletas pagados
        r_doc_FC_pago = df_["n_doc_FC_pago"] / df_["n_doc_FC"], # ratio de documentos facturas pagados

        r_doc_matricula_pago_7d = df_["n_doc_matricula_pago_7d"] / df_["n_doc_matricula"], # ratio de documentos relacionados a matricula pagados hasta 7d desde su vencimiento.
        r_doc_tuition_pago_7d = df_["n_doc_tuition_pago_7d"] / df_["n_doc_tuition"], # ratio de documentos relacionados a tuition pagados hasta 7d desde su vencimiento.

        r_doc_BV_pago_7d = df_["n_doc_BV_pago_7d"] / df_["n_doc_BV"], # ratio de boletas pagados hasta 7d desde su vencimiento
        r_doc_FC_pago_7d = df_["n_doc_FC_pago_7d"] / df_["n_doc_FC"], # ratio de facturas pagados hasta 7d desde su vencimiento

        r_monto_pago = df_["monto_pago"] / df_["monto_emit"], # ratio de monto pagado total
        r_monto_pago_7d = df_["monto_pago_7d"] / df_["monto_emit"], # ratio de monto pagado hasta 7d desde su vencimiento.

        r_monto_matricula_pago = df_["monto_matricula_pago"] / df_["monto_emit_matricula"], # ratio de monto pagado sobre matricula total
        r_monto_matricula_pago_7d = df_["monto_matricula_pago_7d"] / df_["monto_emit_matricula"], # ratio de monto pagado sobre matricula hasta 7d desde su vencimiento.
        
        r_monto_tuition_pago = df_["monto_tuition_pago"] / df_["monto_emit_tuition"], # ratio de monto pagado sobre tuition total
        r_monto_tuition_pago_7d = df_["monto_tuition_pago_7d"] / df_["monto_emit_tuition"], # ratio de monto pagado sobre tuition hasta 7d desde su vencimiento.
    
    )

    if features_dict is not None:
        df_ = df_[features_dict.keys()]

    return df_

def processing_file_perfil_morosidad_eop(
        df: pd.DataFrame,
        dict_peso_mora: dict,
        dict_peso_pago: dict,
        dict_peso_tipo_pago: dict,
        return_periodo: int
    ) -> pd.DataFrame:
    """Procesa datos de cobranzas para calcular perfiles de morosidad y pago.

    Esta función realiza cálculos detallados sobre los patrones de pago y morosidad
    de los alumnos, generando perfiles basados en su comportamiento histórico y actual.

    Args:
        df (pd.DataFrame): DataFrame con los datos de cobranzas.
        dict_peso_mora (dict): Diccionario que mapea perfiles de mora a pesos numéricos.
        dict_peso_pago (dict): Diccionario que mapea perfiles de pago a pesos numéricos.
        dict_peso_tipo_pago (dict): Diccionario que mapea los tipos de perfiles de pago a pesos numéricos.
        return_periodo (int): Periodo específico para el cual se deben devolver los resultados.

    Returns:
        pd.DataFrame: DataFrame con los perfiles de morosidad y pago calculados,
                      incluyendo métricas históricas y actuales.

    Note:
        Esta función realiza múltiples transformaciones y cálculos, incluyendo:
        - Filtrado de datos relevantes.
        - Cálculo de días de mora y flags de pago.
        - Agregación de datos a nivel de alumno y periodo.
        - Cálculo de ratios de mora y pago.
        - Asignación de perfiles de mora y pago.
        - Cálculo de métricas históricas y acumuladas.
    """
    df_cobranzas = df.copy()

    reverse_dict_peso_mora = dict(map(reversed, dict_peso_mora.items()))
    reverse_dict_peso_pago = dict(map(reversed, dict_peso_pago.items()))
    reverse_dict_peso_tipo_pago = dict(map(reversed, dict_peso_tipo_pago.items()))

    # Filtrado para obtener
    # * Periodos Regulares
    # * Cuotas 2, 3, 4, 5
    # * Estados que no sean ANULADO, NO_DETERMINADO
    df_cobranzas = df_cobranzas.loc[
        (~df_cobranzas["periodo"].astype(str).str.endswith("00"))
        & (df_cobranzas["nro_cuota"].isin([2, 3, 4, 5]))
        & (~df_cobranzas["estado"].isin(["ANULADO", "NO_DETERMINADO"]))
    ]

    # delta de dias de pago entre los documentos
    df_cobranzas.loc[:, "delta_dias_pago"] = (
        df_cobranzas["fecha_pago_filtro"].dt.normalize()
        - df_cobranzas["fecha_vencimiento"].dt.normalize()
    ).dt.days

    # calculamos los dias de mora
    df_cobranzas.loc[:, "dias_mora"] = np.where(
        df_cobranzas["fecha_pago_filtro"].isna(),
        (
            df_cobranzas["fecha_corte"].dt.normalize()
            - df_cobranzas["fecha_vencimiento"].dt.normalize()
        ).dt.days,
        df_cobranzas["delta_dias_pago"],
    )

    # Si los dias de mora son mayores a 15 se obtienen los dias de mas
    df_cobranzas.loc[:, "dias_mora"] = np.where(
        df_cobranzas["dias_mora"] < 7, 0, df_cobranzas["dias_mora"] - 6
    )

    # Obtiendo la fecha de vencimiento mas reciente de cada alumno
    df_cobranzas["fecha_vencimiento_mas_reciente"] = df_cobranzas.groupby(
        by=["periodo", "cod_alumno", "fecha_corte"]
    )["fecha_vencimiento"].transform("max")

    # flag mora
    df_cobranzas.loc[:, "flag_mora"] = np.where(df_cobranzas["dias_mora"] > 0, 1, 0)

    # flag pago
    df_cobranzas.loc[:, "flag_pago"] = np.where(
        df_cobranzas["fecha_pago_filtro"].notna(), 1, 0
    )

    # flag para determinar si el pago se realizo despues de la fecha de vencimiento mas reciente
    df_cobranzas["flag_pago_dsps_fec_venc_max"] = np.where(
        df_cobranzas["fecha_pago_filtro"].isna(),
        np.nan,
        np.where(
            df_cobranzas["fecha_pago_filtro"] <= df_cobranzas["fecha_vencimiento_mas_reciente"],
            0,
            1,
        ),
    )

    # Al final del periodo, días que demora en pagar la cuota 2.
    df_cobranzas["flag_cuota2"] = (df_cobranzas["nro_cuota"] == 2).astype(int)
    df_cobranzas["dias_pago_cuota_2"] = np.where(
        df_cobranzas["flag_cuota2"] == 1, df_cobranzas["delta_dias_pago"], np.nan
    )
    df_cobranzas["fecha_pago_cuota_2"] = np.where(
        df_cobranzas["flag_cuota2"] == 1,
        df_cobranzas["fecha_pago_filtro"],
        np.datetime64("NaT"),
    )
    df_cobranzas["fecha_venc_cuota_2"] = np.where(
        df_cobranzas["flag_cuota2"] == 1,
        df_cobranzas["fecha_vencimiento"],
        np.datetime64("NaT"),
    )

    # Al final del periodo, días que demora en pagar la cuota 3.
    df_cobranzas["flag_cuota3"] = (df_cobranzas["nro_cuota"] == 3).astype(int)
    df_cobranzas["dias_pago_cuota_3"] = np.where(
        df_cobranzas["flag_cuota3"] == 1, df_cobranzas["delta_dias_pago"], np.nan
    )
    df_cobranzas["fecha_pago_cuota_3"] = np.where(
        df_cobranzas["flag_cuota3"] == 1,
        df_cobranzas["fecha_pago_filtro"],
        np.datetime64("NaT"),
    )
    df_cobranzas["fecha_venc_cuota_3"] = np.where(
        df_cobranzas["flag_cuota3"] == 1,
        df_cobranzas["fecha_vencimiento"],
        np.datetime64("NaT"),
    )

    # Al final del periodo, días que demora en pagar la cuota 4.
    df_cobranzas["flag_cuota4"] = (df_cobranzas["nro_cuota"] == 4).astype(int)
    df_cobranzas["dias_pago_cuota_4"] = np.where(
        df_cobranzas["flag_cuota4"] == 1, df_cobranzas["delta_dias_pago"], np.nan
    )
    df_cobranzas["fecha_pago_cuota_4"] = np.where(
        df_cobranzas["flag_cuota4"] == 1,
        df_cobranzas["fecha_pago_filtro"],
        np.datetime64("NaT"),
    )
    df_cobranzas["fecha_venc_cuota_4"] = np.where(
        df_cobranzas["flag_cuota4"] == 1,
        df_cobranzas["fecha_vencimiento"],
        np.datetime64("NaT"),
    )

    # Al final del periodo, días que demora en pagar la cuota 5.
    df_cobranzas["flag_cuota5"] = (df_cobranzas["nro_cuota"] == 5).astype(int)
    df_cobranzas["dias_pago_cuota_5"] = np.where(
        df_cobranzas["flag_cuota5"] == 1, df_cobranzas["delta_dias_pago"], np.nan
    )
    df_cobranzas["fecha_pago_cuota_5"] = np.where(
        df_cobranzas["flag_cuota5"] == 1,
        df_cobranzas["fecha_pago_filtro"],
        np.datetime64("NaT"),
    )
    df_cobranzas["fecha_venc_cuota_5"] = np.where(
        df_cobranzas["flag_cuota5"] == 1,
        df_cobranzas["fecha_vencimiento"],
        np.datetime64("NaT"),
    )

    # Obteniendo la fecha de vencimiento mas reciente
    df_cobranzas["fecha_venc_mas_actual"] = (
        df_cobranzas["fecha_venc_cuota_5"]
        .combine_first(df_cobranzas["fecha_venc_cuota_4"])
        .combine_first(df_cobranzas["fecha_venc_cuota_3"])
        .combine_first(df_cobranzas["fecha_venc_cuota_2"])
    )

    df_cobranzas_agg = df_cobranzas.groupby(
        by=["cod_alumno", "periodo", "fecha_corte"], as_index=False, dropna=False
    ).agg(
        count_cuotas=("cod_documento", "count"),
        count_cuotas_mora=("flag_mora", "sum"),
        count_cuotas_pago=("flag_pago", "sum"),
        avg_flag_pago=("flag_pago_dsps_fec_venc_max", "mean"),  # reg o extr
        fecha_pago_cuota_2 = ("fecha_pago_cuota_2", "max"), # fecha de pago de la cuota 2
        fecha_pago_cuota_3 = ("fecha_pago_cuota_3", "max"), # fecha de pago de la cuota 3
        fecha_pago_cuota_4 = ("fecha_pago_cuota_4", "max"), # fecha de pago de la cuota 4
        fecha_pago_cuota_5 = ("fecha_pago_cuota_5", "max"), # fecha de pago de la cuota 5
        fecha_venc_mas_actual = ("fecha_venc_mas_actual", "max"), # fecha de vencimiento mas reciente
    )

    df_cobranzas_agg = df_cobranzas_agg.sort_values(by=["cod_alumno", "periodo"], ignore_index=True)

    df_cobranzas_agg["perfil_mora_actual"] = df_cobranzas_agg["count_cuotas_mora"].map(reverse_dict_peso_mora)
    df_cobranzas_agg["perfil_pago_actual"] = df_cobranzas_agg["count_cuotas_pago"].map(reverse_dict_peso_pago)

    df_cobranzas_agg["perfil_tipo_pago"] = np.where(df_cobranzas_agg["avg_flag_pago"] >= 0.5, "EXT", "REG")

    df_cobranzas_agg["peso_mora_actual"] = df_cobranzas_agg["perfil_mora_actual"].map(dict_peso_mora)
    df_cobranzas_agg["peso_pago_actual"] = df_cobranzas_agg["perfil_pago_actual"].map(dict_peso_pago)
    df_cobranzas_agg["peso_tipo_pago_actual"] = df_cobranzas_agg["perfil_tipo_pago"].map(dict_peso_tipo_pago)

    df_cobranzas_agg["lag_1_peso_mora"] = df_cobranzas_agg.groupby(["cod_alumno"])["peso_mora_actual"].shift(1)
    df_cobranzas_agg["lag_2_peso_mora"] = df_cobranzas_agg.groupby(["cod_alumno"])["peso_mora_actual"].shift(2)
    df_cobranzas_agg["lag_3_peso_mora"] = df_cobranzas_agg.groupby(["cod_alumno"])["peso_mora_actual"].shift(3)

    df_cobranzas_agg["lag_1_peso_pago"] = df_cobranzas_agg.groupby(["cod_alumno"])["peso_pago_actual"].shift(1)
    df_cobranzas_agg["lag_2_peso_pago"] = df_cobranzas_agg.groupby(["cod_alumno"])["peso_pago_actual"].shift(2)
    df_cobranzas_agg["lag_3_peso_pago"] = df_cobranzas_agg.groupby(["cod_alumno"])["peso_pago_actual"].shift(3)

    df_cobranzas_agg["lag_1_tipo_pago"] = df_cobranzas_agg.groupby(["cod_alumno"])["peso_tipo_pago_actual"].shift(1)
    df_cobranzas_agg["lag_2_tipo_pago"] = df_cobranzas_agg.groupby(["cod_alumno"])["peso_tipo_pago_actual"].shift(2)
    df_cobranzas_agg["lag_3_tipo_pago"] = df_cobranzas_agg.groupby(["cod_alumno"])["peso_tipo_pago_actual"].shift(3)

    df_cobranzas_agg["peso_mora_avg_3_ult_periodos_anteriores"] = df_cobranzas_agg[["lag_1_peso_mora", "lag_2_peso_mora", "lag_3_peso_mora"]].mean(axis=1, skipna=True).round()
    df_cobranzas_agg["peso_pago_avg_3_ult_periodos_anteriores"] = df_cobranzas_agg[["lag_1_peso_pago", "lag_2_peso_pago", "lag_3_peso_pago"]].mean(axis=1, skipna=True).round()
    df_cobranzas_agg["peso_tipo_pago_avg_3_ult_periodos_anteriores"] = df_cobranzas_agg[["lag_1_tipo_pago", "lag_2_tipo_pago", "lag_3_tipo_pago"]].mean(axis=1, skipna=True).round()

    df_cobranzas_agg["peso_mora_avg_4_ult_periodos"] = df_cobranzas_agg[["peso_mora_actual", "lag_1_peso_mora", "lag_2_peso_mora", "lag_3_peso_mora"]].mean(axis=1, skipna=True).round()
    df_cobranzas_agg["peso_pago_avg_4_ult_periodos"] = df_cobranzas_agg[["peso_pago_actual", "lag_1_peso_pago", "lag_2_peso_pago", "lag_3_peso_pago"]].mean(axis=1, skipna=True).round()
    df_cobranzas_agg["peso_tipo_pago_avg_4_ult_periodos"] = df_cobranzas_agg[["peso_tipo_pago_actual", "lag_1_tipo_pago", "lag_2_tipo_pago", "lag_3_tipo_pago"]].mean(axis=1, skipna=True).round()

    df_cobranzas_agg["perfil_mora_3_ult_periodos_anteriores"] = df_cobranzas_agg["peso_mora_avg_3_ult_periodos_anteriores"].map(reverse_dict_peso_mora)
    df_cobranzas_agg["perfil_pago_3_ult_periodos_anteriores"] = df_cobranzas_agg["peso_pago_avg_3_ult_periodos_anteriores"].map(reverse_dict_peso_pago)
    df_cobranzas_agg["perfil_tipo_pago_3_ult_periodos_anteriores"] = df_cobranzas_agg["peso_tipo_pago_avg_3_ult_periodos_anteriores"].map(reverse_dict_peso_tipo_pago)

    df_cobranzas_agg["perfil_mora_4_ult_periodos"] = df_cobranzas_agg["peso_mora_avg_4_ult_periodos"].map(reverse_dict_peso_mora)
    df_cobranzas_agg["perfil_pago_4_ult_periodos"] = df_cobranzas_agg["peso_pago_avg_4_ult_periodos"].map(reverse_dict_peso_pago)
    df_cobranzas_agg["perfil_tipo_pago_4_ult_periodos"] = df_cobranzas_agg["peso_tipo_pago_avg_4_ult_periodos"].map(reverse_dict_peso_tipo_pago)

    df_cobranzas_agg["perfil_cobranzas_3_ult_periodos_anteriores"] = df_cobranzas_agg["perfil_mora_3_ult_periodos_anteriores"] + "_" + df_cobranzas_agg["perfil_pago_3_ult_periodos_anteriores"] + "_" + df_cobranzas_agg["perfil_tipo_pago_3_ult_periodos_anteriores"]
    df_cobranzas_agg["perfil_cobranzas_4_ult_periodos"] = df_cobranzas_agg["perfil_mora_4_ult_periodos"] + "_" + df_cobranzas_agg["perfil_pago_4_ult_periodos"] + "_" + df_cobranzas_agg["perfil_tipo_pago_4_ult_periodos"]
    df_cobranzas_agg["perfil_cobranzas_actual"] = df_cobranzas_agg["perfil_mora_actual"] + "_" + df_cobranzas_agg["perfil_pago_actual"] + "_" + df_cobranzas_agg["perfil_tipo_pago"]

    df_cobranzas_agg["perfil_cobranzas_clasico_step_1"] = df_cobranzas_agg.apply(
        lambda x: GetPerfilMoroso(
            x.count_cuotas_mora,
            x.fecha_venc_mas_actual,
            x.fecha_pago_cuota_2,
            x.fecha_pago_cuota_3,
            x.fecha_pago_cuota_4,
            x.fecha_pago_cuota_5,
        ),
        axis=1,
    )

    df_cobranzas_agg["lag_1_perfil_cobranzas_clasico_step_1"] = df_cobranzas_agg.groupby(["cod_alumno"])["perfil_cobranzas_clasico_step_1"].shift(1)
    df_cobranzas_agg["lag_2_perfil_cobranzas_clasico_step_1"] = df_cobranzas_agg.groupby(["cod_alumno"])["perfil_cobranzas_clasico_step_1"].shift(2)

    df_cobranzas_agg["perfil_cobranzas_clasico"] = df_cobranzas_agg.apply(
        lambda x: GetPerfilMorosoFinal(
            x.perfil_cobranzas_clasico_step_1,
            x.lag_1_perfil_cobranzas_clasico_step_1,
            x.lag_2_perfil_cobranzas_clasico_step_1,
        ),
        axis=1,
    )

    df_cobranzas_final = df_cobranzas_agg.drop(columns=["count_cuotas",
       "count_cuotas_mora", "count_cuotas_pago", "avg_flag_pago",
       "fecha_pago_cuota_2", "fecha_pago_cuota_3", "fecha_pago_cuota_4", "fecha_pago_cuota_5", "fecha_venc_mas_actual",
       "perfil_mora_actual", "perfil_pago_actual", "perfil_tipo_pago",
       "peso_mora_actual", "peso_pago_actual", "peso_tipo_pago_actual",
       "lag_1_peso_mora", "lag_2_peso_mora", "lag_3_peso_mora",
       "lag_1_peso_pago", "lag_2_peso_pago", "lag_3_peso_pago",
       "lag_1_tipo_pago", "lag_2_tipo_pago", "lag_3_tipo_pago",
       "peso_mora_avg_3_ult_periodos_anteriores", "peso_pago_avg_3_ult_periodos_anteriores", "peso_tipo_pago_avg_3_ult_periodos_anteriores",
       "peso_mora_avg_4_ult_periodos", "peso_pago_avg_4_ult_periodos", "peso_tipo_pago_avg_4_ult_periodos",
       "perfil_cobranzas_clasico_step_1", "lag_1_perfil_cobranzas_clasico_step_1", "lag_2_perfil_cobranzas_clasico_step_1"])

    df_cobranzas_final = df_cobranzas_final.loc[df_cobranzas_final['periodo'] == return_periodo]

    df_cobranzas_final = df_cobranzas_final.sort_values(by=["cod_alumno", "periodo"], ignore_index=True)

    return df_cobranzas_final

def processing_file_perfil_morosidad_date(
        df: pd.DataFrame,
        list_files: list,
        dict_peso_mora: dict,
        dict_peso_pago: dict,
        dict_peso_tipo_pago: dict,
        fecha_corte: str
    ) -> pd.DataFrame:
    """
    Procesa los datos de cobranzas para calcular perfiles de morosidad y pago.

    Esta función realiza cálculos detallados sobre el comportamiento de pago de los alumnos,
    incluyendo perfiles de morosidad actual, histórico y acumulado. También calcula
    perfiles de pago basados en el momento en que se realizan los pagos en relación
    al fin del ciclo académico.

    Args:
        df (pd.DataFrame): DataFrame con los datos de cobranzas.
        list_files (list): Lista de archivos adicionales para concatenar con df.
        dict_peso_mora (dict): Diccionario que mapea perfiles de mora a pesos numéricos.
        dict_peso_pago (dict): Diccionario que mapea perfiles de pago a pesos numéricos.
        dict_peso_tipo_pago (dict): Diccionario que mapea los tipos de perfiles de pago a pesos numéricos.
        fecha_corte (str): Fecha especifica para la cual se deben devolver los resultados.

    Returns:
        pd.DataFrame: DataFrame con los perfiles de morosidad y pago calculados,
                      incluyendo métricas actuales, de los últimos 3 periodos y acumuladas.

    Note:
        Esta función realiza múltiples transformaciones y cálculos, incluyendo:
        - Filtrado de datos relevantes.
        - Cálculo de días de mora y flags de pago.
        - Agregación de datos a nivel de alumno y periodo.
        - Cálculo de ratios de mora y pago.
        - Asignación de perfiles de mora y pago.
        - Cálculo de métricas históricas y acumuladas.
    """
    df_cobranzas = df.copy()

    reverse_dict_peso_mora = dict(map(reversed, dict_peso_mora.items()))
    reverse_dict_peso_pago = dict(map(reversed, dict_peso_pago.items()))
    reverse_dict_peso_tipo_pago = dict(map(reversed, dict_peso_tipo_pago.items()))

    if list_files:
        df_cobranzas = pd.concat(
            [pd.read_parquet(file) for file in list_files] + [df_cobranzas],
            ignore_index=True,
        )

    # Filtrado para obtener
    # * Periodos Regulares
    # * Cuotas 2, 3, 4, 5
    # * Estados que no sean ANULADO, NO_DETERMINADO
    df_cobranzas = df_cobranzas.loc[
        (~df_cobranzas["periodo"].astype(str).str.endswith("00"))
        & (df_cobranzas["nro_cuota"].isin([2, 3, 4, 5]))
        & (~df_cobranzas["estado"].isin(["ANULADO", "NO_DETERMINADO"]))
    ]

    # delta de dias de pago entre los documentos
    df_cobranzas.loc[:, "delta_dias_pago"] = (
        df_cobranzas["fecha_pago_filtro"].dt.normalize()
        - df_cobranzas["fecha_vencimiento"].dt.normalize()
    ).dt.days

    # calculamos los dias de mora
    df_cobranzas.loc[:, "dias_mora"] = np.where(
        df_cobranzas["fecha_pago_filtro"].isna(),
        (
            df_cobranzas["fecha_corte"].dt.normalize()
            - df_cobranzas["fecha_vencimiento"].dt.normalize()
        ).dt.days,
        df_cobranzas["delta_dias_pago"],
    )

    # Si los dias de mora son mayores a 15 se obtienen los dias de mas
    df_cobranzas.loc[:, "dias_mora"] = np.where(
        df_cobranzas["dias_mora"] < 7, 0, df_cobranzas["dias_mora"] - 6
    )

    # Obtiendo la fecha de vencimiento mas reciente de cada alumno
    df_cobranzas["fecha_vencimiento_mas_reciente"] = df_cobranzas.groupby(
        by=["periodo", "cod_alumno", "fecha_corte"]
    )["fecha_vencimiento"].transform("max")

    # flag mora
    df_cobranzas.loc[:, "flag_mora"] = np.where(df_cobranzas["dias_mora"] > 0, 1, 0)

    # flag pago
    df_cobranzas.loc[:, "flag_pago"] = np.where(
        df_cobranzas["fecha_pago_filtro"].notna(), 1, 0
    )

    # flag para determinar si el pago se realizo despues de la fecha de vencimiento mas reciente
    df_cobranzas["flag_pago_dsps_fec_venc_max"] = np.where(
        df_cobranzas["fecha_pago_filtro"].isna(),
        np.nan,
        np.where(
            df_cobranzas["fecha_pago_filtro"] <= df_cobranzas["fecha_vencimiento_mas_reciente"],
            0,
            1,
        ),
    )

    # Al final del periodo, días que demora en pagar la cuota 2.
    df_cobranzas["flag_cuota2"] = (df_cobranzas["nro_cuota"] == 2).astype(int)
    df_cobranzas["dias_pago_cuota_2"] = np.where(
        df_cobranzas["flag_cuota2"] == 1, df_cobranzas["delta_dias_pago"], np.nan
    )
    df_cobranzas["fecha_pago_cuota_2"] = np.where(
        df_cobranzas["flag_cuota2"] == 1,
        df_cobranzas["fecha_pago_filtro"],
        np.datetime64("NaT"),
    )
    df_cobranzas["fecha_venc_cuota_2"] = np.where(
        df_cobranzas["flag_cuota2"] == 1,
        df_cobranzas["fecha_vencimiento"],
        np.datetime64("NaT"),
    )

    # Al final del periodo, días que demora en pagar la cuota 3.
    df_cobranzas["flag_cuota3"] = (df_cobranzas["nro_cuota"] == 3).astype(int)
    df_cobranzas["dias_pago_cuota_3"] = np.where(
        df_cobranzas["flag_cuota3"] == 1, df_cobranzas["delta_dias_pago"], np.nan
    )
    df_cobranzas["fecha_pago_cuota_3"] = np.where(
        df_cobranzas["flag_cuota3"] == 1,
        df_cobranzas["fecha_pago_filtro"],
        np.datetime64("NaT"),
    )
    df_cobranzas["fecha_venc_cuota_3"] = np.where(
        df_cobranzas["flag_cuota3"] == 1,
        df_cobranzas["fecha_vencimiento"],
        np.datetime64("NaT"),
    )

    # Al final del periodo, días que demora en pagar la cuota 4.
    df_cobranzas["flag_cuota4"] = (df_cobranzas["nro_cuota"] == 4).astype(int)
    df_cobranzas["dias_pago_cuota_4"] = np.where(
        df_cobranzas["flag_cuota4"] == 1, df_cobranzas["delta_dias_pago"], np.nan
    )
    df_cobranzas["fecha_pago_cuota_4"] = np.where(
        df_cobranzas["flag_cuota4"] == 1,
        df_cobranzas["fecha_pago_filtro"],
        np.datetime64("NaT"),
    )
    df_cobranzas["fecha_venc_cuota_4"] = np.where(
        df_cobranzas["flag_cuota4"] == 1,
        df_cobranzas["fecha_vencimiento"],
        np.datetime64("NaT"),
    )

    # Al final del periodo, días que demora en pagar la cuota 5.
    df_cobranzas["flag_cuota5"] = (df_cobranzas["nro_cuota"] == 5).astype(int)
    df_cobranzas["dias_pago_cuota_5"] = np.where(
        df_cobranzas["flag_cuota5"] == 1, df_cobranzas["delta_dias_pago"], np.nan
    )
    df_cobranzas["fecha_pago_cuota_5"] = np.where(
        df_cobranzas["flag_cuota5"] == 1,
        df_cobranzas["fecha_pago_filtro"],
        np.datetime64("NaT"),
    )
    df_cobranzas["fecha_venc_cuota_5"] = np.where(
        df_cobranzas["flag_cuota5"] == 1,
        df_cobranzas["fecha_vencimiento"],
        np.datetime64("NaT"),
    )

    # Obteniendo la fecha de vencimiento mas reciente
    df_cobranzas["fecha_venc_mas_actual"] = (
        df_cobranzas["fecha_venc_cuota_5"]
        .combine_first(df_cobranzas["fecha_venc_cuota_4"])
        .combine_first(df_cobranzas["fecha_venc_cuota_3"])
        .combine_first(df_cobranzas["fecha_venc_cuota_2"])
    )

    df_cobranzas_agg = df_cobranzas.groupby(
        by=["cod_alumno", "periodo", "fecha_corte"], as_index=False, dropna=False
    ).agg(
        count_cuotas=("cod_documento", "count"),
        count_cuotas_mora=("flag_mora", "sum"),
        count_cuotas_pago=("flag_pago", "sum"),
        avg_flag_pago=("flag_pago_dsps_fec_venc_max", "mean"),  # reg o extr
        fecha_pago_cuota_2 = ("fecha_pago_cuota_2", "max"), # fecha de pago de la cuota 2
        fecha_pago_cuota_3 = ("fecha_pago_cuota_3", "max"), # fecha de pago de la cuota 3
        fecha_pago_cuota_4 = ("fecha_pago_cuota_4", "max"), # fecha de pago de la cuota 4
        fecha_pago_cuota_5 = ("fecha_pago_cuota_5", "max"), # fecha de pago de la cuota 5
        fecha_venc_mas_actual = ("fecha_venc_mas_actual", "max"), # fecha de vencimiento mas reciente
    )

    df_cobranzas_agg = df_cobranzas_agg.sort_values(by=["cod_alumno", "periodo"], ignore_index=True)

    df_cobranzas_agg["perfil_mora_actual"] = df_cobranzas_agg["count_cuotas_mora"].map(reverse_dict_peso_mora)
    df_cobranzas_agg["perfil_pago_actual"] = df_cobranzas_agg["count_cuotas_pago"].map(reverse_dict_peso_pago)

    df_cobranzas_agg["perfil_tipo_pago"] = np.where(df_cobranzas_agg["avg_flag_pago"] >= 0.5, "EXT", "REG")

    df_cobranzas_agg["peso_mora_actual"] = df_cobranzas_agg["perfil_mora_actual"].map(dict_peso_mora)
    df_cobranzas_agg["peso_pago_actual"] = df_cobranzas_agg["perfil_pago_actual"].map(dict_peso_pago)
    df_cobranzas_agg["peso_tipo_pago_actual"] = df_cobranzas_agg["perfil_tipo_pago"].map(dict_peso_tipo_pago)

    df_cobranzas_agg["lag_1_peso_mora"] = df_cobranzas_agg.groupby(["cod_alumno"])["peso_mora_actual"].shift(1)
    df_cobranzas_agg["lag_2_peso_mora"] = df_cobranzas_agg.groupby(["cod_alumno"])["peso_mora_actual"].shift(2)
    df_cobranzas_agg["lag_3_peso_mora"] = df_cobranzas_agg.groupby(["cod_alumno"])["peso_mora_actual"].shift(3)

    df_cobranzas_agg["lag_1_peso_pago"] = df_cobranzas_agg.groupby(["cod_alumno"])["peso_pago_actual"].shift(1)
    df_cobranzas_agg["lag_2_peso_pago"] = df_cobranzas_agg.groupby(["cod_alumno"])["peso_pago_actual"].shift(2)
    df_cobranzas_agg["lag_3_peso_pago"] = df_cobranzas_agg.groupby(["cod_alumno"])["peso_pago_actual"].shift(3)

    df_cobranzas_agg["lag_1_tipo_pago"] = df_cobranzas_agg.groupby(["cod_alumno"])["peso_tipo_pago_actual"].shift(1)
    df_cobranzas_agg["lag_2_tipo_pago"] = df_cobranzas_agg.groupby(["cod_alumno"])["peso_tipo_pago_actual"].shift(2)
    df_cobranzas_agg["lag_3_tipo_pago"] = df_cobranzas_agg.groupby(["cod_alumno"])["peso_tipo_pago_actual"].shift(3)

    df_cobranzas_agg["peso_mora_avg_3_ult_periodos_anteriores"] = df_cobranzas_agg[["lag_1_peso_mora", "lag_2_peso_mora", "lag_3_peso_mora"]].mean(axis=1, skipna=True).round()
    df_cobranzas_agg["peso_pago_avg_3_ult_periodos_anteriores"] = df_cobranzas_agg[["lag_1_peso_pago", "lag_2_peso_pago", "lag_3_peso_pago"]].mean(axis=1, skipna=True).round()
    df_cobranzas_agg["peso_tipo_pago_avg_3_ult_periodos_anteriores"] = df_cobranzas_agg[["lag_1_tipo_pago", "lag_2_tipo_pago", "lag_3_tipo_pago"]].mean(axis=1, skipna=True).round()

    df_cobranzas_agg["peso_mora_avg_4_ult_periodos"] = df_cobranzas_agg[["peso_mora_actual", "lag_1_peso_mora", "lag_2_peso_mora", "lag_3_peso_mora"]].mean(axis=1, skipna=True).round()
    df_cobranzas_agg["peso_pago_avg_4_ult_periodos"] = df_cobranzas_agg[["peso_pago_actual", "lag_1_peso_pago", "lag_2_peso_pago", "lag_3_peso_pago"]].mean(axis=1, skipna=True).round()
    df_cobranzas_agg["peso_tipo_pago_avg_4_ult_periodos"] = df_cobranzas_agg[["peso_tipo_pago_actual", "lag_1_tipo_pago", "lag_2_tipo_pago", "lag_3_tipo_pago"]].mean(axis=1, skipna=True).round()

    df_cobranzas_agg["perfil_mora_3_ult_periodos_anteriores"] = df_cobranzas_agg["peso_mora_avg_3_ult_periodos_anteriores"].map(reverse_dict_peso_mora)
    df_cobranzas_agg["perfil_pago_3_ult_periodos_anteriores"] = df_cobranzas_agg["peso_pago_avg_3_ult_periodos_anteriores"].map(reverse_dict_peso_pago)
    df_cobranzas_agg["perfil_tipo_pago_3_ult_periodos_anteriores"] = df_cobranzas_agg["peso_tipo_pago_avg_3_ult_periodos_anteriores"].map(reverse_dict_peso_tipo_pago)

    df_cobranzas_agg["perfil_mora_4_ult_periodos"] = df_cobranzas_agg["peso_mora_avg_4_ult_periodos"].map(reverse_dict_peso_mora)
    df_cobranzas_agg["perfil_pago_4_ult_periodos"] = df_cobranzas_agg["peso_pago_avg_4_ult_periodos"].map(reverse_dict_peso_pago)
    df_cobranzas_agg["perfil_tipo_pago_4_ult_periodos"] = df_cobranzas_agg["peso_tipo_pago_avg_4_ult_periodos"].map(reverse_dict_peso_tipo_pago)

    df_cobranzas_agg["perfil_cobranzas_3_ult_periodos_anteriores"] = df_cobranzas_agg["perfil_mora_3_ult_periodos_anteriores"] + "_" + df_cobranzas_agg["perfil_pago_3_ult_periodos_anteriores"] + "_" + df_cobranzas_agg["perfil_tipo_pago_3_ult_periodos_anteriores"]
    df_cobranzas_agg["perfil_cobranzas_4_ult_periodos"] = df_cobranzas_agg["perfil_mora_4_ult_periodos"] + "_" + df_cobranzas_agg["perfil_pago_4_ult_periodos"] + "_" + df_cobranzas_agg["perfil_tipo_pago_4_ult_periodos"]
    df_cobranzas_agg["perfil_cobranzas_actual"] = df_cobranzas_agg["perfil_mora_actual"] + "_" + df_cobranzas_agg["perfil_pago_actual"] + "_" + df_cobranzas_agg["perfil_tipo_pago"]

    df_cobranzas_agg["perfil_cobranzas_clasico_step_1"] = df_cobranzas_agg.apply(
        lambda x: GetPerfilMoroso(
            x.count_cuotas_mora,
            x.fecha_venc_mas_actual,
            x.fecha_pago_cuota_2,
            x.fecha_pago_cuota_3,
            x.fecha_pago_cuota_4,
            x.fecha_pago_cuota_5,
        ),
        axis=1,
    )

    df_cobranzas_agg["lag_1_perfil_cobranzas_clasico_step_1"] = df_cobranzas_agg.groupby(["cod_alumno"])["perfil_cobranzas_clasico_step_1"].shift(1)
    df_cobranzas_agg["lag_2_perfil_cobranzas_clasico_step_1"] = df_cobranzas_agg.groupby(["cod_alumno"])["perfil_cobranzas_clasico_step_1"].shift(2)

    df_cobranzas_agg["perfil_cobranzas_clasico"] = df_cobranzas_agg.apply(
        lambda x: GetPerfilMorosoFinal(
            x.perfil_cobranzas_clasico_step_1,
            x.lag_1_perfil_cobranzas_clasico_step_1,
            x.lag_2_perfil_cobranzas_clasico_step_1,
        ),
        axis=1,
    )

    df_cobranzas_final = df_cobranzas_agg.drop(columns=["count_cuotas",
       "count_cuotas_mora", "count_cuotas_pago", "avg_flag_pago",
       "fecha_pago_cuota_2", "fecha_pago_cuota_3", "fecha_pago_cuota_4", "fecha_pago_cuota_5", "fecha_venc_mas_actual",
       "perfil_mora_actual", "perfil_pago_actual", "perfil_tipo_pago",
       "peso_mora_actual", "peso_pago_actual", "peso_tipo_pago_actual",
       "lag_1_peso_mora", "lag_2_peso_mora", "lag_3_peso_mora",
       "lag_1_peso_pago", "lag_2_peso_pago", "lag_3_peso_pago",
       "lag_1_tipo_pago", "lag_2_tipo_pago", "lag_3_tipo_pago",
       "peso_mora_avg_3_ult_periodos_anteriores", "peso_pago_avg_3_ult_periodos_anteriores", "peso_tipo_pago_avg_3_ult_periodos_anteriores",
       "peso_mora_avg_4_ult_periodos", "peso_pago_avg_4_ult_periodos", "peso_tipo_pago_avg_4_ult_periodos",
       "perfil_cobranzas_clasico_step_1", "lag_1_perfil_cobranzas_clasico_step_1", "lag_2_perfil_cobranzas_clasico_step_1"])

    fecha_formateada = f"{fecha_corte[:4]}-{fecha_corte[4:6]}-{fecha_corte[6:]}"

    df_cobranzas_final = df_cobranzas_final.loc[df_cobranzas_final['fecha_corte'] == fecha_formateada]

    df_cobranzas_final = df_cobranzas_final.sort_values(by=["cod_alumno", "periodo"], ignore_index=True)

    return df_cobranzas_final
    

def next_cuota_tuition(df: pd.DataFrame, doc_level_eop_path: str, file_name: str) -> pd.DataFrame:
    """
    Calculo de la siguiente cuota.
    Esta cuota no sigue la validacion de fechas de corte, solo se usan para fines de
    construccion del modelo de cobranzas.
    """

    df_ = df.copy().reset_index(drop=True)
    
    doc_level_silver_full_path = os.path.join(doc_level_eop_path, file_name)
    doc_level_silver_df = load_parquet(doc_level_silver_full_path)

    doc_level_silver_df = doc_level_silver_df.loc[doc_level_silver_df["nro_cuota"] > 0] # solo cuotas del 1 hacia adelante
    doc_level_silver_df = doc_level_silver_df[["cod_alumno", "nro_cuota", "fecha_vencimiento", "fecha_pago"]] # estos campos deben estar presentes en la capa silver
    doc_level_silver_df.columns = ["cod_alumno", "next_cuota", "next_cuota_fv", "next_cuota_fp"]

    doc_level_silver_df.loc[:, "next_cuota_delta_dias"] = (doc_level_silver_df["next_cuota_fp"].dt.normalize() - doc_level_silver_df["next_cuota_fv"].dt.normalize()).dt.days
    doc_level_silver_df.loc[:, "next_cuota_flag_pago_7d"] = np.where(doc_level_silver_df["next_cuota_delta_dias"]<=6, 1, 0)

    doc_level_silver_df["next_cuota_fv"] = pd.to_datetime(doc_level_silver_df["next_cuota_fv"].astype(str).str[:10])
    df_["fecha_corte"] = pd.to_datetime(df_["fecha_corte"].astype(str).str[:10])

    # ordenamos para hacer el merge asof
    doc_level_silver_df = doc_level_silver_df.sort_values(by=["next_cuota_fv", "cod_alumno"])
    df_ = df_.sort_values(by=["fecha_corte", "cod_alumno"])

    # hacer el merge asof
    df_ = pd.merge_asof(df_, doc_level_silver_df, left_on=["fecha_corte"], right_on=["next_cuota_fv"], by=["cod_alumno"], direction="forward", allow_exact_matches=False)

    return df_

def GetPerfilMoroso(
    cant_moras, fecha_venc_5, fecha_pago_2, fecha_pago_3, fecha_pago_4, fecha_pago_5
):
    if str(cant_moras).lower() in ["nan", "none", "nat"]:
        perfil = "ND"
    elif (
        (fecha_pago_2 >= fecha_venc_5)
        & (fecha_pago_3 >= fecha_venc_5)
        & (fecha_pago_4 >= fecha_venc_5)
        & (fecha_pago_5 >= fecha_venc_5)
    ):
        perfil = "PERFIL 2"
    elif cant_moras == 0:
        perfil = "PERFIL 1"
    elif cant_moras == 1:
        perfil = "PERFIL 3"
    elif cant_moras == 2:
        perfil = "PERFIL 4"
    elif cant_moras > 2:
        perfil = "PERFIL 5"
    else:
        perfil = "ND"
    return perfil

def GetPerfilMorosoFinal(pm_actual, pm_ant, pm_trasant):
    if (pm_actual == "PERFIL 2") | (pm_ant == "PERFIL 2") | (pm_trasant == "PERFIL 2"):
        pm_final = "PERFIL 2"
    elif (pm_actual == "ND") & (pm_ant == "ND") & (pm_trasant == "ND"):
        pm_final = "ND"
    else:
        try:
            peso_pm_actual = dict_pesos_perfiles[pm_actual]
        except:
            peso_pm_actual = 0
        try:
            peso_pm_ant = dict_pesos_perfiles[pm_ant]
        except:
            peso_pm_ant = 0
        try:
            peso_pm_trasant = dict_pesos_perfiles[pm_trasant]
        except:
            peso_pm_trasant = 0
        if peso_pm_ant == 0:
            peso_pm_ant = max(peso_pm_actual, peso_pm_trasant)
        if peso_pm_trasant == 0:
            peso_pm_trasant = max(peso_pm_actual, peso_pm_ant)
        peso_final = peso_pm_actual + peso_pm_ant + peso_pm_trasant
        if peso_final <= 3:
            pm_final = "PERFIL 1"
        elif peso_final <= 7:
            pm_final = "PERFIL 3"
        elif peso_final <= 11:
            pm_final = "PERFIL 4"
        elif peso_final > 11:
            pm_final = "PERFIL 5"
        else:
            pm_final = "ND"
    return pm_final


dict_pesos_perfiles = {
    "PERFIL 1": 1,
    "PERFIL 3": 2,
    "PERFIL 4": 3,
    "PERFIL 5": 4,
}
