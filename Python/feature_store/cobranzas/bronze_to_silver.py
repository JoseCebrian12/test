import warnings

import numpy as np
import pandas as pd

from ..general.funcs import clean_number, clean_text

warnings.filterwarnings("ignore")

# fmt: off
def processing_bronze_file_raw_level(df: pd.DataFrame, final_cols: list = None) -> pd.DataFrame:
    df_ = df.copy()

    for col_name in df_.columns:
        df_.loc[:, col_name] = df_[col_name].apply(clean_text)

    df_ = df_.loc[((df_['periodo'].notna()) & (df_['cod_alumno'].notna()))]

    df_.loc[:, 'periodo'] = df_['periodo'].apply(clean_number, format='int')
    df_.loc[:, 'monto'] = df_['monto'].apply(clean_number, format='float')
    df_.loc[:, 'monto_pagado'] = df_['monto_pagado'].apply(clean_number, format='float').fillna(0)
    df_.loc[:, 'nro_cuota'] = df_['nro_cuota'].apply(clean_number, format='float').fillna(0)
    
    df_.loc[:, 'fecha_documento'] = pd.to_datetime(df_['fecha_documento'].astype(str).str[:10], errors='coerce')
    df_.loc[:, 'fecha_vencimiento'] = pd.to_datetime(df_['fecha_vencimiento'].astype(str).str[:10], errors='coerce')
    df_.loc[:, 'fecha_pago'] = pd.to_datetime(df_['fecha_pago'].astype(str).str[:10], errors='coerce')

    df_ = df_.loc[(~df_['tipo_documento'].isin(['XB', 'XD', 'PE']))]
    df_ = df_.loc[(~df_['estado_documento'].isin(['AN']))]
    df_ = df_.loc[(df_['monto'] != 0.0)]
    df_ = df_.loc[(df_['monto'].notna())]
    df_ = df_.loc[~((df_.cod_cobranza.isnull()) & (df_.estado_documento=='CO') & (df_.tipo_documento=='BV'))]
    df_ = df_.loc[~((df_['tipo_documento']=='ND') & (df_['cod_cobranza'].isnull()))].reset_index(drop=True)
    
    # creando nuevas columnas necesarias
    df_.loc[:, 'flag_matricula'] = np.where(df_['matricula']=='SI', 1, 0)
    df_.loc[:, 'flag_tuition'] = np.where(df_['tuition']=='SI', 1, 0)
    df_.loc[:, 'flag_tuition'] = np.where(df_['nro_cuota'] != 0, 1, df_['flag_tuition'])

    # eliminamos duplicados
    df_ = df_.sort_values(by=['cod_alumno','periodo','tipo_documento','cod_documento','estado_documento','fecha_documento','fecha_vencimiento','monto','cod_cobranza','fecha_pago','monto_pagado'])
    df_ = df_.drop_duplicates(subset=['cod_alumno','periodo','tipo_documento','cod_documento','estado_documento','fecha_documento','fecha_vencimiento','monto','cod_cobranza','fecha_pago'], keep='first')

    # solo nos quedamos con las columnas necesarias
    if final_cols is not None:
        df_ = df_[final_cols]

    return df_

def processing_file_doc_level(df: pd.DataFrame, final_cols: list = None) -> pd.DataFrame:
    df_ = df.copy()
    
    flag_df_nulo = 0
    df_ = df_.loc[(df_['fecha_vencimiento'] <= df_['fecha_corte'])]
    
    if df_.shape[0] == 0:
        flag_df_nulo = 1
        try:
            df_ = df_.sample(5000)
        except:
            pass

    df_ = df_.reset_index(drop=True)

    # si la fecha de pago es mayor a la fecha de corte
    # se considera que no ha pagado para esa fecha de corte
    b1 = df_['fecha_pago'] > df_['fecha_corte']
    b2 = df_['cod_cobranza'].isnull()

    df_.loc[:, 'monto_pagado_filtro'] = np.where(b1, 0, df_['monto_pagado'])
    df_.loc[:, 'fecha_pago_filtro'] = np.where(b1 , np.datetime64('NaT'), df_['fecha_pago'])
    df_.loc[:, 'cod_cobranza_filtro'] = np.where(b1 | b2, np.nan, df_['cod_cobranza'])

    b3 = df_['cod_cobranza_filtro'].isna()
    df_.loc[:, 'flag_cod_cobranza_filtro'] = np.where(b3, 0, 1)

    map_dict = {
        'BV': 1,  # Boleta de venta
        'ND': 2,  # Nota de débito
        'NC': 3,  # Nota de crédito
        'LC': 4,  # Letra de cambio
        'FC': 5,  # Factura
    }

    df_.loc[:, 'cod_tipo_documento'] = df_['tipo_documento'].map(map_dict)

    # DOCUMENTOS BV FC y LC
    df_primera_import = df_.loc[df_['cod_tipo_documento'].isin([1, 4, 5])].copy()
    df_primera_import = df_primera_import.groupby(['cod_alumno', 'periodo', 'cod_documento'], as_index=False).agg(
        cod_tipo_documento=('cod_tipo_documento', 'max'),
        cod_tipo_documento_unique=('cod_tipo_documento', 'nunique'),
        monto=('monto', 'max'),
        monto_pagado_filtro=('monto_pagado_filtro', 'sum'),
        flag_cod_cobranza=('flag_cod_cobranza_filtro', 'max'),
        fecha_documento=('fecha_documento', 'max'),
        fecha_vencimiento=('fecha_vencimiento', 'max'),
        fecha_pago=('fecha_pago','max'),
        fecha_pago_filtro=('fecha_pago_filtro', 'max'),
        matricula=('flag_matricula', 'max'),
        tuition=('flag_tuition', 'max'),
        nro_cuota=('nro_cuota', 'max'),
        n_doc=('cod_alumno', 'count'),
        fecha_corte=('fecha_corte', 'max'),
    )

    # NOTAS DE CREDITO
    df_nota_cred = df_.loc[(df_['cod_tipo_documento'] == 3)].copy().reset_index(drop=True)
    df_nota_cred.loc[:, 'cod_documento'] = df_nota_cred['cod_documento_asociado'].str[2:]
    df_nota_cred = df_nota_cred.groupby(['periodo', 'cod_alumno', 'cod_documento'], as_index=False).agg(
            monto_nc=('monto', 'sum'),  # Sumar el monto de las notas de crédito
            n_nc=('cod_documento', 'nunique'),  # Contar el número de documentos únicos
        )
    df_nota_cred = df_nota_cred[['periodo', 'cod_alumno', 'cod_documento', 'monto_nc', 'n_nc']]

    # NOTAS DE DEBITO
    # TBD

    # MERGE
    df_ = pd.merge(left=df_primera_import, right=df_nota_cred, on=['periodo', 'cod_alumno', 'cod_documento'], how='left').reset_index(drop=True)
    df_.loc[:, 'monto_nc'] = df_['monto_nc'].fillna(0)
    df_.loc[:, 'n_nc'] = df_['n_nc'].fillna(0)

    # CALCULO DE ESTADOS
    # Estado 1: No Pagado
    b1 = df_['monto_pagado_filtro'] == 0  # Si el monto pagado es 0 
    b2 = df_['flag_cod_cobranza'] == 0  # Y no hay código de cobranza
    b3 = df_['fecha_pago_filtro'].isnull()  # Y no hay fecha de pago
    b4 = df_['monto_nc'] == 0  # Y no hay nota de crédito aplicada
    b5 = df_['fecha_vencimiento'] <= df_['fecha_corte'] # ya se dio la fecha de vencimiento
    bf = b1 & b2 & b3 & b4 & b5
    df_.loc[:, 'estado_1'] = np.where(bf, 'NO_PAGADO', np.nan)

    # Estado 2: Pago Regular
    b1 = df_['monto_pagado_filtro'] >= df_['monto'] * 0.95 # Si el monto pagado es al menos el 95% del monto total
    b2 = df_['flag_cod_cobranza'] == 1  # Y hay un código de cobranza
    b3 = ~df_['fecha_pago_filtro'].isnull()  # Y hay una fecha de pago
    b4 = df_['monto_nc'] == 0  # Y no hay nota de crédito aplicada
    bf = b1 & b2 & b3 & b4
    df_.loc[:, 'estado_2'] = np.where(bf, 'PAGO_REGULAR', np.nan)
    
    # Estado 3: Anulado
    b1 = df_['monto_nc'] + df_['monto'] <= 0 # Si la suma de la nota de crédito y el monto total es menor o igual a 0, quiere decir que la NC anula el documento
    b2 = df_['monto_nc'] != 0 # existe una NC
    bf = b1 & b2
    df_.loc[:, 'estado_3'] = np.where(bf, 'ANULADO', np.nan)

    # Estado 4: Por cobrar
    b1 = df_['monto_pagado_filtro'] == 0  # Si el monto pagado es 0 
    b2 = df_['flag_cod_cobranza'] == 0  # Y no hay código de cobranza
    b3 = df_['fecha_pago_filtro'].isnull()  # Y no hay fecha de pago
    b4 = df_['monto_nc'] == 0  # Y no hay nota de crédito aplicada
    b5 = df_['fecha_vencimiento'] > df_['fecha_corte'] # la fecha de vencimiento ya ha ocurrido
    bf = b1 & b2 & b3 & b4 & b5
    df_.loc[:, 'estado_4'] = np.where(bf, 'POR_COBRAR', np.nan)

    # Estado 5: Pagado con Nota de Crédito
    b1 = df_['monto_nc'] != 0  # Si hay una nota de crédito aplicada
    b2 = df_['monto_nc'] + df_['monto'] > 0 # Y la suma de la nota de crédito y el monto total es mayor que 0, NO ANULA EL DOCUMENTO EN SU TOTALIDAD
    b3 = ~df_['fecha_pago_filtro'].isnull() # Y hay una fecha de pago validada con la fecha de corte
    b4 = df_['monto_pagado_filtro'] >= df_['monto'] * 0.95 # Si el monto pagado es al menos el 95% del monto total
    bf = b1 & b2 & b3 & b4
    df_.loc[:, 'estado_5'] = np.where(bf, 'PAGO_NC', np.nan)

    # Estado 6: Pagado Parcialmente
    b1 = df_['monto_nc'] == 0  # No existe concepto por notas de credito
    b2 = ~df_['fecha_pago_filtro'].isnull() # Y hay una fecha de pago validada con la fecha de corte
    b3 = df_['monto_pagado_filtro'] < df_['monto'] * 0.95 # Si el monto pagado es menor al 95% del monto total
    bf = b1 & b2 & b3
    df_.loc[:, 'estado_6'] = np.where(bf, 'PAGO_PARCIAL', np.nan)

    # Reemplazar los valores de cadena 'nan' con np.nan en las columnas de estado
    df_.loc[:, 'estado_1'] = df_['estado_1'].replace('nan', np.nan)
    df_.loc[:, 'estado_2'] = df_['estado_2'].replace('nan', np.nan)
    df_.loc[:, 'estado_3'] = df_['estado_3'].replace('nan', np.nan)
    df_.loc[:, 'estado_4'] = df_['estado_4'].replace('nan', np.nan)
    df_.loc[:, 'estado_5'] = df_['estado_5'].replace('nan', np.nan)
    df_.loc[:, 'estado_6'] = df_['estado_6'].replace('nan', np.nan)

    # Combinar todos los estados de pago en una sola columna, manteniendo el primer valor no nulo
    df_.loc[:, 'estado'] = (
        df_['estado_1']
        .combine_first(df_['estado_2'])
        .combine_first(df_['estado_3'])
        .combine_first(df_['estado_4'])
        .combine_first(df_['estado_5'])
        .combine_first(df_['estado_6'])
    )

    # para validar la cantidad de estados
    df_.loc[:, 'n_estados'] = df_[['estado_1', 'estado_2', 'estado_3', 'estado_4', 'estado_5', 'estado_6']].notna().sum(axis=1)

    # Rellenar los valores nulos en la columna de estado con 'ND' (NO DETERMINADO)
    df_.loc[:, 'estado'] = df_['estado'].fillna('NO_DETERMINADO')

    # Eliminar las columnas df_orales de los estados de pago
    df_ = df_.drop(columns=['estado_1', 'estado_2', 'estado_3', 'estado_4', 'estado_5', 'estado_6'])

    # reemplazar campo si el df es nulo
    # devolveremos el df
    if flag_df_nulo == 1:
        df_.loc[:, 'cod_alumno'] = '77991144'
        df_ = df_.sample()

    # solo nos quedamos con las columnas necesarias
    if final_cols is not None:
        df_ = df_[final_cols]

    return df_
