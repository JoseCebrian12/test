import numpy as np
import pandas as pd


def processing_file_feature_level(df_retiro: pd.DataFrame, df_base_matricula: pd.DataFrame, features_dict: dict = None) -> pd.DataFrame:
    '''
    Calculo de los features que se encuentran en la capa gold.
    Estos calculos se encuentran a nivel de PERIODO-COD_ALUMNO-FECHA_CORTE.
    '''

    # PROCESAMOS LOS RETIROS DE FORMA HISTORICA
    df_ret = df_retiro.copy().reset_index(drop=True)
    df_ret['contador'] = 1
    
    df_ret = df_ret.groupby(['periodo', 'cod_alumno'], as_index=False, dropna=False).agg(
        flag_retiro_ciclo = ('contador', 'sum'),
    )

    df_ret = df_ret.sort_values(by=['cod_alumno', 'periodo'])
    df_ret.loc[:, 'retiro_ciclo_acum'] = df_ret.groupby(['cod_alumno'])['flag_retiro_ciclo'].transform('cumsum')

    df_ret.loc[:, 'ult_periodo_retiro_ciclo'] = df_ret.groupby(['cod_alumno'])['periodo'].transform(lambda x: x.shift(1))

    df_ret = df_ret.rename({'periodo': 'periodo_retiro_ciclo'}, axis=1)


    # MERGE CON DATA DE MATRICULADOS
    df_matr = df_base_matricula.copy().reset_index(drop=True)
    df_matr = df_matr.sort_values(by=['periodo', 'cod_alumno'])
    df_ret = df_ret.sort_values(by=['periodo_retiro_ciclo', 'cod_alumno'])

    df_matr = pd.merge_asof(df_matr, df_ret, left_on=['periodo'], right_on=['periodo_retiro_ciclo'], by=['cod_alumno'], direction='backward', allow_exact_matches=True)

    # COMPLETAMOS LOS DATOS
    df_matr['flag_retiro_ciclo'] = np.where(df_matr['periodo_retiro_ciclo'] == df_matr['periodo'], 1, 0)
    df_matr['flag_retiro_ciclo'] = df_matr['flag_retiro_ciclo'].fillna(0)
    df_matr['retiro_ciclo_acum'] = df_matr['retiro_ciclo_acum'].fillna(0)
    df_matr = df_matr.drop(['periodo_retiro_ciclo', 'ult_periodo_retiro_ciclo'], axis=1)

    if features_dict is not None:
        df_matr = df_matr[features_dict.keys()]

    return df_matr