import numpy as np
import pandas as pd

from .funcs import load_parquet

# fmt: off
def impute_near_value(df: pd.DataFrame, by_group: list, var_to_impute: str, var_order: str) -> pd.Series:
    '''
    Imputar un valor en funcion de un grupo. La imputacion es en funcion del valor mas cercano,
    anterior o posterior.
    Si siguen quedando campos vacios se completa con el valor mas comun por el parametro "by_group".
    El order de los valores en el parametro "by_group" indicara la forma en la que se ordenan los datos.
    '''

    df = df[by_group + [var_to_impute] + [var_order]].copy().reset_index(drop=True)
    df.loc[:, 'order'] = 1
    df.loc[:, 'order'] = df['order'].cumsum()
    
    vars_to_oder = by_group + [var_order]
    order_prev_val = [True] * len(by_group) + [True]

    df = df.sort_values(by=vars_to_oder, ascending= order_prev_val)
    df[f'{var_to_impute}_prev'] = df.groupby(by_group)[var_to_impute].transform(lambda x: x.bfill())
    df[f'{var_to_impute}_next'] = df.groupby(by_group)[var_to_impute].transform(lambda x: x.ffill())
    df[f'{var_to_impute}_imputed'] = df[var_to_impute].combine_first(df[f'{var_to_impute}_prev']).combine_first(df[f'{var_to_impute}_next'])
    
    df = df.sort_values(by=['order'])

    return df[f'{var_to_impute}_imputed']

def merge_matricula(df: pd.DataFrame, base_matricula_path: str, periodo: int = None) -> pd.DataFrame:
    base_matricula = load_parquet(base_matricula_path)
    
    if periodo is not None:
        base_matricula = base_matricula[base_matricula['periodo'] == periodo]

    base_matricula = base_matricula.merge(df, how='left', on=['cod_alumno', 'periodo'])

    return base_matricula

def formatting(df: pd.DataFrame, format_dict: dict = None) -> pd.DataFrame:
    df_ = df.copy().reset_index(drop=True)
    df_.replace([np.inf, -np.inf], np.nan, inplace=True)

    if format_dict is not None:
        if len(format_dict) > 0:
            for key in format_dict.keys():
                if format_dict[key] != 'skip':
                    try:
                        df_.loc[:, key] = df_[key].fillna(format_dict[key])
                    except Exception as e:
                        print(f'Error de formato {key} / {e}')
                        pass
    return df_

def merge_calendar(df: pd.DataFrame, calendar_path: str) -> pd.DataFrame:
    '''
    Cruce con el calendario general a nivel de fecha de periodo, fecha de corte
    '''
    calendar_df = load_parquet(calendar_path)
    df_ = df.copy().reset_index(drop=True)
    
    df_ = df_.merge(calendar_df, how='left', on=['periodo', 'fecha_corte'])

    return df_
