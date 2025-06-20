import glob
import math
import os
import re
import shutil

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import unidecode

# fmt: off
def get_all_files(folder_path: str, regex_value: str) -> list[str]:
    '''
    Devuelve una lista con la ruta completa de los archivos presentes en una carpeta.
    Se puede especificar un regex para solo extraer archivos que hagan match.
    '''
    regex_path = os.path.join(folder_path, regex_value)
    try:
        files = glob.glob(regex_path)
        return files
    except:
        files = []
        return files

def get_all_folders(root_folder_path: str) -> list[str]:
    '''
    Devuelve una lista de las carpetas que se encuentran disponible 
    dentro de una carpeta especificada.
    '''
    try:
        folder_list = os.listdir(root_folder_path)
        folder_list = [item for item in folder_list if os.path.isdir(os.path.join(root_folder_path, item))]
        return folder_list
    except:
        folder_list = []
        return folder_list
    
def clean_text(value: str) -> str:
    '''
    Limpia un texto, quitando espacios al final y al inicio, quitando espacios en blanco duplicados.
    Devuelve el valor en mayuscula o un valor nulo (numpy NaN)
    '''
    clean_text_value = re.sub('[ ]+', ' ', unidecode.unidecode(str(value).upper())).strip()
    if (clean_text_value == '') or (clean_text_value == 'NAN'):
        return np.nan
    else:
        return clean_text_value
    
def clean_number(value: str, format: str = 'float') -> float | int:
    '''
    Limpia una cadena de texto que se supone debería ser un valor numérico.
    '''
    if format not in ['float', 'int']:
        raise ValueError('format parameter must be "float" or "int".')
    
    # Incluir el signo negativo (-) en la lista de caracteres permitidos.
    clean_text_number_value = re.sub('[^-0-9.,]', '', unidecode.unidecode(str(value).upper())).strip()

    try:
        if format == 'float':
            return float(clean_text_number_value)
        else:
            return int(clean_text_number_value)
    except ValueError:
        try:
            # Convertir comas en puntos si es necesario y reintentar la conversión.
            if format == 'float':
                return float(clean_text_number_value.replace(',', '.'))
            else:
                return int(float(clean_text_number_value.replace(',', '.')))
        except ValueError:
            return np.nan
        
def create_parquet_from_df(df: pd.DataFrame, output_folder: str, filename: str) -> None:
    '''
    Crea un archivo en formato parquet especificando un folder destino y el nombre final del archivo.
    El parámetro "filename" debe contener la extensión ".parquet".
    Si el "output_folder" no existe, se creará el destino.
    '''
    # Asegurarse de que el directorio de salida exista
    os.makedirs(output_folder, exist_ok=True)
    
    # Crear la tabla y escribir el archivo parquet
    tbl = pa.Table.from_pandas(df)
    pq.write_table(tbl, os.path.join(output_folder, filename))
    
def consolidate_parquet_files_from_folder(folder_path: str, parquet_regex_id: str, output_folder_path: str, filename: str, parts: int = None) -> None:
    '''
    Consolida todos los parquet files que sigan la regla del parametro "parquet_regex_id".
    La consolidacion se realizara dentro de un nuevo folder "output_folder_path". El nombre resultante
    del archivo es "filename" (no incluye la extension .parquet) por lo que debe ser especificada
    '''
    if os.path.exists(folder_path):

        if not os.path.exists(output_folder_path):
            os.makedirs(output_folder_path)

        regex_path = os.path.join(folder_path, parquet_regex_id)
        files_to_merge = glob.glob(regex_path)

        if files_to_merge:
            
            if parts is not None:
                sublist_size = math.ceil(len(files_to_merge) / parts)
                sublists = [files_to_merge[i:i+sublist_size] for i in range(0, len(files_to_merge), sublist_size)]
                ix_file = 1
            else:
                sublists = [files_to_merge]

            for sublist in sublists:
                append_result = []
                filename_ = filename

                for file in sublist:
                    dataset = pq.read_table(file)
                    if dataset.num_rows > 0:
                        append_result.append(dataset)

                if parts is not None:
                    filename_ = f'{filename_.split(".")[0]}_part{ix_file}.parquet'
                    ix_file += 1

                try:
                    output_full_path = os.path.join(output_folder_path, filename_)
                    merged_table = pa.concat_tables(append_result, promote_options='permissive')
                    pq.write_table(merged_table, output_full_path)
                except:
                    print("No se pudo concatenar los archivos, estan vacios")
    return None

def refactor_folder_file(path_to_be_refactor: str, flag_folder: int = 1) -> None:
    '''
    Elimina un folder especificado y crea la ruta de forma inmediata.
    '''
    if os.path.exists(path_to_be_refactor):
        if flag_folder == 1:
            shutil.rmtree(path_to_be_refactor)
            os.makedirs(path_to_be_refactor)
        elif flag_folder == 0:
            os.remove(path_to_be_refactor)
    return None

def load_csv(file_path: str) -> None:
    '''
    Cargar los csv procesados para UPC, primero intenta con el UTF8 y luego con el LATIN1.
    '''
    try:
        return pd.read_csv(file_path, sep='|', encoding='utf8', on_bad_lines='skip', dtype=str)
    except:
        return pd.read_csv(file_path, sep='|', encoding='latin1', on_bad_lines='skip', dtype=str)
    
def load_parquet(file_path: str) -> None:
    '''
    Lectura de parquet files
    '''
    return pd.read_parquet(file_path)

def check_columns(df: pd.DataFrame, cols_needed: set) -> bool:
    '''
    Determina si tenemos las columnas necesarias para poder procesar el archivo
    '''
    columns = set(df.columns)
    intersection = cols_needed.intersection(columns)
    
    if cols_needed == intersection:
        return True
    else:
        return False
    
def delta_periodos(df: pd.DataFrame, var_per_1: str, var_per_2: str, max_periodos: int) -> pd.Series:
    df_ = df[[var_per_1, var_per_2]].copy()
    df_['anio_1'] = pd.to_numeric(df_[var_per_1].astype(str).str[:4], errors='coerce')
    df_['anio_2'] = pd.to_numeric(df_[var_per_2].astype(str).str[:4], errors='coerce')
    df_['per_1'] = pd.to_numeric(df_[var_per_1].astype(str).str[4:6], errors='coerce')
    df_['per_2'] = pd.to_numeric(df_[var_per_2].astype(str).str[4:6], errors='coerce')
    df_['result'] = (df_['anio_1'] - df_['anio_2']) * max_periodos + (df_['per_1'] - df_['per_2'])
    return df_['result']
