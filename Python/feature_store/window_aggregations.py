import operator
import time
import warnings
from collections import defaultdict
from typing import Callable

import numpy as np
import pandas as pd

from .general.funcs import load_parquet

warnings.filterwarnings("ignore")

# fmt: off
class WindowAggregation:
    """
    Clase para realizar agregaciones de ventana en un DataFrame.
    """

    def __init__(self, eop_path: str, list_window_agg: list):
        """
        Inicializa la clase con la ruta del archivo y el diccionario de agregaciones de ventana.

        Args:
            eop_path (str): Ruta del archivo parquet.
            list_window_agg (dict): Diccionario de configuraciones de agregaciones de ventana.
        """
        self.dataframe = load_parquet(eop_path)  # Carga el DataFrame desde el archivo parquet
        self.list_window_agg = list_window_agg  # Almacena el diccionario de configuraciones

    def expand_config(self, rolling_window_config: list) -> dict:
        """
        Expande una configuración en un diccionario con listas de columnas 
        agregadas, umbrales y funciones de agregación.

        Args:
            rolling_window_config (dict): Diccionario donde cada clave es una tupla que contiene:
                - base_col (str): Nombre de la columna base.\n
                - thresholds (list of int): Lista de umbrales.\n
                - aggfunc (str): Función de agregación (e.g., "mean", "sum").\n

        Returns:
            dict: Diccionario donde las claves son los nombres de las columnas base
            y los valores son listas que contienen listas con el nombre de la columna
            agregada, el umbral y la función de agregación.
        """
        
        expanded_dict = defaultdict(list)

        for base_col, thresholds, aggfunc in rolling_window_config:
            # Determinar el nombre de la columna agregada basado en la función de agregación
            agg_col_name = "avg" if aggfunc == "mean" else aggfunc
            
            # Crear nombres de columnas agregadas para cada umbral
            for threshold in thresholds:
                col_name = f"{agg_col_name}_{base_col}_{threshold}p_ant"
                expanded_dict[base_col].append([col_name, threshold, aggfunc])

        return expanded_dict

    def window_function(
        self,
        dataframe: pd.DataFrame,
        base_col: str,
        window: int,
        aggregation: Callable,
        partition_col: str = "cod_alumno",
    ) -> pd.Series:
        """
        Aplica una función de agregación de ventana a una columna del DataFrame.

        Args:
            dataframe (pd.DataFrame): DataFrame de entrada.
            base_col (str): Columna base para la agregación.
            window (int): Tamaño de la ventana.
            aggregation (Callable): Función de agregación.
            partition_col (str): Columna para particionar los datos. Por defecto es "cod_alumno".

        Returns:
            pd.Series: Serie con los valores agregados.
        """
        # Especificar la ventana para la agregación
        window_spec = (
            dataframe.groupby(partition_col)[base_col]
            .rolling(window=window, min_periods=1, closed="left")
            .agg(aggregation)
        )

        # Restablecer el índice y llenar valores NaN con 0
        return window_spec.reset_index(level=0, drop=True).fillna(0)

    def create_column_threshold(
        self,
        col: str,
        threshold: int,
        comparison_op_str: str,
    ) -> pd.DataFrame:
        """
        Crea nuevas columnas en el DataFrame basadas en valores umbral.

        Args:
            df (pd.DataFrame): DataFrame de entrada.
            col (str): Nombre de la columna sobre la cual aplicar los umbrales.
            threshold (int): Valor umbral.
            comparison_op_str (str): Operador de comparación en formato string.

        Returns:
            pd.DataFrame: DataFrame con las nuevas columnas agregadas.
        """

        # Diccionario de operadores de comparación
        comparison_operators = {
            "<": operator.lt,
            "<=": operator.le,
            ">": operator.gt,
            ">=": operator.ge,
        }

        # Diccionario de textos de operadores
        operator_texts = {
            "<": "menor",
            "<=": "menor_igual",
            ">": "mayor",
            ">=": "mayor_igual",
        }

        # Obtener la función de comparación correspondiente al operador en formato string
        comparison_op = comparison_operators.get(comparison_op_str)
        if comparison_op is None:
            raise ValueError(f"Operador de comparación no válido: {comparison_op_str}")

        # Obtener el texto del operador
        operator_text = operator_texts.get(comparison_op_str)

        # Crear un nuevo nombre de columna basado en el nombre original, el texto del operador, y el valor del umbral
        col_name = f"{col}_{operator_text}_{threshold}"

        # Aplicar el umbral a la columna y crear la nueva columna
        self.dataframe[col_name] = self.dataframe[col].apply(lambda x: x if comparison_op(x, threshold) else np.nan)


    def get_aggregations(self) -> pd.DataFrame:
        """
        Genera todas las agregaciones de ventana especificadas y las añade al DataFrame original.

        Returns:
            pd.DataFrame: DataFrame con todas las nuevas columnas de agregaciones añadidas.
        """
        # Crear un DataFrame con las columnas 'periodo' y 'cod_alumno'
        dataframe_cod_periodo = self.dataframe[["periodo", "cod_alumno"]].copy()
        dataframe_cod_periodo = dataframe_cod_periodo[~dataframe_cod_periodo['periodo'].astype(str).str.endswith("00")]
        print("Periodos a procesar:", sorted(dataframe_cod_periodo['periodo'].unique()))

        # Creando diccionario de variables
        dict_variables = self.expand_config(rolling_window_config=self.list_window_agg)

        # Iterar sobre cada columna base en el diccionario de agregaciones
        for base_col in dict_variables:

            print(f"Procesando {base_col}")

            # Copiar las columnas necesarias al DataFrame temporal
            dataframe_aggs_temp = self.dataframe[["periodo", "cod_alumno", base_col]].copy()

            # Iterar sobre cada configuración de agregación para la columna base actual
            for list_values in dict_variables[base_col]:

                col_name_agg = list_values[0]
                window = list_values[1]
                aggregation = list_values[2]

                print(f"   Calculo: {col_name_agg}", end=" - ")

                # Medir el tiempo de ejecución de la función de ventana
                start_time = time.time()
                window_values = self.window_function(
                    dataframe=dataframe_aggs_temp,
                    base_col=base_col,
                    window=window,
                    aggregation=aggregation,
                )

                # Añadir la columna agregada al DataFrame temporal
                dataframe_aggs_temp[col_name_agg] = window_values
                end_time = time.time()
                
                elapsed_time = end_time - start_time
                print(f"{elapsed_time:.4f} segundos")

            # Fusionar el DataFrame temporal con el DataFrame de 'periodo' y 'cod_alumno'
            dataframe_cod_periodo = dataframe_cod_periodo.merge(
                dataframe_aggs_temp,
                on=["periodo", "cod_alumno"],
                how="left",
            )

        # Devolver el DataFrame con todas las columnas agregadas, eliminando las columnas base originales
        return dataframe_cod_periodo.drop(columns=list(dict_variables.keys()))
