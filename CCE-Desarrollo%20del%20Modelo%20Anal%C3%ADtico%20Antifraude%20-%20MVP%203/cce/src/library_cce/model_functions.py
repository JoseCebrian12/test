import os
import sys
import datetime
from unidecode import unidecode

sys.path.append(
    os.path.dirname(
        os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    )
)
from src.library_cce.model_functions import *

# Standard Libraries
import json
import time
import urllib

# Third-Party Libraries
import pandas as pd
from tqdm import tqdm
from sklearn.ensemble import IsolationForest
import warnings
import pickle
import psycopg2
from sqlalchemy import create_engine

warnings.simplefilter(action="ignore", category=FutureWarning)

# Constants
DETELE_OBJECTS = False
PREDICTION_PIPELINE = True  # Change to False if training model again
NEW_VARIABLES_FLAG = True


# Utils
def print_time_elapsed(process_name, start_time):
    """
    Prints the process name and the elapsed time since the start time.

    Args:
        process_name (str): The name or description of the process being measured.
        start_time (time): The start time of the process.

    Returns:
        None
    """
    print("finished processing: " + process_name)
    elapsed_time = time.time() - start_time
    print(f"Time elapsed: {elapsed_time} seconds")


def timer_decorator(func):
    """
    Decorator function that measures the execution time of the decorated function.

    Args:
        func (function): The function to be decorated.

    Returns:
        function: The wrapper function that measures the execution time.
    """

    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        execution_time = end_time - start_time
        print(
            f"The function '{func.__name__}' took {execution_time:.4f} seconds to execute."
        )
        return result

    return wrapper


def print_dataframe_size(df):
    """
    Prints the shape and memory usage of a DataFrame.

    Args:
        df (pandas.DataFrame): The DataFrame to analyze.

    Returns:
        None
    """
    print(f"shape: {df.shape}")
    # print(f"Memory usage: {sys.getsizeof(df)/1e6} megabytes")
    print(f"Memory usage: {df.memory_usage(index=True).sum()/1e6} megabytes")


def delete_objects(objects, delete_ad_hoc=False):
    """
    Deletes objects from memory and prints the total memory released.

    Args:
        objects (object or list): The object or list of objects to delete.
        delete_ad_hoc (bool, optional): If True, forces deletion even if DETELE_OBJECTS flag is False.
            Defaults to False.

    Returns:
        None
    """
    if DETELE_OBJECTS or delete_ad_hoc:
        total_memory_released = 0

        if isinstance(objects, list):
            for obj in objects:
                memory_released = sys.getsizeof(obj)
                total_memory_released += memory_released
                del obj
        else:
            memory_released = sys.getsizeof(objects)
            total_memory_released += memory_released
            del objects

        print(f"Total memory released: {total_memory_released/1e6} megabytes")


# 1. Ingest and pre-process
def deserialize_msg_data(df, header_cols, serialized_data_col):
    """
    Deserialize the serialized data in a DataFrame column and combine it with the specified header columns.

    Args:
        df (pandas.DataFrame): The DataFrame containing the data.
        header_cols (list): A list of column names for the header data.
        serialized_data_col (str): The name of the column containing the serialized data.

    Returns:
        pandas.DataFrame: The resulting DataFrame with deserialized data combined with the header columns.
    """
    # Extract the necessary columns from the input DataFrame
    df.reset_index(drop=True, inplace=True)
    series_cabecera = df[header_cols]
    serialized_strings = df[serialized_data_col]

    # Decode the hexadecimal strings and deserialize the JSON objects
    decoded_strings = [
        serialized_string.decode("utf-8")
        for serialized_string in tqdm(serialized_strings, total=len(df))
    ]
    deserialized_objects = [
        json.loads(decoded_string)
        for decoded_string in tqdm(decoded_strings, total=len(df))
    ]
    deserialized_series_list = [
        list(deserialized_object.values())[0]
        for deserialized_object in tqdm(deserialized_objects, total=len(df))
    ]

    deserialized_data = pd.DataFrame(deserialized_series_list)

    # Combine the necessary series into a single DataFrame
    df_combined = pd.concat(
        [series_cabecera, deserialized_data], axis=1
    )  # Resulting DataFrame

    return df_combined


def crear_pk_compuesta(df, columnas_pk):
    """
    Creates a composite primary key column in the DataFrame based on the specified columns.

    Args:
        df (pandas.DataFrame): The DataFrame to modify.
        columnas_pk (list): A list of column names to be used as the composite primary key.

    Returns:
        pandas.DataFrame: The modified DataFrame with the composite primary key column added.
    """
    df["pk"] = df[columnas_pk].astype(str).apply("_".join, axis=1)
    return df


def preprocesamiento(df, header_cols, serialized_data_col, pk_cols, data_cols_to_keep):
    """
    Perform preprocessing steps on the input DataFrame, including deserialization, creating a composite primary key,
    and selecting specific columns.

    Args:
        df (pandas.DataFrame): The DataFrame containing the data.
        header_cols (list): A list of column names for the header data.
        serialized_data_col (str): The name of the column containing the serialized data.
        pk_cols (list): A list of column names to be used as the composite primary key.
        data_cols_to_keep (list): A list of column names to keep in the resulting DataFrame.

    Returns:
        pandas.DataFrame: The resulting preprocessed DataFrame with deserialized data, composite primary key,
            and selected columns.
    """
    df_deserialized = deserialize_msg_data(df, header_cols, serialized_data_col)
    df_deserialized = crear_pk_compuesta(df_deserialized, pk_cols)
    df_deserialized = df_deserialized[["pk"] + header_cols + data_cols_to_keep]
    return df_deserialized


def categorize_hour(hour):
    """
    Categorize the given hour into specific time intervals.

    Args:
        hour (int or str): The hour to be categorized. It can be an integer or a string representation of an integer.

    Returns:
        str: The category corresponding to the input hour.

    """
    # Zero-padding if needed
    hour = str(hour).zfill(6)
    # Extract the hour part
    hour = int(hour[:2])
    # Define time interval categories
    if hour >= 0 and hour < 3:
        return "00 to 03"
    elif hour >= 3 and hour < 6:
        return "03 to 06"
    elif hour >= 6 and hour < 9:
        return "06 to 09"
    elif hour >= 9 and hour < 12:
        return "09 to 12"
    elif hour >= 12 and hour < 15:
        return "12 to 15"
    elif hour >= 15 and hour < 18:
        return "15 to 18"
    elif hour >= 18 and hour < 21:
        return "18 to 21"
    else:  # hour >= 21 or hour < 24
        return "21 to 00"


def generate_combinations(input_list):
    """
    Generate all possible combinations of elements from the given input list.

    The function generates combinations by first including each element in a separate list,
    and then combining two elements at a time to form new combinations.

    Args:
        input_list (list): The list of elements from which to generate combinations.

    Returns:
        list: A list containing all possible combinations of elements.
    """

    output_list = []

    for i in range(len(input_list)):
        # Add each element in a list of 1 item
        output_list.append([input_list[i]])

        for j in range(i + 1, len(input_list)):
            # Add combinations of two elements
            output_list.append([input_list[i], input_list[j]])

    return output_list


def Ratio(dd1, output_list, lista, days_back, max_run_id):  # revisar la fecha
    """
    Calculate ratios based on cumulative counts for specified columns in the input DataFrame.

    The function takes a DataFrame `dd1` and calculates ratios for combinations of columns specified in `output_list`.
    The calculation is performed based on cumulative counts of each combination relative to a total count.

    Args:
        dd1 (pandas.DataFrame): The input DataFrame containing the data.
        output_list (list): A list of lists, where each inner list represents a combination of columns for which ratios
                            are to be calculated based on cumulative counts. The columns in each combination should be
                            present in the DataFrame `dd1`.
        lista (list): A list of column names from the DataFrame `dd1` that will be used to extract data.

    Returns:
        pandas.DataFrame: A DataFrame containing the calculated ratios for each combination specified in `output_list`.
    """

    creditor = ["debtorId"]
    dd2_completed = pd.DataFrame()

    total_days = dd1.creationDate.drop_duplicates().sort_values().reset_index(drop=True)

    if PREDICTION_PIPELINE:
        training_days = [total_days.max()]
    else:
        training_days = total_days[total_days >= "2023-04-06"].reset_index(drop=True)

    for i, _day in enumerate(training_days):
        print(f"processing day {i+1} of {len(training_days)}: {_day}")
        dd2 = (
            dd1[
                (dd1.creationDate <= _day)
                & (dd1.creationDate >= _day - pd.Timedelta(days=days_back))
            ][lista]
            .sort_values(by=["debtorId", "creationDate", "creationTime"])
            .dropna()
            .copy()
        )
        dd2["count_cci"] = dd2.groupby(["debtorId"]).cumcount() + 1
        for i in output_list:
            dd2["_".join(map(str, i)) + "_cumcount"] = (
                dd2.groupby(creditor + i).cumcount() + 1
            )
            dd2["_".join(map(str, i)) + "_ratio"] = (
                dd2["_".join(map(str, i)) + "_cumcount"] / dd2["count_cci"]
            )
        print("SIZE DD2 (before)", days_back, dd2.shape)
        if PREDICTION_PIPELINE:
            # dd2 = dd2[dd2.run_id == max(dd2.run_id)]
            dd2 = dd2[dd2.run_id == max_run_id]
        else:
            dd2 = dd2[dd2.creationDate == _day]
        print("SIZE DD2 (after)", days_back, dd2.shape)
        dd2_completed = pd.concat([dd2_completed, dd2], axis=0, ignore_index=False)

    return dd2_completed.filter(like="ratio")


def predict_isolation_forest(input_data, model):
    print(input_data.columns.tolist())
    y_pred = model.predict(input_data)
    y_pred = [1 if pred == -1 else 0 for pred in y_pred]
    y_pred_series = pd.Series(data=y_pred, index=input_data.index).astype("uint8")
    return y_pred_series


def score_isolation_forest(input_data, model, days_back, new_flag):
    print(input_data.columns.tolist())
    y_pred = model.decision_function(input_data)
    if days_back == 7:
        min_value = -0.10404229802050258
        max_value = 0.1814456418857644
    elif days_back == 90:
        if new_flag:
            min_value = -0.093440210278803
            max_value = 0.17621073805914572
        else:  # old
            min_value = -0.08458841514388882
            max_value = 0.16960560199287533
    # norm_y_pred = (y_pred - min_value) / (max_value - min_value) * 99
    norm_y_pred = (max_value - y_pred) / (max_value - min_value) * 99
    # norm_y_pred = max(0, min(norm_y_pred, 99))
    # y_pred = [1 if pred == -1 else 0 for pred in y_pred]
    norm_y_pred = norm_y_pred.clip(min=0.0, max=99.0)
    # y_pred_series = pd.Series(data=y_pred, index=input_data.index).astype("float64")
    y_pred_series = pd.Series(data=norm_y_pred, index=input_data.index).astype(
        "float64"
    )
    y_pred_series = y_pred_series.round(8)
    return y_pred_series


def load_model(file_path, file_name):
    full_file_path = os.path.join(file_path, file_name)
    with open(full_file_path, "rb") as handle:
        loaded_model = pickle.load(handle)
    return loaded_model


# model function
def fit_isolation_forest(input_data, args=None):
    rs = args["random_state"]
    c = args["contamination"]

    model = IsolationForest(random_state=rs, contamination=c, n_jobs=-1)
    model.fit(input_data)

    return model


# conection to database
_driver = "PostgreSQL ANSI(x64)"
_server = "localhost"
_database = "coemcas"
_username = "coemcas"
_password = "C03$CMa5$2099"
_port = "5432"


def get_database_conection():
    connection = psycopg2.connect(
        host=_server, database=_database, user=_username, password=_password, port=_port
    )
    return connection


def get_database_engine():
    quoted_password = urllib.parse.quote_plus(_password)
    engine = create_engine(
        f"postgresql+psycopg2://{_username}:{quoted_password}@{_server}:{_port}/{_database}"
    )
    return engine


def data_ingestion_from_database():
    # Given the first unrun batch it finds, it retrieves all the transactions from the last 7 days (excluding the unrun batch)
    if PREDICTION_PIPELINE:
        sql = """
            WITH run_id_to_process AS (
                SELECT r.run_id, r.run_date, r.run_end_datetime
                FROM fraud_model_run r
                WHERE r.run_start_datetime = (
                    SELECT MIN(r2.run_start_datetime)
                    FROM fraud_model_run r2
                    WHERE COALESCE(r2.run_process_status, 0) = 0
                )
            ),
            debtor_ids AS (
                SELECT DISTINCT m.debtor_id, r.run_id AS current_run_id, r.run_date AS current_run_date, r.run_end_datetime
                FROM run_id_to_process r
                INNER JOIN stage_ipf_message m ON r.run_id = m.run_id
            )
            SELECT DISTINCT ON (m.pk) m.*, d.current_run_id
            FROM stage_ipf_message m 
            INNER JOIN debtor_ids d ON m.debtor_id = d.debtor_id
            WHERE m.run_id = d.current_run_id
            OR (
                m.run_id != d.current_run_id
                AND m.creation_date BETWEEN d.current_run_date - interval '90 days' AND d.current_run_date
                AND (
                    m.creation_date < d.current_run_date 
                    OR (m.creation_date = d.current_run_date AND m.creation_time <= d.run_end_datetime::time)
                )
            );
        """
    else:
        sql = """
            with debtor_ids as (
                select debtor_id, current_creation_date
                from (
                    select 
                        debtor_id, 
                        current_creation_date,
                        row_number() over (partition by current_creation_date order by random()) as rn
                    from (
                        select 
                            distinct m.debtor_id, 
                            m.creation_date as current_creation_date
                        from stage_ipf_message_old m
                        where m.creation_date between '2022-08-30' and '2022-09-02'
                    ) distinct_pairs
                ) numbered_pairs
                where rn <= 6000
            ),
            filtered_rows as (
                select 
                    m.*, 
                    row_number() over (partition by m.pk) as row_num
                from stage_ipf_message_old m
                inner join debtor_ids d on m.debtor_id = d.debtor_id
                where m.creation_date between d.current_creation_date - interval '7 days' and d.current_creation_date
            )
            select *
            from filtered_rows
            where row_num = 1;
        """

    connection = None
    # Establish the database connection
    connection = get_database_conection()
    cursor = connection.cursor()

    print(
        f"EXEC - Started query to get AV_consolidado rows at {datetime.datetime.now()}"
    )
    cursor.execute(sql)
    print(
        f"EXEC - Ended query (cursor.execute) to get AV_consolidado rows at {datetime.datetime.now()}"
    )
    # Obtener los nombres de las columnas
    column_names = [desc[0] for desc in cursor.description]
    # Obtener los resultados como una lista de listas
    results = cursor.fetchall()
    print(
        f"EXEC - Ended query (cursor.fetchall) to get AV_consolidado rows at {datetime.datetime.now()}"
    )
    cursor.close()
    if connection is not None:
        connection.close()
    # Convertir los resultados en un DataFrame
    AV_consolidado = pd.DataFrame(results, columns=column_names)
    print(column_names)
    return AV_consolidado


def load_predictions_to_database(df_fraud_model_predict, mod_version):
    df_fraud_model_predict["model_version"] = mod_version
    engine = get_database_engine()
    tabla_destino = "fraud_model_predict"
    df_fraud_model_predict["debtor_type_of_person"].fillna("-", inplace=True)
    print(
        f"Started saving results to database at {datetime.datetime.now()}: {df_fraud_model_predict.shape[0]} rows"
    )
    df_fraud_model_predict.to_sql(
        tabla_destino, engine, if_exists="append", index=False
    )
    print(f"Ended saving results to database at {datetime.datetime.now()}")


def update_process_status_to_database(run_id, status, process_datetime_column_name):
    connection = None
    connection = get_database_conection()
    cursor = connection.cursor()

    sql = (
        """ UPDATE fraud_model_run SET run_process_status = %s,"""
        + process_datetime_column_name
        + """= %s WHERE run_id = %s"""
    )
    run_process_status = status
    run_process_end_datetime = datetime.datetime.now()
    cursor.execute(sql, (run_process_status, run_process_end_datetime, run_id))

    # cursor.execute(sql,(run_id, run_process_status, run_process_end_datetime))
    connection.commit()
    print(f"Update process status: {cursor.rowcount} rows")
    connection.close()
    if connection is not None:
        connection.close()


def update_process_for_empty_df():
    sql = """select r.run_id 
        from fraud_model_run r 
        where r.run_start_datetime = (select min(r2.run_start_datetime) from fraud_model_run r2 where coalesce(r2.run_process_status,0) = 0)"""
    connection = None
    connection = get_database_conection()
    cursor = connection.cursor()

    cursor.execute(sql)
    run_id = cursor.fetchone()[0]
    print(f"No data for run_id {run_id}, setting status to -1")
    run_process_end_datetime = datetime.datetime.now()
    sql = """ UPDATE fraud_model_run SET run_process_status = %s, run_process_start_datetime = %s WHERE run_id = %s"""
    cursor.execute(sql, (-1, run_process_end_datetime, run_id))
    connection.commit()


# FRECUENCIA
def create_frequency_features(df, new_cols, freq_days=[1, 7, 30, 90]):
    df = df.sort_values(
        by=["debtorId", "creationDate", "creationTime"]
    )  # .dropna().copy()
    df = df.reset_index(drop=True)
    result_df = pd.DataFrame(index=df.index)
    for days in freq_days:
        # result = (df.groupby('debtorId')
        #                 .rolling(window=f'{days}d', on='creation_date_temp')
        #                 .creation_date_temp
        #                 .count().reset_index(drop=True))
        result = (
            df.groupby("debtorId")
            .rolling(window=f"{days}d", on="creation_date_temp")
            .creation_date_temp.count()
        )  # Rolling count without resetting index
        result = result.reset_index(level=0, drop=True)
        new_col = f"f{days}d"
        result_df[new_col] = result.values / days
        new_cols.append(new_col)
    # print(result_df)
    result_df = result_df.reset_index(drop=True)
    df = pd.concat([df, result_df], axis=1)
    return df, new_cols


# CANTIDAD DE OPERACIONES DEL CLIENTE POR DIA Y POR CANAL
def rolling_total_count(group, days):
    return group.rolling(window=f"{days}D").count()


def create_frequency_per_channel(df, new_cols, freq_days=[1, 7, 30, 90]):
    df = df.sort_values(by=["debtorId", "creationDate", "creationTime"])
    df = df.reset_index(drop=True)
    df.set_index("creation_date_temp", inplace=True)
    df["clean_channel"] = (
        df["channel"]
        .astype(str)
        .apply(unidecode)
        .str.replace(" ", "_", regex="False")
        .str.lower()
    )
    channel_types = df["clean_channel"].unique()
    for channel in channel_types:
        df[f"channel_is_{channel}"] = (df["clean_channel"] == channel).astype(int)
    for days in freq_days:
        # Contar el total de eventos en la ventana de días
        total_counts = (
            df.groupby("debtorId")["clean_channel"]
            .apply(lambda group: rolling_total_count(group, days))
            .reset_index(level=0, drop=True)
        )
        for channel in channel_types:
            # Contar eventos por canal en la ventana de días
            channel_counts = (
                df.groupby("debtorId")[f"channel_is_{channel}"]
                .rolling(window=f"{days}D")
                .sum()
                .reset_index(level=0, drop=True)
            )

            # Calcular la proporción para ese canal y ventana de tiempo
            new_col = f"prop_{channel}_{days}d"
            df[new_col] = channel_counts / total_counts
            new_cols.append(new_col)
    for channel in channel_types:
        df.drop(columns=f"channel_is_{channel}", inplace=True)
    df.drop(columns=["clean_channel"], inplace=True)
    df.reset_index(drop=True, inplace=True)
    return df, new_cols


# DNI DESTINO
def create_frequency_interaction_creditor_id(df, new_cols, freq_days=[1, 7, 30, 90]):
    df["creditorId"] = df["creditorId"].fillna("00000000")
    df = df.sort_values(
        by=["debtorId", "creditorId", "creationDate", "creationTime"]
    )  # .dropna().copy()
    df = df.reset_index(drop=True)

    result_df = pd.DataFrame(index=df.index)
    for days in freq_days:
        # result = (df.groupby(['debtorId', 'creditorId'])
        #                     .rolling(window=f'90d', on='creation_date_temp')
        #                     .creation_date_temp
        #                     .count().reset_index(drop=True))
        result = (
            df.groupby(["debtorId", "creditorId"])
            .rolling(window=f"{days}d", on="creation_date_temp")
            .creation_date_temp.count()
        )
        result = result.reset_index(level=[0], drop=True)
        new_col = f"f{days}d_to_creditor"
        result_df[new_col] = result.values / days
        new_cols.append(new_col)
    # print(result_df)
    result_df = result_df.reset_index(drop=True)
    df = pd.concat([df, result_df], axis=1)
    return df, new_cols


# CUENTAS CON ABONOS DE DIFERENTES ORIGENES
def create_unique_debtors_per_creditor(df, new_cols, freq_days=[1, 7, 30, 90]):
    # Sort by CreditorCCI, debtorId, and transaction date to ensure chronological order
    df["creditorId"] = df["creditorId"].fillna("00000000")
    df = df.sort_values(by=["creditorId", "creationDate", "creationTime"])
    df = df.reset_index(drop=True)

    # Create an empty DataFrame to store the results
    result_df = pd.DataFrame(index=df.index)

    # Use a rolling window per creditor to get unique debtorIds
    # result = (df.groupby('creditorId')
    #             .rolling(window=f'{window_days}d', on='creation_date_temp')
    #             .apply(lambda x: x['debtorId'].unique(), raw=False)
    #             )
    df["hash_id"] = pd.factorize(df["debtorId"])[0]
    print(len(df["hash_id"].unique()))
    for days in freq_days:
        print(days)
        # result = df.groupby('creditorId').apply(lambda x: x.rolling(window=f'{days}d', on='creation_date_temp').hash_id.apply(lambda y: y.nunique() ))
        # Reset the index to avoid issues with multi-indexing
        # result = result.reset_index(level=[0], drop=True)
        result = (
            df.groupby("creditorId")
            .rolling(window=f"{days}d", on="creation_date_temp")["hash_id"]
            .apply(lambda y: y.nunique())
        )
        result = result.reset_index(level=[0], drop=True).values

        # Add the result to the original DataFrame
        new_col = f"unique_debtors_past_{days}d"
        result_df[new_col] = result
        new_cols.append(new_col)
        # Concatenate with the original DataFrame
    df = pd.concat([df, result_df], axis=1)
    df.drop(columns=["hash_id"], inplace=True)

    return df, new_cols
