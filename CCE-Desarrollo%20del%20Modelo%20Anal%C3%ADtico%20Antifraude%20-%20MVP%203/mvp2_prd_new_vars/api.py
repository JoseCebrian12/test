from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import OneHotEncoder
from fastapi import FastAPI, Request
import pandas as pd
import numpy as np
import unicodedata
import pickle
import redis
import dotenv
import json
import os
from sqlalchemy import create_engine, Table, MetaData, insert
import psycopg2
import time
from unidecode import unidecode
from datetime import datetime, timedelta
import pytz

db_params = {
    "dbname": "coemcas",
    "user": "coemcas",
    "password": "C03$CMa5$2099",
    # 'host': '10.201.4.25',
    "host": "localhost",
    "port": "5432",
}

# connection_open = psycopg2.connect(**db_params)
# cursor_open = connection_open.cursor()

current_directory = os.path.dirname(os.path.abspath(__file__))
model_name = "mvp_1_model_90_opt_new_vars_250924.pickle"
model_path = os.path.join(current_directory, model_name)
codificaciones_path = os.path.join(current_directory, "Codificaciones.xlsx")

app = FastAPI()
model: IsolationForest = pickle.load(open(model_path, "rb"))
codificaciones = pd.read_excel(codificaciones_path, sheet_name="codificaciones")
codificaciones["Code"] = codificaciones["Code"].str.strip()
codificaciones_dict = {}
for list_value in codificaciones["List"].unique():
    codificaciones_dict[list_value] = {}
    for index, row in codificaciones[codificaciones["List"] == list_value].iterrows():
        codificaciones_dict[list_value][row["Code"]] = row["Value"]

# TODO: use these variables next time the model is trained
# variables_todo = ['creditorParticipant_ratio', 'creditorParticipant_currency_ratio',
#               'creditorParticipant_channel_ratio',
#               'creditorParticipant_responseCode_ratio',
#                 'creditorParticipant_debtorParticipant_ratio', 'creditorParticipant_Weekday_ratio', 'creditorParticipant_time_interval_ratio', 'creditorParticipant_creditorCCI_ratio', 'currency_ratio', 'currency_channel_ratio', 'currency_responseCode_ratio', 'currency_debtorParticipant_ratio', 'currency_Weekday_ratio', 'currency_time_interval_ratio', 'currency_creditorCCI_ratio', 'channel_ratio', 'channel_responseCode_ratio', 'channel_debtorParticipant_ratio', 'channel_Weekday_ratio', 'channel_time_interval_ratio', 'channel_creditorCCI_ratio', 'responseCode_ratio', 'responseCode_debtorParticipant_ratio', 'responseCode_Weekday_ratio', 'responseCode_time_interval_ratio', 'responseCode_creditorCCI_ratio', 'debtorParticipant_ratio', 'debtorParticipant_Weekday_ratio', 'debtorParticipant_time_interval_ratio', 'debtorParticipant_creditorCCI_ratio', 'Weekday_ratio', 'Weekday_time_interval_ratio', 'Weekday_creditorCCI_ratio', 'time_interval_ratio', 'time_interval_creditorCCI_ratio', 'creditorCCI_ratio', 'hourSin', 'hourCos', 'dayOfYearSin', 'dayOfYearCos', 'dayOfMonthSin', 'dayOfMonthCos', 'dayOfWeekSin', 'dayOfWeekCos','debtorParticipant_bcp','debtorParticipant_interbank','debtorParticipant_citibank','debtorParticipant_scotiabank','debtorParticipant_bbva','debtorParticipant_banco_de_la_nacion','debtorParticipant_comercio','debtorParticipant_banco_pichincha','debtorParticipant_banbif','debtorParticipant_crediscotia_financiera','debtorParticipant_mi_banco','debtorParticipant_gnb','debtorParticipant_banco_falabella','debtorParticipant_banco_ripley','debtorParticipant_alfin_banco_s.a.','debtorParticipant_financiera_oh','debtorParticipant_financiera_efectiva','debtorParticipant_caja_piura','debtorParticipant_caja_trujillo','debtorParticipant_caja_arequipa','debtorParticipant_caja_sullana','debtorParticipant_caja_cusco','debtorParticipant_caja_huancayo','debtorParticipant_caja_ica','debtorParticipant_invalid','creditorParticipant_bcp','creditorParticipant_interbank','creditorParticipant_citibank','creditorParticipant_scotiabank','creditorParticipant_bbva','creditorParticipant_banco_de_la_nacion','creditorParticipant_comercio','creditorParticipant_banco_pichincha','creditorParticipant_banbif','creditorParticipant_crediscotia_financiera','creditorParticipant_mi_banco','creditorParticipant_gnb','creditorParticipant_banco_falabella','creditorParticipant_banco_ripley','creditorParticipant_alfin_banco_s.a.','creditorParticipant_financiera_oh','creditorParticipant_financiera_efectiva','creditorParticipant_caja_piura','creditorParticipant_caja_trujillo','creditorParticipant_caja_arequipa','creditorParticipant_caja_sullana','creditorParticipant_caja_cusco','creditorParticipant_caja_huancayo','creditorParticipant_caja_ica','creditorParticipant_invalid', 'currency_soles', 'channel_banca_movil', 'channel_invalid', 'channel_web', 'responseCode_rejected']

variables = [
    "creditorParticipant_ratio",
    "creditorParticipant_currency_ratio",
    "creditorParticipant_channel_ratio",
    "creditorParticipant_responseCode_ratio",
    "creditorParticipant_debtorParticipant_ratio",
    "creditorParticipant_Weekday_ratio",
    "creditorParticipant_time_interval_ratio",
    "creditorParticipant_creditorCCI_ratio",
    "currency_ratio",
    "currency_channel_ratio",
    "currency_responseCode_ratio",
    "currency_debtorParticipant_ratio",
    "currency_Weekday_ratio",
    "currency_time_interval_ratio",
    "currency_creditorCCI_ratio",
    "channel_ratio",
    "channel_responseCode_ratio",
    "channel_debtorParticipant_ratio",
    "channel_Weekday_ratio",
    "channel_time_interval_ratio",
    "channel_creditorCCI_ratio",
    "responseCode_ratio",
    "responseCode_debtorParticipant_ratio",
    "responseCode_Weekday_ratio",
    "responseCode_time_interval_ratio",
    "responseCode_creditorCCI_ratio",
    "debtorParticipant_ratio",
    "debtorParticipant_Weekday_ratio",
    "debtorParticipant_time_interval_ratio",
    "debtorParticipant_creditorCCI_ratio",
    "Weekday_ratio",
    "Weekday_time_interval_ratio",
    "Weekday_creditorCCI_ratio",
    "time_interval_ratio",
    "time_interval_creditorCCI_ratio",
    "creditorCCI_ratio",
    "hourSin",
    "hourCos",
    "dayOfYearSin",
    "dayOfYearCos",
    "dayOfMonthSin",
    "dayOfMonthCos",
    "dayOfWeekSin",
    "dayOfWeekCos",
    "debtorParticipant_bcp",
    "debtorParticipant_interbank",
    "debtorParticipant_citibank",
    "debtorParticipant_scotiabank",
    "debtorParticipant_bbva",
    "debtorParticipant_banco_de_la_nacion",
    "debtorParticipant_comercio",
    "debtorParticipant_banco_pichincha",
    "debtorParticipant_banbif",
    "debtorParticipant_crediscotia_financiera",
    "debtorParticipant_mi_banco",
    "debtorParticipant_gnb",
    "debtorParticipant_banco_falabella",
    "debtorParticipant_banco_ripley",
    "debtorParticipant_alfin_banco_s.a.",
    "debtorParticipant_financiera_oh",
    "debtorParticipant_financiera_efectiva",
    "debtorParticipant_caja_piura",
    "debtorParticipant_caja_trujillo",
    "debtorParticipant_caja_arequipa",
    "debtorParticipant_caja_sullana",
    "debtorParticipant_caja_cusco",
    "debtorParticipant_caja_huancayo",
    "debtorParticipant_caja_ica",
    "debtorParticipant_invalid",
    "creditorParticipant_bcp",
    "creditorParticipant_interbank",
    "creditorParticipant_citibank",
    "creditorParticipant_scotiabank",
    "creditorParticipant_bbva",
    "creditorParticipant_banco_de_la_nacion",
    "creditorParticipant_comercio",
    "creditorParticipant_banco_pichincha",
    "creditorParticipant_banbif",
    "creditorParticipant_crediscotia_financiera",
    "creditorParticipant_mi_banco",
    "creditorParticipant_gnb",
    "creditorParticipant_banco_falabella",
    "creditorParticipant_banco_ripley",
    "creditorParticipant_alfin_banco_s.a.",
    "creditorParticipant_financiera_oh",
    "creditorParticipant_financiera_efectiva",
    "creditorParticipant_caja_piura",
    "creditorParticipant_caja_trujillo",
    "creditorParticipant_caja_arequipa",
    "creditorParticipant_caja_sullana",
    "creditorParticipant_caja_cusco",
    "creditorParticipant_caja_huancayo",
    "creditorParticipant_caja_ica",
    "creditorParticipant_invalid",
    "currency_soles",
    "channel_banca_movil",
    "channel_invalid",
    "channel_web",
    "responseCode_rejected",
]

new_columns = [
    "f1d",
    "f7d",
    "f30d",
    "f90d",
    "f1d_to_creditor",
    "f7d_to_creditor",
    "f30d_to_creditor",
    "f90d_to_creditor",
    "unique_debtors_past_1d",
    "unique_debtors_past_7d",
    "unique_debtors_past_30d",
    "unique_debtors_past_90d",
    "prop_invalid_1d",
    "prop_banca_movil_1d",
    "prop_web_1d",
    "prop_atm_1d",
    "prop_invalid_7d",
    "prop_banca_movil_7d",
    "prop_web_7d",
    "prop_atm_7d",
    "prop_invalid_30d",
    "prop_banca_movil_30d",
    "prop_web_30d",
    "prop_atm_30d",
    "prop_invalid_90d",
    "prop_banca_movil_90d",
    "prop_web_90d",
    "prop_atm_90d",
]
new_ratios = [
    "creditorParticipant_creditorId_ratio",
    "currency_creditorId_ratio",
    "channel_creditorId_ratio",
    "responseCode_creditorId_ratio",
    "debtorParticipant_creditorId_ratio",
    "Weekday_creditorId_ratio",
    "time_interval_creditorId_ratio",
    "creditorCCI_creditorId_ratio",
    "creditorId_ratio",
]
variables += new_columns + new_ratios

# Load environment variables from .env file
dotenv.load_dotenv()

redis_host = os.getenv("REDIS_HOST")
redis_port = os.getenv("REDIS_PORT")
redis_name = os.getenv("REDIS_NAME")
redis_table = os.getenv("REDIS_TABLE")
redis_password = os.getenv("REDIS_PASSWORD")

conexion = redis.StrictRedis(
    host=redis_host, port=redis_port, password=redis_password, decode_responses=True
)


def decode_av4(av4_df):
    decode_column_list = {
        "debtorParticipantCode": "participants",
        "debtorParticipant": "participants",
        "creditorParticipantCode": "participants",
        "creditorParticipant": "participants",
        "transactionType": "transaction_type",
        "currency": "currency",
        "channel": "channel",
        "responseCode": "response_code",
        "reasonCode": "reason_code",
    }

    # if "creditorParticipant" in av4_df.columns or "debtorParticipant" in av4_df.columns:
    #     av4_df = av4_df.rename(
    #         columns={
    #             "debtorParticipant": "debtorParticipantCode",
    #             "creditorParticipant": "creditorParticipantCode",
    #         }
    #     )
    # for column, type_ in decode_column_list.items():
    #     av4_df[column] = av4_df[column].apply(
    #         lambda x: (
    #             codificaciones_dict[type_][x] if x in codificaciones_dict[type_] else x
    #         )
    #     )
    for column, type_ in decode_column_list.items():
        if column in av4_df.columns:
            av4_df["temp"] = av4_df[column].map(codificaciones_dict[type_])
            av4_df["temp"] = av4_df["temp"].fillna("invalid")
            av4_df[column] = av4_df["temp"]
            av4_df.drop("temp", axis=1, inplace=True)
    av4_df = av4_df.rename(
        columns={
            "debtorParticipantCode": "debtorParticipant",
            "creditorParticipantCode": "creditorParticipant",
        }
    )

    return av4_df


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
    # print(len(df["hash_id"].unique()))
    for days in freq_days:
        # print(days)
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


def data_preprocessing(av4_df: pd.DataFrame) -> pd.DataFrame:
    vars_to_ohe = [
        "debtorParticipant",
        "creditorParticipant",
        "transactionType",
        "currency",
        "channel",
        "responseCode",
    ]  # 3

    # vars_to_feature_engineer
    av4_df.rename(
        columns={
            "creationDate": "creationDate_stage",
            "creationTime": "creationTime_stage",
        },
        inplace=True,
    )

    av4_df["creationDate_stage"] = pd.to_datetime(av4_df["creationDate_stage"])
    av4_df["creationTime_stage"] = (
        av4_df["creationTime_stage"].astype(str).str.replace(":", "", regex=False)
    )
    av4_df["creationTime"] = pd.to_datetime(
        av4_df["creationTime_stage"], format="%H%M%S"
    ).dt.time

    # Creación de características cíclicas
    def create_cyclic_features(df):
        df["hourSin"] = np.sin(
            2 * np.pi * df["creationTime"].apply(lambda x: x.hour) / 24.0
        )
        df["hourCos"] = np.cos(
            2 * np.pi * df["creationTime"].apply(lambda x: x.hour) / 24.0
        )
        df["dayOfYearSin"] = np.sin(
            2 * np.pi * df["creationDate_stage"].dt.dayofyear / 365.0
        )
        df["dayOfYearCos"] = np.cos(
            2 * np.pi * df["creationDate_stage"].dt.dayofyear / 365.0
        )
        df["dayOfMonthSin"] = np.sin(2 * np.pi * df["creationDate_stage"].dt.day / 31.0)
        df["dayOfMonthCos"] = np.cos(2 * np.pi * df["creationDate_stage"].dt.day / 31.0)
        df["dayOfWeekSin"] = np.sin(
            2 * np.pi * df["creationDate_stage"].dt.weekday / 7.0
        )
        df["dayOfWeekCos"] = np.cos(
            2 * np.pi * df["creationDate_stage"].dt.weekday / 7.0
        )
        df["monthSin"] = np.sin(2 * np.pi * df["creationDate_stage"].dt.month / 12.0)
        df["monthCos"] = np.cos(2 * np.pi * df["creationDate_stage"].dt.month / 12.0)
        return df

    vars_to_feature_engineer = ["creationTime"]  # 2

    av4_df = create_cyclic_features(av4_df)

    # Eliminación de columnas intermedias
    av4_df = av4_df.drop(
        columns=["creationDate_stage", "creationTime_stage", *vars_to_feature_engineer]
    )

    for column in vars_to_ohe:
        av4_df[column] = (
            av4_df[column]
            .astype(str)
            .apply(unidecode)
            .str.replace(" ", "_", regex=False)
            .str.lower()
        )

    ohe = OneHotEncoder(sparse=False, handle_unknown="ignore")
    ohe.fit(av4_df[vars_to_ohe])
    # print(ohe.categories_)
    av4_df_stage = pd.DataFrame(
        ohe.transform(av4_df[vars_to_ohe]),
        columns=ohe.get_feature_names_out(vars_to_ohe),
    )
    av4_df = pd.concat([av4_df.drop(vars_to_ohe, axis=1), av4_df_stage], axis=1)
    # print(av4_df)
    # vars_to_ohe
    # av4_df["currency_soles"] = av4_df["currency"].apply(lambda x: 1 if x == "Soles" else 0)
    # av4_df["responseCode_rejected"] = av4_df["responseCode"].apply(lambda x: 1 if x == "Rejected" else 0)

    # def quitar_tildes_y_espacios(x: str) -> str:
    #     x = x.lower()
    #     x = unicodedata.normalize("NFKD", x).encode("ASCII", "ignore").decode("utf-8")
    #     x = x.replace(" ", "_")
    #     return x

    # av4_df[f'debtorParticipant_{quitar_tildes_y_espacios(av4_df["debtorParticipant"].unique()[0])}'] = 1
    # av4_df[f'creditorParticipant_{quitar_tildes_y_espacios(av4_df["creditorParticipant"].unique()[0])}'] = 1

    # av4_df.drop(vars_to_ohe, axis=1, inplace=True)
    # print("XDDD")
    # print(av4_df.columns)
    return av4_df


def categorize_hour(hour):
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
    output_list = []

    for i in range(len(input_list)):
        # Add each element in a list of 1 item
        output_list.append([input_list[i]])

        for j in range(i + 1, len(input_list)):
            # Add combinations of two elements
            output_list.append([input_list[i], input_list[j]])

    return output_list


def Ratio(dd1, output_list, lista):  # revisar la fecha
    creditor = ["debtorId"]
    dd2 = (
        dd1[lista]
        .sort_values(by=["debtorId", "creationDate", "creationTime"])
        .dropna()
        .copy()
    )

    # dd2["count_cci"] = len(dd2)
    dd2["count_cci"] = dd2.groupby(["debtorId"]).cumcount() + 1
    # print(dd2.creditorId)
    # for i in output_list:
    #     dd2["_".join(map(str, i)) + "_cumcount"] = dd2.groupby(creditor + i).cumcount() + 1
    #     dd2["_".join(map(str, i)) + "_ratio"] = dd2[
    #         "_".join(map(str, i)) + "_cumcount"
    #     ] / (len(dd2))
    # print("SIZEEEEE", dd2.shape)
    for i in output_list:
        dd2["_".join(map(str, i)) + "_cumcount"] = (
            dd2.groupby(creditor + i).cumcount() + 1
        )
        dd2["_".join(map(str, i)) + "_ratio"] = (
            dd2["_".join(map(str, i)) + "_cumcount"] / dd2["count_cci"]
        )

    columns_like_ratio = [col for col in dd2.columns if "ratio" in col]
    columns_like_ratio.append("target")

    print(f"DD2 antes: {len(dd2)}")

    dd2 = dd2[columns_like_ratio]
    dd2 = dd2[dd2["target"] == 1]

    print(f"DD2 despues: {len(dd2)}")

    return dd2.drop("target", axis=1)


def get_history_sql(debtor_id, creation_time):
    connection = psycopg2.connect(**db_params)
    try:
        with connection.cursor() as cursor:
            sql_query = f"""
                SELECT * FROM stage_ipf_message
                WHERE debtor_id = '{debtor_id}'
                AND creation_date BETWEEN CURRENT_DATE - interval '90 days' AND CURRENT_DATE
                AND (creation_date < CURRENT_DATE OR (creation_date = CURRENT_DATE AND creation_time <= '{creation_time}'::time));
            """
            # print(sql_query)
            cursor.execute(sql_query)
            results = cursor.fetchall()
            column_names = [desc[0] for desc in cursor.description]
            AV_consolidado = pd.DataFrame(results, columns=column_names)
            # print(AV_consolidado.to_dict(orient="records"))
            # result_values = [tuple(row) for row in results]
            return AV_consolidado

    except Exception as e:
        print(f"An error occurred: {e}")
        return None

    finally:
        connection.close()


def get_history_redis(debtor_id):
    hcols = [
        "debtor_participant_code",
        "creditor_participant_code",
        "creation_date",
        "creation_time",
        "trace",
        "debtor_id",
        "debtor_id_code",
        "transaction_type",
        "channel",
        "instruction_id",
        "response_code",
        "reason_code",
        "creditor_cci",
        "currency",
        "score_datetime",
        "creditor_id",
        "creditor_id_code",
        "same_customer_flag",
        "creditor_credit_card",
    ]

    raw_transactions = conexion.zrange(f"{redis_name}:{redis_table}:{debtor_id}", 0, -1)
    transactions = [json.loads(transaction) for transaction in raw_transactions]
    history = pd.DataFrame(transactions)
    for col in hcols:
        if col not in history.columns:
            history[col] = None
    history = history[hcols]

    return history


def get_historic_vars(debtor_id, creation_time, av4_df):
    history = get_history_redis(debtor_id)
    # for testing only
    # history = get_history_sql(debtor_id, creation_time)

    for col in ["debtor_id", "debtorId"]:
        if col in history.columns:
            history.loc[:, col] = history[col].astype(str)

    print(f"Number of transactions: {len(history)}")
    input_list = [
        "creditorParticipant",
        "currency",
        "channel",
        "responseCode",
        "debtorParticipant",
        "Weekday",
        "time_interval",
        "creditorCCI",
    ]

    # lista += ["creditorId"]
    input_list += ["creditorId"]
    combinations = generate_combinations(input_list)
    # if history.empty:
    #     temp = av4_df.copy()
    #     for i in combinations:
    #         temp["_".join(map(str, i)) + "_ratio"] = 0
    #     columns_like_ratio = [col for col in temp.columns if "ratio" in col]
    #     return temp[columns_like_ratio]

    keep = [
        "debtorId",
        "creditorCCI",
        "creditorParticipant",
        "currency",
        "channel",
        "responseCode",
        "debtorParticipant",
        "creationDate",
        "creationTime",
        "time_interval",
        "Weekday",
        "creditorId",
        "target",
    ]

    cols = {
        "debtor_id": "debtorId",
        "debtor_participant_code": "debtorParticipant",
        "creditor_participant_code": "creditorParticipant",
        "creation_date": "creationDate",
        "creation_time": "creationTime",
        "transaction_type": "transactionType",
        "channel": "channel",
        "response_code": "responseCode",
        "creditor_cci": "creditorCCI",
        "currency": "currency",
        "creditor_id": "creditorId",
    }

    for key in cols.keys():
        if key not in history.columns:
            history[key] = None
    history = history[cols.keys()]
    history.rename(columns=cols, inplace=True)
    # print(history.columns)
    history = decode_av4(history)
    # print(history.head())
    history = history[history["transactionType"] == "Ordinary Transfer"]

    print(f"Ordinary Transfers: {len(history)}")

    history["target"] = 0
    temp = av4_df.copy()
    temp["target"] = 1
    # temp only has one row (transaction we want to predict)
    history = pd.concat([history, temp], axis=0, ignore_index=True)
    print(f"After concat: {len(history)}")
    history["creationDate"] = pd.to_datetime(history["creationDate"])
    history["Weekday"] = (
        history["creationDate"].apply(lambda x: x.weekday()).astype(object)
    )
    history["time_interval"] = history["creationTime"].apply(categorize_hour)
    history["creditorId"] = history["creditorId"].fillna("00000000")
    # history["creditorCCI"] = history["creditorCCI"].fillna("00000000")
    new_cols = []
    new_vars_df = history.copy()
    new_vars_df["creation_date_temp"] = pd.to_datetime(new_vars_df["creationDate"])
    # print(new_vars_df.creation_date_temp)
    new_vars_df, new_cols = create_frequency_features(new_vars_df, new_cols)
    new_vars_df, new_cols = create_frequency_interaction_creditor_id(
        new_vars_df, new_cols
    )
    new_vars_df, new_cols = create_unique_debtors_per_creditor(new_vars_df, new_cols)
    new_vars_df, new_cols = create_frequency_per_channel(new_vars_df, new_cols)
    ratios = Ratio(history, combinations, keep).reset_index(drop=True)
    new_vars_df = new_vars_df[new_vars_df["target"] == 1]
    new_vars_df = new_vars_df[new_cols].reset_index(drop=True)
    return new_vars_df.join(ratios)


def get_current_time():
    # system clock is currently not synchronized with an NTP server. 2 minutes and 15 seconds offset
    return (
        datetime.now(pytz.timezone("America/Lima")) + (timedelta(minutes=2, seconds=15))
    ).strftime("%Y-%m-%d %H:%M:%S")


@app.get("/")
def read_root():
    return "Hello World!!"


@app.post("/fraud")
async def fraud_prediction(request: Request):
    # def fraud_prediction(raw_data):
    body = await request.json()
    # body = raw_data
    raw_data = body["AV4"]
    useful_vars = [
        "debtorParticipantCode",
        "creditorParticipantCode",
        "creationDate",
        "creationTime",
        "debtorId",
        "creditorId",
        "transactionType",
        "creditorCCI",
        "channel",
        "responseCode",
        "currency",
        "sameCustomerFlag",
        "instructionId",
    ]
    data = {key: value for key, value in raw_data.items() if key in useful_vars}
    # print(data)
    if data["sameCustomerFlag"] == "M":
        return {
            "creditorParticipantCode": data["creditorParticipantCode"],
            "time": get_current_time(),
            "pred": 0.0,
        }
    av4_df = pd.DataFrame(data, index=[0])

    av4_df = decode_av4(av4_df)
    creation_time = data["creationTime"]
    creation_time = ":".join(
        [creation_time[i : i + 2] for i in range(0, len(creation_time), 2)]
    )
    # print(creation_time)

    historic_vars = get_historic_vars(data["debtorId"], creation_time, av4_df)
    # print(historic_vars.columns)
    av4_df = data_preprocessing(av4_df)

    av4_df.reset_index(drop=True, inplace=True)
    historic_vars.reset_index(drop=True, inplace=True)

    av4_df = av4_df.join(historic_vars)

    # print(f"Columns: {av4_df.columns}")

    for v in variables:
        if v not in av4_df.columns:
            av4_df[v] = 0
            # print("added this", v)
    # print(av4_df.shape)
    # Find columns in av4_df that are not in variables
    missing_columns = [col for col in av4_df.columns if col not in variables]

    # Print the missing columns
    # print("Columns in av4_df that are not in variables:", missing_columns)
    # print(av4_df.to_dict(orient="records"))
    av4_df = av4_df[variables]

    # print(av4_df.shape)
    av4_df.fillna(-1, inplace=True)
    prediction = model.decision_function(av4_df)
    # prediction2 = model.predict(av4_df)[0]

    # Normalize prediction to a value between 0 and 99
    # TODO: check if this min_value and max_value are correct
    # model_90 (old vars)
    # MODEL_90 (NEW VARS) - CURRENT !
    min_value = -0.093440210278803
    max_value = 0.17621073805914572
    # TODO might need to calculate the min and max reference scale values from training? idk if it is a good practice
    # one alternative would be to use clipping
    # train_anomaly_scores = clf.decision_function(X_train)
    # train_min_score = np.min(train_anomaly_scores)
    # train_max_score = np.max(train_anomaly_scores)

    # 0 means it has the highest probability of being normal, and 99 means it has the lowest probability of being normal
    normalized_prediction = (max_value - prediction) / (max_value - min_value) * 99
    # we clip the normalized_prediction if it falls out of the range
    normalized_prediction = normalized_prediction.clip(min=0.0, max=99.0)
    normalized_prediction = normalized_prediction.round(8)

    return {
        # "data": av4_df.to_dict(orient="records")[0],
        "creditorParticipantCode": data["creditorParticipantCode"],
        "time": get_current_time(),
        "pred": normalized_prediction[0],
    }


# data = {
#    "AV4":{
#       "debtorParticipantCode": "0003",
#       "creditorParticipantCode": "0002",
#       "creationDate": "20240917",
#       "creationTime": "154857",
#       "debtorId": "09639807",
#       "creditorId": "56163198",
#       "transactionType": "325",
#       "creditorCCI": "",
#       "channel": "91",
#       "responseCode": "00",
#       "currency": "604",
#       "sameCustomerFlag": "O",
#       "instructionId": "2024100208012800028115343658"
#    }
# }

# print(fraud_prediction(data))
