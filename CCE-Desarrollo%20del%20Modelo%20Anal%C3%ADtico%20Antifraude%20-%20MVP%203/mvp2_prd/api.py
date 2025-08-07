from sklearn.ensemble import IsolationForest
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

db_params = {
        'dbname': 'coemcas',
        'user': 'coemcas',
        'password': 'C03$CMa5$2099',
        'host': '10.201.4.25',
        'port': '5432'
    }

connection_open = psycopg2.connect(**db_params)
cursor_open = connection_open.cursor()

current_directory = os.path.dirname(os.path.abspath(__file__))
model_path = os.path.join(current_directory, "mvp_1_model_90_opt_200K.pickle")
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
        "creditorParticipantCode": "participants",
        "transactionType": "transaction_type",
        "currency": "currency",
        "channel": "channel",
        "responseCode": "response_code",
    }
    if "creditorParticipant" in av4_df.columns or "debtorParticipant" in av4_df.columns:
        av4_df = av4_df.rename(
            columns={
                "debtorParticipant": "debtorParticipantCode",
                "creditorParticipant": "creditorParticipantCode",
            }
        )
    for column, type in decode_column_list.items():
        av4_df[column] = av4_df[column].apply(
            lambda x: (
                codificaciones_dict[type][x] if x in codificaciones_dict[type] else x
            )
        )

    av4_df = av4_df.rename(
        columns={
            "debtorParticipantCode": "debtorParticipant",
            "creditorParticipantCode": "creditorParticipant",
        }
    )

    return av4_df

def data_preprocessing(av4_df: pd.DataFrame) -> pd.DataFrame:
    vars_to_feature_engineer = ["creationDate", "creationTime"]  # 2
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

    av4_df["creationDate"] = pd.to_datetime(av4_df["creationDate_stage"]).dt.strftime(
        "%Y%m%d"
    )
    av4_df["creationTime_stage"] = (
        av4_df["creationTime_stage"].astype("string").str.replace(":", "")
    )
    av4_df["creationTime"] = pd.to_datetime(av4_df["creationTime_stage"], format="%H%M%S").dt.time

    av4_df.drop(["creationDate_stage", "creationTime_stage"], axis=1, inplace=True)
    av4_df["hourSin"] = av4_df["creationTime"].apply(lambda x: np.sin(2 * np.pi * x.hour / 24.0))
    av4_df["hourCos"] = av4_df["creationTime"].apply(lambda x: np.cos(2 * np.pi * x.hour / 24.0))
    av4_df["creationDate"] = pd.to_datetime(av4_df["creationDate"])

    av4_df["dayOfYearSin"] = av4_df["creationDate"].apply(lambda x: np.sin(2 * np.pi * x.timetuple().tm_yday / 365.0))
    av4_df["dayOfYearCos"] = av4_df["creationDate"].apply(lambda x: np.cos(2 * np.pi * x.timetuple().tm_yday / 365.0))
    av4_df["dayOfMonthSin"] = av4_df["creationDate"].apply(lambda x: np.sin(2 * np.pi * x.day / 31.0))
    av4_df["dayOfMonthCos"] = av4_df["creationDate"].apply(lambda x: np.cos(2 * np.pi * x.day / 31.0))
    av4_df["dayOfWeekSin"] = av4_df["creationDate"].apply(lambda x: np.sin(2 * np.pi * x.weekday() / 7.0))
    av4_df["dayOfWeekCos"] = av4_df["creationDate"].apply(lambda x: np.cos(2 * np.pi * x.weekday() / 7.0))
    av4_df["monthSin"] = av4_df["creationDate"].apply(lambda x: np.sin(2 * np.pi * x.month / 12.0))
    av4_df["monthCos"] = av4_df["creationDate"].apply(lambda x: np.cos(2 * np.pi * x.month / 12.0))

    av4_df.drop(vars_to_feature_engineer, axis=1, inplace=True)

    # vars_to_ohe
    av4_df["currency_soles"] = av4_df["currency"].apply(lambda x: 1 if x == "Soles" else 0)
    av4_df["responseCode_rejected"] = av4_df["responseCode"].apply(lambda x: 1 if x == "Rejected" else 0)

    def quitar_tildes_y_espacios(x: str) -> str:
        x = x.lower()
        x = unicodedata.normalize("NFKD", x).encode("ASCII", "ignore").decode("utf-8")
        x = x.replace(" ", "_")
        return x

    av4_df[f'debtorParticipant_{quitar_tildes_y_espacios(av4_df["debtorParticipant"].unique()[0])}'] = 1
    av4_df[f'creditorParticipant_{quitar_tildes_y_espacios(av4_df["creditorParticipant"].unique()[0])}'] = 1

    av4_df.drop(vars_to_ohe, axis=1, inplace=True)

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
    dd2["count_cci"] = len(dd2)

    for i in output_list:
        dd2["_".join(map(str, i)) + "_cumcount"] = dd2.groupby(creditor + i).cumcount() + 1
        dd2["_".join(map(str, i)) + "_ratio"] = dd2[
            "_".join(map(str, i)) + "_cumcount"
        ] / len(dd2)

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
            cursor.execute(sql_query)
            results = cursor.fetchall()
            column_names = [desc[0] for desc in cursor.description]
            AV_consolidado = pd.DataFrame(results, columns=column_names)
            
            #result_values = [tuple(row) for row in results]
            return AV_consolidado

    except Exception as e:
        print(f"An error occurred: {e}")
        return None
        
    finally:
        connection.close()

def get_history(debtor_id):
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
        "creditor_id",
        "creditor_id_code",
        "reason_code",
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

def get_ratios_sql(debtor_id, creation_time, av4_df):
    
    history = get_history_sql(debtor_id, creation_time)
#    history = get_history(debtor_id)
    print(history.columns.tolist())

    if "debtor_id" in history.columns:
        history.loc[:, "debtor_id"] = history["debtor_id"].astype(str)
    if "debtorId" in history.columns:
        history.loc[:, "debtorId"] = history["debtorId"].astype(str)

    print(f"Number of transactions: {len(history)}")
    combinations = generate_combinations(
        [
            "creditorParticipant",
            "currency",
            "channel",
            "responseCode",
            "debtorParticipant",
            "Weekday",
            "time_interval",
            "creditorCCI",
        ]
    )
    if history.empty:
        temp = av4_df.copy()
        for i in combinations:
            temp["_".join(map(str, i)) + "_ratio"] = 0
        columns_like_ratio = [col for col in temp.columns if "ratio" in col]
        return temp[columns_like_ratio]

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
    }

    for key in cols.keys():
        if key not in history.columns:
            history[key] = None
    history = history[cols.keys()]
    history.rename(columns=cols, inplace=True)

    history = decode_av4(history)
    history = history[history["transactionType"] == "Ordinary Transfer"]

    print(f"Ordinary Transfers: {len(history)}")

    history["target"] = 0
    temp = av4_df.copy()
    temp["target"] = 1
    #temp only has one row (transaction we want to predict)
    history = pd.concat([history, temp], axis=0, ignore_index=True)
    print(f"After concat: {len(history)}")
    history["creationDate"] = pd.to_datetime(history["creationDate"])
    history["Weekday"] = (
        history["creationDate"].apply(lambda x: x.weekday()).astype(object)
    )
    history["time_interval"] = history["creationTime"].apply(categorize_hour)

    return Ratio(history, combinations, keep)


@app.get("/")
def read_root():
    return "Hello World!!"

@app.post("/fraud")
async def fraud_prediction(request: Request):
    body = await request.json()

    data = body["AV4"]
    av4_df = pd.DataFrame(data, index=[0])

    av4_df = decode_av4(av4_df)

    creation_time = data["creationTime"]
    creation_time = ":".join([creation_time[i:i+2] for i in range(0, len(creation_time), 2)])
    #    print(creation_time)

    ratios = get_ratios_sql(data["debtorId"], creation_time, av4_df)

    av4_df = data_preprocessing(av4_df)

    av4_df.reset_index(drop=True, inplace=True)
    ratios.reset_index(drop=True, inplace=True)

    av4_df = av4_df.join(ratios)

    # print(f"Columns: {av4_df.columns}")

    for v in variables:
        if v not in av4_df.columns:
            av4_df[v] = 0

    av4_df.fillna(-1, inplace=True)
    prediction = model.decision_function(av4_df.drop(columns=["debtorId"])[variables])[0]
    # prediction2 = model.predict(av4_df.drop(columns=["debtorId"])[variables])[0]

    # Normalize prediction to a value between 0 and 99
    # TODO: check if this min_value and max_value are correct
    #model_7
    # min_value = -0.10404229802050258 # -0.11774102106942208
    # max_value = 0.1814456418857644 #0.1602303765911105
    #model_90
    min_value = -0.1268560699089064 # -0.11774102106942208
    max_value = 0.1737664642244749 #0.1602303765911105
    # min_value = -0.11774102106942208
    # max_value = 0.1602303765911105
    # TODO might need to calculate the min and max reference scale values from training? idk if it is a good practice
    # one alternative would be to use clipping
    # train_anomaly_scores = clf.decision_function(X_train)
    # train_min_score = np.min(train_anomaly_scores)
    # train_max_score = np.max(train_anomaly_scores)

    # 0 means it has the highest probability of being normal, and 99 means it has the lowest probability of being normal
    normalized_prediction = (max_value - prediction) / (max_value - min_value) * 99
    # we clip the normalized_prediction if it falls out of the range
    normalized_prediction = max(0, min(normalized_prediction, 99))
    normalized_prediction = round(normalized_prediction, 8)

    return {"data": av4_df.to_dict(orient="records")[0], "pred": normalized_prediction}