import os
import sys

sys.path.append(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
)
from src.library_cce.model_functions import *

import pandas as pd
from sklearn.preprocessing import OneHotEncoder
import numpy as np
from unidecode import unidecode
import warnings

warnings.simplefilter(action="ignore", category=FutureWarning)


@timer_decorator
def feature_engineering_exp4():
    codificaciones = pd.read_excel(
        r"/home/cnvdba/cce/src/Data/Codificaciones.xlsx", sheet_name="codificaciones"
    )
    codificaciones.Code = codificaciones.Code.str.strip()
    rename_columns = {
        "creation_date": "creationDate",
        "creation_time": "creationTime",
        "creditor_participant_code": "creditorParticipantCode",
        "debtor_participant_code": "debtorParticipantCode",
        "debtor_type_person": "debtorTypeOfPerson",
        "transaction_type": "transactionType",
        "debtor_id": "debtorId",
        "debtor_id_code": "debtorIdCode",
        "creditor_cci": "creditorCCI",
        "creditor_credit_card": "creditorCreditCard",
        "reason_code": "reasonCode",
        "response_code": "responseCode",
        "creditor_id": "creditorId",
        "creditor_id_code": "creditorIdCode",
    }

    dict_column_list = {
        "debtorParticipantCode": "participants",
        "creditorParticipantCode": "participants",
        "transactionType": "transaction_type",
        "currency": "currency",
        "channel": "channel",
        "responseCode": "response_code",
        "reasonCode": "reason_code",
    }

    # Ingestar de bd AV consolidado de run a procesar
    AV_consolidado = data_ingestion_from_database()
    print("SIZE", AV_consolidado.shape)

    if PREDICTION_PIPELINE:
        if pd.isna(AV_consolidado["run_id"].max()):
            update_process_for_empty_df()
            exit()
        else:
            max_run_id = AV_consolidado.iloc[0]["current_run_id"]
    else:
        max_run_id = "XXX"
    drop_cols_av_cons = (
        ["log_timestamp_replica", "last_modified", "current_run_id"]
        if PREDICTION_PIPELINE
        else ["log_timestamp_replica", "last_modified", "row_num"]
    )
    print("MAX_RUN_ID", max_run_id)
    AV_consolidado = AV_consolidado.drop(columns=drop_cols_av_cons)

    # TODO: REVERT RUN_ID
    run_id_max = AV_consolidado["run_id"].max()
    if pd.isna(run_id_max):
        update_process_for_empty_df()
        exit()

    # update run table, started checking this batch
    if PREDICTION_PIPELINE:
        update_process_status_to_database(max_run_id, 1, "run_process_start_datetime")

    output_directory_path = "/home/cnvdba/cce/src/Features/feature_engineering"

    AV_consolidado = AV_consolidado.rename(columns=rename_columns)
    AV_consolidado_original_stage = AV_consolidado.copy()
    AV_consolidado_original = pd.DataFrame()
    AV_consolidado_original_values_set = pd.DataFrame()
    AV_consolidado_completed = pd.DataFrame()
    # print(dict_column_list)

    # NUEVO
    for column_to_decode, value in dict_column_list.items():
        # Filtrar codificaciones relevantes
        print(column_to_decode)
        codificaciones_stage = codificaciones[codificaciones["List"] == value][
            ["Code", "Value"]
        ]
        codificaciones_dict = codificaciones_stage.set_index("Code")["Value"].to_dict()

        # Rellenar valores nulos en la columna a decodificar
        AV_consolidado[column_to_decode].fillna("", inplace=True)

        # Realizar el merge de forma más eficiente
        AV_consolidado["temp"] = AV_consolidado[column_to_decode].map(
            codificaciones_dict
        )

        if column_to_decode not in ["reasonCode"]:
            AV_consolidado["temp"] = AV_consolidado["temp"].fillna("invalid")

        # Insertar la columna mapeada en el lugar adecuado
        AV_consolidado[column_to_decode] = AV_consolidado["temp"]
        AV_consolidado.drop("temp", axis=1, inplace=True)

    # Renombrar columnas finales in-place
    AV_consolidado.rename(
        columns={
            "debtorParticipantCode": "debtorParticipant",
            "creditorParticipantCode": "creditorParticipant",
        },
        inplace=True,
    )

    # AV_consolidado['flag_invalid'] = (AV_consolidado.astype(str).apply(lambda x: x.str.startswith("invalid_")).any(axis=1)).astype('uint8')
    AV_consolidado["flag_invalid"] = (
        AV_consolidado.astype(str).apply(lambda x: x == "invalid").any(axis=1)
    ).astype("uint8")
    print("done flag invalid")

    AV_consolidado_original_values_set_stage = (
        AV_consolidado.copy()
    )  # DO NOT DELETE THIS LINE

    # AV_consolidado = AV_consolidado[AV_consolidado["flag_invalid"] == 0]
    AV_consolidado.reset_index(drop=True, inplace=True)
    rows_dropped = (
        AV_consolidado_original_values_set_stage.shape[0] - AV_consolidado.shape[0]
    )
    if rows_dropped > 0:
        print(f"{rows_dropped} rows dropped because any value was invalid")
    AV_consolidado.drop(["flag_invalid"], axis=1, inplace=True)

    vars_to_discard = [
        "pk",
        "reasonCode",
        "run_id",
        "trace",
        "instruction_id",
        "message_id",
    ]  # 1
    # vars_to_feature_engineer = ['creationDate', 'creationTime'] # 2
    vars_to_ohe = [
        "debtorTypeOfPerson",
        "debtorParticipant",
        "creditorParticipant",
        "transactionType",
        "currency",
        "channel",
        "responseCode",
    ]  # 3

    # NEW VARIABLES MVP3
    if NEW_VARIABLES_FLAG:
        print("doing new variables")
        AV_consolidado["creation_date_temp"] = pd.to_datetime(
            AV_consolidado["creationDate"]
        )
        new_columns = []
        # in this exact order, to only create "creation_date_temp" once
        AV_consolidado, new_columns = create_frequency_features(
            AV_consolidado, new_columns
        )
        print("first")
        AV_consolidado, new_columns = create_frequency_interaction_creditor_id(
            AV_consolidado, new_columns
        )
        print("second")
        AV_consolidado, new_columns = create_unique_debtors_per_creditor(
            AV_consolidado, new_columns
        )
        print("third")
        AV_consolidado, new_columns = create_frequency_per_channel(
            AV_consolidado, new_columns
        )
        print("fourth")
        AV_consolidado_original_values_set_stage = AV_consolidado.copy()

    # NUEVO
    # 1 vars_to_discard
    AV_consolidado = AV_consolidado.drop(columns=vars_to_discard)

    # 2 vars_to_feature_engineer
    AV_consolidado = AV_consolidado.rename(
        columns={
            "creationDate": "creationDate_stage",
            "creationTime": "creationTime_stage",
        }
    )

    # Conversión de fechas y horas
    AV_consolidado["creationDate_stage"] = pd.to_datetime(
        AV_consolidado["creationDate_stage"]
    )
    AV_consolidado["creationTime_stage"] = (
        AV_consolidado["creationTime_stage"]
        .astype(str)
        .str.replace(":", "", regex=False)
    )
    AV_consolidado["creationTime"] = pd.to_datetime(
        AV_consolidado["creationTime_stage"], format="%H%M%S"
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

    AV_consolidado = create_cyclic_features(AV_consolidado)

    # Eliminación de columnas intermedias
    AV_consolidado = AV_consolidado.drop(
        columns=["creationDate_stage", "creationTime_stage", *vars_to_feature_engineer]
    )

    # 3 vars_to_ohe
    AV_consolidado_3_stage = AV_consolidado.copy()

    # Cambiar formato a "snake_case" sin tildes
    for column in vars_to_ohe:
        AV_consolidado_3_stage[column] = (
            AV_consolidado_3_stage[column]
            .astype(str)
            .apply(unidecode)
            .str.replace(" ", "_", regex=False)
            .str.lower()
        )

    if PREDICTION_PIPELINE:
        ohe = OneHotEncoder(sparse=False, handle_unknown="ignore")
    else:
        ohe = OneHotEncoder(sparse=False, handle_unknown="ignore", drop="first")
    ohe.fit(AV_consolidado_3_stage[vars_to_ohe])

    AV_consolidado_3_ohe_stage = pd.DataFrame(
        ohe.transform(AV_consolidado_3_stage[vars_to_ohe]),
        columns=ohe.get_feature_names_out(vars_to_ohe),
    )
    AV_consolidado = pd.concat(
        [AV_consolidado_3_stage.drop(vars_to_ohe, axis=1), AV_consolidado_3_ohe_stage],
        axis=1,
    )

    AV_consolidado[AV_consolidado_3_ohe_stage.columns] = AV_consolidado[
        AV_consolidado_3_ohe_stage.columns
    ].astype("uint8")
    AV_consolidado_completed = pd.DataFrame()
    AV_consolidado_completed = pd.concat(
        [AV_consolidado_completed, AV_consolidado], axis=0, ignore_index=True
    )

    AV_consolidado_original = pd.DataFrame()
    AV_consolidado_original_stage = AV_consolidado.copy()
    AV_consolidado_original = pd.concat(
        [AV_consolidado_original, AV_consolidado_original_stage],
        axis=0,
        ignore_index=True,
    )
    AV_consolidado_original_values_set = pd.DataFrame()
    AV_consolidado_original_values_set = pd.concat(
        [AV_consolidado_original_values_set, AV_consolidado_original_values_set_stage],
        axis=0,
        ignore_index=True,
    )

    # Fill nulls
    AV_consolidado_completed.fillna(value=0, inplace=True)

    # Set all binary to uint8 (except the ones that are uint8 already)
    id_columns = [
        "debtorId",
        "debtorIdCode",
        "creditorCCI",
        "creditorCreditCard",
        "creditorId",
        "creditorIdCode",
    ]
    excluded_columns = (
        [
            "hourSin",
            "hourCos",
            "dayOfYearSin",
            "dayOfYearCos",
            "dayOfMonthSin",
            "dayOfMonthCos",
            "dayOfWeekSin",
            "dayOfWeekCos",
            "monthSin",
            "monthCos",
            "same_customer_flag",
        ]
        + AV_consolidado_completed.select_dtypes(include="uint8").columns.to_list()
        + id_columns
    )
    print("NEW VARS FLAG", NEW_VARIABLES_FLAG)
    if NEW_VARIABLES_FLAG:
        excluded_columns += new_columns

    # Get a list of column names excluding 'excluded_columns' and id columns
    binary_columns = list(
        set(AV_consolidado_completed.columns).difference(excluded_columns)
    )
    print("BINARY", binary_columns)
    AV_consolidado_completed[binary_columns] = AV_consolidado_completed[
        binary_columns
    ].astype("uint8")

    # Export AV_consolidado_completed
    output_file_path = os.path.join(
        output_directory_path, "cce_ipf_message_feature_engineering.pickle"
    )
    AV_consolidado_completed.to_pickle(output_file_path)

    #    # Export AV_consolidado_original
    #    output_file_path = os.path.join(
    #        output_directory_path, "cce_ipf_message_original.pickle"
    #    )
    #    AV_consolidado_original.to_pickle(output_file_path)

    # Export AV_consolidado_original_values_set
    output_file_path = os.path.join(
        output_directory_path, "cce_ipf_message_original_values_set.pickle"
    )
    AV_consolidado_original_values_set.to_pickle(output_file_path)
    print("end of execution")
    return max_run_id


# feature_engineering_exp4()
