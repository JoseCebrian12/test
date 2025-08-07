# %%
import os
import sys
import datetime

sys.path.append(
    os.path.dirname(
        os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    )
)
from src_dev_new.library_cce.model_functions import *

import pandas as pd

import warnings

warnings.simplefilter(action="ignore", category=FutureWarning)


def predict_model(max_run_id, final_90_same, final_90_black):
    print("final_90_same", final_90_same.shape)
    print("final_90_black", final_90_black.shape)
    # Path and file inputs
    # trained_models_path = '/home/cnvdba/cce/src_dev_new/Models/Train'
    trained_models_path = "/data/mvp1-pickles/"
    trained_model_90_old_vars_file_name = "test_mvp_1_model_90_old_vars_221124.pickle"
    trained_model_90_new_vars_file_name = "test_mvp_1_model_90_new_vars_221124.pickle"

    transformed_data_path = "/data/mvp1-pickles/"
    transformed_data_file_path_90 = os.path.join(
        transformed_data_path, "test_df_output_90.pickle"
    )
    original_values_set_data_file_path = os.path.join(
        transformed_data_path, "test_cce_ipf_message_original_values_set.pickle"
    )

    # Load data
    model_90_old_vars = load_model(
        trained_models_path, trained_model_90_old_vars_file_name
    )
    model_90_new_vars = load_model(
        trained_models_path, trained_model_90_new_vars_file_name
    )

    original_values_set = pd.read_pickle(original_values_set_data_file_path)
    # transformed data
    AV_90 = pd.read_pickle(transformed_data_file_path_90)
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
        "debtorId",
        "creditorCCI",
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
    if NEW_VARIABLES_FLAG:
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
        # DUMMY DATA
        new_columns_dummy = [
            "personeria_debtor_natural",
            "personeria_creditor_natural",
            "fafternoon_1d",
            "fmorning_1d",
            "fevening_1d",
            "fearly morning_1d",
            "fafternoon_7d",
            "fmorning_7d",
            "fevening_7d",
            "fearly morning_7d",
            "fafternoon_30d",
            "fmorning_30d",
            "fevening_30d",
            "fearly morning_30d",
            "fafternoon_90d",
            "fmorning_90d",
            "fevening_90d",
            "fearly morning_90d",
            "prop1d_amount",
            "prop7d_amount",
            "prop30d_amount",
            "prop90d_amount",
            "transaction_amount",
        ]
        # "in_black_debtor",
        # "in_black_creditor",
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
        # DUMMY DATA
        new_ratios_dummy = [
            "creditorParticipant_personeria_creditor_ratio",
            "currency_personeria_creditor_ratio",
            "channel_personeria_creditor_ratio",
            "responseCode_personeria_creditor_ratio",
            "debtorParticipant_personeria_creditor_ratio",
            "Weekday_personeria_creditor_ratio",
            "time_interval_personeria_creditor_ratio",
            "creditorCCI_personeria_creditor_ratio",
            "creditorId_personeria_creditor_ratio",
            "personeria_creditor_ratio",
        ]

        variables += new_columns + new_columns_dummy + new_ratios + new_ratios_dummy

    # Crear columnas si faltaran
    for col in variables:
        if col not in AV_90.columns:
            print("added as 0", col)
            AV_90[col] = 0

    AV_90 = AV_90[variables].copy()
    id_cols = [
        "debtorId",
        "creditorCCI",
        "same_customer_flag",
        "in_black_debtor",
        "in_black_creditor",
    ]
    cols_to_drop = [col for col in id_cols if col in AV_90.columns]
    AV_90.drop(columns=cols_to_drop, inplace=True)
    AV_90.sort_index(axis=1, ascending=True, inplace=True)
    id_descriptive_features_list = [
        "pk",
        "debtorId",
        "creditorCCI",
        "creditorId",
        "creditorIdCode",
        "message_id",
        "trace",
        "instruction_id",
        "run_id",
        "creationDate",
        "creationTime",
        "channel",
        "currency",
        "creditorParticipant",
        "debtorParticipant",
        "debtorTypeOfPerson",
        "transactionType",
        "debtorIdCode",
        "reasonCode",
        "responseCode",
        "same_customer_flag",
        "personeria_debtor",
        "personeria_creditor",
        "transaction_amount",
    ]

    new_col_names = {
        "pk": "pk",
        "debtorId": "debtor_id",
        "creditorCCI": "creditor_cci",
        "creditorId": "creditor_id",
        "creditorIdCode": "creditor_id_code",
        "message_id": "message_id",
        "trace": "trace",
        "instruction_id": "instruction_id",
        "run_id": "run_id",
        "creationDate": "creation_date",
        "creationTime": "creation_time",
        "channel": "channel",
        "currency": "currency",
        "creditorParticipant": "creditor_participant",
        "debtorParticipant": "debtor_participant",
        "debtorTypeOfPerson": "debtor_type_of_person",
        "transactionType": "transaction_type",
        "debtorIdCode": "debtor_id_code",
        "reasonCode": "reason_code",
        "responseCode": "response_code",
        "creditorParticipant_ratio": "creditor_participant_ratio",
        "creditorParticipant_currency_ratio": "creditor_participant_currency_ratio",
        "creditorParticipant_channel_ratio": "creditor_participant_channel_ratio",
        "creditorParticipant_responseCode_ratio": "creditor_participant_response_code_ratio",
        "creditorParticipant_debtorParticipant_ratio": "creditor_participant_debtor_participant_ratio",
        "creditorParticipant_Weekday_ratio": "creditor_participant_weekday_ratio",
        "creditorParticipant_time_interval_ratio": "creditor_participant_time_interval_ratio",
        "creditorParticipant_creditorCCI_ratio": "creditor_participant_creditor_cci_ratio",
        "currency_ratio": "currency_ratio",
        "currency_channel_ratio": "currency_channel_ratio",
        "currency_responseCode_ratio": "currency_response_code_ratio",
        "currency_debtorParticipant_ratio": "currency_debtor_participant_ratio",
        "currency_Weekday_ratio": "currency_weekday_ratio",
        "currency_time_interval_ratio": "currency_time_interval_ratio",
        "currency_creditorCCI_ratio": "currency_creditor_cci_ratio",
        "channel_ratio": "channel_ratio",
        "channel_responseCode_ratio": "channel_response_code_ratio",
        "channel_debtorParticipant_ratio": "channel_debtor_participant_ratio",
        "channel_Weekday_ratio": "channel_weekday_ratio",
        "channel_time_interval_ratio": "channel_time_interval_ratio",
        "channel_creditorCCI_ratio": "channel_creditor_cci_ratio",
        "responseCode_ratio": "response_code_ratio",
        "responseCode_debtorParticipant_ratio": "response_code_debtor_participant_ratio",
        "responseCode_Weekday_ratio": "response_code_weekday_ratio",
        "responseCode_time_interval_ratio": "response_code_time_interval_ratio",
        "responseCode_creditorCCI_ratio": "response_code_creditor_cci_ratio",
        "debtorParticipant_ratio": "debtor_participant_ratio",
        "debtorParticipant_Weekday_ratio": "debtor_participant_weekday_ratio",
        "debtorParticipant_time_interval_ratio": "debtor_participant_time_interval_ratio",
        "debtorParticipant_creditorCCI_ratio": "debtor_participant_creditor_cci_ratio",
        "Weekday_ratio": "weekday_ratio",
        "Weekday_time_interval_ratio": "weekday_time_interval_ratio",
        "Weekday_creditorCCI_ratio": "weekday_creditor_cci_ratio",
        "time_interval_ratio": "time_interval_ratio",
        "time_interval_creditorCCI_ratio": "time_interval_creditor_cci_ratio",
        "creditorCCI_ratio": "creditor_cci_ratio",
        "prediction": "prediction",
        "score": "score",
        "creditorParticipant_creditorId_ratio": "creditor_participant_creditor_id_ratio",
        "currency_creditorId_ratio": "currency_creditor_id_ratio",
        "channel_creditorId_ratio": "channel_creditor_id_ratio",
        "responseCode_creditorId_ratio": "response_code_creditor_id_ratio",
        "debtorParticipant_creditorId_ratio": "debtor_participant_creditor_id_ratio",
        "Weekday_creditorId_ratio": "weekday_creditor_id_ratio",
        "time_interval_creditorId_ratio": "time_interval_creditor_id_ratio",
        "creditorCCI_creditorId_ratio": "creditor_cci_creditor_id_ratio",
        "creditorId_ratio": "creditor_id_ratio",
        "same_customer_flag": "same_customer_flag",
        ###################################################
        "creditorParticipant_personeria_creditor_ratio": "creditor_participant_personeria_creditor_ratio",
        "currency_personeria_creditor_ratio": "currency_personeria_creditor_ratio",
        "channel_personeria_creditor_ratio": "channel_personeria_creditor_ratio",
        "responseCode_personeria_creditor_ratio": "response_code_personeria_creditor_ratio",
        "debtorParticipant_personeria_creditor_ratio": "debtor_participant_personeria_creditor_ratio",
        "Weekday_personeria_creditor_ratio": "weekday_personeria_creditor_ratio",
        "time_interval_personeria_creditor_ratio": "time_interval_personeria_creditor_ratio",
        "creditorCCI_personeria_creditor_ratio": "creditorcci_personeria_creditor_ratio",
        "creditorId_personeria_creditor_ratio": "creditorid_personeria_creditor_ratio",
        "personeria_creditor_ratio": "personeria_creditor_ratio",
        "transaction_amount": "transaction_amount",
        "personeria_debtor": "personeria_debtor",
        "personeria_creditor": "personeria_creditor",
        "in_black_debtor": "in_black_debtor",
        "in_black_creditor": "in_black_creditor",
    }
    black_cols = ["in_black_debtor", "in_black_creditor"]
    if NEW_VARIABLES_FLAG:
        y_pred_90_new_vars = predict_isolation_forest(AV_90, model_90_new_vars)

        y_pred_90_new_vars_same = pd.Series(
            data=[0] * len(final_90_same), index=final_90_same.index
        ).astype("uint8")
        y_pred_90_new_vars_black = pd.Series(
            data=[1] * len(final_90_black), index=final_90_black.index
        ).astype("uint8")

        y_pred_90_new_vars = pd.concat(
            [y_pred_90_new_vars, y_pred_90_new_vars_same, y_pred_90_new_vars_black],
            axis=0,
        )

        y_score_90_new_vars = score_isolation_forest(AV_90, model_90_new_vars, 90)

        y_score_90_new_vars_same = pd.Series(
            data=[0] * len(final_90_same), index=final_90_same.index
        ).astype("float64")
        y_score_90_new_vars_black = pd.Series(
            data=[99] * len(final_90_black), index=final_90_black.index
        ).astype("float64")

        y_score_90_new_vars = pd.concat(
            [y_score_90_new_vars, y_score_90_new_vars_same, y_score_90_new_vars_black],
            axis=0,
        )
        rate_features_list_90_new = AV_90.filter(like="_ratio").columns.to_list()

        # Dataframe fraud_model_predict_temp (features id + descriptive + rate + prediction)
        # df_in_black_debtor_creditor = final_90_black[["in_black_debtor", "in_black_creditor"]]
        df_fraud_model_predict_90_new = original_values_set[
            id_descriptive_features_list
        ].merge(
            pd.concat([AV_90, final_90_same, final_90_black], axis=0)[
                rate_features_list_90_new + black_cols
            ],
            how="inner",
            left_index=True,
            right_index=True,
        )
        df_fraud_model_predict_90_new = df_fraud_model_predict_90_new.merge(
            y_pred_90_new_vars.rename("prediction"),
            how="inner",
            left_index=True,
            right_index=True,
        )
        df_fraud_model_predict_90_new = df_fraud_model_predict_90_new.merge(
            y_score_90_new_vars.rename("score"),
            how="inner",
            left_index=True,
            right_index=True,
        )
        df_fraud_model_predict_90_new.rename(columns=new_col_names, inplace=True)
        df_fraud_model_predict_90_new["in_black_debtor"] = (
            df_fraud_model_predict_90_new["in_black_debtor"].fillna(False)
        )
        df_fraud_model_predict_90_new["in_black_creditor"] = (
            df_fraud_model_predict_90_new["in_black_creditor"].fillna(False)
        )

        print("PREDICTED 90 days new vars model")

    AV_90 = AV_90.drop(new_columns_dummy + new_ratios_dummy, axis=1)
    AV_90.sort_index(axis=1, ascending=True, inplace=True)
    # print("GAAAAAAAA TAMAÑO", AV_90.shape)
    # print(AV_90.columns.tolist())

    y_pred_90_old_vars = predict_isolation_forest(AV_90, model_90_old_vars)

    y_pred_90_old_vars_same = pd.Series(
        data=[0] * len(final_90_same), index=final_90_same.index
    ).astype("uint8")
    y_pred_90_old_vars_black = pd.Series(
        data=[1] * len(final_90_black), index=final_90_black.index
    ).astype("uint8")

    y_pred_90_old_vars = pd.concat(
        [y_pred_90_old_vars, y_pred_90_old_vars_same, y_pred_90_old_vars_black], axis=0
    )

    y_score_90_old_vars = score_isolation_forest(AV_90, model_90_old_vars, 90)

    y_score_90_old_vars_same = pd.Series(
        data=[0] * len(final_90_same), index=final_90_same.index
    ).astype("float64")
    y_score_90_old_vars_black = pd.Series(
        data=[99] * len(final_90_black), index=final_90_black.index
    ).astype("float64")

    y_score_90_old_vars = pd.concat(
        [y_score_90_old_vars, y_score_90_old_vars_same, y_score_90_old_vars_black],
        axis=0,
    )

    # Dataframe fraud_model_predict_temp (features id + descriptive + rate + prediction)
    rate_features_list_90_old = AV_90.filter(like="_ratio").columns.to_list()
    df_fraud_model_predict_90_old = original_values_set[
        id_descriptive_features_list
    ].merge(
        pd.concat([AV_90, final_90_same, final_90_black], axis=0)[
            rate_features_list_90_old + black_cols
        ],
        how="inner",
        left_index=True,
        right_index=True,
    )
    df_fraud_model_predict_90_old = df_fraud_model_predict_90_old.merge(
        y_pred_90_old_vars.rename("prediction"),
        how="inner",
        left_index=True,
        right_index=True,
    )
    df_fraud_model_predict_90_old = df_fraud_model_predict_90_old.merge(
        y_score_90_old_vars.rename("score"),
        how="inner",
        left_index=True,
        right_index=True,
    )
    df_fraud_model_predict_90_old.rename(columns=new_col_names, inplace=True)
    df_fraud_model_predict_90_old["in_black_debtor"] = df_fraud_model_predict_90_old[
        "in_black_debtor"
    ].fillna(False)
    df_fraud_model_predict_90_old["in_black_creditor"] = df_fraud_model_predict_90_old[
        "in_black_creditor"
    ].fillna(False)

    print("PREDICTED 90 days old vars model")

    load_predictions_to_database_temp(
        df_fraud_model_predict_90_old, "MOD_90_OLD_221124_v1"
    )
    load_predictions_to_database_temp(
        df_fraud_model_predict_90_new, "MOD_90_NEW_221124_v1"
    )

    # Update process status
    # run_id = max_run_id
    # update_process_status_to_database(run_id, 2, "run_process_end_datetime")
