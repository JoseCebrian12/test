#%%
import os
import sys
import datetime
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))))
from src_dev.library_cce.model_functions import *

import pandas as pd

import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)


def predict_model(final_90_same):
    # Path and file inputs
    # trained_models_path = '/home/cnvdba/cce/src_dev/Models/Train'
    trained_models_path = '/data/mvp1-pickles/'
    trained_model_90_old_vars_file_name = "mvp_1_model_90_opt_old_vars_230924.pickle"
    trained_model_90_new_vars_file_name = "mvp_1_model_90_opt_new_vars_230924.pickle"
    
    transformed_data_path = '/data/mvp1-pickles/'
    transformed_data_file_path_90 = os.path.join(transformed_data_path, 'df_output_90.pickle')
    original_values_set_data_file_path = os.path.join(transformed_data_path, 'cce_ipf_message_original_values_set.pickle')
    
    # Load data
    model_90_old_vars = load_model(trained_models_path, trained_model_90_old_vars_file_name)
    model_90_new_vars = load_model(trained_models_path, trained_model_90_new_vars_file_name)

    original_values_set = pd.read_pickle(original_values_set_data_file_path)

    # transformed data
    AV_90 = pd.read_pickle(transformed_data_file_path_90)
    # variables=['creditorParticipant_ratio', 'creditorParticipant_currency_ratio',
    #    'creditorParticipant_channel_ratio',
    #    'creditorParticipant_responseCode_ratio',
    #    'creditorParticipant_debtorParticipant_ratio',
    #    'creditorParticipant_Weekday_ratio',
    #    'creditorParticipant_time_interval_ratio',
    #    'creditorParticipant_creditorCCI_ratio', 'currency_ratio',
    #    'currency_channel_ratio', 'currency_responseCode_ratio',
    #    'currency_debtorParticipant_ratio', 'currency_Weekday_ratio',
    #    'currency_time_interval_ratio', 'currency_creditorCCI_ratio',
    #    'channel_ratio', 'channel_responseCode_ratio',
    #    'channel_debtorParticipant_ratio', 'channel_Weekday_ratio',
    #    'channel_time_interval_ratio', 'channel_creditorCCI_ratio',
    #    'responseCode_ratio', 'responseCode_debtorParticipant_ratio',
    #    'responseCode_Weekday_ratio', 'responseCode_time_interval_ratio',
    #    'responseCode_creditorCCI_ratio', 'debtorParticipant_ratio',
    #    'debtorParticipant_Weekday_ratio',
    #    'debtorParticipant_time_interval_ratio',
    #    'debtorParticipant_creditorCCI_ratio', 'Weekday_ratio',
    #    'Weekday_time_interval_ratio', 'Weekday_creditorCCI_ratio',
    #    'time_interval_ratio', 'time_interval_creditorCCI_ratio',
    #    'creditorCCI_ratio', 'hourSin', 'hourCos', 'dayOfYearSin',
    #    'dayOfYearCos', 'dayOfMonthSin', 'dayOfMonthCos', 'dayOfWeekSin',
    #    'dayOfWeekCos', 'monthSin', 'monthCos', 'debtorTypeOfPerson_n',
    #    'debtorParticipant_banco_falabella',
    #    'debtorParticipant_banco_pichincha', 'debtorParticipant_bbva',
    #    'debtorParticipant_bcp', 'debtorParticipant_caja_arequipa',
    #    'debtorParticipant_caja_cusco', 'debtorParticipant_caja_ica',
    #    'debtorParticipant_caja_piura', 'debtorParticipant_caja_sullana',
    #    'debtorParticipant_caja_trujillo', 'debtorParticipant_comercio',
    #    'debtorParticipant_crediscotia_financiera', 'debtorParticipant_gnb',
    #    'debtorParticipant_interbank', 'debtorParticipant_mi_banco',
    #    'debtorParticipant_scotiabank', 'creditorParticipant_banbif',
    #    'creditorParticipant_banco_de_la_nacion',
    #    'creditorParticipant_banco_falabella',
    #    'creditorParticipant_banco_pichincha', 'creditorParticipant_bbva',
    #    'creditorParticipant_bcp', 'creditorParticipant_caja_arequipa',
    #    'creditorParticipant_caja_cusco', 'creditorParticipant_caja_ica',
    #    'creditorParticipant_caja_piura', 'creditorParticipant_caja_sullana',
    #    'creditorParticipant_caja_trujillo', 'creditorParticipant_citibank',
    #    'creditorParticipant_comercio',
    #    'creditorParticipant_crediscotia_financiera', 'creditorParticipant_gnb',
    #    'creditorParticipant_interbank', 'creditorParticipant_mi_banco',
    #    'creditorParticipant_scotiabank', 'currency_soles',
    #    'channel_banca_movil', 'channel_invalid', 'channel_web',
    #    'responseCode_rejected', 'debtorParticipant_financiera_oh',
    #    'creditorParticipant_financiera_oh', 'debtorParticipant_banbif']
    # variables = ['creditorParticipant_ratio', 'creditorParticipant_currency_ratio', 'creditorParticipant_channel_ratio', 'creditorParticipant_responseCode_ratio', 'creditorParticipant_debtorParticipant_ratio', 'creditorParticipant_Weekday_ratio', 'creditorParticipant_time_interval_ratio', 'creditorParticipant_creditorCCI_ratio', 'currency_ratio', 'currency_channel_ratio', 'currency_responseCode_ratio', 'currency_debtorParticipant_ratio', 'currency_Weekday_ratio', 'currency_time_interval_ratio', 'currency_creditorCCI_ratio', 'channel_ratio', 'channel_responseCode_ratio', 'channel_debtorParticipant_ratio', 'channel_Weekday_ratio', 'channel_time_interval_ratio', 'channel_creditorCCI_ratio', 'responseCode_ratio', 'responseCode_debtorParticipant_ratio', 'responseCode_Weekday_ratio', 'responseCode_time_interval_ratio', 'responseCode_creditorCCI_ratio', 'debtorParticipant_ratio', 'debtorParticipant_Weekday_ratio', 'debtorParticipant_time_interval_ratio', 'debtorParticipant_creditorCCI_ratio', 'Weekday_ratio', 'Weekday_time_interval_ratio', 'Weekday_creditorCCI_ratio', 'time_interval_ratio', 'time_interval_creditorCCI_ratio', 'creditorCCI_ratio', 'hourSin', 'hourCos', 'dayOfYearSin', 'dayOfYearCos', 'dayOfMonthSin', 'dayOfMonthCos', 'dayOfWeekSin', 'dayOfWeekCos', 'debtorParticipant_banbif', 'debtorParticipant_banco_de_la_nacion', 'debtorParticipant_banco_falabella', 'debtorParticipant_banco_pichincha', 'debtorParticipant_banco_ripley', 'debtorParticipant_bbva', 'debtorParticipant_bcp', 'debtorParticipant_caja_arequipa', 'debtorParticipant_caja_cusco', 'debtorParticipant_caja_huancayo', 'debtorParticipant_caja_ica', 'debtorParticipant_caja_piura', 'debtorParticipant_caja_sullana', 'debtorParticipant_caja_trujillo', 'debtorParticipant_comercio', 'debtorParticipant_crediscotia_financiera', 'debtorParticipant_financiera_efectiva', 'debtorParticipant_financiera_oh', 'debtorParticipant_gnb', 'debtorParticipant_interbank', 'debtorParticipant_invalid', 'debtorParticipant_mi_banco', 'debtorParticipant_scotiabank', 'creditorParticipant_banbif', 'creditorParticipant_banco_de_la_nacion', 'creditorParticipant_banco_falabella', 'creditorParticipant_banco_pichincha', 'creditorParticipant_banco_ripley', 'creditorParticipant_bbva', 'creditorParticipant_bcp', 'creditorParticipant_caja_arequipa', 'creditorParticipant_caja_cusco', 'creditorParticipant_caja_huancayo', 'creditorParticipant_caja_ica', 'creditorParticipant_caja_piura', 'creditorParticipant_caja_sullana', 'creditorParticipant_caja_trujillo', 'creditorParticipant_citibank', 'creditorParticipant_comercio', 'creditorParticipant_crediscotia_financiera', 'creditorParticipant_financiera_efectiva', 'creditorParticipant_financiera_oh', 'creditorParticipant_gnb', 'creditorParticipant_interbank', 'creditorParticipant_invalid', 'creditorParticipant_mi_banco', 'creditorParticipant_scotiabank', 'currency_soles', 'channel_banca_movil', 'channel_invalid', 'channel_web', 'responseCode_rejected']
    variables = ['creditorParticipant_ratio', 'creditorParticipant_currency_ratio', 'creditorParticipant_channel_ratio', 'creditorParticipant_responseCode_ratio', 'creditorParticipant_debtorParticipant_ratio', 'creditorParticipant_Weekday_ratio', 'creditorParticipant_time_interval_ratio', 'creditorParticipant_creditorCCI_ratio', 'currency_ratio', 'currency_channel_ratio', 'currency_responseCode_ratio', 'currency_debtorParticipant_ratio', 'currency_Weekday_ratio', 'currency_time_interval_ratio', 'currency_creditorCCI_ratio', 'channel_ratio', 'channel_responseCode_ratio', 'channel_debtorParticipant_ratio', 'channel_Weekday_ratio', 'channel_time_interval_ratio', 'channel_creditorCCI_ratio', 'responseCode_ratio', 'responseCode_debtorParticipant_ratio', 'responseCode_Weekday_ratio', 'responseCode_time_interval_ratio', 'responseCode_creditorCCI_ratio', 'debtorParticipant_ratio', 'debtorParticipant_Weekday_ratio', 'debtorParticipant_time_interval_ratio', 'debtorParticipant_creditorCCI_ratio', 'Weekday_ratio', 'Weekday_time_interval_ratio', 'Weekday_creditorCCI_ratio', 'time_interval_ratio', 'time_interval_creditorCCI_ratio', 'creditorCCI_ratio', 'hourSin', 'hourCos', 'dayOfYearSin', 'dayOfYearCos', 'dayOfMonthSin', 'dayOfMonthCos', 'dayOfWeekSin', 'dayOfWeekCos','debtorParticipant_bcp','debtorParticipant_interbank','debtorParticipant_citibank','debtorParticipant_scotiabank','debtorParticipant_bbva','debtorParticipant_banco_de_la_nacion','debtorParticipant_comercio','debtorParticipant_banco_pichincha','debtorParticipant_banbif','debtorParticipant_crediscotia_financiera','debtorParticipant_mi_banco','debtorParticipant_gnb','debtorParticipant_banco_falabella','debtorParticipant_banco_ripley','debtorParticipant_alfin_banco_s.a.','debtorParticipant_financiera_oh','debtorParticipant_financiera_efectiva','debtorParticipant_caja_piura','debtorParticipant_caja_trujillo','debtorParticipant_caja_arequipa','debtorParticipant_caja_sullana','debtorParticipant_caja_cusco','debtorParticipant_caja_huancayo','debtorParticipant_caja_ica','debtorParticipant_invalid','creditorParticipant_bcp','creditorParticipant_interbank','creditorParticipant_citibank','creditorParticipant_scotiabank','creditorParticipant_bbva','creditorParticipant_banco_de_la_nacion','creditorParticipant_comercio','creditorParticipant_banco_pichincha','creditorParticipant_banbif','creditorParticipant_crediscotia_financiera','creditorParticipant_mi_banco','creditorParticipant_gnb','creditorParticipant_banco_falabella','creditorParticipant_banco_ripley','creditorParticipant_alfin_banco_s.a.','creditorParticipant_financiera_oh','creditorParticipant_financiera_efectiva','creditorParticipant_caja_piura','creditorParticipant_caja_trujillo','creditorParticipant_caja_arequipa','creditorParticipant_caja_sullana','creditorParticipant_caja_cusco','creditorParticipant_caja_huancayo','creditorParticipant_caja_ica','creditorParticipant_invalid', 'currency_soles', 'channel_banca_movil', 'channel_invalid', 'channel_web', 'responseCode_rejected']
    if NEW_VARIABLES_FLAG:
        new_columns = ['f1d',
            'f7d',
            'f30d',
            'f90d',
            'f1d_to_creditor',
            'f7d_to_creditor',
            'f30d_to_creditor',
            'f90d_to_creditor',
            'unique_debtors_past_1d',
            'unique_debtors_past_7d',
            'unique_debtors_past_30d',
            'unique_debtors_past_90d',
            'prop_invalid_1d',
            'prop_banca_movil_1d',
            'prop_web_1d',
            'prop_atm_1d',
            'prop_invalid_7d',
            'prop_banca_movil_7d',
            'prop_web_7d',
            'prop_atm_7d',
            'prop_invalid_30d',
            'prop_banca_movil_30d',
            'prop_web_30d',
            'prop_atm_30d',
            'prop_invalid_90d',
            'prop_banca_movil_90d',
            'prop_web_90d',
            'prop_atm_90d',
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

    # Crear columnas si faltaran
    for col in variables:
        if col not in AV_90.columns:
            AV_90[col] = 0

    AV_90 = AV_90[variables].copy()

    id_cols = ['debtorId', 'creditorCCI', 'same_customer_flag']
    cols_to_drop = [col for col in id_cols if col in AV_90.columns]
    AV_90.drop(columns=cols_to_drop, inplace=True)

    
    if NEW_VARIABLES_FLAG:
        print(AV_90.columns.tolist())
        y_pred_90_new_vars = predict_isolation_forest(AV_90, model_90_new_vars)
        y_pred_90_new_vars_same = pd.Series(data=[0] * len(final_90_same), index=final_90_same.index).astype("uint8")
        y_pred_90_new_vars = pd.concat([y_pred_90_new_vars, y_pred_90_new_vars_same], axis=0)

        y_score_90_new_vars = score_isolation_forest(AV_90, model_90_new_vars, 90)
        y_score_90_new_vars_same = pd.Series(data=[0] * len(final_90_same), index=final_90_same.index).astype("float64")
        y_score_90_new_vars = pd.concat([y_score_90_new_vars, y_score_90_new_vars_same], axis=0)
        print("PREDICTED 90 days new vars model")

    AV_90 = AV_90.drop(new_columns + new_ratios, axis=1)
    print(AV_90.columns.tolist())
    y_pred_90_old_vars = predict_isolation_forest(AV_90, model_90_old_vars)
    y_pred_90_old_vars_same = pd.Series(data=[0] * len(final_90_same), index=final_90_same.index).astype("uint8")
    y_pred_90_old_vars = pd.concat([y_pred_90_old_vars, y_pred_90_old_vars_same], axis=0)

    y_score_90_old_vars = score_isolation_forest(AV_90, model_90_old_vars, 90)
    y_score_90_old_vars_same = pd.Series(data=[0] * len(final_90_same), index=final_90_same.index).astype("float64")
    y_score_90_old_vars = pd.concat([y_score_90_old_vars, y_score_90_old_vars_same], axis=0)
    print("PREDICTED 90 days old vars model")
        
    # Path to the directory to sAV_7e the output of prediction
    y_pred_90_old_vars.to_pickle("prediction_model_90.pickle")

    y_score_90_old_vars.to_pickle("score_model_90.pickle")

    # # Dataframe fraud_model_predict (features id + descriptive + rate + prediction)
    # id_descriptive_features_list = \
    # ['pk', 'debtorId', 'creditorCCI', 'creditorId', 'creditorIdCode', 'message_id', 'trace', 'instruction_id', 'run_id',
    #  'creationDate', 'creationTime', 'channel', 'currency', 'creditorParticipant', 'debtorParticipant',
    #  'debtorTypeOfPerson', 'transactionType', 'debtorIdCode', 'reasonCode', 'responseCode', ]
    # rate_features_list_7 = AV_7.filter(like="_ratio").columns.to_list()
    # rate_features_list_90 = AV_90.filter(like="_ratio").columns.to_list()

    # df_fraud_model_predict_7 = original_values_set[id_descriptive_features_list]\
    #     .merge(AV_7[rate_features_list_7], how="inner", left_index=True, right_index=True)
    # df_fraud_model_predict_7 = df_fraud_model_predict_7\
    #     .merge(y_pred_7.rename("prediction"), how="inner", left_index=True, right_index=True)
    # df_fraud_model_predict_7 = df_fraud_model_predict_7\
    #     .merge(y_score_7.rename("score"), how="inner", left_index=True, right_index=True)
    # df_fraud_model_predict_7.columns=["pk","debtor_id","creditor_cci","creditor_id","creditor_id_code","message_id","trace","instruction_id","run_id","creation_date","creation_time","channel","currency","creditor_participant","debtor_participant","debtor_type_of_person","transaction_type","debtor_id_code","reason_code","response_code","creditor_participant_ratio","creditor_participant_currency_ratio","creditor_participant_channel_ratio","creditor_participant_response_code_ratio","creditor_participant_debtor_participant_ratio","creditor_participant_weekday_ratio","creditor_participant_time_interval_ratio","creditor_participant_creditor_cci_ratio","currency_ratio","currency_channel_ratio","currency_response_code_ratio","currency_debtor_participant_ratio","currency_weekday_ratio","currency_time_interval_ratio","currency_creditor_cci_ratio","channel_ratio","channel_response_code_ratio","channel_debtor_participant_ratio","channel_weekday_ratio","channel_time_interval_ratio","channel_creditor_cci_ratio","response_code_ratio","response_code_debtor_participant_ratio","response_code_weekday_ratio","response_code_time_interval_ratio","response_code_creditor_cci_ratio","debtor_participant_ratio","debtor_participant_weekday_ratio","debtor_participant_time_interval_ratio","debtor_participant_creditor_cci_ratio", "weekday_ratio","weekday_time_interval_ratio","weekday_creditor_cci_ratio","time_interval_ratio","time_interval_creditor_cci_ratio","creditor_cci_ratio","prediction", "score"]    # Load predictions to table
    # #df_fraud_model_predict_7["pk"]=df_fraud_model_predict_7["pk"]+"_l1"
    
    # df_fraud_model_predict_90 = original_values_set[id_descriptive_features_list]\
    #     .merge(AV_90[rate_features_list_90], how="inner", left_index=True, right_index=True)
    # df_fraud_model_predict_90 = df_fraud_model_predict_90\
    #     .merge(y_pred_90_old_vars.rename("prediction"), how="inner", left_index=True, right_index=True)
    # df_fraud_model_predict_90 = df_fraud_model_predict_90\
    #     .merge(y_score_90_old_vars.rename("score"), how="inner", left_index=True, right_index=True)
    # df_fraud_model_predict_90.columns=["pk","debtor_id","creditor_cci","creditor_id","creditor_id_code","message_id","trace","instruction_id","run_id","creation_date","creation_time","channel","currency","creditor_participant","debtor_participant","debtor_type_of_person","transaction_type","debtor_id_code","reason_code","response_code","creditor_participant_ratio","creditor_participant_currency_ratio","creditor_participant_channel_ratio","creditor_participant_response_code_ratio","creditor_participant_debtor_participant_ratio","creditor_participant_weekday_ratio","creditor_participant_time_interval_ratio","creditor_participant_creditor_cci_ratio","currency_ratio","currency_channel_ratio","currency_response_code_ratio","currency_debtor_participant_ratio","currency_weekday_ratio","currency_time_interval_ratio","currency_creditor_cci_ratio","channel_ratio","channel_response_code_ratio","channel_debtor_participant_ratio","channel_weekday_ratio","channel_time_interval_ratio","channel_creditor_cci_ratio","response_code_ratio","response_code_debtor_participant_ratio","response_code_weekday_ratio","response_code_time_interval_ratio","response_code_creditor_cci_ratio","debtor_participant_ratio","debtor_participant_weekday_ratio","debtor_participant_time_interval_ratio","debtor_participant_creditor_cci_ratio", "weekday_ratio","weekday_time_interval_ratio","weekday_creditor_cci_ratio","time_interval_ratio","time_interval_creditor_cci_ratio","creditor_cci_ratio","prediction", "score"]    # Load predictions to table
    # #df_fraud_model_predict_90["pk"]=df_fraud_model_predict_90["pk"]+"_l1"
    # # model_version = 'MOD_OLD_7'
    # load_predictions_to_database(df_fraud_model_predict_7, 'MOD_7_280824_v1')
    # load_predictions_to_database(df_fraud_model_predict_90, 'MOD_90_280824_v1')
    
    
    # # Update process status
    # # run_id = df_fraud_model_predict_7.run_id.max()
    # run_id = max_run_id
    # update_process_status_to_database(run_id,2,"run_process_end_datetime")

# predict_model()
# %%