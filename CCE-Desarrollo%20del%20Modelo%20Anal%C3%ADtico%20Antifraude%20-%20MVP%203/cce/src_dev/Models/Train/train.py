#%%
# Model functions library
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))))
from src_dev.library_cce.model_functions import *

import os
import pandas as pd
import pickle


import warnings
warnings.filterwarnings('ignore')
def trainmodel():
# Random state
    RS = 12494328
    #train_data_directory_path = 'C:/Users/dtorres/proyectos_canvia/CEE/src/Features/feature_engineering'
    # train_data_file_name = 'df_output.pickle'
    #train_data_file_path = os.path.join(train_data_directory_path, train_data_file_name)
    # output
    # output_directory_path = '../Data/model/3_training/experiment_4/sample1'
    #output_directory_path = 'C:/Users/dtorres/proyectos_canvia/CEE/src/Models/Train'
    # Load data
    # train data
    # AV_train_7 = pd.read_pickle("/home/cnvdba/cce/src_dev/Features/feature_engineering/df_output.pickle")
    AV_train_90 = pd.read_pickle("/data/mvp1-pickles/df_output_90.pickle")
    print(AV_train_90.shape)
    print(f"AV_train shape: {AV_train_90.shape}")
    #print(AV_train_7)
    # Explore train data
    #print(f"AV_train_7 shape: {AV_train_7.shape}")
    #print(f"AV_train_7 columns: {AV_train_7.columns}")
    #print(f"creditorCCI únicos: {AV_train_7.creditorCCI.nunique()}")
    #print("Top 10 creditor CCI con más operaciones AV:")
    #print(AV_train_7.creditorCCI.value_counts().sort_values(ascending=False).head(10))
    AV_train_90_nunique = AV_train_90.nunique()
    cols_to_drop_unique_value_90 = AV_train_90_nunique[AV_train_90_nunique == 1].index.to_list()
    
    # drop columns with unique value
    if len(cols_to_drop_unique_value_90) > 0:
        AV_train_90.drop(cols_to_drop_unique_value_90, axis=1, inplace=True)
    #print(AV_train_90.shape)
    AV_train_90.dropna(inplace=True)
    #print(AV_train_90.shape)
    
    args_5 = {"random_state":RS, "contamination":0.0196}
    id_cols = ["debtorId",'creditorCCI']
    
    variables = ['creditorParticipant_ratio', 'creditorParticipant_currency_ratio', 'creditorParticipant_channel_ratio', 'creditorParticipant_responseCode_ratio', 'creditorParticipant_debtorParticipant_ratio', 'creditorParticipant_Weekday_ratio', 'creditorParticipant_time_interval_ratio', 'creditorParticipant_creditorCCI_ratio', 'currency_ratio', 'currency_channel_ratio', 'currency_responseCode_ratio', 'currency_debtorParticipant_ratio', 'currency_Weekday_ratio', 'currency_time_interval_ratio', 'currency_creditorCCI_ratio', 'channel_ratio', 'channel_responseCode_ratio', 'channel_debtorParticipant_ratio', 'channel_Weekday_ratio', 'channel_time_interval_ratio', 'channel_creditorCCI_ratio', 'responseCode_ratio', 'responseCode_debtorParticipant_ratio', 'responseCode_Weekday_ratio', 'responseCode_time_interval_ratio', 'responseCode_creditorCCI_ratio', 'debtorParticipant_ratio', 'debtorParticipant_Weekday_ratio', 'debtorParticipant_time_interval_ratio', 'debtorParticipant_creditorCCI_ratio', 'Weekday_ratio', 'Weekday_time_interval_ratio', 'Weekday_creditorCCI_ratio', 'time_interval_ratio', 'time_interval_creditorCCI_ratio', 'creditorCCI_ratio', 'debtorId', 'creditorCCI', 'hourSin', 'hourCos', 'dayOfYearSin', 'dayOfYearCos', 'dayOfMonthSin', 'dayOfMonthCos', 'dayOfWeekSin', 'dayOfWeekCos','debtorParticipant_bcp','debtorParticipant_interbank','debtorParticipant_citibank','debtorParticipant_scotiabank','debtorParticipant_bbva','debtorParticipant_banco_de_la_nacion','debtorParticipant_comercio','debtorParticipant_banco_pichincha','debtorParticipant_banbif','debtorParticipant_crediscotia_financiera','debtorParticipant_mi_banco','debtorParticipant_gnb','debtorParticipant_banco_falabella','debtorParticipant_banco_ripley','debtorParticipant_alfin_banco_s.a.','debtorParticipant_financiera_oh','debtorParticipant_financiera_efectiva','debtorParticipant_caja_piura','debtorParticipant_caja_trujillo','debtorParticipant_caja_arequipa','debtorParticipant_caja_sullana','debtorParticipant_caja_cusco','debtorParticipant_caja_huancayo','debtorParticipant_caja_ica','debtorParticipant_invalid','creditorParticipant_bcp','creditorParticipant_interbank','creditorParticipant_citibank','creditorParticipant_scotiabank','creditorParticipant_bbva','creditorParticipant_banco_de_la_nacion','creditorParticipant_comercio','creditorParticipant_banco_pichincha','creditorParticipant_banbif','creditorParticipant_crediscotia_financiera','creditorParticipant_mi_banco','creditorParticipant_gnb','creditorParticipant_banco_falabella','creditorParticipant_banco_ripley','creditorParticipant_alfin_banco_s.a.','creditorParticipant_financiera_oh','creditorParticipant_financiera_efectiva','creditorParticipant_caja_piura','creditorParticipant_caja_trujillo','creditorParticipant_caja_arequipa','creditorParticipant_caja_sullana','creditorParticipant_caja_cusco','creditorParticipant_caja_huancayo','creditorParticipant_caja_ica','creditorParticipant_invalid', 'currency_soles', 'channel_banca_movil', 'channel_invalid', 'channel_web', 'responseCode_rejected']
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

    for col in variables:
        if col not in AV_train_90.columns:
            AV_train_90[col] = 0

    for x in AV_train_90.columns:
        if x not in variables:
            print("dropping...", x)
    AV_train_90 = AV_train_90[variables]

    #print("variables train",AV_train_7.columns.tolist())
    #AV_train_7.drop(id_cols, axis=1).head(100).to_csv("train.csv")
    # output_directory_path="/home/cnvdba/cce/src_dev/Models/Train"
    output_directory_path = "/data/mvp1-pickles/"
    # file_path_7 = os.path.join(output_directory_path, "mvp_1_model_90_5M.pickle")

    if_model_90_old_vars = fit_isolation_forest(AV_train_90.drop(id_cols + new_columns + new_ratios, axis=1), args_5)
    file_name_90_old_vars = "mvp_1_model_90_opt_old_vars_260924.pickle"
    file_path_90_old_vars = os.path.join(output_directory_path, file_name_90_old_vars)
    with open(file_path_90_old_vars, 'wb') as handle:
        pickle.dump(if_model_90_old_vars, handle, protocol=pickle.HIGHEST_PROTOCOL)
    print("TRAINED 90 days old vars model")

    if NEW_VARIABLES_FLAG:
        if_model_90_new_vars = fit_isolation_forest(AV_train_90.drop(id_cols, axis=1), args_5)
        file_name_90_new_vars = "mvp_1_model_90_opt_new_vars_260924.pickle"
        file_path_90_new_vars = os.path.join(output_directory_path, file_name_90_new_vars)
        with open(file_path_90_new_vars, 'wb') as handle:
            pickle.dump(if_model_90_new_vars, handle, protocol=pickle.HIGHEST_PROTOCOL)
        print("TRAINED 90 days new vars model")
# trainmodel()
# %%
