#%%
# Model functions library
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))))
from src.library_cce.model_functions import *

import os
import pandas as pd
import pickle


import warnings
warnings.filterwarnings('ignore')
def trainmodel():
# Random state
    RS = 12494328
    #train_data_directory_path = 'C:/Users/dtorres/proyectos_canvia/CEE/src/Features/feature_engineering'
    train_data_file_name = 'df_output.pickle'
    #train_data_file_path = os.path.join(train_data_directory_path, train_data_file_name)
    # output
    # output_directory_path = '../Data/model/3_training/experiment_4/sample1'
    #output_directory_path = 'C:/Users/dtorres/proyectos_canvia/CEE/src/Models/Train'
    # Load data
    # train data
    AV_train = pd.read_pickle("/home/cnvdba/cce/src/Features/feature_engineering/df_output.pickle")
    print("nasss",AV_train.isna().sum()/AV_train.shape[0])
    print(AV_train)
    # Explore train data
    print(f"AV_train shape: {AV_train.shape}")
    print(f"AV_train columns: {AV_train.columns}")
    print(f"creditorCCI únicos: {AV_train.creditorCCI.nunique()}")
    print("Top 10 creditor CCI con más operaciones AV:")
    print(AV_train.creditorCCI.value_counts().sort_values(ascending=False).head(10))
    AV_train_nunique = AV_train.nunique()
    cols_to_drop_unique_value = AV_train_nunique[AV_train_nunique == 1].index.to_list()
    # drop columns with unique value
    if len(cols_to_drop_unique_value) > 0:
        AV_train.drop(cols_to_drop_unique_value, axis=1, inplace=True)
    print(AV_train.shape)
    AV_train.dropna(inplace=True)
    print(AV_train.shape)
    
    args_5 = {"random_state":RS, "contamination":0.0196}
    id_cols = ["debtorId",'creditorCCI']
    
    variables = ['creditorParticipant_ratio', 'creditorParticipant_currency_ratio', 'creditorParticipant_channel_ratio', 'creditorParticipant_responseCode_ratio', 'creditorParticipant_debtorParticipant_ratio', 'creditorParticipant_Weekday_ratio', 'creditorParticipant_time_interval_ratio', 'creditorParticipant_creditorCCI_ratio', 'currency_ratio', 'currency_channel_ratio', 'currency_responseCode_ratio', 'currency_debtorParticipant_ratio', 'currency_Weekday_ratio', 'currency_time_interval_ratio', 'currency_creditorCCI_ratio', 'channel_ratio', 'channel_responseCode_ratio', 'channel_debtorParticipant_ratio', 'channel_Weekday_ratio', 'channel_time_interval_ratio', 'channel_creditorCCI_ratio', 'responseCode_ratio', 'responseCode_debtorParticipant_ratio', 'responseCode_Weekday_ratio', 'responseCode_time_interval_ratio', 'responseCode_creditorCCI_ratio', 'debtorParticipant_ratio', 'debtorParticipant_Weekday_ratio', 'debtorParticipant_time_interval_ratio', 'debtorParticipant_creditorCCI_ratio', 'Weekday_ratio', 'Weekday_time_interval_ratio', 'Weekday_creditorCCI_ratio', 'time_interval_ratio', 'time_interval_creditorCCI_ratio', 'creditorCCI_ratio', 'debtorId', 'creditorCCI', 'hourSin', 'hourCos', 'dayOfYearSin', 'dayOfYearCos', 'dayOfMonthSin', 'dayOfMonthCos', 'dayOfWeekSin', 'dayOfWeekCos','debtorParticipant_bcp','debtorParticipant_interbank','debtorParticipant_citibank','debtorParticipant_scotiabank','debtorParticipant_bbva','debtorParticipant_banco_de_la_nacion','debtorParticipant_comercio','debtorParticipant_banco_pichincha','debtorParticipant_banbif','debtorParticipant_crediscotia_financiera','debtorParticipant_mi_banco','debtorParticipant_gnb','debtorParticipant_banco_falabella','debtorParticipant_banco_ripley','debtorParticipant_alfin_banco_s.a.','debtorParticipant_financiera_oh','debtorParticipant_financiera_efectiva','debtorParticipant_caja_piura','debtorParticipant_caja_trujillo','debtorParticipant_caja_arequipa','debtorParticipant_caja_sullana','debtorParticipant_caja_cusco','debtorParticipant_caja_huancayo','debtorParticipant_caja_ica','debtorParticipant_invalid','creditorParticipant_bcp','creditorParticipant_interbank','creditorParticipant_citibank','creditorParticipant_scotiabank','creditorParticipant_bbva','creditorParticipant_banco_de_la_nacion','creditorParticipant_comercio','creditorParticipant_banco_pichincha','creditorParticipant_banbif','creditorParticipant_crediscotia_financiera','creditorParticipant_mi_banco','creditorParticipant_gnb','creditorParticipant_banco_falabella','creditorParticipant_banco_ripley','creditorParticipant_alfin_banco_s.a.','creditorParticipant_financiera_oh','creditorParticipant_financiera_efectiva','creditorParticipant_caja_piura','creditorParticipant_caja_trujillo','creditorParticipant_caja_arequipa','creditorParticipant_caja_sullana','creditorParticipant_caja_cusco','creditorParticipant_caja_huancayo','creditorParticipant_caja_ica','creditorParticipant_invalid', 'currency_soles', 'channel_banca_movil', 'channel_invalid', 'channel_web', 'responseCode_rejected']

    for col in variables:
        if col not in AV_train.columns:
            AV_train[col] = 0

    AV_train = AV_train[variables]

    print("variables train",AV_train.columns.tolist())
    #AV_train.drop(id_cols, axis=1).head(100).to_csv("train.csv")
    if_model_5 = fit_isolation_forest(AV_train.drop(id_cols,axis=1), args_5)
    output_directory_path="/home/cnvdba/cce/src/Models/Train"
    file_path = os.path.join(output_directory_path, "mvp_1_model.pickle")
    with open(file_path, 'wb') as handle:
        pickle.dump(if_model_5, handle, protocol=pickle.HIGHEST_PROTOCOL)
    
# trainmodel()
# %%
