#%%
import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from src_dev.library_cce.model_functions import *

import pandas as pd
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)


def feature_engineering_ratio():
    
    # Specify the directory path where the .parquet files are located
    # directory_path = '/home/cnvdba/cce/src_dev/Features/feature_engineering'
    directory_path = "/data/mvp1-pickles/"
    
    # Define a list of columns to keep in the resulting DataFrame
    lista = ["debtorId", "creditorCCI", "run_id", "creditorParticipant", "currency", "channel", "responseCode",
             "debtorParticipant", "creationDate", "creationTime", "time_interval", "Weekday"]
    
    input_list = ["creditorParticipant", "currency", "channel", "responseCode",
                    "debtorParticipant", "Weekday", "time_interval", "creditorCCI"]
    
    if NEW_VARIABLES_FLAG:
        lista += ["creditorId"]
        input_list += ["creditorId"]

    # Generate all possible combinations of columns to be used for ratio calculations
    output_list = generate_combinations(input_list)

    # Define the filenames for feature engineering and original values set DataFrames
    file_name_feature_engineering = 'cce_ipf_message_feature_engineering.pickle'
    file_name_original_values_set = 'cce_ipf_message_original_values_set.pickle'
    
    # Build the file paths for the feature engineering and original values set DataFrames
    file_path_fe = os.path.join(directory_path, file_name_feature_engineering)
    file_path_o_vs = os.path.join(directory_path, file_name_original_values_set)

    # Read the original values set DataFrame
    AV_o_vs = pd.read_pickle(file_path_o_vs)
    # Filter and preprocess the original values set DataFrame
    AV_o_vs = AV_o_vs[AV_o_vs["transactionType"] == "Ordinary Transfer"]
    # AV_o_vs = AV_o_vs[AV_o_vs['debtorParticipant'].isin(code) | AV_o_vs['creditorParticipant'].isin(code)]
    AV_o_vs["creationDate"] = pd.to_datetime(AV_o_vs["creationDate"])
    df = AV_o_vs.copy()
    df["Weekday"] = df["creationDate"].apply(lambda x: x.weekday()).astype(object)
    df['time_interval'] = df["creationTime"].apply(categorize_hour)    
    
    # Calculate ratios based on cumulative counts for specified columns
    ratios_df_90 = Ratio(df, output_list, lista, 90)
    ratios_df_90.dropna(inplace=True)
    print("Finished RATIOS (90 days)")
    
    # Read the feature engineering DataFrame
    AV_fe = pd.read_pickle(file_path_fe)
    AV_fe = AV_fe.drop(['debtorIdCode', 'creditorCreditCard', 'creditorId', 'creditorIdCode'], axis=1)
    
    # Merge the calculated ratios DataFrame and the feature engineering DataFrame
    final_90 = pd.merge(ratios_df_90, AV_fe, left_index=True, right_index=True)

    final_90_same = final_90[final_90["same_customer_flag"] == 'M']
    final_90_diff = final_90[final_90["same_customer_flag"] != 'M']
    print("final 90", final_90.shape)
    print("final 90 same", final_90_same.shape)
    print("final 90 diff", final_90_diff.shape)

    final_90 = final_90_diff
    print("SHAPE 90 days", final_90.shape)

    # Save the final_7 DataFrame to a pickle file
    final_90.to_pickle(os.path.join(directory_path, "df_output_90.pickle"))
    
    print("FEATURE ENGINEERING RATIOS (90 days) STORED IN", os.path.join(directory_path, "df_output_90.pickle"))

    return final_90_same, final_90_diff
    #final_7.head(100).to_csv("df_output.csv")
#feature_engineering_ratio()
# %%
