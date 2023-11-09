import numpy as np
import pandas as pd
import dice_ml
from dice_ml import Dice
import os
from sklearn.linear_model import Ridge
from fire import Fire

SELECTED_FEATURE_NAMES = ['Mean_Forward_in_mm_per_timeframe', 'Mean_Acceleration_in_mm', 'Proportion_Hand_Movement_Above_0.05', 'log_time']

def read_extracted_data(trainning_or_inference = "training"):
    data_path = os.path.join(os.path.dirname( __file__ ), '..', 'extracted_data')
    X = pd.read_csv(data_path + "/" + trainning_or_inference + "_features.csv")
    y = pd.read_csv(data_path + "/" + trainning_or_inference + "_response.csv")
    X['log_time'] = np.log(X['Time_Used'])
    return X, y

def built_cf(X, y):
    print("Building Counterfactual model...")
    cf_df = X[SELECTED_FEATURE_NAMES]
    cf_df["Penetration"] = y
    ridge = Ridge(1)
    model_ridge = ridge.fit(cf_df.drop('Penetration', axis = 1), cf_df['Penetration'])
    data_cf = dice_ml.Data(dataframe=cf_df, continuous_features=SELECTED_FEATURE_NAMES, outcome_name="Penetration")
    model_cf = dice_ml.Model(model=model_ridge, backend="sklearn", model_type='regressor')
    exp= Dice(data_cf, model_cf, method="genetic")

    return exp

def counterfactual_inference(X_inference, exp, lower_bound = 5.0, upper_bound = 15.0):
    query_instances = X_inference[SELECTED_FEATURE_NAMES]
    generic_cf = exp.generate_counterfactuals(query_instances, total_CFs=1, desired_range=[lower_bound, upper_bound], features_to_vary=SELECTED_FEATURE_NAMES)
    generic_cf.visualize_as_dataframe()

def run(lower_bound = 5.0, upper_bound = 15.0):
    X_train, y_train = read_extracted_data("training")
    exp = built_cf(X_train, y_train)
    X_inference, _ = read_extracted_data("inference")
    counterfactual_inference(X_inference, exp, lower_bound, upper_bound)

if __name__ == "__main__":
    Fire(run)