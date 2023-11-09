import numpy as np
import pandas as pd
from pygam import LinearGAM, s, f
from fire import Fire
import os

SELECTED_FEATURE_NAMES = ['Mean_Forward_in_mm_per_timeframe', 'Mean_Acceleration_in_mm', 'Proportion_Hand_Movement_Above_0.05', 'log_time', 'Type_of_Drill']

def read_extracted_data(trainning_or_inference = "training"):
    data_path = os.path.join(os.path.dirname( __file__ ), '..', 'extracted_data')
    X = pd.read_csv(data_path + "/" + trainning_or_inference + "_features.csv")
    y = pd.read_csv(data_path + "/" + trainning_or_inference + "_response.csv")
    X['log_time'] = np.log(X['Time_Used'])
    return X, y

def built_gam(X, y):
    print("Building Gam model...")
    np_selected_type_drill = X[SELECTED_FEATURE_NAMES].to_numpy()
    gam = LinearGAM(s(0, n_splines=6) + s(1, n_splines=16)+  s(2, n_splines = 8) + s(3, n_splines = 5) + f(4))
    print("Finished building GAM model.")
    gam.fit(np_selected_type_drill, y)

    return gam

def gam_predict(gam, X_inference):
    X_instance = X_inference[SELECTED_FEATURE_NAMES]
    print("Performing inference...")
    return gam.predict(X_instance)

def run():
    X_train, y_train = read_extracted_data("training")
    gam = built_gam(X_train, y_train)
    X_inference, _ = read_extracted_data("inference")
    result = gam_predict(gam, X_inference)
    print("The predicted penetration for three drills respectively are: ")
    print("First drill: " + str(result[0]))
    print("Second drill: " + str(result[1]))
    print("Third drill: " + str(result[2]))

if __name__ == "__main__":
    Fire(run)