import numpy as np
import pandas as pd
import os
import glob
import json
from scipy.spatial.transform import Rotation as R
import math
import statistics
import re
import warnings
from fire import Fire
warnings.filterwarnings('ignore')


def Get_Data_From_JSON(json_dict):
    time_stamps = json_dict["Timestamp"]
    all_tools = json_dict["Recordings"]
    ## find drill
    for tool_rec in all_tools:
        if tool_rec["ToolName"] == "Unium_Drill":
            drill_info = tool_rec["Info"]
            drill_translation = tool_rec["Translation"]
            drill_rotation = tool_rec["Rotation"]
    # basic info loading
    drill_df = pd.DataFrame()
    drill_df["ActivityLabel"] = drill_info["ActivityLabel"]
    drill_df["Translation_X"] = drill_translation["X"]
    drill_df["Translation_Y"] = drill_translation["Y"]
    drill_df["Translation_Z"] = drill_translation["Z"]
    drill_df["Rotation_X"] = drill_rotation["X"]
    drill_df["Rotation_Y"] = drill_rotation["Y"]
    drill_df["Rotation_Z"] = drill_rotation["Z"]
    drill_df["Rotation_W"] = drill_rotation["W"]
    drill_df["time_stamps"] = time_stamps
    return drill_df

# distance helper
def Distance(x1, y1, z1, x2, y2, z2):
    d = 0.0
    d = math.sqrt((x2 - x1)**2 + (y2 - y1)**2 + (z2 - z1)**2)
    return d

# Change-of-location outlier removal
def Outlier_Removal(cur_df):
    # calculate distances btw points
    distances = []
    distances.append(0.0)
    removal_indices = []
    for i in range(cur_df.shape[0] - 1):
        prev = cur_df.iloc[i]
        cur = cur_df.iloc[i + 1]
        distance = Distance(prev["Translation_X"], prev["Translation_Y"], prev["Translation_Z"], cur["Translation_X"], cur["Translation_Y"], cur["Translation_Z"])
        distances.append(distance)
    print("Before removal, there are: " + str(len(distances)) + " points.")
    mean = statistics.mean(distances)
    std = statistics.stdev(distances)
    threshold = mean + 3 * std
    for i in range(len(distances)):
        if distances[i] > threshold:
            removal_indices.append(i)
    print("Following indices are removed:")
    print (*removal_indices, sep=",")
    removed_df = cur_df.drop(removal_indices)
    return removed_df

# Moving average filtering (compression, smoothing outliers, and filtering at same time)
def Moving_Average_Filtering(cur_df, factor=3):
    window_size = factor**2
    cur_df['average_translation_X'] = cur_df["Translation_X"].rolling(window_size).mean()
    cur_df['average_translation_Y'] = cur_df["Translation_Y"].rolling(window_size).mean()
    cur_df['average_translation_Z'] = cur_df["Translation_Z"].rolling(window_size).mean()

    cur_df['average_rotation_X'] = cur_df["Rotation_X"].rolling(window_size).mean()
    cur_df['average_rotation_Y'] = cur_df["Rotation_Y"].rolling(window_size).mean()
    cur_df['average_rotation_Z'] = cur_df["Rotation_Z"].rolling(window_size).mean()
    cur_df['average_rotation_W'] = cur_df["Rotation_W"].rolling(window_size).mean()

    cur_df.dropna(inplace = True)

    cur_df = cur_df.iloc[::factor, :]
    return cur_df

def Create_Manual_Index(time_stamps):
    manual_index = []
    for i in range(len(time_stamps)):
        manual_index.append(i)
    return manual_index

def get_matrix(translation_x, translation_y, translation_z, rotation_x, rotation_y, rotation_z, rotation_w):
    T = np.zeros((4, 4), dtype=np.float32)
    T[:-1, 3] = [translation_x, translation_y, translation_z]
    T[0:3, 0:3] = R.from_quat([rotation_x, rotation_y, rotation_z, rotation_w]).as_matrix()
    T[3, 3] = 1
    return T

def get_inverse(T):
    inv = np.zeros(T.shape, dtype=np.float32)
    inv[0:3, 0:3] = T[0:3, 0:3].T
    trans = -T[0:3, 0:3].T @ T[:-1, 3]
    inv[:-1, 3] = trans
    inv[3, 3] = 1
    return inv

def get_intrinsic_distance(translation_x, translation_y, translation_z, rotation_x, rotation_y, rotation_z, rotation_w):
    intrinsics = []
    for i in range(len(translation_x) - 1):
        T_cur = get_matrix(translation_x[i], translation_y[i], translation_z[i], rotation_x[i], rotation_y[i], rotation_z[i],
                          rotation_w[i])
        T_next = get_matrix(translation_x[i + 1], translation_y[i + 1], translation_z[i + 1], 
                            rotation_x[i + 1], rotation_y[i + 1], rotation_z[i + 1], rotation_w[i+1])
        T_delta = get_inverse(T_cur) @ T_next
        quats = R.from_matrix(T_delta[:3, :3]).as_quat()
        intrinsics.append(np.concatenate((T_delta[:-1, 3], quats)))
    return intrinsics

def Construct_Intrinsic_Features(drill_df):
    intrinsics = get_intrinsic_distance(drill_df['Translation_X'].to_list(), drill_df['Translation_Y'].to_list(), 
                                    drill_df['Translation_Z'].to_list(), drill_df['Rotation_X'].to_list(), 
                                    drill_df['Rotation_Y'].to_list(), drill_df['Rotation_Z'].to_list(), drill_df['Rotation_W'].to_list())
    numpy_intrinsics = np.array(intrinsics)
    intrinsic_translation_x = numpy_intrinsics[:,0]
    intrinsic_translation_x = np.insert(intrinsic_translation_x,0,intrinsic_translation_x[0])
    intrinsic_translation_y = numpy_intrinsics[:,1]
    intrinsic_translation_y = np.insert(intrinsic_translation_y,0,intrinsic_translation_y[0])
    intrinsic_translation_z = numpy_intrinsics[:,2]
    intrinsic_translation_z = np.insert(intrinsic_translation_z,0,intrinsic_translation_z[0])
    intrinsic_rotation_x = numpy_intrinsics[:,3]
    intrinsic_rotation_x = np.insert(intrinsic_rotation_x,0,intrinsic_rotation_x[0])
    intrinsic_rotation_y = numpy_intrinsics[:,4]
    intrinsic_rotation_y = np.insert(intrinsic_rotation_y,0,intrinsic_rotation_y[0])
    intrinsic_rotation_z = numpy_intrinsics[:,5]
    intrinsic_rotation_z = np.insert(intrinsic_rotation_z,0,intrinsic_rotation_z[0])
    intrinsic_rotation_w = numpy_intrinsics[:,6]
    intrinsic_rotation_w = np.insert(intrinsic_rotation_w,0,intrinsic_rotation_w[0])

    # scale the intrinsic feature for future analysis

    drill_df["intrinsic_translation_x"] = intrinsic_translation_x * 1000
    drill_df["intrinsic_translation_y"] = intrinsic_translation_y * 1000
    drill_df["intrinsic_translation_z"] = intrinsic_translation_z * 1000

    drill_df["intrinsic_rotation_x"] = intrinsic_rotation_x
    drill_df["intrinsic_rotation_y"] = intrinsic_rotation_y
    drill_df["intrinsic_rotation_z"] = intrinsic_rotation_z
    drill_df["intrinsic_rotation_w"] = intrinsic_rotation_w


def Extract_Drillin(drill_in_df):
    drill_in_df = drill_in_df[drill_in_df["ActivityLabel"] == 0]
    return drill_in_df

# returns indices of the end of segments
def Find_Segments(df):
    indices = []
    manual_index = df["manual_index"].to_list()
    for i in range(len(manual_index) - 1):
        if not (int(manual_index[i]) + 1 == int(manual_index[i + 1])):
            indices.append(i)
            if len(indices) == 10:
                break
    print("indices found for current file: "+ str(len(indices)))
    return indices

# get three penetration values from the txt file
def Get_Penetration(file_name):
    print("Loading event file: " + file_name)
    penetrations = []
    with open(file_name, 'r') as file:
        text = file.read().replace('\n', '')
        all_indices = [m.end() for m in re.finditer('drillSoftTissuePenetration', text)]
        first_drill_index = all_indices[0] + 2
        second_drill_index = all_indices[1] + 2
        last_drill_index = all_indices[len(all_indices) - 1] + 2

        ## convert to millimeters
        first_penetration = round(float(text[first_drill_index : first_drill_index + 8]) * 1000, 2)
        second_penetration = round(float(text[second_drill_index : second_drill_index + 8]) * 1000, 2)
        last_penetration = round(float(text[last_drill_index : last_drill_index + 8]) * 1000, 2)

        penetrations.extend([first_penetration, second_penetration, last_penetration])
        return penetrations
    
    # Not enabling noise filter here, as MA smoothing will do the job and there's no extreme measurement errors
def Load_All(data_path, storeage_df):
    """Load all data"""
    ### data_path is the path of data from CURRENT directory!
    json_files = glob.glob(os.path.join(os.path.dirname( __file__ ), data_path,"*.json"))
    for file in json_files:
        print("Loading file: " + file)
        with open(file, "rb") as json_file:
            cur_json_dict = json.load(json_file)
            cur_drill_df = Get_Data_From_JSON(cur_json_dict)
            # filter the trajectory here
            cur_manual_index = Create_Manual_Index(cur_drill_df["time_stamps"])
            cur_drill_df["manual_index"] = cur_manual_index
            Construct_Intrinsic_Features(cur_drill_df)
            cur_drill_df = Extract_Drillin(cur_drill_df)
            cur_end_indices = Find_Segments(cur_drill_df)
            # construct segments
            cur_feature_list = []
            first_drill_in_df = cur_drill_df.iloc[:cur_end_indices[0] + 1, :]
            #first_drill_in_df = Outlier_Removal(first_drill_in_df)
            first_drill_in_df = Moving_Average_Filtering(first_drill_in_df, 3)

            second_drill_in_df = cur_drill_df.iloc[cur_end_indices[0] + 1: cur_end_indices[1] + 1, :]
            #second_drill_in_df = Outlier_Removal(second_drill_in_df)
            second_drill_in_df = Moving_Average_Filtering(second_drill_in_df, 3)

            # not using filtering for the third drill here, as the third drill is already short in most cases.
            thrid_drill_in_df = cur_drill_df.iloc[cur_end_indices[2] + 1 :, :]
            #thrid_drill_in_df = Outlier_Removal(thrid_drill_in_df)
            thrid_drill_in_df = Moving_Average_Filtering(thrid_drill_in_df, 2)

            cur_feature_list.extend([first_drill_in_df, second_drill_in_df, thrid_drill_in_df])
            cur_penetrations_in_mm = Get_Penetration(file[:len(file) - 17] + "_EventProtocol.txt")
            # construct key-value pairs for penetration and dictionary
            cur_df = pd.DataFrame(zip(cur_penetrations_in_mm, cur_feature_list), columns=["Penetration", "Feature_df"])
            storeage_df = pd.concat([storeage_df,cur_df], axis=0)
    return storeage_df 

def construct_df(training_or_inference = "traning"):
    path_dataset = training_or_inference + "_data"
    path_dataset = os.path.join('..', path_dataset)
    extracted_features_df = pd.DataFrame()
    extracted_features_df = Load_All(path_dataset, extracted_features_df)
    return extracted_features_df

def Speed_Filter(intrinsic_list, level = 0.001):
    return [x for x in intrinsic_list if abs(x) > level]

def Calculate_Mean_Speed(intrinsic_list):
    filtered_list = Speed_Filter(intrinsic_list)
    return statistics.mean(filtered_list)

def Calculate_ABS_Mean_Speed(intrinsic_list):
    filtered_list = Speed_Filter(intrinsic_list)
    abs_filtered_list = []
    for speed in filtered_list:
        abs_filtered_list.append(abs(speed))
    return statistics.mean(abs_filtered_list)

def Calculate_Standard_Deviation_Speed(intrinsic_list):
    filtered_list = Speed_Filter(intrinsic_list)
    return statistics.stdev(filtered_list)

def Max_Forward(intrinsic_list):
    filtered_list = Speed_Filter(intrinsic_list)
    forward_list = [x for x in filtered_list if x > 0]
    return max(forward_list)

def Max_Backword(intrinsic_list):
    filtered_list = Speed_Filter(intrinsic_list)
    backward_list = [x for x in filtered_list if x < 0]
    abs_backward = []
    for speed in backward_list:
        abs_backward.append(abs(speed))
    return max(abs_backward)

def Forward_Mean_Speed(intrinsic_list):
    filtered_list = Speed_Filter(intrinsic_list)
    forward_list = [x for x in filtered_list if x > 0]
    return statistics.mean(forward_list)

def Backward_Mean_Speed(intrinsic_list):
    filtered_list = Speed_Filter(intrinsic_list)
    backward_list = [x for x in filtered_list if x < 0]
    abs_backward = []
    for speed in backward_list:
        abs_backward.append(abs(speed))
    return statistics.mean(abs_backward)

def Calculate_Acceleration(intrinsic_list):
    filtered_intrinsic = Speed_Filter(intrinsic_list)
    acceleration_list = []
    for i in range(len(filtered_intrinsic) - 1):
        acceleration_list.append(filtered_intrinsic[i + 1] - filtered_intrinsic[i])
    return acceleration_list

def Mean_Acceleration(intrinsic_list):
    acceleration_list = Calculate_Acceleration(intrinsic_list)
    return statistics.mean([x for x in acceleration_list if x > 0])

def Mean_Deacceleration(intrinsic_list):
    acceleration_list = Calculate_Acceleration(intrinsic_list)
    deacceleration = [x for x in acceleration_list if x < 0]
    return abs(statistics.mean(deacceleration))

def Max_Acceleration(intrinsic_list):
    acceleration_list = Calculate_Acceleration(intrinsic_list)
    return max([x for x in acceleration_list if x > 0])

def Max_Deacceleration(intrinsic_list):
    acceleration_list = Calculate_Acceleration(intrinsic_list)
    deacceleration = [x for x in acceleration_list if x < 0]
    return abs(min(deacceleration))

def Ger_Drill_Time(intrinsic_list):
    return len(intrinsic_list)

# Different threshold of movements normalized with time
def Get_Num_Movements_Portion(intrinsic_list, threshold = 0.1):
    accelerations = Calculate_Acceleration(intrinsic_list)
    acceleration_list = [x for x in accelerations if x > 0]
    deacceleration_list = [x for x in accelerations if x < 0]
    count = 0
    for acc in acceleration_list:
        if acc > threshold:
            count += 1
    for deacc in deacceleration_list:
        if abs(deacc) > threshold:
            count += 1
    return count/len(accelerations)

def Get_Speed_Portion(intrinsic_list, threshold = 0.1):
    count = 0
    filtered_list = Speed_Filter(intrinsic_list)
    for speed in filtered_list:
        if abs(speed) > threshold:
            count += 1
    return count/len(filtered_list)

def Construct_Features(df_input):
    df = df_input.copy()
    df.reset_index(drop=True, inplace=True)
    for index, row in df.iterrows():
        intrinsic_x_list = row["Feature_df"]["intrinsic_translation_x"].to_list()
        intrinsic_y_list = row["Feature_df"]["intrinsic_translation_y"].to_list()
        intrinsic_z_list = row["Feature_df"]["intrinsic_translation_z"].to_list()
        #extracted_features_df.loc[index,"Mean_Speed_in_mm_per_timeframe"] = Calculate_Mean_Speed(intrinsic_x_list)
        #df.loc[index,"Mean_ABS_Speed"] = Calculate_ABS_Mean_Speed(intrinsic_x_list)
        df.loc[index,"Sdv_Speed_in_mm"] = Calculate_Standard_Deviation_Speed(intrinsic_x_list)
        #extracted_features_df.loc[index,"Max_Forward_in_mm_per_timeframe"] = Max_Forward(intrinsic_x_list)
        #extracted_features_df.loc[index,"Max_Backward_in_mm_per_timeframe"] = Max_Backword(intrinsic_x_list)
        df.loc[index,"Mean_Forward_in_mm_per_timeframe"] = Forward_Mean_Speed(intrinsic_x_list)
        df.loc[index,"Mean_Backward_in_mm"] = Backward_Mean_Speed(intrinsic_x_list)
        df.loc[index,"Mean_Acceleration_in_mm"] = Mean_Acceleration(intrinsic_x_list)
        df.loc[index,"Mean_Deacceleration_in_mm"] = Mean_Deacceleration(intrinsic_x_list)
        #extracted_features_df.loc[index,"Max_Acceleration_in_mm"] = Max_Acceleration(intrinsic_x_list)
        #extracted_features_df.loc[index,"Max_Deacceleration_in_mm"] = Max_Deacceleration(intrinsic_x_list)
        df.loc[index,"Time_Used"] = Ger_Drill_Time(intrinsic_x_list)
        #df.loc[index,"Proportion_Hand_Movement_Above_0.1"] = Get_Num_Movements_Portion(intrinsic_x_list)
        df.loc[index,"Proportion_Hand_Movement_Above_0.05"] = Get_Num_Movements_Portion(intrinsic_x_list, 0.05)
        #df.loc[index,"Proportion_Hand_Movement_Above_0.15"] = Get_Num_Movements_Portion(intrinsic_x_list, 0.15)
        #df.loc[index,"Proportion_Hand_Movement_Above_0.3"] = Get_Num_Movements_Portion(intrinsic_x_list, 0.3)
        #extracted_features_df.loc[index,"Proportion_Hand_Movement_Above_0.5"] = Get_Num_Movements_Portion(intrinsic_x_list, 0.5)

        df.loc[index,"Proportion_Speed_Above_0.4"] = Get_Speed_Portion(intrinsic_x_list, 0.4)
        #df.loc[index,"Proportion_Speed_Above_0.5"] = Get_Speed_Portion(intrinsic_x_list, 0.5)
        #df.loc[index,"Proportion_Speed_Above_0.8"] = Get_Speed_Portion(intrinsic_x_list, 0.8)

        df.loc[index,"Sdv_Y"] = Calculate_Standard_Deviation_Speed(intrinsic_y_list)
        df.loc[index,"Sdv_Z"] = Calculate_Standard_Deviation_Speed(intrinsic_z_list)
        df.loc[index,"Mean_ABS_Speed_Y"] = Calculate_ABS_Mean_Speed(intrinsic_y_list)
        df.loc[index,"Mean_ABS_Speed_Z"] = Calculate_ABS_Mean_Speed(intrinsic_z_list)
        #extracted_features_df.loc[index,"Mean_Speed_Y_in_mm_per_timeframe"] = Calculate_Mean_Speed(intrinsic_y_list)
        #df.loc[index,"Mean_Speed_Z_in_mm_per_timeframe"] = Calculate_Mean_Speed(intrinsic_z_list)
        
        #extracted_features_df.loc[index,"Proportion_Hand_Movement_0.1_Y"] = Get_Num_Movements_Portion(intrinsic_y_list)
        df.loc[index,"Proportion_Hand_Movement_0.05_Y"] = Get_Num_Movements_Portion(intrinsic_y_list, 0.05)
        #extracted_features_df.loc[index,"Proportion_Hand_Movement_0.15_Y"] = Get_Num_Movements_Portion(intrinsic_y_list, 0.15)
        #extracted_features_df.loc[index,"Proportion_Hand_Movement_0.3_Y"] = Get_Num_Movements_Portion(intrinsic_y_list, 0.3)
        df.loc[index,"Proportion_Speed_Above_0.4_Y"] = Get_Speed_Portion(intrinsic_y_list, 0.4)

        #extracted_features_df.loc[index,"Proportion_Hand_Movement_0.1_Z"] = Get_Num_Movements_Portion(intrinsic_z_list)
        df.loc[index,"Proportion_Hand_Movement_0.05_Z"] = Get_Num_Movements_Portion(intrinsic_z_list, 0.05)
        #extracted_features_df.loc[index,"Proportion_Hand_Movement_0.15_Z"] = Get_Num_Movements_Portion(intrinsic_z_list, 0.15)
        #extracted_features_df.loc[index,"Proportion_Hand_Movement_0.3_Z"] = Get_Num_Movements_Portion(intrinsic_z_list, 0.3)
        df.loc[index,"Proportion_Speed_Above_0.4_Z"] = Get_Speed_Portion(intrinsic_z_list, 0.4)
        if index % 3 != 2:
            df.loc[index, "Type_of_Drill"] = int(0)
        else:
            df.loc[index, "Type_of_Drill"] = int(1)

    return df

def data_pipeline_run(training_or_inference = "training", saving_directory = "extracted_data"):
    key_val_df= construct_df(training_or_inference)
    features_df = Construct_Features(key_val_df)
    features_df = features_df.drop(["Feature_df"], axis=1)
    X = features_df.iloc[:, 1:]
    y = features_df.iloc[:, 0]
    X_path = os.path.join(os.path.dirname( __file__ ), '..', saving_directory, training_or_inference + "_features.csv")
    y_path = os.path.join(os.path.dirname( __file__ ), '..', saving_directory, training_or_inference + "_response.csv")
    X.to_csv(X_path, index=False)
    y.to_csv(y_path, index=False)

if __name__ == '__main__':
    Fire(data_pipeline_run)