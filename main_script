import importlib
import json
import fnmatch
import os
import pandas as pd
import numpy as np
from datetime import datetime, timezone
import matplotlib.pyplot as plt
import math
from sklearn.model_selection import train_test_split
import swifter
import re

def initialize_fulldata_dir(release = "rel05"):
    mainpath = "output_" + release +'/'
    directories = {
        "rawdata_dir" : mainpath + "rawdata/", 
        "fulldata_dir" : mainpath + "fulldata/"
        }
    
    return directories, mainpath

def time_string_one(float_time=None, fmt=None):
    """
    Transforms a single float daytime value into a string representation.

    Parameters
    ----------
    float_time : float, optional
        The input time as a float. If not provided, the current time is used.
    fmt : str, optional
        The format string for time conversion. The default format is '%Y-%m-%d %H:%M:%S.%f'.

    Returns
    -------
    str
        The datetime as a string formatted according to `fmt`. If `float_time` is not provided,
        returns the current datetime formatted as specified.

    Examples
    --------
    >>> from pyspedas import time_string_one
    >>> time_string_one(1679745600.0)
    '2023-03-25 12:00:00.000000'

    >>> time_string_one(1679745600.0, '%Y-%m-%d')
    '2023-03-25'

    >>> time_string_one()
    # Returns the current datetime in the default format
    """

    if fmt is None:
        fmt = '%Y-%m-%d %H:%M:%S.%f'

    if float_time is None:
        str_time = datetime.now().strftime(fmt)
    else:
        str_time = datetime.fromtimestamp(float_time,timezone.utc).strftime(fmt)

    return str_time

def time_string(float_time=None, fmt=None):
    """
    Transforms a list of float daytime values into a list of string representations.

    Parameters
    ----------
    float_time : floats, or list of floats, optional
        The input time(s) as float(s). If not provided, the current time is used.
    fmt : str, optional
        The format string for time conversion. The default format is '%Y-%m-%d %H:%M:%S.%f'.

    Returns
    -------
    str or list of str
        The datetimes as strings formatted according to `fmt`. If `float_time` is not provided, returns
        the current datetime formatted as specified. If `float_time` is a single float, returns
        a single datetime string.

    Examples
    --------
    >>> time_string(1679745600.0)
    '2023-03-25 12:00:00.000000'

    >>> time_string([1679745600.0, 1679832000.0], "%Y-%m-%d %H:%M:%S")
    ['2023-03-25 12:00:00', '2023-03-26 12:00:00']

    >>> time_string()
    #'Returns the current datetime in the default format'

    Notes
    -----
    Compare to https://www.epochconverter.com/
    """
    if float_time is None:
        return time_string_one(None, fmt)
    else:
        if isinstance(float_time, (int, float)):
            return time_string_one(float_time, fmt)
        else:
            time_list = list()
            for t in float_time:
                time_list.append(time_string_one(t, fmt))
            return time_list
        
def read_probes_data(data_dir, fulldata_settings):
    df_full = pd.DataFrame()
    probes = ['a','b']
    
    for iprobe in probes:
        print("Reading csv data for probe " + iprobe, end="\r")
        df = pd.read_csv(data_dir + 'rbsp' + iprobe.capitalize() + '_data_' + fulldata_settings["release"] + '_fulldata.csv')  
        df['probe'] = iprobe
        if iprobe == probes[0]:
            df_full = df
        else:
            df_full = pd.concat([df_full, df], ignore_index=True)  
    
    df_full[fulldata_settings["datetime_name"]] = df_full['time'].apply(lambda x : time_string(x)).astype('datetime64[ns]')
    
    return df_full

def load_coor(directories, fulldataset_csv, fulldata_settings, recalc = False, df_full = [], save_data = True, plot_data = True):
    coor_filename =  fulldataset_csv["df_coor"] + ".csv"
    
    if os.path.exists(coor_filename) & (recalc != True):
        df_coor = pd.read_csv(coor_filename, index_col=False)
        # with open(fulldataset_csv["fulldata_settings_filename"]+'.pkl', 'rb') as file:
        #     fulldata_settings = pickle.load(file)
    else:
        if len(df_full) == 0:
            df_full = read_probes_data(directories["rawdata_dir"], fulldata_settings)
        df_coor, fulldata_settings = scale_corrdinates(df_full, fulldata_settings,  fulldata_settings["datetime_name"], fulldataset_csv["df_coor"], save_data = save_data, plot_data = plot_data)
            
    return df_coor, df_full, fulldata_settings

def scale_arr(arr):
    index = np.isfinite(arr)
    valid_arr = arr[index]
    max_value = max(valid_arr)
    min_value = min(valid_arr)
    mid_value = (max_value + min_value)/2
    scale_value = max_value - min_value
    scaled_arr = (arr - mid_value)/scale_value*2
    # print(max_value, min_value, mid_value, scale_value)
    
    return(scaled_arr, mid_value, scale_value)

def scale_var(df_full, varname, fulldata_settings):
    scaled_var, fulldata_settings[varname+'_mid_vlaue'],  fulldata_settings[varname+'_scale_vlaue'] = scale_arr(df_full[varname])

    return scaled_var, fulldata_settings

def scale_corrdinates(df_full, fulldata_settings, doubletime_name, outputfilename, save_data = True, plot_data = True):       
    df_cos = df_full['mlt'].apply(lambda x: np.cos(x*np.pi/12.0))
    df_sin = df_full['mlt'].apply(lambda x: np.sin(x*np.pi/12.0))
    df_l, fulldata_settings = scale_var(df_full, 'l', fulldata_settings)
    df_lat, fulldata_settings = scale_var(df_full, 'lat', fulldata_settings)
    
    # df_coor = pd.DataFrame(np.array([df_full[doubletime_name],df_cos,df_sin,df_l,df_lat, df_full["mlt"], df_full["l"], df_full["lat"]]), columns=[doubletime_name, 'cos0','sin0','scaled_l','scaled-l', "mlt",'l','lat']).reset_index(drop=True)
    
    df_coor = pd.DataFrame({doubletime_name:df_full[doubletime_name],"cos0":df_cos, "sin0":df_sin, "scaled_l":df_l,"scaled_lat":df_lat })
    # df_coor = pd.concat([df_coor, df_full[["mlt",'l','lat']]], axis=1)
    
    if save_data:
        df_coor.to_csv(outputfilename + '.csv', index=False)
        
    if plot_data:            
        plot_coor_data(df_coor, df_full, fulldata_settings["raw_coor_names"], fulldata_settings["coor_names"], fulldata_settings["datetime_name"], filename = outputfilename)
        
    return df_coor, fulldata_settings

def load_y(directories, fulldataset_csv, fulldata_settings, recalc = False, df_full = [], save_data = True, plot_data = True, energy_bins=(np.array([51767.680, 44428.696, 38130.120, 32724.498, 28085.268, 24103.668, 20686.558, 17753.876, 15236.896, 13076.798, 11222.936, 9631.899, 8266.406, 7094.516, 6088.722, 5225.528, 4484.742, 3848.919, 3303.284, 2834.964, 2433.055, 2088.129, 1792.096, 1538.062, 1319.977, 1132.846, 972.237, 834.421, 716.163, 614.578, 527.484, 452.702, 388.543, 333.459, 286.184, 245.592, 210.769, 180.870, 155.262, 133.243, 114.319, 98.138, 84.209, 72.320, 62.049, 53.255, 45.728, 39.185, 33.627, 28.914, 24.763, 21.246, 18.291, 15.688, 13.437, 11.537, 9.919, 8.512, 7.316, 6.261, 5.347, 4.643, 3.940, 3.377, 2.955, 2.533, 2.181, 1.829, 1.548, 1.337, 1.196, 0.985]) * 1000.).astype(int).astype(str), species_arr=['h','o'] ):   

    for species in species_arr:
        for energy in energy_bins:
            y_name = species + '_flux_'+energy
            log_y_name = 'log_' + y_name
            log_y_filename = fulldataset_csv["df_y"] + '_'+log_y_name
            print(log_y_filename)

            if os.path.exists(log_y_filename+'.csv') & (recalc != True):
                idf_y = pd.read_csv(log_y_filename+'.csv', index_col=False)
            else:
                if len(df_full) == 0:
                    df_full = read_probes_data(directories["rawdata_dir"], fulldata_settings)
                idf_y = df_full[[fulldata_settings["datetime_name"], y_name]]
                idf_y, fulldata_settings = calculate_log_for_y(idf_y, y_name, fulldata_settings, log_y_filename, fulldata_settings["datetime_name"], save_data = save_data, plot_data = plot_data)
                                
            if not 'df_y' in locals():
                df_y = idf_y
            else:
                print(idf_y.columns)
                df_y = pd.concat([df_y, idf_y[[y_name, log_y_name]]], axis=1)
                
    return df_y, df_full, fulldata_settings

def plot_y_data(df_full, y_names, datetime_name, filename = 'dataview_y'):
    view_data(df_full, y_names, y_names,  df_full[datetime_name].astype('datetime64[ns]').reset_index(drop=True), figname = filename)

def calculate_log_for_y(df_y, y_name, fulldata_settings, log_y_filename, datetime_name, save_data = True, plot_data = True, positive_factor = 6):
    log_y_name = "log_"+y_name
    # y_names = [i for i in df_full.columns if re.findall(r'^[a-z]_flux_', i)]  
    index = df_y[y_name] == 0 
    df_y.loc[index, y_name] = 10**(-positive_factor+1)

    # Here we intergrated over for geomatrics and convert the unit  first and then take the log   np.log10(x*1e3*4*math.pi))
    df_y[log_y_name] = np.log10(df_y[y_name]) + positive_factor #### Add a factor of 6 here to ensure all data are positive
    
    fulldata_settings["y_names"].append(y_name)
    fulldata_settings["log_y_names"].append("log_"+y_name) #["log_" + str(x) for x in y_name]
    
    if save_data:
        df_y[[datetime_name, y_name, log_y_name]].to_csv(log_y_filename + '.csv', index = False)
        
    if plot_data:
        plot_y_data(df_y, [y_name,log_y_name], fulldata_settings["datetime_name"], log_y_filename)
    
    return df_y, fulldata_settings

def initialize_data_dir(raw_feature_names, number_history, dL01,  forecast, release = "rel05"):    
    directories, mainpath = initialize_fulldata_dir(release = release)
    
    directories["ml_data"] = mainpath + "ml_data/"
    
    for raw_feature_name in raw_feature_names:
        directories["ml_data"] = directories["ml_data"] + raw_feature_name + '_'
        
    directories["ml_data"]  = directories["ml_data"] + "history" + str(number_history) + "days_dL01" + str(dL01) + "_forecast" + forecast +'/'
    
    return directories

def create_directories(dirs):
    for dir in dirs:
        os.makedirs(dir, exist_ok = True)

def initialize_datacsv(mldata_path, species, energy):    
    dataset_csv = {
        "data_settings_filename" : mldata_path + "data_settings",
        "df_y"    : mldata_path + species+'_'+energy + '_' +  "df_y",
        "df_coor" : mldata_path + "df_coor",
        "df_feature" : mldata_path + "df_features",
        "df_data" : mldata_path + species+'_'+energy + '_' +  "df_data.csv",
        "x_train" : mldata_path + species+'_'+energy + '_' + "x_train.csv",
        "y_train" : mldata_path + species+'_'+energy + '_' +"y_train.csv",
        "x_valid" : mldata_path + species+'_'+energy + '_' +  "x_valid.csv",
        "y_valid" : mldata_path + species+'_'+energy + '_' + "y_valid.csv",
        "x_test"  : mldata_path + species+'_'+energy + '_' + "x_test.csv",
        "y_test"  : mldata_path + species+'_'+energy + '_' + "y_test.csv" 
        }
    return dataset_csv

def create_y_name(energy, species, property = 'flux'):
    return species + '_'+property+'_' + energy

def create_log_y_name(energy, species, property = 'flux'):
    return 'log_' + species + '_'+property+'_' + energy

def initialize_data_settings(energy , species, number_history, raw_feature_names, dL01, forecast, l_min = 1, l_max = 8, rel05_invalid_time = ['2017-10-29', '2017-11-01'], test_ts = '2017-01-01', test_te = '2018-01-01'):
    y_name =  create_y_name(energy, species)
    log_y_name = create_log_y_name(energy, species)
    
    data_settings = {
        "energy" : energy,
        "species" : species,
        "raw_feature_names" : raw_feature_names,
        "feature_names" : ["scaled_" + str(x) for x in raw_feature_names],
        "number_history" : number_history,
        "dL01" : dL01,
        "l_min" : l_min,
        "l_max" : l_max,
        "rel05_invalid_time": rel05_invalid_time,
        "y_name":y_name,
        "log_y_name":log_y_name,
        "forecast" : forecast,
        "test_ts" : test_ts,
        "test_te" : test_te
    }
    
    return data_settings

def initialize_data_var(energy, species, raw_feature_names, number_history, dL01, forecast, test_ts = '2017-01-01', test_te = '2018-01-01', release = 'rel05'):
    # directories, fulldataset_csv, fulldata_settings = initialize_fulldata_var(raw_feature_names = raw_feature_names, number_history=number_history)   
    data_directories = initialize_data_dir(raw_feature_names, number_history, dL01,  forecast, release = release)
    create_directories(data_directories.values())
    print(data_directories["ml_data"])
    
    dataset_csv = initialize_datacsv(mldata_path= data_directories["ml_data"], species = species, energy =energy)   

    data_settings = initialize_data_settings(energy=energy , species=species, number_history=number_history, raw_feature_names=raw_feature_names, dL01=dL01, forecast=forecast, test_ts = test_ts, test_te = test_te)
           
    with open(dataset_csv["data_settings_filename"]+'.json', 'w') as file:
        json.dump(data_settings, file, indent = 4)
            
    return data_directories, dataset_csv, data_settings

def load_csv_data(dataset_csv):
    print("start to load csv data")
    x_train = np.genfromtxt(dataset_csv["x_train"], delimiter=',', dtype='float32')
    x_valid = np.genfromtxt(dataset_csv["x_valid"], delimiter=',', dtype='float32')
    x_test = np.genfromtxt(dataset_csv["x_test"], delimiter=',', dtype='float32')
    y_train = np.genfromtxt(dataset_csv["y_train"], delimiter=',', dtype='float32')
    y_valid = np.genfromtxt(dataset_csv["y_valid"], delimiter=',', dtype='float32')
    y_test = np.genfromtxt(dataset_csv["y_test"], delimiter=',', dtype='float32')
    print("csv data loading complete")

    # x_train = pd.read_csv(dataset_csv["x_train"], index_col=False)
    # y_train = pd.read_csv(dataset_csv["y_train"], index_col=False)
    # x_valid = pd.read_csv(dataset_csv["x_valid"], index_col=False)
    # y_valid = pd.read_csv(dataset_csv["y_valid"], index_col=False)
    # x_test = pd.read_csv(dataset_csv["x_test"], index_col=False)
    # y_test = pd.read_csv(dataset_csv["y_test"], index_col=False)
    return x_train, x_valid, x_test, y_train, y_valid, y_test

def create_feature_history_names(fulldata_settings, scaled_feature_name):
    # m_history is the number of feature history
    m_history = int(fulldata_settings["number_history"]*24*60*60/(fulldata_settings["history_resolution"]) + 1)

    # calculate history of the solar wind driver and geomagentic indexes
    feature_history_names = ["" for x in range(m_history)]

    ihf = 0
    for k in range(m_history):
        feature_history_names[ihf] = scaled_feature_name + '_' + str(k*2)+'h'
        ihf = ihf + 1
        
    return feature_history_names

def scale_feature(df_feature, raw_feature_name, fulldata_settings, feature_filename, plot_data = True): 
    scaled_feature_name = "scaled_"+raw_feature_name
    
    df_feature[scaled_feature_name], fulldata_settings = scale_var(df_feature, raw_feature_name, fulldata_settings)
    
    if plot_data:
        plot_feature_data(df_feature, raw_feature_name, scaled_feature_name, fulldata_settings["datetime_name"], feature_filename)
        
    return df_feature, fulldata_settings

def create_feature_history(df_feature, fulldata_settings, raw_feature_name, scaled_feature_name, feature_history_filename, datetime_name, save_data = True):
    
    # Time reslution is set to be two hours for each feature. For each feature, we will add 2 hours earlier of the parametners: feature_0h no delay, feautre_2h, 2 hours before the observing time, feature_4h, 4 hours before the observation time
    n_history_total_days = fulldata_settings["number_history"]
    n_history_total = n_history_total_days*24*60*60/fulldata_settings["average_time"]

    # m_history is the number of feature history we are going to add
    m_history = int(n_history_total/(fulldata_settings["history_resolution"]/fulldata_settings["average_time"]) + 1)

    # calculate history of the solar wind driver and geomagentic indexes
    index_difference = fulldata_settings["history_resolution"]/fulldata_settings["average_time"]
    feature_history_names = ["" for x in range(m_history)]
    
    index1 = df_feature.index[-1]

    arr_history = np.zeros((df_feature.shape[0], m_history))
    
    ihf = 0
    for k in range(m_history):
        feature_history_names[ihf] = scaled_feature_name + '_' + str(k*2)+'h'
        if k == 0:
            arr_history[:,ihf] = np.array(df_feature.loc[:, scaled_feature_name]) 
        else:
            arr_history[:,ihf] = np.concatenate((np.full(int(index_difference*k), np.nan),  np.array(df_feature.loc[0:(index1-index_difference*k), scaled_feature_name])))
        
        ihf = ihf + 1
    
    df_history = pd.concat([df_feature[datetime_name], pd.DataFrame(arr_history, columns=feature_history_names)],axis=1)
    

def load_features(directories, fulldataset_csv, fulldata_settings, recalc = False, df_full = [], save_data = False, plot_data = False, raw_feature_names = np.array(['symh','asyh','asyd','ae','f10.7','kp','swp','swn','swv','by','bz' ])):
    df_features_history = pd.DataFrame()
    for raw_feature_name in raw_feature_names:
        print("start " + raw_feature_name)
        scaled_feature_name = "scaled_" + raw_feature_name
        feature_history_filename =  fulldataset_csv["df_feature"] + "_" + scaled_feature_name 
        
        feature_history_names = create_feature_history_names(fulldata_settings, scaled_feature_name) # feature names to extract

        if os.path.exists(feature_history_filename+'.csv') & (recalc != True):
            print("Reading from "+feature_history_filename+'.csv')
            
            idf_feature_history = pd.read_csv(feature_history_filename+'.csv', index_col=False, usecols=feature_history_names, low_memory=False, dtype = 'float')
        else:        
            print("Calculate the feature history of " + raw_feature_name)
            if len(df_full) == 0:
                df_full = read_probes_data(directories["rawdata_dir"], fulldata_settings)
            idf_feature = df_full[[fulldata_settings["datetime_name"], raw_feature_name]]
            
            idf_feature, fulldata_settings = scale_feature(idf_feature, raw_feature_name, fulldata_settings, feature_history_filename, plot_data = plot_data)
            
            idf_feature_history, fulldata_settings = create_feature_history(idf_feature, fulldata_settings,raw_feature_name, "scaled_"+raw_feature_name, feature_history_filename, fulldata_settings["datetime_name"], save_data = save_data)
            
        df_features_history[feature_history_names] = idf_feature_history
        fulldata_settings["feature_history_names"] = fulldata_settings["feature_history_names"] + feature_history_names
    
    return df_features_history, df_full, fulldata_settings

def load_fulldata(energy =(np.array([51767.680, 44428.696, 38130.120, 32724.498, 28085.268, 24103.668, 20686.558, 17753.876, 15236.896, 13076.798, 11222.936, 9631.899, 8266.406, 7094.516, 6088.722, 5225.528, 4484.742, 3848.919, 3303.284, 2834.964, 2433.055, 2088.129, 1792.096, 1538.062, 1319.977, 1132.846, 972.237]) * 1000.).astype(int).astype(str), species = ['h','o'], recalc = False, release = 'rel05', average_time = 300, raw_coor_names = ["mlt","l","lat"], coor_names=["cos0", 'sin0', 'scaled_lat','scaled_l'], raw_feature_names = ['symh','asyh','asyd','ae','f10.7','kp','swp','swn','swv','by','bz'], number_history = 30, history_resolution = 2*3600., save_data = False, plot_data = False, df_full = [], create_full_data = False):
    
    """ main function to load full data.

    Args:
        energy (_type_): energy channel selected. If input is []. Full energy channels are: (np.array([51767.680, 44428.696, 38130.120, 32724.498, 28085.268, 24103.668, 20686.558, 17753.876, 15236.896, 13076.798, 11222.936, 9631.899, 8266.406, 7094.516, 6088.722, 5225.528, 4484.742, 3848.919, 3303.284, 2834.964, 2433.055, 2088.129, 1792.096, 1538.062, 1319.977, 1132.846, 972.237, 834.421, 716.163, 614.578, 527.484, 452.702, 388.543, 333.459, 286.184, 245.592, 210.769, 180.870, 155.262, 133.243, 114.319, 98.138, 84.209, 72.320, 62.049, 53.255, 45.728, 39.185, 33.627, 28.914, 24.763, 21.246, 18.291, 15.688, 13.437, 11.537, 9.919, 8.512, 7.316, 6.261, 5.347, 4.643, 3.940, 3.377, 2.955, 2.533, 2.181, 1.829, 1.548, 1.337, 1.196, 0.985]) * 1000.).astype(int).astype(str)
        species (_type_): _description_
        recalc (bool, optional): _description_. Defaults to False.
        release (str, optional): _description_. Defaults to 'rel05'.
        average_time (int, optional): _description_. Defaults to 300.
        raw_coor_names (list, optional): _description_. Defaults to ["mlt","l","lat"].
        coor_names (list, optional): _description_. Defaults to ["cos0", 'sin0', 'scaled_lat','scaled_l'].
        raw_feature_names (list, optional): _description_. Defaults to ['symh','asyh','asyd','ae','f10.7','kp','swp','swn','swv','by','bz'].
        number_history (int, optional): Days of history to calculate and store. Defaults to 30.
        history_resolution (_type_, optional): _description_. Defaults to 2*3600..
        save_data (bool, optional): _description_. Defaults to True.
        plot_data (bool, optional): _description_. Defaults to True.

    Returns:
        _type_: _description_
    """

    if type(energy) == str:
        energy = [energy]
    if type(species) == str:
        species = [species]
        
    directories, fulldataset_csv, fulldata_settings = initialize_fulldata_var(release = release, average_time = average_time, raw_coor_names = raw_coor_names, coor_names = coor_names, raw_feature_names = raw_feature_names, number_history = number_history, history_resolution = history_resolution, energy = energy, species = species)

    df_coor, df_full, fulldata_settings = load_coor(directories, fulldataset_csv, fulldata_settings, recalc = recalc, df_full = df_full, save_data = save_data, plot_data = plot_data)
    
    df_y, df_full, fulldata_settings = load_y(directories, fulldataset_csv, fulldata_settings, recalc = recalc, df_full = df_full, save_data = save_data, plot_data = plot_data, energy_bins = energy, species_arr = species)

    # return True

    df_features_history, df_full, fulldata_settings = load_features(directories, fulldataset_csv, fulldata_settings, recalc = recalc, df_full = df_full, save_data = save_data, plot_data = plot_data, raw_feature_names = raw_feature_names)   
    
    if create_full_data == True:
        print("You have calculated full data.")
        return pd.DataFrame(), directories, fulldataset_csv, fulldata_settings
    
    df_data = pd.concat([df_y, df_coor[fulldata_settings['coor_names']], df_features_history[fulldata_settings['feature_history_names']]], axis=1)
    
    return df_data, directories, fulldataset_csv, fulldata_settings

def initialize_fulldatacsv(fulldata_dir):    
    fulldataset_csv = {
        "fulldata_settings_filename" : fulldata_dir + "fulldata_settings",
        "df_y" : fulldata_dir + "df_hope",
        "df_feature" : fulldata_dir + "df_feature_history",
        "df_coor" : fulldata_dir + "df_coor",
        "fulldata_settings": fulldata_dir + "data_setting",
        } 
    return fulldataset_csv
    
def initialize_fulldata_settings(release, average_time, raw_coor_names,  coor_names, raw_feature_names, number_history, history_resolution, energy, species):
    """Here we create the settings to store the attributes of the dataset, selected feature and history, and model.

    Args:
        average_time (float): _description_
        coor_names (array): _description_
        feature_names (array): _description_
        number_history (float): _description_
        history_resolution (float): _description_

    Returns:
        fulldata_settings: _description_
    """
    
    y_names = []
    for ien in energy:
        for isp in species:
            y_names.append( isp +"_flux_" + ien)
    
    fulldata_settings = {
    "release" : release,
    "average_time" : average_time,
    "number_history" : number_history,
    "history_resolution" : history_resolution,
    "raw_coor_names" : raw_coor_names,
    "coor_names": coor_names,
    "raw_feature_names" : raw_feature_names,
    "feature_names" : ["scaled_" + str(x) for x in raw_feature_names],
    "datetime_name" : "DateTime",
    "doubletime_name" : "time",
    "y_names": y_names,
    "log_y_names":["log_" + str(x) for x in y_names],
    "feature_history_names":[]
    }
    
    return fulldata_settings

def initialize_fulldata_var(release= 'rel05', average_time = 300, raw_coor_names= ["mlt","l","lat"], coor_names=["cos0", 'sin0', 'scaled_lat','scaled_l'], raw_feature_names = ['symh','asyh','asyd','ae','f10.7','kp','swp','swn','swv','by','bz'], number_history = 30, history_resolution = 2*3600., energy = (np.array([51767.680, 44428.696, 38130.120, 32724.498, 28085.268, 24103.668, 20686.558, 17753.876, 15236.896, 13076.798, 11222.936, 9631.899, 8266.406, 7094.516, 6088.722, 5225.528, 4484.742, 3848.919, 3303.284, 2834.964, 2433.055, 2088.129, 1792.096, 1538.062, 1319.977, 1132.846, 972.237, 834.421, 716.163, 614.578, 527.484, 452.702, 388.543, 333.459, 286.184, 245.592, 210.769, 180.870, 155.262, 133.243, 114.319, 98.138, 84.209, 72.320, 62.049, 53.255, 45.728, 39.185, 33.627, 28.914, 24.763, 21.246, 18.291, 15.688, 13.437, 11.537, 9.919, 8.512, 7.316, 6.261, 5.347, 4.643, 3.940, 3.377, 2.955, 2.533, 2.181, 1.829, 1.548, 1.337, 1.196, 0.985]) * 1000.).astype(int).astype(str), species = ['h','o']):

    fulldata_directories, mainpath = initialize_fulldata_dir(release)
    create_directories(fulldata_directories.values())
    
    fulldataset_csv = initialize_fulldatacsv(fulldata_directories['fulldata_dir'])   

    fulldata_settings = initialize_fulldata_settings(release, average_time, raw_coor_names,  coor_names, raw_feature_names, number_history, history_resolution, energy, species)
    
    # with open(fulldataset_csv["fulldata_settings_filename"]+'.pkl', 'wb') as file:
    #     pickle.dump(fulldata_settings, file)
    
    with open(fulldataset_csv["fulldata_settings_filename"]+'.json', 'w') as file:
        json.dump(fulldata_settings, file, indent = 4)
    
    return fulldata_directories, fulldataset_csv, fulldata_settings

def create_ml_data(df_data, index_train, index_valid,index_test, y_name, coor_names, history_feature_names):
    y_train = np.array(df_data.loc[index_train, y_name],dtype='float')
    y_valid = np.array(df_data.loc[index_valid, y_name],dtype='float')
    y_test  = np.array(df_data.loc[index_test, y_name],dtype='float')

    x_train = np.array(df_data.loc[index_train, coor_names + history_feature_names], dtype='float')
    x_valid = np.array(df_data.loc[index_valid, coor_names + history_feature_names], dtype='float')
    x_test  = np.array(df_data.loc[index_test,  coor_names + history_feature_names], dtype='float')

    return x_train, x_valid, x_test, y_train, y_valid, y_test

def create_ml_indexes(df_data, data_settings, test_ts, test_te, train_size=0.8):
    """
    Functions for create train and validation data set with a given test data set the function keeps the "episode time" to 2 days

    Args:
        df_data (df): _description_
        test_ts (time string): _description_
        test_te (time string): _description_
        index_good (index): _description_
        train_size (float, optional): _description_. Defaults to 0.8.

    Returns:
        index_train: index of df_data for training data
        index_valid: index of df_data for validation data
        index_test: index of df_data for test data
    """
    
    
    random_seed = 42
    np.random.seed(random_seed)
    
    index_test = (df_data[data_settings['datetime_name']] >= test_ts ) & (df_data[data_settings['datetime_name']] <= test_te )# & index_good

    #If the test set is randomly split
    #episode_train_full,episode_test=train_test_split(episodes, test_size=0.01, train_size=1.0, random_state=42)
    #episode_train,episode_valid=train_test_split(episode_train_full, test_size=0.2, train_size=0.8, random_state=42)
    
    t0 = min(df_data['time'])
    # t1 = max(df_data['time'])

    episode_time= 86400.0*2 # 2 days
    
    # N_episode = np.ceil((t1-t0)/episode_time).astype(int)
    
    df_data['episodes'] = df_data['time'].apply(lambda x: math.floor((x-t0)/episode_time))

    # episode_train, episode_valid = train_test_split(np.unique(df_data.loc[index_good & ~index_test,'episodes']), test_size=1-train_size, train_size=train_size, random_state=42)

    episode_train, episode_valid = train_test_split(np.unique(df_data.loc[~index_test,'episodes']), test_size=1-train_size, train_size=train_size, random_state=42)

    episode_train = np.array(episode_train)
    episode_valid = np.array(episode_valid)
    
    index_train = df_data.loc[:,'episodes'].apply(lambda x: x in episode_train) & ~index_test # & index_good
    index_valid = df_data.loc[:,'episodes'].apply(lambda x: x in episode_valid) & ~index_test  #& index_good

    np.set_printoptions(precision=3,suppress=True)
    print(sum(index_train), sum(index_valid), sum(index_test))   
    
    return index_train, index_valid, index_test

def remove_features_by_time(feature_history_names, pattern):
    matching_feature_names = fnmatch.filter(feature_history_names, pattern)
    for ifeature_name in matching_feature_names:
        feature_history_names.remove(ifeature_name)  
    return feature_history_names 

def remove_index_features_by_time(feature_history_names, pattern):
    matching_feature_names = fnmatch.filter(feature_history_names, pattern)
    for ifeature_name in matching_feature_names:
        if ifeature_name in ['symh','asyh','asyd','ae','kp']:
            feature_history_names.remove(ifeature_name)  
    return feature_history_names 

def get_dL01_mask(df_full):
    l10 = df_full["l"].swifter.apply(lambda x: np.floor(x*10))
    l10_pre = np.append(0,np.array(l10[0:(len(l10)-1)]))
    index_mask = l10 != l10_pre
    
    return index_mask

def get_good_index(df_full, data_settings, fulldata_settings):
    """
    There are non valid data in the situ plasma, geomagnetic indexes data and solar wind data. Sometimes, the solar wind and index data are pre-processed (interpolated etc.) We need data with no NaN or Inf for the model. Indexes of valid data are created. 
    
    We have previousely reviewed that all coordinates data and all indexes data do not have NaN or Inf data. If solar wind parameters are used, we need to add index_good_sw into the final good index.

    """
    print(df_full.shape[0])

    index_good_coor = (df_full['l'] > data_settings["l_min"]) & (df_full['l'] < data_settings["l_max"]) 
    
    index_good_rel05 = ((df_full[fulldata_settings["datetime_name"]] < '2017-10-29') | (df_full[fulldata_settings["datetime_name"]] > '2017-11-01')) 
    
    index_good_y = np.isfinite(df_full[data_settings["y_name"]]) # 
    # print(sum(index_good_coor))
    # print(sum(index_good_rel05))
    # print(sum(index_good_y))
    index_good = index_good_coor & index_good_rel05 & index_good_y
    # print(sum(index_good))
    for raw_feature_name in set(data_settings["raw_feature_names"]):
        index_good = index_good & np.isfinite(df_full[raw_feature_name])
    # print(sum(index_good))
    if data_settings["dL01"]:
        index_good = index_good & get_dL01_mask(df_full)
    
    return index_good

def save_df_data(df_data,  index_train, index_valid, index_test, dataset_csv):
    df_data["index_train"] = index_train
    df_data["index_valid"] = index_valid
    df_data["index_test"] = index_test

    df_data.to_csv(dataset_csv["df_data"], index=False)
    return True

def save_csv_data(x_train, x_valid, x_test, y_train, y_valid, y_test, dataset_csv):
    
    np.savetxt(dataset_csv["x_train"], x_train, delimiter=',', fmt='%f')
    np.savetxt(dataset_csv["y_train"], y_train, delimiter=',', fmt='%f')
    np.savetxt(dataset_csv["x_valid"], x_valid, delimiter=',', fmt='%f')
    np.savetxt(dataset_csv["y_valid"], y_valid, delimiter=',', fmt='%f')
    np.savetxt(dataset_csv["x_test"], x_test, delimiter=',', fmt='%f')
    np.savetxt(dataset_csv["y_test"], y_test, delimiter=',', fmt='%f')

    # pd.DataFrame(x_train).to_csv(dataset_csv["x_train"], index=False) 
    # pd.DataFrame(y_train).to_csv(dataset_csv["y_train"], index=False) 
    # pd.DataFrame(x_valid).to_csv(dataset_csv["x_valid"], index=False) 
    # pd.DataFrame(y_valid).to_csv(dataset_csv["y_valid"], index=False) 
    # pd.DataFrame(x_test).to_csv(dataset_csv["x_test"], index=False) 
    # pd.DataFrame(y_test).to_csv(dataset_csv["y_test"], index=False) 

def view_data(df_full, varnames, ylabels, time_array, figname ='temp'):
    # print("start viewdata")
    nvar = len(varnames)

    fig1, ax1 = plt.subplots(nvar,1, constrained_layout = True)
    fig1.set_size_inches(8, nvar*2)
    
    for ivar in range(len(varnames)):

        varname = varnames[ivar]
        
        print("start plot " + varname)
        
        ax1[ivar].scatter(time_array,df_full[varname],s = 0.1)
        ax1[ivar].set_ylabel(ylabels[ivar])
        
    plt.tight_layout()    
    plt.savefig(figname + ".png", format = "png", dpi = 300, bbox_inches="tight")

def plot_coor_data(df_coor,  coor_names, datetime_name, filename):
    print("start plot coor data")         

    view_data(df_coor,  coor_names, coor_names, df_coor[datetime_name].astype('datetime64[ns]').reset_index(drop=True),  figname = filename)

def plot_feature_data(df_feature, scaled_feature_names, datetime_name, filename):
    print("start plot feature data")         

    view_data(df_feature,  scaled_feature_names, scaled_feature_names, df_feature[datetime_name].astype('datetime64[ns]').reset_index(drop=True),  figname = filename)

def load_ml_dataset(energy, species, recalc = False, plot_data = False, save_data = True, dL01=True, average_time = 300, raw_feature_names = ['symh','asyh','asyd','ae','f10.7','kp','swp','swn','swv','by','bz'],  forecast = "none", number_history = 7, test_ts = '2017-01-01', test_te = '2018-01-01', skip_loading = False):
    
    # np.set_printoptions(precision=4)
    
    data_directories, dataset_csv, data_settings = initialize_data_var(energy=energy, species=species, raw_feature_names = raw_feature_names, forecast = forecast, number_history = number_history, test_ts=test_ts, test_te=test_te, dL01=dL01)
    
    print(dataset_csv["x_train"])

    if os.path.exists(dataset_csv["x_train"]) & (recalc != True):
        if skip_loading == True:
            print(dataset_csv["x_train"] + ' exists ')
            return True
        else:
            x_train, x_valid, x_test, y_train, y_valid, y_test  = load_csv_data(dataset_csv)
    else:
        df_data, directories, fulldataset_csv, fulldata_settings = load_fulldata(energy, species, recalc = False, raw_feature_names = raw_feature_names, number_history = number_history, save_data = save_data, plot_data = plot_data)
        
        df_full = read_probes_data(directories["rawdata_dir"], fulldata_settings)
        
        df_data[[fulldata_settings['doubletime_name']]] = df_full[[fulldata_settings['doubletime_name']]]
                
        index_good = get_good_index(df_full, data_settings, fulldata_settings)

        if data_settings["forecast"] == "all":
            data_settings["feature_history_names"] = remove_features_by_time(fulldata_settings["feature_history_names"], "*_0h")
        elif data_settings["forecast"] == "index":
            data_settings["feature_history_names"] = remove_index_features_by_time(fulldata_settings["feature_history_names"], "*_0h")
        else:
            data_settings["feature_history_names"] = fulldata_settings["feature_history_names"]            
        
        df_data = df_data.loc[index_good,[fulldata_settings['doubletime_name'], fulldata_settings['datetime_name'],data_settings['y_name'], data_settings['log_y_name']]+ fulldata_settings['coor_names']+fulldata_settings['feature_history_names']]

        df_full = df_full.loc[index_good, :]
        
        #-----------------------------
        # After this line, both df_data and df_full only have good data. no index_good should be used.

        #set test set. Here we use one year (2017) of data for test set 
        index_train, index_valid, index_test = create_ml_indexes(df_data,  fulldata_settings, data_settings["test_ts"], data_settings["test_te"])
        
        # Each round, one can only train one y. If train more than one y, need to  repeat from here
        x_train, x_valid, x_test, y_train, y_valid, y_test = create_ml_data(df_data, index_train, index_valid, index_test, data_settings["log_y_name"], fulldata_settings["coor_names"], data_settings["feature_history_names"])  
        
        print("shapes of x_train, x_valid, x_test, y_train, y_valid, y_test ")
        print(x_train.shape, x_valid.shape, x_test.shape, y_train.shape, y_valid.shape, y_test.shape)

        if save_data:
            save_df_data(df_full[[fulldata_settings['datetime_name'], data_settings["y_name"]] + fulldata_settings["raw_coor_names"] + fulldata_settings["raw_feature_names"]], index_train, index_valid, index_test, dataset_csv)

            save_csv_data(x_train, x_valid, x_test, y_train, y_valid, y_test , dataset_csv)
            
        if plot_data:
            plot_y_data(df_data[ [ fulldata_settings['datetime_name'],  data_settings["y_name"],data_settings["log_y_name"]]], data_settings["y_name"],data_settings["log_y_name"],  fulldata_settings['datetime_name'], dataset_csv["df_y"]+ '_'+ data_settings["log_y_name"])
            
            plot_coor_data(df_data[[fulldata_settings['datetime_name']]+fulldata_settings["coor_names"]], fulldata_settings["coor_names"],  fulldata_settings['datetime_name'], dataset_csv["df_coor"])

            to_plot_feature_name = [s + "_2h" for s in fulldata_settings["feature_names"]]
            plot_feature_data(df_data[[fulldata_settings['datetime_name']] + to_plot_feature_name], to_plot_feature_name, fulldata_settings['datetime_name'], dataset_csv["df_feature"])

        
    return x_train, x_valid, x_test, y_train, y_valid, y_test       

raw_feature_names =  [] #['symh','asyh','asyd','ae','f10.7','kp','swp','swn','swv','by','bz']

# [51767.680, 44428.696, 38130.120, 32724.498, 28085.268, 24103.668, 20686.558, 17753.876, 15236.896, 13076.798, 11222.936, 9631.899, 8266.406, 7094.516, 6088.722, 5225.528, 4484.742
# , 3848.919, 3303.284, 2834.964, 2433.055, 2088.129, 1792.096, 1538.062, 1319.977, 1132.846, 972.237]
number_history_arr = [8]
forecast_arr = ["all", "index","none"]
dL01_arr = [True]
species_arr = ['h', 'o']
energy_arr = ['972237', '9631899', '51767680']  


for number_history in number_history_arr:
    for forecast in forecast_arr:
       for dL01 in dL01_arr:
           for species in species_arr:
               for energy in energy_arr:
               
                    load_ml_dataset(energy, species, recalc = True, plot_data = False, save_data = False
                                                          , dL01=dL01, forecast = forecast, number_history =number_history, raw_feature_names =  raw_feature_names)
