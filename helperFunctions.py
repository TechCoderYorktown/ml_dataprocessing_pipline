import json
import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np
import pandas as pd
import os
# import pickle
import re
from sklearn.metrics import r2_score, mean_squared_error


def plot_y_data(df_full, y_names, datetime_name, filename = 'dataview_y'):
    view_data(df_full, y_names, y_names,  df_full[datetime_name].astype('datetime64[ns]').reset_index(drop=True), figname = filename)

def factor_line_calculation(xrange, factor):
    yrange = xrange
    yrangeup = [xrange[0] + np.log10(factor),xrange[1] + np.log10(factor)]
    yrangelow = [xrange[0] - np.log10(factor),xrange[1] - np.log10(factor)]
    return(yrange, yrangeup, yrangelow)

def plot_correlation_heatmap(y_test_reshaped, y_test_pred_reshaped, xrange=[4,9],figname="tmp", data_type =''):
    corr = r2_score(y_test_reshaped, y_test_pred_reshaped)
    mse_test1 = sum((y_test_pred_reshaped-y_test_reshaped)**2)/len(y_test_reshaped)

    #Plot data vs model predictioin
    grid=0.05 
    
    yrange, yrangeup, yrangelow = factor_line_calculation(xrange, 2)
    
    NX=int((xrange[1]-xrange[0])/grid)
    NY=int((yrange[1]-yrange[0])/grid)
    M_test=np.zeros([NX,NY],dtype=np.int16)

    for k in range(y_test_reshaped.size):
        xk = int(np.clip((y_test_reshaped[k] - xrange[0]) / grid, 0, NX-1))
        yk = int(np.clip((y_test_pred_reshaped[k] - yrange[0]) / grid, 0, NY-1))
        M_test[xk, yk] += 1

    extent = (xrange[0], xrange[1], yrange[0], yrange[1])

    # Boost the upper limit to avoid truncation errors.
    levels = np.arange(0, M_test.max(), 200.0)

    norm = mpl.cm.colors.Normalize(vmax=M_test.max(), vmin=M_test.min())

    fig2=plt.figure(figsize=(10, 8),facecolor='w')
    ax1=fig2.add_subplot(1,1,1)

    im = ax1.imshow(M_test.transpose(),  cmap=mpl.cm.plasma, interpolation='none',#'bilinear',
                origin='lower', extent=[xrange[0],xrange[1],yrange[0],yrange[1]],
                vmax=M_test.max(), vmin=-M_test.min())

    ax1.plot(xrange,yrange,'r')
    ax1.plot(xrange, yrangeup,'r',dashes=[3, 3])
    ax1.plot(xrange, yrangelow,'r', dashes = [3,3])

    ax1.set_title(figname)#,fontsize=10)
    ax1.set_xlabel("Measured flux in log10",fontsize=20)
    ax1.set_ylabel("Predicted flux in log10",fontsize=20)

    plt.text(5,3.6,('R2: %(corr)5.3f' %{"corr": corr}),color='k',fontsize=20)
    plt.text(5,2.5,('mse '+data_type+':%(mse_test)5.3f' %{"mse_test":mse_test1}),color='k',fontsize=20)

    # We change the fontsize of minor ticks label 
    ax1.tick_params(axis='both', which='major', labelsize=16)
    ax1.tick_params(axis='both', which='minor', labelsize=12)
    
    #plt.axis('equal')
    plt.xlim(xrange[0],xrange[1])
    plt.ylim(yrange[0],yrange[1])
    cbar=fig2.colorbar(im, ax=ax1)
    cbar.ax.tick_params(labelsize=16)
    cbar.set_label('# of 5-minute data', fontsize=20)
    
    # plt.savefig(figname+".png", format="png", dpi=300)
    plt.show()


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

    if save_data:
        df_history.to_csv(feature_history_filename+ ".csv", index=False)
        print("Writing csv data completed for " + feature_history_filename)
    fulldata_settings["feature_history_names"] = feature_history_names

    return df_history[feature_history_names], fulldata_settings
    
def load_features(directories, fulldataset_csv, fulldata_settings, recalc = False, df_full = [], save_data = False, plot_data = False, raw_feature_names = np.array(['symh','asyh','asyd','ae','f10.7','kp','swp','swn','swv','by','bz' ])):
    
    features_history_list = []
    all_feature_names = []
    df_features_history = pd.DataFrame()

    for raw_feature_name in raw_feature_names:
        print("start " + raw_feature_name)
        scaled_feature_name = "scaled_" + raw_feature_name
        feature_history_filename =  fulldataset_csv["df_feature"] + "_" + scaled_feature_name 
        
        feature_history_names = create_feature_history_names(fulldata_settings, scaled_feature_name) # feature names to extract

        if os.path.exists(feature_history_filename+'.csv') & (recalc != True):
            print("Reading from "+feature_history_filename+'.csv')
            
            idf_feature_history = pd.read_csv(feature_history_filename+'.csv', index_col=False, usecols=feature_history_names, low_memory=False, dtype = 'float')
            print("Finished reading "+raw_feature_name)
        else:        
            print("Calculate the feature history of " + raw_feature_name)
            if len(df_full) == 0:
                df_full = read_probes_data(directories["rawdata_dir"], fulldata_settings)
            idf_feature = df_full[[fulldata_settings["datetime_name"], raw_feature_name]]
            
            idf_feature, fulldata_settings = scale_feature(idf_feature, raw_feature_name, fulldata_settings, feature_history_filename, plot_data = plot_data)
            
            idf_feature_history, fulldata_settings = create_feature_history(idf_feature, fulldata_settings,raw_feature_name, "scaled_"+raw_feature_name, feature_history_filename, fulldata_settings["datetime_name"], save_data = save_data)
            
        idf_feature_history.columns = feature_history_names

        features_history_list.append(idf_feature_history)

        all_feature_names += feature_history_names

    if len(features_history_list):

        df_features_history = pd.concat(features_history_list, axis=1).copy()

        fulldata_settings["feature_history_names"] = all_feature_names
    
    return df_features_history, df_full, fulldata_settings

def plot_feature_data(df_feature, scaled_feature_names, datetime_name, filename):
    print("start plot feature data")         

    view_data(df_feature,  scaled_feature_names, scaled_feature_names, df_feature[datetime_name].astype('datetime64[ns]').reset_index(drop=True),  figname = filename)

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
        print("Reading csv data for probe " + iprobe)
        # df = pd.read_csv(data_dir + 'rbsp' + iprobe.capitalize() + '_data_' + fulldata_settings["release"] + '_fulldata_with_drift.csv') 
        df = pd.read_csv(r'C:\Users\rhett\Downloads\ml_ringcurrent_ion\rbsp' + iprobe.capitalize() + '_full_esw.csv') 
        df['probe'] = iprobe
        if iprobe == probes[0]:
            df_full = df
        else:
            df_full = pd.concat([df_full, df], ignore_index=True)  
    
    # df_full[fulldata_settings["datetime_name"]] = df_full['time'].apply(lambda x : time_string(x)).astype('datetime64[ns]')
    df_full[fulldata_settings["datetime_name"]] = pd.to_datetime(df_full['time'])
    
    return df_full

def initialize_fulldata_dir(release = "rel05"):
    mainpath = "output_" + release +'/'
    directories = {
        "rawdata_dir" : mainpath + "rawdata/", 
        "fulldata_dir" : mainpath + "fulldata/"
        }
    
    return directories, mainpath

def create_directories(dirs):
    for dir in dirs:
        os.makedirs(dir, exist_ok = True)

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
