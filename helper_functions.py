import pandas as pd
import glob
import re
import matplotlib.pyplot as plt
from scipy import stats
import numpy as np

def load_data(path_to_glob, sheet_to_keep):# Read in Data:
    data = []
    for excel_path in glob.glob(path_to_glob):
        data.append(pd.read_excel(excel_path, sheet_name=sheet_to_keep))

    return pd.concat(data, axis=0, ignore_index=True)

def data_cleaning(data): 
    data.set_index('Unnamed: 0', drop=True, inplace=True) # A column, which contains the time, is unnamed -> Fix it here.
    data = data[data.index != 'Zeitstempel'] # A row with information about unit exists -> Drop it here.
    data.index.name = 'Zeitstempel' # Rename Index.
    data.index = pd.to_datetime(data.index)
    data.columns = [re.sub("\\n.*",'',x) for x in data.columns.to_list()]
    return data

def _create_is_summertime_array(datetimeindex, max_time_delta = pd.Timedelta('45min'), summer_start=False):
    """When changing the time-zone, we need to tell pandas if the current row is summertime. This happens here."""
    summer_time = summer_start
    datetime_before = None
    is_summertime = []
    for datetime in datetimeindex:
        if datetime_before is None:
            datetime_before = datetime
            is_summertime.append(summer_time)
            continue
        if datetime == datetime_before or (datetime - datetime_before) > max_time_delta:
            summer_time = not summer_time
        datetime_before = datetime
        is_summertime.append(summer_time)
    return is_summertime

def localize_time(data, from_='CET', to_='UTC'):
    is_summertime = _create_is_summertime_array(data.index)
    data.index = data.index.tz_localize(from_, ambiguous=is_summertime).tz_convert(to_)
    return data

def calc_num_quantiles(data):
    """
    data: pandas data frame
    
    returns: each quantile in the data
    """
    num_quantiles = len(data)
    return np.linspace(0, 1, num = num_quantiles)

def calc_dist_args(dist, data):
    """
    dist: scipy.stats distribution
    data: pandas data frame
    
    returns: fitted arguments of the provided distribution for the data
    """
    distargs = dist.fit(data)
    distargs = distargs[:len(distargs)-2]
    return distargs

def calculate_m_b(x, y):
    # m and b is needed for linear regression
    """
    x, y: 
    """
    y = y[np.isinf(x)==False]
    x = x[np.isfinite(x)]
    n = len(x)
    m = np.divide((n * np.sum(np.multiply(x, y)) - np.sum(x) * np.sum(y)),
                     (n * np.sum(np.square(x)) - np.square(np.sum(x))))
    b = np.divide(np.sum(np.square(x)) * np.sum(y) - np.sum(x) * np.sum(x * y),
                     (n * np.sum(np.square(x)) - np.square(np.sum(x))))
    # R2 of regression slope
    y_regressed = x * m + b
    y_mean = np.sum(y)/n
    r2error = np.sum(np.square(y_regressed-y))
    r2error /= np.sum(np.square(y-y_mean))
    
    return m, b, 1 - r2error

def qq(data, dist):
    """
    
    """
    
    #Log data if dist.lognorm == True
    is_log_norm = False
    if dist == stats.lognorm:
        data = np.log(data)
        dist = stats.norm
        is_log_norm = True


    #Anzahl Quantile berechnen
    distargs = calc_dist_args(dist, data)
    quantile_list = calc_num_quantiles(data)


    #Bereiche unter der Kurve bestimmen, die eine gleichgrosse Wahrscheinlichkeit haben
    quantile_distribution = dist.ppf(quantile_list, *distargs, loc = 0, scale = 1)
    
    #Linearen Fit berechnen
    m, b, r2_score = calculate_m_b(quantile_distribution[:-1], data[:-1])
    if is_log_norm:
        dist = stats.lognorm
        
    #print(quantile_list)
    #print(quantile_distribution)
    #print(np.quantile(zone1['mass'], quantile_list))
    return r2_score, dist, m, b

def easy_qq(dists, data):
    """
    creates qq plots for multiple distributions at once and displays them.
    
    dists: list of scipy.stats distrubutions
    data: pandas data frame
    
    returns:
    
    r2_best: best r2score
    best_dist: distribution with best r2 score
    r2_list: list of r2 scores
    unusable_dist: list of distributions the function was unable to fit
    error_messages: eventual error messages for the distributions
    distargs: the fitted distributions arguments
    """
    r2_best = -np.inf
    best_dist = "All suck lmao"
    data = np.sort(data)
    
    #Generate highiq plots
    
    r2_list = []
    distributions = []
    ms = []
    bs = []
    unusable_dist = []
    error_messages = []
    
    for dist in dists:
        try:
            r2_now, dist_now, m, b = qq(data, dist)
            r2_list.append(r2_now)
            distributions.append(dist)
            ms.append(m)
            bs.append(b)
            if r2_now > r2_best:
                r2_best = r2_now
                best_dist = dist_now
        except Exception as e:
            print(f'Error with {dist.name}: {e}')
            error_messages.append(e)
            unusable_dist.append(dist.name)
            continue

    print("Best dist:", best_dist.name)
    data_to_sort = np.column_stack((r2_list, distributions, ms, bs))
    data_sorted = data_to_sort[np.argsort(-data_to_sort[:,0])]
    
    quantile_list = calc_num_quantiles(data)
    n_data = len(distributions)
    nr_cols = int(np.rint(np.sqrt(n_data)))
    nr_rows = int(np.ceil(np.sqrt(n_data)))
    
    row = 0 
    col = 0
    _, axes = plt.subplots(nr_rows, nr_cols, figsize = [nr_rows*5,nr_rows*5])
    
    for r2_score, distribution, m, b in data_sorted:
        is_log_norm = False
        if distribution == stats.lognorm:
            distribution = stats.norm
            data = np.log(data)
            is_log_norm = True
            
        distargs_fit = calc_dist_args(distribution, data)    
        quantile_distribution = distribution.ppf(quantile_list, *distargs_fit, loc = 0, scale = 1)
        if not is_log_norm:
            axes[row, col].scatter(quantile_distribution, data)
            axes[row, col].plot(quantile_distribution[:-1], quantile_distribution[:-1] * m + b)
            axes[row, col].set_title('Verteilung: ' + str(distribution.name)+ '. R2-score: ' + str(np.round(r2_score,3)))
        else:
            axes[row, col].scatter(quantile_distribution, data)
            axes[row, col].plot(quantile_distribution[:-1], quantile_distribution[:-1] * m + b)
            axes[row, col].set_title('Verteilung: ' + str(stats.lognorm.name) + '. R2-score: ' + str(np.round(r2_score,3)))
            distribution = stats.lognorm
            data = np.exp(data)
        
        col += 1
        if col == nr_cols:
            col = 0
            row += 1

    plt.show()
    return r2_best, best_dist, r2_list, unusable_dist, error_messages



# Create Features:

def _day_night_cycle(datetimeindex, ratio_solar_winter, longest_day_year = 172):
    day_of_year = datetimeindex.day_of_year

    day_of_year = np.roll(day_of_year.to_numpy(), longest_day_year*4*24) # 15 min intervall: 15 min * 4 * 24 == 24 hours == 1 day
    x_norm = 2*np.pi * day_of_year / day_of_year.max()

    amplitude = (1-ratio_solar_winter) / 2
    middle = (1 + ratio_solar_winter) / 2

    cos_x = np.cos(x_norm) * amplitude + middle

    return cos_x

def create_features(df_, type_, ratio_solar = 0.3, ratio_solar_winter=0.2):
    """Function to generate the data for different scenarios. 
    df_: dataframe generated by the function load_data().
    type_: for which scenario we generate data.
    ratio_solar_winter: how much less electricity gets generated in winter. So if in summer we generate 100%, a value of 0.2 means that we generate 20% instead of 100% in winter. """
    df = df_.copy()
    if type_ == 'normal':
        df['Strombudget Schweiz'] = df['Summe produzierte Energie Regelblock Schweiz'] + df['Import']
        df['Stromverbrauch Schweiz'] = df['Summe endverbrauchte Energie Regelblock Schweiz'] + df['Export']

        df['Strombilanz Schweiz'] = df['Strombudget Schweiz'] - df['Stromverbrauch Schweiz']
        # Select useful Features:
        return df[['Strombilanz Schweiz', 'Strombudget Schweiz', 'Stromverbrauch Schweiz']]

    elif type_ == 'isoliert':
        df['Strombudget Schweiz'] = df['Summe produzierte Energie Regelblock Schweiz'] 
        df['Stromverbrauch Schweiz'] = df['Summe endverbrauchte Energie Regelblock Schweiz']

        df['Strombilanz Schweiz'] = df['Strombudget Schweiz'] - df['Stromverbrauch Schweiz']
        return df[['Strombilanz Schweiz', 'Strombudget Schweiz', 'Stromverbrauch Schweiz']]
    elif type_ == 'solar':
        _factor = _day_night_cycle(df.index, ratio_solar_winter)
        df['Solarstrom Schweiz'] = ratio_solar * df['Summe produzierte Energie Regelblock Schweiz'] * _factor
        df['Strombudget Schweiz'] = df['Summe produzierte Energie Regelblock Schweiz'] * (1-ratio_solar) + df['Import'] + df['Solarstrom Schweiz']
        df['Stromverbrauch Schweiz'] = df['Summe endverbrauchte Energie Regelblock Schweiz'] + df['Export']

        df['Strombilanz Schweiz'] = df['Strombudget Schweiz'] - df['Stromverbrauch Schweiz']
        return df[['Strombilanz Schweiz', 'Strombudget Schweiz', 'Stromverbrauch Schweiz', 'Solarstrom Schweiz']]

    assert type_ in ['normal', 'isoliert', 'solar'], "Error! Type_ has to be either normal, isoliert or solar."
