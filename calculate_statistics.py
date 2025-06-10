import numpy as np
import pandas as pd
from pymannkendall import original_test as mk_test
import os


boxes = ['box_A','box_B','box_C','box_D','box_E']
regions = ['North America', 'Sahara', 'West Asia', 'Southern Africa', 'Australia']

# read in the data
data_dir = ''

# make directories
if not os.path.exists(data_dir+'/tables'):
    os.makedirs(data_dir+'/tables')
if not os.path.exists(data_dir+'/plots'):
    os.makedirs(data_dir+'/plots')

# add basic info to text files ################################################

for box in boxes:

    print('Processing box: ', box)
    print('Region: ', regions[boxes.index(box)])

    # read in the data
    df = pd.read_csv(f'{data_dir}/results/{box}_HL_stats.csv')
    df = df.rename(columns={'Unnamed: 0': 'date'})

    # Step 1: Extract year and month
    df["year"] = pd.to_datetime(df["date"]).dt.year
    df["month"] = pd.to_datetime(df["date"]).dt.month

    print(df)

    daily_stats = df

    # save the stats to a text file
    with open(f'{data_dir}/results/{box}_HLs.txt', 'w') as f:
        print('Basic info for ', box, file=f)
        print('Total number of years: ', daily_stats['year'].nunique(), file=f)
        print('Range of years: ', daily_stats['year'].min(), '-', daily_stats['year'].max(), file=f)
        print('Average LLAT range: ', daily_stats['avg_LLAT'].min(), '-', daily_stats['avg_LLAT'].max(), file=f)
        print('Max LLAT range: ', daily_stats['max_value'].min(), '-', daily_stats['max_value'].max(), file=f)
        print('Minimum number of cells: ', daily_stats['n_cells'].min(), 'equivalent to ', daily_stats['heat_low_size'].min(), ' km^2', file=f)
        print('Maximum number of cells: ', daily_stats['n_cells'].max(), 'equivalent to ', daily_stats['heat_low_size'].max(), ' km^2 , i.e. ', np.sqrt(daily_stats['heat_low_size'].max()), 'x ', np.sqrt(daily_stats['heat_low_size'].max()), ' km', file=f)
        print('Total number of days: ', daily_stats['date'].count(), file=f)

# calculate annual mann kendall trends for each region ##########################

# function to apply Mann-Kendall test for monotonic trend detection
def mann_kendall_trend(time_series):
    result = mk_test(time_series, alpha=0.05)  # significance level 5%
    return {
        "trend": result.trend,  # 'increasing', 'decreasing', or 'no trend'
        "p_value": result.p,  # statistical significance
        "slope": result.slope  # Sen's slope
    }

# initialize a dictionary to store data for each variable across regions
data_by_variable = {
    "avg_LLAT": {},
    "max_value": {},
    "size": {},
    "frequency": {}
}

# process each box
for box in boxes:
    
    # Read in the data
    df = pd.read_csv(f'{data_dir}/results/{box}_HL_stats.csv')
    df = df.rename(columns={'Unnamed: 0': 'date'})
    
    # Remove any row where n_cells is less than or equal to 10
    df = df[df['n_cells'] > 10]
    
    # Extract year and month
    df["year"] = pd.to_datetime(df["date"]).dt.year
    df["month"] = pd.to_datetime(df["date"]).dt.month
    
    # Aggregate data by year
    avg_LLAT = df.groupby("year")["avg_LLAT"].mean()
    max_value = df.groupby("year")["max_value"].mean()
    size = df.groupby("year")["heat_low_size"].mean()
    frequency = df.groupby("year").size()  # frequency is the count of rows
    
    # Store the anomalies for each variable in the dictionary
    data_by_variable["avg_LLAT"][box] = avg_LLAT
    data_by_variable["max_value"][box] = max_value
    data_by_variable["size"][box] = size
    data_by_variable["frequency"][box] = frequency


# initialize a dictionary for filtered data
filtered_data_by_variable = {
    "avg_LLAT": {},
    "max_value": {},
    "size": {},
    "frequency": {}
}

# compute trends
results = {}
for variable in data_by_variable.keys():
    results[variable] = {}
    for box in boxes:
        series = data_by_variable[variable][box]
        results[variable][box] = {"Mann-Kendall": mann_kendall_trend(series)}

# save results to csv
results_df = pd.DataFrame(results)
results_df = results_df.stack().apply(pd.Series)
results_df = results_df.stack().apply(pd.Series)
results_df = results_df.stack().apply(pd.Series)
results_df.to_csv(f'{data_dir}/tables/trends_stats_ERA5.csv')

# calculate seasonal mann kendall trends for each box ##########################

# initialize dictionaries for storing results
size_dict = {}  
freq_dict = {}  

# process each region
for box in boxes:
    
    # read in the data
    df = pd.read_csv(f'{data_dir}/results/{box}_HL_stats.csv')
    df = df.rename(columns={'Unnamed: 0': 'date'})

    # extract year and month
    df["year"] = pd.to_datetime(df["date"]).dt.year
    df["month"] = pd.to_datetime(df["date"]).dt.month

    # compute frequency (number of heat low days per month per year)
    freq_df = df.groupby(["year", "month"]).size().reset_index(name="frequency")
    freq_df = freq_df.pivot_table(index="year", columns="month", values="frequency", fill_value=0)
    freq_df = freq_df.reindex(columns=range(1, 13), fill_value=0)  # Ensure all months are present

    # compute size (average heat low size per month per year)
    size_df = df.groupby(["year", "month"])["heat_low_size"].mean().reset_index()
    size_df = size_df.pivot_table(index="year", columns="month", values="heat_low_size", fill_value=0)
    size_df = size_df.reindex(columns=range(1, 13), fill_value=0)  # Ensure all months are present
    size_df.replace(0, np.nan, inplace=True)

    # store the results
    size_dict[box] = size_df
    freq_dict[box] = freq_df


# initialize dictionaries for storing results
results_freq = {}
results_size = {}

for box in boxes:

    box_results_freq = {}
    box_results_size = {}

    freq_df = freq_dict[box]
    size_df = size_dict[box]

    for month in range(1, 13):

        # extract data for the month
        freq_data = freq_df[month]
        size_data = size_df[month]

        if len(freq_data[freq_data > 0]) > 10:

            # compute Mann Kendall for that month
            freq_trend = mk_test(freq_data)
            freq_trend = dict(zip(freq_trend._fields, freq_trend))

        else:
            freq_trend = {
                "trend": np.nan,
                "h": np.nan,
                "p": np.nan,
                "z": np.nan,
                "Tau": np.nan,
                "s": np.nan,
                "var_s": np.nan,
                "slope": np.nan,
                "intercept": np.nan
            }

        if len(size_data[size_data > 0]) > 10:
            size_trend = mk_test(size_data)
            size_trend = dict(zip(size_trend._fields, size_trend))

        else:
            # assign NaNs if not enough data points
            size_trend = {
                "trend": np.nan,
                "h": np.nan,
                "p": np.nan,
                "z": np.nan,
                "Tau": np.nan,
                "s": np.nan,
                "var_s": np.nan,
                "slope": np.nan,
                "intercept": np.nan
            }

        # save the results in dataframe
        box_results_freq[month] = freq_trend
        box_results_size[month] = size_trend

    results_freq[box] = pd.DataFrame.from_dict(box_results_freq, orient='index')
    results_size[box] = pd.DataFrame.from_dict(box_results_size, orient='index')

    # save to csv
    results_freq[box].to_csv(f'{data_dir}/tables/{box}_freq_trend.csv')
    results_size[box].to_csv(f'{data_dir}/tables/{box}_size_trend.csv')
