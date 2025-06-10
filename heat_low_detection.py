""" 

This script is designed to detect heat lows in ERA5 data. 

Global ERA5 data requirements (see README):
- Daily LLAT data per region located at: {data_dir}{box}_LLAT_1980_2024.nc
- Global daily vertical velocity data (300 hPa) in yearly files at: {daily_w_dir}era5_daily_vertical_velocity_{year}.nc
- Global land-sea mask file located at: {data_dir}era5_land_sea_mask.nc
- ERA5 climatological surface pressure at: {data_dir}era5_avg_sp_1980-2024.nc

Contact: kitty.attwood@ouce.ox.ac.uk


"""

# import packages
import xarray as xr
import numpy as np
import os
import pandas as pd
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
from scipy.ndimage import label

# set the directories
daily_z_dir = '/' # daily geopotential (z) directory
daily_w_dir = '/' # daily vertical velocity (w) directory
dir = '/' # root output directory
output_dir = dir + 'results/' # results output directory
data_dir = dir + 'data/' # data directory

# create the output directories if they don't exist
if not os.path.exists(output_dir):
    os.makedirs(output_dir)
if not os.path.exists(dir+'/checks'):
    os.makedirs(dir+'checks')

# get land sea mask
land_sea_mask = xr.open_dataset(data_dir+'era5_land_sea_mask.nc')

# make sure longitude is -180 to 180
land_sea_mask = land_sea_mask.assign_coords(longitude=(((land_sea_mask.longitude + 180) % 360) - 180)).sortby('longitude')

# define the regions and their levels/coordinates
boxes = ['box_A','box_B','box_C','box_D','box_E']
regions = ['North America','Sahara','West Asia','Southern Africa','Australia']
llat_levels = [[800,650],[900,750],[800,650],[850,700],[900,750]]
min_lon = [-125,-20,30,10,112]
max_lon = [-85,30,75,40,147]
max_lat = [50,40,40,0,-10]
min_lat = [10,5,5,-35,-35]

# open a text file to write the output
for box in boxes:
    with open(f'{output_dir}{box}_HLs.txt', 'w') as f:
        f.write(f'Heat Lows for {box} - {regions[boxes.index(box)]}\n\n')


def define_LLAT_threshold(box):

    box_index = boxes.index(box)

    """

    Defines the LLAT threshold using the 95th percentile of the daily LLAT data for that region.
    - Masks for land grid points only
    - Removes gridpoints below average surface pressure
    * Inputs: global daily LLAT timeseries, land sea mask
    * Outputs: 95th and 99th percentiles of LLAT at each location
    * Saves elevation mask to netCDF, saves land mask image to check
    
    """

    # get daily LLAT timeseries
    LLAT_timeseries = xr.open_dataset(f'{data_dir}/{box}_LLAT_1980_2024.nc')
    # make sure longitude is -180 to 180
    LLAT_timeseries = LLAT_timeseries.assign_coords(longitude=(((LLAT_timeseries.longitude + 180) % 360) - 180)).sortby('longitude')
    # remove level
    LLAT_timeseries = LLAT_timeseries.squeeze()
    # remove bnds dimension
    if 'time_bnds' in LLAT_timeseries:
        LLAT_timeseries = LLAT_timeseries.squeeze('time_bnds')
    # make sure longitude is -180 to 180
    land_sea_mask = land_sea_mask.assign_coords(longitude=(((land_sea_mask.longitude + 180) % 360) - 180)).sortby('longitude')
    # crop to the same lat lon as LLAT timeseries
    land_sea_mask = land_sea_mask.sel(latitude=LLAT_timeseries.latitude, longitude=LLAT_timeseries.longitude)
    # print(land_sea_mask.lsm)
    # apply
    LLAT_timeseries = LLAT_timeseries.where(land_sea_mask.lsm.squeeze() > 0.4, np.nan)

    # check land mask is applied - plot LLAT figure at timestep 1
    fig, ax = plt.subplots(subplot_kw={'projection': ccrs.PlateCarree()})
    LLAT_timeseries['z'][0].plot(ax=ax, transform=ccrs.PlateCarree())
    ax.coastlines()
    plt.savefig(f'{dir}/checks/{box}_land_mask.png')

    # elevation mask
    sp_box = xr.open_dataset(data_dir+'era5_avg_sp_1980-2024.nc')
    # make sure longitude is -180 to 180
    sp_box = sp_box.assign_coords(longitude=(((sp_box.longitude + 180) % 360) - 180)).sortby('longitude')
    # convert to hPa
    sp_box = sp_box/100
    # print(sp_box)
    # crop to the same lat lon as LLAT timeseries
    sp_box = sp_box.sel(latitude=LLAT_timeseries.latitude, longitude=LLAT_timeseries.longitude)
    # mask where surface pressure is below the level
    LLAT_timeseries = LLAT_timeseries.where(sp_box.sp.squeeze() > llat_levels[box_index][0], np.nan)
    
    # check dates
    print('date range of LLAT timeseries for threshold calc:', LLAT_timeseries.time.values.min(), LLAT_timeseries.time.values.max())

    elevation_mask = sp_box.sp.squeeze() > llat_levels[box_index][0]
    # crop to the same lat lon as LLAT timeseries
    elevation_mask = elevation_mask.sel(latitude=LLAT_timeseries.latitude, longitude=LLAT_timeseries.longitude)
    elevation_mask.to_netcdf(f'{data_dir}{box}_elevation_mask.nc')

    print(elevation_mask)

    # flatten array across all timesteps and space
    llat_np = LLAT_timeseries.to_array().values.flatten()

    # get the 99th and 95th percentiles of the daily LLAT data
    LLAT_99th = np.nanpercentile(llat_np, 99)
    LLAT_95th = np.nanpercentile(llat_np, 95)

    with open(f'{output_dir}{box}_HLs.txt', 'w') as f:
        f.write(f'99th percentile LLAT (1980-2024): {LLAT_99th}\n')
        f.write(f'95th percentile LLAT (1980-2024): {LLAT_95th}\n')

    print('99th percentile LLAT (1980-2024):', LLAT_99th)
    print('95th percentile LLAT (1980-2024):', LLAT_95th)

    return LLAT_99th, LLAT_95th
    

def apply_thresholds(daily_LLAT, daily_w, LLAT_threshold, grid_area):

    """

    Applies thresholds to daily LLAT data and returns masked data and statistics.
    1. LLAT threshold applied to daily LLAT field using define_LLAT_threshold()
    2. Grid points above threshold screened for upper level subsidence
    3. Remaining grid points assessed for size (25,000km2 equivalent, contiguous grid cells)
    4. Daily LLAT mask for each HL day is saved as a .nc file, daily statistics are saved in a .csv file 

    """
    
    # print the time range of the data
    print('Processing data from ', daily_LLAT.time.values.min(), 'to', daily_LLAT.time.values.max())
    print('Calculating LLAT...')

    # create an empty list to store the final masked data
    final_masked_data = []

    # create a dataframe to store daily statistics
    daily_stats = pd.DataFrame(columns=['n_cells', 'avg_LLAT', 'max_value', 'max_value_lat', 'max_value_lon', 'heat_low_size'])

    # loop through each timestep
    for timestep in daily_LLAT.time.values:
            
        print('Step 1, processing', timestep)

        # mask ocean values
        daily_LLAT_land_only = daily_LLAT.sel(time=timestep).where(land_sea_mask.lsm > 0.4)

        # STEP 1: apply the LLAT threshold #################################################

        # mask out values below the threshold, discard day if no values exceed the threshold
        daily_LLAT_above_threshold = daily_LLAT_land_only.where(daily_LLAT_land_only > LLAT_threshold, np.nan)
        
        # check if there are any grid points above the threshold
        if daily_LLAT_above_threshold.count() == 0:
            print('No grid points above the LLAT threshold for', timestep)
            continue
     
        print('Grid points above the threshold for', timestep, 'continuing to step 2...')

        # STEP 2: apply vertical velocity criteria ########################################

        # get just date from timestep
        timestep_date = pd.to_datetime(timestep)
        timestep_date = timestep_date.strftime('%Y-%m-%d')
        print('Step 2 - checking vertical velocity:', timestep_date)
        
        # check if the date is in the daily_w dataset
        w = daily_w.sel(time=timestep_date)
        date = pd.to_datetime(timestep)
        date = date.strftime('%Y-%m-%d')
        
        # keep daily_LLAT_masked where w (vertical velocity at 300 hPa) > 0
        daily_LLAT_checked_subsidence = daily_LLAT_above_threshold.where(w > 0, np.nan)


        # STEP 3: apply criteria for size of the area #####################################

        # get the grid box area, ensure longitude is -180 to 180
        grid_area = grid_area.assign_coords(longitude=(((grid_area.longitude + 180) % 360) - 180)).sortby('longitude')
        grid_cell_area = grid_area['cell_area'] 

        grid_cell_area = grid_cell_area.values/1000000 # convert to km^2 from m^2
        
        # Create a mask where LLAT (z variable) is not NaN
        mask = daily_LLAT_checked_subsidence.notnull().values
        mask = mask.squeeze()  # remove time dimension

        # label connected components (8-connectivity: horizontal/vertical/diagonal) to ensure HL is made of contiguous grid points
        structure = np.ones((3, 3))  # defines connectivity structure for 8-connected components
        labeled_mask, num_features = label(mask, structure=structure)

        # initialize a mask to store valid regions
        final_region_mask = np.zeros_like(daily_LLAT_checked_subsidence.isel(time=0), dtype=bool)

        # flag to check if any valid region exists
        valid_region = False

        # define the minimum grid area in km^2
        size_threshold = 25000  # minimum size of the heat low in km^2

        # iterate through each connected component
        for region_id in range(1, num_features + 1):  # region_id starts from 1
            region_mask = (labeled_mask == region_id)
            region_area = (grid_cell_area * region_mask).sum()

            # if the region area exceeds the threshold, add it to the final mask
            if region_area >= size_threshold:
                final_region_mask = np.logical_or(final_region_mask, region_mask)  # add the valid region to the mask
                valid_region = True  # mark that this timestep has at least one valid region

        # break if no valid regions are found
        if valid_region is False:
            print('Step 3 (size): no valid regions found for', timestep)
            continue

        final_region_mask = np.expand_dims(final_region_mask, axis=0)  # Adds time dimension

        final_region_mask = xr.DataArray(
            final_region_mask,
            coords={'time': [timestep], 'latitude': daily_LLAT_checked_subsidence.latitude, 'longitude': daily_LLAT_checked_subsidence.longitude},
            dims=['time', 'latitude', 'longitude']
        )

        daily_LLAT_checked_size = daily_LLAT_checked_subsidence.where(final_region_mask.squeeze(), np.nan)

        # STEP 4: save the data ###########################################################
            
        # Add the time coordinate back explicitly if it gets lost
        daily_LLAT_checked_size = daily_LLAT_checked_size.assign_coords(time=[timestep])

        # save the daily LLAT field of the heat low
        final_masked_data.append(daily_LLAT_checked_size)

        # calculate the number of grid cells, their average LLAT, the location of the cells, the max value and its coords
        n_cells = daily_LLAT_checked_size.count().values
        avg_LLAT = daily_LLAT_checked_size.mean().values
        max_value = daily_LLAT_checked_size.max().values
        masked_area = grid_cell_area * daily_LLAT_checked_size.notnull().squeeze()
        heat_low_size = masked_area.sum(dim=["latitude", "longitude"])
        print('Heat low size:', heat_low_size.values)
        print('Number of cells:', n_cells)

        # get the lat and lon of the max value
        max_value_lat = daily_LLAT_checked_size.where(daily_LLAT_checked_size == max_value, drop=True).latitude
        max_value_lon = daily_LLAT_checked_size.where(daily_LLAT_checked_size == max_value, drop=True).longitude

        # add to row of daily_stats dataframe, index is the date
        daily_stats.loc[date] = [n_cells, avg_LLAT, max_value, max_value_lat.values, max_value_lon.values, heat_low_size.values]
    

    # concatenate all timesteps along the time dimension
    if final_masked_data:
        final_masked_data = xr.concat(final_masked_data, dim='time')
        print('Final masked data:', final_masked_data)
    else:
        print('No valid data after applying thresholds.')
        final_masked_data = xr.DataArray()

    print('Daily stats:', daily_stats)

    return daily_stats, final_masked_data


def get_heat_lows(box, years, months, threshold):

    """ 

    Final function that gets heat lows for a given box, years, months and LLAT threshold 
    Returns masked data in a data array and daily statistics

    """

    # get index
    box_index = boxes.index(box)
    print(box_index)

    masked_data_list = []
    daily_stats_list = []


    # get files for region ###############################################################

    # get daily LLAT timeseries
    LLAT_timeseries = xr.open_dataset(f'{data_dir}/{box}_LLAT_1980_2024.nc')

    # make sure longitude is -180 to 180
    LLAT_timeseries = LLAT_timeseries.assign_coords(longitude=(((LLAT_timeseries.longitude + 180) % 360) - 180)).sortby('longitude')
    
    # remove level
    LLAT_timeseries = LLAT_timeseries.squeeze()

    # remove bnds dimension
    if 'time_bnds' in LLAT_timeseries:
        LLAT_timeseries = LLAT_timeseries.squeeze('time_bnds')

    # create grid cell area file
    if not os.path.exists(f'{data_dir}{box}_gridarea.nc'):
        cmd = f'cdo gridarea {data_dir}/{box}_LLAT_1980_2024.nc {data_dir}{box}_gridarea.nc'
        os.system(cmd)

    grid_area_file = xr.open_dataset(f'{data_dir}{box}_gridarea.nc') # open grid area file created using CDO gridarea function

    # get global average surface pressure file
    sp = xr.open_dataset(data_dir+'era5_avg_sp_1980-2019.nc')

    # make sure longitude is -180 to 180
    sp = sp.assign_coords(longitude=(((sp.longitude + 180) % 360) - 180)).sortby('longitude')

    # apply elevation mask ###############################################################

    # get average surface pressure
    sp_box = sp.sel(latitude=slice(LLAT_timeseries.latitude.max(),LLAT_timeseries.latitude.min()), longitude=slice(LLAT_timeseries.longitude.min(),LLAT_timeseries.longitude.max()))
    sp_box = sp_box/100

    # mask where surface pressure is below the lower pressure level
    LLAT_timeseries = LLAT_timeseries.where(sp_box.sp.squeeze() > llat_levels[box_index][0], np.nan)

    # check elevation mask is applied - plot LLAT timeseries [0] figure
    fig, ax = plt.subplots(subplot_kw={'projection': ccrs.PlateCarree()})
    LLAT_timeseries['z'][0].plot(ax=ax, transform=ccrs.PlateCarree())
    ax.coastlines()
    plt.savefig(f'{dir}checks/{box}_elevation_mask.png')
    plt.close()

    # loop through each year and month ###################################################

    for year in years:
        
        print(year)

        yearly_masked_data = []

        # fix this awkward double file structure ***
        daily_z = LLAT_timeseries.sel(time=LLAT_timeseries.time.dt.year.isin([int(year)]), drop=True)
        daily_w = xr.open_dataset(os.path.join(daily_w_dir, f'era5_daily_vertical_velocity_{year}.nc'))

        # make sure longitude is -180 to 180
        daily_w = daily_w.assign_coords(longitude=(((daily_w.longitude + 180) % 360) - 180)).sortby('longitude')

        # keep only date
        daily_w["time"] = daily_w["time"].dt.floor("D")  # Truncate to date (removes time)
        daily_w = daily_w.assign_coords(time=daily_w["time"].values)  # Ensure coords are updated

        daily_z = daily_z.sel(latitude=slice(max_lat[box_index], min_lat[box_index]), longitude=slice(min_lon[box_index], max_lon[box_index]))
        daily_w = daily_w.sel(latitude=slice(max_lat[box_index],min_lat[box_index]), longitude=slice(min_lon[box_index], max_lon[box_index]))

        for month in months:

            daily_z_crop = daily_z['z'].sel(time=daily_z.time.dt.month.isin([int(month)]), drop=True)

            # file is already pre-processed for LLAT
            daily_LLAT = daily_z_crop
            # make sure longitude is -180 to 180
            daily_LLAT = daily_LLAT.assign_coords(longitude=(((daily_LLAT.longitude + 180) % 360) - 180)).sortby('longitude')

            daily_w_crop = daily_w.sel(time=daily_w.time.dt.month.isin([int(month)]), drop=True)
            daily_w_crop = daily_w_crop['w'].sel(level=300)

            daily_stats, masked_data = apply_thresholds(daily_LLAT, daily_w_crop, threshold, grid_area_file)

            daily_z.close()
            daily_w.close()

            if masked_data is not None and masked_data.dims and masked_data.count() > 0:
                masked_data_list.append(masked_data)  # Add valid data
                yearly_masked_data.append(masked_data)
                daily_stats_list.append(daily_stats)
            else:
                print(f'No data for {year}-{month}, skipping.')
    
        # save yearly results to NetCDF
        if yearly_masked_data:
            yearly_data = xr.concat(yearly_masked_data, dim='time')
            yearly_data.to_netcdf(f'{output_dir}/{box}_LLAT_HLs_{year}.nc')
            print(f"Saved yearly data for {year} to NetCDF.")

    annual_masked_data = xr.concat(masked_data_list,dim='time')
    annual_masked_data = annual_masked_data.assign_attrs({
        "long_name": "LLAT Masked Data",
        "standard_name": "llat", 
        "units": "m",
        "title": "Heat Lows",
        "source": "ERA5 data"
    })

    annual_masked_data = annual_masked_data.transpose("time", "latitude", "longitude")

    return annual_masked_data, daily_stats_list


# apply functions to period of interest

for box in boxes:

    # get the LLAT thresholds for the box from the monthly data
    LLAT_99th, LLAT_95th = define_LLAT_threshold(box)

    # define years and months of interest
    years = [str(year) for year in range(1980,2025)] # get years from 1980 to 2024 inclusive
    months = ['01','02','03','04','05','06','07','08','09','10','11','12']

    # get the heat lows for the box
    annual_masked_data, daily_stats_list = get_heat_lows(box, years, months, LLAT_95th)

    # save the masked data to a netcdf
    annual_masked_data.to_netcdf(f'{output_dir}{box}_HLs.nc')

    # get statistics for the heat lows
    daily_stats_all = pd.concat(daily_stats_list)

    # save the daily stats to a csv
    daily_stats_all.to_csv(f'{output_dir}{box}_HL_stats.csv')

    # if file is created, remove the temporary annual files
    if os.path.exists(f'{output_dir}{box}_HLs.nc'):
        for year in years:
            os.remove(f'{output_dir}{box}_LLAT_HLs_{year}.nc')