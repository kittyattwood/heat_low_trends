import xarray as xr
import cartopy.crs as ccrs
import matplotlib.pyplot as plt
import colormaps as cmaps
import cartopy.feature as cfeature
import numpy as np
from pymannkendall import original_test
import os
from cartopy.mpl.gridliner import LONGITUDE_FORMATTER, LATITUDE_FORMATTER


boxes = ['box_A','box_B','box_C','box_D','box_E']
regions = ['North America', 'Sahara', 'West Asia', 'Southern Africa', 'Australia']
llat_levels = [[800,650],[900,750],[800,650],[850,700],[900,750]]
min_lon = [-125,-20,30,10,112]
max_lon = [-85,30,75,40,147]
max_lat = [50,40,40,0,-10]
min_lat = [10,5,5,-35,-35]

# read in the data
data_dir = ''

if not os.path.exists(data_dir+'/plots'):
    os.makedirs(data_dir+'/plots')

# get global average surface pressure file
sp = xr.open_dataset(f'{data_dir}/data/era5_avg_sp_1980-2019.nc')
# make sure longitude is -180 to 180
sp = sp.assign_coords(longitude=(((sp.longitude + 180) % 360) - 180)).sortby('longitude')

for box in boxes:

    box_index = boxes.index(box)

    # get average surface pressure
    sp = sp.assign_coords(longitude=(((sp.longitude + 180) % 360) - 180)).sortby('longitude')
    sp_box = sp.sel(latitude=slice(max_lat[box_index], min_lat[box_index]), longitude=slice(min_lon[box_index], max_lon[box_index]))
    sp_box = sp_box/100

    # save 2D mask
    elevation_mask = sp_box.sp.squeeze() > llat_levels[box_index][0]
    elevation_mask.to_netcdf(f'{data_dir}/data/{box}_elevation_mask.nc')
    
    print('Processing box:', box)
    print('Region:', regions[boxes.index(box)])
    
    # Read in the data
    data_path = f"{data_dir}/results/{box}_HLs.nc"
    ds = xr.open_dataset(data_path)

    heat_low_days = ds["z"].count(dim="time")
    max_hl_days = heat_low_days.max()
    print('max hl days:', max_hl_days)
    
    heat_low_days_all = heat_low_days.values.flatten()
    top_10 = np.percentile(heat_low_days_all, 90)
    print('threshold:', top_10)
    avg_HL_position_mask = heat_low_days.where(heat_low_days > top_10, other=np.nan)
    one_per_year_mask = heat_low_days.where(heat_low_days > 45, other=np.nan)
    
    # resample the data to annual frequency and count valid days per year
    valid_days_yearly = ds["z"].resample(time='1YE').count()
    valid_days_yearly['time'] = valid_days_yearly['time'].dt.year
    valid_days_yearly = valid_days_yearly.transpose('latitude', 'longitude', 'time')

    lat = ds['latitude'].values
    lon = ds['longitude'].values
    trends = np.zeros((len(lat), len(lon)))
    p_values = np.zeros((len(lat), len(lon)))

    # perform Mann-Kendall test for each grid point
    for i in range(len(lon)):
        for j in range(len(lat)):
            y = valid_days_yearly[j, i, :].values
            if np.sum(~np.isnan(y)) > 1:  # ensure enough valid data points
                mk_result = original_test(y[~np.isnan(y)])
                trends[j, i] = mk_result.slope  # trend magnitude
                p_values[j, i] = mk_result.p  # p-value
            else:
                trends[j, i] = np.nan
                p_values[j, i] = np.nan

    significance_mask = (p_values < 0.05) & (np.abs(trends) > 0.1)
    
    fig, ax = plt.subplots(1, 1, figsize=(12, 8), subplot_kw={'projection': ccrs.PlateCarree()})
    c = ax.pcolormesh(ds["longitude"], ds["latitude"], trends*10, cmap=cmaps.temp_diff_18lev, shading="auto", vmin=-10, vmax=10)
    
    for i in range(len(ds["longitude"])):
        for j in range(len(ds["latitude"])):
            if significance_mask[j, i]:
                ax.scatter(ds["longitude"][i], ds["latitude"][j], color='black', s=0.2, alpha=0.7, transform=ccrs.PlateCarree())
    
    nan_mask = heat_low_days.where(heat_low_days, other=np.nan)

    # mask NaNs to prevent plotting issues
    masked_data = np.ma.masked_invalid(nan_mask)

    # add contour lines to indicate the top 10% of heat low days
    ax.contour(ds["longitude"], ds["latitude"], masked_data, levels=[top_10, max_hl_days], colors='k', linewidths=1.5, transform=ccrs.PlateCarree())

    # add shading where elevation mask is False
    elevation_mask = elevation_mask.where(elevation_mask == False, other=np.nan)
    ax.pcolormesh(sp_box["longitude"], sp_box["latitude"], elevation_mask, cmap=cmaps.greys_r, alpha=0.3)

    # add a colorbar
    cb = plt.colorbar(c, ax=ax, orientation="vertical", label="Change in Heat Low Days per Decade")
    
    # add geographical features for better context
    ax.coastlines()
    ax.add_feature(cfeature.BORDERS, linestyle=':')
    ax.add_feature(cfeature.LAND, facecolor='lightgray', alpha=0.5)

    # set axis labels
    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")

    # add lat lon ticks but no visible grid, only labels on left and bottom axes
    gl = ax.gridlines(crs=ccrs.PlateCarree(), draw_labels=True, linestyle='--', alpha=0.5)
    gl.top_labels = False
    gl.right_labels = False
    gl.xlines = False
    gl.ylines = False
    gl.xformatter = LONGITUDE_FORMATTER
    gl.yformatter = LATITUDE_FORMATTER
    if ax == 0:
        gl.xlabel_style = {'size': 13}
    else:
        gl.xlabel_style = {'size': 12}
        gl.ylabel_style = {'size': 12}
    
    # save the plot
    plt.savefig(f'{data_dir}/plots/{box}_locational_trend.png', dpi=300, bbox_inches='tight')


