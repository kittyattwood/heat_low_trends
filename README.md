# heat_low_trends


Code used in the analysis and production of figures. The workflow is documented below.
Contact: kitty.attwood@ouce.ox.ac.uk

1. Detect heat lows using daily ERA5 data: heat_low_detection.py

    Global ERA5 data requirements:

    - Daily LLAT data per region located at: {data_dir}{box}_LLAT_1980_2024.nc
        LLAT is calculated as the difference in ERA5 geopotential height at the two specified pressure levels. Hourly data is resampled to daily.
        Available from the CDS datastore https://cds.climate.copernicus.eu/datasets/reanalysis-era5-pressure-levels

    - Global daily vertical velocity data (300 hPa) in yearly files at: {daily_w_dir}era5_daily_vertical_velocity_{year}.nc
        Hourly vertical velocity is resampled to daily. Files are aggregated by year. Available from the CDS datastore: https://cds.climate.copernicus.eu/datasets/reanalysis-era5-pressure-levels
        
    - Global land-sea mask file located at: {data_dir}era5_land_sea_mask.nc
        Invariant, downloadable from the CDS datastore: https://cds.climate.copernicus.eu/datasets/reanalysis-era5-single-levels
    
    - ERA5 climatological surface pressure at: {data_dir}era5_avg_sp_1980-2024.nc
        Calculated from daily surface pressure. Available at: https://cds.climate.copernicus.eu/datasets/reanalysis-era5-single-levels


3. Generate statistics on trends and save to csv files: calculate_statistics.py

4. Plot seasonal size and frequency trends using csv files created in (2): plot_seasonal_trends.py

5. Plot spatial trends in heat low frequency using netCDF files created in (1): plot_spatial_trends.py
