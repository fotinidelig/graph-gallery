import marimo

__generated_with = "0.16.5"
app = marimo.App(width="medium")


@app.cell
def _():
    import matplotlib.pyplot as plt
    import matplotlib as mpl
    from matplotlib.colors import LinearSegmentedColormap
    from matplotlib.ticker import FormatStrFormatter
    from matplotlib.animation import FuncAnimation
    import pandas as pd
    from pypalettes import load_cmap, create_cmap
    from highlight_text import fig_text, ax_text
    from drawarrow import ax_arrow, fig_arrow
    import geopandas as gpd
    from shapely.geometry import Point
    import xarray as xr
    from pyfonts import load_google_font, set_default_font
    import numpy as np
    from pathlib import Path
    import marimo as mo
    return (
        LinearSegmentedColormap,
        Path,
        Point,
        ax_arrow,
        ax_text,
        fig_text,
        gpd,
        load_google_font,
        np,
        pd,
        plt,
        set_default_font,
        xr,
    )


@app.cell
def _(Path, Point, gpd, np):
    # Get data from files into dictionaries
    CURRENT_DIR = Path(__file__).resolve().parent
    print(f"Current Dir: {CURRENT_DIR}")

    def load_greece_mask(da):
        """Load Greece boundaries from GeoJSON and create a mask for the data array."""
        world = gpd.read_file(CURRENT_DIR.parent / 'data' / 'world.geo.json')
        greece = world[world["name"] == "Greece"]

        if len(greece) == 0:
            raise ValueError("Greece not found in GeoJSON file!")

        greece_geom = greece.geometry.iloc[0]
        print(f"Greece bounds: {greece.total_bounds}")

        # Get lat/lon coordinates
        lons, lats = np.meshgrid(da.longitude.values, da.latitude.values)

        # Create mask: True if point is within Greece
        mask = np.zeros(lats.shape, dtype=bool)
        for i in range(lats.shape[0]):
            for j in range(lats.shape[1]):
                point = Point(lons[i, j], lats[i, j])
                mask[i, j] = greece_geom.contains(point) or point.within(greece_geom)

        return mask
    return CURRENT_DIR, load_greece_mask


@app.cell
def _(load_greece_mask, np, xr):
    def read_monthly_series_greece(
        filename: str,
        var_name: str,
        title_label: str,
    ) -> xr.DataArray:
        """Load one E-OBS file, compute monthly spatial means for Greece, and plot."""
        print(f"\n=== Processing {filename} ({var_name}) ===")
        ds = xr.open_dataset(filename, engine="netcdf4")

        print("Dataset info:")
        print(f"  Dimensions: {dict(ds.sizes)}")
        print(f"  Data variables: {list(ds.data_vars)}")
        print(f"  Units: {ds[var_name].attrs.get('units', 'NA')}")
        print("\n" + "=" * 50 + "\n")

        da = ds[var_name]

        # --- Select region using Greece boundaries from GeoJSON ---
        print("Selecting region (Greece using GeoJSON boundaries)...")

        # First, get a rough bounding box to reduce data size
        # Greece bounds: [19.63, 34.82, 28.24, 41.75] (minx, miny, maxx, maxy)
        region_rough = da.sel(
            latitude=slice(34.5, 42.0),
            longitude=slice(19.0, 28.5),
        )

        # Create mask for Greece
        greece_mask = load_greece_mask(region_rough)

        # Apply mask: set values outside Greece to NaN
        # Create a DataArray mask with same dimensions as region_rough
        mask_da = xr.DataArray(
            greece_mask,
            dims=["latitude", "longitude"],
            coords={"latitude": region_rough.latitude, "longitude": region_rough.longitude},
        )

        # Apply mask to the data
        region = region_rough.where(mask_da)

        print(f"Region shape: {region.shape}")
        print(f"  Time steps: {region.sizes['time']}")
        print(f"  Latitude points: {region.sizes['latitude']}")
        print(f"  Longitude points: {region.sizes['longitude']}")
        print(f"  Valid points (within Greece): {mask_da.sum().values}")
        print("\n" + "=" * 50 + "\n")

        # Process year by year to avoid memory issues
        print("Processing data year by year to manage memory...")
        years = np.unique(region.time.dt.year.values)
        monthly_series = []
        yearly_extremes = []  # Store yearly max/min values

        for year in years:
            year_data = region.sel(time=region.time.dt.year == year)

            # Data is already in Celsius
            spatial_avg = year_data.mean(dim=["latitude", "longitude"])

            # Resample to monthly means, keeping the time dimension
            monthly_year = spatial_avg.resample(time="1MS").mean()

            if var_name == "tx":
                yearly_max = np.nanmax(monthly_year.values)
                    # Find the month when the maximum occurs
                max_idx = np.nanargmax(monthly_year.values)
                extreme_time = monthly_year.time.values[max_idx]
                yearly_extremes.append({
                    "time": extreme_time,
                    "value": yearly_max,
                })
            elif var_name == "tn":  # Minimum temperature: track yearly minimum
                yearly_min = np.nanmin(monthly_year.values)
                # Find the month when the minimum occurs
                min_idx = np.nanargmin(monthly_year.values)
                extreme_time = monthly_year.time.values[min_idx]
                yearly_extremes.append({
                    "time": extreme_time,
                    "value": yearly_min,
                })

            # Store monthly values with their time coordinates
            for i in range(len(monthly_year.time)):
                monthly_series.append(
                    {
                        "time": monthly_year.time.values[i],
                        "temp": monthly_year.values[i],
                    }
                )

        # Create a DataArray from the monthly series
        monthly_times = [item["time"] for item in monthly_series]
        monthly_temps = [item["temp"] for item in monthly_series]

        monthly_data = xr.DataArray(
            data=monthly_temps,
            dims=["time"],
            coords={"time": monthly_times},
        )

        if len(yearly_extremes) > 0:
            yearly_times = [item["time"] for item in yearly_extremes]
            yearly_values = [item["value"] for item in yearly_extremes]
            yearly_extremes = xr.DataArray(
                data=yearly_values,
                dims=["time"],
                coords={"time": yearly_times},
            )
        else:
            yearly_extremes = None

        print(f"\nMonthly time series shape: {monthly_data.shape}")
        print(
            f"Time range: {monthly_data.time.min().values} "
            f"to {monthly_data.time.max().values}"
        )
        print("\n" + "=" * 50 + "\n")

        # Print some statistics
        print("\nTime series statistics:")
        print(f"  Total months: {len(monthly_data)}")
        print(f"  Min temperature: {np.nanmin(monthly_data.values):.2f} °C")
        print(f"  Max temperature: {np.nanmax(monthly_data.values):.2f} °C")
        print(f"  Mean temperature: {np.nanmean(monthly_data.values):.2f} °C")

        return monthly_data, yearly_extremes
    return (read_monthly_series_greece,)


@app.cell
def _(CURRENT_DIR, read_monthly_series_greece):
    mean_data, _ = read_monthly_series_greece(filename=CURRENT_DIR / "temperature-data/tg_ens_mean_0.25deg_reg_2011-2024_v31.0e.nc",
                                              var_name="tg", title_label="Mean (tg)")
    minim_data, minim_extremes = read_monthly_series_greece(
        filename=CURRENT_DIR / "temperature-data/tn_ens_mean_0.25deg_reg_2011-2024_v31.0e.nc", 
                                              var_name="tn", title_label="Minimum (tn)")
    maxim_data, maxim_extremes = read_monthly_series_greece(
        filename=CURRENT_DIR / "temperature-data/tx_ens_mean_0.25deg_reg_2011-2024_v31.0e.nc", 
                                              var_name="tx", title_label="Maximum (tx)")
    return maxim_data, minim_data


@app.cell
def _(load_greece_mask, np, xr):
    def read_monthly_series_range_greece(
        filename: str,
        var_name: str,
        title_label: str,
    ) -> xr.DataArray:
        """Load one E-OBS file, compute monthly spatial means for Greece, and plot."""
        ds = xr.open_dataset(filename, engine="netcdf4")

        print("Dataset info:")
        print(f"  Dimensions: {dict(ds.sizes)}")
        print(f"  Data variables: {list(ds.data_vars)}")
        print(f"  Units: {ds[var_name].attrs.get('units', 'NA')}")
        print("\n" + "=" * 50 + "\n")

        da = ds[var_name]

        # --- Select region using Greece boundaries from GeoJSON ---

        # First, get a rough bounding box to reduce data size
        # Greece bounds: [19.63, 34.82, 28.24, 41.75] (minx, miny, maxx, maxy)
        region_rough = da.sel(
            latitude=slice(34.5, 42.0),
            longitude=slice(19.0, 28.5),
        )

        # Create mask for Greece
        greece_mask = load_greece_mask(region_rough)

        # Apply mask: set values outside Greece to NaN
        # Create a DataArray mask with same dimensions as region_rough
        mask_da = xr.DataArray(
            greece_mask,
            dims=["latitude", "longitude"],
            coords={"latitude": region_rough.latitude, "longitude": region_rough.longitude},
        )

        # Apply mask to the data
        region = region_rough.where(mask_da)

        # Process year by year to avoid memory issues
        years = np.unique(region.time.dt.year.values)
        monthly_ranges = []

        for year in years:
            year_data = region.sel(time=region.time.dt.year == year)

            # Data is already in Celsius
            spatial_avg = year_data.mean(dim=["latitude", "longitude"])

            # Group by month and compute max/min of daily values within each month
            for month in range(1, 13):
                month_data = spatial_avg.sel(time=spatial_avg.time.dt.month == month)

                if len(month_data) > 0:
                    # Compute max and min of daily values for this month
                    month_max = np.nanmax(month_data.values)
                    month_min = np.nanmin(month_data.values)

                    # Use first day of month for timestamp
                    month_time = month_data.time.values[0]

                    monthly_ranges.append({
                        "time": month_time,
                        "max": month_max,
                        "min": month_min,
                    })

        # Create DataArrays for plotting
        monthly_times = [item["time"] for item in monthly_ranges]
        monthly_max = [item["max"] for item in monthly_ranges]
        monthly_min = [item["min"] for item in monthly_ranges]

        upper_bound = xr.DataArray(
            data=monthly_max,
            dims=["time"],
            coords={"time": monthly_times},
        )

        lower_bound = xr.DataArray(
            data=monthly_min,
            dims=["time"],
            coords={"time": monthly_times},
        )

        print(f"\nMonthly range time series shape: {upper_bound.shape}")
        print(
            f"Time range: {upper_bound.time.min().values} "
            f"to {upper_bound.time.max().values}"
        )
        print("\n" + "=" * 50 + "\n")

        return lower_bound, upper_bound
    return (read_monthly_series_range_greece,)


@app.cell
def _(CURRENT_DIR, read_monthly_series_range_greece):
    minim_lower_b, minim_upper_b = read_monthly_series_range_greece(
        filename=CURRENT_DIR / "temperature-data/tn_ens_mean_0.25deg_reg_2011-2024_v31.0e.nc", 
                                              var_name="tn", title_label="Minimum (tn)")
    maxim_lower_b, maxim_upper_b = read_monthly_series_range_greece(
        filename=CURRENT_DIR / "temperature-data/tx_ens_mean_0.25deg_reg_2011-2024_v31.0e.nc", 
                                              var_name="tx", title_label="Maximum (tx)")
    return maxim_lower_b, maxim_upper_b, minim_lower_b, minim_upper_b


@app.cell
def _(load_greece_mask, np, pd, xr):
    def find_extreme_temperature(filename: str, var_name: str, find_maximum: bool = True):
        """
        Find the extreme temperature (max or min) and its location (coordinates and time).

        Parameters:
        -----------
        filename : str
            Path to the NetCDF file
        var_name : str
            Variable name ('tx' or 'tn')
        find_maximum : bool
            If True, find maximum temperature; if False, find minimum temperature
        """
        ds = xr.open_dataset(filename, engine="netcdf4")

        da = ds[var_name]

        # First, get a rough bounding box to reduce data size
        region_rough = da.sel(
            latitude=slice(34.5, 42.0),
            longitude=slice(19.0, 28.5),
        )

        # Create mask for Greece
        greece_mask = load_greece_mask(region_rough)

        # Apply mask: set values outside Greece to NaN
        mask_da = xr.DataArray(
            greece_mask,
            dims=["latitude", "longitude"],
            coords={"latitude": region_rough.latitude, "longitude": region_rough.longitude},
        )

        # Apply mask to the data
        region = region_rough.where(mask_da)

        # Find the extreme value
        if find_maximum:
            extreme_value = region.max().values
            extreme_idx = np.unravel_index(np.nanargmax(region.values), region.shape)
        else:
            extreme_value = region.min().values
            extreme_idx = np.unravel_index(np.nanargmin(region.values), region.shape)

        # Get coordinates and time
        time_idx, lat_idx, lon_idx = extreme_idx
        extreme_time = region.time.values[time_idx]
        extreme_lat = region.latitude.values[lat_idx]
        extreme_lon = region.longitude.values[lon_idx]

        # Print results
        print(f"{'Maximum' if find_maximum else 'Minimum'} Temperature Found:")
        print(f"  Temperature: {extreme_value:.2f} °C")
        print(f"  Date: {pd.Timestamp(extreme_time).strftime('%Y-%m-%d')}")
        print(f"  Latitude: {extreme_lat:.4f}°")
        print(f"  Longitude: {extreme_lon:.4f}°")
        print("\n" + "=" * 50 + "\n")

        return {
            'temperature': extreme_value,
            'time': extreme_time,
            'latitude': extreme_lat,
            'longitude': extreme_lon,
        }
    return (find_extreme_temperature,)


@app.cell
def _(CURRENT_DIR, find_extreme_temperature):
    find_extreme_temperature(filename=CURRENT_DIR / "temperature-data/tn_ens_mean_0.25deg_reg_2011-2024_v31.0e.nc", 
                                  var_name="tn", find_maximum=False)
    find_extreme_temperature(filename=CURRENT_DIR / "temperature-data/tx_ens_mean_0.25deg_reg_2011-2024_v31.0e.nc", 
                                  var_name="tx", find_maximum=True)
    return


@app.cell
def _(load_greece_mask, np, pd, xr):
    def compute_monthly_extremes(filename: str, var_name: str):
        """
        Compute the extreme temperature (max for tx, min for tn) per month per year.

        Parameters:
        -----------
        filename : str
            Path to the NetCDF file
        var_name : str
            Variable name ('tx' for maximum or 'tn' for minimum)

        Returns:
        --------
        pandas.DataFrame with columns: year, month, extreme_temperature, date, latitude, longitude
        """
        ds = xr.open_dataset(filename, engine="netcdf4")

        da = ds[var_name]

        # First, get a rough bounding box to reduce data size
        region_rough = da.sel(
            latitude=slice(34.5, 42.0),
            longitude=slice(19.0, 28.5),
        )

        # Create mask for Greece
        greece_mask = load_greece_mask(region_rough)

        # Apply mask: set values outside Greece to NaN
        mask_da = xr.DataArray(
            greece_mask,
            dims=["latitude", "longitude"],
            coords={"latitude": region_rough.latitude, "longitude": region_rough.longitude},
        )

        # Apply mask to the data
        region = region_rough.where(mask_da)

        years = np.unique(region.time.dt.year.values)
        results = []

        is_maximum = (var_name == "tx")  # True for tx (find max), False for tn (find min)

        for year in years:
            year_data = region.sel(time=region.time.dt.year == year)

            # Process each month
            for month in range(1, 13):
                month_data = year_data.sel(time=year_data.time.dt.month == month)

                if len(month_data) == 0:
                    continue

                # Find extreme value for this month
                if is_maximum:
                    extreme_value = month_data.max().values
                    # Find where the maximum occurs
                    extreme_idx = np.unravel_index(
                        np.nanargmax(month_data.values), 
                        month_data.shape
                    )
                else:
                    extreme_value = month_data.min().values
                    # Find where the minimum occurs
                    extreme_idx = np.unravel_index(
                        np.nanargmin(month_data.values), 
                        month_data.shape
                    )

                # Get coordinates and time
                time_idx, lat_idx, lon_idx = extreme_idx
                extreme_time = month_data.time.values[time_idx]
                extreme_lat = month_data.latitude.values[lat_idx]
                extreme_lon = month_data.longitude.values[lon_idx]

                results.append({
                    'year': year,
                    'month': month,
                    'extreme_temperature': float(extreme_value),
                    'date': pd.Timestamp(extreme_time),
                    'latitude': float(extreme_lat),
                    'longitude': float(extreme_lon),
                })

        # Create DataFrame
        df = pd.DataFrame(results)

        print(f"\nComputed extremes for {len(df)} month-year combinations")

        return df
    return (compute_monthly_extremes,)


@app.cell
def _(CURRENT_DIR, compute_monthly_extremes):
    maximum_extremes = compute_monthly_extremes(filename=CURRENT_DIR / "temperature-data/tx_ens_mean_0.25deg_reg_2011-2024_v31.0e.nc", 
                                  var_name="tx")
    minimum_extremes = compute_monthly_extremes(filename=CURRENT_DIR / "temperature-data/tn_ens_mean_0.25deg_reg_2011-2024_v31.0e.nc", 
                                  var_name="tn")
    maximum_extremes = maximum_extremes.sort_values('date')
    minimum_extremes = minimum_extremes.sort_values('date')

    minimum_extremes.columns
    return maximum_extremes, minimum_extremes


@app.cell
def _(
    CURRENT_DIR,
    LinearSegmentedColormap,
    ax_arrow,
    ax_text,
    fig_text,
    load_google_font,
    maxim_data,
    maxim_lower_b,
    maxim_upper_b,
    maximum_extremes,
    minim_data,
    minim_lower_b,
    minim_upper_b,
    minimum_extremes,
    np,
    pd,
    plt,
    set_default_font,
):
    def plot_ranges():
        fig, ax = plt.subplots(figsize=(15,6), dpi=150, layout="tight")
        font = load_google_font("Space Mono", weight="regular", italic=False)
        bold_font = load_google_font("Space Mono", weight="bold", italic=False)
        italic_font = load_google_font("Space Mono", weight="regular", italic=True)
        set_default_font(font)

        # Colors
        background = '#f0f0f0' #'#d9e9ff'
        maxim_color = '#98231B'
        minim_color = '#008080'#'#007041'
        too_hot_color = '#cc6633'#'#cc6633'
        too_cold_color = '#29a6a6'#'#33cccc'
        tickcolor = '#333333'
    
        ax.set_facecolor(background)
        fig.set_facecolor(background)

        xlim = (pd.Timestamp('2010-06-01'), pd.Timestamp('2025-06-01')) # go beyond data range on the left to place ticks
        ylim = (-60, 80)

        ax.hlines(y=0, xmin=min(minim_data.time.values), xmax=max(minim_data.time.values), alpha=.2, 
                  color=too_cold_color, zorder=-2, ls='dashed')
        ax.hlines(y=30, xmin=min(minim_data.time.values), xmax=max(minim_data.time.values), alpha=.2, 
                  color=too_hot_color, zorder=-2, ls='dashed')

        # Loop through years and add alternating background
        years = range(2011, 2025)
        light_grey = background
        dark_grey = '#dedede'
        for i, year in enumerate(years):
            # January to June (first half)
            ax.axvspan(
                pd.Timestamp(f'{year}-01-01'),
                pd.Timestamp(f'{year}-12-31') if year < 2024 else pd.Timestamp('2024-12-30'),
                ymin=.3, ymax=.75,
                color=light_grey if i % 2 == 0 else dark_grey,
                alpha=0.3,
                linewidth=0,
                zorder=-1  # Place behind the plot
            )

        # Area above 30 degrees 
        colors = [too_hot_color, background]
        n_bins = 100
        cmap = LinearSegmentedColormap.from_list('gradient', colors, N=n_bins)

        # Create a small image for the gradient
        gradient = np.linspace(0, 1, n_bins).reshape(n_bins, 1)
        ax.imshow(
            gradient,
            aspect='auto',
            cmap=cmap,
            extent=[xlim[0], xlim[1], 30, ylim[1]],
            alpha=0.3,
            zorder=0
        )

        # Area bellow 0 degrees
        colors = [background, too_cold_color] # bottom color to top color
        cmap = LinearSegmentedColormap.from_list('gradient', colors, N=n_bins)

        # Create a small image for the gradient
        gradient = np.linspace(0, 1, n_bins).reshape(n_bins, 1)

        ax.imshow(
            gradient,
            aspect='auto',
            cmap=cmap,
            extent=[xlim[0], xlim[1], ylim[0], 0],
            alpha=0.3,
            zorder=0
        )

        # range for minimum temperatures 
        ax.fill_between(
            minim_upper_b.time.values,
            minim_lower_b.values,
            minim_upper_b.values,
            zorder=0,
            alpha=0.1,
            linewidth=0,
            color=minim_color,
        )
        # range for maximum temperatures
        ax.fill_between(
            maxim_upper_b.time.values,
            maxim_lower_b.values,
            maxim_upper_b.values,
            zorder=2,
            alpha=0.2,
            linewidth=0,
            color=maxim_color,
        )

        ax.plot(minim_data.time.values, minim_data.values, color=minim_color, 
                alpha=.6, linewidth=2, zorder=1)

        ax.plot(maxim_data.time.values, maxim_data.values, color=maxim_color, 
                alpha=.6, linewidth=2, zorder=3)

        # Plot maximum and minimum temperatures across Greece per year
        yearly_max = maximum_extremes.loc[maximum_extremes.groupby('year')['extreme_temperature'].idxmax()]
        yearly_min = minimum_extremes.loc[minimum_extremes.groupby('year')['extreme_temperature'].idxmin()]
        extremes_date = [pd.Timestamp(f'{y}-07-01') for y in years]
        ax.scatter(extremes_date, yearly_max['extreme_temperature'], 
             marker='^', label='Monthly Maximum (tx)', color=too_hot_color, alpha=0.8)
        ax.scatter(extremes_date, yearly_min['extreme_temperature'], 
             marker='v', label='Monthly Minimum (tn)', color=too_cold_color, alpha=0.8)

        ax.set_xlim(xlim[0], xlim[1])
        ax.set_ylim(ylim[0], ylim[1])

        years = range(2011, 2025) # data ranges from 2011 to 2024
        xtick_dates = [pd.Timestamp(f'{y}-07-01') for y in years[::2]] #for m in [1, 7]]
        labels = years[::2]
        ax.set_xticks(xtick_dates)
        # labels = [f'{y}' if m == 1 else 'Jul' for y in years for m in [1, 7]]
        ax.set_xticklabels(labels, rotation=0, ha='center')
        yticks = [-10, 0, 10, 20, 30, 40]
        ax.set_yticks(yticks)
        ax.set_yticklabels([f'{t}°C' for t in yticks])
        ax.spines[['top', 'left', 'right', 'bottom']].set_visible(False)
        ax.tick_params(length=0, labelsize=7, labelcolor=tickcolor)
        ax.tick_params(axis='y', pad=-22)
        ax.tick_params(axis='x', pad=-95)

        # Annotations for max and min temperature
        max_data = maximum_extremes.loc[maximum_extremes['extreme_temperature'].idxmax()]
        max_year, max_temp = max_data['year'], max_data['extreme_temperature']
        max_data_txt = f'Maximum temperature reached <{max_temp:.1f}°C>\njust outside Athens'
        ax_arrow(head_position=(pd.Timestamp(f'{max_year}-07-01'), max_temp+2),
                tail_position=(pd.Timestamp(f'{max_year-1}-09-01'), max_temp+15),
                radius=-.5, color=too_hot_color, head_width=4)
        ax_text(x=pd.Timestamp(f'{max_year-1}-09-01'), y=max_temp+18,
              s=max_data_txt, ha='right', textalign='center',
                color=too_hot_color, highlight_textprops=[{'font': bold_font}])

        min_data = minimum_extremes.loc[minimum_extremes['extreme_temperature'].idxmin()]
        min_year, min_temp = min_data['year'], min_data['extreme_temperature']
        min_data_txt = f'Minimum temperature reached <{min_temp:.1f}°C> \nnear the Greek-Albanian boarder'
        ax_arrow(head_position=(pd.Timestamp(f'{min_year}-07-01'), min_temp-2),
                tail_position=(pd.Timestamp(f'{min_year+1}-03-01'), min_temp-17),
                radius=-.5, color=too_cold_color, head_width=4)
        ax_text(x=pd.Timestamp(f'{min_year+1}-03-01'), y=min_temp-22,
              s=min_data_txt, ha='left', textalign='center', va='bottom',
                color=too_cold_color, highlight_textprops=[{'font': bold_font}])

        highest_min_data = yearly_min.loc[yearly_min['extreme_temperature'].idxmax()]
        highest_min_year, highest_min_temp = highest_min_data['year'], highest_min_data['extreme_temperature'] 
        highest_min_txt = \
        f'2024\'s lowest temperature was <{highest_min_temp:.1f}°C>, \ncontinuing a trend of Greece <"warming up">\nsince 2019'
        ax_arrow(head_position=(pd.Timestamp(f'{highest_min_year}-07-01'), highest_min_temp-2),
                tail_position=(pd.Timestamp(f'{highest_min_year}-01-01'), highest_min_temp-20),
                radius=.2, color=too_cold_color, head_width=4)
        ax_text(x=pd.Timestamp('2022-12-01'), y=highest_min_temp-22,
              s=highest_min_txt, ha='center', textalign='center', va='top',
                color=too_cold_color, highlight_textprops=[{'font': bold_font}, {'font': bold_font}])

        # Legend
        ax.scatter([pd.Timestamp('2011-03-01')], [52], 
             marker='^', s=20, color=tickcolor, alpha=0.8)
        ax_text(x=pd.Timestamp('2011-04-01'), y=52, size=7, va='center', color=tickcolor,
               s='Temperature extremes per year (<maximum> and <minimum>)',
               highlight_textprops=[{'color': too_hot_color}, {'color': too_cold_color}])
        ax.plot([pd.Timestamp('2011-02-15'), pd.Timestamp('2011-04-01')], [48, 48], 
             linewidth=1.5, color=tickcolor, alpha=0.8)
        ax_text(x=pd.Timestamp('2011-05-01'), y=48, size=7, va='center', color=tickcolor,
               s='Average daily temperature (<highest> and <lowest> daily)',
               highlight_textprops=[{'color': maxim_color}, {'color': minim_color}])
    
        fig_text(fig=fig, x=.5, y=.94, s='Is Greece already experiencing global warming?', 
                 ha='center', size=17)

        ax_text(x=pd.Timestamp('2025-05-01'), y=-52, 
                s='Data Source: https://cds.climate.copernicus.eu/datasets/\nDataset: E-OBS', size=7, 
                textalign='right', ha='right')

        plt.savefig(CURRENT_DIR / 'greece_temperatures.svg')
        plt.show()

    plot_ranges()
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
