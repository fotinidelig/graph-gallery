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
        FuncAnimation,
        LinearSegmentedColormap,
        Path,
        Point,
        ax_arrow,
        ax_text,
        fig_text,
        gpd,
        load_google_font,
        mo,
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
    return maximum_extremes, minimum_extremes


@app.cell
def _(mo):
    mo.md(r"""### Line plot with annotations""")
    return


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
def _(mo):
    mo.md(r"""### What if we go polar?""")
    return


@app.cell
def _(
    CURRENT_DIR,
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
    def plot_polar():
        # --- Step 1: Create polar figure ---
        fig, ax = plt.subplots(
            figsize=(10, 10), dpi=150, layout="tight",
            subplot_kw={'projection': 'polar'},
        )
        font = load_google_font("Space Mono", weight="regular", italic=False)
        bold_font = load_google_font("Space Mono", weight="bold", italic=False)
        set_default_font(font)

        # Colors (same as Cartesian version)
        background = '#f0f0f0'
        maxim_color = '#98231B'
        minim_color = '#008080'
        too_hot_color = '#cc6633'
        too_cold_color = '#29a6a6'
        tickcolor = '#333333'

        ax.set_facecolor(background)
        fig.set_facecolor(background)

        # --- Step 2: Helper — convert datetime to x and to angle ---
        datetimes = list(minim_data.time.values)
        datetimes.append(pd.Timestamp('2025-01-01')) # two additional datetimes to create space between 2011 and 2025
        datetimes.append(pd.Timestamp('2025-02-01'))

        global_x = np.arange(0, len(minim_data.time.values)+2)
        global_x_min = min(global_x)
        global_x_max = max(global_x)

        datetime_to_x = dict(zip(minim_data.time.values, global_x))

        def x_to_theta(x):
            """Map datetime values linearly to [0, 2π]."""
            return 2 * np.pi * (x - global_x_min) / (global_x_max - global_x_min)

        # --- Step 3: Radius offset (so negative temps become positive radii) ---
        r_offset = 15  # 0°C sits at radius 15; tune to taste


        # --- Step 4: Plot the temperature lines ---
        x_min = np.arange(0, len(minim_data.time.values))
        x_max = np.arange(0, len(maxim_data.time.values))

        theta_min_line = x_to_theta(x_min)
        theta_max_line = x_to_theta(x_max)


        ax.plot(theta_min_line, minim_data.values + r_offset,
                color=minim_color, alpha=0.6, linewidth=1.5, zorder=1)

        ax.plot(theta_max_line, maxim_data.values + r_offset,
                color=maxim_color, alpha=0.6, linewidth=1.5, zorder=3)

        # --- Step 5: Fill-between bands ---
        x_minim_upper_b = np.arange(0, len(minim_upper_b.time.values))
        theta_band_min = x_to_theta(x_minim_upper_b)
        ax.fill_between(
            theta_band_min,
            minim_lower_b.values + r_offset,
            minim_upper_b.values + r_offset,
            color=minim_color, alpha=0.1, linewidth=0, zorder=0,
        )

        x_maxim_upper_b = np.arange(0, len(maxim_upper_b.time.values))
        theta_band_max = x_to_theta(x_maxim_upper_b)
        ax.fill_between(
            theta_band_max,
            maxim_lower_b.values + r_offset,
            maxim_upper_b.values + r_offset,
            color=maxim_color, alpha=0.2, linewidth=0, zorder=2,
        )

        # --- Step 6: Scatter — yearly extremes ---
        years = range(2011, 2025)
        yearly_max = maximum_extremes.loc[
            maximum_extremes.groupby('year')['extreme_temperature'].idxmax()
        ]

        yearly_min = minimum_extremes.loc[
            minimum_extremes.groupby('year')['extreme_temperature'].idxmin()
        ]

        extremes_date = np.array(
            [pd.Timestamp(f'{y}-07-01') for y in years], dtype='datetime64[ns]'
        )

        x_extremes = [datetime_to_x.get(date) for date in extremes_date]
        theta_extremes = x_to_theta(x_extremes)

        ax.scatter(theta_extremes,
                   yearly_max['extreme_temperature'].values + r_offset,
                   marker='o', color=too_hot_color, alpha=0.8, s=30, zorder=5)
        ax.scatter(theta_extremes,
                   yearly_min['extreme_temperature'].values + r_offset,
                   marker='o', color=too_cold_color, alpha=0.8, s=30, zorder=5)

        # --- Step 7: Reference circles at 0°C and 30°C ---
        full_theta = np.linspace(0, 2 * np.pi, 300)
        ax.plot(full_theta, np.full_like(full_theta, 0 + r_offset),
                color=too_cold_color, alpha=0.25, ls='dashed', lw=0.8, zorder=-1)
        ax.plot(full_theta, np.full_like(full_theta, 30 + r_offset),
                color=too_hot_color, alpha=0.25, ls='dashed', lw=0.8, zorder=-1)

        # --- Step 8: Angular tick labels (years) ---
        year_ticks = [pd.Timestamp(f'{y}-01-01') for y in range(2011, 2025)]
        # year_ticks.append(pd.Timestamp('2025-02-01')) # push 2025 tick slightly before in order to avoid overlap with 2011

        x_ticks = [datetime_to_x.get(date) for date in year_ticks]
        year_thetas = x_to_theta(
            np.array(x_ticks)
        )
        ax.set_xticks(year_thetas)
        ax.set_xticklabels([str(y) for y in range(2011, 2025)],
                           size=8, color=tickcolor)

        # --- Step 9: Radial tick labels (temperatures) ---
        temp_labels = [-10, 0, 10, 20, 30, 40]
        temp_ticks = [-7, 0, 10, 20, 30, 40] # push -10 to the -7 position for better visibility
        ax.set_yticks([t + r_offset for t in temp_ticks])
        ax.set_yticklabels([f'{t}°C' for t in temp_labels],
                           size=7, color=tickcolor)
        ax.set_rlabel_position(0)  # place radial labels on the left

        # Radial limits
        ax.set_ylim(0, 46 + r_offset)

        # Start angle: put Jan 2011 at the top (90° in polar = top)
        # ax.set_theta_offset(np.pi / 2)
        ax.set_theta_direction(-1)  # clockwise

        # Light grid styling
        ax.grid(True, color='grey', alpha=0.15, linewidth=0.5)
        ax.spines['polar'].set_visible(False)

        # --- Title ---
        fig_text(fig=fig, x=0.5, y=0.97,
                 s='Is Greece already experiencing global warming?',
                 ha='center', size=15)
            # --- Legend ---
        fig_text(fig=fig, x=0.5, y=0.02,
                 s='o Temperature extremes per year (<maximum> and <minimum>)',
                 ha='center', size=8, color=tickcolor,
                 highlight_textprops=[{'color': too_hot_color}, {'color': too_cold_color}])
        fig_text(fig=fig, x=0.5, y=0.04,
                 s='— Average daily temperature (<highest> and <lowest> daily)',
                 ha='center', size=8, color=tickcolor,
                 highlight_textprops=[{'color': maxim_color}, {'color': minim_color}])

        plt.savefig(CURRENT_DIR / 'greece_temperatures_polar.png')
        plt.show()

    plot_polar()
    return


@app.cell
def _(mo):
    mo.md(r"""### Animated version of polar line plot""")
    return


@app.cell
def _(
    CURRENT_DIR,
    FuncAnimation,
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
    def animate_expanding_time():
        """Animation 1: Data grows clockwise month by month, like a flower blooming."""
        font = load_google_font("Space Mono", weight="regular", italic=False)
        set_default_font(font)

        # Colors
        background = '#f0f0f0'
        maxim_color = '#98231B'
        minim_color = '#008080'
        too_hot_color = '#cc6633'
        too_cold_color = '#29a6a6'
        tickcolor = '#333333'

        # Precompute theta mapping (same as static version)
        total_months = len(minim_data.time.values)
        global_x = np.arange(0, total_months + 2)
        global_x_min = min(global_x)
        global_x_max = max(global_x)
        datetime_to_x = dict(zip(minim_data.time.values, global_x))

        def x_to_theta(x):
            return 2 * np.pi * (x - global_x_min) / (global_x_max - global_x_min)

        r_offset = 15

        # Precompute all theta arrays
        theta_min_line = x_to_theta(np.arange(len(minim_data.time.values)))
        theta_max_line = x_to_theta(np.arange(len(maxim_data.time.values)))
        theta_band_min = x_to_theta(np.arange(len(minim_upper_b.time.values)))
        theta_band_max = x_to_theta(np.arange(len(maxim_upper_b.time.values)))

        # Yearly extremes
        years = range(2011, 2025)
        yearly_max_df = maximum_extremes.loc[
            maximum_extremes.groupby('year')['extreme_temperature'].idxmax()
        ]
        yearly_min_df = minimum_extremes.loc[
            minimum_extremes.groupby('year')['extreme_temperature'].idxmin()
        ]
        extremes_date = np.array(
            [pd.Timestamp(f'{y}-07-01') for y in years], dtype='datetime64[ns]'
        )
        x_extremes = np.array([datetime_to_x.get(d) for d in extremes_date])
        theta_extremes = x_to_theta(x_extremes)

        # Year tick positions
        year_ticks = [pd.Timestamp(f'{y}-01-01') for y in range(2011, 2025)]
        x_ticks = [datetime_to_x.get(d) for d in year_ticks]
        year_thetas = x_to_theta(np.array(x_ticks))

        # Frame indices: reveal 2 months per frame
        step = 2
        frame_months = list(range(step, total_months + 1, step))
        if frame_months[-1] != total_months:
            frame_months.append(total_months)

        # Create figure
        fig, ax = plt.subplots(
            figsize=(10, 10), dpi=100,
            subplot_kw={'projection': 'polar'},
        )
        fig.set_facecolor(background)
        fig.suptitle(
            'Is Greece already experiencing global warming?',
            fontsize=15, y=0.97,
        )

        def update(frame_idx):
            ax.clear()
            ax.set_facecolor(background)
            n = frame_months[frame_idx]

            # Reference circles
            full_theta = np.linspace(0, 2 * np.pi, 300)
            ax.plot(full_theta, np.full_like(full_theta, 0 + r_offset),
                    color=too_cold_color, alpha=0.25, ls='dashed', lw=0.8, zorder=-1)
            ax.plot(full_theta, np.full_like(full_theta, 30 + r_offset),
                    color=too_hot_color, alpha=0.25, ls='dashed', lw=0.8, zorder=-1)

            # Temperature lines up to month n
            ax.plot(theta_min_line[:n], minim_data.values[:n] + r_offset,
                    color=minim_color, alpha=0.6, linewidth=1.5, zorder=1)
            ax.plot(theta_max_line[:n], maxim_data.values[:n] + r_offset,
                    color=maxim_color, alpha=0.6, linewidth=1.5, zorder=3)

            # Fill bands up to month n
            nb_min = min(n, len(theta_band_min))
            ax.fill_between(
                theta_band_min[:nb_min],
                minim_lower_b.values[:nb_min] + r_offset,
                minim_upper_b.values[:nb_min] + r_offset,
                color=minim_color, alpha=0.1, linewidth=0, zorder=0,
            )
            nb_max = min(n, len(theta_band_max))
            ax.fill_between(
                theta_band_max[:nb_max],
                maxim_lower_b.values[:nb_max] + r_offset,
                maxim_upper_b.values[:nb_max] + r_offset,
                color=maxim_color, alpha=0.2, linewidth=0, zorder=2,
            )

            # Scatter for completed years only
            completed_years = min(n // 12, 14)
            if completed_years > 0:
                ax.scatter(
                    theta_extremes[:completed_years],
                    yearly_max_df['extreme_temperature'].values[:completed_years] + r_offset,
                    marker='o', color=too_hot_color, alpha=0.8, s=30, zorder=5,
                )
                ax.scatter(
                    theta_extremes[:completed_years],
                    yearly_min_df['extreme_temperature'].values[:completed_years] + r_offset,
                    marker='o', color=too_cold_color, alpha=0.8, s=30, zorder=5,
                )

            # Axis formatting
            ax.set_xticks(year_thetas)
            ax.set_xticklabels([str(y) for y in range(2011, 2025)],
                               size=8, color=tickcolor)
            temp_labels = [-10, 0, 10, 20, 30, 40]
            temp_ticks = [-7, 0, 10, 20, 30, 40]
            ax.set_yticks([t + r_offset for t in temp_ticks])
            ax.set_yticklabels([f'{t}°C' for t in temp_labels],
                               size=7, color=tickcolor)
            ax.set_rlabel_position(0)
            ax.set_ylim(0, 46 + r_offset)
            ax.set_theta_direction(-1)
            ax.grid(True, color='grey', alpha=0.15, linewidth=0.5)
            ax.spines['polar'].set_visible(False)
            return []

        anim = FuncAnimation(
            fig, update, frames=len(frame_months), interval=100, blit=False,
        )
        anim.save(
            CURRENT_DIR / 'polar_anim_expanding_time.gif',
            writer='pillow', fps=10, dpi=100,
        )
        plt.close(fig)
        print("Saved: polar_anim_expanding_time.gif")

    animate_expanding_time()
    return


@app.cell
def _(
    CURRENT_DIR,
    FuncAnimation,
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
    def animate_expanding_ylim():
        """Animation 2: Radial limit expands outward, revealing the full chart."""
        font = load_google_font("Space Mono", weight="regular", italic=False)
        set_default_font(font)

        # Colors
        background = '#f0f0f0'
        maxim_color = '#98231B'
        minim_color = '#008080'
        too_hot_color = '#cc6633'
        too_cold_color = '#29a6a6'
        tickcolor = '#333333'

        # Theta mapping
        total_months = len(minim_data.time.values)
        global_x = np.arange(0, total_months + 2)
        global_x_min = min(global_x)
        global_x_max = max(global_x)
        datetime_to_x = dict(zip(minim_data.time.values, global_x))

        def x_to_theta(x):
            return 2 * np.pi * (x - global_x_min) / (global_x_max - global_x_min)

        r_offset = 15

        theta_min_line = x_to_theta(np.arange(len(minim_data.time.values)))
        theta_max_line = x_to_theta(np.arange(len(maxim_data.time.values)))
        theta_band_min = x_to_theta(np.arange(len(minim_upper_b.time.values)))
        theta_band_max = x_to_theta(np.arange(len(maxim_upper_b.time.values)))

        # Yearly extremes
        years = range(2011, 2025)
        yearly_max_df = maximum_extremes.loc[
            maximum_extremes.groupby('year')['extreme_temperature'].idxmax()
        ]
        yearly_min_df = minimum_extremes.loc[
            minimum_extremes.groupby('year')['extreme_temperature'].idxmin()
        ]
        extremes_date = np.array(
            [pd.Timestamp(f'{y}-07-01') for y in years], dtype='datetime64[ns]'
        )
        x_extremes = np.array([datetime_to_x.get(d) for d in extremes_date])
        theta_extremes = x_to_theta(x_extremes)

        # Year tick positions
        year_ticks = [pd.Timestamp(f'{y}-01-01') for y in range(2011, 2025)]
        x_ticks = [datetime_to_x.get(d) for d in year_ticks]
        year_thetas = x_to_theta(np.array(x_ticks))

        # ylim expansion with ease-out curve
        final_ylim = 46 + r_offset  # = 61
        start_ylim = r_offset - 5   # = 10 (shows almost nothing)
        n_frames = 60
        t = np.linspace(0, 1, n_frames)
        eased_t = 1 - (1 - t) ** 2  # quadratic ease-out for smooth deceleration
        ylim_values = start_ylim + (final_ylim - start_ylim) * eased_t

        # Create figure
        fig, ax = plt.subplots(
            figsize=(10, 10), dpi=100,
            subplot_kw={'projection': 'polar'},
        )
        fig.set_facecolor(background)
        fig.suptitle(
            'Is Greece already experiencing global warming?',
            fontsize=15, y=0.97,
        )

        def update(frame_idx):
            ax.clear()
            ax.set_facecolor(background)
            current_ylim = ylim_values[frame_idx]

            # Reference circles
            full_theta = np.linspace(0, 2 * np.pi, 300)
            ax.plot(full_theta, np.full_like(full_theta, 0 + r_offset),
                    color=too_cold_color, alpha=0.25, ls='dashed', lw=0.8, zorder=-1)
            ax.plot(full_theta, np.full_like(full_theta, 30 + r_offset),
                    color=too_hot_color, alpha=0.25, ls='dashed', lw=0.8, zorder=-1)

            # All data plotted (clipped by ylim)
            ax.plot(theta_min_line, minim_data.values + r_offset,
                    color=minim_color, alpha=0.6, linewidth=1.5, zorder=1)
            ax.plot(theta_max_line, maxim_data.values + r_offset,
                    color=maxim_color, alpha=0.6, linewidth=1.5, zorder=3)

            ax.fill_between(
                theta_band_min,
                minim_lower_b.values + r_offset,
                minim_upper_b.values + r_offset,
                color=minim_color, alpha=0.1, linewidth=0, zorder=0,
            )
            ax.fill_between(
                theta_band_max,
                maxim_lower_b.values + r_offset,
                maxim_upper_b.values + r_offset,
                color=maxim_color, alpha=0.2, linewidth=0, zorder=2,
            )

            ax.scatter(
                theta_extremes,
                yearly_max_df['extreme_temperature'].values + r_offset,
                marker='o', color=too_hot_color, alpha=0.8, s=30, zorder=5,
            )
            ax.scatter(
                theta_extremes,
                yearly_min_df['extreme_temperature'].values + r_offset,
                marker='o', color=too_cold_color, alpha=0.8, s=30, zorder=5,
            )

            # Axis formatting
            ax.set_xticks(year_thetas)
            ax.set_xticklabels([str(y) for y in range(2011, 2025)],
                               size=8, color=tickcolor)
            temp_labels = [-10, 0, 10, 20, 30, 40]
            temp_ticks = [-7, 0, 10, 20, 30, 40]
            ax.set_yticks([t + r_offset for t in temp_ticks])
            ax.set_yticklabels([f'{t}°C' for t in temp_labels],
                               size=7, color=tickcolor)
            ax.set_rlabel_position(0)
            ax.set_ylim(0, current_ylim)  # <-- this is what animates
            ax.set_theta_direction(-1)
            ax.grid(True, color='grey', alpha=0.15, linewidth=0.5)
            ax.spines['polar'].set_visible(False)
            return []

        anim = FuncAnimation(
            fig, update, frames=n_frames, interval=80, blit=False,
        )
        anim.save(
            CURRENT_DIR / 'polar_anim_expanding_ylim.gif',
            writer='pillow', fps=12, dpi=100,
        )
        plt.close(fig)
        print("Saved: polar_anim_expanding_ylim.gif")

    animate_expanding_ylim()
    return


@app.cell
def _(
    CURRENT_DIR,
    FuncAnimation,
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
    def animate_pulsing_scatter():
        """Animation 3: Extreme value markers pulse between two sizes."""
        font = load_google_font("Space Mono", weight="regular", italic=False)
        set_default_font(font)

        # Colors
        background = '#f0f0f0'
        maxim_color = '#98231B'
        minim_color = '#008080'
        too_hot_color = '#cc6633'
        too_cold_color = '#29a6a6'
        tickcolor = '#333333'

        # Theta mapping
        total_months = len(minim_data.time.values)
        global_x = np.arange(0, total_months + 2)
        global_x_min = min(global_x)
        global_x_max = max(global_x)
        datetime_to_x = dict(zip(minim_data.time.values, global_x))

        def x_to_theta(x):
            return 2 * np.pi * (x - global_x_min) / (global_x_max - global_x_min)

        r_offset = 15

        theta_min_line = x_to_theta(np.arange(len(minim_data.time.values)))
        theta_max_line = x_to_theta(np.arange(len(maxim_data.time.values)))
        theta_band_min = x_to_theta(np.arange(len(minim_upper_b.time.values)))
        theta_band_max = x_to_theta(np.arange(len(maxim_upper_b.time.values)))

        # Yearly extremes
        years = range(2011, 2025)
        yearly_max_df = maximum_extremes.loc[
            maximum_extremes.groupby('year')['extreme_temperature'].idxmax()
        ]
        yearly_min_df = minimum_extremes.loc[
            minimum_extremes.groupby('year')['extreme_temperature'].idxmin()
        ]
        extremes_date = np.array(
            [pd.Timestamp(f'{y}-07-01') for y in years], dtype='datetime64[ns]'
        )
        x_extremes = np.array([datetime_to_x.get(d) for d in extremes_date])
        theta_extremes = x_to_theta(x_extremes)

        # Year tick positions
        year_ticks = [pd.Timestamp(f'{y}-01-01') for y in range(2011, 2025)]
        x_ticks = [datetime_to_x.get(d) for d in year_ticks]
        year_thetas = x_to_theta(np.array(x_ticks))

        # Create figure with full static plot (drawn once)
        fig, ax = plt.subplots(
            figsize=(10, 10), dpi=100,
            subplot_kw={'projection': 'polar'},
        )
        fig.set_facecolor(background)
        ax.set_facecolor(background)
        fig.suptitle(
            'Is Greece already experiencing global warming?',
            fontsize=15, y=0.97,
        )

        # Reference circles
        full_theta = np.linspace(0, 2 * np.pi, 300)
        ax.plot(full_theta, np.full_like(full_theta, 0 + r_offset),
                color=too_cold_color, alpha=0.25, ls='dashed', lw=0.8, zorder=-1)
        ax.plot(full_theta, np.full_like(full_theta, 30 + r_offset),
                color=too_hot_color, alpha=0.25, ls='dashed', lw=0.8, zorder=-1)

        # Lines and bands (static)
        ax.plot(theta_min_line, minim_data.values + r_offset,
                color=minim_color, alpha=0.6, linewidth=1.5, zorder=1)
        ax.plot(theta_max_line, maxim_data.values + r_offset,
                color=maxim_color, alpha=0.6, linewidth=1.5, zorder=3)
        ax.fill_between(
            theta_band_min,
            minim_lower_b.values + r_offset,
            minim_upper_b.values + r_offset,
            color=minim_color, alpha=0.1, linewidth=0, zorder=0,
        )
        ax.fill_between(
            theta_band_max,
            maxim_lower_b.values + r_offset,
            maxim_upper_b.values + r_offset,
            color=maxim_color, alpha=0.2, linewidth=0, zorder=2,
        )

        # Scatter — keep references for animation
        scat_hot = ax.scatter(
            theta_extremes,
            yearly_max_df['extreme_temperature'].values + r_offset,
            marker='o', color=too_hot_color, alpha=0.8, s=30, zorder=5,
        )
        scat_cold = ax.scatter(
            theta_extremes,
            yearly_min_df['extreme_temperature'].values + r_offset,
            marker='o', color=too_cold_color, alpha=0.8, s=30, zorder=5,
        )

        # Axis formatting (static)
        ax.set_xticks(year_thetas)
        ax.set_xticklabels([str(y) for y in range(2011, 2025)],
                           size=8, color=tickcolor)
        temp_labels = [-10, 0, 10, 20, 30, 40]
        temp_ticks_vals = [-7, 0, 10, 20, 30, 40]
        ax.set_yticks([tv + r_offset for tv in temp_ticks_vals])
        ax.set_yticklabels([f'{tl}°C' for tl in temp_labels],
                           size=7, color=tickcolor)
        ax.set_rlabel_position(0)
        ax.set_ylim(0, 46 + r_offset)
        ax.set_theta_direction(-1)
        ax.grid(True, color='grey', alpha=0.15, linewidth=0.5)
        ax.spines['polar'].set_visible(False)

        # Pulsing parameters
        s_min, s_max = 15, 80
        n_points = len(theta_extremes)
        n_frames = 60  # 3 full cycles at 20 frames per cycle

        def update(frame_idx):
            # Sine wave oscillation: period = 20 frames
            sine_val = np.sin(2 * np.pi * frame_idx / 20)
            s = s_min + (s_max - s_min) * (sine_val + 1) / 2
            scat_hot.set_sizes(np.full(n_points, s))
            scat_cold.set_sizes(np.full(n_points, s))
            return [scat_hot, scat_cold]

        anim = FuncAnimation(
            fig, update, frames=n_frames, interval=50, blit=False,
        )
        anim.save(
            CURRENT_DIR / 'polar_anim_pulsing_scatter.gif',
            writer='pillow', fps=20, dpi=100,
        )
        plt.close(fig)
        print("Saved: polar_anim_pulsing_scatter.gif")

    animate_pulsing_scatter()
    return


@app.cell
def _(
    CURRENT_DIR,
    FuncAnimation,
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
    def animate_combined():
        """Combined animation: time + ylim expand together, then scatter pulses."""
        font = load_google_font("Space Mono", weight="regular", italic=False)
        set_default_font(font)

        # Colors
        background = '#f0f0f0'
        maxim_color = '#98231B'
        minim_color = '#008080'
        too_hot_color = '#cc6633'
        too_cold_color = '#29a6a6'
        tickcolor = '#333333'

        # Theta mapping
        total_months = len(minim_data.time.values)
        global_x = np.arange(0, total_months + 2)
        global_x_min = min(global_x)
        global_x_max = max(global_x)
        datetime_to_x = dict(zip(minim_data.time.values, global_x))

        def x_to_theta(x):
            return 2 * np.pi * (x - global_x_min) / (global_x_max - global_x_min)

        r_offset = 15

        theta_min_line = x_to_theta(np.arange(len(minim_data.time.values)))
        theta_max_line = x_to_theta(np.arange(len(maxim_data.time.values)))
        theta_band_min = x_to_theta(np.arange(len(minim_upper_b.time.values)))
        theta_band_max = x_to_theta(np.arange(len(maxim_upper_b.time.values)))

        # Yearly extremes
        years = range(2011, 2025)
        yearly_max_df = maximum_extremes.loc[
            maximum_extremes.groupby('year')['extreme_temperature'].idxmax()
        ]
        yearly_min_df = minimum_extremes.loc[
            minimum_extremes.groupby('year')['extreme_temperature'].idxmin()
        ]
        extremes_date = np.array(
            [pd.Timestamp(f'{y}-07-01') for y in years], dtype='datetime64[ns]'
        )
        x_extremes = np.array([datetime_to_x.get(d) for d in extremes_date])
        theta_extremes = x_to_theta(x_extremes)

        # Year tick positions
        year_ticks = [pd.Timestamp(f'{y}-01-01') for y in range(2011, 2025)]
        x_ticks = [datetime_to_x.get(d) for d in year_ticks]
        year_thetas = x_to_theta(np.array(x_ticks))

        # --- Phase 1: expanding time + ylim together ---
        step = 2
        phase1_months = list(range(step, total_months + 1, step))
        if phase1_months[-1] != total_months:
            phase1_months.append(total_months)
        n_phase1 = len(phase1_months)

        # ylim eases out over phase 1
        final_ylim = 46 + r_offset
        start_ylim = r_offset - 5
        t_ease = np.linspace(0, 1, n_phase1)
        eased_t = 1 - (1 - t_ease) ** 2
        ylim_phase1 = start_ylim + (final_ylim - start_ylim) * eased_t

        # --- Phase 2: pulsing scatter ---
        n_phase2 = 60  # 3 full sine cycles at 20 frames per cycle
        s_min, s_max = 15, 80

        total_frames = n_phase1 + n_phase2

        # Create figure
        fig, ax = plt.subplots(
            figsize=(10, 10), dpi=100,
            subplot_kw={'projection': 'polar'},
        )
        fig.set_facecolor(background)
        fig.suptitle(
            'Is Greece already experiencing global warming?',
            fontsize=15, y=0.97,
        )

        n_points = len(theta_extremes)

        def update(frame):
            ax.clear()
            ax.set_facecolor(background)

            if frame < n_phase1:
                # --- Phase 1: growing data + expanding ylim ---
                n = phase1_months[frame]
                current_ylim = ylim_phase1[frame]
                scatter_size = 30  # constant during phase 1
            else:
                # --- Phase 2: full data, pulsing scatter ---
                n = total_months
                current_ylim = final_ylim
                pulse_frame = frame - n_phase1
                sine_val = np.sin(2 * np.pi * pulse_frame / 20)
                scatter_size = s_min + (s_max - s_min) * (sine_val + 1) / 2

            # Reference circles
            full_theta = np.linspace(0, 2 * np.pi, 300)
            ax.plot(full_theta, np.full_like(full_theta, 0 + r_offset),
                    color=too_cold_color, alpha=0.25, ls='dashed', lw=0.8, zorder=-1)
            ax.plot(full_theta, np.full_like(full_theta, 30 + r_offset),
                    color=too_hot_color, alpha=0.25, ls='dashed', lw=0.8, zorder=-1)

            # Temperature lines
            ax.plot(theta_min_line[:n], minim_data.values[:n] + r_offset,
                    color=minim_color, alpha=0.6, linewidth=1.5, zorder=1)
            ax.plot(theta_max_line[:n], maxim_data.values[:n] + r_offset,
                    color=maxim_color, alpha=0.6, linewidth=1.5, zorder=3)

            # Fill bands
            nb_min = min(n, len(theta_band_min))
            ax.fill_between(
                theta_band_min[:nb_min],
                minim_lower_b.values[:nb_min] + r_offset,
                minim_upper_b.values[:nb_min] + r_offset,
                color=minim_color, alpha=0.1, linewidth=0, zorder=0,
            )
            nb_max = min(n, len(theta_band_max))
            ax.fill_between(
                theta_band_max[:nb_max],
                maxim_lower_b.values[:nb_max] + r_offset,
                maxim_upper_b.values[:nb_max] + r_offset,
                color=maxim_color, alpha=0.2, linewidth=0, zorder=2,
            )

            # Scatter for completed years
            completed_years = min(n // 12, 14)
            if completed_years > 0:
                ax.scatter(
                    theta_extremes[:completed_years],
                    yearly_max_df['extreme_temperature'].values[:completed_years] + r_offset,
                    marker='o', color=too_hot_color, alpha=0.8,
                    s=scatter_size, zorder=5,
                )
                ax.scatter(
                    theta_extremes[:completed_years],
                    yearly_min_df['extreme_temperature'].values[:completed_years] + r_offset,
                    marker='o', color=too_cold_color, alpha=0.8,
                    s=scatter_size, zorder=5,
                )

            # Axis formatting
            ax.set_xticks(year_thetas)
            ax.set_xticklabels([str(y) for y in range(2011, 2025)],
                               size=8, color=tickcolor)
            temp_labels = [-10, 0, 10, 20, 30, 40]
            temp_ticks = [-7, 0, 10, 20, 30, 40]
            ax.set_yticks([tv + r_offset for tv in temp_ticks])
            ax.set_yticklabels([f'{tl}°C' for tl in temp_labels],
                               size=7, color=tickcolor)
            ax.set_rlabel_position(0)
            ax.set_ylim(0, current_ylim)
            ax.set_theta_direction(-1)
            ax.grid(True, color='grey', alpha=0.15, linewidth=0.5)
            ax.spines['polar'].set_visible(False)
            return []

        anim = FuncAnimation(
            fig, update, frames=total_frames, interval=80, blit=False,
        )
        anim.save(
            CURRENT_DIR / 'polar_anim_combined.gif',
            writer='pillow', fps=12, dpi=100,
        )
        plt.close(fig)
        print(f"Saved: polar_anim_combined.gif ({n_phase1} growth frames + {n_phase2} pulse frames = {total_frames} total)")

    animate_combined()
    return


@app.cell
def _():
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
