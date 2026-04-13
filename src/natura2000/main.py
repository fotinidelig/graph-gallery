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
    import seaborn as sb
    from pypalettes import load_cmap, create_cmap
    from highlight_text import fig_text, ax_text
    from drawarrow import ax_arrow, fig_arrow
    import geopandas as gpd
    import cartopy.crs as ccrs
    from shapely.geometry import Point
    from pyfonts import load_google_font, set_default_font
    import numpy as np
    from pathlib import Path
    import os
    import marimo as mo

    font = load_google_font("Datatype", weight="regular", italic=False)
    bold_font = load_google_font("Datatype", weight="bold", italic=False)
    set_default_font(font)
    os.getcwdb()
    return (
        Path,
        ax_text,
        bold_font,
        ccrs,
        fig_text,
        gpd,
        load_cmap,
        mo,
        mpl,
        np,
        os,
        pd,
        plt,
    )


@app.cell
def _(Path, pd):
    csv_pth = Path('data/natura2000_2024/csv/')

    species = pd.read_csv(f'{csv_pth}/SPECIES.csv')
    species = pd.read_csv(f'{csv_pth}/SPECIES.csv')
    other_species = pd.read_csv(f'{csv_pth}/OTHERSPECIES.csv')

    other_species = other_species.rename(columns={'SPECIESGROUP': 'SPGROUP'})
    common_cols = other_species.columns.intersection(species.columns)

    all_species = pd.concat([other_species[common_cols], species[common_cols]], ignore_index=True)
    all_species = all_species[all_species.SPGROUP != pd.NA ]
    all_species = all_species.rename(columns={'SITECODE': 'sitecode'})


    sites = pd.read_csv(f'{csv_pth}/NATURA2000SITES.csv')#[['SITECODE', 'AREAHA', 'COUNTRY_CODE']]
    sites = sites.drop_duplicates(subset='SITECODE')
    sites['AREAKM'] = (sites['AREAHA'] / 100).astype(float) # ha to km2
    sites = sites.rename(columns={'SITECODE': 'sitecode'})

    # Load country name and code data
    country_codes = pd.read_json(f'data/europe_countries.json')[['country_name', 'alpha_2']]
    country_codes = {row['alpha_2']: row['country_name'] for _, row in country_codes.iterrows()}
    sites['country'] = sites.COUNTRY_CODE.map(country_codes)
    all_species['country'] = all_species.COUNTRY_CODE.map(country_codes)
    return all_species, sites


@app.cell
def _(all_species, sites):
    cover_per_country = sites.groupby('country')['AREAKM'].sum().reset_index(name='cover').sort_values(by='cover',ascending=False)

    sites_per_country = sites.groupby('country')['sitecode'].nunique().reset_index(name='num_sites').sort_values(
        by='num_sites',ascending=False)

    species_per_country = all_species.groupby('country')['SPECIESNAME'].nunique().reset_index(name="num_species").sort_values(
        by='num_species', ascending=False)
    return cover_per_country, sites_per_country


@app.cell
def _(mo):
    mo.md(r"""### Polar Bar Plot""")
    return


@app.cell
def _(Path, mpl, np, pd, sites_per_country):
    # Create colormap based on country's size

    def create_custom_cmap(start_color, end_color, n_colors, name='custom_cmap'):
        """
        Create a custom colormap that interpolates between two colors.

        Parameters:
        -----------
        start_color : str or tuple
            Starting color (darkest). Can be hex string, color name, or RGB tuple.
        end_color : str or tuple
            Ending color (brightest). Can be hex string, color name, or RGB tuple.
        n_colors : int
            Number of colors in the colormap (including start and end).
        name : str
            Name for the colormap.

        Returns:
        --------
        matplotlib.colors.ListedColormap
            A colormap with n_colors interpolated between start and end.
        """
        start_rgb = np.array(mpl.colors.to_rgb(start_color))
        end_rgb = np.array(mpl.colors.to_rgb(end_color))
        t = np.linspace(0, 1, n_colors)

        colors = np.array([start_rgb + (end_rgb - start_rgb) * ti for ti in t])

        cmap = mpl.colors.ListedColormap(colors, name=name)
        return cmap

    countries = list(sites_per_country.country.unique())

    world_data = Path("./data/europe_countries.json")
    world_data = pd.read_json(world_data)
    world_data = world_data[world_data['country_name'].isin(countries)][['country_name', 'area_km2', 'area_comment']].sort_values(by='area_km2', ascending=False)
    # fix area according to area_comment
    world_data.loc[world_data['country_name'] == 'France', 'area_km2'] = 643801
    world_data.loc[world_data['country_name'] == 'Spain', 'area_km2'] = 505990
    world_data = world_data.drop(['area_comment'], axis=1)
    # countries sorted by area
    countries_by_area = world_data.country_name.to_list()

    dark_color = '#000033'
    light_color = '#f06dfc' #ce93d8
    cmap = create_custom_cmap(dark_color, light_color, len(countries))  # Dark blue to light gray

    hex_colors = [mpl.colors.to_hex(c, keep_alpha=False) for c in cmap.colors]
    world_data = world_data.assign(color=hex_colors)

    # Create a mapping of country to color (cycling through colors if more than 27 countries)
    country_color_map = dict(zip(world_data.country_name.to_list(), world_data.color.to_list()))
    return countries, country_color_map


@app.cell
def _(ax_text, country_color_map, np, plt):
    # create plot

    _fig, _ax = plt.subplots(subplot_kw={'projection': 'polar'})
    _ax.axis(False)

    def plot_radial_bar(ax, df, column, type='', offset=0, invert=False, text=True):
        heights = df[column].to_numpy()
        max_height = df[column].max()
        # scale from current scale to 0-1
        deviation = df[column].max()-df[column].min()
        heights = heights/deviation

        ax.set_ylim(-.2, 1)
        width = 2*np.pi/len(heights)

        angles = [offset+i*width for i in range(0, len(heights))]
        if invert:
            angles = [np.pi-o for o in angles]
        _countries = df['country'].to_list()
        current_colors = [country_color_map[c] for c in _countries]

        bars = ax.bar(
            x=angles, 
            height=heights, 
            width=width, 
            color=current_colors,
            linewidth=1, 
            edgecolor='white',)

        # Annotations
        num_annotations = 6 # number of bars to annotate, starting from highest bar (sorted) 
    
        values = df[column].to_numpy() #
        sum_ = sum(values)
        values = [int(v) for v in values]
        postfix = ''
        if max(values) >= 10000:
            values = [int(v/1000) for v in values]
            postfix = 'k'

        if not text:
            return

        pad = 0.01  # radial padding in your normalized units    
        for i, (patch, val) in enumerate(zip(bars.patches[:num_annotations], values[:num_annotations])):
            country = _countries[i]
        
            theta0 = patch.get_x()
            dtheta = patch.get_width()
            r_top  = patch.get_height()
            theta_mid = theta0 + dtheta / 2
            rotation_val = np.degrees(theta_mid-np.pi/2) # subtract np.pi/2 so that text is parallel to bar
            ax.text(
                theta_mid, r_top + pad, str(val)+postfix,
                ha="center", va="bottom",
                rotation=rotation_val,
                rotation_mode="anchor",
                color="black", fontsize=8,
            )
        
            rotation_country = np.degrees(theta_mid if not invert else  theta_mid-np.pi)
            ax.text(theta_mid, r_top - pad, s=country, ha='right' if not invert else 'left', va='top', 
                    rotation=rotation_country, rotation_mode="anchor", color='white', size=8)
        
        if sum_ > 10000:
            sum_ = np.round(sum_/1000, decimals=2)
            postfix = 'k'
        else:
            postfix = ''
        ax_text(ax=ax, s=f"Total:\n{sum_}{postfix}\n{type}", x=0, y=-.2, ha='center', va='center', 
                textalign='center', color='black', size=9)
    return (plot_radial_bar,)


@app.cell
def _(
    bold_font,
    ccrs,
    countries,
    country_color_map,
    cover_per_country,
    fig_text,
    gpd,
    os,
    plot_radial_bar,
    plt,
    sites_per_country,
):
    _fig, _ax = plt.subplots(figsize=(11,8), layout='tight')

    back_color = '#E9EEE8'
    _fig.set_facecolor(back_color)
    _ax.set_facecolor(back_color)
    _ax.axis(False)

    _axx_cover = _ax.inset_axes([.25, 0, 0.7, 1], projection='polar')
    plot_radial_bar(_axx_cover, cover_per_country, 'cover', type='km2', offset=0)
    _axx_sites = _ax.inset_axes([.05, 0, 0.7, 1], projection='polar')
    plot_radial_bar(_axx_sites, sites_per_country, 'num_sites', type='sites', offset=0, invert=True)
    _axx_species = _ax.inset_axes([-.1, 0, 0.6, 1], projection='polar')
    # plot_radial_bar(_axx_species, plants_per_country, 'num_species', offset=np.pi/3, invert=True)

    for a in [ _axx_sites, _axx_cover, _axx_species]:
        a.axis(False)

    # Countries Legend

    world = gpd.read_file('./data/eu_asia_africa.geo.json')

    projection = ccrs.Mercator()
    previous_proj = ccrs.PlateCarree() # default projection of GeoPandas
    world = world.to_crs(projection.to_proj4())
    world = world[world['name'].isin(countries)]

    _axx_legend = _ax.inset_axes([.6, -.1, .6, .5])
    _axx_legend.axis(False)

    _axx_legend.set_xlim(-.2*1e7, .4*1e7)
    _axx_legend.set_ylim(.37*1e7, 1.15*1e7)

    world_map_colors = [country_color_map[k] for k in world.name]
    world.plot(ax=_axx_legend, color=world_map_colors, edgecolor=back_color, linewidth=.6)

    # Information

    bbox = dict(facecolor=(1,1,1,1), edgecolor='none', pad=1)

    # 1. Spain's largest site
    fig_text(s='<Spain> has the highest protected area, with the largest being <Montes de Toledo>, spanning almost <2.2k> km2.\nIt\'s the most important site for the protection of the <iberian lynx>.', textalign='left', highlight_textprops=[{'font': bold_font, 'color': country_color_map['Spain'], 'bbox': dict(facecolor=(1,1,1,1), edgecolor='none', pad=1)}, {'font': bold_font, 'bbox': dict(facecolor=(1,1,1,1), edgecolor='none', pad=1)}, {'font': bold_font}, {'font': bold_font, 'bbox': dict(facecolor=(1,1,1,1), edgecolor='none', pad=1)}], x=.15, y=.4, size=9)

    # 2. largest site in Sweden
    fig_text(s='The largest protected site is <Vindelfjällen> in <Sweden>, with an area of <5.5k> km2.\nThe <arctic fox> breeds there, as well as other protected large mammals.', textalign='left', highlight_textprops=[{'font': bold_font, 'bbox': bbox}, {'font': bold_font, 'color': country_color_map['Sweden'], 'bbox': bbox}, {'font': bold_font}, {'font': bold_font, 'bbox': bbox}],
            x=.2, y=.3, size=9)

    fig_text(s='The most common plant living in <18> EU countries is\nthe moss species <dicranum viride>, found mostly in woodlands.', textalign='left', highlight_textprops=[{'font': bold_font}, {'font': bold_font, 'color': '#158109', 'bbox': bbox}], x=.25, y=.2, size=9,)

    fig_text(s='Natura2000, world\'s largest protected area database, shows Europe\'s reach nature.', size=13,
            font=bold_font, color='#421655', x=.5, y=.85, ha='center', textalign='center')

    plt.savefig(f'{os.path.dirname(__file__)}/natura2000_polarbar.svg')
    plt.show()
    return


@app.cell
def _(mo):
    mo.md(r"""## Lolipop Plot""")
    return


@app.cell
def _(all_species, sites):
    sites_species = all_species.groupby(['country', 'sitecode'])['SPECIESNAME'].nunique().reset_index(name='COUNT')
    # remove sites with <= 1 species
    sites_species = sites_species[sites_species['COUNT'] > 1].reset_index(drop=True)
    area_data = (
        sites.groupby(["sitecode", "SITENAME"], as_index=False)["AREAKM"]
        .sum()
        .rename(columns={"AREAKM": "AREA"})
        .sort_values("AREA", ascending=False)
        .drop_duplicates(subset=["SITENAME"], keep="first")
        .reset_index(drop=True)
    )
    area_data = area_data[area_data['AREA'] > 1].reset_index(drop=True)
    return area_data, sites_species


@app.cell
def _(area_data, np, sites_species):
    data = area_data.merge(sites_species, on='sitecode', how='left').dropna(axis=0)
    data = data.sort_values(by='AREA', ascending=False).reset_index(drop=True)
    data["x_plot"] = np.nan
    return (data,)


@app.cell
def _(ax_text, data, load_cmap, np, os, plt):
    scatter_only = False

    _fig ,_ax = plt.subplots(layout='tight', figsize=(10,7))
    _fig.subplots_adjust(left=.2, right=.8, top=.8, bottom=.2)
    _ax.spines[["top", "right", "left", "bottom"]].set_visible(False)
    # if scatter_only:
    #     background_color = 'none'
    # else:
    #     background_color = 'white'
    background_color = 'white'
    _fig.set_facecolor(background_color)
    _ax.set_facecolor(background_color)

    primary_text = '#8DC56A'#'#025940'
    big_text = '#0F1926'

    max_area, min_area = data['AREA'].max(), data['AREA'].min()
    max_species, min_species = data['COUNT'].max(), data['COUNT'].min()

    data['area_norm'] = (data['AREA']-min_area) / (max_area-min_area)
    data['species_norm'] = (data['COUNT']-min_species) / (max_species-min_species)

    area_cmap = load_cmap("Acanthostracion_polygonius", cmap_type="continuous", reverse=True) #BluGrn
    species_cmap = load_cmap("SunsetDark", cmap_type="continuous")

    y_margin = .05
    n_y_ticks = 5
    area_span = max_area - min_area
    spec_span = max_species - min_species
    area_tick_vals = np.array([8750, 17500, 35000, 52500, 70000])
    area_tick_y = (area_tick_vals - min_area) / area_span + y_margin
    spec_tick_vals = np.array([100, 200, 400, 600, 800])
    spec_tick_y = -((spec_tick_vals - min_species) / spec_span)

    tick_y = np.concatenate([spec_tick_y, area_tick_y])
    tick_labels = [f"{v:.0f}" for v in spec_tick_vals] + [f"{v:,.0f}" for v in area_tick_vals]
    order = np.argsort(tick_y)
    _ax.set_yticks(tick_y[order])
    _ax.set_yticklabels(np.array(tick_labels)[order], fontsize=7)
    _ax.set_ylim(-1.02, 1.02 + y_margin)
    _ax.tick_params(axis="y", length=0, pad=-10, labelcolor=primary_text)
    _ax.set_xticks([],[])

    if not scatter_only:
        for tick in tick_y:
            _ax.hlines(y=tick, xmin=0, xmax=len(data), color='#E0D890', linewidth=0.3, alpha=.5)

    x_idx = 0
    area_norm_global = plt.Normalize(vmin=float(data["area_norm"].min()), vmax=float(data["area_norm"].max()))
    species_norm_global = plt.Normalize(
        vmin=float(data["species_norm"].min()), vmax=float(data["species_norm"].max())
    )
    # Create a small image for the gradient
    # n_bins = 100
    # colors = ['white', background_color] # left to right color
    # gradient = np.linspace(0, 1, n_bins).reshape(1, n_bins)
    # background_cmap = LinearSegmentedColormap.from_list('gradient', colors, N=n_bins)

    for country in data["country"].unique():
        c_data = data[data['country'] == country]
        y_area = c_data['area_norm']
        y_species = c_data['species_norm']
        num = len(c_data['sitecode'])

        x = np.arange(x_idx, x_idx + num)
        data.loc[c_data.index, "x_plot"] = x # set mapping from index to x position

        if not scatter_only:
            _ax.vlines(x_idx, -1, 1, colors='lightgrey', linewidth=0.2, alpha=.4)

        x_idx += num
        colors_area = area_cmap(area_norm_global(y_area.to_numpy()))
        colors_species = species_cmap(species_norm_global(y_species.to_numpy()))


        _ax.vlines(x, y_margin, y_area+y_margin, colors=colors_area, linewidth=0.2, alpha=.3)
        _ax.vlines(x, -y_species, 0, colors=colors_species, linewidth=0.2, alpha=.3)

        s_min, s_max = 2, 10
        s_area = s_min + (s_max - s_min) * y_area.to_numpy()
        s_species = s_min + (s_max - s_min) * y_species.to_numpy()

        _ax.scatter(x, y_area + y_margin, c=colors_area, s=s_area)
        _ax.scatter(x, -y_species, c=colors_species, s=s_species)

        if not scatter_only:
            country_name = c_data["country"].iloc[0]
            ax_text(
                x=x_idx - num,
                y=0.04,
                # s=str(country_name).capitalize() if pd.notna(country_name) else "",
                s=country,
                textalign="left",
                size=8,
            )

    # Compute Outliers

    if not scatter_only:

        outliers_area = data[data['area_norm'] > 0.5]
        outliers_species = data[data['species_norm'] > 0.7]
        # Get indices as numpy array
        outliers_area_idx = outliers_area.index.values
        outliers_species_idx = outliers_species.index.values

        # Get corresponding y and x values for annotation
        y_outliers_area = data['area_norm'].to_numpy()[outliers_area_idx]
        y_outliers_species = data['species_norm'].to_numpy()[outliers_species_idx]

        x_outliers_area = outliers_area["x_plot"].to_numpy()
        x_outliers_species = outliers_species["x_plot"].to_numpy()

        # Annotate area outliers
        outliers_code_spec = []
        outliers_code_area = []
        for idx, (x_pos, y_pos) in enumerate(zip(x_outliers_area, y_outliers_area)):
            _ax.annotate(
                outliers_area.iloc[idx]['sitecode'],  # or whatever label you want
                xy=(x_pos, y_pos),
                xytext=(5, 5),
                textcoords='offset points',
                fontsize=7
            )
            outliers_code_area.append(outliers_area.iloc[idx]['sitecode'])

        # Annotate species outliers similarly
        for idx, (x_pos, y_pos) in enumerate(zip(x_outliers_species, -y_outliers_species)):
            _ax.annotate(
                outliers_species.iloc[idx]['sitecode'],
                xy=(x_pos, y_pos),
                xytext=(5, -5),
                textcoords='offset points',
                fontsize=7
            )
            outliers_code_spec.append(outliers_species.iloc[idx]['sitecode'])


    if not scatter_only:
        # plt.savefig(f'{os.path.dirname(__file__)}/natura2000_lolipop.svg')
        plt.savefig(f'{os.path.dirname(__file__)}/natura2000_lolipop.png')
    else:
        plt.savefig(f'{os.path.dirname(__file__)}/natura2000_lolipop_data.png', dpi=1200)

    plt.show()
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
