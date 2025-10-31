import marimo

__generated_with = "0.16.5"
app = marimo.App(width="medium")


@app.cell
def _():
    import matplotlib.pyplot as plt
    import matplotlib as mpl
    import pandas as pd
    from pypalettes import load_cmap
    from highlight_text import fig_text, ax_text
    from drawarrow import ax_arrow
    import seaborn as sns
    import geopandas as gpd
    import cartopy.crs as ccrs
    from pyfonts import load_google_font, set_default_font
    from matplotlib.lines import Line2D
    import numpy as np
    import regex as re
    from pathlib import Path
    from matplotlib.animation import FuncAnimation
    from PIL import Image, ImageSequence
    import marimo as mo
    return (
        FuncAnimation,
        Image,
        ImageSequence,
        Path,
        ax_arrow,
        ax_text,
        ccrs,
        fig_text,
        gpd,
        load_cmap,
        load_google_font,
        mo,
        mpl,
        np,
        pd,
        plt,
        re,
        set_default_font,
    )


@app.cell
def _(mo):
    mo.md(
        r"""
    ###Bubble Map 
    ####with bubble color (number of fires) and size (burned areas in hectares) dimension & animation for time dimension
    """
    )
    return


@app.cell
def _():
    return


@app.cell
def _(Path, load_cmap, mpl, pd, re):
    # Get data from files into dictionaries
    CURRENT_DIR = Path(__file__).resolve().parent

    # Navigate relative to this file’s location
    DATA_DIR = CURRENT_DIR.parent.parent / "data" / "fire-data"
    pattern = re.compile(r'estimates-by-country_([A-Z]{3})_2006_2024')

    country_dfs: dict[str, pd.DataFrame] = {}

    MIN_YEAR, MAX_YEAR = 2006, 2023

    first_df = True
    for file in DATA_DIR.glob('*.csv'):
        match = pattern.match(file.name)
        if not match:
            continue
        code = match.group(1)
        _country_df = pd.read_csv(file)
        _country_df = _country_df.assign(Code=lambda x: code)[_country_df['Year'].between(MIN_YEAR, MAX_YEAR)]
        _country_df = _country_df.rename(columns={"Burned Area [ha]": "Burned Area (ha)"})
        if first_df:
            fire_df = _country_df
            first_df = False
            continue
        fire_df = pd.concat([fire_df, _country_df])

    # NOTE: Emissions not used for the moment
    emission_pattern = re.compile(
        r"MCD64\.006\.yearly-gfed-emissions-burned-area\.\d{4}-\d{4}\.([A-Z]{3})_[A-Za-z]+"
    )
    first_emission_df = True
    for _file in DATA_DIR.glob('*.csv'):
        _match = emission_pattern.match(_file.name)
        if not _match:
            continue
        code = _match.group(1)
        _country_df = pd.read_csv(_file)
        _country_df = _country_df.assign(Code=lambda x: code)
        _country_df = _country_df[['Year', 'CO2 - Carbon Dioxide [t]', 'PM2.5 - Fine Particulate Matter [t]', 'Code']]
        _country_df = _country_df[_country_df['Year'].between(MIN_YEAR, MAX_YEAR)]
        if first_emission_df:
            emision_df = _country_df
            first_emission_df = False
            continue
        emision_df = pd.concat([emision_df, _country_df])

    fire_df = fire_df.merge(emision_df, left_on=['Year', 'Code'], right_on=['Year', 'Code'])

    # Create column for bubble sizes based on burned area
    min_s, max_s = 10, 200
    max_ba = fire_df['Burned Area (ha)'].max()
    min_ba = fire_df['Burned Area (ha)'].min()
    fire_df = fire_df.assign(
        bubble_size=lambda x: min_s + (x['Burned Area (ha)']-min_ba)*(max_s-min_s)/(max_ba-min_ba)
    )
    # Create column for bubble color based on number of fires
    colormap = load_cmap('pal12', cmap_type='continuous', reverse=False)

    norm = mpl.colors.Normalize(
        vmin=fire_df["Number of Fires"].min(),
        vmax=fire_df["Number of Fires"].max()
    )
    sm = mpl.cm.ScalarMappable(cmap=colormap, norm=norm)
    fire_df['bubble_color'] = fire_df['Number of Fires'].apply(lambda v: mpl.colors.to_hex(sm.to_rgba(v)))
    return CURRENT_DIR, DATA_DIR, fire_df, max_ba, max_s, min_ba, min_s, sm


@app.cell
def _(DATA_DIR, ccrs, gpd, pd):
    world = gpd.read_file(DATA_DIR / 'eu_asia_africa.geo.json')

    projection = ccrs.Mercator()
    previous_proj = ccrs.PlateCarree() # default projection of GeoPandas
    world = world.to_crs(projection.to_proj4())
    world = world[['name', 'adm0_a3', 'geometry']]

    capitals = pd.read_csv(DATA_DIR / "capitals.csv")[['iso3','lon','lat','country']]

    new_coords = projection.transform_points(
        previous_proj,
        capitals["lon"],
        capitals["lat"]
    )
    capitals = capitals.assign(lon=new_coords[:, 0])
    capitals = capitals.assign(lat=new_coords[:, 1])

    world = world.merge(capitals, left_on='adm0_a3', right_on='iso3')
    return (world,)


@app.cell
def _(
    CURRENT_DIR,
    FuncAnimation,
    Image,
    ImageSequence,
    ax_arrow,
    ax_text,
    fig_text,
    fire_df,
    load_google_font,
    max_ba,
    max_s,
    min_ba,
    min_s,
    np,
    plt,
    set_default_font,
    sm,
    world,
):
    # GLOBAL UNCHANGED ATTRIBUTES
    font = load_google_font("Space Mono", weight="regular", italic=True)
    bold_font = load_google_font("Space Mono", weight="bold", italic=True)
    set_default_font(font)
    background = '#a6bddb'
    details_color = '#3c4856'
    annot_color = '#683a3a'
    muted_color = '#a0acbd'
    bubble_alpha = .4

    yearly_ba = fire_df.groupby(['Year'])['Burned Area (ha)'].sum().reset_index()
    max_ba_year = int(yearly_ba.loc[yearly_ba['Burned Area (ha)'].idxmax(), 'Year'])
    max_ba_yearly = yearly_ba['Burned Area (ha)'].max()


    # Greece Data
    GR_fire_df = fire_df[fire_df['Code'] == 'GRC']
    GR_max_ba_year = int(GR_fire_df.loc[GR_fire_df['Burned Area (ha)'].idxmax(), 'Year'])
    GR_max_ba = GR_fire_df['Burned Area (ha)'].max()

    _fig, _ax = plt.subplots(figsize=(8, 6), dpi=150,)
    _fig.subplots_adjust(left=0, right=1, top=1, bottom=0)

    _fig.set_facecolor(background)
    _ax.set_facecolor(background)

    # Elements that need update
    cbar = _fig.colorbar(sm, ax=_ax)

    def update(frame):
        '''frame: the year in our case'''
        global cbar

        _ax.clear()
        _ax.set_position([0, 0, 1, 1])  # [left, bottom, width, height]
        _ax.axis('off')
        _ax.set_xlim(-.25*1e7, .55*1e7)
        _ax.set_ylim(.17*1e7, .77*1e7)

        YEAR = frame
        year_fire_df = fire_df[fire_df['Year'] == YEAR]
        year_fire_world = world.merge(year_fire_df, left_on='adm0_a3', right_on='Code')
        world.plot(ax=_ax, color=muted_color)
        year_fire_world.plot(ax=_ax, color='grey', edgecolor=details_color, linewidth=.2)

        # Bubbles for fire data
        x = year_fire_world['lon'].values
        y = year_fire_world['lat'].values

        scatter = _ax.scatter(x, y, s=year_fire_world['bubble_size'], alpha=bubble_alpha, 
                              c=year_fire_world['bubble_color'], 
                              linewidth=1.5,)

        # Annotate Greece's largest fires
        if YEAR >= GR_max_ba_year:
            gr_x = year_fire_world.loc[year_fire_world['Code'] == 'GRC', 'lon'].values[0]
            gr_y = year_fire_world.loc[year_fire_world['Code'] == 'GRC', 'lat'].values[0]
            ax_text(ax=_ax, x=gr_x, y=gr_y*1.18, 
                    s=f'Greece saw its largest fires \nduring <{GR_max_ba_year}>: <{int(GR_max_ba/1000)}k> hectares \nwere burned',
                   bbox=dict(facecolor=(1,1,1,0.4), edgecolor='none', pad=1),
                   highlight_textprops=[{'font': bold_font}, {'font': bold_font}])
            ax_arrow(head_position=[gr_x, gr_y], tail_position=[gr_x, gr_y*1.1], 
                    radius=.6, head_length=5, fill_head=False, color=details_color)

        # Create simple line plot showing total burned area for all countries
        x = yearly_ba[yearly_ba['Year']<=YEAR]['Year'].values.tolist()
        if len(x) >= 2:
            mid_year = yearly_ba[yearly_ba['Year']<=YEAR]['Year'].mean()
            y = yearly_ba[yearly_ba['Year']<=YEAR]['Burned Area (ha)'].values.tolist()

            yearly_ba_ax = _ax.inset_axes([.72, .03, .2, .2])
            yearly_ba_ax.plot(x, y, color=details_color)
            yearly_ba_ax.fill_between(x, y, alpha=.5, color=details_color)
            yearly_ba_ax.scatter(x[-1], y[-1], color=details_color, zorder=1)
            yearly_ba_ax.text(x[-1]+.5, y[-1], s=f"{int(max(y)/1000)}k", color=details_color, size=8)

            yearly_ba_ax.set_ylim(min(y), max(y)*1.6)
            yearly_ba_ax.text(
                0.5, .89, " Total burned area (ha)",
                transform=yearly_ba_ax.transAxes,
                ha='center', va='bottom',
                fontsize=8, color=details_color,
            )
            yearly_ba_ax.set_xticks([min(x), max(x)],[min(x), max(x)])
            yearly_ba_ax.set_yticks([])
            yearly_ba_ax.tick_params(labelsize=7, length=2, labelcolor=details_color,
                                color=details_color, pad=1) 
            yearly_ba_ax.spines[['bottom','top','right', 'left']].set_visible(False)
            yearly_ba_ax.set_facecolor((1, 1, 1, 0.4))

            # Custom annotations for specific years
            if YEAR >= max_ba_year:
                ax_text(ax=yearly_ba_ax, x=max_ba_year, y=max_ba_yearly*1.1, 
                        s=f'<{max_ba_year}> saw the \n<most> burned area', 
                      color=annot_color, va='bottom', ha='center', textalign='center', clip_on=False, size=7, 
                        highlight_textprops=[{'font': bold_font}, {'font': bold_font}])

            if YEAR >= max_ba_year:
                yearly_ba_ax.scatter(max_ba_year, max_ba_yearly, color=annot_color, zorder=1)
                yearly_ba_ax.vlines(x=max_ba_year, ymin=min(y), ymax=max_ba_yearly, 
                                    color=annot_color, linestyles='dashed')

        # Add the colorbar explaining number of fires values to a sub axis
        bar_ax = _ax.inset_axes([.93, .67, .1, .3])
        bar_ax.axis('off')
        cbar.remove()
        cbar = _fig.colorbar(sm, ax=bar_ax, fraction=1)
        # sm.set_norm(mpl.colors.Normalize(
        #     vmin=year_fire_world["Number of Fires"].min(),
        #     vmax=year_fire_world["Number of Fires"].max()
        # ))
        # cbar.update_normal(sm)
        cbar.outline.set_visible(False) 
        cbar.ax.tick_params(labelsize=7, length=0, labelcolor=details_color,
                            color=details_color, pad=1) 
        cbar.set_label("Number of Fires", size=8, color=details_color)
        cbar.ax.yaxis.set_label_position("left")

        # Add legend for bubble size
        bubble_ax = _ax.inset_axes([.81, .67, .1, .3])
        _x = [0.2, 0.2, 0.2, 0.2]
        _y = [.15, .25, .35, .45]

        bubble_ax.set_xlim(0.1, .5)
        bubble_ax.set_ylim(0, .58)
        b_sizes = np.linspace(min_s+(max_s-min_s)/3, max_s, 4)
        bubble_ax.scatter(_x, _y, s=b_sizes, color=details_color, alpha=bubble_alpha)
        bubble_ax.set_facecolor((0, 0, 0, 0))
        for spine in bubble_ax.spines.values():
            spine.set_visible(False)
        bubble_ax.tick_params(
            left=False, bottom=False, labelleft=False, labelbottom=False
        )
        bubble_ax.set_ylabel('Burned Area (ha)', size=8, color=details_color,labelpad=0)

        b_labels = (
            (min_ba + (b_sizes - min_s) * (max_ba - min_ba) / \
             (max_s - min_s))
        ).astype(int)
        b_labels = [
            str(int(v/1000))+"k" if v > 1000 else
            str(v)
            for v in b_labels
        ]

        for i, lab in enumerate(b_labels):
            bubble_ax.text(x=.26, y=_y[i], s=lab, size=7, color=details_color, va='center',)

        ax_text(ax=_ax, x=.5, y=.87, s=f'Year: <{YEAR}>', size=13, color=details_color, 
                highlight_textprops=[{'font': bold_font}], va='center', ha='center',
                annotationbbox_kw={'boxcoords': _fig.transFigure},
                bbox=dict(facecolor=(1,1,1,0.4), edgecolor='none', pad=.3))

        fig_text(fig=_fig, x=.5, y=.9, s=f'Fires in mediterranean countries \nshow an <increasing trend> after <2019>',
                va='bottom', ha='center', color=details_color, size=15, textalign='center',
                highlight_textprops=[{'font': bold_font}, {'font': bold_font}], clip_on=False)

        return _ax

    years = fire_df['Year'].unique().tolist()#[:7]
    ani = FuncAnimation(
        fig=_fig,
        func=update,
        frames=years,
        interval=500, # in ms,
    )
    gif_path = CURRENT_DIR / 'web-animation-with-text.gif'
    ani.save(gif_path, writer='imagemagick', fps=5)

    # get initial frames
    with Image.open(gif_path) as img:
        frames = [frame.copy() for frame in ImageSequence.Iterator(img)]

    n_frames = len(years)
    initial_duration = 300 # in ms
    pause_frames = [max_ba_year, GR_max_ba_year]
    pause_frames = [y-min(years) for y in pause_frames]

    for i, frame in enumerate(frames):

        frame_duration = initial_duration

        if i in pause_frames:
            frame_duration = 500
        if i == len(frames) - 1:
            frame_duration = 5000

        frame.info['duration'] = frame_duration

    frames[0].save(gif_path, save_all=True, append_images=frames[1:], loop=0)
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
