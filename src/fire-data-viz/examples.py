import marimo

__generated_with = "0.16.5"
app = marimo.App(width="medium")


@app.cell
def _(mo):
    mo.md(r"""### Multiple line plots - grayed out lines""")
    return


@app.cell
def _():
    import matplotlib.pyplot as plt
    import pandas as pd
    from pypalettes import load_cmap
    import seaborn as sns

    url = "https://raw.githubusercontent.com/JosephBARBIERDARNAL/data-matplotlib-journey/refs/heads/main/economic/economic.csv"
    df = pd.read_csv(url)

    fig, axs = plt.subplots(
       nrows=3, ncols=3,
       figsize=(12, 8),
       layout="tight"
    )

    # list of all country names
    countries = df["country"].unique()

    for country, ax in zip(countries, axs.flat):

       # draw other lines in the background
       other_df = df[df["country"] != country]
       for other_country in other_df["country"].unique():
          x = other_df.loc[other_df["country"] == other_country, "date"]
          y = other_df.loc[other_df["country"] == other_country, "unemployment rate"]
          ax.plot(x, y, alpha=0.2, color="lightgrey")

       x = df.loc[df["country"] == country, "date"]
       y = df.loc[df["country"] == country, "unemployment rate"]

       ax.plot(x, y, color="cyan")
       ax.set_ylim(0, 15)
       ax.set_xlim("2020-01-01", "2024-01-01")
       ax.spines[["top", "right"]].set_visible(False)
       ax.set_xticks(
          ["2020-01-01", "2021-01-01", "2022-01-01", "2023-01-01", "2024-01-01"],
          labels=[2020, 2021, 2022, 2023, 2024]
       )
       ax.text(
          x="2024-01-01", y=9,
          s=country.upper(), # upper case
          ha="right",
          size=12,
       )

    plt.show()
    return load_cmap, pd, plt, sns


@app.cell
def _(pd, plt, sns):

    _url = "https://raw.githubusercontent.com/JosephBARBIERDARNAL/data-matplotlib-journey/refs/heads/main/economic/economic.csv"
    _df = pd.read_csv(_url)

    _fig, _axs = plt.subplots(
        ncols=3, nrows=3, layout="tight"
    )

    for _country, _ax in zip(_df["country"].unique(), _axs.flat):
      _x = _df.loc[_df["country"] == _country, "consumer confidence"]
      sns.kdeplot(_x, ax=_ax, fill=True, color='purple')
      _ax.text(x=-49, y=.07, size=7, s=f'Consumer confidence in \n{_country}')
      _ax.set_xlim(-60, 130)
      _ax.set_ylim(0, .1)
      _ax.set_xlabel("")
      _ax.set_ylabel("")
      _ax.set_yticks([],[])
      _ax.set_xticks([-50, 0, 50, 100], [-50, 0, 50, 100])
      _ax.tick_params(length=0)
      _ax.spines[['top', 'left', 'right']].set_visible(False)


    plt.show()
    return


@app.cell
def _(load_cmap, pd, plt):
    def multiple_line_plot_grid(url):
        df = pd.read_csv(url)

        colors = load_cmap("Antique", reverse=True).colors
        background_color = "#f2f2f2"

        fig, axs = plt.subplots(nrows=3, ncols=3, figsize=(8, 6), layout='tight')

        countries = df["country"].unique()

        color_map = dict(zip(countries, colors))

        for country, ax in zip(countries, axs.flat):
          x = df.loc[df['country'] == country, 'date']
          y = df.loc[df['country'] == country, 'consumer confidence']

          ax.plot(x, y, color=color_map[country])
          ax.text(x=45, y=120, s=country.capitalize(), 
                  size=10, weight='bold', ha='right',
                  color=color_map[country])

          other_countries = df[df['country'] != country]['country'].unique()
          for other_country in other_countries:
            x = df.loc[df['country'] == other_country, 'date']
            y = df.loc[df['country'] == other_country, 'consumer confidence']

            ax.plot(x, y, color='#A7AAA9', linewidth=.5)

          ax.tick_params(length=0, labelsize=6)
          ax.grid(axis='y', alpha=.5)
          ax.set_ylim(-60, 140)
          ax.set_xlim("2020-01-01", "2024-01-01")
          ax.spines[["top", "right", "bottom"]].set_visible(False)
          ax.set_xticks(
            ["2020-01-01", "2022-01-01", "2024-01-01"],
            labels=[2020, 2022, 2024]
          )
          ax.set_yticks([-50, 0, 50, 100], [-50, 0, 50, 100])

          ax.set_facecolor(background_color)
        fig.set_facecolor(background_color)

        plt.show()

    _url = "https://raw.githubusercontent.com/JosephBARBIERDARNAL/data-matplotlib-journey/refs/heads/main/economic/economic.csv"
    multiple_line_plot_grid(_url)
    return


@app.cell
def _(mo):
    mo.md(r"""### Subplot Mosaics""")
    return


@app.cell
def _(pd, plt):
    def mosaic(url):

        df = pd.read_csv(url)
        x = df["sepal_length"]
        y = df["sepal_width"]

        scheme = """
        BBB.
        AAAC
        AAAC
        AAAC
        """

        fig, axs = plt.subplot_mosaic(scheme)
        fig.subplots_adjust(wspace=0, hspace=0)
        fig.set_facecolor('#f1f1f1')

        for ax in axs:
          axs[ax].set_facecolor('#f1f1f1')

        axs["B"].axis("off") # remove all spines and ticks
        axs["C"].axis("off") # remove all spines and ticks

        axs["A"].scatter(df["sepal_length"], df["sepal_width"],color='black')
        axs["B"].hist(df["sepal_length"], color='#95170E', 
                      edgecolor='black', linewidth=1)
        axs["C"].hist(df["sepal_width"], orientation="horizontal", color='#950E8C',
                     edgecolor='black', linewidth=1)

        axs['A'].spines[['bottom', 'left']].set_visible(False)
        axs['A'].set_xticks([],[])
        axs['A'].set_yticks([],[])

        axs['A'].text(x=8, y=4.3, s='Sepal Length', color='#95170E',
                     ha='right', weight='bold')
        axs['A'].text(x=8, y=2, s='Sepal Width', color='#950E8C',
                     rotation=270, ha='right', weight='bold')

        plt.show()


    _url = "https://raw.githubusercontent.com/holtzy/The-Python-Graph-Gallery/refs/heads/master/static/data/iris.csv"
    mosaic(_url)
    return


@app.cell
def _(mo):
    mo.md(r"""### Axes within another axes""")
    return


@app.cell
def _(plt):
    import numpy as np
    from PIL import Image
    import requests
    from io import BytesIO

    def plot_img_in_ax(img_url):
        response = requests.get(img_url)
        img = Image.open(BytesIO(response.content))
        img = np.array(img)

        fig, ax = plt.subplots()
        ax.imshow(img, alpha=0.3) # alpha for transparency of image
        ax.axis("off")

        child_ax = ax.inset_axes([0.2, 0.2, 0.3, 0.3])
        child_ax.plot([1, 2, 3], [1, 2, 3])
        child_ax.text(x=1, y=3.3, s="Time spent sleeping")
        child_ax.patch.set_alpha(0) # transparency of smaller plot

        plt.show()

    img_url = "https://raw.githubusercontent.com/JosephBARBIERDARNAL/data-matplotlib-journey/refs/heads/main/misc/cat.png"
    plot_img_in_ax(img_url)
    return


@app.cell
def _(mo):
    mo.md(r"""### Maps and Choropleth""")
    return


@app.cell
def _(load_cmap, pd, plt):
    import geopandas as gpd
    from drawarrow import ax_arrow

    def example_choropleth():
        url = "https://raw.githubusercontent.com/JosephBARBIERDARNAL/data-matplotlib-journey/refs/heads/main/world/world.geojson"
        world = gpd.read_file(url)

        url = "https://raw.githubusercontent.com/JosephBARBIERDARNAL/data-matplotlib-journey/refs/heads/main/CO2/CO2.csv"
        df = pd.read_csv(url)

        world = world.merge(df, left_on="code_adm", right_on="ISO")
        europe = world[world["continent"] == "Europe"]
        europe = europe[europe["name"] != "Russia"]

        cmap = load_cmap("Sunset2", cmap_type="continuous")

        fig, ax = plt.subplots(layout="tight")
        fig.set_facecolor('lightgrey')
        ax.set_facecolor('lightgrey')
        europe.plot(
          column="Total",
          cmap=cmap,
          edgecolor="black",
          linewidth=0.4,
          legend=True,
          legend_kwds={"shrink": 0.5},
          ax=ax
        )

        ax.set_xlim(-25, 41)
        ax.set_ylim(33, 82)
        ax.axis("off")

        ax_arrow(
          [0, 65], [5.5, 50],
          color="black",
          radius=0.2,
          fill_head=False
        )
        ax.text(x=-5, y=66, s="Luxembourg")

        ax.text(
          x=-25, y=73,
          s="CO2/Capita in Europe in 2021",
          size=16
        )

        plt.show()

    example_choropleth()
    return (gpd,)


@app.cell
def _(gpd, load_cmap, pd, plt):
    def multiple_maps():
        url = "https://raw.githubusercontent.com/JosephBARBIERDARNAL/data-matplotlib-journey/refs/heads/main/world/world.geojson"
        world = gpd.read_file(url)

        url = "https://raw.githubusercontent.com/JosephBARBIERDARNAL/data-matplotlib-journey/refs/heads/main/CO2/CO2.csv"
        df = pd.read_csv(url)

        world = world.merge(df, left_on="code_adm", right_on="ISO")
        africa = world[world["continent"] == "Africa"]

        fig, ax = plt.subplots(ncols=2, figsize=(8,5))

        cmap = load_cmap('Exter', cmap_type="continuous")

        plot_kws = {
          'edgecolor': 'darkred',
          'linewidth': .5,
          'cmap': cmap,
          'legend': True,
          'legend_kwds': {
            'shrink': .5,
            'extend': 'both',
            # 'ticks': [],
            },
        }

        africa.plot(ax=ax[0], column='Coal', 
                    vmin=0, vmax=3, **plot_kws)

        africa.plot(ax=ax[1], column='Gas', **plot_kws)

        for axx in ax:
          axx.axis('off')
        ax[0].text(x=75, y=-30, s='low')
        ax[0].text(x=75, y=30, s='high')
        plt.show()

    multiple_maps()
    return


@app.cell
def _(mo):
    mo.md(r"""### Bubble Maps""")
    return


@app.cell
def _(gpd, pd, plt):
    import cartopy.crs as ccrs

    def bubble_map():
        url = "https://raw.githubusercontent.com/JosephBARBIERDARNAL/data-matplotlib-journey/refs/heads/main/world/world.geojson"
        world = gpd.read_file(url)

        url = "https://raw.githubusercontent.com/JosephBARBIERDARNAL/data-matplotlib-journey/refs/heads/main/earthquakes/earthquakes.csv"
        df = pd.read_csv(url)

        projection = ccrs.Mercator()
        previous_proj = ccrs.PlateCarree() # default projection of GeoPandas
        world = world.to_crs(projection.to_proj4())

        new_coords = projection.transform_points(
            previous_proj,
            df["Longitude"],
            df["Latitude"]
        )
        x = new_coords[:, 0] # new longitude
        y = new_coords[:, 1] # new latitude

        min_s = 10
        max_s = 1000
        s = df["Depth"]
        s = min_s + (s - s.min()) * (max_s - min_s) / (s.max() - s.min())

        fig, ax = plt.subplots(
          subplot_kw={"projection": projection},
          layout="tight"
        )

        world.plot(ax=ax, color="lightgrey")
        scatter = ax.scatter(x, y, s=s, alpha=.2, color='lightgreen')

        handles, labels = scatter.legend_elements(prop='sizes', num=3, alpha=.7, color='lightgreen') # <-- good way to add legend!
        ax.legend(handles, labels, loc=(0.9, 0.9), title='depth', frameon=False, framealpha=0)
        ax.axis("off")

        plt.show()

    bubble_map()
    return


@app.cell
def _(mo):
    mo.md(r"""### Example KDE map with transparent map as background""")
    return


@app.cell
def _(gpd, pd, plt, sns):
    def kde_map():
        url = "https://raw.githubusercontent.com/JosephBARBIERDARNAL/data-matplotlib-journey/refs/heads/main/newyork/newyork.geojson"
        newyork = gpd.read_file(url)

        url = "https://raw.githubusercontent.com/JosephBARBIERDARNAL/data-matplotlib-journey/refs/heads/main/newyork-airbnb/newyork-airbnb.csv"
        df = pd.read_csv(url)
        df = df.sample(2000) # random sample

        fig, ax = plt.subplots(layout="tight")

        sns.kdeplot(
           x=df["longitude"],
           y=df["latitude"],
           fill=True,
           bw_adjust=0.2,
        )
        newyork.plot(ax=ax, color=(1,1,1,0), edgecolor="black") # <-- (1,1,1,0) is RGBA format to control opacity (here fully transparent)
        ax.axis('off')

        plt.show()

    kde_map()
    return


@app.cell
def _():
    import marimo as mo
    return (mo,)


if __name__ == "__main__":
    app.run()
