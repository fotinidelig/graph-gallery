import marimo

__generated_with = "0.16.5"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo
    import plotly.graph_objects as go
    import plotly.express as px
    import plotly.io as pio
    pio.renderers.default = "browser" 
    import pandas as pd
    import geopandas as gpd
    import json
    import numpy as np
    import matplotlib.pyplot as plt
    import matplotlib.colors as mcolors
    from pathlib import Path
    from pyfonts import load_google_font, set_default_font
    from scipy import stats

    from habitat_colors import (
        assign_cluster,
        assign_cluster_color,
        CLUSTER_COLORS,
    )

    from config import COUNTRY_CODES_NAMES
    from visualizations import create_colorscale, get_text_color_for_background
    return (
        CLUSTER_COLORS,
        Path,
        assign_cluster,
        create_colorscale,
        get_text_color_for_background,
        go,
        gpd,
        json,
        mcolors,
        np,
        pd,
        px,
    )


@app.cell
def _(Path, assign_cluster, pd):
    csv_pth = Path('./data/natura2000_2024/csv/')
    habitats = pd.read_csv(csv_pth / 'HABITATS.csv')

    habitatclass = pd.read_csv(csv_pth / 'HABITATCLASS.csv')

    habitats['DETAIL_DESCRIPTION'] = habitats['DESCRIPTION']
    habitats = habitats.drop(['DESCRIPTION'], axis=1)
    habitats = habitats.merge(habitatclass[['DESCRIPTION', 'SITECODE']], on='SITECODE', how='left')
    habitats['CLUSTER'] = habitats['DESCRIPTION'].map(
        lambda r: assign_cluster(r)
    )

    habitats['COVER_KM'] = habitats['COVER_HA'] / 100
    habitats = habitats.drop(['COVER_HA'], axis=1)

    print(habitats.columns)
    print(habitatclass.columns)
    return csv_pth, habitatclass, habitats


@app.cell
def _(habitatclass, habitats):
    habitat_uniques = habitats.groupby('HABITATCODE')['DESCRIPTION'].nunique().reset_index(name='uniques')
    zeros = habitat_uniques.uniques == 0
    habitat_uniques[zeros]
    habitatclass.SITECODE.nunique(), habitats.SITECODE.nunique()
    return


@app.cell
def _(mo):
    mo.md(r"""### Scatter plot spiral""")
    return


@app.cell
def _(np):
    def generate_squashed_spiral(
        N,
        a=0.1,
        b=0.8,
        theta_max=6*np.pi,
        scale_x=1.2,
        scale_y=0.5,
        center=(0.0, 0.0)
    ):
        """
        Generate N points on a squashed Archimedean spiral.

        Parameters
        ----------
        N : int
            Number of points.
        a : float
            Initial radius offset.
        b : float
            Radial growth factor.
        theta_max : float
            Maximum angle (controls number of turns).
        scale_x : float
            Horizontal scaling factor.
        scale_y : float
            Vertical scaling factor (squash factor).
        center : tuple(float, float)
            Spiral center (cx, cy).

        Returns
        -------
        points : np.ndarray shape (N, 2)
        """
        cx, cy = center

        theta = np.linspace(0, theta_max, N)
        r = a + b * theta

        x = r * np.cos(theta)
        y = r * np.sin(theta)

        # Squash transformation
        x *= scale_x
        y *= scale_y

        x += cx
        y += cy

        return np.column_stack((x, y))
    return (generate_squashed_spiral,)


@app.cell
def _(CLUSTER_COLORS, assign_cluster, habitatclass, habitats):
    description_count = habitatclass.groupby('DESCRIPTION')['HABITATCODE'].count().reset_index(name='COUNT')

    cover = habitats.groupby(['DESCRIPTION', 'CLUSTER'])['COVER_KM'].sum().reset_index(name='COVER_KM')
    cover['COVER_NORM'] = cover['COVER_KM'] / cover['COVER_KM'].max()

    description_count_sorted = description_count.sort_values(by='COUNT', ascending=False)
    description_cover_sorted = description_count_sorted.merge(cover, on='DESCRIPTION', how='left')
    description_cover_sorted['COLOR'] = description_cover_sorted['DESCRIPTION'].map(
        lambda r: CLUSTER_COLORS[assign_cluster(r)]
    )
    return (description_cover_sorted,)


@app.cell
def _(
    CLUSTER_COLORS,
    description_cover_sorted,
    generate_squashed_spiral,
    np,
    px,
):
    xy = generate_squashed_spiral(
        len(description_cover_sorted['COUNT'].values),
        # cover_values=description_cover_sorted['COVER_KM'].values,
        # aspect_ratio=1.5  # Elliptical (wider)
    )
    x = xy[:,0]
    y = xy[:,1]

    description_cover_sorted['x'] = x
    description_cover_sorted['y'] = y
    description_cover_sorted['# of habitats (k)'] = (description_cover_sorted.COUNT/1000).map(lambda x: np.round(x, decimals=2))
    description_cover_sorted['Cover (k km2)'] = (description_cover_sorted.COVER_KM/1000).map(lambda x: int(x))


    # Create scatter plot
    _fig = px.scatter(
        description_cover_sorted,
        x='x',
        y='y',
        size='COVER_KM',  # Circle size = cover
        color='CLUSTER',
        color_discrete_map=CLUSTER_COLORS,
        hover_name='DESCRIPTION',
        hover_data={'x': False, 'y': False, 'COVER_KM': False,
                   'DESCRIPTION': False, 'COUNT': False, '# of habitats (k)': True, 'Cover (k km2)': True},
        size_max=45,  # Adjust max circle size
    )

    # Add "Cover Area" title above legend
    _fig.update_layout(
        plot_bgcolor='white',
        height=500,
        width=800,
        margin=dict(l=0, r=0, t=0, b=0),
        showlegend=True,
        font_family='Noto Sans Mono',
    )

    # Configure axes
    _fig.update_layout(
        xaxis=dict(
            showgrid=False,
            showticklabels=False,
            zeroline=False,
            title=""
        ),
        yaxis=dict(
            showgrid=False,
            showticklabels=False,
            zeroline=False,
            title=""
        ),
        plot_bgcolor='white',
        height=500,
        width=1000,
        margin=dict(l=0, r=0, t=0, b=0),
        showlegend=True,
        font_family='Noto Sans Mono',
        legend=dict(
            title="Habitat Cluster",
            x=1.02,
            y=1,
            xanchor='left',
            yanchor='top'
        ),
        hovermode='closest',
        hoverdistance=5,
        hoverlabel=dict(
            bordercolor='white',
            font_size=12,
            font_family='Noto Sans Mono',
            font_color='white',
            align='left'
        ),
        legend_itemclick="toggleothers",
    )
    # Point to a specific bubble (e.g., index 0, or find by description)

    target_idx = description_cover_sorted[description_cover_sorted['DESCRIPTION'] == 'Evergreen woodland'].index[0]
    target_x = description_cover_sorted.loc[target_idx, 'x']
    target_y = description_cover_sorted.loc[target_idx, 'y']

    _fig.add_annotation(
        x=target_x,
        y=target_y,
        text="The larger the bubble,<br>the more area is covered<br>(kilo km2)",
        showarrow=True,
        arrowhead=2,
        arrowsize=1.3,
        arrowwidth=1.5,
        arrowcolor='rgba(123, 138, 110, 0.9)',
        ax=90,
        ay=-25,
        bgcolor='rgba(123, 138, 110, 0.9)',
        bordercolor='rgba(123, 138, 110, 0.9)',
        borderwidth=0,
        font=dict(size=12, family='Noto Sans Mono', color='#FCFFF7'),
        xanchor='left',
        yanchor='bottom'
    )

    target_idx = description_cover_sorted[description_cover_sorted['DESCRIPTION'] == 'Broad-leaved deciduous woodland'].index[0]
    target_x = description_cover_sorted.loc[target_idx, 'x']
    target_y = description_cover_sorted.loc[target_idx, 'y']

    _fig.add_annotation(
        x=target_x,
        y=target_y,
        text="Habitats closer to the center are more popular<br>(by count of habitat occurences).",
        showarrow=True,
        clicktoshow=False,
        arrowhead=2,
        arrowsize=1.3,
        arrowwidth=1.5,
        arrowcolor='rgba(123, 138, 110, 0.7)',
        ax=120,
        ay=-30,
        bgcolor='rgba(123, 138, 110, 0.7)',
        bordercolor='rgba(123, 138, 110, 0.7)',
        borderwidth=0,
        font=dict(size=12, family='Noto Sans Mono', color='#FCFFF7'),
        xanchor='left',
        yanchor='middle',
    )

    _fig.write_html("./graph_gallery/src/natura2000_interactive/habitat_type_scatter.html")
    _fig.show()
    return


@app.cell
def _(mo):
    mo.md(r"""### Sankey Diagram""")
    return


@app.cell
def _(habitats):
    # habitat_country_area = habitat[['HABITATCODE', 'SITECODE', 'COUNTRY_CODE', 'DESCRIPTION', 'CLUSTER', 'COVER_HA']]
    habitat_country_cover = habitats.groupby(['CLUSTER', 'DESCRIPTION', 'COUNTRY_CODE'])['COVER_KM'].sum().reset_index(name='COVER_KM')
    habitat_country_cover['COVER_PERCENT'] = (
        habitat_country_cover.groupby('DESCRIPTION')['COVER_KM']
        .transform(lambda x: x / x.sum())
    )

    def rgb_to_rgba(rgb, alpha=0.6):
        """Convert RGB tuple (0-1) to rgba string (0-255)."""
        r, g, b = rgb
        return f'rgba({int(r * 255)}, {int(g * 255)}, {int(b * 255)}, {alpha})'
    return habitat_country_cover, rgb_to_rgba


@app.cell
def _(CLUSTER_COLORS, go, habitat_country_cover, mcolors, rgb_to_rgba):
    # Step 1: Get all unique nodes
    clusters = habitat_country_cover["CLUSTER"].unique()
    descriptions = habitat_country_cover["DESCRIPTION"].unique()
    countries = habitat_country_cover["COUNTRY_CODE"].unique()

    # Step 2: Create node labels (all nodes in order)
    node_labels = list(clusters) + list(descriptions) + list(countries)

    # Step 3: Create index mapping
    cluster_indices = {cluster: i for i, cluster in enumerate(clusters)}
    desc_indices = {desc: len(clusters) + i for i, desc in enumerate(descriptions)}
    country_indices = {
        country: len(clusters) + len(descriptions) + i
        for i, country in enumerate(countries)
    }

    # Step 4: Create links CLUSTER -> DESCRIPTION
    cluster_desc_links = (
        habitat_country_cover.groupby(["CLUSTER", "DESCRIPTION"])["COVER_KM"]
        .sum()
        .reset_index()
    )

    source_cluster_desc = [
        cluster_indices[cluster] for cluster in cluster_desc_links["CLUSTER"]
    ]
    target_cluster_desc = [
        desc_indices[desc] for desc in cluster_desc_links["DESCRIPTION"]
    ]
    value_cluster_desc = cluster_desc_links["COVER_KM"].tolist()

    # Step 5: Create links DESCRIPTION -> COUNTRY_CODE
    desc_country_links = (
        habitat_country_cover.groupby(["DESCRIPTION", "COUNTRY_CODE"])["COVER_KM"]
        .sum()
        .reset_index()
    )

    source_desc_country = [
        desc_indices[desc] for desc in desc_country_links["DESCRIPTION"]
    ]
    target_desc_country = [
        country_indices[country] for country in desc_country_links["COUNTRY_CODE"]
    ]
    value_desc_country = desc_country_links["COVER_KM"].tolist()

    # Step 6: Combine all links
    source = source_cluster_desc + source_desc_country
    target = target_cluster_desc + target_desc_country
    value = value_cluster_desc + value_desc_country

    desc_to_cluster = (
        habitat_country_cover.groupby("DESCRIPTION")["CLUSTER"].first().to_dict()
    )
    node_colors = []
    node_x = []
    node_y = []
    for i, label in enumerate(node_labels):
        if i < len(clusters):
            node_colors.append(CLUSTER_COLORS.get(label, "#bbbaba"))
            node_x.append(0)
            node_y.append(i / max(len(clusters) - 1, 1))
        elif i < len(clusters) + len(descriptions):
            desc_name = descriptions[i - len(clusters)]
            desc_cluster = desc_to_cluster.get(desc_name)
            cluster_color = CLUSTER_COLORS.get(desc_cluster, "#bbbaba")
            node_colors.append(cluster_color)
            node_x.append(0.3)
            desc_idx = i - len(clusters)
            node_y.append(desc_idx / max(len(descriptions) - 1, 1))
        else:
            # Third tier: countries - light gray
            node_colors.append("#f0f0f0")
            node_x.append(1.0)
            country_idx = i - len(clusters) - len(descriptions)
            node_y.append(country_idx / max(len(countries) - 1, 1))

    # Color links by source cluster
    link_colors = []

    for src_idx in source:
        if src_idx < len(clusters):
            # Link from cluster
            cluster_name = node_labels[src_idx]
            base_color_hex = CLUSTER_COLORS.get(cluster_name, "#bbbaba")
            rgb = mcolors.to_rgb(base_color_hex)
            rgba = rgb_to_rgba(rgb, alpha=0.6)
            link_colors.append(rgba)
        else:
            # Link from description (DESCRIPTION -> COUNTRY_CODE)
            desc_name = node_labels[src_idx]
            desc_cluster = desc_to_cluster.get(desc_name)
            base_color_hex = CLUSTER_COLORS.get(desc_cluster, "#bbbaba")
            rgb = mcolors.to_rgb(base_color_hex)
            rgba = rgb_to_rgba(rgb, alpha=0.4)
            link_colors.append(rgba)

    node_text_labels = list(clusters) + [""] * len(descriptions) + list(countries)
    _fig = go.Figure(
        data=[
            go.Sankey(
                node=dict(
                    label=node_text_labels,
                    pad=15,
                    thickness=25,
                    line=dict(color="white", width=0),
                    color=node_colors,
                    customdata=node_labels,
                    hovertemplate="<b>%{customdata}</b><br>Cover: %{value} km2<extra></extra>",
                    x=node_x,  # Manual x positions
                    y=node_y,  # Manual y positions
                ),
                link=dict(
                    source=source,
                    target=target,
                    value=value,
                    color=link_colors,
                    customdata=node_labels,
                    hovertemplate="<b>%{target.customdata}</b><br>"+
                    "%{source.customdata}<br>"
                    + "Cover: %{value}"
                    + "<extra></extra>",
                ),
            )
        ]
    )

    _fig.update_layout(
        title="Habitat Distribution: Cluster → Description → Country",
        font_size=10,
        height=800,
    )

    _fig.write_html(
        "./graph_gallery/src/natura2000_interactive/habitat_country_sankey.html"
    )

    _fig.show()
    return (countries,)


@app.cell
def _(CLUSTER_COLORS, go, habitat_country_cover, mcolors, rgb_to_rgba):
    def cluster_to_country():
        # Step 1: Get all unique nodes
        clusters = habitat_country_cover["CLUSTER"].unique()
        descriptions = habitat_country_cover["DESCRIPTION"].unique()
        countries = habitat_country_cover["COUNTRY_CODE"].unique()

        # Step 2: Create node labels (all nodes in order)
        node_labels = list(descriptions) + list(clusters) + list(countries)

        # Step 3: Create index mapping
        desc_indices = {desc: i for i, desc in enumerate(descriptions)}
        cluster_indices = {cluster: len(descriptions) + i for i, cluster in enumerate(clusters)}
        country_indices = {
            country: len(clusters) + len(descriptions) + i
            for i, country in enumerate(countries)
        }

        # Step 4: Create links CLUSTER -> DESCRIPTION
        desc_cluster_links = (
            habitat_country_cover.groupby(["DESCRIPTION", "CLUSTER"])["COVER_KM"]
            .sum()
            .reset_index()
        )

        source_desc_cluster = [
            desc_indices[desc] for desc in desc_cluster_links["DESCRIPTION"]
        ]
        target_desc_cluster = [
            cluster_indices[cluster] for cluster in desc_cluster_links["CLUSTER"]
        ]
        value_cluster_desc = desc_cluster_links["COVER_KM"].tolist()

        # Step 5: Create links DESCRIPTION -> COUNTRY_CODE
        cluster_country_links = (
            habitat_country_cover.groupby(["CLUSTER", "COUNTRY_CODE"])["COVER_KM"]
            .sum()
            .reset_index()
        )

        source_cluster_country = [
            cluster_indices[cluster] for cluster in cluster_country_links["CLUSTER"]
        ]
        target_cluster_country = [
            country_indices[country] for country in cluster_country_links["COUNTRY_CODE"]
        ]
        value_cluster_country = cluster_country_links["COVER_KM"].tolist()

        # Step 6: Combine all links
        source = source_desc_cluster + source_cluster_country
        target = target_desc_cluster + target_cluster_country
        value = value_cluster_desc + value_cluster_country

        desc_to_cluster = (
            habitat_country_cover.groupby("DESCRIPTION")["CLUSTER"].first().to_dict()
        )
        node_colors = []
        node_x = []
        node_y = []
        for i, label in enumerate(node_labels):
            if i < len(descriptions):
                desc_cluster = desc_to_cluster.get(label)
                node_colors.append(CLUSTER_COLORS.get(desc_cluster, "#bbbaba"))
                node_x.append(0)
                node_y.append(i / max(len(clusters) - 1, 1))
            elif i < len(clusters) + len(descriptions):
                cluster_name = clusters[i - len(descriptions)]
                cluster_color = CLUSTER_COLORS.get(cluster_name, "#bbbaba")
                node_colors.append(cluster_color)
                node_x.append(0.3)
                cluster_idx = i - len(descriptions)
                node_y.append(cluster_idx / max(len(clusters) - 1, 1))
            else:
                # Third tier: countries - light gray
                node_colors.append("#f0f0f0")
                node_x.append(1.0)
                country_idx = i - len(clusters) - len(descriptions)
                node_y.append(country_idx / max(len(countries) - 1, 1))

        # Color links by source cluster
        link_colors = []

        for src_idx in source:
            if src_idx < len(descriptions):
                # Link from description
                desc_name = node_labels[src_idx]
                desc_cluster = desc_to_cluster.get(desc_name)
                base_color_hex = CLUSTER_COLORS.get(desc_cluster, "#bbbaba")
                rgb = mcolors.to_rgb(base_color_hex)
                rgba = rgb_to_rgba(rgb, alpha=0.6)
                link_colors.append(rgba)
            else:
                # Link from description (DESCRIPTION -> COUNTRY_CODE)
                cluster_name = node_labels[src_idx]
                base_color_hex = CLUSTER_COLORS.get(cluster_name, "#bbbaba")
                rgb = mcolors.to_rgb(base_color_hex)
                rgba = rgb_to_rgba(rgb, alpha=0.4)
                link_colors.append(rgba)

        node_text_labels = [""] * len(descriptions) + list(clusters) + list(countries)
        _fig = go.Figure(
            data=[
                go.Sankey(
                    node=dict(
                        label=node_text_labels,
                        pad=15,
                        thickness=25,
                        line=dict(color="white", width=0),
                        color=node_colors,
                        customdata=node_labels,
                        hovertemplate="<b>%{customdata}</b><br>Cover: %{value} km2<extra></extra>",
                        x=node_x,  # Manual x positions
                        y=node_y,  # Manual y positions
                    ),
                    link=dict(
                        source=source,
                        target=target,
                        value=value,
                        color=link_colors,
                        customdata=node_labels,
                        hovertemplate="<b>%{target.customdata}</b><br>"+
                        "%{source.customdata}<br>"
                        + "Cover: %{value}"
                        + "<extra></extra>",
                    ),
                )
            ]
        )

        _fig.update_layout(
            title="Habitat Distribution: Description → Cluster → Country",
            font_size=10,
            height=800,
        )

        _fig.write_html(
            "./graph_gallery/src/natura2000_interactive/cluster_country_sankey.html"
        )
        _fig.show()
    cluster_to_country()
    return


@app.cell
def _(mo):
    mo.md(r"""### Chloropleth""")
    return


@app.cell
def _(Path, gpd):
    map_countries = gpd.read_file(Path('./graph_gallery/src/data/world.geo.json'))
    return (map_countries,)


@app.cell
def _(Path, countries, map_countries, pd):
    country_codes = pd.read_json(Path('./data/europe_countries.json'))[['country_name', 'alpha_2']]
    country_codes = {row['alpha_2']: row['country_name'] for _, row in country_codes.iterrows()}
    country_names = [country_codes[c] for c in countries]

    europe = map_countries[map_countries.iso_a2_eh.isin(countries)]
    return country_codes, europe


@app.cell
def _(europe, habitat_country_cover):
    cluster_country_cover = (
        habitat_country_cover.groupby(["COUNTRY_CODE", "CLUSTER"])["COVER_KM"]
        .sum()
        .reset_index()
    )

    choropleth_data = (
        cluster_country_cover.merge(
            europe[['geometry', 'iso_a2_eh']], left_on='COUNTRY_CODE', right_on='iso_a2_eh', how='left')
        .drop(['iso_a2_eh'], axis=1)
    )
    # choropleth_data
    return (choropleth_data,)


@app.cell
def _(CLUSTER_COLORS, choropleth_data, europe, go, json, mcolors):
    from plotly.subplots import make_subplots
    from config import COLORS, FONT_FAMILY,INLINE_FONTSIZE

    europe_geojson = json.loads(europe.to_json())

    # Use up to 8 _clusters (4x2 grid), prefer explicit CLUSTER_COLORS order
    available_clusters = [
        c for c in CLUSTER_COLORS.keys()
        if c in choropleth_data["CLUSTER"].unique() and c != "Other"
    ]
    _clusters = available_clusters[:8]


    rows, cols = 2, 4
    specs = [[{"type": "choropleth"} for _ in range(cols)] for _ in range(rows)]
    _fig = make_subplots(
        rows=rows,
        cols=cols,
        specs=specs,
        subplot_titles=_clusters,
        horizontal_spacing=0.0,
        vertical_spacing=0.0,
    )
    for ann in _fig.layout.annotations:
        # Move titles a bit down towards the maps (tune the value)
        ann.y = ann.y - 0.03
        ann.yanchor = "top"  

    def _cluster_colorscale(hex_color):
        """Build a light→full colorscale from a cluster hex color."""
        r, g, b = mcolors.to_rgb(hex_color)
        # Light version, slightly tinted toward background
        bg_r, bg_g, bg_b = mcolors.to_rgb(COLORS["background"])
        light = (
            0.8 * bg_r + 0.2 * r,
            0.8 * bg_g + 0.2 * g,
            0.8 * bg_b + 0.2 * b,
        )
        light_rgba = f"rgba({int(light[0]*255)}, {int(light[1]*255)}, {int(light[2]*255)}, 0.5)"
        full_rgba = f"rgba({int(r*255)}, {int(g*255)}, {int(b*255)}, 1.0)"
        return [
            [0.0, light_rgba],
            [1.0, full_rgba],
        ]

    for idx, cluster in enumerate(_clusters):
        r = idx // cols + 1
        c = idx % cols + 1

        df_cluster = choropleth_data[choropleth_data["CLUSTER"] == cluster]
        df_cluster = df_cluster.assign(
            COVER_NORM=lambda x:  x.COVER_KM /df_cluster.COVER_KM.sum()
        )

        if df_cluster.empty:
            continue

        base_hex = CLUSTER_COLORS.get(cluster, "#bbbaba")
        colorscale = _cluster_colorscale(base_hex)
        if idx==0:
            print(colorscale)

        _fig.add_trace(
            go.Choropleth(
                locations=df_cluster["COUNTRY_CODE"],
                # z=df_cluster["COVER_KM"],
                z=df_cluster["COVER_NORM"],
                geojson=europe_geojson,
                featureidkey="properties.iso_a2_eh",
                colorscale=colorscale,
                marker_line_width=0.4,
                marker_line_color=COLORS["white"],
                showscale=False,
                customdata = df_cluster["COVER_KM"]/1000,
                hovertemplate="<b>%{location}</b><br>"
                "Cover: %{customdata:.0f}k km²<extra></extra>",
            ),
            row=r,
            col=c,
        )

    # Minimal map styling for all geos
    _fig.update_geos(
        # fitbounds="locations",
        visible=False,
        showcountries=False,
        showcoastlines=False,
        showland=True,
        projection_scale=5.0,
        center=dict(lat=55, lon=20),
        landcolor=COLORS["white"],
        bgcolor=COLORS["white"],
    )

    _fig.update_layout(
        height=380,
        width=800,
        plot_bgcolor=COLORS["white"],
        paper_bgcolor=COLORS["white"],
        font=dict(
            family=FONT_FAMILY,
            size=INLINE_FONTSIZE,
            color=COLORS["text_primary"],
        ),
        margin=dict(l=0, r=0, t=0, b=0),
    )
    _fig.show()
    return COLORS, FONT_FAMILY, INLINE_FONTSIZE


@app.cell
def _(mo):
    mo.md(r"""### Pictogram""")
    return


@app.cell
def _(csv_pth, pd):
    species = pd.read_csv(f'{csv_pth}/SPECIES.csv')
    species = pd.read_csv(f'{csv_pth}/SPECIES.csv')
    other_species = pd.read_csv(f'{csv_pth}/OTHERSPECIES.csv')

    other_species = other_species.rename(columns={'SPECIESGROUP': 'SPGROUP'})
    common_cols = other_species.columns.intersection(species.columns)

    all_species = pd.concat([other_species[common_cols], species[common_cols]], ignore_index=True)
    all_species.SPGROUP = all_species.SPGROUP.astype(str)
    all_species = all_species[all_species.SPGROUP != 'nan']
    return (all_species,)


@app.cell
def _(all_species):
    all_species.columns
    return


@app.cell
def _(all_species):
    types = all_species.SPGROUP.unique().tolist()
    species_count = all_species.groupby('SPGROUP').SPECIESNAME.nunique().reset_index(name='COUNT')
    species_count['COUNT_HUNDREDS'] = species_count.COUNT / 100
    species_count = species_count.sort_values(by='COUNT', ascending=False)
    print(species_count.SPGROUP)

    SPECIES_COLORS = {
        'Amphibians': '#5A9A8B', #'#4ECDC4',  # Amphibians - Bright teal
        'Birds': '#4A7BA7', #'#5B9BD5',  # Birds - Light blue
        'Fish': '#2E86AB', #'#2E75B6',  # Fish - Deep blue
        'Invertebrates': '#9B7BB8', #'#C55AD5',  # Invertebrates - Magenta
        'Lichens': '#8B8B7A', #'#A0A0A0',  # Lichens - Medium gray
        'Mammals': '#8B6F47', #'#C55A5A',  # Mammals - Terracotta
        'Plants': '#6B8E5A', #'#70AD47',  # Plants - Fresh green
        'Reptiles': '#A67C52' #'#D2691E'   # Reptiles - Chocolate
    }

    SPECIES_GROUPS = {
        0: 'Amphibians',
        1: 'Birds',
        2: 'Fish',
        3: 'Invertebrates',
        4: 'Lichens',
        5: 'Mammals',
        6: 'Plants',
        7: 'Reptiles'
    }

    def get_species_color(species_group):
        """
        Get the color for a species group.

        Parameters:
        -----------
        species_group : string
            The species group 

        Returns:
        --------
        str
            Hex color code for the species group
        """
        return SPECIES_COLORS.get(species_group, '#bbbaba')  # Default gray if not found
    return SPECIES_COLORS, get_species_color, species_count


@app.cell
def _(
    COLORS,
    FONT_FAMILY,
    INLINE_FONTSIZE,
    get_species_color,
    get_text_color_for_background,
    go,
    np,
    species_count,
):
    species_count_data = species_count.copy()
    species_count_data['COUNT_HUNDREDS'] = species_count_data['COUNT'] / 100

    # Spiral parameters
    initial_radius = 0  # Starting radius at the center
    angular_spacing = 0.1  # Angular increment in radians (controls spacing along spiral)
    radius_growth_rate = 0.2  # How much radius increases per angular step

    # Track position along spiral (continuous angle, increasing radius)
    current_angle = 0.0  # Start at angle 0
    current_radius = initial_radius
    series = []

    for _i, row in species_count_data.iterrows():
        _x = []
        _y = []
        count = int(row.COUNT_HUNDREDS)
        stype = row.SPGROUP
    
        for j in range(0, count):
            # Calculate position along spiral
            # Radius increases gradually as we go around
            current_radius = initial_radius + radius_growth_rate * current_angle
        
            # Convert polar to Cartesian coordinates
            curr_x = current_radius * np.cos(current_angle)
            curr_y = current_radius * np.sin(current_angle)
        
            _x.append(curr_x)
            _y.append(curr_y)
        
            # Move to next position along spiral
            current_angle += angular_spacing
    
        species_color = get_species_color(stype)
        text_color = get_text_color_for_background(species_color)
        series.append(
            go.Scatter(
                x=_x, 
                y=_y, 
                mode='markers', 
                marker={
                    'symbol': 'circle', 
                    'line': dict(width=1, color=COLORS['white']),
                    'size': 12, 
                    'color': species_color
                }, 
                name=f'{stype} ({row.COUNT_HUNDREDS})',
                hovertemplate=f'{stype} ({row.COUNT_HUNDREDS})<extra></extra>',
                hoverlabel=dict(
                    bgcolor=species_color,
                    bordercolor='rgba(0,0,0,0)',
                    font_size=INLINE_FONTSIZE,
                    font_family=FONT_FAMILY,
                    font_color=text_color,
                ),
            ),
        )

    _fig = go.Figure(
        dict(
            data=series, 
            layout=go.Layout(
                # title={'text': "Species", 'x': 0.5, 'xanchor': 'center'},
                paper_bgcolor=COLORS['background'],
                plot_bgcolor='rgba(0,0,0,0)',
                xaxis=dict(
                    showgrid=False,
                    zeroline=False, 
                    showline=False, 
                    visible=False, 
                    showticklabels=False,
                    scaleanchor='y',
                    scaleratio=1  # Keep aspect ratio 1:1 for circular appearance
                ),
                yaxis=dict(
                    showgrid=False,
                    zeroline=False, 
                    showline=False, 
                    visible=False, 
                    showticklabels=False
                ),
                legend=dict(
                    title_text="Species count (in hundreds)",
                    title_font=dict(
                        family=FONT_FAMILY,
                        size=INLINE_FONTSIZE,
                        color=COLORS['text_primary']
                    ),
                    font=dict(
                        family=FONT_FAMILY,
                        size=INLINE_FONTSIZE - 1,
                        color=COLORS['text_primary']
                    )
                ),
                font=dict(
                    family=FONT_FAMILY,
                    size=INLINE_FONTSIZE,
                    color=COLORS['text_primary']
                ),
            )
        )
    )
    _fig.show()
    return (species_count_data,)


@app.cell
def _(
    COLORS,
    FONT_FAMILY,
    INLINE_FONTSIZE,
    get_species_color,
    get_text_color_for_background,
    go,
    np,
    species_count_data,
):

    def bar():
        n_species = len(species_count_data)
        # Calculate angular width for each bar (360 degrees / number of species)
        angular_width = 360 / n_species
    
        # Create polar bar chart
        fig = go.Figure()
    
        # Get unique count values for grid lines (only show values that exist in the data)
        unique_counts = sorted(species_count_data['COUNT'].unique())
        min_count = species_count_data['COUNT'].min()
        max_count = species_count_data['COUNT'].max()
    
        # Transform counts to log scale for visualization
        # Add small epsilon to avoid log(0) issues
        epsilon = 1e-10
        log_counts = [np.log10(count + epsilon) for count in species_count_data['COUNT']]
        log_min = np.log10(min_count + epsilon)
        log_max = np.log10(max_count + epsilon)
        log_unique_counts = [np.log10(c + epsilon) for c in unique_counts]
    
        # Add bars for each species type (using log-transformed values)
        for idx, (_, row) in enumerate(species_count_data.iterrows()):
            stype = row.SPGROUP
            count = row.COUNT
            count_log = log_counts[idx]
            species_color = get_species_color(stype)
            text_color = get_text_color_for_background(species_color)
        
            # Calculate the angle for this bar (center of the angular segment)
            # Start from top (90 degrees) and go clockwise
            theta = 90 - (idx * angular_width + angular_width / 2)
        
            # Add the radial bar (using log-transformed value)
            fig.add_trace(go.Barpolar(
                r=[count_log],
                theta=[theta],
                width=[angular_width * 0.9],  # Slightly smaller than full width for spacing
                marker_color=species_color,
                marker_line_color=species_color,
                marker_line_width=0,
                name=stype,
                hovertemplate=f'<b>{stype}</b><br>Count: {count:,}<extra></extra>',
                hoverlabel=dict(
                    bgcolor=species_color,
                    bordercolor='rgba(0,0,0,0)',
                    font_size=INLINE_FONTSIZE,
                    font_family=FONT_FAMILY,
                    font_color=text_color,
                ),
            ))
    
        # Update layout for polar coordinates with log scale (transformed data)
        fig.update_layout(
            polar=dict(
                radialaxis=dict(
                    visible=True,
                    range=[log_min, log_max * 1.15],  # Add padding for labels
                    showgrid=False,
                    gridcolor=COLORS['white'],  # Subtle grid
                    gridwidth=.8,
                    tickmode='array',
                    tickvals=log_unique_counts,  # Grid lines at actual bar values (log scale)
                    # ticktext=[f'{c:,}' for c in unique_counts],  # Format tick labels with original values
                    ticktext=['' for c in unique_counts],  # Format tick labels with original values
                    tickfont=dict(
                        family=FONT_FAMILY,
                        size=INLINE_FONTSIZE - 2,
                        color=COLORS['text_primary']
                    ),
                    tickcolor=COLORS['text_primary'],
                    showline=False,
                    linecolor=COLORS['text_primary'],
                ),
                angularaxis=dict(
                    visible=True,
                    rotation=90,  # Start from top
                    direction='clockwise',
                    showgrid=False,
                    showline=False,
                    showticklabels=False,
                ),
                bgcolor='rgba(0,0,0,0)',
            ),
            paper_bgcolor=COLORS['background'],
            plot_bgcolor='rgba(0,0,0,0)',
            showlegend=False,
            font=dict(
                family=FONT_FAMILY,
                size=INLINE_FONTSIZE,
                color=COLORS['text_primary']
            ),
            margin=dict(l=0, r=0, t=0, b=0),
        )
    
        # Add text annotations for species types around the perimeter using scatterpolar
        # Position labels outside the chart area
        # annotation_radius_log = log_max * 1.05
    
        # Collect annotation data and add connecting lines
        for idx, (_, row) in enumerate(species_count_data.iterrows()):
            stype = row.SPGROUP
            count_log = log_counts[idx]
            # Calculate angle for annotation (same as bar center)
            theta = 90 - (idx * angular_width + angular_width / 2)
        
            # Add connecting line from end of bar to label (dashed line)
            line_radius_log = count_log * 1.11
            if row.SPGROUP in ['Amphibians', 'Reptiles']:
                line_radius_log = count_log * 1.26
        
            fig.add_trace(go.Scatterpolar(
                r=[count_log, line_radius_log],
                theta=[theta, theta],
                mode='lines',
                line=dict(
                    color=COLORS['white'],#'rgba(0,0,0,0.2)',
                    width=1,
                    dash='dash'
                ),
                showlegend=False,
                hoverinfo='skip',
            ))
        
            # Add a small black dot at the end of the bar
            fig.add_trace(go.Scatterpolar(
                r=[count_log],
                theta=[theta],
                mode='markers',
                marker=dict(
                    size=4,
                    color=COLORS['white'],#'rgba(0,0,0,0.7)',
                    symbol='circle',
                    line=dict(width=0)
                ),
                showlegend=False,
                hoverinfo='skip',
            ))
        
            annotation_text = f"{row.SPGROUP}: {row.COUNT}"
            annotation_radius_log = count_log * 1.14
            if row.SPGROUP in ['Amphibians', 'Reptiles']:
                annotation_radius_log = count_log * 1.3
            
            fig.add_trace(go.Scatterpolar(
                r=[annotation_radius_log],
                theta=[theta],
                mode='text',
                text=annotation_text,
                textfont=dict(
                    family=FONT_FAMILY,
                    size=INLINE_FONTSIZE - 1,
                    color=COLORS['text_primary']
                ),
                showlegend=False,
                hoverinfo='skip',
                )
             )
        fig.show()

    bar()
    return


@app.cell
def _(all_species, csv_pth, pd):
    sites = pd.read_csv(csv_pth / 'NATURA2000SITES.csv')
    species_sites = all_species.groupby(['SITECODE', 'SPGROUP'])['SPECIESNAME'].nunique().reset_index(name='COUNT')
    # print(sites.columns)
    species_count_sites = all_species.merge(sites[['LATITUDE', 'LONGITUDE', 'AREAHA', 'SITECODE']], on='SITECODE', how='left')
    species_count_sites = (species_count_sites
        .groupby(
            ['SPGROUP', 'SITECODE', 'LATITUDE', 'LONGITUDE', 'COUNTRY_CODE']
        )['SPECIESNAME']
            .nunique().reset_index(name='COUNT')
    )
    return (species_count_sites,)


@app.cell
def _(
    COLORS,
    FONT_FAMILY,
    INLINE_FONTSIZE,
    SPECIES_COLORS,
    country_codes,
    create_colorscale,
    go,
    mcolors,
    px,
    species_count_sites,
):
    _species_type = 'Fish'
    species_data = species_count_sites[
        species_count_sites['SPGROUP'] == _species_type
    ].copy()

    species_data['COUNTRY'] = species_data.COUNTRY_CODE.map(country_codes)
    species_data = species_data[species_data.COUNT > 1]
    species_data['hover_text'] = species_data.apply(
        lambda row: f"No. {_species_type.lower()}: {row.COUNT}<br>Country: {row.COUNTRY}", axis=1)

    species_data = species_data.sort_values(by='COUNT')

    def _cluster_colorscale(hex_color):
        """Build a light→full colorscale from a cluster hex color."""
        r, g, b = mcolors.to_rgb(hex_color)
        # Light version, slightly tinted toward background
        bg_r, bg_g, bg_b = mcolors.to_rgb(COLORS["background"])
        light = (
            0.8 * bg_r + 0.2 * r,
            0.8 * bg_g + 0.2 * g,
            0.8 * bg_b + 0.2 * b,
        )
        light_rgba = f"rgba({int(light[0]*255)}, {int(light[1]*255)}, {int(light[2]*255)}, 0.5)"
        full_rgba = f"rgba({int(r*255)}, {int(g*255)}, {int(b*255)}, 1.0)"
        return [
            [0.0, light_rgba],
            [1.0, full_rgba],
        ]

    # Get base color for this species type
    _base_color = SPECIES_COLORS[_species_type]
    _colorscale = create_colorscale(_base_color)

    # Normalize COUNT values for colorscale (0 to 1)
    min_count = species_data['COUNT'].min()
    max_count = species_data['COUNT'].max()

    if max_count == min_count:

        # All counts are the same, use uniform color
        normalized_counts = [0.5] * len(species_data)
    else:
        normalized_counts = (species_data['COUNT'] - min_count) / (max_count - min_count)

    # Create scatter map plot
    _fig = go.Figure()

    light_background, _ = create_colorscale(COLORS['background']) # get the color for the map background
    map_color = light_background[1]

    _fig = px.scatter_geo(
            species_data,
            lat='LATITUDE',
            lon='LONGITUDE',
            color='COUNT',
            hover_name='SITECODE',
            color_continuous_scale=_colorscale,
            opacity=0.7,
            labels={'COUNT': 'Species Count', 'COUNTRY': 'Country'},
            size='COUNT',
            size_max=15,
            custom_data=['hover_text'],
        )

    _fig.update_traces(
        marker=dict(
                # size=10,
                line=dict(width=0.1, color=COLORS['white'])
            ),
        hovertemplate=(
                "<b>%{hovertext}</b><br>" +
                "%{customdata[0]}" +
                "<extra></extra>"
            )
        )

    _fig.update_geos(
        scope="europe",
        visible=False,
        showcountries=False,
        showcoastlines=False,
        showland=True,
        projection_scale=2.50,
        center=dict(lat=55, lon=15),
        landcolor=map_color,
        bgcolor=COLORS["white"],
    )

    # Update layout
    _fig.update_layout(
        mapbox=dict(
            style="open-street-map",  # Free map style
            center=dict(
                lat=species_data['LATITUDE'].mean(),
                lon=species_data['LONGITUDE'].mean()
            ),
            zoom=4  # Adjust zoom level as needed
        ),
        hoverlabel=dict(
            bordercolor=COLORS['white'],
            font_size=12,
            font_family='Noto Sans Mono',
            font_color=COLORS['text_primary'],
            align='left'
        ),
        title=dict(
            text=f"{_species_type} Distribution Across Natura 2000 Sites",
            x=0.5,
            xanchor='center',
            font=dict(family=FONT_FAMILY, size=INLINE_FONTSIZE + 2, color=COLORS['text_primary'])
        ),
        height=600,
        width=900,
        plot_bgcolor=COLORS['background'],
        paper_bgcolor=COLORS['background'],
        font=dict(
            family=FONT_FAMILY,
            size=INLINE_FONTSIZE,
            color=COLORS['text_primary']
        ),
        margin=dict(l=0, r=0, t=50, b=0)
    )

    _fig.show()
    return


@app.cell
def _(
    COLORS,
    INLINE_FONTSIZE,
    SPECIES_COLORS,
    create_colorscale,
    go,
    species_count,
    species_per_country,
):
    _fig = go.Figure()
    species_types = species_count.SPGROUP.unique()

    y_positions = {species: i for i, species in enumerate(species_types)}


    for species_type in species_types:
        data = species_per_country[species_per_country.SPGROUP == species_type]
        if len(data) == 0:
            continue
        y_pos = [y_positions[species_type]] * len(data)

        # Get base color for this species type
        base_color = SPECIES_COLORS[species_type]
        _colorscale = create_colorscale(base_color, 1)

        _fig = _fig.add_trace(go.Scatter(
            x=data.COUNTRY_CODE,
            y=y_pos,
            mode='markers',
            name=species_type,
            marker=dict(
                size=data.COUNT,
                sizemode='diameter',
                sizeref=data.COUNT.max() / 50 if data.COUNT.max() > 0 else 1,
                color=data.COUNT,
                colorscale=_colorscale,
                opacity=1,
                showscale=False,  # Only show colorbar for last trace
                cmin=species_per_country['COUNT'].min(),
                cmax=species_per_country['COUNT'].max(),
                line=dict(width=1, color='white')
            ),
            text=data.COUNTRY_CODE,
            hovertemplate=f'<b>{species_type}</b><br>Country: %{{text}}<br>Count: %{{marker.size}}<extra></extra>'
        ))

    # Update y-axis
    _fig.update_layout(
        yaxis=dict(
            tickmode='array',
            tickvals=list(range(len(species_types))),
            ticktext=species_types,
            title="Species Type",
            zeroline=False,
            gridwidth=.8,
            gridcolor='rgba(252, 255, 247, 0.3)',
        ),
        xaxis=dict(
            title="Country Code", 
            zeroline=False,
            gridcolor='rgba(252, 255, 247, 0.3)',
            gridwidth=.8,
        ),
        plot_bgcolor=COLORS['background'],
        paper_bgcolor=COLORS['background'],
        height=600,
        showlegend=True,
        font_size=INLINE_FONTSIZE,
    )
    _fig.show()
    return


@app.cell
def _():
    import marimo as mo
    return (mo,)


if __name__ == "__main__":
    app.run()
