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

    from habitat_colors import (
        assign_cluster,
        assign_cluster_color,
        CLUSTER_COLORS,
    )
    return (
        CLUSTER_COLORS,
        Path,
        assign_cluster,
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
    return habitatclass, habitats


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
def _(habitat_country_cover):
    cluster_country_links = (
        habitat_country_cover.groupby(["CLUSTER", "COUNTRY_CODE"])["COVER_KM"]
        .sum()
        .reset_index()
    )
    cluster_country_links
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
    return (europe,)


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
def _(choropleth_data, europe, px):
    _cluster_name = 'Agricultural'
    data = choropleth_data[choropleth_data.CLUSTER==_cluster_name]
    _fig = px.choropleth(
        data,
        geojson=europe,
        featureidkey="properties.iso_a2_eh",
        locations='COUNTRY_CODE',
        color='COVER_KM',
        scope="europe",          # limit to Europe
        fitbounds="locations",
    )
    _fig.show()
    return


@app.cell
def _(choropleth_data):
    choropleth_data[choropleth_data["CLUSTER"] == 'Other']
    return


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
    return


@app.cell
def _():
    import marimo as mo
    return (mo,)


if __name__ == "__main__":
    app.run()
