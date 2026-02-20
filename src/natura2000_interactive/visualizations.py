"""
Visualization functions for Natura 2000 data.

This module contains functions to create various plots and visualizations.
"""

import json

import numpy as np
import plotly.graph_objects as go
import plotly.express as px
import matplotlib.colors as mcolors
from plotly.subplots import make_subplots

from config import SCATTER_SPIRAL_CONFIG, FONT_FAMILY, COLORS, INLINE_FONTSIZE, COUNTRY_CODES_NAMES
from habitat_colors import CLUSTER_COLORS, rgb_to_rgba
from species_colors import SPECIES_COLORS, get_species_color


def create_colorscale(hex_color, alpha=0.5):
    """Build a light→full colorscale from a hex color."""
    r, g, b = mcolors.to_rgb(hex_color)
    # Light version, slightly tinted toward background
    bg_r, bg_g, bg_b = mcolors.to_rgb(COLORS["background"])
    light = (
        0.8 * bg_r + 0.2 * r,
        0.8 * bg_g + 0.2 * g,
        0.8 * bg_b + 0.2 * b,
    )
    light_rgba = f"rgba({int(light[0]*255)}, {int(light[1]*255)}, {int(light[2]*255)}, {alpha})"
    full_rgba = f"rgba({int(r*255)}, {int(g*255)}, {int(b*255)}, 1.0)"
    return [
        [0.0, light_rgba],
        [1.0, full_rgba],
    ]


def get_text_color_for_background(bg_color):
    """
    Determine whether text should be white or dark based on background color brightness.
    
    Parameters
    ----------
    bg_color : str
        Background color in hex format (e.g., '#RRGGBB') or rgba format
        
    Returns
    -------
    str
        COLORS['white'] for dark backgrounds, COLORS['text_primary'] for light backgrounds
    """
    # Handle rgba format
    if bg_color.startswith('rgba'):
        # Extract RGB values from rgba string
        import re
        match = re.search(r'rgba\((\d+),\s*(\d+),\s*(\d+)', bg_color)
        if match:
            r, g, b = int(match.group(1)), int(match.group(2)), int(match.group(3))
        else:
            return COLORS['text_primary']
    else:
        # Handle hex format
        r, g, b = mcolors.to_rgb(bg_color)
        r, g, b = int(r * 255), int(g * 255), int(b * 255)
    
    # Calculate relative luminance (simplified)
    luminance = (0.299 * r + 0.587 * g + 0.114 * b) / 255
    
    # Use white text for dark backgrounds (luminance < 0.5), dark text for light backgrounds
    return COLORS['white'] if luminance < 0.5 else COLORS['text_primary']

def generate_squashed_spiral(N, **kwargs):
    """
    Generate N points on a squashed Archimedean spiral.
    
    Parameters
    ----------
    N : int
        Number of points.
    **kwargs : dict
        Optional parameters to override config defaults:
        - a: Initial radius offset
        - b: Radial growth factor
        - theta_max: Maximum angle (in multiples of pi)
        - scale_x: Horizontal scaling factor
        - scale_y: Vertical scaling factor (squash factor)
        - center: Spiral center (cx, cy)
        
    Returns
    -------
    np.ndarray
        Array of shape (N, 2) with x, y coordinates
    """
    # Use config defaults, but allow override via kwargs
    config = SCATTER_SPIRAL_CONFIG.copy()
    config.update(kwargs)
    
    a = config['a']
    b = config['b']
    theta_max = config['theta_max'] * np.pi  # Convert to radians
    scale_x = config['scale_x']
    scale_y = config['scale_y']
    center = config['center']
    
    cx, cy = center
    theta = np.linspace(0, theta_max, N)
    r = a + b * theta
    x = r * np.cos(theta)
    y = r * np.sin(theta)
    x *= scale_x
    y *= scale_y
    x += cx
    y += cy
    return np.column_stack((x, y))


def create_scatter_plot(scatter_data):
    """
    Create the scatter plot spiral visualization.
    
    Parameters
    ----------
    scatter_data : pandas.DataFrame
        DataFrame containing habitat data with columns:
        - COUNT: Number of sites
        - COVER_KM: Total coverage in km²
        - CLUSTER: Habitat cluster name
        - DESCRIPTION: Habitat description
        
    Returns
    -------
    plotly.graph_objects.Figure
        Scatter plot figure
    """
    xy = generate_squashed_spiral(len(scatter_data['COUNT'].values))
    x = xy[:, 0]
    y = xy[:, 1]
    
    scatter_data_copy = scatter_data.copy()
    scatter_data_copy['x'] = x
    scatter_data_copy['y'] = y
    scatter_data_copy['Count (k)'] = (scatter_data_copy.COUNT/1000).map(
        lambda x: np.round(x, decimals=2)
    )
    scatter_data_copy['Cover (k km2)'] = (scatter_data_copy.COVER_KM/1000).map(
        lambda x: int(x)
    )
    
    fig = px.scatter(
        scatter_data_copy,
        x='x',
        y='y',
        size='COVER_KM',
        color='CLUSTER',
        color_discrete_map=CLUSTER_COLORS,
        hover_name='DESCRIPTION',
        hover_data={
            'x': False, 'y': False, 'COVER_KM': False,
            'DESCRIPTION': False, 'COUNT': False, 
            'Count (k)': True, 'Cover (k km2)': True
        },
        size_max=50,
    )
    
    # Update traces with marker styling and per-trace hoverlabel
    for trace in fig.data:
        cluster_name = trace.name
        if cluster_name in CLUSTER_COLORS:
            cluster_color = CLUSTER_COLORS[cluster_name]
            text_color = get_text_color_for_background(cluster_color)
            trace.update(
                marker=dict(
                    line=dict(
                        width=1.2,
                        color=COLORS['white']
                    ),
                    opacity=0.9,
                ),
                hoverlabel=dict(
                    bgcolor=cluster_color,
                    bordercolor='rgba(0,0,0,0)',
                    font_size=INLINE_FONTSIZE,
                    font_family=FONT_FAMILY,
                    font_color=text_color,
                    align='left'
                )
            )
        else:
            trace.update(
                marker=dict(
                    line=dict(
                        width=1.2,
                        color=COLORS['white']
                    ),
                    opacity=0.9,
                ),
                hoverlabel=dict(
                    bgcolor=COLORS['background'],
                    bordercolor='rgba(0,0,0,0)',
                    font_size=INLINE_FONTSIZE,
                    font_family=FONT_FAMILY,
                    font_color=COLORS['text_primary'],
                    align='left'
                )
            )
    
    fig.update_layout(
        plot_bgcolor=COLORS['background'],
        paper_bgcolor=COLORS['background'],
        height=500,
        width=900,
        margin=dict(l=0, r=0, t=0, b=0),
        showlegend=True,
        font_family=FONT_FAMILY,
        font_color=COLORS['text_primary'],
        xaxis=dict(
            title="",
            showgrid=False,
            showticklabels=False,
            zeroline=False
        ),
        yaxis=dict(
            title="",
            showgrid=False,
            showticklabels=False,
            zeroline=False
        ),
        hovermode='closest',
        hoverdistance=5,
        legend_itemclick="toggleothers",
    )
    
    # Point to a specific bubble (e.g., index 0, or find by description)

    target_idx = scatter_data_copy[scatter_data_copy['DESCRIPTION'] == 'Evergreen woodland'].index[0]
    target_x = scatter_data_copy.loc[target_idx, 'x']
    target_y = scatter_data_copy.loc[target_idx, 'y']

    # fig.add_annotation(
    #     x=target_x,
    #     y=target_y,
    #     text="The larger the bubble,<br>the more area is covered<br>(kilo km2)",
    #     showarrow=True,
    #     arrowhead=2,
    #     arrowsize=1.3,
    #     arrowwidth=1.5,
    #     arrowcolor=rgb_to_rgba(mcolors.to_rgb(COLORS['white']), 0.7),
    #     ax=90,
    #     ay=-25,
    #     bgcolor=rgb_to_rgba(mcolors.to_rgb(COLORS['white']), 0.7),
    #     bordercolor=rgb_to_rgba(mcolors.to_rgb(COLORS['white']), 0.7),
    #     borderwidth=0,
    #     font=dict(size=12, family='Noto Sans Mono', color=COLORS['text_primary']),
    #     xanchor='left',
    #     yanchor='bottom'
    # )

    target_idx = scatter_data_copy[scatter_data_copy['DESCRIPTION'] == 'Broad-leaved deciduous woodland'].index[0]
    target_x = scatter_data_copy.loc[target_idx, 'x']
    target_y = scatter_data_copy.loc[target_idx, 'y']

    # fig.add_annotation(
    #     x=target_x,
    #     y=target_y,
    #     text="Habitats closer to the center are more popular<br>(by count of habitat occurences).",
    #     showarrow=True,
    #     arrowhead=2,
    #     arrowsize=1.3,
    #     arrowwidth=1.5,
    #     arrowcolor=rgb_to_rgba(mcolors.to_rgb(COLORS['white']), 0.7),
    #     ax=120,
    #     ay=-30,
    #     bgcolor=rgb_to_rgba(mcolors.to_rgb(COLORS['white']), 0.7),
    #     bordercolor=rgb_to_rgba(mcolors.to_rgb(COLORS['white']), 0.7),
    #     borderwidth=0,
    #     font=dict(size=12, family='Noto Sans Mono', color=COLORS['text_primary']),
    #     xanchor='left',
    #     yanchor='middle',
    # )

    return fig


def create_sankey_diagram(sankey_data, selected_node_index=None, dim_color=COLORS['background']):
    """
    Create the Sankey diagram visualization with optional node filtering.
    
    Parameters
    ----------
    sankey_data : pandas.DataFrame
        DataFrame containing habitat-country data with columns:
        - CLUSTER: Habitat cluster name
        - DESCRIPTION: Habitat description
        - COUNTRY_CODE: Country code
        - COVER_KM: Coverage in km²
    selected_node_index : int, optional
        If provided, only links connected to this node will be shown.
        Other nodes will be dimmed. If None, all links are shown.
        
    Returns
    -------
    plotly.graph_objects.Figure
        Sankey diagram figure
    """
    def rgb_to_rgba(rgb, alpha=0.6):
        """Convert RGB tuple (0-1) to rgba string (0-255)."""
        r, g, b = rgb
        return f'rgba({int(r * 255)}, {int(g * 255)}, {int(b * 255)}, {alpha})'
    
    def wrap_text(text, max_chars_per_line=30):
        """
        Wrap text by adding newlines at word boundaries.
        
        Parameters
        ----------
        text : str
            Text to wrap
        max_chars_per_line : int
            Maximum characters per line
            
        Returns
        -------
        str
            Wrapped text with newlines
        """
        if len(text) <= max_chars_per_line:
            return text
        
        words = text.split()
        lines = []
        current_line = []
        current_length = 0
        
        for word in words:
            # Check if adding this word would exceed the limit
            if current_length + len(word) + (1 if current_line else 0) > max_chars_per_line:
                if current_line:
                    lines.append(' '.join(current_line))
                    current_line = [word]
                    current_length = len(word)
                else:
                    # Word itself is longer than max_chars_per_line, split it
                    lines.append(word[:max_chars_per_line])
                    current_line = [word[max_chars_per_line:]]
                    current_length = len(word[max_chars_per_line:])
            else:
                current_line.append(word)
                current_length += len(word) + (1 if len(current_line) > 1 else 0)
        
        if current_line:
            lines.append(' '.join(current_line))
        
        return '<br>'.join(lines)
    
    # Get unique nodes
    clusters = sankey_data["CLUSTER"].unique()
    descriptions = sankey_data["DESCRIPTION"].unique()
    countries = sankey_data["COUNTRY_CODE"].unique()
    
    # Map country codes to country names
    country_names = [COUNTRY_CODES_NAMES.get(code, code) for code in countries]

    # ---------------------------------------------------------------------
    # Tier order: DESCRIPTION -> CLUSTER -> COUNTRY
    # ---------------------------------------------------------------------
    node_labels = list(descriptions) + list(clusters) + country_names

    # Create index mapping
    desc_indices = {desc: i for i, desc in enumerate(descriptions)}
    cluster_indices = {
        cluster: len(descriptions) + i for i, cluster in enumerate(clusters)
    }
    country_indices = {
        country: len(descriptions) + len(clusters) + i
        for i, country in enumerate(countries)
    }

    # Create links DESCRIPTION -> CLUSTER
    desc_cluster_links = (
        sankey_data.groupby(["DESCRIPTION", "CLUSTER"])["COVER_KM"]
        .sum()
        .reset_index()
    )
    source_desc_cluster = [
        desc_indices[desc] for desc in desc_cluster_links["DESCRIPTION"]
    ]
    target_desc_cluster = [
        cluster_indices[cluster] for cluster in desc_cluster_links["CLUSTER"]
    ]
    value_desc_cluster = desc_cluster_links["COVER_KM"].tolist()

    # Create links CLUSTER -> COUNTRY_CODE
    cluster_country_links = (
        sankey_data.groupby(["CLUSTER", "COUNTRY_CODE"])["COVER_KM"]
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

    # Combine all links
    source = source_desc_cluster + source_cluster_country
    target = target_desc_cluster + target_cluster_country
    value = value_desc_cluster + value_cluster_country
    
    # Determine which links and nodes are connected if a node is selected
    if selected_node_index is not None:
        # Find all links directly connected to the selected node
        connected_link_indices = set([
            i for i, (s, t) in enumerate(zip(source, target))
            if s == selected_node_index or t == selected_node_index
        ])
        
        # Find all nodes connected to the selected node (directly connected)
        connected_nodes = {selected_node_index}
        for link_idx in connected_link_indices:
            connected_nodes.add(source[link_idx])
            connected_nodes.add(target[link_idx])
        
        # Keep ALL links, but set unconnected ones to 0 (invisible)
        # This preserves node positions
        # filtered_value = []
        # for i, v in enumerate(value):
        #     if i in connected_link_indices:
        #         filtered_value.append(v)  # Keep original value
        #     else:
        #         filtered_value.append(0)  # Hide link by setting value to 0
        # value = filtered_value
    else:
        connected_nodes = set(range(len(node_labels)))  # All nodes are connected
        connected_link_indices = set(range(len(source)))  # All links are connected
    
    # Create node colors and positions
    desc_to_cluster = (
        sankey_data.groupby("DESCRIPTION")["CLUSTER"].first().to_dict()
    )
    node_colors = []
    node_x = []
    node_y = []
    
    for i, label in enumerate(node_labels):
        if i < len(descriptions):
            # First tier: descriptions (colored by their assigned cluster)
            desc_cluster = desc_to_cluster.get(label)
            node_colors.append(CLUSTER_COLORS.get(desc_cluster, "#bbbaba"))
            # Keep nodes away from the very top/bottom to avoid clipping
            node_x.append(0.02)
            node_y.append((i + 0.5) / max(len(descriptions), 1))
        elif i < len(descriptions) + len(clusters):
            # Second tier: clusters
            cluster_name = clusters[i - len(descriptions)]
            node_colors.append(CLUSTER_COLORS.get(cluster_name, "#bbbaba"))
            # node_x.append(0.5)
            cluster_idx = i - len(descriptions)
            if clusters[cluster_idx] == 'Forest/Woodland':
                node_x.append(0.42)
                node_y.append((cluster_idx + 0.4) / max(len(clusters), 1))
            else:
                node_x.append(0.5)
                node_y.append((cluster_idx + 0.5) / max(len(clusters), 1))
        else:
            # Third tier: countries
            node_colors.append(COLORS["details"])
            country_idx = i - len(descriptions) - len(clusters)
            node_y.append((country_idx + 0.5) / max(len(countries), 1))
            # add a different x position for country == 'ES' as it has the largest node lentgh
            if countries[country_idx] == 'ES':
                node_x.append(0.87)
            else:
                node_x.append(0.98)
    
    # Color links by source cluster and dim if filtered
    link_colors = []
    for i, src_idx in enumerate(source):
        # Determine base color
        if src_idx < len(descriptions):
            # Link from description -> cluster (color by description's cluster)
            desc_name = node_labels[src_idx]
            desc_cluster = desc_to_cluster.get(desc_name)
            base_color_hex = CLUSTER_COLORS.get(desc_cluster, "#bbbaba")
            rgb = mcolors.to_rgb(base_color_hex)
            # Dim unconnected links
            if selected_node_index is not None and i not in connected_link_indices:
                link_colors.append(rgb_to_rgba(rgb, alpha=0.05))  # Very dim
            else:
                link_colors.append(rgb_to_rgba(rgb, alpha=0.6))
        else:
            # Link from cluster -> country (color by cluster)
            cluster_name = node_labels[src_idx]
            base_color_hex = CLUSTER_COLORS.get(cluster_name, "#bbbaba")
            rgb = mcolors.to_rgb(base_color_hex)
            # Dim unconnected links
            if selected_node_index is not None and i not in connected_link_indices:
                link_colors.append(rgb_to_rgba(rgb, alpha=0.05))  # Very dim
            else:
                link_colors.append(rgb_to_rgba(rgb, alpha=0.4))
    
    # # Dim nodes that aren't connected to the selected node
    if selected_node_index is not None:
        dimmed_node_colors = []
        for i, color in enumerate(node_colors):
            if i in connected_nodes:
                dimmed_node_colors.append(color)
            else:
                # Convert to rgba and reduce opacity
                if isinstance(color, str) and color.startswith('#'):
                    rgb = mcolors.to_rgb(color)
                    dimmed_node_colors.append(
                        f'rgba({int(rgb[0]*255)}, {int(rgb[1]*255)}, {int(rgb[2]*255)}, 0.2)'
                    )
                else:
                    dimmed_node_colors.append(color)
        node_colors = dimmed_node_colors

    # Hide description labels (too many); show clusters + countries
    node_text_labels = [""] * len(descriptions) + list(clusters) + country_names

    # If a node is selected, show tier-1 (descriptions) as *annotations* instead
    # of Sankey node labels, because Plotly Sankey has a single global font size
    # for all node labels. Annotations let us use a smaller font just for tier-1.
    description_label_annotations = []
    if selected_node_index is not None and selected_node_index < len(descriptions) + len(clusters):
        for i in sorted(connected_nodes):
            if i >= len(descriptions):
                continue

            # label_wrapped = wrap_text(node_labels[i], max_chars_per_line=28)
            node_text_labels[i] = node_labels[i].split('(')[0]
            # label_wrapped = node_labels[i].split('(')[0]

            # # Sankey uses y=0 at the top; paper coordinates use y=0 at the bottom.
            # # We flip y so the annotation sits on the same horizontal band.
            # y_paper = 1 - node_y[i]

            # description_label_annotations.append(
            #     dict(
            #         xref="paper",
            #         yref="paper",
            #         x=node_x[i] + 0.01,
            #         y=y_paper,
            #         text=label_wrapped,
            #         showarrow=False,
            #         xanchor="left",
            #         yanchor="middle",
            #         align="left",
            #         bgcolor="rgba(252, 255, 247, 0.3)",  # subtle white background (COLORS['white'] at 30% opacity)
            #         bordercolor="rgba(252, 255, 247, 0.0)",
            #         borderpad=1,
            #         font=dict(
            #             family=FONT_FAMILY,
            #             size=max(9, INLINE_FONTSIZE - 3),
            #             color=COLORS["text_primary"],
            #         ),
            #     )
            # )
            
    fig = go.Figure(
        data=[
            go.Sankey(
                node=dict(
                    label=node_text_labels,
                    pad=10,
                    thickness=18,
                    line=dict(color="white", width=0),
                    color=node_colors,
                    customdata=node_labels,
                    hovertemplate="<b>%{customdata}</b><br>Cover: %{value} km²<extra></extra>",
                    x=node_x,
                    y=node_y,
                ),
                link=dict(
                    source=source,
                    target=target,
                    value=value,
                    color=link_colors,
                    customdata=node_labels,
                    hovertemplate=(
                        "<b>%{target.customdata}</b><br>"
                        "%{source.customdata}<br>"
                        "Cover: %{value} km²<extra></extra>"
                    ),
                ),
            )
        ]
    )
    
    fig.update_layout(
        plot_bgcolor=COLORS['background'],
        paper_bgcolor=COLORS['background'],
        title="",
        font_size=INLINE_FONTSIZE,
        font_color=COLORS['text_primary'],
        height=800,
        margin=dict(l=0, r=0, t=10, b=30),
        font_family=FONT_FAMILY,
        annotations=description_label_annotations,
        hoverlabel=dict(
            bordercolor='rgba(0,0,0,0)',
            font_size=INLINE_FONTSIZE,
            font_family=FONT_FAMILY,
            # Note: Sankey hoverlabel bgcolor and font_color are set globally
            # Individual node colors are handled by the node color array
            bgcolor=COLORS['background'],
            font_color=COLORS['text_primary'],
        ),
    )
    
    return fig


def create_cluster_choropleth_grid(chloropleth_data, europe_gdf):
    """
    Create a 2x4 grid of small choropleth subplots.

    Each subplot represents one habitat cluster, showing per-country COVER_KM
    using a single-hue colorscale derived from CLUSTER_COLORS.

    Parameters
    ----------
    chloropleth_data : pandas.DataFrame
        DataFrame with at least columns:
        - COUNTRY_CODE
        - CLUSTER
        - COVER_KM
    europe_gdf : geopandas.GeoDataFrame
        GeoDataFrame with geometry and ISO alpha-2 codes in column 'iso_a2'.

    Returns
    -------
    plotly.graph_objects.Figure
        Figure with a 2x4 grid of choropleth maps.
    """
    if chloropleth_data is None or europe_gdf is None:
        # Return empty figure if data is not available
        fig = go.Figure()
        fig.add_annotation(
            text="Choropleth data not available",
            xref="paper", yref="paper",
            x=0.5, y=0.5,
            showarrow=False,
            font=dict(size=16, color=COLORS['text_primary'])
        )
        fig.update_layout(
            height=380,
            width=850,
            plot_bgcolor=COLORS['white'],
            paper_bgcolor=COLORS['white'],
        )
        return fig
    
    # Build geojson once from Europe geometries
    europe_geojson = json.loads(europe_gdf.to_json())

    # Use up to 8 clusters (2x4 grid), prefer explicit CLUSTER_COLORS order
    available_clusters = [
        c for c in CLUSTER_COLORS.keys()
        if c in chloropleth_data["CLUSTER"].unique() and c != "Other"
    ]
    clusters = available_clusters[:8]

    rows, cols = 2, 4
    specs = [[{"type": "choropleth"} for _ in range(cols)] for _ in range(rows)]
    fig = make_subplots(
        rows=rows,
        cols=cols,
        specs=specs,
        subplot_titles=clusters,
        horizontal_spacing=0.0,
        vertical_spacing=0.01,
    )
    
    # Move titles closer to maps
    for ann in fig.layout.annotations:
        ann.y = ann.y - 0.03
        ann.yanchor = "top"
        # Add slightly white, slightly transparent background
        ann.bgcolor = "rgba(252, 255, 247, 0.7)"  # Using COLORS['white'] with 70% opacity
        ann.bordercolor = "rgba(252, 255, 247, 0.0)"  # No border
        ann.borderpad = 3  # Padding around text

    for idx, cluster in enumerate(clusters):
        r = idx // cols + 1
        c = idx % cols + 1

        df_cluster = chloropleth_data[chloropleth_data["CLUSTER"] == cluster].copy()
        if df_cluster.empty:
            continue

        # Normalize coverage within each cluster (proportion of cluster's total coverage)
        df_cluster = df_cluster.assign(
            COVER_NORM=lambda x: x.COVER_KM / df_cluster.COVER_KM.sum()
        )

        base_hex = CLUSTER_COLORS.get(cluster, "#bbbaba")
        colorscale = create_colorscale(base_hex)
        
        # Map country codes to country names for hover
        df_cluster = df_cluster.copy()
        df_cluster['COUNTRY'] = df_cluster['COUNTRY_CODE'].map(COUNTRY_CODES_NAMES)
        df_cluster['COUNTRY'] = df_cluster['COUNTRY'].fillna(df_cluster['COUNTRY_CODE'])
        
        # Prepare customdata as numpy array for proper formatting
        customdata_array = np.column_stack([
            df_cluster['COUNTRY'].values,
            (df_cluster['COVER_KM'] / 1000).values
        ])

        fig.add_trace(
            go.Choropleth(
                locations=df_cluster["COUNTRY_CODE"],
                z=df_cluster["COVER_NORM"],
                geojson=europe_geojson,
                featureidkey="properties.iso_a2_eh",
                colorscale=colorscale,
                marker_line_width=0.4,
                marker_line_color=COLORS["white"],
                showscale=False,
                # Keep absolute values and country name in customdata for hover
                customdata=customdata_array,
                hovertemplate="<b>%{customdata[0]}</b><br>"
                "Cover: %{customdata[1]:.0f}k km²<br>"
                "Proportion: %{z:.1%}<extra></extra>",
                hoverlabel=dict(
                    bgcolor=base_hex,
                    bordercolor='rgba(0,0,0,0)',
                    font_size=INLINE_FONTSIZE,
                    font_family=FONT_FAMILY,
                    font_color=get_text_color_for_background(base_hex),
                ),
            ),
            row=r,
            col=c,
        )

    # Minimal map styling for all geos
    fig.update_geos(
        scope="europe",
        visible=False,
        showcountries=False,
        showcoastlines=False,
        showland=True,
        projection_scale=2.30,
        center=dict(lat=50, lon=10),
        landcolor=COLORS["white"],
        bgcolor=COLORS["white"],
    )

    fig.update_layout(
        height=400,
        width=850,
        plot_bgcolor=COLORS["background"],
        paper_bgcolor=COLORS["background"],
        font=dict(
            family=FONT_FAMILY,
            size=INLINE_FONTSIZE,
            color=COLORS["text_primary"],
        ),
        margin=dict(l=0, r=0, t=0, b=0),
    )

    return fig


def create_species_pictogram(species_count_data):
    """
    Create a pictogram-style visualization showing species counts by type.
    Each marker represents 100 species, arranged in a spiral pattern (polar coordinates).
    
    Parameters
    ----------
    species_count_data : pandas.DataFrame
        DataFrame with columns: SPGROUP, COUNT
        Should be sorted by count in descending order
        
    Returns
    -------
    plotly.graph_objects.Figure
        Pictogram figure
    """
    # Prepare data
    species_count_data = species_count_data.copy()
    species_count_data['COUNT_HUNDREDS'] = species_count_data['COUNT'] / 100
    
    # Spiral parameters
    initial_radius = 0.5  # Starting radius at the center
    angular_spacing = 0.1  # Angular increment in radians (controls spacing along spiral)
    radius_growth_rate = 0.09  # How much radius increases per angular step
    
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
            x = current_radius * np.cos(current_angle)
            y = current_radius * np.sin(current_angle)
            
            _x.append(x)
            _y.append(y)
            
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
                    'line': dict(width=0), 
                    'size': 10, 
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
    
    fig = go.Figure(
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
    
    return fig


def create_species_per_country_scatter(species_per_country, species_count_data):
    """
    Create a scatter plot showing species distribution across countries.
    
    Each species type is shown on a separate row (y-axis), with countries on the x-axis.
    Marker size and color encode the species count per country.
    
    Parameters
    ----------
    species_per_country : pandas.DataFrame
        DataFrame with columns: COUNTRY_CODE, SPGROUP, COUNT
    species_count_data : pandas.DataFrame
        DataFrame with columns: SPGROUP, COUNT (used to get species types order)
        
    Returns
    -------
    plotly.graph_objects.Figure
        Scatter plot figure
    """
    fig = go.Figure()
    
    # Get species types in the order they appear in species_count_data
    species_types = species_count_data.SPGROUP.unique()
    species_per_country['COUNTRY'] = species_per_country.apply(lambda x: COUNTRY_CODES_NAMES[x.COUNTRY_CODE], axis=1)
    # Create y-position mapping for each species type
    y_positions = {species: i for i, species in enumerate(species_types)}
    
    # # Get global min/max for consistent color scaling
    # global_min = species_per_country['COUNT'].min()
    # global_max = species_per_country['COUNT'].max()
    
    # Add a trace for each species type
    for species_type in species_types:
        data = species_per_country[species_per_country.SPGROUP == species_type]
        data = data.sort_values(by='COUNT', ascending=False)

        if len(data) == 0:
            continue

        local_min = data.COUNT.min()
        local_max = data.COUNT.max()
        
        y_pos = [y_positions[species_type]] * len(data)
        
        # Get base color for this species type
        base_color = get_species_color(species_type)
        colorscale = create_colorscale(base_color, alpha=1.0)
        text_color = get_text_color_for_background(base_color)
        
        # Calculate size reference (max size = 50 pixels)
        sizeref = data.COUNT.max() / 50 if data.COUNT.max() > 0 else 1
        
        fig.add_trace(go.Scatter(
            x=data.COUNTRY_CODE,
            y=y_pos,
            mode='markers',
            name=species_type,
            marker=dict(
                size=data.COUNT,
                sizemode='diameter',
                sizeref=sizeref,
                color=data.COUNT,
                # color=base_color,
                colorscale=colorscale,
                opacity=1,
                showscale=False,
                cmin=local_min,
                cmax=local_max,
                line=dict(width=1.3, color=COLORS['white'])
            ),
            text=data.COUNTRY,
            hovertemplate=(
                f'<b>{species_type}</b><br>' +
                'Country: %{text}<br>' +
                'Count: %{marker.size}<extra></extra>'
            ),
            hoverlabel=dict(
                bgcolor=base_color,
                bordercolor='rgba(0,0,0,0)',
                font_size=INLINE_FONTSIZE,
                font_family=FONT_FAMILY,
                font_color=text_color,
            ),
        ))
    
    # Update layout
    fig.update_layout(
        yaxis=dict(
            tickmode='array',
            tickvals=list(range(len(species_types))),
            ticktext=species_types,
            title="Species Type",
            zeroline=False,
            gridwidth=0.8,
            gridcolor='rgba(252, 255, 247, 0.3)',
        ),
        xaxis=dict(
            title="Country",
            zeroline=False,
            gridcolor='rgba(252, 255, 247, 0.3)',
            gridwidth=0.8,        ),
        plot_bgcolor=COLORS['background'],
        paper_bgcolor=COLORS['background'],
        height=600,
        width=900,
        showlegend=True,
        font=dict(family=FONT_FAMILY, size=INLINE_FONTSIZE, color=COLORS['text_primary']),
        legend=dict(
            font=dict(family=FONT_FAMILY, size=INLINE_FONTSIZE - 2, color=COLORS['text_primary']),
            bgcolor=COLORS['background'],
            bordercolor=COLORS['details'],
            borderwidth=0
        ),
        margin=dict(l=60, r=20, t=20, b=60)
    )
    
    return fig


def create_species_scatter_map(species_count_sites, species_type, europe_gdf):
    """
    Create a scatter map plot showing species distribution across Natura 2000 sites
    for a specific species type, with Europe map background.
    
    Parameters
    ----------
    species_count_sites : pandas.DataFrame
        DataFrame with columns: SPGROUP, SITECODE, LATITUDE, LONGITUDE, COUNT
        Contains species counts per site per species group
    species_type : str
        The species group to visualize (e.g., 'Birds', 'Plants', 'Mammals', etc.)
    europe_gdf : geopandas.GeoDataFrame
        GeoDataFrame with Europe country geometries (same as used in choropleth)
        
    Returns
    -------
    plotly.graph_objects.Figure
        Scatter map figure with markers colored by species count intensity
    """
    # Filter data for the specified species type
    species_data = species_count_sites[
        species_count_sites['SPGROUP'] == species_type
    ].copy()
    
    if species_data.empty:
        # Return empty figure with message
        print(f'Warning: No data available for species type: {species_type}, switching to default: Fish data')
        species_data = species_count_sites[
            species_count_sites['SPGROUP'] == 'Fish'
        ].copy()
    
    # Get base color for this species type
    base_color = get_species_color(species_type)
    
    colorscale = create_colorscale(base_color)
    
    # Create hover text
    species_data['COUNTRY'] = species_data.COUNTRY_CODE.map(COUNTRY_CODES_NAMES)

    species_data['hover_text'] = species_data.apply(
        lambda row: f"No. {species_type.lower()}: {row.COUNT}<br>Country: {row.COUNTRY}", axis=1)
    
    light_background, _ = create_colorscale(COLORS['background']) # get the color for the map background
    map_color = light_background[1]

    # Use plotly express scatter_geo with both size and color encoding
    fig = px.scatter_geo(
        species_data,
        lat='LATITUDE',
        lon='LONGITUDE',
        color='COUNT',
        size='COUNT',
        hover_name='SITECODE',
        custom_data=['hover_text'],
        color_continuous_scale=colorscale,
        size_max=16,  # Maximum marker size
        opacity=0.7,
        labels={'COUNT': 'Species Count'},
    )
    
    # Update marker styling and add custom hover template
    text_color = get_text_color_for_background(base_color)
    fig.update_traces(
        marker=dict(
            line=dict(width=0.1, color=COLORS['white'])
        ),
        hovertemplate=(
            "<b>%{hovertext}</b><br>" +
            "%{customdata[0]}" +
            "<extra></extra>"
        ),
        hoverlabel=dict(
            bgcolor=base_color,
            bordercolor='rgba(0,0,0,0)',
            font_size=INLINE_FONTSIZE,
            font_family=FONT_FAMILY,
            font_color=text_color,
        ),
    )
    
    # Update layout with same geo settings as choropleth, but with visible borders
    fig.update_geos(
        scope="europe",
        visible=False,
        showcountries=True,
        showcoastlines=True,
        showland=True,
        projection_scale=2,
        center=dict[str, int](lat=50, lon=15),
        landcolor=map_color,
        bgcolor=COLORS["white"],
        countrycolor=COLORS['white'],
        coastlinecolor=map_color,
        # lataxis_range=[35, 72],
        # lonaxis_range=[-12, 42],
    )
    
    # Update layout styling
    fig.update_layout(
        title=dict(
            # text=f"{species_type} Distribution Across Natura 2000 Sites",
            x=0.5,
            xanchor='center',
            font=dict(family=FONT_FAMILY, size=INLINE_FONTSIZE + 2, color=COLORS['text_primary'])
        ),
        height=550,
        width=800,
        plot_bgcolor=COLORS['background'],
        paper_bgcolor=COLORS['background'],
        font=dict(
            family=FONT_FAMILY,
            size=INLINE_FONTSIZE,
            color=COLORS['text_primary']
        ),
        margin=dict(l=0, r=0, t=50, b=0),
        coloraxis_colorbar=dict(
            title=dict(
                text="Species Count",
                font=dict(family=FONT_FAMILY, size=INLINE_FONTSIZE, color=COLORS['text_primary'])
            ),
            bgcolor=COLORS['background'],
            bordercolor=COLORS['details'],
            borderwidth=0
        )
    )
    
    return fig
