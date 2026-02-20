"""
Natura 2000 Interactive Visualization - Storytelling Web App

A minimal, flowing data visualization web app with scroll-triggered animations
and text annotations for storytelling.
"""

import dash
from dash import dcc, html, Input, Output
import dash_bootstrap_components as dbc
import plotly.graph_objects as go

from data_loader import (
    load_natura2000_data, 
    prepare_scatter_data, 
    prepare_sankey_data, 
    prepare_choropleth_data,
    load_europe_geodataframe,
    load_species_data,
    prepare_species_count_data,
    prepare_species_scatter_map_data,
    prepare_species_per_country_data
)
from visualizations import (
    create_scatter_plot, 
    create_sankey_diagram, 
    create_cluster_choropleth_grid,
    create_species_pictogram,
    create_species_per_country_scatter,
    create_species_scatter_map
)
from config import APP_PORT, APP_DEBUG, FONT_FAMILY, COLORS


# Initialize app with minimal Bootstrap theme
app = dash.Dash(
    __name__,
    external_stylesheets=[dbc.themes.BOOTSTRAP],
    suppress_callback_exceptions=True
)

# Load data
print("Loading Natura 2000 data...")
data = load_natura2000_data()
habitats = data['habitats']
habitatclass = data['habitatclass']

# Prepare visualization data
scatter_data = prepare_scatter_data(habitats, habitatclass)
sankey_data = prepare_sankey_data(habitats)

# Load Europe GeoDataFrame and prepare choropleth data
print("Loading Europe GeoDataFrame...")
try:
    europe_gdf = load_europe_geodataframe()
    choropleth_data = prepare_choropleth_data(habitats, europe_gdf)
except Exception as e:
    print(f"Warning: Could not load Europe GeoDataFrame: {e}")
    print("Choropleth visualization will be skipped.")
    europe_gdf = None
    choropleth_data = None

# Load species data
print("Loading species data...")
try:
    species_data = load_species_data()
    species_count_data = prepare_species_count_data(species_data)
    species_count_sites = prepare_species_scatter_map_data(species_data)
    species_per_country = prepare_species_per_country_data(species_data)
except Exception as e:
    print(f"Warning: Could not load species data: {e}")
    print("Species visualization will be skipped.")
    species_count_data = None
    species_count_sites = None
    species_per_country = None


# ============================================================================
# Custom CSS for Storytelling Style
# ============================================================================

# Format CSS with config values
font_family_css = FONT_FAMILY.replace(' ', '+')
app.index_string = '''
<!DOCTYPE html>
<html>
    <head>
        {%metas%}
        <title>Natura 2000 - Interactive Exploration</title>
        {%favicon%}
        {%css%}
        <style>
            @import url('https://fonts.googleapis.com/css2?family=''' + font_family_css + ''':wght@300;400;500;600&display=swap');
            
            * {
                margin: 0;
                padding: 0;
                box-sizing: border-box;
            }
            
            body {
                font-family: ''' + FONT_FAMILY + ''', monospace;
                background:  ''' + COLORS["background"] + ''';
                color: ''' + COLORS['text_primary'] + ''';
                line-height: 1.6;
                overflow-x: hidden;
            }
            
            .story-section {
                min-height: 100vh;
                display: flex;
                align-items: center;
                justify-content: center;
                padding: 4rem 2rem;
            }
            
            .story-container {
                max-width: 1200px;
                width: 100%;
                margin: 0 auto;
            }
            
            .story-text {
                font-size: 1.1rem;
                line-height: 1.8;
                margin-bottom: 3rem;
                color: ''' + COLORS['text_primary'] + ''';
                max-width: 800px;
                margin-left: auto;
                margin-right: auto;
            }
            
            .story-text h1 {
                font-size: 2.5rem;
                font-weight: 600;
                margin-bottom: 1.5rem;
                color: ''' + COLORS['text_primary'] + ''';
            }
            
            .story-text h2 {
                font-size: 1.8rem;
                font-weight: 500;
                margin-bottom: 1rem;
                color: ''' + COLORS['text_primary'] + ''';
            }
            
            .story-text p {
                margin-bottom: 1.5rem;
            }
            
            .plot-container {
                position: relative;
                display: flex;
                align-items: center;
                gap: 3rem;
                margin: 4rem 0;
            }
            
            .plot-container.reverse {
                flex-direction: row-reverse;
            }
            
            .plot-wrapper {
                flex: 1;
                position: relative;
            }
            
            .graph-annotation {
                position: absolute;
                z-index: 10;
                background: ''' + COLORS['white'] + ''';
                background: rgba(252, 255, 247, 0.7);
                padding: 0.75rem 1rem;
                border-radius: 4px;
                box-shadow: 0 2px 8px rgba(0,0,0,0.15);
                font-size: 0.75rem;
                color: ''' + COLORS['text_primary'] + ''';
                max-width: 200px;
                line-height: 1.3;
                border-left: 2px solid ''' + COLORS['details'] + ''';
                backdrop-filter: blur(4px);
            }
            
            .graph-annotation::before {
                content: '';
                position: absolute;
                width: 0;
                height: 0;
                border-style: solid;
            }
            
            /* Arrow pointing right */
            .graph-annotation.arrow-right::before {
                left: -8px;
                top: 50%;
                transform: translateY(-50%);
                border-width: 6px 8px 6px 0;
                border-color: transparent ''' + COLORS['details'] + ''' transparent transparent;
            }
            
            /* Arrow pointing left */
            .graph-annotation.arrow-left::before {
                right: -8px;
                top: 50%;
                transform: translateY(-50%);
                border-width: 6px 0 6px 8px;
                border-color: transparent transparent transparent ''' + COLORS['details'] + ''';
            }
            
            /* Arrow pointing up */
            .graph-annotation.arrow-up::before {
                bottom: -8px;
                left: 50%;
                transform: translateX(-50%);
                border-width: 8px 6px 0 6px;
                border-color: ''' + COLORS['details'] + ''' transparent transparent transparent;
            }
            
            /* Arrow pointing down */
            .graph-annotation.arrow-down::before {
                top: -8px;
                left: 50%;
                transform: translateX(-50%);
                border-width: 0 6px 8px 6px;
                border-color: transparent transparent ''' + COLORS['details'] + ''' transparent;
            }
            
            /* Connecting line (optional, for longer distances) */
            .graph-annotation-line {
                position: absolute;
                background: ''' + COLORS['details'] + ''';
                z-index: 9;
            }
            
            .graph-annotation-line.horizontal {
                height: 2px;
            }
            
            .graph-annotation-line.vertical {
                width: 2px;
            }
            
            .annotation {
                flex: 0 0 300px;
                padding: 1.5rem;
                background: ''' + COLORS['white'] + ''';
                border-left: 3px solid ''' + COLORS['details'] + ''';
                box-shadow: 0 2px 8px rgba(0,0,0,0.1);
                position: relative;
            }
            
            .plot-container.reverse .annotation {
                border-left: none;
                border-right: 3px solid ''' + COLORS['details'] + ''';
            }
            
            .annotation::before {
                content: '';
                position: absolute;
                width: 2px;
                height: 60px;
                background: ''' + COLORS['details'] + ''';
                left: -3px;
                top: 50%;
                transform: translateY(-50%) rotate(-45deg);
                transform-origin: center;
            }
            
            .plot-container.reverse .annotation::before {
                left: auto;
                right: -3px;
                transform: translateY(-50%) rotate(45deg);
            }
            
            .annotation h3 {
                font-size: 1.2rem;
                font-weight: 500;
                margin-bottom: 0.5rem;
                color: ''' + COLORS['text_primary'] + ''';
            }
            
            .annotation p {
                font-size: 0.95rem;
                color: ''' + COLORS['text_primary'] + ''';
                opacity: 0.8;
                line-height: 1.6;
            }
            
            @media (max-width: 968px) {
                .plot-container {
                    flex-direction: column;
                }
                
                .plot-container.reverse {
                    flex-direction: column;
                }
                
                .annotation {
                    flex: 1;
                    border-left: 3px solid ''' + COLORS['details'] + ''';
                    border-right: none;
                }
                
                .plot-container.reverse .annotation {
                    border-left: 3px solid ''' + COLORS['details'] + ''';
                    border-right: none;
                }
                
                .annotation::before {
                    display: none;
                }
            }
            
            /* Modern Dropdown Styling - Simple and targeted */
            .dash-dropdown-wrapper {
                border: none !important;
            }
            
            .dash-dropdown {
                border: none !important;
                border-radius: 8px !important;
                box-shadow: 0 2px 4px rgba(0,0,0,0.1) !important;
                background-color: ''' + COLORS['white'] + ''' !important;
            }
            
            .dash-dropdown:hover {
                box-shadow: 0 4px 8px rgba(0,0,0,0.15) !important;
            }
            
            .dash-dropdown:focus,
            .dash-dropdown:focus-within {
                box-shadow: 0 4px 12px rgba(0,0,0,0.2) !important;
                outline: none !important;
            }
            
            .dash-dropdown-grid-container {
                border: none !important;
            }
        </style>
    </head>
    <body>
        {%app_entry%}
        <footer>
            {%config%}
            {%scripts%}
            {%renderer%}
        </footer>
        <script>
            // Suppress source map warnings
            if (window.console && console.warn) {
                const originalWarn = console.warn;
                console.warn = function(...args) {
                    if (args[0] && typeof args[0] === 'string' && args[0].includes('Source map')) {
                        return; // Suppress source map warnings
                    }
                    originalWarn.apply(console, args);
                };
            }
        </script>
    </body>
</html>
'''


# ============================================================================
# App Layout - Storytelling Structure
# ============================================================================

app.layout = html.Div([
    # Title Section
    html.Div([
        html.Div([
            html.Div([
                html.H1("Natura 2000", className="text-center"),
                html.P(
                    "Exploring Europe's protected habitats through data visualization",
                    className="text-center",
                    style={'fontSize': '1.2rem', 'color': COLORS['details'], 'marginTop': '1rem'}
                )
            ], className="story-text")
        ], className="story-container")
    ], className="story-section", style={'minHeight': '60vh'}),
    
    # Introduction Text
    html.Div([
        html.Div([
            html.Div([
                html.H2("Understanding Habitat Distribution", className="text-center",),
                html.P([
                    "The Natura 2000 network is the largest coordinated network of protected areas in the world, ",
                    "covering over 18% of the EU's land area and more than 8% of its marine territory. ",
                    "This visualization explores the diversity and distribution of habitats across Europe."
                ], className="text-center",)
            ], className="story-text")
        ], className="story-container")
    ], className="story-section"),
    
    # Scatter Plot Section
    html.Div([
        html.Div([
            html.Div([
                html.Div([
                    html.Div([
                        html.H3("Habitat Diversity", className="text-center",),
                        html.P([
                            "Each point represents a unique habitat type, positioned in a spiral pattern. ",
                            "The size of each circle reflects the total area covered, while colors indicate ",
                            "the habitat cluster. Hover to explore individual habitats."
                        ], className="text-center",)
                    ], className="annotation"),
                    html.Div([
                        dcc.Graph(
                            id='scatter-plot',
                            figure=create_scatter_plot(scatter_data),
                            config={'displayModeBar': False}
                        ),
                        # Example annotation - you can add more or remove this
                        html.Div(
                            "Largest habitats by area",
                            className="graph-annotation arrow-right",
                            style={'top': '45%', 'right': '10%'}
                        ),
                    ], className="plot-wrapper", style={'position': 'relative'})
                ], className="plot-container")
            ], className="story-container")
        ], className="story-container")
    ], className="story-section"),
    
    # Middle Text Section
    html.Div([
        html.Div([
            html.Div([
                html.H2("From Clusters to Countries", className="text-center",),
                html.P([
                    "The following Sankey diagram shows how habitat clusters break down into specific ",
                    "habitat types, and how these are distributed across European countries. ",
                    "The flow width represents the total area covered."
                ], className="text-center",)
            ], className="story-text")
        ], className="story-container")
    ], className="story-section"),
    
    # Sankey Diagram Section
    html.Div([
        html.Div([
            html.Div([
                html.Div([
                    html.Div([
                        dcc.Store(id='sankey-selected-node', data=None),  # Store selected node index
                        dcc.Graph(
                            id='sankey-plot',
                            figure=create_sankey_diagram(sankey_data),
                            config={'displayModeBar': False}
                        ),
                        # Example annotation - you can add more or remove this
                        # html.Div(
                        #     "Habitat clusters",
                        #     className="graph-annotation arrow-left",
                        #     style={'top': '15%', 'right': '5%'}
                        # ),
                    ], className="plot-wrapper", style={'position': 'relative'}),
                    html.Div([
                        html.H3("Geographic Distribution", className="text-center",),
                        html.P([
                            "This flow diagram reveals the geographic patterns of habitat distribution. ",
                            "Notice how certain habitat clusters are concentrated in specific regions, ",
                            "reflecting Europe's diverse ecosystems from Mediterranean coasts to Nordic forests."
                        ], className="text-center",)
                    ], className="annotation")
                ], className="plot-container reverse")
            ], className="story-container")
        ], className="story-container")
    ], className="story-section"),
    
    # Middle Text Section - Transition to Country View
    html.Div([
        html.Div([
            html.Div([
                html.H2("Country-Specific Distribution", className="text-center",),
                html.P([
                    "Zooming into individual countries reveals how each habitat cluster is distributed ",
                    "across Europe. Each map shows the total coverage of one cluster type, with darker ",
                    "shades indicating greater coverage."
                ], className="text-center",)
            ], className="story-text")
        ], className="story-container")
    ], className="story-section"),
    
    # Choropleth Grid Section (only if data is available)
] + ([
    html.Div([
        html.Div([
            html.Div([
                dcc.Graph(
                    id='choropleth-grid',
                    figure=create_cluster_choropleth_grid(choropleth_data, europe_gdf),
                    config={'displayModeBar': False}
                )
            ], className="story-container")
        ], className="story-container")
    ], className="story-section")
] if choropleth_data is not None and europe_gdf is not None else []) + [
    
    # Middle Text Section - Transition to Species View
    html.Div([
        html.Div([
            html.Div([
                html.H2("Protected Species", className="text-center",),
                html.P([
                    "Beyond habitats, the Natura 2000 network protects thousands of species across Europe. ",
                    "The following visualization shows the diversity of protected species by type, ",
                    "with each marker representing 100 species."
                ], className="text-center",)
            ], className="story-text")
        ], className="story-container")
    ], className="story-section"),
    
    # Species Pictogram Section (only if data is available)
] + ([
    html.Div([
        html.Div([
            html.Div([
                html.Div([
                    html.Div([
                        html.H3("Species Diversity", className="text-center",),
                        html.P([
                            "Each circle represents 100 species, arranged in a grid. ",
                            "Colors distinguish different species groups, from plants and invertebrates ",
                            "to birds, mammals, and more. Explore the legend to see the count for each group."
                        ], className="text-center",)
                    ], className="annotation"),
                    html.Div([
                        dcc.Graph(
                            id='species-pictogram',
                            figure=create_species_pictogram(species_count_data),
                            config={'displayModeBar': False}
                        ),
                    ], className="plot-wrapper", style={'position': 'relative'})
                ], className="plot-container")
            ], className="story-container")
        ], className="story-container")
    ], className="story-section")
] if species_count_data is not None else []) + [
    
    # Middle Text Section - Transition to Species by Country
    html.Div([
        html.Div([
            html.Div([
                html.H2("Species Distribution by Country", className="text-center",),
                html.P([
                    "The following visualization shows how different species types are distributed across European countries. ",
                    "Each row represents a species type, and each marker shows the number of species found in that country. ",
                    "Marker size and color intensity both encode the species count."
                ], className="text-center",)
            ], className="story-text")
        ], className="story-container")
    ], className="story-section") if species_per_country is not None and species_count_data is not None else [],
    
    # Species per Country Scatter Plot Section (only if data is available)
] + ([
    html.Div([
        html.Div([
            html.Div([
                html.Div([
                    html.Div([
                        dcc.Graph(
                            id='species-per-country-scatter',
                            figure=create_species_per_country_scatter(species_per_country, species_count_data),
                            config={'displayModeBar': False}
                        ),
                    ], className="plot-wrapper", style={'position': 'relative'})
                ], className="plot-container", style={'flexDirection': 'column'})
            ], className="story-container")
        ], className="story-container")
    ], className="story-section")
] if species_per_country is not None and species_count_data is not None else []) + [
    
    # Middle Text Section - Transition to Species Distribution Map
    html.Div([
        html.Div([
            html.Div([
                html.H2("Species Distribution Across Sites", className="text-center",),
                html.P([
                    "Explore the geographic distribution of different species types across Natura 2000 sites. ",
                    "Select a species type from the dropdown to see where these species are found. ",
                    "Marker size and color intensity both represent the number of species at each site."
                ], className="text-center",)
            ], className="story-text")
        ], className="story-container")
    ], className="story-section") if species_count_sites is not None and europe_gdf is not None else [],
    
    # Species Scatter Map Section (only if data is available)
] + ([
    html.Div([
        html.Div([
            html.Div([
                html.Div([
                    html.Div([
                        html.Label("Select Species Type:", style={'fontSize': '1rem', 'color': COLORS['text_primary'], 'marginBottom': '0.5rem'}),
                        html.Div([
                            dcc.Dropdown(
                                id='species-type-dropdown',
                                clearable=False,
                                searchable=False,
                                options=[
                                    {'label': stype, 'value': stype} 
                                    for stype in sorted(species_count_sites['SPGROUP'].unique())
                                ],
                                value=sorted(species_count_sites['SPGROUP'].unique())[0] if len(species_count_sites['SPGROUP'].unique()) > 0 else None,
                                style={
                                    'backgroundColor': COLORS['white'],
                                    'color': COLORS['text_primary'],
                                    'fontFamily': FONT_FAMILY
                                }
                            ),
                        ], className='modern-dropdown-wrapper'),
                    ], style={
                        'maxWidth': '200px',
                        'marginTop': 'auto',      # Vertical positioning (top margin) - use px/rem values like '20px', '1rem'
                        'marginBottom': '1rem', # Vertical positioning (bottom margin)
                        'marginLeft': 'auto',   # Horizontal positioning (left margin - 'auto' centers, or use px/% values like '20px', '10%')
                        'marginRight': 'auto'  # Horizontal positioning (right margin - 'auto' centers, or use px/% values)
                    }),
                    html.Div([
                        dcc.Graph(
                            id='species-scatter-map',
                            config={'displayModeBar': False}
                        ),
                    ], className="plot-wrapper", style={'position': 'relative'})
                ], className="plot-container", style={'flexDirection': 'column'})
            ], className="story-container")
        ], className="story-container")
    ], className="story-section")
] if species_count_sites is not None and europe_gdf is not None else []) + [
    
    # Closing Text
    html.Div([
        html.Div([
            html.Div([
                html.H2("Exploring Further", className="text-center",),
                html.P([
                    "These visualizations provide just a glimpse into the rich biodiversity data ",
                    "available through the Natura 2000 network. Each habitat tells a story of ",
                    "conservation efforts and ecological importance across Europe."
                ], className="text-center",)
            ], className="story-text")
        ], className="story-container")
    ], className="story-section", style={'minHeight': '40vh'}),
    
], style={'padding': 0})


# ============================================================================
# Callbacks for Interactive Features
# ============================================================================

@app.callback(
    [Output('sankey-plot', 'figure'),
     Output('sankey-selected-node', 'data')],
    [Input('sankey-plot', 'clickData')],
    [dash.dependencies.State('sankey-selected-node', 'data')],
    prevent_initial_call=True
)
def update_sankey_on_click(clickData, current_selected_node):
    """
    Handle node clicks on the Sankey diagram.
    When a node is clicked, filter links to show only those connected to it.
    Click the same node again to reset (show all links).
    """
    if clickData is None:
        # No click data, return original figure
        return create_sankey_diagram(sankey_data), None
    
    # Extract node index from click data
    # Plotly Sankey clickData structure: {'points': [{'pointNumber': <node_index>, ...}]}
    point_number = clickData.get('points', [{}])[0].get('pointNumber')
    
    if point_number is None:
        return create_sankey_diagram(sankey_data), None
    
    # Toggle: if clicking the same node, reset to show all links
    if current_selected_node == point_number:
        # Reset: show all links
        return create_sankey_diagram(sankey_data), None
    else:
        # Filter: show only links connected to clicked node
        filtered_fig = create_sankey_diagram(sankey_data, selected_node_index=point_number)
        return filtered_fig, point_number


@app.callback(
    Output('species-scatter-map', 'figure'),
    [Input('species-type-dropdown', 'value')],
    prevent_initial_call=False
)
def update_species_scatter_map(selected_species_type):
    """
    Update the species scatter map based on the selected species type.
    """
    if species_count_sites is None or europe_gdf is None or selected_species_type is None:
        # Return empty figure
        fig = go.Figure()
        fig.add_annotation(
            text="No data available",
            xref="paper", yref="paper",
            x=0.5, y=0.5,
            showarrow=False,
            font=dict(size=16, color=COLORS['text_primary'])
        )
        fig.update_layout(
            height=600,
            width=900,
            plot_bgcolor=COLORS['background'],
            paper_bgcolor=COLORS['background'],
        )
        return fig
    
    # Create the scatter map for the selected species type
    fig = create_species_scatter_map(species_count_sites, selected_species_type, europe_gdf)
    return fig


if __name__ == '__main__':
    app.run(debug=APP_DEBUG, port=APP_PORT)
