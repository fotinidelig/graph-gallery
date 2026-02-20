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
from config import APP_PORT, APP_DEBUG, FONT_FAMILY, FONT_FAMILY_STORY, COLORS


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
font_family_story_css = FONT_FAMILY_STORY.replace(' ', '+')
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
            @import url('https://fonts.googleapis.com/css2?family=Playfair+Display:wght@400;500;600;700&display=swap');
            
            * {
                margin: 0;
                padding: 0;
                box-sizing: border-box;
            }
            
            body {
                font-family: ''' + FONT_FAMILY_STORY + ''', sans-serif;
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
                font-family: ''' + FONT_FAMILY_STORY + ''', sans-serif;
                font-size: 1.1rem;
                line-height: 1.8;
                margin-bottom: 3rem;
                color: ''' + COLORS['text_primary'] + ''';
                max-width: 800px;
                margin-left: auto;
                margin-right: auto;
            }
            
            .story-text h1 {
                font-family: ''' + FONT_FAMILY_STORY + ''', sans-serif;
                font-size: 2.5rem;
                font-weight: 600;
                margin-bottom: 1.5rem;
                color: ''' + COLORS['text_primary'] + ''';
            }
            
            .story-text h2 {
                font-family: ''' + FONT_FAMILY_STORY + ''', sans-serif;
                font-size: 1.8rem;
                font-weight: 500;
                margin-bottom: 1rem;
                color: ''' + COLORS['text_primary'] + ''';
            }
            
            .story-text p {
                font-family: ''' + FONT_FAMILY_STORY + ''', sans-serif;
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
                font-family: ''' + FONT_FAMILY + ''', monospace;
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
                font-family: ''' + FONT_FAMILY + ''', monospace;
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
                font-family: ''' + FONT_FAMILY + ''', monospace;
                font-size: 1.2rem;
                font-weight: 500;
                margin-bottom: 0.5rem;
                color: ''' + COLORS['text_primary'] + ''';
            }
            
            .annotation p {
                font-family: ''' + FONT_FAMILY + ''', monospace;
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
            
            /* Story content wrapper */
            /* PADDING ADJUSTMENT: Change padding values to adjust spacing around content */
            /* Format: padding: [top] [right] [bottom] [left] or padding: [vertical] [horizontal] */
            .story-content {
                position: relative;
                z-index: 2;
                max-width: 1400px;
                margin: 0 auto;
                padding: 2rem 2rem;  /* ADJUST THIS: Reduced from 4rem vertical. Increase for more space, decrease for less */
            }
            
            
            /* Story point styling - plain text, no boxes */
            /* MARGIN ADJUSTMENT: Change the values below to adjust spacing between story points and plots */
            /* Format: margin: [top] [right] [bottom] [left] or margin: [vertical] [horizontal] */
            /* Examples: '3rem 0' (3rem top/bottom, 0 left/right), '2rem 0 4rem 0' (2rem top, 0 right, 4rem bottom, 0 left) */
            .story-point {
                position: relative;
                z-index: 2;
                margin: 3rem 0;  /* ADJUST THIS: Reduced from 8rem. Increase for more space, decrease for less */
                max-width: 500px;
            }
            
            .story-point.left {
                margin-left: 25%;
                margin-right: auto;
                text-align: left;
            }
            
            .story-point.right {
                margin-left: auto;
                margin-right: 25%;
                text-align: right;
            }
            
            .story-point.center {
                margin-left: auto;
                margin-right: auto;
                text-align: center;
            }
            
            .story-point p {
                font-family: ''' + FONT_FAMILY_STORY + ''', sans-serif;
                font-size: 1.5rem;  /* ADJUST THIS: Increased from 1.3rem for Pompiere. Change to adjust story text size */
                line-height: 1.8;
                color: ''' + COLORS['text_primary'] + ''';
                margin: 0;
            }
            
            /* Make numbers bold in story text */
            .story-point p strong {
                font-weight: bold;
            }
            
            /* Story point with plot side by side */
            /* MARGIN ADJUSTMENT: Change margin to adjust spacing for side-by-side story and plot */
            .story-point-with-plot {
                display: flex;
                align-items: center;
                gap: 2rem;
                margin: 3rem 0;  /* ADJUST THIS: Reduced from 6rem. Increase for more space, decrease for less */
                max-width: 1400px;
            }
            
            .story-point-with-plot .story-point {
                flex: 0 0 45%;
                margin: 0;
            }
            
            .story-point-with-plot .plot-wrapper {
                flex: 0 0 50%;
            }
            
            /* Natura 2000 text styling */
            .natura-text {
                color: ''' + COLORS['natura_blue'] + ''';
                font-weight: 600;
            }
            
            /* Main title */
            .main-title {
                position: relative;
                z-index: 2;
                text-align: center;
                font-family: ''' + FONT_FAMILY_STORY + ''', sans-serif;
                font-size: 3.5rem;
                font-weight: 600;
                margin: 4rem auto 8rem auto;
                max-width: 800px;
                color: ''' + COLORS['text_primary'] + ''';
            }
            
            /* Plot with toggleable annotation */
            /* MARGIN ADJUSTMENT: Change margin values to adjust spacing above/below plots */
            .plot-with-annotation {
                position: relative;
                margin: 3rem 0;  /* ADJUST THIS: Reduced from 6rem. Increase for more space, decrease for less */
                display: flex;
                justify-content: center;  /* Changed from flex-end to center the plot */
            }
            
            .plot-right {
                flex: 0 0 80%;  /* Increased from 70% to make centered plot wider */
                max-width: 1200px;  /* Added max-width to prevent it from getting too wide */
                position: relative;
            }
            
            /* Toggleable graph annotation */
            .toggle-annotation {
                position: absolute;
                left: -60px;
                top: 10px;
                z-index: 10;
                display: flex;
                align-items: flex-start;
            }
            
            .toggle-annotation-button {
                width: 40px;
                height: 40px;
                border-radius: 50%;
                background: ''' + COLORS['white'] + ''';
                border: 2px solid ''' + COLORS['details'] + ''';
                cursor: pointer;
                display: flex;
                align-items: center;
                justify-content: center;
                box-shadow: 0 2px 8px rgba(0,0,0,0.15);
                transition: all 0.3s ease;
                font-family: ''' + FONT_FAMILY + ''', monospace;
                font-size: 1.2rem;
                color: ''' + COLORS['text_primary'] + ''';
                z-index: 11;
            }
            
            .toggle-annotation-button:hover {
                background: ''' + COLORS['details'] + ''';
                color: ''' + COLORS['white'] + ''';
                transform: scale(1.1);
            }
            
            .toggle-annotation-content {
                position: absolute;
                right: 50px;
                top: 0;
                background: ''' + COLORS['white'] + ''';
                background: rgba(252, 255, 247, 0.95);
                padding: 0.75rem 1rem;
                border-radius: 8px;
                box-shadow: 0 4px 12px rgba(0,0,0,0.2);
                font-family: ''' + FONT_FAMILY + ''', monospace;
                font-size: 0.85rem;
                color: ''' + COLORS['text_primary'] + ''';
                max-width: 320px;
                width: 320px;
                line-height: 1.4;
                border-right: 3px solid ''' + COLORS['details'] + ''';
                backdrop-filter: blur(4px);
                opacity: 0;
                visibility: hidden;
                transition: all 0.3s ease;
                transform: translateX(10px);
                white-space: normal;
            }
            
            .toggle-annotation-content.visible {
                opacity: 1;
                visibility: visible;
                transform: translateX(0);
            }
            
            .toggle-annotation-content h3 {
                font-family: ''' + FONT_FAMILY + ''', monospace;
                font-size: 1rem;
                font-weight: 600;
                margin-bottom: 0.5rem;
                color: ''' + COLORS['text_primary'] + ''';
            }
            
            .toggle-annotation-content p {
                font-family: ''' + FONT_FAMILY + ''', monospace;
                margin: 0;
                font-size: 0.85rem;
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
    # Story content wrapper
    html.Div([
        
        # Main Title
        html.H1([
            "Exploring the ",
            html.Span("Natura 2000", className="natura-text"),
            " network"
        ], className="main-title center"),
        
        # Story Point 1 - positioned left
        html.Div([
            html.P([
                html.Span("Natura 2000", className="natura-text"),
                " is the largest network of protected areas in the world."
            ])
        ], className="story-point left"),
        
        # Story Point 2 - positioned right
        html.Div([
            html.P([
                "Spanning all across Europe and covering more than ",
                html.Strong("18%"),
                " land and ",
                html.Strong("7%"),
                " marine area as of ",
                html.Strong("2017"),
                "."
            ])
        ], className="story-point right"),
        
        # Story Point 3 - positioned left
        html.Div([
            html.P([
                "There are ",
                html.Strong("27"),
                " different habitats, which we can cluster into ",
                html.Strong("8"),
                " categories, and over ",
                html.Strong("280 thousand"),
                " protected sites."
            ])
        ], className="story-point left"),
        
        # Scatter Plot with Toggleable Annotation
        html.Div([
            html.Div([
                html.Div([
                    html.Button(
                        "ℹ",
                        id="toggle-annotation-btn",
                        className="toggle-annotation-button",
                        n_clicks=0
                    ),
                    html.Div([
                        html.H3("Explore the data"),
                        html.P([
                            "Explore the data by hovering over them, clicking, zooming in/out, or selecting various regions."
                        ])
                    ], id="toggle-annotation-content", className="toggle-annotation-content")
                ], className="toggle-annotation"),
                dcc.Graph(
                    id='scatter-plot',
                    figure=create_scatter_plot(scatter_data),
                    config={'displayModeBar': False}
                ),
            ], className="plot-right")
        ], className="plot-with-annotation"),
        
    ], className="story-content"),  # Close story-content wrapper
    
    # Story Point: These habitats are found in 27 countries
    html.Div([
        html.Div([
            html.P([
                "These habitats are found in ",
                html.Strong("27"),
                " countries, in different countries."
            ])
        ], className="story-point left")
    ], className="story-content"),
    
    # Sankey Diagram with Toggleable Annotation
    html.Div([
        html.Div([
            html.Div([
                html.Div([
                    html.Button(
                        "ℹ",
                        id="toggle-sankey-annotation-btn",
                        className="toggle-annotation-button",
                        n_clicks=0
                    ),
                    html.Div([
                        html.H3("Explore the data"),
                        html.P([
                            "Notice how certain habitat clusters are concentrated in specific regions, reflecting Europe's diverse ecosystems from Mediterranean coasts to Nordic forests."
                        ])
                    ], id="toggle-sankey-annotation-content", className="toggle-annotation-content")
                ], className="toggle-annotation"),
                dcc.Store(id='sankey-selected-node', data=None),
                dcc.Graph(
                    id='sankey-plot',
                    figure=create_sankey_diagram(sankey_data),
                    config={'displayModeBar': False}
                ),
            ], className="plot-right")
        ], className="plot-with-annotation")
    ], className="story-content"),
    
    # Story Point: Spain has the most protected area
    html.Div([
        html.Div([
            html.P([
                "Spain notably has the most protected area..."
            ])
        ], className="story-point right")
    ], className="story-content"),
    
    # Story Point: For almost every category
    html.Div([
        html.Div([
            html.P([
                "...for almost every category of habitat, which we can observe when looking at the distribution of habitat coverage over countries."
            ])
        ], className="story-point left")
    ], className="story-content"),
    
    # Choropleth Grid Section (centered)
] + ([
    html.Div([
        html.Div([
            dcc.Graph(
                id='choropleth-grid',
                figure=create_cluster_choropleth_grid(choropleth_data, europe_gdf),
                config={'displayModeBar': False}
            )
        ])  # Removed maxWidth - uses full width of story-content wrapper. ADJUST: Change '3rem' in margin to adjust vertical spacing
    ], className="story-content", style={'maxWidth': '850px'})
] if choropleth_data is not None and europe_gdf is not None else []) + [
    
    # Story Point: Network tracks over 35 thousand species with Pictogram
] + ([
    html.Div([
        html.Div([
            html.Div([
                html.P([
                    "The network also tracks over ",
                    html.Strong("35 thousand"),
                    " threatened species, such as birds, plants, and more."
                ])
            ], className="story-point left"),
            html.Div([
                dcc.Graph(
                    id='species-pictogram',
                    figure=create_species_pictogram(species_count_data),
                    config={'displayModeBar': False}
                ),
            ], className="plot-wrapper")
        ], className="story-point-with-plot")
    ], className="story-content")
] if species_count_data is not None else []) + [
    
    # Story Point: Species spread across Europe
    html.Div([
        html.Div([
            html.P([
                "These species spread across Europe, though the largest proportion lives in the Mediterranean and Atlantic regions."
            ])
        ], className="story-point right")
    ], className="story-content"),
    
    # Story Point: Protected sites with varying richness
    html.Div([
        html.Div([
            html.P([
                html.Span("Natura 2000", className="natura-text"),
                " comprises protected sites with varying richness in terms of species."
            ])
        ], className="story-point left")
    ], className="story-content"),
    
    # Species Scatter Map Section (only if data is available)
] + ([
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
                'marginTop': 'auto',
                'marginBottom': '1rem',
                'marginLeft': 'auto',
                'marginRight': 'auto'
            }),
            html.Div([
                dcc.Graph(
                    id='species-scatter-map',
                    config={'displayModeBar': False}
                ),
            ], className="plot-wrapper", style={'position': 'relative'})
        ], className="plot-container", style={'flexDirection': 'column', 'maxWidth': '1200px', 'margin': '0 auto'})
    ], className="story-content")
] if species_count_sites is not None and europe_gdf is not None else []) + [
    
    # Closing Text
    html.Div([
        html.Div([
            html.Div([
                html.H2("Exploring Further", className="text-center",),
                html.P([
                    "These visualizations provide just a glimpse into the rich biodiversity data ",
                    "available through the ",
                    html.Span("Natura 2000", className="natura-text"),
                    " network. Each habitat tells a story of ",
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
    Output('toggle-annotation-content', 'className'),
    Input('toggle-annotation-btn', 'n_clicks'),
    prevent_initial_call=True
)
def toggle_annotation(n_clicks):
    if n_clicks is None or n_clicks == 0:
        return "toggle-annotation-content"
    # Toggle visibility based on even/odd clicks
    if n_clicks % 2 == 1:
        return "toggle-annotation-content visible"
    else:
        return "toggle-annotation-content"

@app.callback(
    Output('toggle-sankey-annotation-content', 'className'),
    Input('toggle-sankey-annotation-btn', 'n_clicks'),
    prevent_initial_call=True
)
def toggle_sankey_annotation(n_clicks):
    if n_clicks is None or n_clicks == 0:
        return "toggle-annotation-content"
    # Toggle visibility based on even/odd clicks
    if n_clicks % 2 == 1:
        return "toggle-annotation-content visible"
    else:
        return "toggle-annotation-content"

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
