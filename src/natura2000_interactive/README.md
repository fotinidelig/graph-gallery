# Natura 2000 Interactive Visualization Project

Interactive data visualization project exploring Natura 2000 protected sites across Europe.

## Stack Recommendation

**Primary: Plotly + Dash**

- **Plotly**: Interactive visualizations (maps, charts, annotations, arrows)
- **Dash**: Web app framework (Python-only, no JS required)
- **Dash Bootstrap Components**: Professional UI components

## Project Structure

```
natura2000_interactive/
├── README.md                    # This file
├── requirements.txt             # Python dependencies
├── config.py                    # Configuration (paths, colors, settings)
├── app.py                       # Main Dash application
├── data_loader.py               # Data loading utilities
├── visualizations.py            # Plot creation functions
├── habitat_colors.py            # Habitat color mappings
├── export_gpkg_to_csv.py       # Script to export Natura2000 GeoPackage data to CSV
├── dash_example.py              # Minimal Dash app example
└── plotly_styling_example.py    # Examples of Plotly styling capabilities
```

## Data Requirements

The app requires the following CSV files in `data/natura2000_2024/csv/`:
- `HABITATS.csv`
- `HABITATCLASS.csv`
- `NATURA2000SITES.csv`
- `SPECIES.csv`
- `OTHERSPECIES.csv`

Additionally, a GeoJSON file for Europe is needed at `graph_gallery/src/data/world.geo.json` (or configure the path in `data_loader.py`).

## Quick Start

### 1. Install Dependencies

```bash
python -m venv venv
source venv/bin/activate  # or `venv\Scripts\activate` on Windows
pip install -r requirements.txt
```

### 2. Prepare Data

First, download the GeoPackage of Natura2000 (the most recent one as of January 2026 is [here](https://sdi.eea.europa.eu/data/91357f39-7866-41ce-b447-43905c364ec8).  
The app expects CSV files in the `data/natura2000_2024/csv/` directory. If you have a GeoPackage file (`Natura2000_end2024.gpkg`), you can export it to CSV using the provided script:

```bash
python export_gpkg_to_csv.py
```

This script will:
- Read all layers from `Natura2000_end2024.gpkg`
- Export each layer to a CSV file in `data/natura2000_2024/csv/`
- Automatically handle geometry columns (dropped for CSV export)

**Note**: Update the paths in `export_gpkg_to_csv.py` and `config.py` to match your local setup.

### 3. Configure Paths

Edit `config.py` to set the correct data directory path:

```python
# Use relative path from project root (default)
BASE_DATA_DIR = Path(__file__).parent.parent.parent.parent.parent / 'data' / 'natura2000_2024' / 'csv'

# Or use absolute path
# BASE_DATA_DIR = Path('/path/to/your/data/natura2000_2024/csv')
```

### 4. Run the App

```bash
python app.py
# Open http://localhost:8050 in browser
```

### 5. Explore Examples

- `dash_example.py`: Minimal Dash app with filters and interactivity
- `plotly_styling_example.py`: Examples of annotations, arrows, multiple subplots

## Current Features

- **Scatter Plot Spiral**: Visualizes habitat diversity with custom spiral layout
- **Sankey Diagram**: Shows flow from habitat descriptions → clusters → countries
- **Choropleth Grid**: 2×4 grid of normalized habitat coverage maps per country
- **Storytelling Layout**: Scroll-based narrative with inline annotations
- **Custom Color Palette**: Aesthetic color scheme matching the design requirements

## Key Features to Implement

- **Interactive Map**: Natura 2000 sites with click interactions
- **Filters**: Species, habitat type, country, time period
- **Linked Views**: Select in one plot → highlight in another
- **Detailed Info**: Click sites → show detailed information panel
- **Multiple Visualizations**: Maps, bar charts, time series, treemaps

## Learning Resources

- [Dash Documentation](https://dash.plotly.com/)
- [Plotly Python Documentation](https://plotly.com/python/)
- [Dash Bootstrap Components](https://dash-bootstrap-components.opensource.faculty.ai/)
- [Plotly Mapbox Tutorial](https://plotly.com/python/mapbox-layers/)

## Data Export Script

The `export_gpkg_to_csv.py` script converts GeoPackage layers to CSV format:

```bash
python export_gpkg_to_csv.py
```

**What it does:**
- Lists all layers in `Natura2000_end2024.gpkg`
- Reads each layer using geopandas
- Exports to CSV (geometry columns are automatically dropped)
- Saves files to `data/natura2000_2024/csv/`

**Configuration:**
Update paths in `export_gpkg_to_csv.py` to match your local setup before running.

## Next Steps

1. Configure paths in `config.py` and `export_gpkg_to_csv.py`
2. Run `export_gpkg_to_csv.py` to prepare your data
3. Run `app.py` to start the visualization app
4. Explore the codebase to understand the implementation
