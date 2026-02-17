"""
Script to export GeoPackage layers to CSV files.

This script reads all layers from Natura2000_end2024.gpkg and exports each
layer to a corresponding CSV file in the csv directory.
"""

import fiona
import geopandas as gpd
from pathlib import Path
import pandas as pd

# Paths - adjust these to match your local setup
# Default: Use relative path from project root
PROJECT_ROOT = Path(__file__).parent.parent.parent.parent
BASE_DATA_DIR = PROJECT_ROOT / 'data' / 'natura2000_2024'
GPKG_PATH = BASE_DATA_DIR / 'Natura2000_end2024.gpkg'
CSV_DIR = BASE_DATA_DIR / 'csv'

# Alternative: Use absolute paths
# Uncomment and adjust if you prefer absolute paths:
# BASE_DATA_DIR = Path('/path/to/your/data/natura2000_2024')
# GPKG_PATH = BASE_DATA_DIR / 'Natura2000_end2024.gpkg'
# CSV_DIR = BASE_DATA_DIR / 'csv'

# Ensure CSV directory exists
CSV_DIR.mkdir(parents=True, exist_ok=True)

# List all layers in the GeoPackage
print(f"Reading GeoPackage: {GPKG_PATH}")
layers = fiona.listlayers(str(GPKG_PATH))
print(f"Found {len(layers)} layers: {layers}")

# Export each layer to CSV
for layer in layers:
    print(f"\nProcessing layer: {layer}")
    
    # Read layer as GeoDataFrame
    gdf = gpd.read_file(str(GPKG_PATH), layer=layer)
    
    # Convert to DataFrame (drop geometry column for CSV)
    if 'geometry' in gdf.columns:
        df = pd.DataFrame(gdf.drop(columns=['geometry']))
    else:
        df = pd.DataFrame(gdf)
    
    # Save to CSV
    csv_path = CSV_DIR / f"{layer}.csv"
    df.to_csv(csv_path, index=False)
    
    print(f"  Exported {len(df)} rows to {csv_path}")
    print(f"  Columns: {list(df.columns)}")

print(f"\n✓ Export complete! CSV files saved to: {CSV_DIR}")
