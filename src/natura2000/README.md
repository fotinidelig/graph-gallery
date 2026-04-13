# Natura 2000 Exploration

## Data Requirements

By default, the app expects CSV files under `graph_gallery/src/data/natura2000_2024/csv/`:

- `HABITATS.csv`
- `HABITATCLASS.csv`
- `NATURA2000SITES.csv` (or the lighter `NATURA2000SITES_processed.csv`)
- `SPECIES.csv`
- `OTHERSPECIES.csv`

Additionally, the map visualizations require a Europe/world JSON file to map countries and country codes (see `graph_gallery/data/europe_countries.json`). 

Adjust the data paths to match where you keep your files.

### Optional: From GeoPackage to CSV

If you start from the official Natura 2000 GeoPackage (`Natura2000_end2024.gpkg`), you can generate the CSV files with:

```bash
python export_gpkg_to_csv.py
```

This will:

- Read layers from the GeoPackage
- Export them to CSV under `data/natura2000_2024/csv/`
- Drop geometry columns (since this app works with tabular + lat/lon data)
