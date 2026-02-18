"""
Data loading utilities for Natura 2000 visualization.
"""

import pandas as pd
from pathlib import Path
from habitat_colors import assign_cluster, CLUSTER_COLORS
from config import BASE_DATA_DIR

# Try to import geopandas, but make it optional
try:
    import geopandas as gpd
    HAS_GEOPANDAS = True
except ImportError:
    HAS_GEOPANDAS = False
    gpd = None


def load_natura2000_data(data_dir=None):
    """
    Load and prepare Natura 2000 datasets.
    
    Parameters
    ----------
    data_dir : str or Path, optional
        Path to data directory. If None, uses BASE_DATA_DIR from config.
        
    Returns
    -------
    dict
        Dictionary containing processed datasets:
        - 'habitats': DataFrame with habitat data
        - 'habitatclass': DataFrame with habitat class data
        - 'sites': DataFrame with site data (if available)
    """
    if data_dir is None:
        base_path = BASE_DATA_DIR
    else:
        base_path = Path(data_dir)
    
    # Load datasets
    habitats = pd.read_csv(base_path / 'HABITATS.csv')
    habitatclass = pd.read_csv(base_path / 'HABITATCLASS.csv')
    
    # Merge and prepare habitats data
    habitats['DETAIL_DESCRIPTION'] = habitats['DESCRIPTION']
    habitats = habitats.drop(['DESCRIPTION'], axis=1, errors='ignore')
    habitats = habitats.merge(
        habitatclass[['DESCRIPTION', 'SITECODE']], 
        on='SITECODE', 
        how='left'
    )
    
    # Add cluster assignment
    habitats['CLUSTER'] = habitats['DESCRIPTION'].map(lambda r: assign_cluster(r))
    
    # Convert cover from hectares to square kilometers
    habitats['COVER_KM'] = habitats['COVER_HA'] / 100
    habitats = habitats.drop(['COVER_HA'], axis=1, errors='ignore')
    
    return {
        'habitats': habitats,
        'habitatclass': habitatclass
    }


def prepare_scatter_data(habitats, habitatclass):
    """
    Prepare data for scatter plot visualization.
    
    Parameters
    ----------
    habitats : DataFrame
        Habitats dataframe
    habitatclass : DataFrame
        Habitat class dataframe
        
    Returns
    -------
    DataFrame
        Prepared data for scatter plot
    """
    import numpy as np
    
    # Count descriptions
    description_count = habitatclass.groupby('DESCRIPTION')['HABITATCODE'].count().reset_index(name='COUNT')
    description_count['COUNT_NORM'] = description_count['COUNT'] / description_count['COUNT'].max()
    
    # Calculate cover by description
    cover = habitats.groupby(['DESCRIPTION', 'CLUSTER'])['COVER_KM'].sum().reset_index(name='COVER_KM')
    cover['COVER_NORM'] = cover['COVER_KM'] / cover['COVER_KM'].max()
    
    # Merge and sort
    description_count_sorted = description_count.sort_values(by='COUNT', ascending=False)
    description_cover_sorted = description_count_sorted.merge(cover, on='DESCRIPTION', how='left')
    
    # Add colors
    description_cover_sorted['COLOR'] = description_cover_sorted['DESCRIPTION'].map(
        lambda r: CLUSTER_COLORS[assign_cluster(r)]
    )
    
    return description_cover_sorted


def prepare_sankey_data(habitats):
    """
    Prepare data for Sankey diagram visualization.
    
    Parameters
    ----------
    habitats : DataFrame
        Habitats dataframe
        
    Returns
    -------
    DataFrame
        Prepared data for Sankey diagram
    """
    habitat_country_cover = habitats.groupby(
        ['CLUSTER', 'DESCRIPTION', 'COUNTRY_CODE']
    )['COVER_KM'].sum().reset_index(name='COVER_KM')
    
    habitat_country_cover['COVER_PERCENT'] = (
        habitat_country_cover.groupby('DESCRIPTION')['COVER_KM']
        .transform(lambda x: x / x.sum())
    )
    
    return habitat_country_cover


def load_europe_geodataframe(data_dir=None):
    """
    Load Europe GeoDataFrame for choropleth visualizations.
    
    Parameters
    ----------
    data_dir : str or Path, optional
        Path to data directory. If None, uses default relative path.
        
    Returns
    -------
    geopandas.GeoDataFrame
        GeoDataFrame with Europe country geometries
        
    Raises
    ------
    ImportError
        If geopandas is not installed
    FileNotFoundError
        If Europe GeoJSON file is not found
    """
    if not HAS_GEOPANDAS:
        raise ImportError("geopandas is required for choropleth visualizations. Install it with: pip install geopandas")
    
    if data_dir is None:
        # Use the world.geo.json file from graph_gallery/src/data/
        # Path relative to this file: data_loader.py is in graph_gallery/src/natura2000_interactive/
        # So we go up to graph_gallery/src/ then into data/
        current_file_path = Path(__file__).parent  # graph_gallery/src/natura2000_interactive/
        europe_path = current_file_path.parent / 'data' / 'world.geo.json'

    else:
        europe_path = Path(data_dir) / 'world.geo.json'
    
    if not europe_path.exists():
        raise FileNotFoundError(f"World GeoJSON file not found at: {europe_path}")
    
    europe_gdf = gpd.read_file(europe_path)
    
    # Filter to only countries that have data (will be done in prepare_choropleth_data)
    # But we can also filter to Europe if continent column exists
    if 'continent' in europe_gdf.columns:
        europe_gdf = europe_gdf[europe_gdf['continent'] == 'Europe']
    
    # Ensure iso_a2 is not null
    europe_gdf = europe_gdf[europe_gdf['iso_a2_eh'].notna()]
    
    return europe_gdf


def prepare_choropleth_data(habitats, europe_gdf):
    """
    Prepare data for choropleth grid visualization.
    
    This function matches the user's notebook processing:
    1. Groups habitats by COUNTRY_CODE and CLUSTER
    2. Filters europe_gdf to only countries present in the data
    3. Merges the grouped data with geometries
    
    Parameters
    ----------
    habitats : DataFrame
        Habitats dataframe with COUNTRY_CODE, CLUSTER, COVER_KM
    europe_gdf : geopandas.GeoDataFrame
        GeoDataFrame with geometry and ISO alpha-2 codes in column 'iso_a2'
        
    Returns
    -------
    DataFrame
        Merged DataFrame ready for choropleth visualization
    """
    # Group by country and cluster
    cluster_country_cover = (
        habitats.groupby(["COUNTRY_CODE", "CLUSTER"])["COVER_KM"]
        .sum()
        .reset_index()
    )
    
    # Get unique countries from the data
    countries = cluster_country_cover["COUNTRY_CODE"].unique()
    
    # Filter europe_gdf to only countries present in the data
    # (matching user's code: europe = map_countries[map_countries.iso_a2_eh.isin(countries)])
    europe = europe_gdf[europe_gdf['iso_a2'].isin(countries)]
    
    # Merge with Europe geometries
    chloropleth_data = (
        cluster_country_cover.merge(
            europe[['geometry', 'iso_a2']],
            left_on='COUNTRY_CODE',
            right_on='iso_a2',
            how='left'
        )
        .drop(['iso_a2'], axis=1, errors='ignore')
    )
    
    return chloropleth_data
