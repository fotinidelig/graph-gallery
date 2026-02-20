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
    europe = europe_gdf[europe_gdf['iso_a2_eh'].isin(countries)]
    
    # Merge with Europe geometries
    chloropleth_data = (
        cluster_country_cover.merge(
            europe[['geometry', 'iso_a2_eh']],
            left_on='COUNTRY_CODE',
            right_on='iso_a2_eh',
            how='left'
        )
        .drop(['iso_a2_eh'], axis=1, errors='ignore')
    )
    
    return chloropleth_data


def load_species_data(data_dir=None):
    """
    Load and merge species datasets (SPECIES.csv and OTHERSPECIES.csv).
    
    Parameters
    ----------
    data_dir : str or Path, optional
        Path to data directory. If None, uses BASE_DATA_DIR from config.
        
    Returns
    -------
    pandas.DataFrame
        Merged DataFrame with all species data
    """
    if data_dir is None:
        base_path = BASE_DATA_DIR
    else:
        base_path = Path(data_dir)
    
    # Load species datasets
    species = pd.read_csv(base_path / 'SPECIES.csv')
    other_species = pd.read_csv(base_path / 'OTHERSPECIES.csv')
    
    # Rename column for consistency
    other_species = other_species.rename(columns={'SPECIESGROUP': 'SPGROUP'})
    
    # Find common columns
    common_cols = other_species.columns.intersection(species.columns)
    
    # Merge the datasets
    all_species = pd.concat(
        [other_species[common_cols], species[common_cols]], 
        ignore_index=True
    )
    
    # Clean up SPGROUP column
    all_species['SPGROUP'] = all_species['SPGROUP'].astype(str)
    all_species = all_species[all_species['SPGROUP'] != 'nan']
    
    return all_species


def prepare_species_count_data(species_data):
    """
    Prepare species count data grouped by species type.
    
    Parameters
    ----------
    species_data : pandas.DataFrame
        DataFrame with species data containing SPGROUP and SPECIESNAME columns
        
    Returns
    -------
    pandas.DataFrame
        DataFrame with columns: SPGROUP, COUNT
        Sorted by count in descending order
    """
    species_count = (
        species_data.groupby('SPGROUP')
        .SPECIESNAME.nunique()
        .reset_index(name='COUNT')
    )
    species_count = species_count.sort_values(by='COUNT', ascending=False)
    
    return species_count


def prepare_species_scatter_map_data(species_data, data_dir=None):
    """
    Prepare species count data per site for scatter map visualization.
    
    This function merges species data with site location data (latitude, longitude)
    and groups by species type and site to count unique species per site.
    
    Parameters
    ----------
    species_data : pandas.DataFrame
        DataFrame with species data containing SPGROUP, SITECODE, and SPECIESNAME columns
    data_dir : str or Path, optional
        Path to data directory. If None, uses BASE_DATA_DIR from config.
        
    Returns
    -------
    pandas.DataFrame
        DataFrame with columns: SPGROUP, SITECODE, LATITUDE, LONGITUDE, COUNT, COUNTRY_CODE
        Contains species counts per site per species group
    """
    if data_dir is None:
        base_path = BASE_DATA_DIR
    else:
        base_path = Path(data_dir)
    
    # Load sites data
    sites = pd.read_csv(base_path / 'NATURA2000SITES.csv')
    
    # Merge species data with site location data
    species_count_sites = (
        species_data.merge(
            sites[['LATITUDE', 'LONGITUDE', 'SITECODE']], 
            on='SITECODE', 
            how='left'
        )
        .groupby(['SPGROUP', 'SITECODE', 'LATITUDE', 'LONGITUDE', 'COUNTRY_CODE'])['SPECIESNAME']
        .nunique().reset_index(name='COUNT')
    )
    
    # Remove rows with missing coordinates
    species_count_sites = species_count_sites.dropna(subset=['LATITUDE', 'LONGITUDE'])
    
    return species_count_sites


def prepare_species_per_country_data(species_data, data_dir=None):
    """
    Prepare species count data grouped by country and species type.
    
    Parameters
    ----------
    species_data : pandas.DataFrame
        DataFrame with species data containing SPGROUP, SITECODE, and SPECIESNAME columns
    data_dir : str or Path, optional
        Path to data directory. If None, uses BASE_DATA_DIR from config.
        
    Returns
    -------
    pandas.DataFrame
        DataFrame with columns: COUNTRY_CODE, SPGROUP, COUNT
        Contains species counts per country per species group
    """
    if data_dir is None:
        base_path = BASE_DATA_DIR
    else:
        base_path = Path(data_dir)
    
    # Load sites data to get COUNTRY_CODE
    sites = pd.read_csv(base_path / 'NATURA2000SITES.csv')
    
    # Merge species data with sites to get COUNTRY_CODE
    species_with_country = species_data.merge(
        sites[['SITECODE']],
        on='SITECODE',
        how='left'
    )
    
    # Group by country and species type
    species_per_country = (
        species_with_country.groupby(['COUNTRY_CODE', 'SPGROUP'])['SPECIESNAME']
        .nunique()
        .reset_index(name='COUNT')
    )
    
    # Remove rows with missing country codes
    species_per_country = species_per_country.dropna(subset=['COUNTRY_CODE'])
    
    return species_per_country
