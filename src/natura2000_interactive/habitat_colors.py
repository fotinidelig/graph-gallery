"""
Habitat color mapping for Natura 2000 visualization.

This module provides color assignments and clustering for different habitat types.
"""

import pandas as pd

def rgb_to_rgba(rgb, alpha=0.6):
    """Convert RGB tuple (0-1) to rgba string (0-255)."""
    r, g, b = rgb
    return f'rgba({int(r * 255)}, {int(g * 255)}, {int(b * 255)}, {alpha})'

# Define habitat clusters
HABITAT_CLUSTERS = {
    # Water/Marine (Blues)
    'Water/Marine': [
        'Inland water bodies (Standing water, Running water)',
        'Marine areas, Sea inlets',
        'Tidal rivers, Estuaries, Mud flats, Sand flats, Lagoons (including saltwork basins)',
        'Bogs, Marshes, Water fringed vegetation, Fens',
        'Marine and coastal habitats (general)'
    ],
    
    # Forest/Woodland (Greens)
    'Forest/Woodland': [
        'Broad-leaved deciduous woodland',
        'Mixed woodland',
        'Coniferous woodland',
        'Evergreen woodland',
        'Artificial forest monoculture (e.g. Plantations of poplar or Exotic trees)',
        'Woodland habitats (general)'
    ],
    
    # Grassland (Yellows/Greens)
    'Grassland': [
        'Dry grassland, Steppes',
        'Humid grassland, Mesophile grassland',
        'Improved grassland',
        'Alpine and sub-Alpine grassland',
        'Grassland and scrub habitats (general)'
    ],
    
    # Coastal (Teals/Cyans)
    'Coastal': [
        'Coastal sand dunes, Sand beaches, Machair',
        'Shingle, Sea cliffs, Islets',
        'Salt marshes, Salt pastures, Salt steppes'
    ],
    
    # Agricultural (Oranges/Browns)
    'Agricultural': [
        'Other arable land',
        'Extensive cereal cultures (including Rotation cultures with regular fallowing)',
        'Non-forest areas cultivated with woody plants (including Orchards, groves, Vineyards, Dehesas)',
        'Ricefields',
        'Agricultural habitats (general)'
    ],
    
    # Heath/Scrub (Purples/Pinks)
    'Heath/Scrub': [
        'Heath, Scrub, Maquis and Garrigue, Phygrana'
    ],
    
    # Rocky/Alpine (Grays)
    'Rocky/Alpine': [
        'Inland rocks, Screes, Sands, Permanent Snow and ice'
    ],
    
    # Urban/Artificial (Reds/Pinks)
    'Urban/Artificial': [
        'Other land (including Towns, Villages, Roads, Waste places, Mines, Industrial sites)'
    ],
    
    # Other
    'Other': [None]  # for nan values
}

# Color palette for each cluster
CLUSTER_COLORS = {
    'Water/Marine': '#3481b8',        # Blue
    'Forest/Woodland': '#4b934b',      # Green
    'Grassland': '#B8CA8B',            # Light orange/yellow
    'Coastal': '#60d2c8',              # Cyan/Teal
    'Agricultural': '#cc6a3d',         # Red/Orange
    'Heath/Scrub': '#CD9FDC',          # Purple
    'Rocky/Alpine': '#8e6d67',        # Brown/Gray
    'Urban/Artificial': '#983256',     # Pink
    'Other': '#bbbaba'                 # Gray
}


def assign_cluster(habitat):
    """
    Assign a habitat to its cluster.
    
    Parameters:
    -----------
    habitat : str or None
        The habitat description
        
    Returns:
    --------
    str
        The cluster name
    """
    if pd.isna(habitat):
        return 'Other'
    
    for cluster, habitats in HABITAT_CLUSTERS.items():
        if habitat in habitats:
            return cluster
    
    return 'Other'


def assign_cluster_color(habitat):
    """
    Get the color for a habitat's cluster.
    
    Parameters:
    -----------
    habitat : str or None
        The habitat description
        
    Returns:
    --------
    str
        Hex color code for the cluster
    """
    cluster = assign_cluster(habitat)
    return CLUSTER_COLORS[cluster]


# Example usage:
if __name__ == "__main__":
    # Test the functions
    test_habitats = [
        'Broad-leaved deciduous woodland',
        'Marine areas, Sea inlets',
        'Dry grassland, Steppes',
        None
    ]
    
    print("Habitat Clustering Test:")
    for habitat in test_habitats:
        cluster = assign_cluster(habitat)
        color = assign_cluster_color(habitat)
        print(f"{habitat}:")
        print(f"  Cluster: {cluster}")
        print(f"  Cluster Color: {color}")
        print()
