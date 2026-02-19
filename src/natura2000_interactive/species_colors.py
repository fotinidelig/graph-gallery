"""
Species color mapping for Natura 2000 visualization.

This module provides color assignments for different species groups.
Colors are designed to work harmoniously with the app's sage green background (#BCCAB9).
"""

# Color palette for species groups
# Colors are chosen to:
# 1. Work harmoniously with sage green background (#BCCAB9)
# 2. Have semantic meaning (e.g., blue for aquatic, green for plants)
# 3. Maintain good contrast and readability
# 4. Complement existing habitat color scheme
SPECIES_COLORS = {
    'Amphibians': '#5A9A8B',  # Amphibians - Teal (water + land)
    'Birds': '#4A7BA7',  # Birds - Sky blue (air)
    'Fish': '#2E86AB',  # Fish - Deep blue (water)
    'Invertebrates': '#9B7BB8',  # Invertebrates - Purple (diverse)
    'Lichens': '#8B8B7A',  # Lichens - Muted gray-green (subtle)
    'Mammals': '#8B6F47',  # Mammals - Earth brown (land)
    'Plants': '#6B8E5A',  # Plants - Forest green (nature)
    'Reptiles': '#A67C52'   # Reptiles - Warm brown (earth)
}

# Alternative palette option (more vibrant, if preferred):
# SPECIES_COLORS_ALT = {
#     0: '#4ECDC4',  # Amphibians - Bright teal
#     1: '#5B9BD5',  # Birds - Light blue
#     2: '#2E75B6',  # Fish - Deep blue
#     3: '#C55AD5',  # Invertebrates - Magenta
#     4: '#A0A0A0',  # Lichens - Medium gray
#     5: '#C55A5A',  # Mammals - Terracotta
#     6: '#70AD47',  # Plants - Fresh green
#     7: '#D2691E'   # Reptiles - Chocolate
# }

# Muted palette option (more subtle, blends better with background):
# SPECIES_COLORS_MUTED = {
#     0: '#6B9A8B',  # Amphibians - Muted teal
#     1: '#6B8BA7',  # Birds - Muted blue
#     2: '#5A7A9B',  # Fish - Muted deep blue
#     3: '#8B7BA8',  # Invertebrates - Muted purple
#     4: '#7B7B6A',  # Lichens - Muted gray-green
#     5: '#7B6F57',  # Mammals - Muted brown
#     6: '#6B8E5A',  # Plants - Forest green (same as main)
#     7: '#9B7C62'   # Reptiles - Muted warm brown
# }


def get_species_color(species_type):
    """
    Get the color for a species group.
    
    Parameters:
    -----------
    species_type : str
        The species group name (e.g., 'Amphibians', 'Birds', etc.)
        
    Returns:
    --------
    str
        Hex color code for the species group
    """
    return SPECIES_COLORS.get(species_type, '#bbbaba')  # Default gray if not found


# Example usage:
if __name__ == "__main__":
    print("Species Color Palette:")
    print("=" * 50)
    for species_type, color in SPECIES_COLORS.items():
        print(f"{species_type:15}: {color}")
    print("=" * 50)
    print(f"\nBackground color: #BCCAB9 (sage green)")
    print("All colors are designed to work harmoniously with this background.")
