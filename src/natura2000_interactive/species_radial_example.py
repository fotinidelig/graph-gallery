"""
Example script demonstrating the species pictogram visualization.

This script loads species data and creates a pictogram showing
the count of species for each type (each marker represents 100 species).
"""

from data_loader import load_species_data, prepare_species_count_data
from visualizations import create_species_pictogram

# Load species data
print("Loading species data...")
species_data = load_species_data()

# Prepare count data
print("Preparing species count data...")
species_count = prepare_species_count_data(species_data)

print("\nSpecies counts:")
print(species_count)

# Create pictogram
print("\nCreating species pictogram...")
fig = create_species_pictogram(species_count)

# Show the plot
fig.write_html(
        "./graph_gallery/src/natura2000_interactive/polar_bar_species.html"
    )
fig.show()

print("\nDone! The species pictogram should be displayed in your browser.")
