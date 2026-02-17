"""
Configuration file for Natura 2000 visualization app.

Edit the paths below to match your local setup.
"""

from pathlib import Path

# Base data directory - adjust this to your local path
# Default: Use relative path from project root
# Change this to match your local setup
BASE_DATA_DIR = Path(__file__).parent.parent.parent.parent / 'data' / 'natura2000_2024' / 'csv'

# Alternative: Use absolute path
# Uncomment and adjust if you prefer absolute paths:
# BASE_DATA_DIR = Path('/path/to/your/data/natura2000_2024/csv')

# App settings
APP_PORT = 8050
APP_DEBUG = True

# Visualization settings
SCATTER_SPIRAL_CONFIG = {
    'a': 0.1,           # Initial radius offset
    'b': 0.2,           # Radial growth factor
    'theta_max': 6,     # Number of turns (multiplied by pi)
    'scale_x': 1.2,     # Horizontal scaling
    'scale_y': 0.2,     # Vertical scaling (squash factor)
    'center': (0.0, 0.0)  # Spiral center
}

# Styling settings
FONT_FAMILY = 'Noto Sans Mono'
PRIMARY_COLOR = '#3498db'  # Keep for accent/links if needed
INLINE_FONTSIZE = 12

# Color palette
COLORS = {
    'background': '#BCCAB9',
    'text_primary': '#2A1800',
    'details': '#7B8A6E',
    'white': '#FCFFF7'
}
