"""
Pre-process NATURA2000SITES.csv to create a smaller file with only needed columns.

This script extracts only the columns needed for visualizations:
- SITECODE (for merging)
- LATITUDE (for map visualizations)
- LONGITUDE (for map visualizations)
- COUNTRY_CODE (for country-based aggregations)

Usage:
    python preprocess_sites.py [--input INPUT_FILE] [--output OUTPUT_FILE]
"""

import pandas as pd
from pathlib import Path
import argparse
from config import BASE_DATA_DIR


def preprocess_sites(input_file=None, output_file=None):
    """
    Pre-process sites CSV to extract only needed columns.
    
    Parameters
    ----------
    input_file : str or Path, optional
        Path to input NATURA2000SITES.csv file. If None, uses BASE_DATA_DIR.
    output_file : str or Path, optional
        Path to output processed file. If None, saves as NATURA2000SITES_processed.csv
        in the same directory as input.
    """
    # Determine input file path
    if input_file is None:
        input_path = BASE_DATA_DIR / 'NATURA2000SITES.csv'
    else:
        input_path = Path(input_file)
    
    if not input_path.exists():
        raise FileNotFoundError(f"Input file not found: {input_path}")
    
    print(f"Loading sites data from: {input_path}")
    print(f"Original file size: {input_path.stat().st_size / (1024*1024):.2f} MB")
    
    # Load only the columns we need
    required_columns = ['SITECODE', 'LATITUDE', 'LONGITUDE', 'COUNTRY_CODE']
    
    # First, read just to check which columns exist
    sample = pd.read_csv(input_path, nrows=1)
    available_columns = sample.columns.tolist()
    
    # Check if all required columns exist
    missing_columns = [col for col in required_columns if col not in available_columns]
    if missing_columns:
        print(f"Warning: Missing columns: {missing_columns}")
        print(f"Available columns: {available_columns}")
        # Try to find similar column names (case-insensitive)
        column_mapping = {}
        for req_col in missing_columns:
            for avail_col in available_columns:
                if req_col.lower() == avail_col.lower():
                    column_mapping[req_col] = avail_col
                    print(f"  Found similar column: {req_col} -> {avail_col}")
        if column_mapping:
            required_columns = [column_mapping.get(col, col) for col in required_columns]
    
    # Read only the columns we need
    print(f"Reading columns: {required_columns}")
    sites = pd.read_csv(input_path, usecols=required_columns)
    
    # Remove rows with missing critical data
    initial_rows = len(sites)
    sites = sites.dropna(subset=['SITECODE'])  # SITECODE is essential
    sites = sites.dropna(subset=['LATITUDE', 'LONGITUDE'], how='all')  # At least one coordinate needed
    
    removed_rows = initial_rows - len(sites)
    if removed_rows > 0:
        print(f"Removed {removed_rows} rows with missing critical data")
    
    # Rename columns back to standard names if we had to use alternatives
    if 'column_mapping' in locals():
        reverse_mapping = {v: k for k, v in column_mapping.items()}
        sites = sites.rename(columns=reverse_mapping)
    
    # Determine output file path
    if output_file is None:
        output_path = input_path.parent / 'NATURA2000SITES_processed.csv'
    else:
        output_path = Path(output_file)
    
    # Save processed file
    print(f"Saving processed file to: {output_path}")
    sites.to_csv(output_path, index=False)
    
    output_size = output_path.stat().st_size / (1024*1024)
    print(f"Processed file size: {output_size:.2f} MB")
    print(f"Size reduction: {((input_path.stat().st_size - output_path.stat().st_size) / input_path.stat().st_size * 100):.1f}%")
    print(f"Rows: {len(sites):,}")
    print(f"Columns: {list(sites.columns)}")
    
    return output_path


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Pre-process NATURA2000SITES.csv to create a smaller file'
    )
    parser.add_argument(
        '--input',
        type=str,
        help='Path to input NATURA2000SITES.csv file (default: from config)'
    )
    parser.add_argument(
        '--output',
        type=str,
        help='Path to output processed file (default: NATURA2000SITES_processed.csv in same directory)'
    )
    
    args = parser.parse_args()
    
    try:
        output_path = preprocess_sites(args.input, args.output)
        print(f"\n✅ Success! Processed file saved to: {output_path}")
        print("\nNext steps:")
        print("1. Replace NATURA2000SITES.csv with the processed version, OR")
        print("2. Update data_loader.py to use NATURA2000SITES_processed.csv")
    except Exception as e:
        print(f"\n❌ Error: {e}")
        raise
