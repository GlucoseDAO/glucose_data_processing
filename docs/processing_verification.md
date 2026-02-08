# Processing Verification Guide

This document describes the steps to verify the complete data processing pipeline, from repository setup to multi-database consolidation.

## 1. Repository Setup

Clone the repository and enter the project directory:

```bash
git clone https://github.com/GlucoseDAO/glucose_data_processing.git
cd glucose_data_processing
```

## 2. Installation

This project uses `uv` for dependency management. Install all required dependencies:

```bash
uv sync
```

## 3. Download Datasets

Use the `download.py` script to fetch the HUPA and UOM datasets. The script will automatically filter HUPA to only download the consolidated CSV files required for processing.

```bash
uv run download.py by-names "HUPA" "T1D-UOM"
```

The datasets will be saved in the `DATA/hupa` and `DATA/uom` directories.

## 4. Multi-Database Processing

Process the downloaded HUPA and UOM datasets along with the local Dexcom small sample. This command uses a specialized configuration file to combine and standardize data from all three sources into a single unified output.

```bash
uv run glucose_cli.py DATA/hupa DATA/uom test_data/dexcom_small --config hupa_uom_dexcomraw_combined.yaml
```

### What this command does:
1.  **Consolidation**: Merges all CSV files from the three input folders.
2.  **Standardization**: Applies the mappings defined in the configuration to unify field names (e.g., mapping various heart rate or glucose columns to standard names).
3.  **Preprocessing**: Runs the full pipeline:
    *   Detects and fills small gaps (interpolation).
    *   Handles high/low glucose value replacements.
    *   Removes sensor calibration periods and subsequent "noisy" data.
    *   Filters out short sequences (default minimum 200 points).
4.  **Sequence Management**: Automatically tracks and offsets sequence IDs across the different databases to ensure each sequence remains unique in the final file.
5.  **Output**: Generates a unified CSV file in the `OUTPUT/` folder (e.g., `hupa_uom_dexcom_small_ml_ready.csv`) containing the processed ML-ready data.

## 5. Verification

After the process completes, check the `OUTPUT/` folder for the results:
1.  **ML-ready CSV**: `hupa_uom_dexcom_small_ml_ready.csv` containing the combined data.
2.  **Statistics Report**: `hupa_uom_dexcom_small_ml_ready.txt` containing detailed processing statistics for each dataset and the combined total.

You can verify the success by checking the "DATA QUALITY" section at the end of the statistics report, which should show high completeness for the mapped fields.
