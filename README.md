# Glucose Data Preprocessing for Machine Learning

A comprehensive tool for preprocessing glucose monitoring data from continuous glucose monitors (CGM) to prepare it for machine learning applications. This project handles data consolidation, gap detection, interpolation, calibration smoothing, and sequence filtering.

## 🚀 Quick Start

### 1. Setup Project Using UV

This project uses [UV](https://docs.astral.sh/uv/) as the package manager for fast and reliable dependency management.

#### Prerequisites

- Python 3.11 or higher
- UV package manager

#### Installation

1. **Install UV** (if not already installed):

   ```bash
   # On macOS and Linux
   curl -LsSf https://astral.sh/uv/install.sh | sh

   # On Windows
   powershell -c "irm https://astral.sh/uv/install.ps1 | iex"

   # Or using pip
   pip install uv
   ```

2. **Clone and setup the project**:

   ```bash
   git clone <your-repo-url>
   cd glucose_data_processing

   # Install dependencies using UV
   uv sync

   # Activate the virtual environment
   uv shell
   ```

3. **Verify installation**:
   ```bash
   python glucose_cli.py --help
   ```

### 2. Dependencies

The project requires the following Python packages (automatically managed by UV):

- **pandas** (≥2.3.2) - Data manipulation and analysis
- **polars** (≥1.33.1) - Fast DataFrame library for large datasets
- **pyarrow** (≥21.0.0) - Columnar data format for efficient data processing
- **typer** (≥0.12.0) - Command-line interface framework

All dependencies are specified in `pyproject.toml` and will be automatically installed with `uv sync`.

## 📋 Project Overview

### What is this project about?

This project is designed to preprocess continuous glucose monitoring (CGM) data for machine learning applications. It addresses common challenges in glucose data:

- **Data Fragmentation**: CGM data often comes in multiple CSV files that need consolidation
- **Time Gaps**: Missing data points due to device disconnections or calibration periods
- **Data Quality Issues**: Spikes during calibration events and irregular sampling intervals
- **Sequence Management**: Identifying continuous data segments suitable for ML training

### Project Goals

1. **Data Consolidation**: Merge multiple CSV files into a single, chronologically ordered dataset
2. **Gap Detection**: Identify time gaps and create sequence boundaries
3. **Smart Interpolation**: Fill small gaps while preserving sequence integrity
4. **Calibration Handling**: Remove or smooth calibration-related data spikes
5. **ML-Ready Output**: Generate clean, continuous sequences suitable for time-series ML models

## 🖥️ CLI Usage

### Basic Usage

```bash
python glucose_cli.py <input_folder> [OPTIONS]
```

### Command Line Options

| Option                             | Short | Default                | Description                                             |
| ---------------------------------- | ----- | ---------------------- | ------------------------------------------------------- |
| `--output`, `-o`                   |       | `glucose_ml_ready.csv` | Output file path for ML-ready data                      |
| `--interval`, `-i`                 |       | `5`                    | Time discretization interval in minutes                 |
| `--gap-max`, `-g`                  |       | `15`                   | Maximum gap size to interpolate in minutes              |
| `--min-length`, `-l`               |       | `200`                  | Minimum sequence length to keep for ML training         |
| `--calibration/--no-calibration`   |       | `True`                 | Interpolate calibration glucose values                  |
| `--verbose`, `-v`                  |       | `False`                | Enable verbose output                                   |
| `--stats/--no-stats`               |       | `True`                 | Show processing statistics                              |
| `--save-intermediate`, `-s`        |       | `False`                | Save intermediate files after each step                 |
| `--calibration-period`, `-c`       |       | `165`                  | Gap duration considered as calibration period (minutes) |
| `--remove-after-calibration`, `-r` |       | `24`                   | Hours of data to remove after calibration period        |

### Examples

```bash
# Basic processing with default settings
python glucose_cli.py ./000-csv

# Custom parameters with verbose output
python glucose_cli.py ./000-csv --output my_glucose_data.csv --interval 5 --gap-max 10 --verbose

# Process with calibration period detection and data removal
python glucose_cli.py ./000-csv --calibration-period 165 --remove-after-calibration 24 --save-intermediate

# Quick processing without statistics
python glucose_cli.py ./000-csv --no-stats --output quick_output.csv
```

## 📁 Input File Requirements

### Folder Structure

Place your CSV files in a folder (e.g., `000-csv/`) with the following structure:

```
000-csv/
├── 000-14 oct-28 oct 2019.csv
├── 001-28 oct-10 nov 2019.csv
├── 002-11 nov-24 nov 2019.csv
└── ... (additional CSV files)
```

### CSV File Format

Each CSV file should contain glucose monitoring data with the following required columns:

| Column Name                       | Description           | Example                                 |
| --------------------------------- | --------------------- | --------------------------------------- |
| `Timestamp (YYYY-MM-DDThh:mm:ss)` | ISO timestamp         | `2019-10-28T16:42:37`                   |
| `Event Type`                      | Type of glucose event | `EGV`, `Calibration`, `Insulin`, `Carb` |
| `Glucose Value (mg/dL)`           | Glucose reading       | `120.0`                                 |
| `Insulin Value (u)`               | Insulin amount        | `2.5`                                   |
| `Carb Value (grams)`              | Carbohydrate amount   | `30.0`                                  |

### Supported Event Types

- **EGV**: Estimated Glucose Value (main glucose readings)
- **Calibration**: Calibration events (can cause data spikes)
- **Insulin**: Insulin administration
- **Carb**: Carbohydrate intake

## 📊 Processing Information Fields Explained

### Step-by-Step Processing Output

#### 1. Consolidation Phase

```
Found 34 CSV files to consolidate
Processing: 000-14 oct-28 oct 2019.csv
  ✓ Extracted 4,014 records
Total records collected: 1,234,567
Records with valid timestamps: 1,200,000
Date range: 2019-10-14T16:42:37 to 2025-09-17T13:30:12
```

#### 2. Gap Detection and Sequences

```
Created 1,245 sequences
Found 1,244 gaps > 15 minutes
Calibration Periods Detected: 89
Records Removed After Calibration: 45,678
```

#### 3. Interpolation Analysis

```
Small Gaps Identified and Processed: 2,456 gaps
Interpolated Data Points Created: 4,123 points
Total Field Interpolations: 8,246 values
Glucose Interpolations: 4,123 values
Large Gaps Skipped: 1,244 gaps
```

#### 4. Sequence Filtering

```
Original Sequences: 1,245
Sequences After Filtering: 892
Sequences Removed: 353
Original Records: 1,154,322
Records After Filtering: 987,654
```

### Why More Small Gaps Than Interpolations?

The difference between "Small Gaps Identified" and actual interpolations occurs due to several factors:

#### 1. **Gap Size Thresholds**

- **Small Gap Detection**: Any gap between `expected_interval` (5 min) and `small_gap_max` (15 min)
- **Interpolation Logic**: Only gaps that represent 1-2 missing data points are interpolated

#### Example:

- Gap of 10 minutes detected (2 missing 5-minute intervals)
- Gap of 12 minutes detected (2.4 missing intervals) → Only 2 points interpolated
- Gap of 20 minutes detected → Skipped (exceeds small_gap_max)

#### 2. **Data Quality Requirements**

- Both previous and next values must be valid numeric data
- If either value is missing or non-numeric, interpolation is skipped
- Only glucose, insulin, and carb values are interpolated (not timestamps or event types)

#### 3. **Sequence Boundaries**

- Gaps at sequence boundaries are not interpolated
- Large gaps (>15 min) create new sequences instead of being filled

### Why Fewer Actual Interpolated Values?

The "Total Field Interpolations" count includes:

- **Glucose Value interpolations**: Most common
- **Insulin Value interpolations**: Only when both values are valid
- **Carb Value interpolations**: Only when both values are valid

**Common scenarios where interpolation is skipped:**

1. **Missing adjacent values**: Previous or next value is empty
2. **Non-numeric values**: Text or invalid data in adjacent cells
3. **Mixed event types**: Different event types around the gap
4. **Boundary conditions**: Gaps at the start/end of sequences

## 🔧 Advanced Configuration

### Calibration Period Detection

The system can automatically detect calibration periods (typically 2-3 hours) and remove subsequent data to avoid inaccurate readings:

```bash
python glucose_cli.py ./000-csv --calibration-period 165 --remove-after-calibration 24
```

### Custom Interpolation Settings

Fine-tune interpolation behavior:

```bash
# More aggressive interpolation (up to 30-minute gaps)
python glucose_cli.py ./000-csv --gap-max 30 --interval 5

# Conservative interpolation (only 10-minute gaps)
python glucose_cli.py ./000-csv --gap-max 10 --interval 5
```

### Debugging and Analysis

Enable intermediate file saving to inspect each processing step:

```bash
python glucose_cli.py ./000-csv --save-intermediate --verbose
```

This creates files like:

- `consolidated_data.csv`
- `step2_sequences_created.csv`
- `step3_interpolated_values.csv`
- `step4_calibration_interpolated.csv`
- `step5_filtered_sequences.csv`
- `step6_ml_ready.csv`

## 📈 Output Data Format

The final ML-ready dataset contains:

| Column                            | Type    | Description                                    |
| --------------------------------- | ------- | ---------------------------------------------- |
| `sequence_id`                     | Integer | Unique identifier for continuous data segments |
| `Timestamp (YYYY-MM-DDThh:mm:ss)` | String  | ISO formatted timestamp                        |
| `Event Type`                      | String  | Type of glucose event                          |
| `Glucose Value (mg/dL)`           | Float64 | Glucose reading (interpolated where needed)    |
| `Insulin Value (u)`               | Float64 | Insulin amount                                 |
| `Carb Value (grams)`              | Float64 | Carbohydrate amount                            |

## 🎯 Machine Learning Applications

The processed data is optimized for:

- **Time Series Forecasting**: Predict future glucose levels
- **Anomaly Detection**: Identify unusual glucose patterns
- **Sequence Modeling**: LSTM, GRU, or Transformer models
- **Classification**: Event type prediction or risk assessment

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

## 📄 License

[Add your license information here]

## 🆘 Support

For issues or questions:

1. Check the verbose output with `--verbose` flag
2. Enable intermediate file saving with `--save-intermediate`
3. Review the processing statistics for data quality insights
4. Open an issue on GitHub with sample data and error messages
