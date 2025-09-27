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
- **pyyaml** (≥6.0) - YAML configuration file support

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
2. **Data Standardization**: Replace textual glucose values (High/Low) with numeric equivalents
3. **Calibration Removal**: Remove calibration events to create interpolatable gaps
4. **Gap Detection**: Identify time gaps and create sequence boundaries
5. **Smart Interpolation**: Fill small gaps while preserving sequence integrity
6. **ML-Ready Output**: Generate clean, continuous sequences suitable for time-series ML models

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
| `--glucose-only`                   |       | `False`                | Output only glucose data with simplified fields         |

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

# Use configuration file
python glucose_cli.py ./000-csv --config glucose_config.yaml

# Override config file parameters with CLI arguments
python glucose_cli.py ./000-csv --config glucose_config.yaml --interval 10 --gap-max 30

# Process with glucose-only output (simplified format)
python glucose_cli.py ./000-csv --glucose-only --output glucose_only_data.csv

# Combine glucose-only with other options
python glucose_cli.py ./000-csv --glucose-only --min-length 100 --verbose
```

## ⚙️ YAML Configuration File

### Overview

For complex configurations or repeated processing, you can use a YAML configuration file to specify all parameters. Command line arguments always take precedence over configuration file values.

### Configuration File Usage

```bash
# Use configuration file
python glucose_cli.py ./000-csv --config glucose_config.yaml

# Override specific parameters from config file
python glucose_cli.py ./000-csv --config glucose_config.yaml --interval 10 --gap-max 30

# Use config file with custom output
python glucose_cli.py ./000-csv --config glucose_config.yaml --output custom_output.csv
```

### Configuration File Structure

The `glucose_config.yaml` file contains the essential configuration options:

```yaml
# Basic processing parameters
expected_interval_minutes: 5
small_gap_max_minutes: 15
remove_calibration: true
min_sequence_len: 200
save_intermediate_files: false

# Calibration period detection
calibration_period_minutes: 165
remove_after_calibration_hours: 24

# Glucose value replacement (configurable High/Low values)
glucose_value_replacement:
  high_value: 401 # Replace 'High' with 401 mg/dL
  low_value: 39 # Replace 'Low' with 39 mg/dL
  enabled: true

# Glucose-only mode (simplified output)
glucose_only: false # Output only glucose data with simplified fields
```

### Key Configuration Features

#### **Core Processing Parameters**

- **`expected_interval_minutes`**: Time discretization interval (default: 5 minutes)
- **`small_gap_max_minutes`**: Maximum gap size to interpolate (default: 15 minutes)
- **`remove_calibration`**: Remove calibration events to create interpolatable gaps
- **`min_sequence_len`**: Minimum sequence length for ML training (default: 200 records)
- **`save_intermediate_files`**: Save intermediate files for debugging

#### **Calibration Period Detection**

- **`calibration_period_minutes`**: Gap duration considered as calibration period (default: 165 minutes)
- **`remove_after_calibration_hours`**: Hours of data to remove after calibration (default: 24 hours)

#### **Configurable High/Low Values**

- **`high_value`**: Numeric value to replace 'High' glucose readings (default: 401 mg/dL)
- **`low_value`**: Numeric value to replace 'Low' glucose readings (default: 39 mg/dL)
- **`enabled`**: Enable/disable High/Low value replacement

#### **Glucose-Only Mode**

- **`glucose_only`**: When enabled, outputs only glucose data with simplified fields (default: false)

### Priority Order

1. **Command line arguments** (highest priority)
2. **Configuration file values**
3. **Default values** (lowest priority)

Example:

```bash
# Config file sets interval=5, CLI overrides to 10
python glucose_cli.py ./000-csv --config glucose_config.yaml --interval 10
# Result: interval=10 (CLI wins)
```

### Creating Your Own Configuration

1. Copy `glucose_config.yaml` to create your custom configuration
2. Modify values as needed for your specific use case (focus on the core parameters)
3. Use the `--config` parameter to specify your file

```bash
# Use custom configuration
python glucose_cli.py ./000-csv --config my_custom_config.yaml
```

**Note**: The configuration file includes only the essential parameters that are actively used by the preprocessing pipeline. This keeps the configuration simple and focused on the most important settings.

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

#### 2. High/Low Value Replacement

```
Replaced 245 'High' values with 401
Replaced 89 'Low' values with 39
Total replacements: 334
✓ Glucose field converted to Float64 type
```

#### 3. Calibration Event Removal

```
Found 245 calibration events to remove
Removed 245 calibration events
Records before removal: 1,200,000
Records after removal: 1,199,755
✓ Calibration events removed - gaps can now be interpolated
```

#### 4. Gap Detection and Sequences

```
Created 1,245 sequences
Found 1,244 gaps > 15 minutes
Calibration Periods Detected: 89
Records Removed After Calibration: 45,678
```

#### 5. Interpolation Analysis

```
Small Gaps Identified and Processed: 2,456 gaps
Interpolated Data Points Created: 4,123 points
Total Field Interpolations: 8,246 values
Glucose Interpolations: 4,123 values
Large Gaps Skipped: 1,244 gaps
```

#### 6. Sequence Filtering

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

### Why Calibration Removal Instead of Interpolation?

The new approach removes calibration events entirely rather than interpolating their values because:

1. **Cleaner Data**: Calibration events often contain spikes or inaccurate readings
2. **Better Interpolation**: Removing calibration events creates gaps that can be interpolated using surrounding EGV (glucose) values
3. **Consistent Methodology**: All gaps are treated uniformly in the interpolation step
4. **Reduced Noise**: Eliminates potential artifacts from calibration spikes

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

### High/Low Value Replacement

The system automatically replaces textual glucose values with numeric equivalents:

- **"High"** → **401 mg/dL** (configurable): Represents glucose readings above the device's upper measurement limit
- **"Low"** → **39 mg/dL** (configurable): Represents glucose readings below the device's lower measurement limit

#### Configurable Replacement Values

You can customize the High/Low replacement values through:

1. **Configuration File** (`glucose_config.yaml`):

   ```yaml
   glucose_value_replacement:
     high_value: 401 # Custom high value
     low_value: 39 # Custom low value
     enabled: true
   ```

2. **Command Line** (when using config file):
   ```bash
   python glucose_cli.py ./000-csv --config glucose_config.yaml --high-value 450 --low-value 30
   ```

#### Why This Replacement is Necessary

1. **ML Compatibility**: Machine learning models require numeric data
2. **Consistent Data Types**: Ensures the glucose field is Float64 throughout the dataset
3. **Meaningful Values**: 401 and 39 are clinically relevant threshold values that preserve the information that readings were outside normal measurement ranges
4. **Flexibility**: Allows customization for different devices or clinical requirements

The replacement occurs early in the pipeline (Step 2) to ensure all subsequent processing works with numeric glucose values.

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
- `step2_high_low_replaced.csv`
- `step3_calibrations_removed.csv`
- `step4_sequences_created.csv`
- `step5_interpolated_values.csv`
- `step6_filtered_sequences.csv`
- `step7_ml_ready.csv`

## 📈 Output Data Format

### Standard Output Format

The final ML-ready dataset contains:

| Column                            | Type    | Description                                    |
| --------------------------------- | ------- | ---------------------------------------------- |
| `sequence_id`                     | Integer | Unique identifier for continuous data segments |
| `Timestamp (YYYY-MM-DDThh:mm:ss)` | String  | ISO formatted timestamp                        |
| `Event Type`                      | String  | Type of glucose event                          |
| `Glucose Value (mg/dL)`           | Float64 | Glucose reading (interpolated where needed)    |
| `Insulin Value (u)`               | Float64 | Insulin amount                                 |
| `Carb Value (grams)`              | Float64 | Carbohydrate amount                            |

### Glucose-Only Output Format

When using the `--glucose-only` option, the output is simplified to contain only glucose data:

| Column                            | Type    | Description                                    |
| --------------------------------- | ------- | ---------------------------------------------- |
| `sequence_id`                     | Integer | Unique identifier for continuous data segments |
| `Timestamp (YYYY-MM-DDThh:mm:ss)` | String  | ISO formatted timestamp                        |
| `Glucose Value (mg/dL)`           | Float64 | Glucose reading (interpolated where needed)    |

#### Key Differences in Glucose-Only Mode:

- **Removed Fields**: `Event Type`, `Insulin Value (u)`, `Carb Value (grams)`
- **Filtered Data**: Only rows with non-null glucose values are included
- **Simplified Output**: Focuses exclusively on glucose time-series data
- **Smaller File Size**: Reduced data volume for glucose-specific analysis

#### Example Output Comparison:

**Standard Format:**

```csv
sequence_id,Timestamp (YYYY-MM-DDThh:mm:ss),Event Type,Glucose Value (mg/dL),Insulin Value (u),Carb Value (grams)
0,2019-10-14T16:47:37,EGV,55.0,,
0,2019-10-14T16:52:37,EGV,55.0,,
```

**Glucose-Only Format:**

```csv
sequence_id,Timestamp (YYYY-MM-DDThh:mm:ss),Glucose Value (mg/dL)
0,2019-10-14T16:47:37,55.0
0,2019-10-14T16:52:37,55.0
```

## 🎯 Machine Learning Applications

The processed data is optimized for:

- **Time Series Forecasting**: Predict future glucose levels
- **Anomaly Detection**: Identify unusual glucose patterns
- **Sequence Modeling**: LSTM, GRU, or Transformer models
- **Classification**: Event type prediction or risk assessment

### When to Use Glucose-Only Mode

The `--glucose-only` option is particularly useful for:

- **Pure Glucose Analysis**: When you only need glucose time-series data
- **Simplified Models**: For models that don't require insulin or carb information
- **Data Reduction**: When working with large datasets and need to reduce file size
- **Glucose Forecasting**: For models focused exclusively on glucose prediction
- **Research Applications**: When studying glucose patterns without confounding variables

### Use Cases for Each Format

**Standard Format** (all fields):

- Multi-variate time series analysis
- Models that consider insulin and carb effects
- Comprehensive diabetes management applications
- Research requiring full context

**Glucose-Only Format**:

- Pure glucose forecasting models
- Anomaly detection in glucose patterns
- Simplified time series analysis
- Educational or demonstration purposes

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
