# Glucose Data Preprocessing for Machine Learning

A professional pipeline for normalizing and preparing continuous glucose monitoring (CGM) data for time-series machine learning models.

## 🏛️ System Architecture

The system is designed as a modular pipeline with specialized components for each transformation stage.

```mermaid
classDiagram
    class GlucoseMLPreprocessor {
        +process()
        +process_multiple_databases()
    }
    class DatabaseDetector {
        +detect_database_type()
    }
    class DatabaseConverter {
        <<interface>>
        +consolidate_data()
        +iter_user_event_frames()
    }
    class SequenceGapDetector {
        +detect_gaps_and_sequences()
    }
    class SequenceInterpolator {
        +interpolate_missing_values()
    }
    class SequenceFilterStep {
        +filter_sequences_by_length()
        +filter_glucose_only()
    }
    class FixedFrequencyGenerator {
        +create_fixed_frequency_data()
    }
    class MLDataPreparer {
        +prepare_ml_data()
    }

    GlucoseMLPreprocessor --> DatabaseDetector
    GlucoseMLPreprocessor --> DatabaseConverter
    GlucoseMLPreprocessor --> SequenceGapDetector
    GlucoseMLPreprocessor --> SequenceInterpolator
    GlucoseMLPreprocessor --> SequenceFilterStep
    GlucoseMLPreprocessor --> FixedFrequencyGenerator
    GlucoseMLPreprocessor --> MLDataPreparer
    DatabaseConverter <|-- UoMDatabaseConverter
    DatabaseConverter <|-- DexcomDatabaseConverter
    DatabaseConverter <|-- AIReadyDatabaseConverter
```

## 🛠️ Processing Pipeline

The preprocessor executes the following steps in sequence:

1.  **Consolidation**: Normalizes various database formats (CSV, JSON, ZIP) into a standardized multi-user event stream.
2.  **Gap Detection**: Identifies time gaps exceeding the threshold and splits data into contiguous sequences.
3.  **Smart Interpolation**: Performs linear interpolation on "continuous" fields for small gaps while preserving "occasional" events.
4.  **Length Filtering**: Discards sequences that do not meet the minimum required length for ML training.
5.  **Fixed-Frequency Generation**: Resamples sequences to a consistent time interval (e.g., 5 minutes) using averaging for continuous fields and bucket-shifting for events.
6.  **ML Preparation**: Applies final schema constraints and exports the ML-ready dataset.

## 📂 Project Structure

- `glucose_cli.py`: Primary entry point for the application.
- `glucose_ml_preprocessor.py`: Orchestration class for the pipeline.
- `formats/`: Database-specific converters and [Schema Definitions](docs/schemas.md).
- `processing/`: Component logic for gap detection, interpolation, and resampling.
- `docs/`: Detailed documentation for [Configuration](docs/config.md), [CLI Commands](docs/cli.md), and [Schemas](docs/schemas.md).

## 📊 Supported Datasets

The project supports multiple CGM data formats. See [docs/datasets.csv](docs/datasets.csv) for a comprehensive list of 50+ public glucose datasets with download links.

### Datasets with Format Converters

| Format | Dataset | Description |
|--------|---------|-------------|
| `uom` | [T1D-UOM](https://zenodo.org/records/15806142) | University of Manchester multi-modality T1D dataset |
| `hupa` | [HUPA](https://data.mendeley.com/datasets/3hbcscwz44/1) | CGM with heart rate, steps, meals, insulin |
| `uc_ht` | [UCHTT1DM](https://github.com/fisiologiacuantitativauc/UC_HT_T1DM) | Type 1 + healthy controls with CGM, HR, steps |
| `loop` | [Loop System](https://public.jaeb.org) | DIY Loop insulin delivery observational study |
| `minidose1` | [Mini-dose Glucagon](https://public.jaeb.org) | Mini-dose glucagon for hypoglycemia prevention |
| `ai_ready` | [AI-READI](https://aireadi.org/) | Comprehensive health dataset (manual access) |
| `dexcom` | Dexcom G6 | Standardized export format from Dexcom receivers |
| `libre3` | FreeStyle Libre 3 | Abbott's CGM data format |
| `medtronic` | Medtronic | Medtronic pump/CGM export format |

### Downloading Datasets

Use the built-in download tool to fetch publicly available datasets:

```bash
# List all available datasets for download
glucose-download list

# Download a specific dataset by name
glucose-download by-name "HUPA"

# Download multiple datasets by name
glucose-download by-names "HUPA" "T1D-UOM"

# Download a specific dataset by ID
glucose-download by-id 14

# Download all programmatically accessible datasets
glucose-download all

# Force redownload even if exists
glucose-download by-name "T1D-UOM" --force
```

Downloaded datasets are saved to the `DATA/` folder with folder names matching their format converters (e.g., `DATA/hupa/`, `DATA/uom/`).

**Note**: Some datasets require credentials:
- **PhysioNet** datasets (CGMacros, BIG IDEAs): Set `PHYSIONET_USERNAME` and `PHYSIONET_PASSWORD` in `.env`
- **Manual access** datasets (AI-READI, some JAEB DirecNet studies): Require registration on their respective portals

## 📦 Installation

This project uses [uv](https://github.com/astral-sh/uv), a fast Python package installer and resolver.

### Installing uv

**Windows (PowerShell):**
```powershell
powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"
```

**macOS and Linux:**
```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

After installation, **restart your terminal** to make `uv` available in your PATH. The installer will automatically configure your PATH for you.

**Verify installation:**
```bash
uv --version
```

You should see the version number (e.g., `uv 0.x.x`).

For alternative installation methods (including package managers), see the [official uv documentation](https://github.com/astral-sh/uv#installation).

### Setting up the project

Once `uv` is installed, sync the project dependencies:

```bash
uv sync
```

**What this command does:**
- Creates a virtual environment (if needed) for the project
- Reads `pyproject.toml` to determine required dependencies
- Downloads and installs all Python packages needed by the project
- Makes the project ready to run without manual dependency management

This is typically only needed once after cloning the repository or when dependencies change.

## 🚀 Quick Start

### Basic Usage

After installation, you can use the following commands:

```bash
# Process a single dataset (output saved to OUTPUT folder automatically)
glucose-process <path/to/your/data>

# Process with custom output filename
glucose-process <path/to/your/data> -o my_custom_output.csv

# Combine multiple databases
glucose-process <path/to/dataset1> <path/to/dataset2>

# Compare two checkpoint files
glucose-compare checkpoint1.csv checkpoint2.csv
```

**Note**: You can also use `uv run glucose-process ...` if you prefer not to install the package globally.

**Command explanations:**

1. **`glucose-process <input> [-o <output>]`**: Processes glucose monitoring data through the ML preprocessing pipeline. 
   - `<input>`: Path to your input data folder (CSV files) or ZIP file (for AI-READY format)
   - `-o <output>`: (Optional) Custom output filename. If not provided, filename is resolved based on config or database type.
   - **Output location**: All output files are automatically saved to the `OUTPUT/` folder in the project root.
   - **Automatic naming**: When `-o` is not specified, the output filename is resolved using the following priority:
     1. Configuration file `output_file` setting.
     2. Folder-name-based: input folder/ZIP names joined with underscores and suffixed with `_ml_ready.csv` (e.g., `OUTPUT/uom_ml_ready.csv`).
   - The command automatically detects the database format (UoM, Dexcom, AI-READY, Libre3) and applies the appropriate conversion.
   - Multiple input paths can be provided to combine datasets from different sources.

2. **`glucose-compare <file1> <file2>`**: Compares two checkpoint CSV files and provides detailed statistics on schema, sequences, and values.

### Expected Output

When running the commands above, you should see output similar to:

**For `uv sync`:**
```
Resolved X packages in Yms
Downloaded X packages in Yms
Installed X packages in Yms
```

**For `glucose-process <input>`:**
```
Processing completed successfully!
Output: X,XXX records in XX sequences
Saved to: OUTPUT/uom_ml_ready.csv

Summary:
   Date range: YYYY-MM-DD to YYYY-MM-DD
   Longest sequence: X,XXX records
   Average sequence: XXX.X records
   Data preserved: XX.X% (X,XXX/X,XXX records)
   Gaps processed: XX gaps
   Data points created: XXX points
   Field interpolations: XXX values
   Sequences filtered: X removed
```

The output CSV file will be saved in the `OUTPUT/` folder and contain ML-ready glucose data with:
- Fixed-frequency time intervals (default: 5 minutes)
- Interpolated continuous fields
- Filtered sequences meeting minimum length requirements
- Standardized schema across all supported database formats

**Note**: The `OUTPUT/` folder is automatically created if it doesn't exist. All processed files are saved there to keep the project root clean.

For advanced configuration, refer to the [Pipeline Configuration](docs/config.md) and [CLI Documentation](docs/cli.md).
