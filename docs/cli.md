# Command Line Interface (CLI)

The project provides three main CLI tools:
- `glucose-process`: Processes one or multiple glucose databases into a unified ML-ready format.
- `glucose-compare`: Compares two checkpoint CSV files and provides detailed statistics.
- `glucose-download`: Downloads publicly available glucose datasets.

## Usage: glucose-process

```bash
glucose-process [INPUT_FOLDERS]... [OPTIONS]
```

### Arguments

- `INPUT_FOLDERS`: One or more paths to database folders (e.g., `DATA/uom_small`) or ZIP files (for AI-READI).

### Options

| Option | Shorthand | Description |
|--------|-----------|-------------|
| `--config` | `-c` | Path to a YAML configuration file. Auto-loaded from `glucose_config.yaml` if present. |
| `--output` | `-o` | Filename for the final ML-ready CSV (placed in the `OUTPUT/` folder). |
| `--interval` | `-i` | Time discretization interval (minutes). |
| `--gap-max` | `-g` | Max gap size to interpolate (minutes). |
| `--min-length` | `-l` | Minimum sequence length to preserve. |
| `--remove-calibration/--keep-calibration` | | Remove calibration events to create interpolatable gaps (default: enabled). |
| `--calibration-period` | `-p` | Gap duration considered a calibration period (minutes, default: 165). |
| `--remove-after-calibration` | `-r` | Hours of data to remove after a calibration period (default: 24). |
| `--glucose-only` | | Filter output to only include glucose values. |
| `--fixed-frequency/--no-fixed-frequency` | | Enable or disable resampling to fixed time buckets (default: enabled). |
| `--last-step` | | Last processing step to execute (1–7). Omit or use 0 for all steps. |
| `--round-precision` | | Decimal digits for rounding numeric fields. Can be negative (default: 3). |
| `--verbose` | `-v` | Enable detailed logging. |
| `--stats/--no-stats` | | Show or suppress the summary statistics printout (default: shown). |
| `--save-intermediate` | `-s` | Export CSVs after each processing stage. |
| `--first-n-users` | | Limit processing to the first `N` users found. |

### Config auto-loading

If `--config` is not provided, the tool automatically loads `glucose_config.yaml` from the current directory when it exists. CLI arguments always override config file values.

### Output file naming

The output filename is resolved in the following order:

1. **`--output` CLI option** – filename provided by the user, placed in `OUTPUT/`.
2. **Config `output_file` setting** – from the YAML config, placed in `OUTPUT/`.
3. **Folder-name-based** – generated from the input folder/ZIP names joined with underscores and suffixed with `_ml_ready.csv` (e.g., `OUTPUT/hupa_uom_ml_ready.csv`).

## Multi-Database Processing

The CLI supports combining different databases in a single run:

```bash
glucose-process DATA/uom DATA/hupa DATA/dexcom_small -o combined_data.csv
```

The preprocessor automatically:
1. Detects the database type for each input.
2. Tracks global `sequence_id` to prevent collisions.
3. Normalizes all data to the same time resolution and field set.

## Download Tool: glucose-download

```bash
glucose-download [COMMAND] [OPTIONS]
```

| Command | Description |
|---------|-------------|
| `list` | List all datasets available for download. |
| `all` | Download all programmatically accessible datasets. |
| `by-name` | Download a single dataset by name. |
| `by-names` | Download multiple datasets by name. |
| `by-id` | Download a dataset by its numeric ID. |

Common options:

| Option | Description |
|--------|-------------|
| `--force` | Re-download even if the file already exists. |

Examples:

```bash
glucose-download list
glucose-download by-name "HUPA"
glucose-download by-names "HUPA" "T1D-UOM"
glucose-download by-id 14
glucose-download by-name "T1D-UOM" --force
```

Downloaded datasets are saved to the `DATA/` folder with subdirectory names matching their format converters (e.g., `DATA/hupa/`, `DATA/uom/`).

**Note**: Some datasets require credentials:
- **PhysioNet** datasets: Set `PHYSIONET_USERNAME` and `PHYSIONET_PASSWORD` in `.env`.
- **Manual-access** datasets (AI-READI, some JAEB DirecNet studies): Require registration on their respective portals.

## Comparison Tool: glucose-compare

```bash
glucose-compare [FILE1] [FILE2] [OPTIONS]
```

This tool compares two checkpoint files to ensure processing results are consistent.

### Arguments

- `FILE1`: Path to the first checkpoint file.
- `FILE2`: Path to the second checkpoint file.

### Options

| Option | Shorthand | Description |
|--------|-----------|-------------|
| `--key-columns` | `-k` | Key columns for row matching. |
| `--tolerance` | `-t` | Numeric tolerance for approximate matches. |
| `--no-streaming` | | Disable Polars streaming. |

## Processing Statistics

At the end of a successful run, the CLI displays a summary including:
- Total records collected and preserved.
- Number of sequences created and filtered.
- Interpolation and gap statistics.
- Longest and average sequence lengths.
