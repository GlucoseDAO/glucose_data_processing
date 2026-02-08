#!/usr/bin/env python3
"""CLI tool for downloading public glucose datasets."""

import csv
import hashlib
import os
import shutil
import subprocess
import tempfile
import zipfile
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Optional, Union

import requests
import typer
from loguru import logger


class FileFilter(ABC):
    @abstractmethod
    def should_download(self, filename: str) -> bool:
        pass


class HupaFileFilter(FileFilter):
    def __init__(self) -> None:
        import re
        self.pattern = re.compile(r"HUPA(\d+)([a-zA-Z])\.csv", re.IGNORECASE)

    def should_download(self, filename: str) -> bool:
        return bool(self.pattern.fullmatch(filename))


def get_filter_for_dataset(dataset_name: str) -> Optional[FileFilter]:
    if dataset_name == "HUPA":
        return HupaFileFilter()
    return None


# Constants
DATASETS_CSV = Path(__file__).parent / "docs" / "datasets.csv"
DATA_DIR = Path(__file__).parent / "DATA"

# CSV column indices (0-based)
COL_ID = 0  # #
COL_NAME = 1  # Dataset Name
COL_STUDY = 2  # Study
COL_SOURCE = 6  # Source
COL_SOURCE_URL = 7  # Source URL
COL_DOWNLOAD = 9  # Download (programmatic/manual)

# Mapping from dataset names to format folder names (matching formats/ directory)
# This ensures downloaded datasets go into folders that match their format converters
DATASET_TO_FORMAT_FOLDER: dict[str, str] = {
    "AI Ready": "ai_ready",
    "HUPA": "hupa",
    "UCHTT1DM": "uc_ht",
    "Loop System": "loop",
    "T1D-UOM": "uom",
    "Mini-dose Glucagon": "minidose1",
    "Mini-dose Glucagon 1": "minidose1_exercise",  # Second mini-dose study
}

# Timeout for HTTP requests (seconds)
REQUEST_TIMEOUT = 30
DOWNLOAD_TIMEOUT = 600  # 10 minutes for large files

# User agent for HTTP requests
USER_AGENT = "glucose-dataset-downloader/1.0"

app = typer.Typer(
    name="glucose-download",
    help="Download public glucose datasets to DATA folder.",
    no_args_is_help=True,
)


def load_datasets() -> list[dict]:
    """Load datasets from CSV and return list of dataset info dicts."""
    datasets = []
    with open(DATASETS_CSV, newline="", encoding="utf-8") as f:
        reader = csv.reader(f)
        next(reader)  # Skip header row
        
        for row in reader:
            if len(row) < 10:
                continue
            
            if not row[COL_NAME].strip():
                continue
                
            dataset = {
                "id": row[COL_ID].strip(),
                "name": row[COL_NAME].strip(),
                "study": row[COL_STUDY].strip(),
                "source": row[COL_SOURCE].strip() if len(row) > COL_SOURCE else "",
                "source_url": row[COL_SOURCE_URL].strip() if len(row) > COL_SOURCE_URL else "",
                "download": row[COL_DOWNLOAD].strip() if len(row) > COL_DOWNLOAD else "",
            }
            datasets.append(dataset)
    
    return datasets


def get_downloadable_datasets() -> list[dict]:
    """Get datasets that can be downloaded programmatically with valid URLs."""
    all_datasets = load_datasets()
    downloadable = []
    
    for ds in all_datasets:
        if ds["download"] != "programmatic":
            continue
        
        url = ds["source_url"]
        if not url or url == "N/A":
            continue
        
        if not any(url.startswith(proto) for proto in ["http://", "https://", "s3://", "ftp://"]):
            continue
        
        downloadable.append(ds)
    
    return downloadable


def get_folder_name(name: str) -> str:
    """Get folder name for a dataset, using format mapping if available.
    
    If the dataset has a matching format converter, use that folder name.
    Otherwise, generate a safe folder name from the dataset name.
    """
    # Check if dataset has a mapped format folder
    if name in DATASET_TO_FORMAT_FOLDER:
        return DATASET_TO_FORMAT_FOLDER[name]
    
    # Fallback: generate safe folder name
    return sanitize_folder_name(name)


def sanitize_folder_name(name: str) -> str:
    """Convert dataset name to a safe folder name (fallback for unmapped datasets)."""
    safe_name = name.replace(" ", "_").replace("/", "-").replace("\\", "-")
    safe_name = safe_name.replace("(", "").replace(")", "").replace(",", "")
    safe_name = safe_name.replace(":", "-").replace("&", "and")
    while "__" in safe_name:
        safe_name = safe_name.replace("__", "_")
    while "--" in safe_name:
        safe_name = safe_name.replace("--", "-")
    return safe_name.strip("_-")


def compute_hash(file_path: Path, algorithm: str = "md5") -> str:
    """Compute hash of a file using specified algorithm (md5, sha256)."""
    if algorithm == "md5":
        h = hashlib.md5()
    elif algorithm == "sha256":
        h = hashlib.sha256()
    else:
        raise ValueError(f"Unsupported hash algorithm: {algorithm}")
        
    with open(file_path, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()


def get_zenodo_files(record_url: str) -> list[dict]:
    """Get file download info from Zenodo record using API."""
    # Extract record ID from URL
    parts = record_url.rstrip("/").split("/")
    record_id = parts[-1]
    
    api_url = f"https://zenodo.org/api/records/{record_id}"
    logger.info(f"Fetching Zenodo record metadata: {api_url}")
    
    # Use browser-like headers to avoid being blocked
    headers = {
        "Accept": "application/json",
        "User-Agent": "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko)",
    }
    
    response = requests.get(api_url, headers=headers, timeout=REQUEST_TIMEOUT)
    response.raise_for_status()
    
    data = response.json()
    
    # Check if access is restricted
    access_right = data.get("metadata", {}).get("access_right", "open")
    if access_right == "restricted":
        logger.warning("Dataset has restricted access - requires Zenodo login and access request")
        return []
    
    files = []
    for f in data.get("files", []):
        files.append({
            "filename": f["key"],
            "url": f["links"]["self"],
            "size": f["size"],
            "checksum": f.get("checksum", ""),  # Format: "md5:xxxx"
        })
    
    return files


def download_zenodo_record(record_url: str, output_dir: Path) -> bool:
    """Download all files from a Zenodo record."""
    files = get_zenodo_files(record_url)
    if not files:
        logger.error("No files found in Zenodo record (may be restricted)")
        return False
    
    for file_info in files:
        # Flatten path: save to dataset dir directly (e.g. data/uom/file.zip), not sharpic/file.zip
        dest = output_dir / Path(file_info["filename"]).name
        if not download_file_with_validation(
            file_info["url"],
            dest,
            file_info.get("size"),
            file_info.get("checksum"),
        ):
            return False
        
        # Extract ZIP files
        if dest.suffix.lower() == ".zip":
            extract_zip(dest, output_dir)
            dest.unlink()  # Remove ZIP after extraction
    
    return True


def get_figshare_files(article_url: str) -> list[dict]:
    """Get file download info from Figshare article."""
    # Extract article ID: https://figshare.com/articles/dataset/diabetes_datasets_zip/21600933
    parts = article_url.rstrip("/").split("/")
    article_id = parts[-1]
    
    api_url = f"https://api.figshare.com/v2/articles/{article_id}"
    logger.info(f"Fetching Figshare article metadata: {api_url}")
    
    headers = {
        "Accept": "application/json",
        "User-Agent": "glucose-dataset-downloader/1.0",
    }
    
    response = requests.get(api_url, headers=headers, timeout=REQUEST_TIMEOUT)
    response.raise_for_status()
    
    data = response.json()
    files = []
    
    for f in data.get("files", []):
        files.append({
            "filename": f["name"],
            "url": f["download_url"],
            "size": f["size"],
            "checksum": f.get("computed_md5", ""),
        })
    
    return files


def get_mendeley_files(dataset_url: str) -> list[dict]:
    """Get file download info from Mendeley Data using public API."""
    # URL: https://data.mendeley.com/datasets/3hbcscwz44/1
    parts = dataset_url.rstrip("/").split("/")
    version = parts[-1]
    dataset_id = parts[-2]
    
    # Mendeley public API endpoint
    # We need to explicitly request the 'files' field. 
    # Note: specifying version with fields can sometimes return 0 files on Mendeley.
    api_url = f"https://data.mendeley.com/public-api/datasets/{dataset_id}?fields=files"
    logger.info(f"Fetching Mendeley dataset metadata: {api_url}")
    
    headers = {
        "Accept": "application/json",
        "User-Agent": "glucose-dataset-downloader/1.0",
    }
    
    response = requests.get(api_url, headers=headers, timeout=REQUEST_TIMEOUT)
    response.raise_for_status()
    
    data = response.json()
    files = []
    
    for f in data.get("files", []):
        # Handle both old and new Mendeley API structures
        download_url = f.get("download_url")
        size = f.get("size", 0)
        checksum = f.get("checksum", "")
        
        if not download_url and "content_details" in f:
            details = f["content_details"]
            download_url = details.get("download_url")
            if not size:
                size = details.get("size", 0)
            if not checksum:
                # Use sha256 if available, prefixing it so validator knows
                sha256 = details.get("sha256_hash")
                if sha256:
                    checksum = f"sha256:{sha256}"
            
        if not download_url:
            continue
            
        files.append({
            "filename": f["filename"],
            "url": download_url,
            "size": size,
            "checksum": checksum,
        })
    
    return files


def get_figshare_collection_files(collection_url: str) -> list[dict]:
    """Get file download info from Figshare collection (multiple articles)."""
    # Extract collection ID: https://figshare.com/collections/Shanghai_Type_1.../6310860
    parts = collection_url.rstrip("/").split("/")
    collection_id = parts[-1]
    
    api_url = f"https://api.figshare.com/v2/collections/{collection_id}/articles"
    logger.info(f"Fetching Figshare collection metadata: {api_url}")
    
    headers = {
        "Accept": "application/json",
        "User-Agent": "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko)",
    }
    
    response = requests.get(api_url, headers=headers, timeout=REQUEST_TIMEOUT)
    response.raise_for_status()
    
    articles = response.json()
    logger.info(f"Found {len(articles)} articles in collection")
    
    all_files = []
    for article in articles:
        article_id = article["id"]
        article_title = article.get("title", f"article_{article_id}")
        
        # Get article details with file list
        article_api_url = f"https://api.figshare.com/v2/articles/{article_id}"
        response = requests.get(article_api_url, headers=headers, timeout=REQUEST_TIMEOUT)
        response.raise_for_status()
        article_data = response.json()
        
        for f in article_data.get("files", []):
            all_files.append({
                "filename": f["name"],
                "url": f["download_url"],
                "size": f["size"],
                "checksum": f.get("computed_md5", ""),
            })
    
    return all_files


def get_physionet_credentials() -> tuple[Optional[str], Optional[str]]:
    """Get PhysioNet credentials from environment variables or .env file."""
    # Try to load from .env file if present
    env_file = Path(__file__).parent / ".env"
    if env_file.exists():
        for line in env_file.read_text().splitlines():
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            if "=" in line:
                key, value = line.split("=", 1)
                key = key.strip()
                value = value.strip().strip('"').strip("'")
                if key and key not in os.environ:
                    os.environ[key] = value
    
    username = os.environ.get("PHYSIONET_USERNAME")
    password = os.environ.get("PHYSIONET_PASSWORD")
    
    return username, password


def download_physionet_file(
    url: str,
    dest_path: Path,
    username: str,
    password: str,
) -> bool:
    """Download a file from PhysioNet with authentication."""
    logger.info(f"Downloading (authenticated): {url}")
    
    response = requests.get(
        url,
        auth=(username, password),
        stream=True,
        allow_redirects=True,
        timeout=DOWNLOAD_TIMEOUT,
    )
    response.raise_for_status()
    
    downloaded = 0
    with open(dest_path, "wb") as f:
        for chunk in response.iter_content(chunk_size=8192):
            if chunk:
                f.write(chunk)
                downloaded += len(chunk)
    
    logger.success(f"Downloaded: {dest_path.name} ({downloaded:,} bytes)")
    return True


def download_physionet_dataset(url: str, output_dir: Path, dataset_name: str) -> bool:
    """Download dataset from PhysioNet with authentication."""
    username, password = get_physionet_credentials()
    
    if not username or not password:
        logger.error(
            "PhysioNet credentials not found. "
            "Set PHYSIONET_USERNAME and PHYSIONET_PASSWORD environment variables or in .env file"
        )
        return False
    
    # Transform content URL to files URL
    # https://physionet.org/content/cgmacros/1.0.0/ -> https://physionet.org/files/cgmacros/1.0.0/
    files_url = url.replace("/content/", "/files/").rstrip("/")
    
    # Handle specific datasets
    if "cgmacros" in url.lower():
        # CGMacros: Download the main zip file
        zip_url = f"{files_url}/CGMacros_dateshifted365.zip"
        zip_file = output_dir / "CGMacros_dateshifted365.zip"
        
        if not download_physionet_file(zip_url, zip_file, username, password):
            return False
        
        extract_zip(zip_file, output_dir)
        zip_file.unlink()
        return True
    
    elif "big-ideas" in url.lower():
        # BIG IDEAs: Download individual subject files
        for subj_num in range(1, 17):  # subjects 001-016
            file_name = f"Dexcom_{subj_num:03d}.csv"
            file_url = f"{files_url}/{subj_num:03d}/{file_name}"
            output_file = output_dir / file_name
            
            if output_file.exists():
                logger.info(f"File exists, skipping: {file_name}")
                continue
            
            if not download_physionet_file(file_url, output_file, username, password):
                return False
        
        return True
    
    else:
        logger.error(f"PhysioNet download for {dataset_name} not implemented")
        return False


def download_file_with_validation(
    url: str, 
    dest_path: Path, 
    expected_size: Optional[int] = None,
    expected_checksum: Optional[str] = None,
) -> bool:
    """Download a file with optional size and checksum validation."""
    logger.info(f"Downloading: {url}")
    
    # Use browser-like headers to avoid being blocked
    headers = {
        "User-Agent": "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko)",
    }
    
    response = requests.get(url, headers=headers, stream=True, timeout=DOWNLOAD_TIMEOUT)
    response.raise_for_status()
    
    # Get size from headers if not provided
    content_length = response.headers.get("content-length")
    if content_length:
        remote_size = int(content_length)
        logger.info(f"File size: {remote_size:,} bytes")
    else:
        remote_size = None
    
    # Ensure parent directories exist (Zenodo filenames can include path segments, e.g. sharpic/file.zip)
    dest_path.parent.mkdir(parents=True, exist_ok=True)

    # Download with progress
    downloaded = 0
    with open(dest_path, "wb") as f:
        for chunk in response.iter_content(chunk_size=8192):
            f.write(chunk)
            downloaded += len(chunk)
    
    logger.success(f"Downloaded: {dest_path.name} ({downloaded:,} bytes)")
    
    # Validate size
    if expected_size and downloaded != expected_size:
        logger.warning(f"Size mismatch: expected {expected_size:,}, got {downloaded:,}")
    
    # Validate checksum if provided
    if expected_checksum:
        if expected_checksum.startswith("md5:"):
            expected_hash = expected_checksum[4:]
            algorithm = "md5"
        elif expected_checksum.startswith("sha256:"):
            expected_hash = expected_checksum[7:]
            algorithm = "sha256"
        else:
            expected_hash = expected_checksum
            algorithm = "md5"  # default
        
        actual_hash = compute_hash(dest_path, algorithm)
        if actual_hash != expected_hash:
            logger.error(f"Checksum mismatch ({algorithm}): expected {expected_hash}, got {actual_hash}")
            return False
        logger.success(f"Checksum verified ({algorithm}): {actual_hash}")
    
    return True


def clone_github_repo(repo_url: str, dest_dir: Path) -> bool:
    """Clone a GitHub repository."""
    logger.info(f"Cloning GitHub repository: {repo_url}")
    
    result = subprocess.run(
        ["git", "clone", "--depth", "1", repo_url, str(dest_dir)],
        capture_output=True,
        text=True,
    )
    
    if result.returncode != 0:
        logger.error(f"Git clone failed: {result.stderr}")
        return False
    
    # Remove .git folder to save space
    git_dir = dest_dir / ".git"
    if git_dir.exists():
        shutil.rmtree(git_dir)
    
    logger.success(f"Cloned repository to: {dest_dir}")
    return True


def extract_zip(zip_path: Path, extract_dir: Path) -> bool:
    """Extract a ZIP file to the specified directory.
    If the archive has a single top-level directory, its contents are moved
    into extract_dir so the pipeline sees a flat structure (no intermediate folder).
    """
    logger.info(f"Extracting: {zip_path.name}")

    with zipfile.ZipFile(zip_path, "r") as zf:
        zf.extractall(extract_dir)

    # Flatten single top-level folder (e.g. sharpic-ManchesterCSCoordinatedDiabetesStudy-fdbd74f)
    subdirs = [p for p in extract_dir.iterdir() if p.is_dir()]
    if len(subdirs) == 1:
        single_dir = subdirs[0]
        for child in single_dir.iterdir():
            shutil.move(str(child), str(extract_dir / child.name))
        single_dir.rmdir()
        logger.info(f"Flattened top-level folder into: {extract_dir}")

    logger.success(f"Extracted to: {extract_dir}")
    return True


def download_dataset(dataset: dict, data_dir: Path, force: bool = False) -> bool:
    """Download a single dataset based on its source type."""
    name = dataset["name"]
    url = dataset["source_url"]
    source = dataset["source"]
    
    folder_name = get_folder_name(name)
    dataset_dir = data_dir / folder_name
    
    # Get file filter for this dataset if available
    file_filter = get_filter_for_dataset(name)
    if file_filter:
        logger.info(f"Applying file filter for {name}")

    if dataset_dir.exists() and not force:
        logger.warning(f"Dataset folder already exists: {dataset_dir}")
        logger.info("Use --force to redownload")
        return False
    
    # Create/clear dataset directory
    if dataset_dir.exists():
        shutil.rmtree(dataset_dir)
    dataset_dir.mkdir(parents=True, exist_ok=True)
    
    # Handle different sources
    if source == "GitHub":
        return clone_github_repo(url, dataset_dir)
    
    elif source == "Zenodo":
        return download_zenodo_record(url, dataset_dir)
    
    elif source == "Figshare":
        # Detect if it's a collection or article URL
        if "/collections/" in url:
            files = get_figshare_collection_files(url)
        else:
            files = get_figshare_files(url)
        
        if not files:
            logger.error("No files found in Figshare")
            return False
        
        for file_info in files:
            filename = file_info["filename"]
            if file_filter and not file_filter.should_download(filename):
                continue

            dest = dataset_dir / filename
            if not download_file_with_validation(
                file_info["url"],
                dest,
                file_info.get("size"),
                file_info.get("checksum"),
            ):
                return False
            
            if dest.suffix.lower() == ".zip":
                extract_zip(dest, dataset_dir)
                dest.unlink()
        
        return True
    
    elif source == "PhysioNet":
        return download_physionet_dataset(url, dataset_dir, name)
    
    elif source == "Mendeley":
        files = get_mendeley_files(url)
        if not files:
            logger.error("No files found in Mendeley dataset")
            return False
        
        for file_info in files:
            filename = file_info["filename"]
            if file_filter and not file_filter.should_download(filename):
                continue

            dest = dataset_dir / filename
            if not download_file_with_validation(
                file_info["url"],
                dest,
                file_info.get("size"),
                file_info.get("checksum"),
            ):
                return False
            
            if dest.suffix.lower() == ".zip":
                extract_zip(dest, dataset_dir)
                dest.unlink()
        
        return True
    
    else:
        # Direct download (JAEB S3, PhysioNet, Kaggle, etc.)
        # Get filename from URL
        url_path = url.split("?")[0]
        filename = Path(url_path).name
        if not filename:
            filename = f"{folder_name}.zip"
        
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_path = Path(tmpdir) / filename
            
            if not download_file_with_validation(url, tmp_path):
                return False
            
            if filename.lower().endswith(".zip"):
                extract_zip(tmp_path, dataset_dir)
            else:
                shutil.copy(tmp_path, dataset_dir / filename)
        
        return True


def list_datasets(datasets: list[dict]) -> None:
    """Print a formatted list of datasets."""
    typer.echo("\nAvailable datasets for download:")
    typer.echo("-" * 100)
    typer.echo(f"  {'ID':>3}  {'Name':<35} {'Folder':<25} {'Source':<15}")
    typer.echo("-" * 100)
    
    for ds in datasets:
        name = ds["name"]
        source = ds["source"]
        folder = get_folder_name(name)
        typer.echo(f"  [{ds['id']:>2}] {name:<35} {folder:<25} ({source})")
    
    typer.echo("-" * 100)
    typer.echo(f"Total: {len(datasets)} datasets\n")


@app.command("list")
def list_cmd() -> None:
    """List all datasets available for programmatic download."""
    datasets = get_downloadable_datasets()
    list_datasets(datasets)


@app.command("all")
def download_all(
    force: bool = typer.Option(False, "--force", "-f", help="Force redownload even if exists"),
    dry_run: bool = typer.Option(False, "--dry-run", "-n", help="Show what would be downloaded"),
) -> None:
    """Download all datasets with programmatic access."""
    datasets = get_downloadable_datasets()
    
    if not datasets:
        typer.echo("No datasets available for download.")
        raise typer.Exit(1)
    
    typer.echo(f"Found {len(datasets)} datasets available for download\n")
    
    if dry_run:
        list_datasets(datasets)
        typer.echo("Dry run - no files downloaded.")
        return
    
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    
    success_count = 0
    fail_count = 0
    skip_count = 0
    
    for ds in datasets:
        typer.echo(f"\n{'='*60}")
        typer.echo(f"Dataset: {ds['name']}")
        typer.echo(f"Source: {ds['source']}")
        typer.echo("=" * 60)
        
        folder_name = get_folder_name(ds["name"])
        dataset_dir = DATA_DIR / folder_name
        
        if dataset_dir.exists() and not force:
            typer.echo(f"Skipping (already exists): {dataset_dir}")
            skip_count += 1
            continue
        
        if download_dataset(ds, DATA_DIR, force):
            success_count += 1
        else:
            fail_count += 1
    
    typer.echo(f"\n{'='*60}")
    typer.echo("Download Summary:")
    typer.echo(f"  Success: {success_count}")
    typer.echo(f"  Failed:  {fail_count}")
    typer.echo(f"  Skipped: {skip_count}")
    typer.echo("=" * 60)


@app.command("by-names")
def download_by_names(
    names: list[str] = typer.Argument(..., help="One or more dataset names (or partial matches)"),
    force: bool = typer.Option(False, "--force", "-f", help="Force redownload even if exists"),
) -> None:
    """Download multiple datasets by name."""
    datasets = get_downloadable_datasets()
    
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    
    success_count = 0
    fail_count = 0
    
    for name in names:
        typer.echo(f"\nProcessing: {name}")
        name_lower = name.lower()
        matches = [ds for ds in datasets if name_lower in ds["name"].lower()]
        
        if not matches:
            typer.echo(f"  No dataset found matching: {name}")
            fail_count += 1
            continue
        
        if len(matches) > 1:
            typer.echo(f"  Multiple datasets match '{name}':")
            for m in matches:
                typer.echo(f"    - {m['name']}")
            typer.echo("  Please provide a more specific name for this match.")
            fail_count += 1
            continue
        
        dataset = matches[0]
        typer.echo(f"  Found dataset: {dataset['name']}")
        typer.echo(f"  Source: {dataset['source']}")
        
        if download_dataset(dataset, DATA_DIR, force):
            typer.echo("  Download complete!")
            success_count += 1
        else:
            typer.echo("  Download failed!")
            fail_count += 1

    typer.echo(f"\n{'='*60}")
    typer.echo("Summary:")
    typer.echo(f"  Success: {success_count}")
    typer.echo(f"  Failed:  {fail_count}")
    typer.echo("=" * 60)
    
    if fail_count > 0:
        raise typer.Exit(1)


@app.command("by-name")
def download_by_name(
    name: str = typer.Argument(..., help="Dataset name (or partial match)"),
    force: bool = typer.Option(False, "--force", "-f", help="Force redownload even if exists"),
) -> None:
    """Download a specific dataset by name."""
    datasets = get_downloadable_datasets()
    
    name_lower = name.lower()
    matches = [ds for ds in datasets if name_lower in ds["name"].lower()]
    
    if not matches:
        typer.echo(f"No dataset found matching: {name}")
        typer.echo("\nAvailable datasets:")
        list_datasets(datasets)
        raise typer.Exit(1)
    
    if len(matches) > 1:
        typer.echo(f"Multiple datasets match '{name}':")
        list_datasets(matches)
        typer.echo("Please provide a more specific name.")
        raise typer.Exit(1)
    
    dataset = matches[0]
    typer.echo(f"Found dataset: {dataset['name']}")
    typer.echo(f"Source: {dataset['source']}")
    typer.echo(f"URL: {dataset['source_url']}\n")
    
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    
    if download_dataset(dataset, DATA_DIR, force):
        typer.echo("\nDownload complete!")
    else:
        typer.echo("\nDownload failed!")
        raise typer.Exit(1)


@app.command("by-id")
def download_by_id(
    dataset_id: int = typer.Argument(..., help="Dataset ID number"),
    force: bool = typer.Option(False, "--force", "-f", help="Force redownload even if exists"),
) -> None:
    """Download a specific dataset by ID number."""
    datasets = get_downloadable_datasets()
    
    matches = [ds for ds in datasets if ds["id"] == str(dataset_id)]
    
    if not matches:
        typer.echo(f"No downloadable dataset found with ID: {dataset_id}")
        typer.echo("\nAvailable datasets:")
        list_datasets(datasets)
        raise typer.Exit(1)
    
    dataset = matches[0]
    typer.echo(f"Found dataset: {dataset['name']}")
    typer.echo(f"Source: {dataset['source']}")
    typer.echo(f"URL: {dataset['source_url']}\n")
    
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    
    if download_dataset(dataset, DATA_DIR, force):
        typer.echo("\nDownload complete!")
    else:
        typer.echo("\nDownload failed!")
        raise typer.Exit(1)


if __name__ == "__main__":
    app()
