"""Tests for dataset downloader functionality."""

import shutil
import tempfile
from pathlib import Path

import pytest

from download import (
    DATASET_TO_FORMAT_FOLDER,
    compute_hash,
    download_file_with_validation,
    get_downloadable_datasets,
    get_figshare_collection_files,
    get_figshare_files,
    get_folder_name,
    get_physionet_credentials,
    get_zenodo_files,
    load_datasets,
    sanitize_folder_name,
)


class TestGetFolderName:
    """Tests for folder name mapping to format converters."""

    def test_mapped_datasets_use_format_folder(self) -> None:
        """Verify datasets with format converters use the mapped folder name."""
        assert get_folder_name("HUPA") == "hupa"
        assert get_folder_name("UCHTT1DM") == "uc_ht"
        assert get_folder_name("Loop System") == "loop"
        assert get_folder_name("T1D-UOM") == "uom"
        assert get_folder_name("Mini-dose Glucagon") == "minidose1"
        assert get_folder_name("Mini-dose Glucagon 1") == "minidose1_exercise"
        assert get_folder_name("AI Ready") == "ai_ready"

    def test_unmapped_datasets_use_sanitized_name(self) -> None:
        """Verify datasets without format converters use sanitized names."""
        assert get_folder_name("Unknown Dataset") == "Unknown_Dataset"
        assert get_folder_name("CGMacros") == "CGMacros"
        assert get_folder_name("ShanghaiT1&2DM") == "ShanghaiT1and2DM"

    def test_all_mappings_exist_in_formats_directory(self) -> None:
        """Verify all mapped folder names correspond to format directories."""
        formats_dir = Path(__file__).parent.parent / "formats"
        expected_format_folders = {"ai_ready", "hupa", "uc_ht", "loop", "uom", "minidose1"}
        
        for folder in expected_format_folders:
            format_path = formats_dir / folder
            assert format_path.exists(), f"Format folder missing: {folder}"


class TestSanitizeFolderName:
    """Tests for folder name sanitization."""

    def test_spaces_replaced_with_underscore(self) -> None:
        assert sanitize_folder_name("My Dataset Name") == "My_Dataset_Name"

    def test_special_characters_removed(self) -> None:
        assert sanitize_folder_name("Test (v1.0)") == "Test_v1.0"
        assert sanitize_folder_name("Data: Analysis") == "Data-_Analysis"

    def test_slashes_replaced(self) -> None:
        assert sanitize_folder_name("Type 1/Type 2") == "Type_1-Type_2"

    def test_ampersand_replaced(self) -> None:
        assert sanitize_folder_name("T1&T2") == "T1andT2"


class TestLoadDatasets:
    """Tests for CSV loading functionality."""

    def test_loads_datasets_from_csv(self) -> None:
        datasets = load_datasets()
        assert len(datasets) > 0

    def test_dataset_has_required_fields(self) -> None:
        datasets = load_datasets()
        required_fields = {"id", "name", "study", "source", "source_url", "download"}
        for ds in datasets:
            assert required_fields.issubset(ds.keys())


class TestGetDownloadableDatasets:
    """Tests for filtering downloadable datasets."""

    def test_returns_only_programmatic_downloads(self) -> None:
        datasets = get_downloadable_datasets()
        for ds in datasets:
            assert ds["download"] == "programmatic"

    def test_returns_only_datasets_with_valid_urls(self) -> None:
        datasets = get_downloadable_datasets()
        for ds in datasets:
            url = ds["source_url"]
            assert url.startswith(("http://", "https://", "s3://", "ftp://"))

    def test_includes_expected_sources(self) -> None:
        datasets = get_downloadable_datasets()
        sources = {ds["source"] for ds in datasets}
        # Should include at least JAEB and Zenodo
        assert "JAEB" in sources
        assert "Zenodo" in sources


class TestZenodoAPI:
    """Tests for Zenodo API integration."""

    @pytest.mark.parametrize(
        "record_url,expected_has_files",
        [
            # T1GDUJA - open access record with files
            ("https://zenodo.org/records/11284018", True),
            # T1DiabetesGranada - restricted access (no files)
            ("https://zenodo.org/records/10050944", False),
        ],
    )
    def test_get_zenodo_files(self, record_url: str, expected_has_files: bool) -> None:
        files = get_zenodo_files(record_url)
        if expected_has_files:
            assert len(files) > 0
            for f in files:
                assert "filename" in f
                assert "url" in f
                assert "size" in f
        else:
            assert len(files) == 0


class TestFigshareAPI:
    """Tests for Figshare API integration."""

    def test_get_figshare_files(self) -> None:
        # ShanghaiT1&2DM dataset (article URL)
        url = "https://figshare.com/articles/dataset/diabetes_datasets_zip/21600933"
        files = get_figshare_files(url)
        assert len(files) > 0
        for f in files:
            assert "filename" in f
            assert "url" in f
            assert "size" in f

    def test_get_figshare_collection_files(self) -> None:
        # Shanghai T1DM collection URL
        url = "https://figshare.com/collections/Shanghai_Type_1_Diabetes_Mellitus_Dataset/6310860"
        files = get_figshare_collection_files(url)
        assert len(files) > 0
        for f in files:
            assert "filename" in f
            assert "url" in f
            assert "size" in f


class TestPhysioNetCredentials:
    """Tests for PhysioNet credentials handling."""

    def test_get_physionet_credentials_returns_tuple(self) -> None:
        # Just verify the function returns a tuple (credentials may or may not be set)
        result = get_physionet_credentials()
        assert isinstance(result, tuple)
        assert len(result) == 2


class TestDownloadWithValidation:
    """Tests for file download with validation."""

    def test_download_small_file_with_checksum(self) -> None:
        """Download a small file from Zenodo and verify checksum."""
        # Get the T1GDUJA record files
        files = get_zenodo_files("https://zenodo.org/records/11284018")
        assert len(files) > 0

        # Find a small file (the PNG)
        small_file = next((f for f in files if f["filename"].endswith(".png")), None)
        assert small_file is not None

        with tempfile.TemporaryDirectory() as tmpdir:
            dest = Path(tmpdir) / small_file["filename"]
            result = download_file_with_validation(
                small_file["url"],
                dest,
                small_file["size"],
                small_file["checksum"],
            )
            assert result is True
            assert dest.exists()
            assert dest.stat().st_size == small_file["size"]


class TestComputeHash:
    """Tests for hash checksum computation."""

    def test_compute_hash_known_content(self) -> None:
        """Verify MD5 computation with known content."""
        with tempfile.NamedTemporaryFile(delete=False) as f:
            f.write(b"Hello, World!")
            tmp_path = Path(f.name)

        try:
            md5 = compute_hash(tmp_path)
            # Known MD5 hash for "Hello, World!"
            assert md5 == "65a8e27d8879283831b664bd8b7f0ad4"
        finally:
            tmp_path.unlink()
