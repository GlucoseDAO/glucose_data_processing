#!/usr/bin/env python3
"""
ShanghaiT1DM / ShanghaiT2DM database converter.

One Excel workbook per recording period, named ``<patient>_<period>_<YYYYMMDD>.xls[x]``
(e.g. ``1002_0_20210504.xls``). A patient may have several periods; they are merged into
one user. Only the CGM column is read (docs/DECISIONS.md D2, formats/shanghai/SHANGHAI.md).
"""

import re
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Union

import polars as pl
from loguru import logger

from formats.database_converters import FILE_MATCHED, FILE_NO_CONVERTER, DatabaseConverter

SHANGHAI_WORKBOOK_PATTERN = re.compile(r"^(?P<patient>\d+)_(?P<period>\d+)_(?P<date>\d{8})\.xlsx?$")
SHANGHAI_DATE_COLUMN = "Date"
# 'CGM (mg / dl)' in most workbooks, 'CGM ' (no unit, same mg/dL range) in two T2DM files.
SHANGHAI_CGM_COLUMN_PREFIX = "CGM"
SHANGHAI_EVENT_TYPE = "EGV"


def find_shanghai_workbooks(root: Path) -> List[Path]:
    """All workbooks under ``root`` whose name follows the patient_period_date pattern, sorted."""
    return sorted(
        p for p in root.glob("**/*.xls*") if p.is_file() and SHANGHAI_WORKBOOK_PATTERN.match(p.name)
    )


def shanghai_cgm_column(columns: List[str]) -> Optional[str]:
    """The CGM column of a Shanghai workbook, or None if it has none."""
    matches = [c for c in columns if c.strip().startswith(SHANGHAI_CGM_COLUMN_PREFIX)]
    return matches[0] if len(matches) == 1 else None


class ShanghaiDatabaseConverter(DatabaseConverter):
    """Converter for the ShanghaiT1DM and ShanghaiT2DM workbooks."""

    def get_database_name(self) -> str:
        return "Shanghai T1DM/T2DM Database"

    def consolidate_data(self, data_folder: Union[str, Path], output_file: Optional[Union[str, Path]] = None) -> pl.DataFrame:
        frames = list(self.iter_user_event_frames(data_folder, interval_minutes=5))
        if not frames:
            raise ValueError(f"No valid data found in {data_folder}: {self.describe_file_report()}")
        df = pl.concat(frames, how="diagonal_relaxed")
        if output_file:
            logger.info(f"Writing consolidated data to: {output_file}")
            df.write_csv(output_file)
        return df

    def iter_user_event_frames(self, data_folder: Union[str, Path], *, interval_minutes: int) -> Iterable[pl.DataFrame]:
        root = Path(data_folder)
        if not root.is_dir():
            raise ValueError(f"Input must be a directory, got: {data_folder}")

        by_patient: Dict[str, List[Path]] = {}
        for workbook in find_shanghai_workbooks(root):
            patient = SHANGHAI_WORKBOOK_PATTERN.match(workbook.name).group("patient")
            by_patient.setdefault(patient, []).append(workbook)
        logger.info(f"Found {sum(map(len, by_patient.values()))} workbooks for {len(by_patient)} patients")

        for patient, workbooks in sorted(by_patient.items()):
            periods = [f for f in (self._read_cgm(w) for w in workbooks) if f is not None]
            if not periods:
                continue
            df = (
                pl.concat(periods)
                .unique("timestamp", keep="first", maintain_order=True)
                .sort("timestamp")
                .with_columns(
                    pl.lit(patient).alias("user_id"),
                    pl.lit(SHANGHAI_EVENT_TYPE).alias("event_type"),
                )
            )
            yield self._enforce_output_schema(df)

    def _read_cgm(self, workbook: Path) -> Optional[pl.DataFrame]:
        sheet = pl.read_excel(workbook)
        cgm_col = shanghai_cgm_column(sheet.columns)
        if cgm_col is None or SHANGHAI_DATE_COLUMN not in sheet.columns:
            self.file_report[FILE_NO_CONVERTER] += 1
            return None
        self.file_report[FILE_MATCHED] += 1
        return sheet.select(
            pl.col(SHANGHAI_DATE_COLUMN).cast(pl.Datetime("us")).alias("timestamp"),
            pl.col(cgm_col).cast(pl.Float64, strict=False).alias("glucose_value_mgdl"),
        ).drop_nulls()
