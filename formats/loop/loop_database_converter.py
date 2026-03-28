#!/usr/bin/env python3
"""
Loop database converter.

Raw data reality:
  - 6 CGM files × ~19M rows = 111M rows, but only ~9M unique (user, timestamp) records
  - Each file is a full-history snapshot export → massive cross-file duplication
  - Polars cannot sort/dedup 111M rows on 8GB RAM (crashes with access violation)

Solution: DuckDB handles the initial scan + dedup + sort out-of-core (spills to disk
automatically). After dedup the result is ~14M rows total which Polars handles fine.

Cache: per-modality parquet files written to <data_folder>/.loop_cache/.
       Reused across runs while source files are unchanged (mtime comparison).
"""

from pathlib import Path
from typing import List, Optional, Dict, Any, Iterable, Union
import shutil
import polars as pl
import duckdb
from loguru import logger

from formats.database_converters import DatabaseConverter
from formats.glucose_bounds import dexcom_style_bounds

MMOL_TO_MGDL: float = 18.018
CACHE_DIR_NAME: str = ".loop_cache"
_DEDUP_KEYS: List[str] = ["user_id", "timestamp", "event_type"]


class LoopDatabaseConverter(DatabaseConverter):
    """Converter for Loop database using DuckDB for dedup and Polars for per-user iteration."""

    def get_database_name(self) -> str:
        return "Loop Database"

    def consolidate_data(
        self,
        data_folder: Union[str, Path],
        output_file: Optional[Union[str, Path]] = None,
    ) -> pl.DataFrame:
        """
        Consolidate Loop data by iterating per-user frames.
        Kept for backwards compatibility and tests.
        """
        interval_minutes = int(self.config.get("expected_interval_minutes", 5))
        frames: List[pl.DataFrame] = []
        for user_df in self.iter_user_event_frames(data_folder, interval_minutes=interval_minutes):
            frames.append(user_df)

        if not frames:
            raise ValueError(f"No Loop records produced from {data_folder}")

        df = pl.concat(frames, how="diagonal").sort(["user_id", "timestamp"])

        if output_file:
            logger.info(f"Writing consolidated data to: {output_file}")
            df.write_csv(output_file)

        return df

    def iter_user_event_frames(
        self,
        data_folder: Union[str, Path],
        *,
        interval_minutes: int,
    ) -> Iterable[pl.DataFrame]:
        """
        Yield one deduplicated, sorted DataFrame per user.

        First call: DuckDB streams each modality to a parquet cache, then merges them
        into a single combined parquet sorted by user_id (~first 10-20 min).
        Subsequent calls: cache is reused (<5 sec to enumerate users, then
        per-user DuckDB filter reads one user at a time — O(1) memory).
        """
        data_path = Path(data_folder)
        if not data_path.exists():
            raise FileNotFoundError(f"Data folder not found: {data_folder}")

        combined_pq = self._ensure_combined_parquet(data_path)

        # Enumerate users cheaply via DuckDB (no full load)
        con = duckdb.connect()
        user_ids: list[str] = [
            row[0] for row in con.execute(
                f"SELECT DISTINCT user_id FROM read_parquet('{combined_pq.as_posix()}') ORDER BY user_id"
            ).fetchall()
        ]
        con.close()

        start_user_id = self._get_start_with_user_id()
        if start_user_id:
            found = False
            for i, uid in enumerate(user_ids):
                if uid == start_user_id:
                    user_ids = user_ids[i:]
                    found = True
                    break
            if not found:
                logger.warning(f"start_with_user_id '{start_user_id}' not found in Loop database.")

        first_n_users = self.config.get("first_n_users")
        if first_n_users and int(first_n_users) > 0:
            user_ids = user_ids[: int(first_n_users)]

        logger.info(f"Processing {len(user_ids)} Loop users...")

        # Yield one user at a time — DuckDB pushes the user_id filter into parquet
        # row-group statistics, reading only the relevant pages (~O(1) per user).
        pq_path_posix = combined_pq.as_posix()
        for user_id in user_ids:
            con = duckdb.connect()
            user_df = con.execute(
                f"SELECT * FROM read_parquet('{pq_path_posix}') "
                f"WHERE user_id = '{user_id}' ORDER BY timestamp"
            ).pl()
            con.close()

            if len(user_df) == 0:
                continue

            user_df = self._enforce_output_schema(user_df.lazy()).collect()
            user_df = self._apply_glucose_trim_to_frame(user_df)
            if len(user_df) > 0:
                logger.info(f"  User {user_id}: {len(user_df)} records")
                yield user_df

    # ------------------------------------------------------------------
    # Glucose clip: same dexcom config keys (low/high mg/dL) as Dexcom; Loop values are numeric only
    # ------------------------------------------------------------------

    def _apply_glucose_trim_to_frame(self, df: pl.DataFrame) -> pl.DataFrame:
        """
        Clip numeric CGM/BGM glucose to [low_glucose_value, high_glucose_value] (Dexcom
        config keys). Loop data are numeric only; idempotent if SQL already clipped.
        """
        if "glucose_value_mgdl" not in df.columns or "event_type" not in df.columns:
            return df

        low_g, high_g = dexcom_style_bounds(self.config)
        ev = pl.col("event_type")
        g = pl.col("glucose_value_mgdl")
        is_glucose_event = ev.is_in(["EGV", "BGM"])
        g_num = g.cast(pl.Float64, strict=False)
        trimmed = pl.when(g_num.is_not_null()).then(g_num.clip(low_g, high_g)).otherwise(None)
        out_col = pl.when(is_glucose_event).then(trimmed).otherwise(g).cast(pl.Float64, strict=False)

        return df.with_columns(out_col.alias("glucose_value_mgdl"))

    # ------------------------------------------------------------------
    # Cache management
    # ------------------------------------------------------------------

    def _ensure_combined_parquet(self, data_path: Path) -> Path:
        """
        Ensure the fully-deduplicated combined parquet exists and return its path.

        Build order (each step cached):
          1. Per-modality parquets via DuckDB (deduplicated within each modality).
          2. Combined parquet via DuckDB UNION ALL + DISTINCT ON (dedup across modalities).

        DuckDB spills to C:\\Temp automatically — never crashes from memory pressure.
        """
        cache_dir = data_path / CACHE_DIR_NAME
        cache_dir.mkdir(exist_ok=True)

        all_files = list(data_path.glob("**/*.txt"))
        cgm_files   = [f for f in all_files if "cgm"   in f.name.lower()]
        bgm_files   = [f for f in all_files if "bgm"   in f.name.lower()]
        basal_files = [f for f in all_files if "basal" in f.name.lower()]
        bolus_files = [f for f in all_files if "bolus" in f.name.lower()]
        food_files  = [f for f in all_files if "food"  in f.name.lower()]

        logger.info(
            f"Found {len(cgm_files)} CGM files, {len(bgm_files)} BGM, "
            f"{len(basal_files)} Basal, {len(bolus_files)} Bolus, {len(food_files)} Food"
        )

        low_g, high_g = dexcom_style_bounds(self.config)
        logger.info(f"Loop glucose clip: numeric CGM/BGM to [{low_g:g}, {high_g:g}] mg/dL")

        # Each modality definition: (name, source_files, duckdb_select_sql)
        # The SELECT must produce columns: user_id, timestamp (TIMESTAMP), event_type,
        # and any modality-specific value columns.
        modality_defs: List[tuple[str, List[Path], str]] = []

        if cgm_files:
            modality_defs.append((
                "cgm", cgm_files,
                """
                SELECT DISTINCT ON (user_id, timestamp, event_type)
                    CAST(PtID AS VARCHAR) AS user_id,
                    TRY_CAST(UTCDtTm AS TIMESTAMP) AS timestamp,
                    'EGV' AS event_type,
                    CASE
                        WHEN TRY_CAST(CGMVal AS DOUBLE) IS NULL THEN NULL
                        WHEN lower(Units) = 'mmol/l'
                            THEN least({high}, greatest({low},
                                TRY_CAST(CGMVal AS DOUBLE) * {mmol}))
                        ELSE least({high}, greatest({low},
                            TRY_CAST(CGMVal AS DOUBLE)))
                    END AS glucose_value_mgdl
                FROM read_csv({{files}}, delim='|', header=true, ignore_errors=true,
                              null_padding=true, strict_mode=false)
                WHERE PtID IS NOT NULL
                  AND TRY_CAST(UTCDtTm AS TIMESTAMP) IS NOT NULL
                ORDER BY user_id, timestamp
                """.format(mmol=MMOL_TO_MGDL, low=low_g, high=high_g),
            ))

        if bgm_files:
            modality_defs.append((
                "bgm", bgm_files,
                """
                SELECT DISTINCT ON (user_id, timestamp, event_type)
                    CAST(PtID AS VARCHAR) AS user_id,
                    TRY_CAST(UTCDtTm AS TIMESTAMP) AS timestamp,
                    'BGM' AS event_type,
                    CASE
                        WHEN TRY_CAST(BGMVal AS DOUBLE) IS NULL THEN NULL
                        WHEN lower(Units) = 'mmol/l'
                            THEN least({high}, greatest({low},
                                TRY_CAST(BGMVal AS DOUBLE) * {mmol}))
                        ELSE least({high}, greatest({low},
                            TRY_CAST(BGMVal AS DOUBLE)))
                    END AS glucose_value_mgdl
                FROM read_csv({{files}}, delim='|', header=true, ignore_errors=true,
                              null_padding=true, strict_mode=false)
                WHERE PtID IS NOT NULL
                  AND TRY_CAST(UTCDtTm AS TIMESTAMP) IS NOT NULL
                ORDER BY user_id, timestamp
                """.format(mmol=MMOL_TO_MGDL, low=low_g, high=high_g),
            ))

        if basal_files:
            modality_defs.append((
                "basal", basal_files,
                """
                SELECT DISTINCT ON (user_id, timestamp, event_type)
                    CAST(PtID AS VARCHAR) AS user_id,
                    TRY_CAST(UTCDtTm AS TIMESTAMP) AS timestamp,
                    'Basal' AS event_type,
                    CAST(Rate AS VARCHAR) AS basal_rate
                FROM read_csv({files}, delim='|', header=true, ignore_errors=true,
                              null_padding=true, strict_mode=false)
                WHERE PtID IS NOT NULL
                  AND TRY_CAST(UTCDtTm AS TIMESTAMP) IS NOT NULL
                ORDER BY user_id, timestamp
                """,
            ))

        if bolus_files:
            modality_defs.append((
                "bolus", bolus_files,
                """
                SELECT DISTINCT ON (user_id, timestamp, event_type)
                    CAST(PtID AS VARCHAR) AS user_id,
                    TRY_CAST(UTCDtTm AS TIMESTAMP) AS timestamp,
                    'Bolus' AS event_type,
                    CAST(Normal AS VARCHAR) AS fast_acting_insulin_u
                FROM read_csv({files}, delim='|', header=true, ignore_errors=true,
                              null_padding=true, strict_mode=false)
                WHERE PtID IS NOT NULL
                  AND TRY_CAST(UTCDtTm AS TIMESTAMP) IS NOT NULL
                ORDER BY user_id, timestamp
                """,
            ))

        if food_files:
            modality_defs.append((
                "food", food_files,
                """
                SELECT DISTINCT ON (user_id, timestamp, event_type)
                    CAST(PtID AS VARCHAR) AS user_id,
                    TRY_CAST(UTCDtTm AS TIMESTAMP) AS timestamp,
                    'Meal' AS event_type,
                    CAST(CarbsNet AS VARCHAR) AS carb_grams
                FROM read_csv({files}, delim='|', header=true, ignore_errors=true,
                              null_padding=true, strict_mode=false)
                WHERE PtID IS NOT NULL
                  AND TRY_CAST(UTCDtTm AS TIMESTAMP) IS NOT NULL
                ORDER BY user_id, timestamp
                """,
            ))

        if not modality_defs:
            raise ValueError("No valid Loop data found in the provided directory.")

        parquet_files: List[Path] = []
        for name, source_files, sql_template in modality_defs:
            pq_path = cache_dir / f"{name}.parquet"
            if self._cache_is_fresh(pq_path, source_files):
                logger.info(f"  Using cached parquet for {name}: {pq_path.name} "
                            f"({pq_path.stat().st_size / 1024**2:.1f} MB)")
            else:
                logger.info(f"  Building {name} cache from {len(source_files)} files → {pq_path.name} ...")
                self._duckdb_dedup_to_parquet(source_files, sql_template, pq_path)
                logger.info(f"  Done: {pq_path.stat().st_size / 1024**2:.1f} MB")
            parquet_files.append(pq_path)

        logger.info("Reading parquet cache and combining modalities...")
        # Use DuckDB to do the cross-modality combine, dedup, and sort — writing a single
        # combined parquet sorted by user_id. This avoids loading all modalities into Polars
        # simultaneously (which would require ~1 GB RAM and crash on Windows).
        combined_pq = cache_dir / "combined.parquet"
        if not self._cache_is_fresh(combined_pq, parquet_files):
            logger.info("  Building combined parquet (DuckDB merge)...")
            # UNION ALL BY NAME pads missing columns with NULL — critical for combining
            # modality parquets that each have different value columns.
            union_parts = " UNION ALL BY NAME ".join(
                f"SELECT * FROM read_parquet('{p.as_posix()}')" for p in parquet_files
            )
            combine_sql = f"""
            COPY (
                SELECT DISTINCT ON (user_id, timestamp, event_type) *
                FROM ({union_parts})
                ORDER BY user_id, timestamp
            ) TO '{combined_pq.as_posix()}' (FORMAT PARQUET, COMPRESSION SNAPPY)
            """
            # Spill dir on D: alongside the cache — now 32 GB free.
            tmp_dir = cache_dir / "_tmp_combined"
            tmp_dir.mkdir(parents=True, exist_ok=True)
            db_path = cache_dir / "combined.duckdb"
            try:
                con = duckdb.connect(str(db_path))
                con.execute("SET memory_limit = '6GB'")
                con.execute(f"SET temp_directory = '{tmp_dir.as_posix()}'")
                con.execute("SET max_temp_directory_size = '25GiB'")
                con.execute("PRAGMA threads=4")
                con.execute("SET preserve_insertion_order = false")
                con.execute(combine_sql)
                con.close()
            finally:
                for ext in (".duckdb", ".duckdb.wal"):
                    p_db = db_path if ext == ".duckdb" else Path(str(db_path) + ".wal")
                    if p_db.exists():
                        p_db.unlink()
                if tmp_dir.exists():
                    shutil.rmtree(tmp_dir, ignore_errors=True)
            logger.info(f"  Combined parquet: {combined_pq.stat().st_size / 1024**2:.1f} MB")

        logger.info(f"  Combined parquet ready: {combined_pq.stat().st_size / 1024**2:.1f} MB")
        return combined_pq

    def _cache_is_fresh(self, cache_path: Path, source_files: List[Path]) -> bool:
        """True if cache file exists, is a valid parquet (≥12 bytes), and newer than all sources."""
        if not cache_path.exists() or cache_path.stat().st_size < 12:
            return False
        cache_mtime = cache_path.stat().st_mtime
        return all(cache_mtime >= f.stat().st_mtime for f in source_files)

    def _duckdb_dedup_to_parquet(
        self,
        source_files: List[Path],
        sql_template: str,
        output_path: Path,
    ) -> None:
        """
        Use DuckDB to scan source CSV files, deduplicate by (user_id, timestamp, event_type),
        sort, and write the result to a parquet file.

        DuckDB is connected to a temporary on-disk database (db_path) so it can spill
        intermediate sort/hash data to disk automatically when memory is tight.
        The temp DB file is deleted after the query completes.
        """
        files_sql = "[" + ", ".join(f"'{f.as_posix()}'" for f in source_files) + "]"
        select_sql = sql_template.replace("{files}", files_sql)

        copy_sql = f"""
        COPY (
            {select_sql}
        ) TO '{output_path.as_posix()}' (FORMAT PARQUET, COMPRESSION SNAPPY)
        """

        # File-based DuckDB DB so it can spill sort/hash pages to disk.
        # Spill dir lives alongside the parquet output — D: now has 32 GB free.
        db_path = output_path.with_suffix(".duckdb")
        tmp_dir = output_path.parent / f"_tmp_{output_path.stem}"
        tmp_dir.mkdir(parents=True, exist_ok=True)
        try:
            con = duckdb.connect(str(db_path))
            con.execute("SET memory_limit = '6GB'")
            con.execute(f"SET temp_directory = '{tmp_dir.as_posix()}'")
            con.execute("SET max_temp_directory_size = '25GiB'")
            con.execute("PRAGMA threads=4")
            con.execute("SET preserve_insertion_order = false")
            con.execute(copy_sql)
            con.close()
        finally:
            for ext in (".duckdb", ".duckdb.wal"):
                p = output_path.with_suffix(ext) if ext == ".duckdb" else Path(str(output_path.with_suffix("")) + ext)
                if p.exists():
                    p.unlink()
            if tmp_dir.exists():
                shutil.rmtree(tmp_dir, ignore_errors=True)

    def _read_loop_files_lazy(self, files: List[Path], schema: Dict[str, Any]) -> pl.LazyFrame:
        """Read multiple pipe-separated files as a LazyFrame (kept for any external callers)."""
        if not files:
            return pl.LazyFrame()
        ldf = pl.scan_csv(
            files,
            separator="|",
            schema_overrides=schema,
            ignore_errors=True,
            truncate_ragged_lines=True,
        )
        ldf_schema = ldf.collect_schema()
        available_cols = [c for c in schema.keys() if c in ldf_schema.names()]
        if not available_cols:
            return pl.LazyFrame()
        return ldf.select(available_cols)
