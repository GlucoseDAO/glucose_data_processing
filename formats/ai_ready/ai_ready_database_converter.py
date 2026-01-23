#!/usr/bin/env python3
"""
AI-READI database converter (zip-backed).

Reads AI-READI modality files directly from `ai-ready.zip` without extracting and
normalizes them into the project's standard field names.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
import io
import json
from pathlib import Path
from typing import Any, Iterable, Optional, Union
import zipfile

import polars as pl
from loguru import logger

from formats.database_converters import DatabaseConverter


@dataclass(frozen=True)
class _AIReadyZipLayout:
    dataset_root: str

    def participants_tsv(self) -> str:
        return f"{self.dataset_root}participants.tsv"

    def dexcom_cgm_json(self, user_id: str) -> str:
        return (
            f"{self.dataset_root}"
            f"wearable_blood_glucose/continuous_glucose_monitoring/dexcom_g6/{user_id}/{user_id}_DEX.json"
        )

    def garmin_file(self, modality: str, user_id: str, suffix: str) -> str:
        return (
            f"{self.dataset_root}"
            f"wearable_activity_monitor/{modality}/garmin_vivosmart5/{user_id}/{user_id}_{suffix}.json"
        )


def _find_dataset_root(zip_ref: zipfile.ZipFile) -> str:
    """
    Find the prefix inside the zip that ends with `/dataset/`.
    Falls back to empty string (meaning zip root).
    """
    # AI-READI zips typically contain something like:
    # ai-ready/downloads/<uuid>/dataset/...
    for name in zip_ref.namelist():
        if name.endswith("/dataset/"):
            return name
    # Some zips may not include the explicit directory entry; search for participants.tsv
    for name in zip_ref.namelist():
        if name.endswith("/dataset/participants.tsv"):
            return name[: -len("participants.tsv")]
    return ""


def _json_load_from_zip(zip_ref: zipfile.ZipFile, member_path: str) -> dict[str, Any]:
    with zip_ref.open(member_path) as f:
        return json.load(f)


def _dig(obj: Any, path: str) -> Any:
    cur: Any = obj
    for part in path.split("."):
        if isinstance(cur, dict) and part in cur:
            cur = cur[part]
        else:
            return None
    return cur


def _parse_timestamp_to_naive_utc(ts: str) -> Optional[datetime]:
    """
    Parse ISO-like timestamps found in AI-READI into naive UTC datetime.

    Examples:
      - 2025-03-17T19:54:36Z
      - 2025-03-17T19:54:36+00:00
      - 2025-03-17 19:12:30
    """
    if not ts:
        return None
    ts = ts.strip()
    if not ts:
        return None
    # Normalize Zulu suffix to ISO offset
    if ts.endswith("Z"):
        ts = ts[:-1] + "+00:00"
    try:
        dt = datetime.fromisoformat(ts)
    except ValueError:
        # Last resort: try common format without offset
        try:
            dt = datetime.strptime(ts, "%Y-%m-%dT%H:%M:%S")
        except ValueError:
            try:
                dt = datetime.strptime(ts, "%Y-%m-%d %H:%M:%S")
            except ValueError:
                return None
    if dt.tzinfo is not None:
        dt = dt.astimezone(timezone.utc).replace(tzinfo=None)
    return dt


class AIReadyDatabaseConverter(DatabaseConverter):
    """
    Zip-backed converter for the AI-READY dataset.

    Produces a multi-user event stream with standard field names. This initial
    implementation returns an in-memory DataFrame; a streaming path is added
    separately (see todo: streaming-processing).
    """

    def get_database_name(self) -> str:
        return "AI-READY (zip) Database"

    def consolidate_data(self, data_folder: Union[str, Path], output_file: Optional[Union[str, Path]] = None) -> pl.DataFrame:
        zip_path = Path(data_folder)
        if not zip_path.exists():
            raise FileNotFoundError(f"AI-READY zip not found: {data_folder}")
        if zip_path.suffix.lower() != ".zip":
            raise ValueError(f"AI-READY input must be a .zip file path, got: {data_folder}")

        # NOTE: consolidate_data is kept for backwards compatibility, but AI-READY is expected
        # to be processed via a streaming path in the preprocessor. For safety, we still support
        # limiting users via config and resampling to the preprocessor interval if present.
        interval_minutes = int(self.config.get("expected_interval_minutes", 5))

        frames: list[pl.DataFrame] = []
        for user_df in self.iter_user_event_frames(zip_path, interval_minutes=interval_minutes):
            frames.append(user_df)

        if not frames:
            raise ValueError("No AI-READY records produced (check zip contents / selected users).")

        df = pl.concat(frames).sort(["user_id", "timestamp"])
        if output_file:
            df.write_csv(output_file)
        return df

    def iter_user_event_frames(self, data_folder: Union[str, Path], *, interval_minutes: int) -> Iterable[pl.DataFrame]:
        """
        Yield per-user, resampled event frames from the AI-READY zip.

        The output is a wide, time-binned DataFrame per user at `interval_minutes` granularity,
        with one row per time bucket and standard field names as columns.
        """
        zip_path = Path(data_folder)
        with zipfile.ZipFile(zip_path, "r") as zip_ref:
            layout = _AIReadyZipLayout(dataset_root=_find_dataset_root(zip_ref))
            participants = self._read_participants(zip_ref, layout)

            user_ids = sorted(participants.keys(), key=lambda x: int(x) if x.isdigit() else x)
            
            # Apply start_with_user_id skipping if specified
            start_user_id = self._get_start_with_user_id()
            if start_user_id:
                start_index = 0
                found = False
                for i, uid in enumerate(user_ids):
                    if uid == start_user_id:
                        start_index = i
                        found = True
                        break
                if found:
                    logger.info(f"Skipping AI-READY users before {start_user_id} (found at index {start_index})")
                    user_ids = user_ids[start_index:]
                else:
                    logger.info(f"Warning: start_with_user_id '{start_user_id}' not found in AI-READY database.")

            # Apply first_n_users filtering if specified
            first_n_users = self.config.get("first_n_users")
            if first_n_users and int(first_n_users) > 0:
                user_ids = user_ids[: int(first_n_users)]

            logger.info(f"Processing {len(user_ids)} AI-READY users...")

            for user_id in user_ids:
                df = self._extract_user_frame(zip_ref, layout, user_id, participants[user_id], interval_minutes=interval_minutes)
                if len(df) > 0:
                    yield df

    def _read_participants(
        self, zip_ref: zipfile.ZipFile, layout: _AIReadyZipLayout
    ) -> dict[str, dict[str, Any]]:
        """
        Read participants.tsv and return mapping user_id -> participant metadata.
        """
        path = layout.participants_tsv()
        with zip_ref.open(path) as f:
            raw = f.read()
        df = pl.read_csv(
            io.BytesIO(raw),
            separator="\t",
            infer_schema_length=10_000,
            ignore_errors=True,
        )
        if "person_id" not in df.columns:
            raise ValueError("participants.tsv missing required column: person_id")

        # Normalize to string IDs.
        df = df.with_columns(pl.col("person_id").cast(pl.Utf8).alias("person_id"))

        cols_keep = [c for c in ["person_id", "clinical_site", "study_group", "age", "study_visit_date", "recommended_split"] if c in df.columns]
        df = df.select(cols_keep)

        out: dict[str, dict[str, Any]] = {}
        for row in df.iter_rows(named=True):
            pid = str(row["person_id"])
            out[pid] = row
        return out

    def _extract_user_rows(
        self,
        zip_ref: zipfile.ZipFile,
        layout: _AIReadyZipLayout,
        user_id: str,
        meta: dict[str, Any],
    ) -> list[dict[str, Any]]:
        rows: list[dict[str, Any]] = []

        # Dexcom CGM
        rows.extend(self._extract_cgm(zip_ref, layout, user_id, meta))

        # Garmin modalities
        rows.extend(self._extract_garmin_heart_rate(zip_ref, layout, user_id, meta))
        rows.extend(self._extract_garmin_stress(zip_ref, layout, user_id, meta))
        rows.extend(self._extract_garmin_calories(zip_ref, layout, user_id, meta))
        rows.extend(self._extract_garmin_steps(zip_ref, layout, user_id, meta))
        rows.extend(self._extract_garmin_sleep(zip_ref, layout, user_id, meta))
        rows.extend(self._extract_garmin_respiratory_rate(zip_ref, layout, user_id, meta))
        rows.extend(self._extract_garmin_oxygen_saturation(zip_ref, layout, user_id, meta))

        return rows

    def _extract_user_frame(
        self,
        zip_ref: zipfile.ZipFile,
        layout: _AIReadyZipLayout,
        user_id: str,
        meta: dict[str, Any],
        *,
        interval_minutes: int,
    ) -> pl.DataFrame:
        """
        Extract and resample all supported modalities for a single user into a single wide frame.
        """
        interval = f"{int(interval_minutes)}m"

        frames: list[pl.DataFrame] = []

        cgm = self._extract_cgm_df(zip_ref, layout, user_id)
        if cgm is not None:
            frames.append(self._resample(cgm, interval, agg="last"))

        heart_rate = self._extract_series_df(
            zip_ref,
            layout.garmin_file("heart_rate", user_id, "heartrate"),
            records_path="body.heart_rate",
            timestamp_path="effective_time_frame.date_time",
            value_path="heart_rate.value",
            value_col="heart_rate",
        )
        if heart_rate is not None:
            frames.append(self._resample(heart_rate, interval, agg="mean"))

        stress = self._extract_series_df(
            zip_ref,
            layout.garmin_file("stress", user_id, "stress"),
            records_path="body.stress",
            timestamp_path="effective_time_frame.date_time",
            value_path="stress.value",
            value_col="stress_level",
        )
        if stress is not None:
            frames.append(self._resample(stress, interval, agg="mean"))

        calories = self._extract_series_df(
            zip_ref,
            layout.garmin_file("physical_activity_calorie", user_id, "calorie"),
            records_path="body.activity",
            timestamp_path="effective_time_frame.date_time",
            value_path="calories_value.value",
            value_col="active_kcal",
        )
        if calories is not None:
            frames.append(self._resample(calories, interval, agg="sum"))

        steps = self._extract_series_df(
            zip_ref,
            layout.garmin_file("physical_activity", user_id, "activity"),
            records_path="body.activity",
            timestamp_path="effective_time_frame.time_interval.start_date_time",
            value_path="base_movement_quantity.value",
            value_col="step_count",
        )
        if steps is not None:
            frames.append(self._resample(steps, interval, agg="sum"))

        rr = self._extract_series_df(
            zip_ref,
            layout.garmin_file("respiratory_rate", user_id, "respiratoryrate"),
            records_path="body.breathing",
            timestamp_path="effective_time_frame.date_time",
            value_path="respiratory_rate.value",
            value_col="respiratory_rate",
        )
        if rr is not None:
            frames.append(self._resample(rr, interval, agg="mean"))

        ox = self._extract_series_df(
            zip_ref,
            layout.garmin_file("oxygen_saturation", user_id, "oxygensaturation"),
            records_path="body.breathing",
            timestamp_path="effective_time_frame.date_time",
            value_path="oxygen_saturation.value",
            value_col="oxygen_saturation_percent",
        )
        if ox is not None:
            frames.append(self._resample(ox, interval, agg="mean"))

        sleep = self._extract_sleep_df(zip_ref, layout, user_id)
        if sleep is not None:
            frames.append(self._resample(sleep, interval, agg="last"))

        if not frames:
            return pl.DataFrame({"user_id": [], "timestamp": [], "event_type": []})

        df = self._outer_join_all(frames)
        df = df.with_columns(
            pl.lit(int(user_id) if user_id.isdigit() else user_id).alias("user_id"),
            pl.lit("AI_READY").alias("event_type"),
            pl.lit(meta.get("clinical_site", "")).alias("clinical_site"),
            pl.lit(meta.get("study_group", "")).alias("study_group"),
            pl.lit(meta.get("recommended_split", "")).alias("recommended_split"),
            pl.lit(meta.get("age", "")).alias("age"),
            pl.lit(meta.get("study_visit_date", "")).alias("study_visit_date"),
        )
        return df.sort("timestamp")

    def _extract_cgm_df(
        self, zip_ref: zipfile.ZipFile, layout: _AIReadyZipLayout, user_id: str
    ) -> Optional[pl.DataFrame]:
        member = layout.dexcom_cgm_json(user_id)
        try:
            obj = _json_load_from_zip(zip_ref, member)
        except KeyError:
            return None

        records = _dig(obj, "body.cgm")
        if not isinstance(records, list):
            return None

        ts_list: list[datetime] = []
        val_list: list[float] = []
        for rec in records:
            if not isinstance(rec, dict):
                continue
            ts = _dig(rec, "effective_time_frame.time_interval.start_date_time")
            bg = _dig(rec, "blood_glucose.value")
            if not isinstance(ts, str) or bg is None:
                continue
            dt = _parse_timestamp_to_naive_utc(ts)
            if dt is None:
                continue
            try:
                val = float(bg)
            except Exception:
                continue
            ts_list.append(dt)
            val_list.append(val)

        if not ts_list:
            return None
        return pl.DataFrame({"timestamp": ts_list, "glucose_value_mgdl": val_list})

    def _extract_sleep_df(
        self, zip_ref: zipfile.ZipFile, layout: _AIReadyZipLayout, user_id: str
    ) -> Optional[pl.DataFrame]:
        member = layout.garmin_file("sleep", user_id, "sleep")
        try:
            obj = _json_load_from_zip(zip_ref, member)
        except KeyError:
            return None

        records = _dig(obj, "body.sleep")
        if not isinstance(records, list):
            return None

        ts_list: list[datetime] = []
        lvl_list: list[str] = []
        for rec in records:
            if not isinstance(rec, dict):
                continue
            ts = _dig(rec, "effective_time_frame.time_interval.start_date_time")
            lvl = rec.get("sleep_stage_state")
            if not isinstance(ts, str) or not isinstance(lvl, str):
                continue
            dt = _parse_timestamp_to_naive_utc(ts)
            if dt is None:
                continue
            ts_list.append(dt)
            lvl_list.append(lvl)

        if not ts_list:
            return None
        return pl.DataFrame({"timestamp": ts_list, "sleep_level": lvl_list})

    def _extract_series_df(
        self,
        zip_ref: zipfile.ZipFile,
        member: str,
        *,
        records_path: str,
        timestamp_path: str,
        value_path: str,
        value_col: str,
    ) -> Optional[pl.DataFrame]:
        try:
            obj = _json_load_from_zip(zip_ref, member)
        except KeyError:
            return None

        records = _dig(obj, records_path)
        if not isinstance(records, list):
            return None

        ts_list: list[datetime] = []
        val_list: list[float] = []
        for rec in records:
            if not isinstance(rec, dict):
                continue
            ts = _dig(rec, timestamp_path)
            val = _dig(rec, value_path)
            if not isinstance(ts, str) or val is None:
                continue
            dt = _parse_timestamp_to_naive_utc(ts)
            if dt is None:
                continue
            try:
                fv = float(val)
            except Exception:
                continue
            ts_list.append(dt)
            val_list.append(fv)

        if not ts_list:
            return None
        return pl.DataFrame({"timestamp": ts_list, value_col: val_list})

    def _resample(self, df: pl.DataFrame, every: str, *, agg: str) -> pl.DataFrame:
        """
        Resample a time series DataFrame to fixed buckets using a simple aggregation.
        """
        if len(df) == 0:
            return df
        if df.schema.get("timestamp") != pl.Datetime:
            df = df.with_columns(pl.col("timestamp").cast(pl.Datetime, strict=False))

        value_cols = [c for c in df.columns if c != "timestamp"]
        if not value_cols:
            return df

        truncated = df.with_columns(pl.col("timestamp").dt.truncate(every).alias("timestamp"))
        agg_exprs: list[pl.Expr] = []
        for c in value_cols:
            if agg == "sum":
                agg_exprs.append(pl.col(c).sum().alias(c))
            elif agg == "last":
                agg_exprs.append(pl.col(c).last().alias(c))
            else:
                agg_exprs.append(pl.col(c).mean().alias(c))

        return truncated.group_by("timestamp").agg(agg_exprs).sort("timestamp")

    def _outer_join_all(self, frames: list[pl.DataFrame]) -> pl.DataFrame:
        """
        Outer-join a list of frames on timestamp to build a wide time-bucket table.
        """
        out = frames[0]
        for df in frames[1:]:
            out = out.join(df, on="timestamp", how="outer_coalesce")
        return out

    def _base_row(self, user_id: str, meta: dict[str, Any], *, event_type: str, timestamp: str) -> dict[str, Any]:
        return {
            "user_id": user_id,
            "timestamp": timestamp,
            "event_type": event_type,
            "clinical_site": meta.get("clinical_site", ""),
            "study_group": meta.get("study_group", ""),
            "recommended_split": meta.get("recommended_split", ""),
            "age": meta.get("age", ""),
            "study_visit_date": meta.get("study_visit_date", ""),
        }

    def _extract_cgm(
        self, zip_ref: zipfile.ZipFile, layout: _AIReadyZipLayout, user_id: str, meta: dict[str, Any]
    ) -> list[dict[str, Any]]:
        member = layout.dexcom_cgm_json(user_id)
        try:
            obj = _json_load_from_zip(zip_ref, member)
        except KeyError:
            return []

        records = _dig(obj, "body.cgm")
        if not isinstance(records, list):
            return []

        out: list[dict[str, Any]] = []
        for rec in records:
            if not isinstance(rec, dict):
                continue
            ts = _dig(rec, "effective_time_frame.time_interval.start_date_time")
            if not isinstance(ts, str) or not ts:
                continue
            bg = _dig(rec, "blood_glucose.value")
            row = self._base_row(user_id, meta, event_type=str(rec.get("event_type", "EGV")), timestamp=ts)
            if bg is not None:
                row["glucose_value_mgdl"] = bg
            out.append(row)
        return out

    def _extract_garmin_heart_rate(
        self, zip_ref: zipfile.ZipFile, layout: _AIReadyZipLayout, user_id: str, meta: dict[str, Any]
    ) -> list[dict[str, Any]]:
        member = layout.garmin_file("heart_rate", user_id, "heartrate")
        return self._extract_simple_value_series(
            zip_ref,
            member,
            user_id,
            meta,
            event_type="HeartRate",
            records_path="body.heart_rate",
            timestamp_path="effective_time_frame.date_time",
            value_path="heart_rate.value",
            output_field="heart_rate",
        )

    def _extract_garmin_stress(
        self, zip_ref: zipfile.ZipFile, layout: _AIReadyZipLayout, user_id: str, meta: dict[str, Any]
    ) -> list[dict[str, Any]]:
        member = layout.garmin_file("stress", user_id, "stress")
        return self._extract_simple_value_series(
            zip_ref,
            member,
            user_id,
            meta,
            event_type="Stress",
            records_path="body.stress",
            timestamp_path="effective_time_frame.date_time",
            value_path="stress.value",
            output_field="stress_level",
        )

    def _extract_garmin_calories(
        self, zip_ref: zipfile.ZipFile, layout: _AIReadyZipLayout, user_id: str, meta: dict[str, Any]
    ) -> list[dict[str, Any]]:
        member = layout.garmin_file("physical_activity_calorie", user_id, "calorie")
        return self._extract_simple_value_series(
            zip_ref,
            member,
            user_id,
            meta,
            event_type="Calories",
            records_path="body.activity",
            timestamp_path="effective_time_frame.date_time",
            value_path="calories_value.value",
            output_field="active_kcal",
            extra_fields={"activity_name": "activity_name"},
        )

    def _extract_garmin_steps(
        self, zip_ref: zipfile.ZipFile, layout: _AIReadyZipLayout, user_id: str, meta: dict[str, Any]
    ) -> list[dict[str, Any]]:
        member = layout.garmin_file("physical_activity", user_id, "activity")
        # activity uses time_interval.start_date_time
        return self._extract_simple_value_series(
            zip_ref,
            member,
            user_id,
            meta,
            event_type="Activity",
            records_path="body.activity",
            timestamp_path="effective_time_frame.time_interval.start_date_time",
            value_path="base_movement_quantity.value",
            output_field="step_count",
            extra_fields={"activity_name": "activity_name"},
        )

    def _extract_garmin_sleep(
        self, zip_ref: zipfile.ZipFile, layout: _AIReadyZipLayout, user_id: str, meta: dict[str, Any]
    ) -> list[dict[str, Any]]:
        member = layout.garmin_file("sleep", user_id, "sleep")
        try:
            obj = _json_load_from_zip(zip_ref, member)
        except KeyError:
            return []

        records = _dig(obj, "body.sleep")
        if not isinstance(records, list):
            return []

        out: list[dict[str, Any]] = []
        for rec in records:
            if not isinstance(rec, dict):
                continue
            ts = _dig(rec, "effective_time_frame.time_interval.start_date_time")
            if not isinstance(ts, str) or not ts:
                continue
            sleep_stage = rec.get("sleep_stage_state")
            row = self._base_row(user_id, meta, event_type="Sleep", timestamp=ts)
            if sleep_stage is not None:
                row["sleep_level"] = sleep_stage
            out.append(row)
        return out

    def _extract_garmin_respiratory_rate(
        self, zip_ref: zipfile.ZipFile, layout: _AIReadyZipLayout, user_id: str, meta: dict[str, Any]
    ) -> list[dict[str, Any]]:
        member = layout.garmin_file("respiratory_rate", user_id, "respiratoryrate")
        return self._extract_simple_value_series(
            zip_ref,
            member,
            user_id,
            meta,
            event_type="RespiratoryRate",
            records_path="body.breathing",
            timestamp_path="effective_time_frame.date_time",
            value_path="respiratory_rate.value",
            output_field="respiratory_rate",
        )

    def _extract_garmin_oxygen_saturation(
        self, zip_ref: zipfile.ZipFile, layout: _AIReadyZipLayout, user_id: str, meta: dict[str, Any]
    ) -> list[dict[str, Any]]:
        member = layout.garmin_file("oxygen_saturation", user_id, "oxygensaturation")
        return self._extract_simple_value_series(
            zip_ref,
            member,
            user_id,
            meta,
            event_type="OxygenSaturation",
            records_path="body.breathing",
            timestamp_path="effective_time_frame.date_time",
            value_path="oxygen_saturation.value",
            output_field="oxygen_saturation_percent",
        )

    def _extract_simple_value_series(
        self,
        zip_ref: zipfile.ZipFile,
        member: str,
        user_id: str,
        meta: dict[str, Any],
        *,
        event_type: str,
        records_path: str,
        timestamp_path: str,
        value_path: str,
        output_field: str,
        extra_fields: Optional[dict[str, str]] = None,
    ) -> list[dict[str, Any]]:
        try:
            obj = _json_load_from_zip(zip_ref, member)
        except KeyError:
            return []

        records = _dig(obj, records_path)
        if not isinstance(records, list):
            return []

        out: list[dict[str, Any]] = []
        for rec in records:
            if not isinstance(rec, dict):
                continue
            ts = _dig(rec, timestamp_path)
            if not isinstance(ts, str) or not ts:
                continue
            val = _dig(rec, value_path)
            row = self._base_row(user_id, meta, event_type=event_type, timestamp=ts)
            if val is not None:
                row[output_field] = val
            if extra_fields:
                for source_key, out_key in extra_fields.items():
                    v = rec.get(source_key)
                    if v is not None:
                        row[out_key] = v
            out.append(row)
        return out


