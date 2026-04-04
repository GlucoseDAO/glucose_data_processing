"""Logic for preparing final ML dataset."""

import polars as pl
from typing import Dict, Any, List, Optional
from loguru import logger
from processing.core.fields import StandardFieldNames
from formats.base_converter import CSVFormatConverter

class MLDataPreparer:
    """
    Prepares final DataFrame for machine learning with sequence_id as first column.
    """
    
    def __init__(self, config: Dict[str, Any]) -> None:
        self.config = config
        self.round_precision = int(config.get("round_precision", 3))

    def _apply_rounding(self, expr: pl.Expr) -> pl.Expr:
        """Applies rounding based on self.round_precision (supports negative)."""
        if self.round_precision >= 0:
            return expr.round(self.round_precision)
        else:
            factor = 10 ** abs(self.round_precision)
            return (expr / factor).round(0) * factor

    def prepare_ml_data(self, df: pl.DataFrame, field_categories_dict: Optional[Dict[str, Any]] = None) -> pl.DataFrame:
        """
        Dynamically casts all fields and renames them to display names.
        """
        logger.debug(f"Preparing final ML dataset (rounding precision: {self.round_precision})...")
        
        ts_col = StandardFieldNames.TIMESTAMP
        seq_id_col = StandardFieldNames.SEQUENCE_ID
        user_id_col = StandardFieldNames.USER_ID

        cast_exprs: List[pl.Expr] = []
        
        continuous_fields = set(field_categories_dict.get('continuous', [])) if field_categories_dict else set()
        occasional_fields = set(field_categories_dict.get('occasional', [])) if field_categories_dict else set()
        service_fields = set(field_categories_dict.get('service', [])) if field_categories_dict else set()
        
        output_fields = CSVFormatConverter.get_output_fields()
        all_output_fields = set(output_fields)
        
        for col in df.columns:
            if col == seq_id_col:
                cast_exprs.append(pl.col(col).cast(pl.Int64, strict=False).alias(col))
                continue

            if col == user_id_col:
                cast_exprs.append(pl.col(col).cast(pl.Utf8, strict=False).alias(col))
                continue

            current_type = df.schema.get(col)
            
            # Skip booleans
            if current_type == pl.Boolean or str(current_type).startswith("Boolean"):
                cast_exprs.append(pl.col(col))
                continue

            if col == ts_col:
                if current_type == pl.Datetime:
                    cast_exprs.append(pl.col(ts_col).dt.strftime('%Y-%m-%dT%H:%M:%S').alias(ts_col))
                else:
                    cast_exprs.append(pl.col(ts_col).cast(pl.Utf8, strict=False).alias(ts_col))
                continue
            
            if col in continuous_fields or col in occasional_fields:
                expr = pl.col(col).cast(pl.Float64, strict=False)
                cast_exprs.append(self._apply_rounding(expr).alias(col))
            elif col in service_fields:
                cast_exprs.append(pl.col(col).cast(pl.Utf8, strict=False).alias(col))
            elif col in all_output_fields:
                current_type = df.schema.get(col)
                if current_type in [pl.Float64, pl.Float32, pl.Int64, pl.Int32]:
                    expr = pl.col(col).cast(pl.Float64, strict=False)
                    cast_exprs.append(self._apply_rounding(expr).alias(col))
                elif current_type == pl.Datetime:
                    cast_exprs.append(pl.col(col).dt.strftime('%Y-%m-%dT%H:%M:%S').alias(col))
                else:
                    cast_exprs.append(pl.col(col).cast(pl.Utf8, strict=False).alias(col))
            else:
                current_type = df.schema.get(col)
                if current_type == pl.Datetime:
                    cast_exprs.append(pl.col(col).dt.strftime('%Y-%m-%dT%H:%M:%S').alias(col))
                elif current_type in [pl.Float64, pl.Float32, pl.Int64, pl.Int32]:
                    expr = pl.col(col).cast(pl.Float64, strict=False)
                    cast_exprs.append(self._apply_rounding(expr).alias(col))
                else:
                    cast_exprs.append(pl.col(col).cast(pl.Utf8, strict=False).alias(col))

        if cast_exprs:
            df = df.with_columns(cast_exprs)

        output_fields = CSVFormatConverter.get_output_fields()
        
        preferred = [seq_id_col] if seq_id_col in df.columns else []
        preferred.extend([f for f in output_fields if f != seq_id_col])
        if user_id_col in df.columns and user_id_col not in preferred:
            preferred.append(user_id_col)
        
        ordered = [c for c in preferred if c in df.columns] + [c for c in df.columns if c not in set(preferred)]
        ml_df = df.select(ordered)
        
        field_to_display_map = CSVFormatConverter.get_field_to_display_name_map()
        
        rename_map: Dict[str, str] = {}
        for col in ml_df.columns:
            if col in field_to_display_map:
                rename_map[col] = field_to_display_map[col]
        
        if rename_map:
            ml_df = ml_df.rename(rename_map)

        # Final rounding pass for any float columns that might have been missed or introduced
        # We do this before string conversion/null-filling
        float_cols = [col for col, dtype in ml_df.schema.items() if dtype in [pl.Float64, pl.Float32]]
        if float_cols:
            ml_df = ml_df.with_columns([
                self._apply_rounding(pl.col(col)).alias(col) for col in float_cols
            ])

        if bool(self.config.get("restrict_output_to_config_fields", False)):
            service_allow = self.config.get("service_fields_allowlist")
            if isinstance(service_allow, list):
                service_keep = {str(x) for x in service_allow}
            else:
                service_keep = set(service_fields)

            allowed_standard = set(output_fields) | service_keep | {seq_id_col}
            if user_id_col in df.columns:
                allowed_standard.add(user_id_col)

            allowed_cols = {field_to_display_map.get(c, c) for c in allowed_standard}

            if seq_id_col in ml_df.columns:
                allowed_cols.add(seq_id_col)

            ml_df = ml_df.select([c for c in ml_df.columns if c in allowed_cols])
        
        # Final step: Fill nulls with empty strings for all Utf8 (string) columns
        # This is the "last step" where empty strings are safe.
        string_cols = [col for col, dtype in ml_df.schema.items() if dtype == pl.Utf8]
        if string_cols:
            ml_df = ml_df.with_columns([
                pl.col(col).fill_null("") for col in string_cols
            ])
        
        return ml_df

