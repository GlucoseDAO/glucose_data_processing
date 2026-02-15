"""Sentinel value cleaning for known invalid/indicator values in covariate fields.

Cleans sentinel codes (e.g. stress < 0, heart_rate <= 0) BEFORE any averaging
or interpolation so these indicators don't pollute downstream aggregations.
"""

import polars as pl
from typing import Tuple, Dict, Any, Optional, List
from loguru import logger


# Default cleaning rules: field_name -> (condition_description, filter_expression)
# Each rule converts matching values to NULL.
_DEFAULT_RULES: Dict[str, Dict[str, Any]] = {
    "stress_level": {
        "description": "stress < 0 (Garmin offline/rest sentinel codes)",
        "expr": lambda col: pl.col(col) < 0,
    },
    "respiratory_rate": {
        "description": "respiratory_rate < 0 (invalid/sentinel)",
        "expr": lambda col: pl.col(col) < 0,
    },
    "heart_rate": {
        "description": "heart_rate <= 0 (device offline / invalid)",
        "expr": lambda col: pl.col(col) <= 0,
    },
}


class SentinelCleaner:
    """
    Replaces known sentinel / invalid values with NULL before aggregation.

    This step must run BEFORE gap detection and interpolation so that
    sentinel codes (e.g. Garmin stress = -1, -2) are not averaged with
    real measurements.
    """

    def __init__(self, extra_rules: Optional[Dict[str, Dict[str, Any]]] = None) -> None:
        """
        Args:
            extra_rules: Additional cleaning rules in the same format as _DEFAULT_RULES.
                         Merged with defaults (extras override on key collision).
        """
        self.rules = dict(_DEFAULT_RULES)
        if extra_rules:
            self.rules.update(extra_rules)

    def clean_sentinel_values(
        self,
        df: pl.DataFrame,
        field_categories_dict: Optional[Dict[str, Any]] = None,
    ) -> Tuple[pl.DataFrame, Dict[str, Any]]:
        """
        Replace sentinel / invalid values with NULL.

        Args:
            df: Input DataFrame (pre-aggregation).
            field_categories_dict: Optional field categories (unused currently,
                                   kept for API consistency).

        Returns:
            (cleaned DataFrame, statistics dict)
        """
        if df.is_empty():
            return df, {"sentinel_cleaned": {}, "total_cleaned": 0}

        stats: Dict[str, Any] = {}
        total_cleaned = 0

        set_null_exprs: List[pl.Expr] = []

        for field_name, rule in self.rules.items():
            if field_name not in df.columns:
                continue

            # Count how many values match the sentinel condition
            condition = rule["expr"](field_name)
            count_before = df.select(condition.sum()).item()

            if count_before is None:
                count_before = 0
            count_before = int(count_before)

            if count_before > 0:
                total_non_null = df.select(pl.col(field_name).is_not_null().sum()).item()
                total_non_null = int(total_non_null) if total_non_null else 0
                pct = round(count_before / max(1, total_non_null) * 100, 2)

                stats[field_name] = {
                    "cleaned": count_before,
                    "total_non_null": total_non_null,
                    "pct_of_non_null": pct,
                    "rule": rule["description"],
                }
                total_cleaned += count_before

                # Build expression: where condition is true, set to NULL
                set_null_exprs.append(
                    pl.when(condition)
                    .then(None)
                    .otherwise(pl.col(field_name))
                    .alias(field_name)
                )

                logger.info(
                    f"Sentinel cleaning: {field_name} — {count_before} values "
                    f"({pct}% of non-null) matched '{rule['description']}' → NULL"
                )
            else:
                stats[field_name] = {"cleaned": 0, "rule": rule["description"]}

        if set_null_exprs:
            df = df.with_columns(set_null_exprs)

        result_stats = {
            "sentinel_cleaned": stats,
            "total_cleaned": total_cleaned,
        }

        if total_cleaned > 0:
            logger.info(f"Sentinel cleaning total: {total_cleaned} values set to NULL")
        else:
            logger.info("Sentinel cleaning: no sentinel values found")

        return df, result_stats
