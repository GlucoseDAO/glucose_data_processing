#!/usr/bin/env python3
"""
Write docs/CGM_FORMAT_PARITY.md: native converters vs the cgm_format backend, per user.

    uv run python scripts/cgm_format_parity.py test_data/dexcom_small test_data/d1namo_small/diabetes DATA/bigideas

Numbers come straight from formats.cgm_format_input.parity.compare_corpus, the same code
tests/test_cgm_format_parity.py asserts on.
"""

from datetime import datetime, timezone
from importlib.metadata import version
from pathlib import Path
from typing import List, Optional

import typer
from loguru import logger

from formats.cgm_format_input.parity import PARITY_SUM_FIELDS, CorpusParity, compare_corpus

CONFIG = {"dexcom": {"high_glucose_value": 401, "low_glucose_value": 39, "remove_calibration": True}}
LEDGER = Path(__file__).parent.parent / "docs" / "CGM_FORMAT_PARITY.md"

app = typer.Typer(help="Compare native converters with the cgm_format backend and write the ledger.")


def _fmt(value: Optional[float]) -> str:
    return "—" if value is None else f"{value:,.1f}"


def _section(parity: CorpusParity) -> List[str]:
    users = parity.users
    lines = [
        f"### `{parity.root.as_posix()}` (native: `{parity.native_type}`)",
        "",
        f"Users: native {len(parity.native_users)}, cgm_format {len(parity.adapter_users)}, "
        f"same set: {'yes' if parity.native_users == parity.adapter_users else 'NO'}.",
        "",
        "| user | EGV native | EGV cgm_format | only native | only cgm_format | glucose mismatches | median ratio | "
        + " | ".join(f"{f} native / cgm_format" for f in PARITY_SUM_FIELDS)
        + " |",
        "|" + "---|" * (7 + len(PARITY_SUM_FIELDS)),
    ]
    for u in users:
        sums = " | ".join(f"{_fmt(u.sums[f][0])} / {_fmt(u.sums[f][1])}" for f in PARITY_SUM_FIELDS)
        ratio = "—" if u.glucose_ratio is None else f"{u.glucose_ratio:.6f}"
        lines.append(
            f"| {u.user_id} | {u.egv_native:,} | {u.egv_adapter:,} | {u.only_native} | {u.only_adapter} | "
            f"{len(u.glucose_mismatches)} | {ratio} | {sums} |"
        )
    return lines + [""]


@app.command()
def main(roots: List[Path] = typer.Argument(..., help="Inputs both backends can read")) -> None:
    sections: List[str] = []
    for root in roots:
        logger.info(f"Comparing {root}")
        sections.extend(_section(compare_corpus(root, CONFIG)))
    generated = datetime.now(timezone.utc).isoformat(timespec="seconds")
    header = [
        "# cgm_format parity ledger",
        "",
        f"Generated {generated} by `scripts/cgm_format_parity.py` with cgm-format "
        f"{version('cgm-format')} and glucose-dataset {version('glucose-dataset')}. "
        "Do not edit by hand; rerun the script.",
        "",
        "Compares the per-user frames each backend yields before the processing pipeline. "
        "Expected differences are explained in docs/DECISIONS.md D3 and asserted in "
        "tests/test_cgm_format_parity.py.",
        "",
    ]
    LEDGER.write_text("\n".join(header + sections), encoding="utf-8")
    print(f"Wrote {LEDGER}")


if __name__ == "__main__":
    app()
