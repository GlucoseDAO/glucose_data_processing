# Agent Guidelines

This document outlines the coding standards and practices for this project.

## Repository layout (uv workspace)

This repo is a **uv workspace**

### Running the App

The recommended way to start the application is:

- `glucose-process <input_folder>`
- Or using uv: `uv run glucose-process <input_folder>`
- `glucose-process <input_folder>`
- Or using uv: `uv run glucose-process <input_folder>`

## Coding Standards

- **Avoid nested try-catch**: try catch often just hide errors, put them only when errors is what we consider unavoidable in the use-case
- **Type hints**: Mandatory for all Python code.
- **Pathlib**: Always use for all file paths.
- **No relative imports**: Always use absolute imports.
- **Polars**: Prefer over Pandas. Consider using polars expressions to speed up code where relevant. 
- **Be memory efficient**: Use lazyframes (`scan_parquet`) and streaming (`sink_parquet`) for efficiency if applicable, use engine="streaming". Also, remember that you can load multiple files with one scan and can avoid python loops in many cases. Avoid redundant collect-s
- **Memory efficient joins**:  if you make joins make sure largest dataframe goes first. Joins often materializes whole second lazyframe. Pre-filter dataframes before joining to avoid materialization.
- **Polars**: Prefer over Pandas. Consider using polars expressions to speed up code where relevant. 
- **Be memory efficient**: Use lazyframes (`scan_parquet`) and streaming (`sink_parquet`) for efficiency if applicable, use engine="streaming". Also, remember that you can load multiple files with one scan and can avoid python loops in many cases. Avoid redundant collect-s
- **Memory efficient joins**:  if you make joins make sure largest dataframe goes first. Joins often materializes whole second lazyframe. Pre-filter dataframes before joining to avoid materialization.
- **Typer CLI**: Mandatory for all CLI tools.
- **No placeholders**: Never use `/my/custom/path/` in code.
- **No legacy support**: Refactor aggressively; do not keep old API functions.
- **Dependency Management**: Use `uv sync` and `uv add`. NEVER use `uv pip install`.
- **Versions**: Do not hardcode versions in `__init__.py`; use `project.toml`.
- **Literals**: Prefer declaring constants than using hardcoded strings. Constants easyer to reuse and update.
- **Logging**: For logging use loguru library. It gave additional flexibility when choosing loging channel.

## Testing & Docs

- **Tests**: After major code updates always run tests in tests folder with pytest. If introducing new feature or business logic update look through tests and also update them to reflect changes.
- **Regression tests**: Before code update make checkpoint result with CLI script for DATA/uom_small dataset. After code update make one more checkpoint and compare them with compare_checkpoints.py. Result checkpoints should match. Exception is only when there are changes in business logic that modify data or fields. In this case compare differences with expected outcomes. Save checkpoints in to checkpoints folder.
- **Docs**: Put all new markdown files (except README/AGENTS) in `docs/`.

### Test Generation Guidelines

- **Real data + ground truth**: Use actual source data, auto-download if needed, and compute expected values at runtime.
- **Deterministic coverage**: Use fixed seeds or explicit filters; include representative and edge cases.
- **Meaningful assertions**: Prefer relationships and aggregates over existence-only checks.

#### What to Validate

- **Counts & aggregates**: Row counts, sums/min/max/means, distinct counts, and distributions.
- **Joins**: Pre/post counts, key coverage, cardinality expectations, nulls introduced by outer joins, and a few spot-checks.
- **Transformations**: Round-trip survival, subset/superset semantics, value mapping, key preservation.
- **Data quality**: Format/range checks, outliers, malformed entries, duplicates, referential integrity.

#### Avoiding LLM "Reward Hacking" in Tests

- **Runtime ground truth**: Query source data at test time instead of hardcoding expectations.
- **Seeded sampling**: Validate random records with a fixed seed, not just known examples.
- **Negative & boundary tests**: Ensure invalid inputs fail; probe min/max, empty, unicode.
- **Derived assertions**: Test relationships (e.g., input vs output counts), not magic numbers.
- **Allow expected failures**: Use `pytest.mark.xfail` for known data quality issues with a clear reason.

#### Test Structure Best Practices

- **Parameterize over duplicate**: If testing the same logic on multiple outputs, use `@pytest.mark.parametrize` instead of copy-pasting tests.
- **Set equality over counts**: Prefer `assert set_a == set_b` over `assert len(set_a) == 270` - set comparison catches both missing and extra values.
- **Delete redundant tests**: If test A (e.g., set equality) fully covers test B (e.g., count check), keep only test A.
- **Domain constants are OK**: Hardcoding expected enum values or well-known constants from specs is fine; hardcoding row counts or unique counts derived from data inspection is not.

#### Verifying Bug-Catching Claims

When claiming a test "would have caught" a bug, **demonstrate it**:

1. **Isolate the buggy logic** in a test or script
2. **Run it and show failure** against correct expectations
3. **Then show the fix passes** the same test

Never claim "tests would have caught this" without running the buggy code against the test.

#### Anti-Patterns to Avoid

- Testing only "happy path" with trivial data
- Hardcoding expected values that drift from source (use derived ground truth)
- Mocking data transformations instead of running real pipelines
- Ignoring edge cases (nulls, empty strings, boundary values, unicode, malformed data)
- **Claiming tests "would catch bugs" without demonstrating failure on buggy code**

**Meaningless Tests to Avoid** (common AI-generated anti-patterns):

```python
# BAD: Existence-only checks as the sole validation
assert "name" in df.columns
assert len(df) > 0

# BAD: Hardcoded counts derived from data inspection
assert len(source_ids) == 270  # will break when source changes

# BAD: Redundant with set equality test
assert len(output_cats) == 12  # already covered by subset check

# ACCEPTABLE: Required columns as prerequisites
required_cols = {"id", "name", "value"}
assert required_cols.issubset(df.columns)

# GOOD: Set equality from source data
source_ids = set(source_df["id"].unique().drop_nulls().to_list())
output_ids = set(output_df["id"].unique().drop_nulls().to_list())
assert source_ids == output_ids

# GOOD: Domain knowledge constants (from spec, not data inspection)
assert valid_states == {"active", "inactive", "pending"}  # from API spec
```
