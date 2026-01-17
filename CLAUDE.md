# CLAUDE.md

## Development Guidelines

1. **Reuse existing components** - Reuse existing components, methods, and functionality when possible to reduce redundant code.

2. **Efficient solutions** - Write runtime and memory efficient solutions using **polars** for dataframe operations and **altair** for visualization.

3. **Concise documentation** - Keep Markdown files concise and avoid cluttering the repo with excessive markdown files.

4. **TODO for simplifications** - When implementation requires simplifications or assumptions (e.g., hardcoded calibration values), add a `TODO` comment with the recommended data-driven enhancement approach.

5. **NO synthetic/fake data** - Never generate synthetic, fake, or placeholder data. All data must come from real API sources (FRED, SEC EDGAR, CFTC, NY Fed, etc.). If an API is unavailable, either find an alternative real data source or clearly report that the data is unavailable - never fabricate data.
