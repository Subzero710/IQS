# IQS — Image Quality Sorter (Usage)

IQS batch-sorts images from a dataset into quality-based groups using dataset-driven statistics and thresholds.

## Workflow

### Step 1 — Compile dataset statistics
Use `create_stat_analysis.py` to compile the statistics of your dataset into a CSV.

### Step 2 — Generate thresholds/settings
Use `generate_thresholds.py` to generate the settings needed later in a JSON file.

### Step 3 — Batch-sort images (latest version)
Use `prototype2.py` as it is the latest version.

After Step 3, your dataset will be batch sorted into the `Result/` folder.
