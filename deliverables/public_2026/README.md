# Public-source research handoff (2026)

This immutable snapshot contains 364 raw public posts and **335 unique prepared records** (261 bluesky, 74 mastodon). Only Bluesky and Mastodon contributed observations. It is a bounded API sample of posts published during the elapsed portion of 2026, collected on the timestamps recorded in the source manifests. Government relevance and human sentiment labels still require review. Public post IDs, source URLs and text can identify authors; this dataset is not anonymous.

## Run the notebook

Download the entire repository branch, keeping `src/`, `configs/`, `notebooks/`, and this folder together. Follow the root `README.md` installation steps (Python 3.11+) and open `notebooks/philippine_government_2026.ipynb` in that environment. Choose Run All. The notebook runs offline against this snapshot and writes student outputs under `data/notebook-run/`. It does not contact X, Threads or any collection API.

## Files

- `raw/`: original public post CSVs and collection manifests.
- `processed/posts.csv`: cleaned starting corpus; `preprocessing_audit.csv` explains exclusions.
- `labels/annotation.csv`: canonical records with blank `gold_label`, with no model predictions.
- `collection_report.csv`, `run_summary.csv`, `source_summary.csv`: actual collection coverage/counts; paths are relative to this folder.
- `analysis/` (if present): reproducible exploratory LDA topics, rule sentiment, summaries and cosine/Jaccard/Euclidean pairs. These are not human labels or final accuracy results.
- `data_dictionary.csv`, `script_output_remarks.csv`: field definitions and script/output/remarks reference.
- `file_manifest.csv`: relative filenames, row counts, sizes and SHA-256 checksums for every other snapshot file.
- `config.toml`: collection and analysis settings, containing no credentials.

## Group responsibilities

1. Review relevance to the 2026 Philippine national government and document exclusions. Keep raw files and this snapshot unchanged. Maintain at least 200 final valid records from different sources; collect more if review reduces the corpus below that minimum.
2. The notebook creates a working annotation CSV under `data/notebook-run/labels/`. Independently fill `gold_label` with `positive`, `negative` or `neutral`, resolving disagreements. Avoid model-output files while annotating. Do not manufacture labels or class counts.
3. Follow the notebook guidance when selecting the reviewed corpus. Rerun preparation and analysis after changing it; name the three fitted topics from their words and representative posts.
4. Enable `RUN_EVALUATION` only after independently labeling the final corpus. The fixed rule classifier reserves a 10% holdout and applies ten-fold evaluation on the remaining 90%; required class counts are checked. No final evaluation metrics are supplied in this unlabelled handoff.
5. Interpret actual outputs, discuss sampling/language/source imbalance, complete the instructor's report and peer evaluation, and save the final notebook as your actual `GROUP NAME.ipynb`. Submit uncompressed files according to the assignment.

Post sentiment does not measure person-level approval or objective government performance. Do not describe API query coverage as an exhaustive archive, uniform full-year sample or representative survey.
