# Philippine government sentiment research (2026)

A Python pipeline and Jupyter notebook for studying public discussions of the Philippine national administration. The project handoff uses **Bluesky and Mastodon only**. It includes real collected posts, reproducible preprocessing, three-topic LDA, text proximity measures, and an English/Filipino/Taglish rule-based sentiment baseline. All tabular datasets and results are **CSV**.

**Start with the committed dataset in [deliverables/public_2026](deliverables/public_2026/): 335 cleaned unique records awaiting relevance review (261 Bluesky, 74 Mastodon), prepared from 289 Bluesky and 75 Mastodon posts.** Group members can run the notebook against this snapshot without social-media credentials or live API requests. X and Threads are excluded from this handoff; their optional adapters remain available in the source code.

**Read the [notebook validation report](deliverables/validation_2026/README.md) before interpreting the supplied outputs.** Execution and numerical checks pass, but provisional assisted review found 184 clearly relevant records, 82 off-target records and 69 requiring context. These screening flags need human confirmation. The minimum of 200 eligible final records is not yet established, topic contamination needs correction, and no-cue neutral predictions are not confirmed neutral opinion. This is an exploratory collection/code handoff for review and annotation, not a completed sentiment study.

The code covers the supplied midterm activity's computational stages. Independent human labeling, interpretation, the final report, and the peer-evaluation form remain group work. Sentiment predictions are not ground truth or an objective measure of government performance.

## Group handoff: run the notebook

Use the `feature/research/public-data-handoff` branch and keep the repository's folders together. Deliver the files uncompressed, following the activity's submission instructions.

From the repository root, install Python 3.11+ dependencies and register the environment's notebook kernel:

```powershell
# Run from the repository root (PowerShell).
python -m venv .venv
./.venv/Scripts/python.exe -m pip install -e '.[notebook]'
./.venv/Scripts/python.exe -m ipykernel install --user --name ph-government-sentiment --display-name "Philippine government sentiment (.venv)"
./.venv/Scripts/python.exe -m jupyterlab notebooks/philippine_government_2026.ipynb
```

In JupyterLab, select **Philippine government sentiment (.venv)** and run all cells. VS Code can also open [the notebook](notebooks/philippine_government_2026.ipynb) using that kernel. No `.env` setup is needed. Installation downloads Python packages; the notebook's default research run reads the bundled CSVs offline.

The notebook uses `deliverables/public_2026` as its input and writes working outputs beneath `data/notebook-run/`. It creates a blank working annotation template before analysis and preserves an existing annotation file. Evaluation is disabled by default (`RUN_EVALUATION = False`) until independent human labels are ready. The committed snapshot remains the original collection record.

For the group:

1. Check the platform/month counts, preprocessing audit, and relevance of the retained posts. The activity requires at least **200 valid unique records from different sources** after review. If posts are removed, document the exclusions and rerun analysis on the revised corpus.
2. Give independent annotators `data/notebook-run/labels/annotation.csv`. They should fill `gold_label` with `positive`, `negative`, or `neutral` without seeing the rule predictions. Keep the record IDs and other columns intact.
3. After annotation and adjudication, set `RUN_EVALUATION = True` and rerun the evaluation cells. The evaluator requires all three classes and enough development examples per class for ten-fold validation.
4. Name and interpret the topics, discuss model errors and sampling limitations, and complete the report and peer-evaluation form. Rename a submission copy of the notebook to the instructor's required `GROUP NAME.ipynb`.

Running the notebook produces descriptive model outputs. It cannot calculate valid accuracy or other supervised metrics from a blank annotation template. Do not copy its predictions into `gold_label`.

## Included scraper deliverables

| Snapshot file | Purpose |
| --- | --- |
| `config.toml` | Collection window, public-source queries/budgets, and analysis settings used for this snapshot. |
| `raw/bluesky.csv`, `raw/mastodon.csv` | Actual public posts returned by the two collectors. |
| `raw/*_collection_manifest.csv` | Collection timestamp, query/hashtag settings, budgets, and coverage warnings. |
| `processed/posts.csv` | Valid, cleaned, deduplicated records used by the notebook. |
| `processed/preprocessing_audit.csv` | Retained/dropped records and preparation reasons. |
| `labels/annotation.csv` | Blank canonical human-label template, with no model predictions. |
| `collection_report.csv`, `run_summary.csv` | Source statuses, row counts, and run outcome. |
| `source_summary.csv` | Actual source distribution and observed publication coverage. |
| `file_manifest.csv` | Relative paths, sizes, and SHA-256 hashes for verifying the snapshot files. |
| `script_output_remarks.csv` | Script/input/output/remarks mapping for the activity. |
| `analysis/*.csv` | Reproducible descriptive baseline outputs; these are not human-labeled evaluation results. |

Read the snapshot's collection timestamp and observed date range when writing the report. Collection on October 12, 2026 cannot cover the remainder of the year. The configured January–December window is a filter, not proof of complete historical coverage. No X or Threads observations are included.

The snapshot is not anonymous: public post IDs, source URLs, and text can identify authors. No separate profile metadata is collected. Share the handoff with the research group and review quoted text before including it in a report. CSV export protects formula-like text with reversible leading-apostrophe escaping; use the project's CSV reader for a round trip.

## Collect and export a new public sample

Use a new output directory to keep the handoff snapshot reproducible:

```powershell
# Run from the repository root (PowerShell).
./.venv/Scripts/python.exe -m ph_sentiment collect-public --output-dir data/public/my-run
./.venv/Scripts/python.exe -m ph_sentiment handoff --run-dir data/public/my-run --output-dir deliverables/my-new-snapshot --config configs/public.example.toml
```

Collection uses `configs/public.example.toml`, retrieves Bluesky and Mastodon posts without tokens, prepares the corpus, and runs analysis when at least 200 valid unique records are available. `--skip-analysis` collects and prepares only. `--platform bluesky` or `--platform mastodon` selects one source. Without `--output-dir`, collection creates a timestamped folder under `data/public/`. The handoff command exports a shareable snapshot separately from the ignored working data.

Every completed source is saved immediately. Partial collection retains received posts and reports its limitations; empty or failed sources produce explicit statuses rather than invented rows. Check `collection_report.csv`, `run_summary.csv`, the manifests, and the retained platform counts before interpreting a new run.

Dates are **Asia/Manila calendar dates**, with inclusive `start_date` and exclusive `end_date`. The public preset requests January 1, 2026 through January 1, 2027; only already published posts can be returned. Review search terms for national-administration relevance and include balanced policy vocabulary rather than selecting only praise or criticism.

Bluesky uses public AppView search at `api.bsky.app`. The preset uses one page per query because a later cursor request was denied in the verified deployment. Mastodon uses public hashtag timelines on `mastodon.social`, filtering for both country and government terms. Its results cover content visible to that instance. These are bounded discussion samples with different search semantics, not a census of either network. References: [Bluesky API hosts](https://bsky.network/docs/api-directory/), [Bluesky search schema](https://github.com/bluesky-social/atproto/blob/main/lexicons/app/bsky/feed/searchPosts.json), [Mastodon timelines](https://docs.joinmastodon.org/methods/timelines/).

## Run individual research stages

The notebook coordinates these same Python functions. CLI commands can reproduce the stages without Jupyter:

```powershell
# Run from the repository root (PowerShell); write to a new working folder.
./.venv/Scripts/python.exe -m ph_sentiment prepare --input deliverables/public_2026/raw/bluesky.csv --input deliverables/public_2026/raw/mastodon.csv --config deliverables/public_2026/config.toml --output data/manual-run/posts.csv --audit-output data/manual-run/preprocessing_audit.csv
./.venv/Scripts/python.exe -m ph_sentiment label --input data/manual-run/posts.csv --config deliverables/public_2026/config.toml --output data/manual-run/annotation.csv
./.venv/Scripts/python.exe -m ph_sentiment analyze --input data/manual-run/posts.csv --config deliverables/public_2026/config.toml --output-dir data/manual-run/analysis
# Run only after independent human labeling and adjudication.
./.venv/Scripts/python.exe -m ph_sentiment evaluate --input data/manual-run/annotation.csv --config deliverables/public_2026/config.toml --output-dir data/manual-run/evaluation
```

Existing outputs are protected unless explicitly overwritten. Never overwrite completed human labels with a new blank template. After changing the final corpus, recreate its annotation template in a new file and regenerate downstream analysis.

Preparation validates timestamps, removes empty/deleted content, normalizes text, preserves negation and hashtag words, and removes duplicate IDs and normalized token sequences. Its audit records every retained or dropped row. It does not establish political relevance automatically; that requires human review.

| Analysis/evaluation output | Purpose |
| --- | --- |
| `analyzed_documents.csv` | Record-level topic assignment, rule sentiment, heuristic score/evidence, model version, and provenance. |
| `topic_terms.csv`, `topic_metrics.csv` | LDA terms, NPMI coherence, topic diversity, and explicitly in-sample perplexity. |
| `proximity_pairs.csv` | Deterministically sampled TF-IDF cosine similarity, token-set Jaccard similarity, and TF-IDF Euclidean distance. |
| `sentiment_summary.csv` | Counts/shares by topic, platform, publication type, Manila month, and predicted sentiment. |
| `evaluation_metrics.csv` | Accuracy, macro precision/recall/F1, and balanced accuracy, with development and holdout scopes separate. |
| `classification_report.csv`, `confusion_matrix.csv` | Per-class performance and error counts. |
| `split_assignments.csv`, `evaluation_predictions.csv` | Reproducible split membership and record-level prediction audit. |

Evaluation reserves a stratified **10% holdout first**, then runs **10-fold stratified validation on the 90% development portion**. The rule-based classifier has no training step; folds describe performance variation of the fixed baseline. Human labels must include all three classes, with at least ten examples per class in development. Improve rules using development errors and keep the holdout untouched until final assessment.

Rule polarity can miss sarcasm, code switching, quotations, and an opinion's target. Negative wording about an event does not automatically express criticism of the government. Scores and topic weights are not calibrated confidence. Findings describe the collected corpus, not a representative national approval survey or an objective performance assessment. See [the methodology](docs/methodology.md) for annotation and interpretation guidance.

## Import other authorized CSVs

Imported records use this canonical schema; optional fields may be blank:

| Column | Requirement |
| --- | --- |
| `record_id` | Required stable record ID, unique together with platform. |
| `platform` | Required; handoff sources are `bluesky` and `mastodon`. Historical imports also accept `x`, `threads`, `reddit`, `facebook`, `news`, `blog`, or `other`. |
| `created_at` | Required timezone-aware ISO timestamp, e.g. `2026-06-01T12:00:00Z`. |
| `text` | Required nonempty text; quote commas and multiline content correctly. |
| `source_url` | Optional public HTTPS provenance URL. |
| `query` | Optional collection/search phrase. |
| `record_type` | Optional `post` (default), `comment`, or `article`. |
| `collected_at` | Optional timezone-aware ISO timestamp. |
| `gold_label` | Blank until independently annotated: `positive`, `negative`, or `neutral`. |

Include only canonical record CSVs as `prepare --input`, not collection manifests or summaries. Malformed rows fail validation instead of being silently discarded. Test fixtures contain synthetic software examples and are not part of the real research dataset.

## Optional API adapters and developer checks

X and Threads adapters are retained for a later, separately scoped collection. They are not required by the notebook or the public handoff. Their live results were unavailable because X credits were depleted and Threads public-search approval was unconfirmed. Token scopes alone do not establish public-search approval. See [X access/pricing](https://docs.x.com/x-api/getting-started/pricing) and [Threads keyword-search access](https://developers.facebook.com/docs/threads/keyword-search/) before attempting them.

Use local `.env` values `X_BEARER_TOKEN` and `THREADS_ACCESS_TOKEN` only for those optional sources. Environment variables take precedence. Keep tokens out of TOML, Git, archives, and chat; `.env`, `.venv`, and generated `data/` are ignored. `check-access` writes safe credential-readiness metadata to CSV without making post-search requests. `collect-live --platform x --platform threads` explicitly selects the optional sources. Read each command's `--help` before use; normal API usage charges can apply.

To run the software checks:

```powershell
# Run from the repository root (PowerShell).
./.venv/Scripts/python.exe -m pip install -e '.[dev,notebook]'
./.venv/Scripts/python.exe -m pytest
./.venv/Scripts/python.exe -m ruff check .
./.venv/Scripts/python.exe -m ruff format --check .
```

Architecture: `collectors/` owns network access; `schema.py`, `csvio.py`, and `preprocessing.py` own data integrity; `sentiment.py`, `analysis.py`, and `evaluation.py` own modeling and metrics; `cli.py` and the notebook coordinate stages. Runtime dependencies are in `pyproject.toml`; `requirements.lock.txt` records the verified development environment. The handoff's committed CSV snapshot is the only intended generated-data exception to the ignored `data/` working directory.
