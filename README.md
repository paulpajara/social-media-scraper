# Philippine government sentiment research (2026)

A Python pipeline and Jupyter notebook for studying public discussions of the Philippine national administration. The project handoff uses **Bluesky and Mastodon only**. It includes real collected posts, reproducible preprocessing, three-topic LDA, text proximity measures, and an English/Filipino/Taglish rule-based sentiment baseline. All tabular datasets and results are **CSV**.

**Start with [deliverables/labeled_2026](deliverables/labeled_2026/): the curated corpus, complete AI-assisted reference labels, annotation reasons/context, and reproducible analysis/evaluation CSVs.** the notebook can run without social-media credentials, live API requests, or manual labeling. Read `source_summary.csv` and `curation_summary.csv` for the delivered source, class, and inclusion/exclusion counts. X and Threads are excluded; their optional adapters remain in the source code.

**Evaluation measures agreement with AI-assisted reference labels, not human-validated accuracy.** Reference labels were assigned from the posts and available public context independently of the fixed rule classifier. Their target is favorable/adverse content about national governance, including claims in reporting; they do not establish an author's personal approval. The schema's `gold_label` column name does not imply gold-standard validation. Decisions and reasons are supplied for audit. Two initial AI assistants annotated disjoint batches; a third blind AI reviewer annotated a simple random 60-candidate sample (seed `20261012`), followed by root AI adjudication. The recorded AI agreement measures consistency among AI reviewers, not human-validated accuracy. See [the annotation protocol](docs/annotation_protocol.md).

The earlier [public collection](deliverables/public_2026/) and [validation report](deliverables/validation_2026/README.md) remain unchanged as historical records. The original 335 candidates required relevance correction; the default notebook uses the curated labeled snapshot instead. Focus on documentation, topic naming, interpretation, the final report, and peer evaluation. Optional independent manual validation can strengthen later claims. Corpus sentiment is not an objective measure of government performance.

## Group handoff: run the notebook

Use the `feature/research/ai-labeled-handoff` branch and keep the repository's folders together. Deliver the files uncompressed, following the activity's submission instructions.

From the repository root, install Python 3.11+ dependencies and register the environment's notebook kernel:

```powershell
# Run from the repository root (PowerShell).
python -m venv .venv
./.venv/Scripts/python.exe -m pip install -e '.[notebook]'
./.venv/Scripts/python.exe -m ipykernel install --user --name ph-government-sentiment --display-name "Philippine government sentiment (.venv)"
./.venv/Scripts/python.exe -m jupyterlab notebooks/philippine_government_2026.ipynb
```

In JupyterLab, select **Philippine government sentiment (.venv)** and run all cells. VS Code can also open [the notebook](notebooks/philippine_government_2026.ipynb) using that kernel. No `.env` setup is needed. Installation downloads Python packages; the notebook's default research run reads the bundled CSVs offline.

The notebook defaults to `deliverables/labeled_2026` and writes beneath `data/labeled-notebook-run/`. It verifies the complete file manifest, copies the already curated prepared CSV and supplied reference labels, then runs analysis and evaluation (`RUN_EVALUATION = True`, `LABEL_SOURCE = "ai_assisted"`). It does not prepare the full raw collection again, which would reintroduce excluded records. Existing working labels are preserved.

Repeat runs reuse results only when input, configuration, implementation, Python/model-library versions, and output hashes match. A changed input or incomplete run produces an error; set `FORCE_REBUILD = True` to intentionally regenerate derived CSVs. To use separately completed human-reviewed labels, preserve their provenance, edit only the working annotation copy, and set `LABEL_SOURCE = "human"`. Completed annotations are never replaced by a blank template.

For the group:

1. Run all notebook cells. The notebook checks at least **200 curated unique records from different sources**, complete reference labels, and the class counts needed for ten-fold development validation.
2. Read the source/curation summaries and annotation decisions/context; describe the AI-assisted procedure and coverage limits honestly in the methodology.
3. Name and interpret the topics, inspect actual proximity examples, and discuss evaluation errors and reference-label limitations. Keep development and holdout scores separate.
4. Complete the report and peer-evaluation form, and rename a submission copy to the instructor's required `GROUP NAME.ipynb`.

The default run produces all analysis and evaluation outputs from supplied labels. The notebook displays supplied reference summaries separately from `analysis/sentiment_summary.csv`, which counts the fixed rule predictions. The reference `content_kind` distinguishes news/commentary purpose; reference polarity may use available public embed context, while the rule model uses canonical post text only. Do not replace reference labels with rule predictions; that would make the evaluation circular.

## Included scraper deliverables

| Snapshot file | Purpose |
| --- | --- |
| `config.toml` | Collection window, public-source queries/budgets, and analysis settings used for this snapshot. |
| `raw/bluesky.csv`, `raw/mastodon.csv` | Actual public posts returned by the two collectors. |
| `raw/*_collection_manifest.csv` | Collection timestamp, query/hashtag settings, budgets, and coverage warnings. |
| `processed/posts.csv` | Curated canonical records, including supplied reference labels. |
| `labels/annotation.csv` | Complete AI-assisted reference labels, independently assigned from the rule model. |
| `labels/annotation_decisions.csv` | Candidate relevance/label decisions, reasons, and annotation provenance. |
| `labels/post_context.csv` | Available public embed context used during review. |
| `labels/independent_review.csv`, `labels/adjudication.csv`, `labels/annotation_quality.csv` | Independent AI sample review, adjudication decisions, and measured agreement. |
| `annotation_summary.csv`, `labels/reference_sentiment_summary.csv` | Supplied reference counts and polarity grouped by topic, source, and content purpose. |
| `curation_summary.csv` | Actual candidate inclusion/exclusion and reference-class counts. |
| `source_summary.csv` | Actual source distribution and observed publication coverage. |
| `file_manifest.csv` | Relative paths, sizes, and SHA-256 hashes for verifying the snapshot files. |
| `analysis/*.csv` | Reproducible descriptive baseline outputs. |
| `evaluation/*.csv` | Split assignments, predictions, and agreement metrics against AI-assisted references. |

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
./.venv/Scripts/python.exe -m ph_sentiment analyze --input deliverables/labeled_2026/processed/posts.csv --config deliverables/labeled_2026/config.toml --output-dir data/manual-run/analysis
./.venv/Scripts/python.exe -m ph_sentiment evaluate --input deliverables/labeled_2026/labels/annotation.csv --config deliverables/labeled_2026/config.toml --output-dir data/manual-run/evaluation --label-source ai_assisted
```

Existing outputs are protected unless explicitly overwritten. Never overwrite completed human labels with a new blank template. After changing the final corpus, recreate its annotation template in a new file and regenerate downstream analysis.

For a new collection, `prepare` validates timestamps, removes empty/deleted content, normalizes text, preserves negation and hashtag words, and removes duplicate IDs and normalized token sequences. Its audit records every retained/dropped row. Query matches and cleaning do not establish relevance; candidate review is a separate stage. `label` still creates a blank template for a separately scoped future annotation exercise.

| Analysis/evaluation output | Purpose |
| --- | --- |
| `analyzed_documents.csv` | Record-level topic assignment, rule sentiment, heuristic score/evidence, model version, and provenance. |
| `topic_terms.csv`, `topic_metrics.csv` | LDA terms, NPMI coherence, topic diversity, and explicitly in-sample perplexity. |
| `proximity_pairs.csv` | Deterministically sampled TF-IDF cosine similarity, token-set Jaccard similarity, and TF-IDF Euclidean distance. |
| `sentiment_summary.csv` | Counts/shares by topic, platform, publication type, Manila month, and predicted sentiment. |
| `evaluation_metrics.csv` | Accuracy, macro precision/recall/F1, and balanced accuracy, with development and holdout scopes separate. |
| `classification_report.csv`, `confusion_matrix.csv` | Per-class performance and error counts. |
| `split_assignments.csv`, `evaluation_predictions.csv` | Reproducible split membership and record-level prediction audit. |

Evaluation reserves a stratified **10% holdout first**, then runs **10-fold stratified validation on the 90% development portion**. The rule classifier has no training step; folds describe performance variation of the fixed baseline. Reference labels must include all three classes, with at least ten examples per class in development. The default CLI label source is human; this handoff explicitly passes `--label-source ai_assisted` and records that scope in the CSVs. Improve rules using development errors and keep the holdout untouched until final assessment.

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
| `gold_label` | Independent reference label: `positive`, `negative`, or `neutral`; blank for new unannotated imports. State human/AI-assisted provenance separately. |

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
