# Philippine government sentiment research (2026)

A Python command-line pipeline and notebook for collecting public social-media discussions, preparing a reproducible corpus, extracting at least three topics, comparing posts, and evaluating a rule-based English/Filipino/Taglish sentiment baseline. Every dataset, audit, prediction, and metric export is **CSV**. No fabricated observations or political conclusions are included.

The implementation follows the supplied midterm activity's computational requirements. The documentation PDF and peer-evaluation form are outside this code-focused phase. Rename a submission copy of the notebook to your actual `GROUP NAME.ipynb`.

## Install (PowerShell, Python 3.11+)

```powershell
python -m venv .venv
./.venv/Scripts/python.exe -m pip install -e '.[dev,notebook]'
Copy-Item .env.example .env
Copy-Item configs/research.example.toml configs/research.toml
```

Use `./.venv/Scripts/python.exe` for the commands below; activation is optional. Only collection needs platform credentials. Add them to `.env` locally or set environment variables. Existing environment variables take precedence. `.env` and generated `data/` files are ignored by Git; credentials must never be committed or put in the TOML configuration. Do not share raw research data by accident.

`configs/research.toml` specifies search terms, sample/page budgets, platform settings, and model settings. Dates are **Asia/Manila calendar dates**, inclusive `start_date` and exclusive `end_date`. The default window is January 1, 2026 through January 1, 2027; collectors only return records that already exist. Review your queries for relevance to the national administration. Broad terms can return unrelated content; document manual exclusions and sampling choices. Add balanced policy terms such as education, health, prices, jobs, and transport as needed, rather than selecting only approving or critical language.

## Collection access and coverage

### One-command live collection

The credential placeholders for **all four sources** are in `.env.example`. The blank values deliberately disable unavailable sources. A local `.env` copy can be filled progressively; you do not need credentials for every platform to run the others. For Facebook, also supply supported `graph_version` and authorized `page_ids` in your research configuration. No credentials are included in this repository.

```powershell
# Uses the example research configuration and your local .env by default.
# Attempt credential-ready sources; prepare and analyze the collected CSVs.
./.venv/Scripts/python.exe -m ph_sentiment collect-live

# Select sources/config explicitly when needed (repeat --platform for more).
./.venv/Scripts/python.exe -m ph_sentiment collect-live --platform reddit --platform threads --config configs/research.toml
```

Each run creates a new timestamped directory in `data/live/`, containing per-source raw CSVs and manifests, `collection_report.csv`, `run_summary.csv`, `processed/posts.csv`, a preprocessing audit, and analysis CSVs when the configured minimum is met (200 by default). Use `--output-dir` to choose a directory or `--skip-analysis` to collect/prepare only. Explicit platform selection and automatic source selection use the same real API collectors.

Missing credentials, empty search results, successful collection, and incomplete collection have distinct statuses. Every completed source is saved immediately. A later API failure retains already received records and marks that source `partial`; other selected sources continue. A run with no valid collected records exits with an actionable error and CSV readiness/status reports. No placeholder rows are added to public datasets. `research.include_comments = true` applies to Reddit/Facebook in multi-source runs; X/Threads keep their supported search-only behavior.

Human sentiment evaluation remains a separate step requiring independent labels. Adding credentials enables real collection; it does not guarantee platform approval, query coverage, or sufficient records for the activity.

| Source | Implemented collection | Required configuration / access |
| --- | --- | --- |
| Reddit | OAuth post search in configured subreddits; optional comments on matched posts | `REDDIT_CLIENT_ID`, `REDDIT_CLIENT_SECRET`, and identifying `REDDIT_USER_AGENT`; approved API access where required |
| X | Recent or full-archive post search with pagination | `X_BEARER_TOKEN`; recent search covers only seven days, so full-year research needs entitled archive access |
| Threads | Public keyword/tag search with cursor pagination | `THREADS_ACCESS_TOKEN`, including approved `threads_keyword_search` permission |
| Facebook | Authorized configured public Page posts and optional comments; literal phrase filtering of post messages | `FACEBOOK_ACCESS_TOKEN`, supported `graph_version`, and numeric `page_ids`; suitable Page permissions or public-content access |

These APIs have different search semantics and quotas. Reddit search does not provide unrestricted keyword search over all comments. Threads ranking and historical coverage can limit the sample. Facebook does not provide a general public-post keyword search in this implementation; Page-authored posts are distinct from citizen comments. Private profiles, private groups, login bypasses, and unrestricted website scraping are not implemented.

Set `research.include_comments = true` for Reddit or Facebook comment collection. Use a separate configuration with it set to `false` for X/Threads, which collect search results. Every collector filters publication timestamps locally, deduplicates record IDs, respects bounded pagination, and records API-specific limitations in a collection manifest. Missing permission, quota, or authentication produces a clear error; an empty response is not a completed research dataset.

Current interfaces are based on the official [Reddit API reference](https://www.reddit.com/dev/api/), [X search documentation](https://docs.x.com/x-api/posts/search/introduction), [Meta Threads search example](https://www.postman.com/meta/threads/request/m9j4i2x/search-for-threads-posts), and [Facebook Pages API documentation](https://developers.facebook.com/docs/pages-api/posts/). Live access depends on your application's permissions and has not been verified using your credentials.

```powershell
# Run one or more authorized sources. Each also writes a separate manifest CSV.
./.venv/Scripts/python.exe -m ph_sentiment collect --platform reddit --config configs/research.toml --output data/raw/reddit.csv
./.venv/Scripts/python.exe -m ph_sentiment collect --platform x --config configs/research.toml --output data/raw/x.csv
./.venv/Scripts/python.exe -m ph_sentiment collect --platform threads --config configs/research.toml --output data/raw/threads.csv
./.venv/Scripts/python.exe -m ph_sentiment collect --platform facebook --config configs/research.toml --output data/raw/facebook.csv
```

The example intentionally leaves Facebook Page IDs/version empty. Fill them with values approved for your application. For X recent search, use a window within the last seven days and `search_mode = "recent"`; do not expect it to retrieve earlier 2026 records. Existing outputs are protected; `--overwrite` explicitly permits replacement.

## Import existing public exports

If API access is unavailable, supply an independently obtained public/authorized CSV using these columns. Import is offline and uses the same validation and preprocessing stages.

| Column | Requirement |
| --- | --- |
| `record_id` | Required stable record ID, unique together with platform |
| `platform` | Required: `reddit`, `x`, `threads`, `facebook`, `news`, `blog`, or `other` |
| `created_at` | Required timezone-aware ISO timestamp, e.g. `2026-06-01T12:00:00Z` |
| `text` | Required nonempty post/comment text; quote commas and multiline text correctly |
| `source_url` | Optional public HTTPS provenance URL |
| `query` | Optional collection/search phrase |
| `record_type` | Optional `post` (default), `comment`, or `article` |
| `collected_at` | Optional timezone-aware ISO timestamp |
| `gold_label` | Blank until independently annotated: `positive`, `negative`, or `neutral` |

Account usernames/IDs are deliberately omitted. Public source URLs remain as research provenance and may indirectly identify authors; restrict data sharing accordingly. Spreadsheet formula-like text is prefixed with an apostrophe on CSV export and recovered on canonical import. Existing leading apostrophes are escaped for a reversible round trip. Do not remove protection in spreadsheets or execute imported cells as formulas.

## Run the research stages

```powershell
# Include only canonical post CSVs, not *_collection_manifest.csv.
# Repeat --input to combine different platforms or authorized external sources.
./.venv/Scripts/python.exe -m ph_sentiment prepare --input data/raw/reddit.csv --input data/raw/x.csv --config configs/research.toml --output data/processed/posts.csv --audit-output data/processed/preprocessing_audit.csv

./.venv/Scripts/python.exe -m ph_sentiment analyze --input data/processed/posts.csv --config configs/research.toml --output-dir data/analysis

# Annotators fill gold_label without seeing rule predictions.
./.venv/Scripts/python.exe -m ph_sentiment label --input data/processed/posts.csv --config configs/research.toml --output data/labels/annotation.csv

# Run only after independent human labeling and adjudication are complete.
./.venv/Scripts/python.exe -m ph_sentiment evaluate --input data/labels/annotation.csv --config configs/research.toml --output-dir data/evaluation
```

Preparation normalizes text, removes URLs/mentions/markup, preserves negation and hashtag words, filters the date window, and removes empty/deleted/duplicate records. The audit explains every retained or dropped record. Cross-platform duplicates are detected using normalized token sequences. Raw CSVs are retained so cleaning decisions can be inspected. Each input CSV must have a valid schema; malformed rows fail instead of being silently discarded.

Analysis and evaluation require **200 valid unique records by default**. Collection may need substantially more raw records to reach that net count. The CLI reports single-platform sampling so you can address the activity's different-source requirement. `--minimum-records 3` is an explicit exploratory/smoke override and does not satisfy the final activity requirements. Human labels must include all three classes, with at least ten examples per class remaining in development for ten-fold validation. Do not manufacture records or change valid labels to meet these constraints.

| Output | Purpose |
| --- | --- |
| `analyzed_documents.csv` | Document-level topic, sentiment score/evidence, model version, and provenance |
| `topic_terms.csv` | LDA topic terms and their weights |
| `topic_metrics.csv` | Document-presence NPMI coherence, topic diversity, and explicitly in-sample perplexity |
| `proximity_pairs.csv` | Deterministically sampled cosine similarity, Jaccard similarity, and Euclidean distance |
| `sentiment_summary.csv` | Counts/shares by topic, platform, publication type, Manila month, and sentiment |
| `evaluation_metrics.csv` | Accuracy, macro precision/recall/F1, balanced accuracy; development folds and holdout kept separate |
| `classification_report.csv`, `confusion_matrix.csv` | Per-class and error diagnostics |
| `split_assignments.csv`, `evaluation_predictions.csv` | Reproducible membership and predictions for audit |

Evaluation reserves a stratified **10% unseen holdout first**, then runs **10-fold stratified validation on the 90% development portion**. The rule-based model has no learned parameters; folds measure performance variation of a fixed baseline. Do not present this as model training. Only independent human labels establish accuracy. Iterative rule improvements must use development errors; inspecting and tuning against the holdout invalidates an untouched final assessment.

The lexicon describes linguistic polarity and can miss sarcasm, code switching, quotations, and the actual target of an opinion. A negative event description is not automatically criticism of the administration. Topic weights and rule scores are not calibrated probabilities. Observed social-media sentiment describes the collected corpus; it is neither a representative national approval survey nor an objective measure of government performance. Read [the methodology](docs/methodology.md) for annotation guidance and interpretation.

## Notebook and verification

Open [the notebook](notebooks/philippine_government_2026.ipynb) in VS Code using the `.venv` Python kernel, or run:

```powershell
./.venv/Scripts/jupyter-lab.exe notebooks/philippine_government_2026.ipynb
./.venv/Scripts/python.exe -m pytest
./.venv/Scripts/python.exe -m ruff check .
./.venv/Scripts/python.exe -m ruff format --check .
```

The notebook separates collection, data exploration, cleaning, modeling, human annotation, validation, and interpretation, with a scripts/outputs/remarks table. Network collection and evaluation are disabled until explicitly enabled by the researcher. Test data are synthetic software fixtures only and are not counted toward the real research dataset.

Architecture: `collectors/` owns network access; `schema.py`, `csvio.py`, and `preprocessing.py` own data integrity; `sentiment.py`, `analysis.py`, and `evaluation.py` own models/metrics; `cli.py` and the notebook coordinate the stages. Runtime dependencies are specified in `pyproject.toml`; `requirements.lock.txt` records the verified development environment. No frontend or web service is needed for these code/script deliverables.
