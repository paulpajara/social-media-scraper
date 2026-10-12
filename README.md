# Philippine government sentiment research (2026)

A Python command-line pipeline and notebook for collecting public social-media discussions, preparing a reproducible corpus, extracting at least three topics, comparing posts, and evaluating a rule-based English/Filipino/Taglish sentiment baseline. Every dataset, audit, prediction, and metric export is **CSV**. No fabricated observations or political conclusions are included.

The active collection platforms are **Threads, X, Bluesky, and Mastodon**. `collect-live` attempts all four independently; only X and Threads need tokens. Reddit and Facebook are outside the active collection scope.

The implementation follows the supplied midterm activity's computational requirements. The documentation PDF and peer-evaluation form are outside this code-focused phase. Rename a submission copy of the notebook to your actual `GROUP NAME.ipynb`.

## Run now without credentials

After installation, this command collects real public posts from **Bluesky and Mastodon**, prepares the CSV dataset, and runs topic/sentiment/proximity analysis when at least 200 valid unique records are available:

```powershell
./.venv/Scripts/python.exe -m ph_sentiment collect-public
```

No signup, access token, or `.env` value is required for this public-source command. It uses `configs/public.example.toml` and creates a new timestamped directory under `data/public/`. Use `--output-dir data/public/my-run` for a chosen folder, `--platform bluesky` to select one source, or `--skip-analysis` to collect and prepare only. Each run saves real raw records, source manifests/statuses, a preprocessing audit, a cleaned CSV, and the analysis CSVs when the configured minimum is reached.

This command selects the two credential-free platforms only. It does not attempt X or Threads. Search/hashtag results form a bounded discussion sample; review relevance and distinguish news/reporting from expressed opinions before using it as the final research corpus. Mastodon requires public preview on the configured instance and filters hashtag posts for both country and government terms. Public API access can change, and an empty or failed source is reported without inventing rows.

Bluesky requests use its documented public AppView at `api.bsky.app`; Mastodon uses the public `mastodon.social` hashtag timelines. References: [Bluesky API hosts](https://bsky.network/docs/api-directory/), [Bluesky search schema](https://github.com/bluesky-social/atproto/blob/main/lexicons/app/bsky/feed/searchPosts.json), [Mastodon public timelines](https://docs.joinmastodon.org/methods/timelines/).

The public preset limits Bluesky to one page per configured query because the verified deployment denied a later cursor request. Source-level `max_pages` can override the shared page budget. The manifest records the effective budget and query list; result coverage is deliberately bounded.

## Install (PowerShell, Python 3.11+)

```powershell
python -m venv .venv
./.venv/Scripts/python.exe -m pip install -e '.[dev,notebook]'
if (-not (Test-Path .env)) { Copy-Item .env.example .env }
if (-not (Test-Path configs/research.toml)) { Copy-Item configs/research.example.toml configs/research.toml }
```

Use `./.venv/Scripts/python.exe` for the commands below; activation is optional. X and Threads collection needs platform credentials. Add `X_BEARER_TOKEN` and `THREADS_ACCESS_TOKEN` to `.env` locally or set environment variables. Existing environment variables take precedence. Preserve a configured `.env` when installing or updating the project. `.env` and generated `data/` files are ignored by Git; credentials must never be committed or put in the TOML configuration. Do not share raw research data by accident.

`configs/research.toml` specifies search terms, sample/page budgets, platform settings, and model settings. Dates are **Asia/Manila calendar dates**, inclusive `start_date` and exclusive `end_date`. The default window is January 1, 2026 through January 1, 2027; collectors only return records that already exist. Review your queries for relevance to the national administration. Broad terms can return unrelated content; document manual exclusions and sampling choices. Add balanced policy terms such as education, health, prices, jobs, and transport as needed, rather than selecting only approving or critical language.

## Collection access and coverage

### One-command live collection

The two credential placeholders in `.env.example` are for **X and Threads**. Blank values leave those sources unavailable; Bluesky and Mastodon can still run. No credentials are included in this repository.

```powershell
# Uses the example research configuration and your local .env by default.
# Attempt Threads, X, Bluesky, and Mastodon; prepare and analyze the CSVs.
./.venv/Scripts/python.exe -m ph_sentiment collect-live

# Select sources/config explicitly when needed (repeat --platform for more).
./.venv/Scripts/python.exe -m ph_sentiment collect-live --platform x --platform threads --config configs/research.toml
```

Each run creates a new timestamped directory in `data/live/`, containing per-source raw CSVs and manifests, `collection_report.csv`, `run_summary.csv`, `processed/posts.csv`, a preprocessing audit, and analysis CSVs when the configured minimum is met (200 by default). Threads and X also save `*_query_diagnostics.csv` with API-returned and locally retained counts per query. Use `--output-dir` to choose a directory or `--skip-analysis` to collect/prepare only. Explicit platform selection and automatic source selection use the same real API collectors.

Missing credentials, empty search results, successful collection, and incomplete collection have distinct statuses. Every completed source is saved immediately. A later API failure retains already received records and marks that source `partial`; other selected sources continue. A run with no valid collected records exits with an actionable error and CSV readiness/status reports. No placeholder rows are added to public datasets. The active sources collect search or hashtag results, so keep `research.include_comments = false`.

Human sentiment evaluation remains a separate step requiring independent labels. Adding credentials enables real collection; it does not guarantee platform approval, query coverage, or sufficient records for the activity.

For a bounded live verification, use the test preset below. It attempts all four sources, limits X to one request with at most ten returned posts, and records source-level results in CSV. Normal API usage charges can apply; the program does not purchase credits or change billing settings.

```powershell
./.venv/Scripts/python.exe -m ph_sentiment collect-live --config configs/live-test.example.toml --skip-analysis
```

`collection_report.csv` includes `http_status` and `error_category` for failed HTTP requests. Recognized X HTTP 402 credit failures report `credits_depleted`; other payment failures report `payment_required`. Query diagnostics distinguish `empty_api` (the API supplied no posts), `filtered_empty` (returned posts were rejected locally), and failed or partial queries. Threads HTTP 200 with zero records does not prove public-search approval or neutral public sentiment. Threads requests clamp the exclusive end timestamp to the current moment, as its API requires. Only counts, allowlisted metadata, and safe diagnostic messages are logged; upstream error bodies and token values are not logged.

Single-source `collect` exits with code 2 when no records are retained and preserves the empty CSV and query diagnostics. Use a new output directory for each research run; timestamped directories are the default. When using `--overwrite`, rely on the current source report rather than treating older files remaining in the directory as newly collected observations.

### Check tokens and account readiness

Run this before collection to inspect Threads token validity, required scopes, and expiry, and read X's API credit balance:

```powershell
./.venv/Scripts/python.exe -m ph_sentiment check-access --output data/live/access.csv

# Select either source, or repeat --platform to select both.
./.venv/Scripts/python.exe -m ph_sentiment check-access --platform threads --platform x --output data/live/access.csv --overwrite
```

This command calls the read-only [Threads token debugger](https://developers.facebook.com/documentation/threads/troubleshooting/debug-access-token) and [X credit-balance endpoint](https://docs.x.com/x-api/usage/get-usage-credits). It makes no post-search requests and does not purchase credits. The CSV records scope flags, expiry in UTC, and the credit balance in USD without exporting tokens or account identities. Existing output requires `--overwrite`. The command exits with code 2 when a selected check is blocked, unverified, or missing credentials; a successful metadata check does not establish public-search approval or endpoint entitlement.

Threads collection checks token metadata by default and stops before search when the token is invalid, expired, or known to lack `threads_basic` or `threads_keyword_search`. It warns near expiry. If the debugger is unavailable, collection continues with an explicit warning that permissions remain unverified. Set `check_token = false` in `[threads]` only when token validity and scopes have already been checked externally; this does not change Meta's public-access requirements.

If Threads still returns `empty_api`, confirm that your app's **public `threads_keyword_search` access has passed App Review** in Meta's app dashboard. Token scopes alone do not establish that approval: [Meta's keyword-search documentation](https://developers.facebook.com/docs/threads/keyword-search/) describes restricted results for apps without approval. Authorize both required scopes and regenerate the token after changing permissions. For an expiring token, follow Meta's [long-lived token instructions](https://developers.facebook.com/docs/threads/get-started/long-lived-tokens/), then replace `THREADS_ACCESS_TOKEN` locally and rerun `check-access`.

For X `credits_depleted`, check Billing/Credits in the [X Developer Console](https://console.x.com/) and restore the account's credit balance before retrying. Both recent and archive search require available access; changing the date window cannot repair depleted credits. Review current [X pricing](https://docs.x.com/x-api/getting-started/pricing) and spending limits before funding the account. The program cannot restore credits automatically, and a positive balance does not guarantee every endpoint is authorized.

| Source | Implemented collection | Required configuration / access |
| --- | --- | --- |
| X | Recent or full-archive post search with pagination | `X_BEARER_TOKEN`; recent search covers only seven days, so full-year research needs entitled archive access |
| Threads | Keyword/tag search with cursor pagination | `THREADS_ACCESS_TOKEN`; approved `threads_keyword_search` permission is required for public keyword coverage |
| Bluesky | Public AppView post search | No credentials; one page per query in the default preset |
| Mastodon | Public hashtag timelines on `mastodon.social`, filtered for Philippine government relevance | No credentials; instance visibility and public-preview access limit coverage |

These APIs have different search semantics, quotas, and visibility limits. A valid Threads token does not establish approval for public keyword search: without the required approval, returned content can be restricted to the authorized account. Threads ranking and historical coverage can also limit the sample. Mastodon covers content visible to the selected instance rather than the entire network. Inspect record counts and source URLs before reporting coverage.

Every collector filters publication timestamps locally, deduplicates record IDs, respects bounded pagination, and records API-specific limitations in a collection manifest. Missing permission, quota, or authentication produces a source error; the other selected platforms continue. Count actual retained records by platform rather than assuming that a four-platform request yielded four-platform observations. An unavailable or empty source is not neutral sentiment.

Current interfaces are based on the official [X search documentation](https://docs.x.com/x-api/posts/search/introduction), [Threads keyword-search documentation](https://developers.facebook.com/docs/threads/keyword-search/), [Bluesky search schema](https://github.com/bluesky-social/atproto/blob/main/lexicons/app/bsky/feed/searchPosts.json), and [Mastodon public timelines](https://docs.joinmastodon.org/methods/timelines/). X search defaults to `post.fields`, matching the current [official OpenAPI schema](https://docs.x.com/openapi.json); `[x] fields_parameter = "tweet.fields"` remains available for legacy integrations. Live access depends on your application's permissions and current endpoint availability; each run's status reports describe the access actually observed.

```powershell
# Run one or more authorized sources. Each also writes a separate manifest CSV.
./.venv/Scripts/python.exe -m ph_sentiment collect --platform x --config configs/research.toml --output data/raw/x.csv
./.venv/Scripts/python.exe -m ph_sentiment collect --platform threads --config configs/research.toml --output data/raw/threads.csv
./.venv/Scripts/python.exe -m ph_sentiment collect --platform bluesky --config configs/research.toml --output data/raw/bluesky.csv
./.venv/Scripts/python.exe -m ph_sentiment collect --platform mastodon --config configs/research.toml --output data/raw/mastodon.csv
```

The research preset requests X archive search for the 2026 study window. If your application cannot use archive search, that source is reported as unavailable; it does not silently switch to a different date window. For a separate X recent-search pilot, explicitly use a window within the last seven days and `search_mode = "recent"`; do not expect it to retrieve earlier 2026 records. Existing outputs are protected; `--overwrite` explicitly permits replacement.

## Import existing public exports

If API access is unavailable, supply an independently obtained public/authorized CSV using these columns. Import is offline and uses the same validation and preprocessing stages.

| Column | Requirement |
| --- | --- |
| `record_id` | Required stable record ID, unique together with platform |
| `platform` | Required: active sources `x`, `threads`, `bluesky`, `mastodon`; canonical imports also accept `reddit`, `facebook`, `news`, `blog`, or `other` |
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
./.venv/Scripts/python.exe -m ph_sentiment prepare --input data/raw/x.csv --input data/raw/threads.csv --input data/raw/bluesky.csv --input data/raw/mastodon.csv --config configs/research.toml --output data/processed/posts.csv --audit-output data/processed/preprocessing_audit.csv

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
