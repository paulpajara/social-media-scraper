# Philippine government sentiment study: methodology and scripts

This project studies sentiment **expressed in collected public posts about the Philippine national administration during 2026**. Sentiment is an observation about the selected posts; it is not a representative approval rating, a causal estimate, or an objective measurement of government performance. Performance-related discussion can be grouped into policy topics, but factual performance claims require separate evidence.

The attached *Midterm Activity 1* is used as an assignment reference. The implementation provides collection, preparation, analysis, evaluation code, and a reproducible notebook. The handoff includes real scraper outputs, but independent annotation, final relevance decisions, interpretation, the research report, and peer evaluation remain group responsibilities.

## Study scope and collection

1. Define an inclusive search vocabulary in `configs/public.example.toml`: administration names and verified aliases, institutions, and relevant policy areas. Include English, Filipino, and Taglish expressions. Keep the query wording and collection dates with the research notes. Avoid selecting only approving or only critical terms. The committed snapshot includes its exact settings in `deliverables/public_2026/config.toml`.
2. Use public data accessible through supported official APIs, or import a legitimately obtained CSV into the same schema. The project handoff uses **Bluesky and Mastodon only**. X and Threads are outside this collection scope. Public search, instance visibility, retention, and collection time limit coverage differently for each source.
3. Collect at least **200 final cleaned records from different sources** for the activity. A pilot with fewer records is useful for debugging but does not satisfy that requirement. Count distinct retained records after cleaning, not downloaded rows. Inspect the actual platform distribution; requesting several platforms does not guarantee that each returned usable data.
4. Use the interval beginning `2026-01-01` and ending before `2027-01-01`, with timestamps normalized consistently. Data collected before the end of 2026 covers only the elapsed portion of that year. Do not describe it as a full-year sample.
5. Keep all tabular datasets in `.csv` files. Keep raw, prepared, annotated, analyzed, and evaluation datasets separate so each transformation is traceable.

Use a sampling log to record collection time, search vocabulary, requested platform, actual coverage, and failures. Inspect the collection command's diagnostics and preprocessing audit. If a source is inaccessible or returns no usable records, report that limitation and obtain another authorized source. Do not treat an empty source as neutral sentiment.

## Offline group handoff

The repository includes a reproducible scraper snapshot under `deliverables/public_2026`. Collection on October 12, 2026 returned **289 Bluesky and 75 Mastodon posts**. Preparation retained **335 unique records: 261 Bluesky and 74 Mastodon**; the audit records 24 duplicate-text exclusions and five empty/removed exclusions. These are computationally prepared records awaiting human relevance review, not an automatically established final research corpus. All human labels are blank.

The notebook defaults to this snapshot and does not make live collection requests. Install the package and notebook dependencies as described in the README, select the environment's Python kernel, and run `notebooks/philippine_government_2026.ipynb`. No `.env` setup is required. Working CSV outputs go to `data/notebook-run`; the committed snapshot remains unchanged.

Before model outputs are produced, the notebook creates a working blank annotation template at `data/notebook-run/labels/annotation.csv`, preserving an existing annotation file. Independent annotators should use this template without seeing model predictions. Evaluation stays disabled until the template is independently labeled and adjudicated. Then set `RUN_EVALUATION = True` and run the evaluation cells. Blank labels cannot yield meaningful accuracy metrics.

The snapshot includes raw records, source manifests, a cleaned corpus, its preprocessing audit, a blank annotation template, descriptive baseline analysis, source/run summaries, a file-hash manifest, and `script_output_remarks.csv`. Record-level IDs and raw data allow the stages to be traced. `source_summary.csv` describes actual platform distribution and observed publication dates; the retained posts span January 16–October 12, 2026, with uneven month coverage. The dataset does not represent a full-year or balanced sample.

Keep handoff/submission files uncompressed, following the activity's instruction. Rename a submission copy of the notebook to `GROUP NAME.ipynb`. The group must complete the report, interpretation, and peer-evaluation form separately.

## Collect a new public sample

The credential-free command collects **Bluesky and Mastodon**, then performs the same CSV preparation and analysis stages:

```console
# Run from the repository root.
python -m ph_sentiment collect-public --output-dir data/public/my-run
python -m ph_sentiment handoff --run-dir data/public/my-run --output-dir deliverables/my-new-snapshot --config configs/public.example.toml
```

Collection uses `configs/public.example.toml`; neither source requires a token. Without an explicit output directory, results go to a new timestamped folder under `data/public`. Each run writes `collection_report.csv` and `run_summary.csv`; successful or partial sources write `raw/<platform>.csv` and a collection manifest. When records were returned, preparation writes `processed/posts.csv` and `processed/preprocessing_audit.csv`. Analysis runs automatically once the configured retained-record minimum is met (200 by default). The `handoff` command exports a separate shareable snapshot from a completed run.

Use `--platform bluesky` or `--platform mastodon` to select a single source, or `--skip-analysis` to collect and prepare only. A fresh output folder preserves earlier runs. Inspect partial-source warnings, retained counts, actual source coverage, and analysis status before interpreting the outputs. Source failures are recorded separately; report the platforms that actually contributed data. Public search/hashtag visibility limits, query relevance, reporting versus opinion, and language coverage still require review.

Bluesky's preset uses one result page per query because a later cursor request was denied in the verified deployment. Mastodon hashtag timelines reflect content visible to `mastodon.social`, filtered for country and government terms. Review these different coverage limits when comparing sources. Neither source provides a complete platform-wide history.

The notebook's separate stages remain available for manual imports and review. Change its input snapshot setting only when intentionally using another completed handoff, and keep collection disabled when inspecting existing data. Optional X/Threads adapters are retained in the software for separately scoped use; they are not part of the delivered corpus or the default notebook workflow.

## Data preparation and quality checks

The `prepare` command applies the configured date restrictions, normalizes records, and excludes unusable or duplicate material. Its audit CSV records preparation outcomes. Search queries guide collection; preparation does not establish political relevance automatically. Inspect retained samples and imported records for relevance before committing to the final corpus; overbroad queries can include unrelated uses of government or political names.

Retain platform identity, timestamp, text, and the provenance fields defined by the project schema. No separate profile metadata is collected, but public post IDs, source URLs, and text can identify authors; this snapshot is not anonymous. Do not add user-level political classifications. Review quoted text before sharing outputs because it can contain names or other identifying information.

For the final dataset, inspect:

- The retained count and distribution by platform and month.
- Missing or invalid timestamps, empty text, relevance errors, and duplicates.
- Cross-platform reposts and near-duplicate campaigns; simple exact deduplication cannot establish independent authorship or detect every coordinated campaign.
- Language coverage, source imbalance, and uneven API search windows.

Keep raw records unchanged. If an annotation or data-quality correction is needed, document it and rerun downstream stages. Do not quietly reuse an old analysis after changing the prepared dataset.

## Topic modeling

The analyzer uses **Latent Dirichlet Allocation (LDA)** on a bag-of-words count matrix. It requests three topics by default, satisfying the activity's minimum topic count when the retained corpus has sufficient usable vocabulary. Topic probabilities describe the fitted text model; they are not probabilities of an author's beliefs.

Inspect `topic_terms.csv` together with representative records in `analyzed_documents.csv`. Assign human-readable names only after checking both. A topic number has no intrinsic political meaning, and labels should describe the observed discussion rather than force a preferred narrative. Review overlap and whether low-information documents receive a useful assignment.

`topic_metrics.csv` includes document-presence NPMI coherence, topic diversity, and explicitly **in-sample** LDA perplexity. Coherence measures term co-occurrence, diversity measures overlap among top terms, and in-sample perplexity describes fit to the supplied corpus. None proves that the topics are socially meaningful or will generalize. Human inspection is required. A small or homogeneous corpus may not support three distinct interpretable topics.

Keep record types and publication purposes separate when interpreting public sentiment. Institutional posts, news reports, and personal opinions express different kinds of content. `sentiment_summary.csv` groups by `record_type` as well as topic, platform, and Manila-local month; document the type of selected content before describing it as public opinion.

For exploratory topic changes, record the settings and regenerate outputs. Freeze the choices before interpreting final results. Topic modeling is descriptive and separate from the human-labeled sentiment evaluation.

## Proximity measures

`proximity_pairs.csv` compares selected document pairs using three representations:

| Measure | Representation | Interpretation |
| --- | --- | --- |
| Cosine similarity | Normalized TF-IDF vectors | Larger values mean more similar lexical directions. |
| Jaccard similarity | Sets of analyzed tokens | Larger values mean a greater fraction of shared unique tokens. |
| Euclidean distance | Normalized TF-IDF vectors | Smaller values mean closer vectors. |

The pair limit bounds time and output size; the CSV is not necessarily the complete set of document pairs. Check `pair_status` before interpreting a row. Empty representations do not provide usable evidence of textual similarity. These measures compare wording, not agreement, policy accuracy, or shared authorship.

## Rule-based sentiment and human labels

The baseline uses a versioned English/Filipino/Taglish lexicon with local negation, intensifiers, and contrast handling. It predicts `positive`, `negative`, or `neutral`, and stores the lexical evidence. The bounded rule score is a **heuristic score**, not calibrated confidence or a probability.

The rules can miss sarcasm, idioms, code switching, quoted accusations, target ambiguity, and mixed sentiment. They can also detect negative emotional language in a report of a harmful event without establishing that the author criticizes the administration. Do not interpret every predicted negative post as political opposition.

Use the notebook's working template at `data/notebook-run/labels/annotation.csv`, or create a new template with the `label` command. The committed `deliverables/public_2026/labels/annotation.csv` remains the original blank template. Templates contain blank human labels and no model predictions. Preserve completed annotations; regenerate a template into a new filename if the prepared corpus changes.

Use this annotation protocol:

1. Define the target as the Philippine national administration or its governance during the research period. Evaluate the author's expressed stance toward that target, including performance-related policies and actions. Emotional tone about an unrelated person or event is insufficient.
2. Mark `positive` for clear approval or praise, `negative` for clear criticism or disapproval, and `neutral` for information or discussion without a clear stance toward the target. Neutral is not a substitute for an annotation the reviewer cannot resolve.
3. Have reviewers label without seeing rule predictions. For ambiguous or mixed cases, record a review note separately and adjudicate before evaluation. Exclude unresolved examples transparently instead of guessing.
4. Where feasible, have a second reviewer label a subset independently. Report the number reviewed, agreement, disagreements, and adjudication process. Do not claim measured agreement unless it was calculated.
5. Use only these independent human labels as evaluation ground truth. Copying rule predictions into the human-label column would make the evaluation circular.

Keep the annotation guidelines fixed during final labeling. Review class and platform counts before evaluation; collecting more data is preferable to changing valid labels to satisfy a split.

## Sentiment evaluation

The evaluator performs a reproducible **stratified 90% development / 10% holdout split**, then **stratified 10-fold cross-validation on the development portion**. All three classes must be present, and each development class must have enough examples for ten folds. Duplicate or insufficient labels are rejected with an actionable error.

The rule classifier is immutable: it has no training step. Cross-validation therefore describes variation in a fixed baseline's performance across development folds; it is not evidence that the classifier learned from the training folds. If a trainable model is added later, fit its preprocessing and model inside each fold.

Report accuracy, macro precision, macro recall, macro F1, and balanced accuracy, together with per-class results and confusion matrices. Prefer macro F1 and class-level results when the corpus is imbalanced. Read the CSV scope fields to distinguish development cross-validation results from holdout results. Do not pool them into one score.

Inspect development errors before making a rule revision. Once the holdout has been inspected for a revision, it no longer provides an untouched final assessment; a revised model needs a fresh holdout. Preserve `split_assignments.csv`, model version, and settings so the split and predictions can be reproduced.

## CSV artifacts and activity mapping

| Script / command | Input | Output | Remarks |
| --- | --- | --- | --- |
| `python -m ph_sentiment collect-public` | Public collection configuration | A run's `raw/`, `processed/`, optional `analysis/`, `collection_report.csv`, and `run_summary.csv` | Bluesky and Mastodon only; no social-media credentials required. |
| `python -m ph_sentiment handoff` | Completed public run and its configuration | A shareable snapshot such as `deliverables/public_2026` | Raw/prepared records, blank annotation template, summaries, baseline outputs, and file hashes. |
| `python -m ph_sentiment collect` | Public collection configuration and selected platform | `data/raw/<platform>.csv` | Public records actually returned by Bluesky or Mastodon; collection can be incomplete. |
| `python -m ph_sentiment prepare` | One or more raw/import CSVs | `data/processed/posts.csv`, `data/processed/preprocessing_audit.csv` | Count the retained rows here toward the minimum 200. |
| `python -m ph_sentiment analyze` | Prepared CSV | `analyzed_documents.csv` | Normalized records, topic assignment, rule sentiment, score, evidence, and model version. |
| Same analysis command | Prepared CSV | `topic_terms.csv`, `topic_metrics.csv` | At least three requested topics, top terms, and explicit topic-evaluation scope. |
| Same analysis command | Prepared CSV | `proximity_pairs.csv` | Cosine, Jaccard, and Euclidean comparisons with pair status. |
| Same analysis command | Prepared CSV | `sentiment_summary.csv` | Counts and shares by topic, platform, record type, Manila-local month, and label; no population estimate. |
| `python -m ph_sentiment label` | Prepared CSV | A working `labels/annotation.csv` | Blank human-label template, independently edited and reviewed by people. |
| `python -m ph_sentiment evaluate` | Independently annotated CSV | `evaluation_metrics.csv`, `classification_report.csv`, `confusion_matrix.csv` | Holdout and development-fold sentiment metrics. |
| Same evaluation command | Annotated CSV | `split_assignments.csv`, `evaluation_predictions.csv` | Reproducible split membership and prediction-level error inspection. |
| Notebook | Bundled public snapshot and its configuration | Working analysis/evaluation CSVs in `data/notebook-run`, notebook cells, and previews | Offline by default; evaluation requires independent labels. Rename a submission copy to `GROUP NAME.ipynb`. |

Analysis files are written under `data/notebook-run/analysis`; evaluation files are written under `data/notebook-run/evaluation` in the default notebook workflow. The snapshot's `script_output_remarks.csv` also provides an activity mapping. Run `python -m ph_sentiment --help` and the relevant subcommand's `--help` for the complete current options. The README supplies installation, handoff, collection, and import-schema instructions.

## Interpretation and final research notes

Complete the notebook's interpretation prompts using the real data and independent human annotations. Report the final retained number of records after relevance review, sources, observed time coverage, sampling procedure, topic naming process, evaluation metrics, and limitations. Include the activity's scripts/outputs/remarks table and evidence-backed discussion in the eventual submission. If evaluation has not been run on completed independent labels, report it as pending rather than substituting model predictions.

Restrict findings to this collected corpus. Social media users self-select, platform search APIs impose coverage limits, repeated content can distort counts, and sentiment rules have language and target errors. Different platform shares can reflect collection design. A change in collected sentiment is not evidence that government performance caused the change. Corroborate specific performance questions with documented indicators or primary sources in a separate research step.
