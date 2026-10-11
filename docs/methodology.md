# Philippine government sentiment study: methodology and scripts

This project studies sentiment **expressed in collected public posts about the Philippine national administration during 2026**. Sentiment is an observation about the selected posts; it is not a representative approval rating, a causal estimate, or an objective measurement of government performance. Performance-related discussion can be grouped into policy topics, but factual performance claims require separate evidence.

The attached *Midterm Activity 1* is used as an assignment reference. The current implementation focuses on collection, preparation, analysis, evaluation, and a reproducible notebook. It does not supply a completed research report or fabricate the required observations.

## Study scope and collection

1. Define an inclusive search vocabulary in `configs/research.example.toml`: administration names and verified aliases, institutions, and relevant policy areas. Include English, Filipino, and Taglish expressions. Keep the query wording and collection dates with the research notes. Avoid selecting only approving or only critical terms.
2. Use public data accessible through supported official APIs, or import a legitimately obtained CSV into the same schema. Reddit, X, Threads, and Facebook have different access, search, and historical coverage limits. The available results depend on credentials, API permissions, endpoint capabilities, and collection time.
3. Collect at least **200 final cleaned records from different sources** for the activity. A pilot with fewer records is useful for debugging but does not satisfy that requirement. Count distinct retained records after cleaning, not downloaded rows. Inspect the actual platform distribution; requesting several platforms does not guarantee that each returned usable data.
4. Use the interval beginning `2026-01-01` and ending before `2027-01-01`, with timestamps normalized consistently. Data collected before the end of 2026 covers only the elapsed portion of that year. Do not describe it as a full-year sample.
5. Keep all tabular datasets in `.csv` files. Keep raw, prepared, annotated, analyzed, and evaluation datasets separate so each transformation is traceable.

Use a sampling log to record collection time, search vocabulary, requested platform, actual coverage, and failures. Inspect the collection command's diagnostics and preprocessing audit. If a source is inaccessible or returns no usable records, report that limitation and obtain another authorized source. Do not treat an empty source as neutral sentiment.

## One-command live workflow

For the recommended automated workflow, copy `.env.example` to a local `.env`, fill the credentials for the sources you can access, and run:

```console
python -m ph_sentiment collect-live --config configs/research.example.toml
```

The command attempts Reddit, X, Threads, and Facebook independently. Blank credentials are reported as `missing_credentials`; they do not produce invented records. Results go to a new timestamped folder under `data/live` by default. Each run writes `collection_report.csv` and `run_summary.csv`; successful or partial sources write `raw/<platform>.csv` and a collection manifest. When records were returned, preparation writes `processed/posts.csv` and `processed/preprocessing_audit.csv`. Analysis runs automatically once the configured retained-record minimum is met (200 by default).

Repeat `--platform` to select sources, use `--skip-analysis` to collect and prepare only, or provide `--output-dir` for a specific run folder. Existing CSV outputs require explicit `--overwrite`; a fresh folder preserves earlier runs. Inspect partial-source warnings, retained counts, actual source coverage, and analysis status before interpreting the outputs. Credentials and API permissions are still required for live observations; no live data is included merely by configuring placeholders.

The notebook's separate stages remain available for manual imports and review. To inspect an automated run there, point its data paths to the selected live run's `raw`, `processed`, and `analysis` folders.

## Data preparation and quality checks

The `prepare` command applies the configured date restrictions, normalizes records, and excludes unusable or duplicate material. Its audit CSV records preparation outcomes. Search queries guide collection; preparation does not establish political relevance automatically. Inspect retained samples and imported records for relevance before committing to the final corpus; overbroad queries can include unrelated uses of government or political names.

Retain source identity, timestamp, text, and the provenance fields defined by the project schema. Do not add usernames, profile identifiers, or user-level political classifications. Review quoted text before sharing outputs because post text can still contain names or other identifying information.

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

Keep record types separate when interpreting public sentiment. Facebook Page posts can be institutional communications, while comments may reflect public reactions. `sentiment_summary.csv` groups by `record_type` as well as topic, platform, and Manila-local month; document who produced the selected content before describing it as public opinion.

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

Create `data/labels/annotation.csv` using the `label` command. The template should contain blank human labels and no model predictions. Preserve a completed annotation file; regenerate a template into a new filename if the prepared corpus changes.

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
| `python -m ph_sentiment collect-live` | Research configuration and locally filled credentials | A live run's `raw/`, `processed/`, optional `analysis/`, `collection_report.csv`, and `run_summary.csv` | Recommended one-command workflow; missing credentials, failures, and partial coverage remain explicit. |
| `python -m ph_sentiment collect` | Research configuration and platform credentials | `data/raw/<platform>.csv` | Public records actually returned by the supported source; collection can be incomplete. |
| `python -m ph_sentiment prepare` | One or more raw/import CSVs | `data/processed/posts.csv`, `data/processed/preprocessing_audit.csv` | Count the retained rows here toward the minimum 200. |
| `python -m ph_sentiment analyze` | Prepared CSV | `analyzed_documents.csv` | Normalized records, topic assignment, rule sentiment, score, evidence, and model version. |
| Same analysis command | Prepared CSV | `topic_terms.csv`, `topic_metrics.csv` | At least three requested topics, top terms, and explicit topic-evaluation scope. |
| Same analysis command | Prepared CSV | `proximity_pairs.csv` | Cosine, Jaccard, and Euclidean comparisons with pair status. |
| Same analysis command | Prepared CSV | `sentiment_summary.csv` | Counts and shares by topic, platform, record type, Manila-local month, and label; no population estimate. |
| `python -m ph_sentiment label` | Prepared CSV | `data/labels/annotation.csv` | Blank human-label template, edited and reviewed by people. |
| `python -m ph_sentiment evaluate` | Independently annotated CSV | `evaluation_metrics.csv`, `classification_report.csv`, `confusion_matrix.csv` | Holdout and development-fold sentiment metrics. |
| Same evaluation command | Annotated CSV | `split_assignments.csv`, `evaluation_predictions.csv` | Reproducible split membership and prediction-level error inspection. |
| Notebook | Configuration plus generated CSVs | Saved notebook cells and CSV previews | `notebooks/philippine_government_2026.ipynb`; rename a submission copy to the instructor's required `GROUP NAME.ipynb`. |

Analysis files are written under `data/analysis`; evaluation files are written under `data/evaluation` in the notebook workflow. Run `python -m ph_sentiment --help` and the relevant subcommand's `--help` for the complete current options. The README supplies installation, configuration, credential, and import-schema instructions.

## Interpretation and final research notes

Complete the notebook's interpretation prompts only after real data and human annotations are available. Report the retained number of records, sources, observed time coverage, sampling procedure, topic naming process, evaluation metrics, and limitations. Include the activity's scripts/outputs/remarks table and evidence-backed discussion in the eventual submission.

Restrict findings to this collected corpus. Social media users self-select, platform search APIs impose coverage limits, repeated content can distort counts, and sentiment rules have language and target errors. Different platform shares can reflect collection design. A change in collected sentiment is not evidence that government performance caused the change. Corroborate specific performance questions with documented indicators or primary sources in a separate research step.
