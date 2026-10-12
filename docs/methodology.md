# Philippine government sentiment study: methodology and scripts

This project studies sentiment **expressed in collected public posts about the Philippine national administration during 2026**. Sentiment is an observation about the selected posts; it is not a representative approval rating, a causal estimate, or an objective measurement of government performance. Performance-related discussion can be grouped into policy topics, but factual performance claims require separate evidence.

The attached *Midterm Activity 1* is an assignment reference. The implementation provides collection, preparation, analysis, evaluation code, and an offline notebook. The latest handoff supplies curated real records and complete AI-assisted reference labels so the group can focus on documentation, interpretation, the report, and peer evaluation. Its evaluation is agreement with AI-assisted references, not human-validated accuracy.

## Study scope and collection

1. Define an inclusive search vocabulary in `configs/public.example.toml`: administration names and verified aliases, institutions, and relevant policy areas. Include English, Filipino, and Taglish expressions. Keep the query wording and collection dates with the research notes. Avoid selecting only approving or only critical terms. Each committed snapshot retains its exact settings in `config.toml`.
2. Use public data accessible through supported official APIs, or import a legitimately obtained CSV into the same schema. The project handoff uses **Bluesky and Mastodon only**. X and Threads are outside this collection scope. Public search, instance visibility, retention, and collection time limit coverage differently for each source.
3. Collect at least **200 final cleaned records from different sources** for the activity. A pilot with fewer records is useful for debugging but does not satisfy that requirement. Count distinct retained records after cleaning, not downloaded rows. Inspect the actual platform distribution; requesting several platforms does not guarantee that each returned usable data.
4. Use the interval beginning `2026-01-01` and ending before `2027-01-01`, with timestamps normalized consistently. Data collected before the end of 2026 covers only the elapsed portion of that year. Do not describe it as a full-year sample.
5. Keep all tabular datasets in `.csv` files. Keep raw, prepared, annotated, analyzed, and evaluation datasets separate so each transformation is traceable.

Use a sampling log to record collection time, search vocabulary, requested platform, actual coverage, and failures. Inspect the collection command's diagnostics and preprocessing audit. If a source is inaccessible or returns no usable records, report that limitation and obtain another authorized source. Do not treat an empty source as neutral sentiment.

## Offline group handoff

The notebook defaults to `deliverables/labeled_2026`, which supplies the curated canonical corpus and complete reference labels. `labels/annotation_decisions.csv` records candidate decisions and reasons; `labels/post_context.csv` preserves available public embed context used in review. Read `curation_summary.csv` and `source_summary.csv` for actual delivered counts, class balance, and observed date coverage. Excluded and unresolved candidates are documented instead of being assigned neutral labels or silently reintroduced.

The original `deliverables/public_2026` snapshot preserves **289 Bluesky and 75 Mastodon downloaded posts**, with **335 computationally prepared unique candidates**. The earlier [validation report](../deliverables/validation_2026/README.md) found topic contamination and context-dependent relevance. Those historical files stay unchanged; their original blank template and exploratory model results are distinct from the latest curated labeled handoff.

Install the package and notebook dependencies following the README, select that Python kernel, and run all cells in `notebooks/philippine_government_2026.ipynb`. No `.env` setup or live requests are required. The notebook verifies a complete SHA-256 manifest, copies the supplied curated prepared CSV and reference labels, and generates analysis/evaluation beneath `data/labeled-notebook-run`. It does not run preparation on all raw candidates again.

Evaluation is enabled by default with `LABEL_SOURCE = "ai_assisted"`; no group labeling step is required. Existing working annotations are preserved. Input, configuration, implementation, Python/model-library versions, and output fingerprints prevent silently reusing stale results; an intentional rebuild regenerates derived CSVs while preserving labels. A separately documented human review can replace working labels with `LABEL_SOURCE = "human"`.

The AI-assisted references were assigned from text and available public context independently of the fixed rule classifier. The canonical `gold_label` column is a compatibility name, not a claim of human validation. Two initial AI assistants annotated disjoint batches. A third blind AI reviewer independently annotated a simple random sample of 60 candidates (seed `20261012`), followed by root AI adjudication. `labels/annotation_quality.csv` reports actual AI review agreement, with the independent decisions and adjudication retained separately. This consistency measurement among AI reviewers does not establish human-validated correctness; no measured human agreement is claimed. See [the annotation protocol](annotation_protocol.md). News reporting and announcements are included when relevant; their target-related polarity does not establish the poster's personal approval. The sample has uneven source/month/language coverage and does not represent a complete year or a representative national opinion survey.

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

For exploratory topic changes, record the settings and regenerate outputs. Freeze the choices before interpreting final results. Topic modeling is descriptive and separate from the reference-label sentiment evaluation.

## Proximity measures

`proximity_pairs.csv` compares selected document pairs using three representations:

| Measure | Representation | Interpretation |
| --- | --- | --- |
| Cosine similarity | Normalized TF-IDF vectors | Larger values mean more similar lexical directions. |
| Jaccard similarity | Sets of analyzed tokens | Larger values mean a greater fraction of shared unique tokens. |
| Euclidean distance | Normalized TF-IDF vectors | Smaller values mean closer vectors. |

The pair limit bounds time and output size; the CSV is not necessarily the complete set of document pairs. Check `pair_status` before interpreting a row. Empty representations do not provide usable evidence of textual similarity. These measures compare wording, not agreement, policy accuracy, or shared authorship.

## Rule sentiment and AI-assisted reference annotation

The baseline uses a versioned English/Filipino/Taglish lexicon with local negation, intensifiers, and contrast handling. It predicts `positive`, `negative`, or `neutral` and stores lexical evidence. Its bounded score is a heuristic, not calibrated confidence. Zero-cue neutral classifications are fallback results, not verified neutral opinions.

The rules can miss sarcasm, idioms, code switching, quotations, target ambiguity, and mixed sentiment. Negative language about an unrelated harmful event does not establish criticism of the administration. Reference evaluation measures discrepancies with the supplied content-based annotations.

The default working reference file is `data/labeled-notebook-run/labels/annotation.csv`, copied from the curated snapshot. It contains complete AI-assisted labels independently assigned from the rule classifier. Do not copy rule outputs into this file. The earlier blank template in `deliverables/public_2026` is retained only as historical collection provenance; `label` remains available for future annotation exercises.

The supplied annotation protocol is:

1. Establish relevance to the Philippine national administration or its governance during 2026 using post text and available public embed context. Document off-target or unresolved candidates separately and exclude them from the labeled analysis corpus.
2. Label the target-related polarity of the content: `positive` for approval, favorable assessment, attributed governance success, or beneficial government action; `negative` for criticism, adverse assessment, alleged misconduct/failure, or adverse government action; `neutral` for relevant information without a sufficiently clear favorable/adverse assessment or balanced content without a dominant polarity.
3. Include target-related favorable/adverse claims conveyed in news reporting. Describe these as content polarity; they do not prove the posting author's personal stance or the truth of an allegation.
4. Record concise reasons, context availability, and AI-assisted provenance. Treat unresolved relevance as an exclusion, rather than fabricating a neutral sentiment label. Keep model predictions separate from reference annotation.
5. Independently review a simple random 60-candidate sample with a third AI annotator who is blind to initial labels and rule outputs; record actual relevance/label agreement and root AI adjudication. Report these as AI consistency diagnostics. Evaluation remains agreement with AI-assisted references; no human ground-truth validation or measured human agreement is claimed. An optional documented human review can strengthen future results.

`annotation_summary.csv` reports actual reference counts. `labels/reference_sentiment_summary.csv` groups supplied reference polarity by topic, platform, and assisted `content_kind`; it is separate from `analysis/sentiment_summary.csv`, which counts rule predictions. Reference review may use public embed context while the fixed rule model uses canonical post text only; document that input limitation when interpreting errors. Neither content purpose nor conveyed polarity establishes an account owner's personal political stance.

Keep definitions fixed during annotation and report actual class/source distributions. Do not change defensible labels to satisfy a split or manufacture a metric. The evaluator rejects blank, invalid, duplicate, or insufficient labels. Human-reviewed replacement annotations should preserve IDs/text, document their method, and select `--label-source human`.

## Sentiment evaluation

The evaluator performs a reproducible **stratified 90% development / 10% holdout split**, then **stratified 10-fold cross-validation on the development portion**. All three classes must be present, and each development class must have enough examples for ten folds. Duplicate or insufficient labels are rejected with an actionable error.

The rule classifier is immutable: it has no training step. Cross-validation therefore describes variation in a fixed baseline's performance across development folds; it is not evidence that the classifier learned from the training folds. If a trainable model is added later, fit its preprocessing and model inside each fold.

Report accuracy, macro precision, macro recall, macro F1, and balanced accuracy, together with per-class results and confusion matrices. For this handoff, these metric names quantify agreement with AI-assisted references; they are not human-validated accuracy. The notebook explicitly passes `--label-source ai_assisted`, and evaluation CSVs record that provenance. Prefer macro F1 and class-level results when the corpus is imbalanced. Read the CSV scope fields to distinguish development cross-validation results from holdout results. Do not pool them into one score.

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
| Supplied reference-label copy | Curated prepared CSV and supplied labels | `data/labeled-notebook-run/labels/annotation.csv` | Complete independent AI-assisted references; existing working labels are preserved. |
| `python -m ph_sentiment evaluate --label-source ai_assisted` | Supplied reference-labeled CSV | `evaluation_metrics.csv`, `classification_report.csv`, `confusion_matrix.csv` | Holdout/development agreement with AI-assisted references. |
| Same evaluation command | Annotated CSV | `split_assignments.csv`, `evaluation_predictions.csv` | Reproducible split membership and prediction-level error inspection. |
| Notebook | Bundled curated labeled snapshot and configuration | Working CSVs in `data/labeled-notebook-run`, cells, and previews | Offline analysis and evaluation enabled by default; rename a submission copy to `GROUP NAME.ipynb`. |

Analysis files are written under `data/labeled-notebook-run/analysis`; evaluation files are written under `data/labeled-notebook-run/evaluation` in the default notebook workflow. The snapshot's `script_output_remarks.csv` also provides an activity mapping. Run `python -m ph_sentiment --help` and the relevant subcommand's `--help` for the complete current options. The README supplies installation, handoff, collection, and import-schema instructions.

## Interpretation and final research notes

Complete the notebook's interpretation prompts using the real data and supplied AI-assisted references, with their provenance stated explicitly. Report the final retained number of records after relevance review, sources, observed time coverage, sampling procedure, topic naming process, evaluation metrics, and limitations. Include the activity's scripts/outputs/remarks table and evidence-backed discussion in the eventual submission. Do not describe the supplied AI-assisted agreement scores as human-validated accuracy or substitute model predictions for independent reference labels.

Restrict findings to this collected corpus. Social media users self-select, platform search APIs impose coverage limits, repeated content can distort counts, and sentiment rules have language and target errors. Different platform shares can reflect collection design. A change in collected sentiment is not evidence that government performance caused the change. Corroborate specific performance questions with documented indicators or primary sources in a separate research step.
