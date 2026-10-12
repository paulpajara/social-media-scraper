# Labeled research handoff — 2026

This snapshot contains **231 curated, labeled records** from Bluesky and Mastodon. Independent AI annotation screened 335 prepared candidates: 96 excluded and 8 unresolved. Sentiment references: 104 negative, 110 neutral, 17 positive. The original public API acquisition and text remain traceable in raw CSVs, source manifests and the per-record decision audit.

## Run All

Download this entire repository branch, install `python -m pip install -e ".[notebook]"`, select that Python environment, and run `notebooks/philippine_government_2026.ipynb`. No credentials or live API requests are needed. Analysis and evaluation run by default, writing CSV files under `data/labeled-notebook-run/`. Existing working labels are preserved. See the root README for setup.

## What is complete

Relevance screening, positive/negative/neutral reference annotations, three-topic LDA, rule sentiment, cosine/Jaccard/Euclidean proximity, and stratified 90/10 plus ten-fold development evaluation are supplied. `file_manifest.csv` records every other snapshot file's row count, byte size and SHA-256 checksum. `scripts/build_labeled_handoff.py` reproduces the snapshot from the decision/context inputs without creating new labels or using an annotation API.

`processed/posts.csv` and `labels/annotation.csv` contain identical labeled canonical records. The historical schema calls the reference column `gold_label`; here it is **AI-assisted**, not human gold truth. `labels/annotation_decisions.csv` covers every candidate with inclusion/exclusion rationale, annotator provenance, rubric version and text hash. `labels/post_context.csv` preserves the public embed context available to annotators. Read `annotation_protocol.md` for exact definitions and scope.

## Interpretation and remaining group work

The labels describe expressed target polarity, including allegations conveyed in news reports; they do not establish whether those allegations are true or measure each author's personal approval. Some target/polarity judgments use public link-preview context while the fixed lexical classifier sees post text only. Evaluation reports **agreement with AI-assisted references**, not human-validated predictive accuracy. The annotation audit reports any independently reviewed AI sample and its actual agreement separately; this does not establish human validation. Optional independent human validation would strengthen research conclusions.

Finish the topic names, examples, discussion, limitations, instructor's report and peer evaluation. Replace the notebook's group name and save a submission copy as `GROUP NAME.ipynb`. Keep submission files uncompressed as instructed. No compulsory manual labeling step remains in this handoff.

This is a bounded, source-imbalanced sample during the elapsed portion of 2026, dominated by publicly indexed/hashtagged posts and news text. It is not a representative survey, an exhaustive archive, or a measure of objective government performance. Public post text, URLs and IDs can identify authors. Use aggregate findings and necessary traceable examples in documentation.
