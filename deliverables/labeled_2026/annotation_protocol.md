# AI-assisted target-polarity annotation protocol

Version: `ph_target_polarity_v1`. Annotation date: 2026-10-12.

## Scope and process

Two AI assistant annotators read every candidate's full normalized post text in disjoint batches. They received the same rubric, canonical records and matching public embed context. Their tasks explicitly prohibited reading the sentiment classifier, rule predictions, previous relevance flags or evaluation outputs. Annotation was completed before splitting/evaluating the labeled corpus. The fixed rule model was not tuned to these labels.

Every decision records a rationale, annotator, source, rubric version and SHA-256 digest of the exact prepared post text. A third blind AI reviewer independently annotated a simple random sample of 60 of the 335 candidates, selected using Python's `random.Random(20261012).sample`. Before adjudication, exact decision agreement was 54/60 (90%), relevance agreement 56/60 (93.3%), and sentiment agreement 38/40 (95%) among records both annotators included. These are AI consistency measurements; shared model limitations can produce shared errors, and no human validation is claimed. Root AI adjudication resolved all six sample disagreements and separately excluded one nonpolitical bond advertisement. Original decisions, independent review, actual agreement and the seven adjudications are preserved as CSVs. The final corpus contains 231 included records, 96 excluded records and 8 unresolved records.

## Relevance

Include posts about the Philippine national government during 2026: the current executive and vice president, national agencies, national legislature, judiciary, public policy, national governance and present national-government actions relating to previous officials. A historical Marcos Sr reference by itself, unrelated foreign politics, acronym collisions, purely local affairs, generic personal text and nonpolitical commercial promotion are excluded. Regional affairs qualify when a national actor/policy is clearly involved. A bare link, missing comparison, absent cartoon or ambiguous target is unresolved unless matching public context resolves it. Unresolved and excluded candidates receive no sentiment label and are absent from analysis/evaluation.

## Sentiment labels

- `positive`: explicit praise/support, or a clearly beneficial achievement/outcome credited to a Philippine national-government target.
- `negative`: explicit criticism/disapproval, or a clearly adverse allegation, failure or governance outcome targeting Philippine officials/institutions.
- `neutral`: factual government actions/statements/events without an evaluative target polarity, or balanced/mixed text with neither polarity dominant.

Promises and planned actions alone do not establish successful performance. Negative remarks directed only at a foreign country do not imply negative sentiment about the Philippine government. The presence of a politician's name is not a sentiment cue. Reported allegations can convey negative target polarity without proving the allegation or demonstrating the author's own belief. Quoted criticism is annotated as the post's conveyed target polarity. Mixed targets, sarcasm and missing context require careful reading rather than a default neutral label.

`content_kind` distinguishes news reports and personal commentary for interpretation. It is an AI assessment of the text, not a verified account classification. `record_type=post` remains the acquisition type. Reference sentiment may reflect public link-preview content, whereas the baseline classifier uses the canonical post text only; discuss this input limitation when interpreting errors.

## Context and provenance

The public Bluesky `app.bsky.feed.getPosts` endpoint supplies post text, external-preview title/description, image alt text and quoted-post text. Only those text fields are saved to CSV; no author profiles, media files or linked websites are collected. Context is usable only when its normalized current post text matches the original frozen candidate. Missing, failed or changed context cannot resolve the old post's meaning. Mastodon candidates use their complete previously collected text; no profile inference is made.

The canonical field name `gold_label` is retained for compatibility but denotes an AI-assisted reference label in this snapshot. All evaluation CSVs carry `label_source=ai_assisted` and explicit limitations. Accuracy, precision, recall and F1 against these references are agreement measurements. They are not human-validated accuracy, population approval, or evidence of objective government performance.

## Reproducibility

The provided decision CSV is a fixed research input. Offline rebuilding joins exact IDs/platforms and verifies text hashes; it never assigns new labels or copies predictions. Date filtering, normalization and duplicate handling precede curation. A stratified 10% holdout is reserved before ten development folds. Every development class must support ten folds and the holdout must include all three classes; valid labels are never changed to meet those conditions. Preserve the original and labeled snapshots separately and version any future revision.
