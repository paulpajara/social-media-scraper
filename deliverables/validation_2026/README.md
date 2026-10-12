# Notebook validation: execution passes; final research quality is pending

The actual repository notebook was executed on October 12, 2026 with external network access blocked. All eight code cells completed. Its 335 prepared records and five analysis CSVs reproduce the supplied snapshot exactly. The executed copy is saved locally as `data/notebook-validation/philippine_government_2026.executed.ipynb`; the tracked source notebook remains clean for group members to rerun.

**Decision: suitable as an exploratory collection/code handoff for review and annotation. Do not submit the current topic or sentiment outputs as final evidence of Philippine government approval or performance.**

## What passed

- Snapshot checksums, canonical schemas, valid 2026 dates, and identity/provenance links.
- 335 unique prepared records: 261 Bluesky and 74 Mastodon.
- Three fitted LDA topics with ten terms each; finite, valid numerical outputs.
- 5,000 unique nonself proximity pairs; valid cosine/Jaccard bounds and Euclidean consistency.
- Sentiment summary counts reconcile to all 335 records.
- All human labels remain blank. Evaluation correctly stays disabled rather than supplying artificial accuracy scores.

## What prevents final research approval

1. **Relevance:** provisional AI-assisted inspection flagged 184 records as clearly relevant, 82 as off-target, and 69 as needing context. This is a screening aid, not independently confirmed human ground truth. The requirement for at least 200 government-relevant final records has not yet been established. Do not turn uncertain or irrelevant content into neutral sentiment labels.
2. **Topic contamination:** one off-target education/AI article has 2,022 total tokens and 1,240 retained model-feature tokens. It contributes 52.8% of all tokens and 49.9% of retained feature tokens in documents assigned to topic 1. Holding its original top terms fixed, removing that one document changes NPMI coherence from 0.49735 to -0.68243. This sensitivity check does not refit LDA; it shows why a seemingly coherent topic can be inappropriate for the study target. Topic 2 also contains radio/Canada/publisher boilerplate.
3. **Sentiment noncoverage:** the rule model predicts 297 neutral, 22 positive and 16 negative labels. Every neutral prediction has zero matched cues. The 88.7% neutral share describes the default lexical classification, not confirmed neutral attitudes toward the government. Target attribution, sarcasm and quoted news need independent human evaluation.
4. **Sampling:** 75.2% of posts are from September–October, 77.9% from Bluesky. News reporting is common, and repeated coverage and author concentration limit independence. These are post-level observations, not a population approval survey or a uniform full-year sample. Only 8.94% of possible proximity pairs are sampled.

## Files for the group

- `execution_checks.csv`: actual notebook execution and baseline reproduction checks.
- `quality_checks.csv`: numerical successes, interpretation cautions and pending requirements.
- `relevance_review.csv`: all 335 original record IDs, provisional relevance flags, content types and reasons. `reviewer_type` explicitly identifies assisted review and `human_confirmation` remains pending. No sentiment labels are supplied.
- `review_summary.csv`: source/status totals from that provisional review.
- `topic_sensitivity.csv`: fixed-term coherence sensitivity to the longest document, with traceable record ID.
- `mastodon_screen_comparison.csv`: effect of the tightened future collection preset on the old candidate corpus. A term match is not a human relevance verdict; the tighter screen can also miss relevant short posts.
- `file_manifest.csv`: hashes and row counts for these CSV reports.

Keep `deliverables/public_2026` unchanged as the original collection record. The updated collection presets require country markers independent of government markers; bare BBM tags and generic corporate/university president mentions no longer establish governmental relevance. Those fixes affect future collection, not this historical snapshot.

Before final analysis, have the group confirm or correct the relevance flags, follow provenance links for missing context, document exclusions, and retain at least 200 eligible records from both sources. Obtain additional suitably scoped observations if required. Keep news/reporting distinct from expressed opinions when interpreting sentiment. Create and adjudicate independent gold labels without seeing per-record model predictions, then rebuild preparation, topics, summaries and the 90/10 plus ten-fold evaluation for the actual final corpus. Class-count requirements must be met through actual data, never invented or changed labels.
