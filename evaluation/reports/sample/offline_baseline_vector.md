# Evaluation report — Helix Systems demo benchmark

> **Synthetic / demo evaluation.** The benchmark documents are fictional sample policies created for this project. Scores describe this demo corpus only.

Run `981320b2788f482da6274e64605f4a65` · label: offline baseline (vector) · created 2026-11-22T16:12:23.114603

## Configuration

| Setting | Value |
|---|---|
| top_k | 5 |
| similarity_threshold | 0.1 |
| mode | vector |
| hybrid_alpha | 0.5 |
| reranker | none |
| document_ids | None |
| embedding_provider | hash |
| embedding_model | sentence-transformers/all-MiniLM-L6-v2 |
| llm_provider | extractive |
| llm_model | extractive |
| chunk_size | 800 |
| chunk_overlap | 120 |
| max_questions | None |
| ingest_dataset_documents | True |

## Retrieval (K = 5) — deterministic, ground truth = labelled relevant documents

| Metric | Value |
|---|---|
| Precision@K | 0.746 |
| Recall@K | 0.950 |
| MRR | 0.950 |
| Hit rate | 0.950 |
| Avg retrieval latency (ms) | 2.431 |

## Generation — judge: heuristic (automated estimates — not human-verified ground truth)

| Metric | Value |
|---|---|
| Faithfulness | 0.950 |
| Answer relevance | 0.387 |
| Context relevance | 0.221 |
| Citation coverage | 0.950 |
| Answer token-F1 vs reference | 0.544 |

## Unanswerable questions

Correct abstentions: 2 / 3 (accuracy 0.667)

## Performance

Avg total latency: 3.174 ms · avg generation latency: 0.377 ms · total tokens: n/a

## Per-question results

| ID | Question | P@K | RR | Faithful | Cited | Status |
|---|---|---|---|---|---|---|
| q01 | How many days of paid annual leave do full-time employees ge | 1.000 | 1.000 | 1.000 | 1.000 | success |
| q02 | How many casual leaves are allowed per year? | 1.000 | 1.000 | 1.000 | 1.000 | success |
| q03 | How much unused annual leave can be carried forward and when | 1.000 | 1.000 | 1.000 | 1.000 | success |
| q04 | When is a medical certificate required for sick leave? | 1.000 | 1.000 | 1.000 | 1.000 | success |
| q05 | How long is paternity leave for non-birthing parents? | 0.750 | 1.000 | 1.000 | 1.000 | success |
| q06 | What is the notice period for managers after confirmation? | 0.667 | 1.000 | 1.000 | 1.000 | success |
| q07 | How long is the probation period for new employees? | 1.000 | 1.000 | 1.000 | 1.000 | success |
| q08 | How many days per week can employees work remotely? | 1.000 | 1.000 | 1.000 | 1.000 | success |
| q09 | What is the minimum password length for company accounts? | 0.000 | 0.000 | 0.000 | 0.000 | no_context |
| q10 | Within how much time must a lost laptop be reported? | 0.667 | 1.000 | 1.000 | 1.000 | success |
| q11 | How quickly must a suspected phishing email be reported and  | 1.000 | 1.000 | 1.000 | 1.000 | success |
| q12 | Which systems require multi-factor authentication? | 0.500 | 1.000 | 1.000 | 1.000 | success |
| q13 | What is the daily meal allowance for international travel? | 0.600 | 1.000 | 1.000 | 1.000 | success |
| q14 | When is business class allowed on flights? | 1.000 | 1.000 | 1.000 | 1.000 | success |
| q15 | What is the deadline for submitting expense claims? | 0.500 | 1.000 | 1.000 | 1.000 | success |
| q16 | What is the mileage reimbursement rate for a personal car? | 0.500 | 1.000 | 1.000 | 1.000 | success |
| q17 | How many approving reviews does a pull request need? | 1.000 | 1.000 | 1.000 | 1.000 | success |
| q18 | On which days and times are production deployments allowed? | 0.400 | 1.000 | 1.000 | 1.000 | success |
| q19 | How fast must the on-call engineer acknowledge a page? | 0.667 | 1.000 | 1.000 | 1.000 | success |
| q20 | What is the annual learning budget for engineers? | 0.667 | 1.000 | 1.000 | 1.000 | success |
| q21 | What is the company's stock option vesting schedule? | n/a | n/a | n/a | n/a | no_context |
| q22 | Who is the CEO of Helix Systems? | n/a | n/a | n/a | n/a | success |
| q23 | Does the company provide free gym memberships? | n/a | n/a | n/a | n/a | no_context |
