# Capstone Report — Refresh / Content Opportunity Scoring

- **Author:** Mayur Kusmade
- **Lane:** Refresh / Content Opportunity Scoring (locked in ML-07)
- **Repo:** https://github.com/mayurkusmade/flyrank-ml-internship-starter
- **Paper:** https://mayurkusmade.github.io/flyrank-ml-internship-starter/
- **Date:** October 2026

## 0. Abstract

A content editor with fifty review slots a week needs to know which pages to inspect before next month;
the obvious rule — big pages that are already slipping — is cheap to write but had never been tested on a
month it could not see. Using the FlyRank warehouse release, I built monthly features per page (traffic,
in-month momentum, CTR, position, age), labelled a page "at risk" when next month's impressions fell at
least 20%, and fit a logistic model on March→April 2026. Validated with folds grouped by client and tested
once on May→June, the model reached 49 of the top 50 (precision@50) against 47 of 50 for the frozen
hand-written baseline — a two-page difference with overlapping bootstrap intervals. The methodological
finding is larger than the modelling one: a random split made gradient boosting look near-perfect (0.984)
and a forest nearly as good (0.972), while grouping folds by client collapsed them to 0.836 and 0.704,
showing that most of that apparent skill was client memorisation. The output is a decision-support queue
with reason codes and three actions, not evidence that refreshing a page recovers traffic.

## 1. Problem framing

- **Unit:** one content item for one client, summarised over one calendar month.
- **Output:** ranked queue — score, one reason code, one action per row.
- **Human action:** `check_indexing`, `refresh_content`, `monitor`.
- **Cost of a wrong call:** a wasted editorial day, and a decline that runs a month longer than needed.
- **Why ML at all:** the ranking signal is a weighted combination of several weak, noisy measurements.
  A model can combine them; a rule can only multiply two of them. Whether that combination actually helps
  is the question this work answers, rather than assumes.

## 2. Data safety

Release `FlyRank/internship-warehouse`, build `flyrank_pseudonymized_warehouse_release_v20260703`, read
with DuckDB over `hf://`. Tables: `fact_content_daily_performance` (2026-03, 2026-04, 2026-05, 2026-06)
and `dim_content` (creation date only).

**Excluded, with reasons:** `fact_content_query_90d` (its window, 2026-04-02 → 2026-06-30, contains the
outcome month); every current-state field in `dim_content` — `content_updated_date`, `last_optimized_date`,
`is_published`, `is_deleted`, `word_count` — because they describe the July snapshot and would import
post-decision activity (ML-07 measured 82.1% of items as censored that way); rows with
`gsc_data_available IS NOT TRUE`; items under 100 monthly impressions; and all pricing/keyword metadata
that only exists as a July snapshot. IDs are pseudonyms, truncated in every table, and used for grouping
and splitting only. No client names, domains, URLs, raw queries or tokens appear anywhere in `work/`.

## 3. Baseline

The ML-07 rule, frozen: eligible when `imp >= 500`, `imp_h1 > 0` and `momentum < 0.8`; score =
`imp × (1 − momentum)`; one reason code (`slipping_but_visible`), one action (`refresh_content`). It is a
fair comparison because it uses the same frame, the same K and the same scoring moment, and it was never
re-tuned after the test month was opened. On the future month it reaches **47/50** at K=50 (P@50 0.940,
95% CI 0.86–1.00) and 0.920 at K=500.

## 4. Model / analysis

**Features (8, all measured inside the feature month):** `log_imp`, `days_with_imp`, `momentum_f`
(2nd half ÷ 1st half, clipped at 3), `has_h1` (zero first-half traffic flag), `ctr`, `pos`,
`pos_missing`, `age_days` (decision date − creation date).

**Label:** `label = 1` when the next month's impressions < 0.8 × the feature month's. A defined proxy over
observed outcomes; the ±20% band is a choice and is stated as one.

**Method:** logistic regression (standardised) — chosen because it survived the grouped split, not because
it won the random one. Random forest and gradient boosting are carried as challengers. Final deployable
score: `P(decline) × log(1 + impressions)` over the ≥ 500-impression population.

## 5. Evaluation

**Split 1 — grouped by client (5 × 20%, training month):** rule 0.816; logit 0.844 (AUC 0.677); forest
0.704 (0.655); boosting 0.836 (0.656). **Random split of the same rows:** rule 0.876; logit 0.900; forest
0.972; boosting 0.984. The gap (rule −0.060, logit −0.056, forest −0.268, boosting −0.148) is the
memorisation the random split was hiding.

**Split 2 — future window (fit March, predict May, score June), used once:** on the ≥ 500-impression
population (59,874 pages, base rate 0.665) — logit **0.980** P@50 / 0.705 AUC; rule **0.940** / 0.615;
forest 0.880 / 0.696; boosting 0.900 / 0.667. On all pages (111,075, base 0.656) the rule leads at K=50
(0.940 vs 0.900), because the model spends its top on small pages — which is why the deployed score weights
by log impressions.

**Errors:** the top pick of the backtested queue (88,175 May impressions, momentum 0.016) grew 22% in June
— a trough, not a break. Two further misses in the top 200 had momentum ≈ 0.04–0.09 and recovered. The
pattern is consistent: in-month collapse is the strongest signal and occasionally a temporary dip.

## 6. Interpretation

Momentum dominates (standardised coefficient −0.595), then the zero-first-half flag (−0.424) and CTR
(−0.370); age is −0.083, i.e. essentially no signal — the third independent confirmation that staleness is
not the driver the refresh heuristic assumes. The two rankings agree on precision and disagree on pages:
the rule's top 50 has a median of 55,348 impressions across 11 clients; the model's probability-only queue
spans 20 clients with a median of 674, and the log-weighted queue returns to 29,009 impressions across 14
clients at the same 0.98 precision.

## 7. Recommendation

1. Run the playbook queue (`P × log(1+imp)`, ≥ 500 impressions) as the default weekly list.
2. Read the action before the row: `check_indexing` for momentum < 0.10 (a 15-minute technical check, not a
   rewrite), `refresh_content` for a visible slip, `monitor` when the score is high but no pattern is clear.
3. Keep the rule in production as an auditable fallback and as the baseline every future model must beat.
4. Close the loop: after four weeks, compare refreshed pages against the `visible_slip` baseline rate. That
   comparison — not this ranking — is what would let anyone talk about refresh *effects*.

## 8. Reproducibility

`SEED = 42` for every split, forest and bootstrap; logistic regression is deterministic under lbfgs.
From a fresh clone: `pip install -r requirements.txt`, put an `HF_TOKEN` in `.env` (gitignored), then run
`work/notebooks/capstone.ipynb` top to bottom — four month partitions on the first run (~2–4 minutes),
cached afterwards. Receipts: `work/outputs/capstone_metrics.json` (committed) holds every number quoted
here and on the paper; `work/outputs/capstone_action_queue.csv` is regenerated on every run and is
gitignored. The June partition was unread until the test and used once as the outcome month; there is no
second holdout.

## 9. Acknowledgments & data credit

Built on the **FlyRank ML Internship** dataset — <https://flyrank.ai>. The release is a pseudonymized,
gated warehouse snapshot; nothing client-identifying is used or shown.

---

**Claims checklist:** observed / measured / directional / decision-support language throughout · no causal
claims about refresh impact · no "predicted Google's algorithm" · base rate next to every precision number ·
numbers match `capstone_metrics.json`, which a fresh re-run reproduces (logistic + rule stable to three
decimals; tree ensembles can move a point or two across scikit-learn versions).
