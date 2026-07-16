![logo_ironhack_blue](https://user-images.githubusercontent.com/23629340/40541063-a07a0a8a-601a-11e8-91b5-2f13e4e6b441.png)

# Automated Customer Reviews
### NLP-Powered Sentiment Classification, Product Clustering & Generative AI Summarisation

**Team:** Lydia · Don · Marcelo
**Ironhack AI Engineering Bootcamp — 2026**

> The original assignment brief provided by Ironhack is kept for reference in [Readme_exercise.md](Readme_exercise.md). This document describes what was actually built.

---

## 1. Project Summary

This project delivers an end-to-end NLP pipeline applied to Amazon product reviews, addressing the three core tasks required by the business case:

1. **Sentiment Classification** — classify reviews as positive, neutral or negative using fine-tuned Transformer models.
2. **Product Category Clustering** — group raw product categories into business-relevant meta-categories using sentence embeddings and K-Means.
3. **Generative AI Summarisation** — produce sentence-grounded, quality-audited review-intelligence articles per meta-category using BART.

The pipeline was built on **129,000 balanced reviews** from three complementary datasets. Standard off-the-shelf sentiment models are binary (positive/negative) and structurally cannot predict a neutral class, so DistilBERT and RoBERTa were fine-tuned on a balanced 3-class dataset. The results power a dual-mode **Gradio dashboard** with separate consumer and seller workflows.

**Production model:** DistilBERT fine-tuned — **82.95% Macro F1**, selected for its 2.5x inference speed advantage over RoBERTa v2 at a cost of only 0.95 percentage points of accuracy.

---

## 2. Repository Structure

```
.
├── README.md                                            # this file
├── Readme_exercise.md                                  # original Ironhack assignment brief
├── project_nlp_business_case_automated_customers_reviews_FinalVersion.ipynb   # final pipeline notebook
├── Automated_Customer_Reviews_Report_Updated_Clustering_Summarisation.docx    # full technical report
├── Automated_Customer_Reviews_Presentation_FINAL.pptx  # stakeholder presentation
├── consumer-reviews-of-amazon-products-metadata.json   # Croissant metadata for the Kaggle dataset
├── EDA/                                                 # exploratory data analysis notebooks + charts
│   ├── 01_Primary_Dataset_EDA.ipynb
│   ├── Merged Dataset _ eda.ipynb
│   ├── eda_amazon_reviews.ipynb
│   └── eda_outputs/                                     # exported plots (rating/sentiment distributions, wordclouds, etc.)
└── data/
    ├── primary/                                         # Datafiniti/Kaggle Amazon reviews (backbone dataset)
    ├── large/                                            # sampled McAuley-Lab Amazon Reviews 2023
    └── merged/                                           # cleaned, merged dataset used for modelling
```

Large CSVs are gitignored; see [Readme_exercise.md](Readme_exercise.md) for original dataset sources.

---

## 3. Data & Preprocessing

Three complementary sources were combined:

| Source | Role | Notes |
|---|---|---|
| **Primary — Datafiniti/Kaggle** (`1429_1.csv`) | Backbone: clean product names + category labels | 34,660 reviews, ~93% positive, ~4% neutral, ~2% negative — severely imbalanced on its own |
| **McAuley-Lab Amazon Reviews 2023** | Augmentation (neutral + negative only) | Streamed via HuggingFace Datasets, filtered at read time; still insufficient alone to close the class gap |
| **HuggingFace `SetFit/amazon_reviews_multi_en`** | Second augmentation pass (neutral + negative only) | Closed the remaining gap to reach the balance target |

**Pipeline steps:** text cleaning (lowercasing, HTML/URL stripping, punctuation normalisation, drop reviews <10 chars) → star-rating → sentiment mapping (1–2★ negative, 3★ neutral, 4–5★ positive) → schema alignment across sources → stratified 80/20 train/test split **before** augmentation (so the test set reflects real-world class distribution) → downsampling to a perfect 1:1:1 balance.

**Final dataset:** 129,000 reviews — exactly 43,000 per sentiment class, 35,073 unique products, 74 raw categories.

---

## 4. Task 1 — Sentiment Classification

**Problem:** off-the-shelf sentiment models (e.g. `distilbert-base-uncased-finetuned-sst-2-english`) are binary and cannot predict "neutral". This was empirically confirmed: zero-shot baseline scored **0% recall on the neutral class**.

Both models were adapted by replacing the pretrained 2-class head with a fresh 3-class head (`AutoModelForSequenceClassification`, `num_labels=3`) and fine-tuned with the HuggingFace `Trainer` API.

| Model | Accuracy | Macro F1 | Weighted F1 | Params | Notes |
|---|---|---|---|---|---|
| DistilBERT zero-shot (binary) | 61.00% | 0.4900 | 0.4900 | 66M | 0% recall on neutral |
| **DistilBERT fine-tuned** | **82.97%** | **0.8295** | **0.8295** | 66M | **Selected for production** |
| RoBERTa v1 fine-tuned | 83.94% | 0.8389 | 0.8389 | 125M | lr=2e-5, batch=8 |
| RoBERTa v2 fine-tuned | 83.94% | 0.8390 | 0.8390 | 125M | lr=1e-5, batch=16 — best overall |
| RoBERTa v3 fine-tuned | 83.53% | 0.8347 | 0.8347 | 125M | + warmup + fp16 + dynamic padding — regressed |

**Key findings:**
- Fine-tuning closes the gap dramatically (Macro F1 0.49 → 0.83+), with neutral-class recall going from 0% to ~75%.
- RoBERTa v2 is the highest-accuracy configuration, but the gain over DistilBERT (+0.95pp Macro F1) does not justify 2x the parameters and ~2.5x the training time for a real-time review platform.
- Performance ceiling sits around ~84% due to label noise: mapping star ratings to sentiment is an imperfect proxy (a 3-star "great product but terrible shipping" review is genuinely ambiguous, even for a human annotator).

**Production decision:** **DistilBERT fine-tuned** — best balance of accuracy, latency (~2.5x faster than RoBERTa) and memory footprint (~250MB vs ~500MB). Saved to `outputs/baseline_distilbert/` and reused by the Gradio app.

---

## 5. Task 2 — Product Category Clustering

Clustering was performed on the **primary dataset** (stable product names and category labels), at the **category level**, not per individual review — this avoids high-volume categories dominating the embedding space and produces business-interpretable clusters.

**Pipeline:**
1. Filter to 41 usable raw categories (67,959 reviews).
2. Build an aggregated profile per category: name, top 5 products, sentiment shares, sample of up to 50 reviews.
3. Embed sampled reviews with `sentence-transformers/all-MiniLM-L6-v2` — reviews are chunked (limited token window) and chunk embeddings are mean-pooled + L2-normalised into one 384-dim vector per category.
4. Run K-Means for k = 4, 5, 6 with a fixed random seed, using silhouette score to guide the choice.
5. Manually name and validate each cluster against category, product and sample-review inspection (with an assertion guarding against silent mismatches).

**K selection:** k=6 had the numerically highest silhouette score (0.2430) vs. k=5 (0.2329), but the marginal 0.0101 gain produced extra fragmentation without a comparable improvement in business interpretability — **k = 5 was selected deliberately.**

**Final meta-categories:**

| Cluster | Meta-category | Scale |
|---|---|---|
| 0 | Kindle, Tablets, Batteries and Office Products | Largest — 53,284 reviews / 15 categories |
| 1 | Small Accessories, Cases, Stands and Pet Supplies | Low support — only 102 reviews / 11 categories |
| 2 | Fire TV, Refurbished and Promotional Electronics | — |
| 3 | Echo, Alexa and Home Entertainment | — |
| 4 | Chargers, Cables and Device Accessories | — |

**Limitation:** the "Small Accessories" cluster has very few supporting reviews and should be interpreted more cautiously than the high-volume clusters.

---

## 6. Task 3 — Generative AI Summarisation

Rather than asking a generative model to freely write a recommendation article (which risked mixing unrelated products or copying instructions), the final pipeline **separates deterministic analytics from generation**:

1. Map each review to its meta-category; clean and deduplicate text.
2. Rank products with a Bayesian-adjusted rating that accounts for review volume.
3. Split reviews into sentence-level evidence; classify each sentence positive/neutral/negative with the production **DistilBERT** model.
4. Select representative sentences per sentiment via **TF-IDF centrality**, preserving diversity across products and categories.
5. Use **`facebook/bart-large-cnn`** only to compress the selected evidence into 3 sections: *What customers value*, *Mixed feedback*, *Common concerns* — mapped to a controlled business-theme taxonomy.
6. Run a **mandatory quality audit** before export: checks that sentiment percentages sum to 100%, sections are complete and distinct, themes belong to the controlled taxonomy, summaries stay grounded in evidence, and no prompt text leaked into the output. Export is blocked on failure.

**Output:** 5 validated review-intelligence articles (15 sentiment-specific sections total), all passing the audit. Saved to:
- `results/summarization/generated_articles.csv` — permanent project result
- `data/app/generated_articles.csv` — direct input for the Gradio app

When no recurring theme is supported by the evidence, the system reports "insufficient recurring evidence" rather than fabricating an insight.

---

## 7. Deployment — Gradio Review Intelligence Dashboard

The dashboard is built with **Gradio Blocks**, launched from Google Colab (mounted to Google Drive) with a public `share=True` link. All heavy computation (training, clustering, summarisation) runs **offline in the notebook** — the app only loads pre-exported artefacts (`data/app/`) plus the fine-tuned DistilBERT model at runtime, keeping the interface fast and its analytical outputs reproducible.

**Consumer view:** pick a plain-language product group → get sentiment distribution, average rating, review count, controlled value/concern themes, and representative review excerpts.

**Seller view** (4 tabs):
- **Overview** — portfolio volume, weighted sentiment distribution, strongest/highest-priority meta-categories.
- **Category Intelligence** — 3D PCA map of clusters + audited summaries from `generated_articles.csv`.
- **Product Performance** — top/bottom-ranked products with Bayesian smoothing against low-review-count products.
- **Review Analyzer** — the only live-inference workflow: type a review, get the DistilBERT prediction, class probabilities and confidence.

> Note: the current deployment relies on an ephemeral Colab share link, not a persistent public host (see [Future Work](#9-future-improvements)).

---

## 8. How to Reproduce

1. Open [project_nlp_business_case_automated_customers_reviews_FinalVersion.ipynb](project_nlp_business_case_automated_customers_reviews_FinalVersion.ipynb) in Colab or Jupyter.
2. Run **Section 1** to install dependencies (`pandas`, `numpy`, `scikit-learn`, `transformers`, `datasets`, `evaluate`, `accelerate`, `sentence-transformers`, `gradio`, `plotly`).
3. Run sections in order — each is numbered per the in-notebook roadmap (Acquisition → Cleaning → Label Creation → EDA → Balancing → Baseline → Comparison → Evaluation → Clustering → Summarisation → App).
4. Section 12 exports the deployment artefacts to `data/app/` (review sample, cluster table with PCA coordinates, weighted cluster profiles, validated summaries).
5. Launch the Gradio app (`app.py`, generated alongside the exported artefacts) with `share=True` for a public link, or run it locally against the same `data/app/` artefacts.

---

## 9. Future Improvements

Directly identified during evaluation and documented in the technical report:

- **Benchmark against an instruction-tuned LLM** for the summarisation step, while retaining grounding checks and offline pre-generation to control cost.
- **Expand the dataset beyond Amazon-branded products** to improve clustering diversity and generalisation.
- **Explore human-annotated sentiment labels** to bypass star-rating noise and raise the ~84% Macro F1 ceiling.
- **Real-time review ingestion** to turn the dashboard into a live aggregator instead of a static snapshot.
- **Persistent public deployment** (e.g. Hugging Face Spaces) instead of the current ephemeral Colab share link.

## 10. Known Limitations

- Dataset composition is exclusively Amazon-branded products — meta-categories may not generalise to other retailers.
- Star-rating → sentiment mapping is a pragmatic approximation; ambiguous 3-star reviews cap achievable accuracy.
- BART is a document summariser, not an instruction-following BI model — the controlled theme layer trades some linguistic freedom for reliability.
- The "Small Accessories" cluster (102 reviews) has materially lower statistical support than the other four clusters.

---

## 11. Deliverables

| Deliverable | File |
|---|---|
| Final notebook (source code) | [project_nlp_business_case_automated_customers_reviews_FinalVersion.ipynb](project_nlp_business_case_automated_customers_reviews_FinalVersion.ipynb) |
| Technical report | [Automated_Customer_Reviews_Report_Updated_Clustering_Summarisation.docx](Automated_Customer_Reviews_Report_Updated_Clustering_Summarisation.docx) |
| Stakeholder presentation | [Automated_Customer_Reviews_Presentation_FINAL.pptx](Automated_Customer_Reviews_Presentation_FINAL.pptx) |
| EDA notebooks & charts | [EDA/](EDA/) |
| Original assignment brief | [Readme_exercise.md](Readme_exercise.md) |

---

## Team & Credits

**Lydia · Don · Marcelo** — Ironhack NLP Business Case Project, July 2026.
