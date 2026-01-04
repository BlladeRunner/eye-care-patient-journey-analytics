# Executive Summary — Eye Care Patient Journey Analytics

## Project Overview

This project analyzes the **end-to-end patient journey in eye care**, from clinical recommendation to purchase, retention, and downstream revenue.

The analysis is built on a **fully synthetic, privacy-safe dataset** designed to replicate real-world healthcare and consumer behavior while remaining fully reproducible.

The primary goal is to **quantify the business impact of clinical recommendations** and identify which patient segments, product tiers, and channels drive the highest adoption and value.

---

## Key Business Questions

- Do clinical recommendations increase product adoption?
- How large is the adoption uplift compared to a control group?
- Does higher adoption translate into higher short-term revenue?
- Which segments (market type, tier, channel) benefit the most?
- Do adopters remain active over time?

---

## Data & Methodology

- **Dataset:** Fully synthetic, relational healthcare-style data  
- **Population:** ~50,000 patients  
- **Key entities:** patients, visits, recommendations, transactions, outcomes  
- **Design:** Recommended vs control comparison (observational uplift)

### Key Methods
- Funnel analysis (recommendation → adoption)
- 30-day adoption window
- 60-day revenue attribution window
- Cohort-based retention analysis
- Bootstrap-based uplift estimation with confidence intervals

---

## Key Findings

### 1. Recommendation → Adoption Funnel
- Adoption is **not guaranteed after recommendation**, confirming meaningful funnel drop-off.
- Recommendations are **strongly associated with higher adoption** within 30 days compared to control.

### 2. Revenue Impact
- Patients who receive recommendations generate **higher 60-day revenue on average**.
- This indicates measurable business value beyond conversion alone.

### 3. Segment-Level Uplift
- Adoption uplift varies significantly by **market type** (e.g. developed vs emerging).
- Uplift also differs by **recommended product tier / category**, suggesting opportunities for:
  - better tier alignment
  - more targeted recommendations

### 4. Retention Behavior
- Cohort analysis shows that **adopters remain active longer** than non-adopters.
- Retention curves indicate sustained engagement beyond the initial purchase window.

### 5. Channel Effects (Important Limitation)
- Channel uplift among purchasers could **not be reliably estimated** due to:
  - lack of a true control group within post-purchase channels
- This highlights a **data design limitation**, not a modeling issue.

---

## Business Implications

- Clinical recommendations deliver **clear adoption and revenue uplift**.
- Not all patients benefit equally — **segmentation matters**.
- Product tier and market context should influence recommendation strategy.
- Retention impact suggests recommendations drive **longer-term value**, not just one-off sales.

---

## Recommended Actions

1. **Targeting**
   - Prioritize segments with the highest adoption uplift and positive revenue impact.

2. **Product Strategy**
   - Adjust recommended tier mix by segment to maximize conversion and retention.

3. **Channel Strategy**
   - Improve tracking of **pre-purchase channel exposure** to enable true channel uplift analysis.

4. **Measurement Plan**
   - Track adoption, revenue, and retention monthly.
   - Validate findings on new cohorts to ensure stability over time.

---

## Limitations & Next Steps

- Observational uplift (not randomized experiment)
- Synthetic data used for demonstration purposes
- Channel attribution requires upstream exposure data

**Next Steps**
- Validate methodology on real-world or anonymized datasets
- Introduce pre-recommendation channel features
- Extend analysis to longer revenue and retention windows

---

## Why This Project Matters

This project demonstrates the ability to:
- Design realistic analytical datasets
- Build end-to-end analytics pipelines
- Apply causal-style reasoning in non-experimental data
- Translate analysis into clear business decisions

It reflects how data analysts operate in real healthcare and product analytics environments.