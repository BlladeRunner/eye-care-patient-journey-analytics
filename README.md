# Eye Care Patient Journey Analytics

End-to-end analytics project analyzing the **patient journey in eye care** — from clinical recommendation to purchase, retention, and downstream revenue.

The project is built on a **fully synthetic, privacy-safe dataset** and demonstrates how data analytics can quantify recommendation impact and support business decisions in healthcare-like environments.

---

## Project Goals

- Measure adoption uplift from clinical recommendations  
- Quantify short-term revenue impact (60-day window)  
- Analyze retention behavior using cohort analysis  
- Identify high-performing segments by market, tier, and channel  
- Translate analytics into clear business actions  

---

## Dataset

- Fully **synthetic and reproducible**
- Healthcare-style relational schema
- ~50k patients
- No sensitive or proprietary data

**Core tables**
- `patients`
- `visits`
- `recommendations`
- `transactions`
- `outcomes`

All raw data is generated via notebook **01_generate_data.ipynb**.

---

## Analysis Structure

| Notebook | Purpose |
|--------|--------|
| `01_generate_data.ipynb` | Synthetic data generation (reproducible) |
| `02_data_understanding.ipynb` | Data validation & exploration |
| `03_funnel_adoption.ipynb` | Recommendation → adoption funnel |
| `04_retention_cohorts.ipynb` | Cohort-based retention analysis |
| `05_recommendations.ipynb` | Uplift & business impact analysis |

---

## Key Findings

- Clinical recommendations are **strongly associated with higher adoption** within 30 days  
- Recommended patients generate **higher 60-day revenue**  
- Adoption uplift varies by **market type and product tier**  
- Retention analysis shows **longer engagement among adopters**  
- Channel uplift could not be reliably estimated due to control-group limitations (documented)

---

## Business Impact

- Demonstrates measurable value of recommendations beyond conversion
- Highlights where segmentation improves effectiveness
- Shows how analytics can guide:
  - targeting strategy
  - product tier alignment
  - retention optimization

---

## Tools & Skills

- Python (pandas, numpy, matplotlib)
- Cohort & funnel analysis
- Bootstrap-based uplift estimation
- Business-oriented storytelling
- Reproducible analytics design

---

## Executive Summary

A full business-facing executive summary is available in:

---

## Interview Pitch (30 sec)

> “I built an end-to-end patient journey analytics project for eye care.  
> I generated a realistic synthetic dataset, analyzed adoption as a funnel from recommendation to purchase, measured retention using cohorts, and quantified recommendation uplift on adoption and revenue.  
> The result is a clear set of business actions around targeting, product tier strategy, and measurement.”

---

## Notes

This project mirrors real-world healthcare analytics constraints:
- privacy-first data design
- observational uplift (non-experimental)
- focus on decision-making, not just metrics

🔙 [Back to Portfolio](https://github.com/BlladeRunner)
