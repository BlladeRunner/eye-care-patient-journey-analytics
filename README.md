# Eye Care Patient Journey & Adoption Analytics

## Overview
This project analyzes the **patient journey in eye care** to understand how patients adopt and continue using vision care products after clinical recommendations.

The focus is on:
- product adoption after recommendation
- retention and dropout behavior
- differences across markets, channels, and product tiers
- translating analytical insights into **business-ready recommendations**

The project is inspired by real-world challenges in global eye care and health-tech analytics.

## Project Goals
- Understand patient population and visit behavior
- Measure adoption rates after product recommendation
- Analyze retention using cohort-based approaches
- Identify key drivers of dropout and sustained usage
- Provide actionable insights for improving patient outcomes and access

## Dataset
The project uses a **synthetic dataset** generated in Python to simulate realistic eye care scenarios while avoiding sensitive personal or medical data.

Generated tables include:
- patients
- visits
- recommendations
- transactions
- outcomes

All data is reproducible via a data generation script.

## Repository Structure
- data/ synthetic raw and processed data
- notebooks/ analytical notebooks (step-by-step narrative)
- src/ reusable Python modules (data generation, metrics, cohorts)
- reports/ executive summary and business recommendations


## Analysis Workflow
1. Data generation and validation
2. Patient landscape and visit behavior analysis
3. Adoption funnel: recommendation → first purchase
4. Retention and cohort analysis
5. Business interpretation and recommendations

## How to Run
1. Create a virtual environment and install dependencies:
```bash
python -m pip install -r requirements.txt

2. Generate synthetic data:
python -m src.data_gen

3. Run notebooks in order:
- 02_data_understanding.ipynb
- 03_funnel_adoption.ipynb
- 04_retention_cohorts.ipynb
- 05_recommendations.ipynb