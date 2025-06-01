# Kaggle ML & Data Science Trends Forecast (2025 Hackathon)

This repository contains our project submission for the **2025 Machine Learning Hackathon**.  
Our task was to analyze the **Kaggle Machine Learning & Data Science Survey (2017–2025)** and forecast the **most popular tools, libraries, and job roles** for **2026** using time-series modeling.

---

## Problem Statement

- Analyze trends from past Kaggle ML survey datasets
- Forecast future usage/popularity of:
  - Programming tools (e.g., Jupyter, VS Code)
  - ML libraries (e.g., TensorFlow, scikit-learn)
  - Job roles (e.g., Data Scientist, ML Engineer)
- Use time-series or regression-based modeling approaches to generate predictions

---

## Approach

We used a **Linear Regression model** (`sklearn.linear_model.LinearRegression`) to fit and forecast yearly trends for each tool/library/job role based on their usage percentages.

---

## Repository Structure

```bash
   data/
   ├── kaggle_survey_2017.csv
   ├── ...
   └── kaggle_survey_2025.csv
 Analysis.py                             # Main notebook with data analysis
 Matplotlib forecast.py                  # Most popular Library forecast
 Programming language forecast.py        # Most popular tool forecast
 Student forecast.py                     # Most polpular role forecast
 README.md                               # This file
