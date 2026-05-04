# Sovereign Credit Rating Prediction

This repository implements a machine learning pipeline for predicting sovereign credit rating categories (Default, Junk, Investment Grade) using a combination of macroeconomic indicators and sentiment features.
The project includes data processing, feature engineering, model training, and evaluation components, structured as a modular Python codebase.

## Project Structure

- `src/` — Core pipeline code (data processing, feature engineering, models, evaluation)
- `tests/` — Unit tests for key components
- `notebooks/` — Exploratory analysis and experiment workflows
- `data/` — Input datasets (not included or partially included)
- `results/` — Model outputs and evaluation results
- `config/` — Configuration files

## Models Implemented

- Ordered Logistic Regression
- XGBoost
- LSTM (attention-based sequence model)

## Setup

Build and run using Docker:

```bash
docker build -t credit-rating .
docker run credit-rating
```

## Running Tests

```bash
pip install -r requirements.txt
pytest tests/ -v
```
