# Sovereign Credit Rating Prediction

ML pipeline for predicting sovereign credit ratings (Default / Junk / Investment Grade)
for 17 African and emerging market countries using FinBERT sentiment + market signals.

## Models
- Ordered Logistic Regression (baseline)
- XGBoost (grid search tuned)
- LSTM with Attention (2-layer, 12-month lookback)

## Countries
**Africa:** South Africa, Kenya, Ghana, Egypt, Nigeria, Ethiopia, Botswana, Morocco, Zambia  
**Benchmark:** United States, United Kingdom, Japan, Brazil, Germany, India, China, Mexico

## Features
- FinBERT sentiment from central bank statements (S_CB)
- FinBERT sentiment from market news (S_MKT)
- Bond yield changes, FX returns
- Macro indicators: GDP growth, inflation, debt/GDP, reserves

## Notebooks
| Notebook | Description |
|----------|-------------|
| 01_data_download.ipynb | Downloads all raw data |
| 02_feature_engineering.ipynb | Builds feature matrix |
| 03_model_training.ipynb | Trains all 3 models |
| 04_evaluation_bias_analysis.ipynb | Evaluation + Africa bias analysis |

## Results
See `results/` folder for confusion matrices, ROC curves, bias analysis plots.

## Author
George Simei — AfterQuery ML Project 2025
