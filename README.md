# Sovereign Credit Rating Prediction

ML pipeline predicting sovereign credit ratings (Default / Junk / Investment Grade)
for 17 countries with different economies using FinBERT sentiment + market signals.

## Models
- Ordered Logistic Regression (baseline)
- XGBoost (grid search tuned)
- LSTM with Attention (2-layer, 12-month lookback)

## Countries
**Africa:** South Africa, Kenya, Ghana, Egypt, Nigeria, Ethiopia, Botswana, Morocco, Zambia  
**Benchmark:** United States, United Kingdom, Japan, Brazil, Germany, India, China, Mexico

## Notebooks — Run in Order
| Notebook | Description |
|----------|-------------|
| 01_data_download.ipynb | Downloads all raw data |
| 02_feature_engineering.ipynb | Builds feature matrix |
| 03_model_training.ipynb | Trains all 3 models |
| 04_evaluation_bias_analysis.ipynb | Evaluation + bias analysis |

## Author
George Nyatangi
