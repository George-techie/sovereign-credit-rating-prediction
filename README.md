# Sovereign Credit Rating Prediction

This repository implements a machine learning pipeline for predicting sovereign credit rating categories (Default, Junk, Investment Grade) using a combination of macroeconomic indicators and sentiment features.
The project includes data processing, feature engineering, model training, and evaluation components, structured as a modular Python codebase.

It runs two experiments:

- **Experiment A (80 countries, yearly, 2011–2025):** next year's S&P rating from macroeconomic and fiscal fundamentals.
- **Experiment B (17 countries, monthly, 2018–2024):** the original sentiment-based model, re-examined.

## Project Structure

- `src/` — Core pipeline code (data processing, feature engineering, models, evaluation)
- `scripts/` — Data collection and experiment runners
- `tests/` — Unit tests for key components
- `notebooks/` — Exploratory analysis and experiment workflows (the original 17-country pipeline)
- `data/` — Input datasets
- `results/` — Model outputs and evaluation results (`results/annual/`, `results/monthly/` for the experiments below)
- `config/` — Configuration files

## Data

| Source | Coverage | File |
|---|---|---|
| World Bank WDI (growth, inflation, current account, reserves, unemployment, GDP per capita, trade, FDI) | 80 countries, 2000–2025 | `data/raw/macro/worldbank_macro.csv` |
| IMF World Economic Outlook, Oct 2026 (government debt and fiscal balance, % of GDP; IMF estimate years flagged) | 80 countries, 2000–2025; debt 97% filled, balance 99%; 2025 actual for 69 countries | `data/raw/macro/imf_weo_fiscal.csv` |
| S&P long-term foreign-currency sovereign ratings and outlooks, via countryeconomy.com | 71 of 80 countries; complete from 2011 | `data/raw/credit_ratings/sp_rating_history.csv`, `sp_ratings_annual.csv` |
| FX, 10-year yields, GDELT news tone, central-bank sentiment, consensus ratings | 17 countries, monthly 2018–2024 | `data/raw/...`, `data/processed/model_ready_features.csv` |

Rating classes follow `src/features.py`: Default (SD, D), Junk (BB+ to C), Investment Grade (BBB- and above).

Notes on the data:

- **S&P labels.** S&P's own rating-history pages block scripted access, so the labels come from countryeconomy.com's transcription of S&P actions. That source lists only 5–48 countries a year before 2011, so Experiment A starts in 2011. Not rated by S&P there: Algeria, Ethiopia, Libya, Mali, Namibia, Sudan, Tanzania, Vietnam, Zimbabwe. Ethiopia and Vietnam are in fact rated by S&P, so those two are gaps in the source. Spot checks match the known defaults: Argentina 2014/2019–20, Ecuador, Lebanon and Zambia 2020, Sri Lanka and Ghana 2022, Ukraine 2015/2022, Mozambique 2016, Cameroon 2023. One listed selective default, Bolivia in November 2017, is unverified.
- **Data vintage.** The macro data are the latest published vintage, not what was known at the time, so past years include later revisions. This is a common simplification, but it slightly flatters any model.

## Updating the data

```bash
python scripts/collect_worldbank.py                       # World Bank indicators (public API)
python scripts/parse_weo.py data/raw/macro/<WEO file>     # IMF debt and fiscal balance
python scripts/collect_sp_ratings.py --refresh            # S&P histories (3 s between requests)
```

The IMF also blocks scripted downloads. To update the IMF data, download the newest WEO dataset in a browser from the [IMF Data portal](https://data.imf.org/datasets/WEO), save it in `data/raw/macro/`, and run `parse_weo.py` on it. Both the portal's CSV and the classic tab-delimited `.xls` are read.

The October 2026 edition revised 350 country-years of debt by more than 2 points of GDP. The largest change is Senegal: 2023–2025 debt rose from about 81% to 121–136% of GDP, after a 2024 audit found debt that had not been reported.

## Models Implemented

- Ordered Logistic Regression
- XGBoost
- LSTM sequence model
- Persistence baseline (the rating stays the same), which every model is compared with

## Experiment A: next-year S&P rating, 80 countries

`python scripts/run_annual_experiment.py` builds the panel and writes `results/annual/`.

- **Setup.** Features from year *t* predict the rating at the end of *t*+1, split by the year predicted: train 2012–2017, validation 2018–2019, test 2020–2025.
- **Test set.** 71 countries, 420 country-years, of which only 12 change class.
- **Missing values.** Filled from the same country's earlier years only, then with training medians.

**Next-year class (test 2020–2025)**

| Model | Accuracy | Macro F1 | Accuracy on the 12 class changes |
|---|---|---|---|
| Persistence (same class) | **0.971** | **0.913** | 0.00 |
| XGBoost, fundamentals + current rating | 0.943 | 0.822 | 0.50 |
| Ordered logit, fundamentals + current rating | 0.941 | 0.891 | 0.08 |
| LSTM, fundamentals + current rating | 0.848 | 0.583 | 0.33 |
| XGBoost, fundamentals only | 0.843 | 0.576 | 0.50 |

**Downgrade / no change / upgrade (notch direction, 46 downgrades and 31 upgrades in test)**

| Model | Macro F1 | Downgrade recall | Upgrade recall |
|---|---|---|---|
| Persistence (no change) | 0.300 | 0.00 | 0.00 |
| XGBoost, fundamentals only | 0.399 | 0.17 | 0.13 |
| XGBoost, fundamentals + current rating | **0.466** | **0.30** | **0.23** |

**Actual rating minus what fundamentals imply (notches, test years)**

| Region | Countries | Mean residual | Clustered s.e. |
|---|---|---|---|
| Middle East | 6 | −2.27 | 1.12 |
| Africa | 21 | **−1.02** | 0.40 |
| Latin America | 13 | −0.58 | 0.71 |
| Europe & Central Asia | 15 | −0.12 | 0.45 |
| Asia-Pacific | 14 | +1.31 | 0.62 |
| North America | 2 | +1.61 | 0.49 |

What these results say:

- **No model beats persistence on the class.** Over 2020–2025, only 12 of 420 country-years changed class. A model that copies the current class is right 97% of the time, and no model beat that.
- **The useful signal is in the direction of the next notch move.** With the current rating and outlook included, XGBoost catches about a third of downgrades and a fifth of upgrades, where persistence catches none.
- **Africa is rated about one notch below what its fundamentals imply.** The gap is −1.0 notch, about 2.5 standard errors from zero. The Middle East is further below, but with only six countries the estimate is noisy. Asia-Pacific and North America sit above.
- **This gap is not by itself evidence of agency bias.** The fundamentals model leaves out institutional strength, governance and debt-payment history, which agencies weigh explicitly. The residual is what those factors and any bias explain *together*.

## Experiment B: the 17-country monthly model

`python scripts/run_monthly_experiment.py` writes `results/monthly/`. It uses the original split: train before 2020, validation 2020–2021, test 2022–2024.

**The original monthly labels do not change over time.** In `data/processed/model_ready_features.csv`, every country carries its 2024 consensus class for all 84 months. Ghana and Zambia, for example, are "Default" from January 2018, although Zambia defaulted in late 2020 and Ghana in December 2022. As a result:

- **The models were not predicting rating changes.** None exist in these labels, so the models could only learn which country a row belongs to.
- **The class lags leak the target.** `cls_lag1`–`cls_lag3` are the same constant as the target, so the original high accuracies in `results/*.csv` should not be read as predictive skill.

The experiment therefore compares three versions:

| Variant | What it is |
|---|---|
| As is | Original labels and features |
| No sentiment | Without the central-bank and market sentiment features |
| S&P labels | Month-end S&P class (16 countries; Ethiopia has no S&P history in the source), so ratings change when they really changed |

**Next-month class (test 2022–2024)**

| Model | As is | No sentiment | S&P labels |
|---|---|---|---|
| Persistence (last month's class) | **1.000** | **1.000** | **0.997** |
| XGBoost | 0.902 | 0.902 | 0.892 |
| Ordered logit | 0.691 | 0.775 | 0.811 |
| LSTM (first 12 test months per country need history, so fewer rows) | 0.858 | 0.902 | 0.875 |
| Class changes in the test period | 0 | 0 | 2 |

What these results say:

- **The original setup cannot measure predictive skill.** With constant labels, persistence is trivially perfect, and the models score below it.
- **Sentiment does not help.** Dropping the central-bank and market-sentiment features leaves XGBoost unchanged and improves the ordered logit and the LSTM.
- **Real monthly labels leave almost nothing to predict.** With S&P labels, the 16 countries change class only twice in 2022–2024, and no model catches a default.

At a monthly horizon with 16 countries there are almost no rating events to learn from. Experiment A's 80-country yearly panel is the more informative design.

The monthly sentiment inputs (GDELT tone and central-bank statements) end in 2024. Extending Experiment B needs those collected again and the FinBERT scoring rerun.

## Setup

Build and run using Docker:

```bash
docker build -t credit-rating .
docker run credit-rating
```

Or with conda (the pinned versions need Python 3.11):

```bash
conda create -n sovereign python=3.11
conda activate sovereign
pip install -r requirements.txt
```

## Running Tests

```bash
pip install -r requirements.txt
pytest tests/ -v
```
