"""
collect_sp_ratings.py
---------------------
Collects S&P long-term foreign-currency sovereign ratings for the 80 countries
in data/raw/macro/worldbank_macro.csv, from countryeconomy.com, and writes:

    data/raw/credit_ratings/sp_rating_history.csv   every rating/outlook action
    data/raw/credit_ratings/sp_ratings_annual.csv   rating in force each 31 Dec

Pages are cached in data/raw/credit_ratings/countryeconomy_html/ (not in git),
so rerunning parses offline; pass --refresh to download them again.
countryeconomy.com's robots.txt allows /ratings/; requests are spaced 3 s apart.

Run from the repository root:
    python scripts/collect_sp_ratings.py [--refresh]
"""

import argparse
import sys
import time
from pathlib import Path

import pandas as pd
import requests

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.ratings import parse_sp_history, year_end_ratings  # noqa: E402

BASE_URL = "https://countryeconomy.com/ratings/"
COUNTRIES_CSV = Path("data/raw/macro/worldbank_macro.csv")
OUT_DIR = Path("data/raw/credit_ratings")
CACHE_DIR = OUT_DIR / "countryeconomy_html"
FIRST_YEAR, LAST_YEAR = 2000, 2025
DELAY_S = 3.0
HEADERS = {"User-Agent": "Mozilla/5.0 (research; sovereign-credit-rating-prediction)"}

# Our country name -> countryeconomy.com slug, where the slug is not simply the
# lower-cased, hyphenated name.
SLUG_OVERRIDES = {
    "Cape Verde": "cabo-verde",
    "Democratic Republic of Congo": "democratic-republic-congo",
}


def slug_for(country):
    return SLUG_OVERRIDES.get(country, country.lower().replace(" ", "-"))


def fetch_page(country, refresh):
    path = CACHE_DIR / f"{slug_for(country)}.html"
    if path.exists() and not refresh:
        return path.read_text(encoding="utf-8")
    response = requests.get(BASE_URL + slug_for(country), headers=HEADERS, timeout=30)
    response.raise_for_status()
    path.write_text(response.text, encoding="utf-8")
    time.sleep(DELAY_S)
    return response.text


def main():
    parser = argparse.ArgumentParser(description="Collect S&P sovereign rating histories")
    parser.add_argument("--refresh", action="store_true", help="download pages again")
    args = parser.parse_args()

    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    countries = sorted(pd.read_csv(COUNTRIES_CSV)["country"].unique())
    histories, annual, failed = [], [], []
    for country in countries:
        try:
            history = parse_sp_history(fetch_page(country, args.refresh))
        except requests.RequestException as error:
            failed.append(f"{country}: {error}")
            continue
        if history.empty:
            failed.append(f"{country}: no S&P rating on the page")
            continue
        histories.append(history.assign(country=country))
        annual.append(
            year_end_ratings(history, range(FIRST_YEAR, LAST_YEAR + 1)).assign(country=country)
        )
        print(f"  {country:<30} {len(history):>3} actions, first {history['date'].min():%Y-%m-%d}")

    history_df = pd.concat(histories, ignore_index=True)[["country", "date", "rating", "outlook"]]
    annual_df = pd.concat(annual, ignore_index=True)[
        ["country", "year", "rating", "outlook", "notch", "rating_class"]
    ]
    history_df.to_csv(OUT_DIR / "sp_rating_history.csv", index=False)
    annual_df.to_csv(OUT_DIR / "sp_ratings_annual.csv", index=False)

    rated = annual_df["rating"].notna()
    print(f"\nCountries with S&P history: {history_df['country'].nunique()} / {len(countries)}")
    print(f"Rated country-years {FIRST_YEAR}-{LAST_YEAR}: {rated.sum()} / {len(annual_df)}")
    print(annual_df.loc[rated, "rating_class"].value_counts().sort_index().to_string())
    if failed:
        print("\nNot collected:\n  " + "\n  ".join(failed))


if __name__ == "__main__":
    main()
