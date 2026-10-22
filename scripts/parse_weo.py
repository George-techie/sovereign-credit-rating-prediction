"""
parse_weo.py
------------
Extracts general government gross debt and the fiscal balance (% of GDP) for
the 80 countries from an IMF WEO "entire dataset" file, into
data/raw/macro/imf_weo_fiscal.csv. The World Bank file is left untouched; the
two are joined when the panel is built (src/annual_panel.py).

The IMF site blocks scripted downloads, so get the newest edition in a browser:
IMF Data portal -> World Economic Outlook -> download the entire dataset, by
countries, tab-delimited. Then run, from the repository root:
    python scripts/parse_weo.py data/raw/macro/WEOOct2024all.xls
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.weo import DEBT_CODE, extract_indicator, read_weo  # noqa: E402

FISCAL_BALANCE_CODE = "GGXCNL_NGDP"  # general government net lending/borrowing, % of GDP
OUTPUT = Path("data/raw/macro/imf_weo_fiscal.csv")
FIRST_YEAR, LAST_YEAR = 2000, 2025


def main():
    parser = argparse.ArgumentParser(description="Extract IMF WEO fiscal indicators")
    parser.add_argument("weo_file", type=Path)
    args = parser.parse_args()

    weo = read_weo(args.weo_file)
    debt = extract_indicator(weo, DEBT_CODE, FIRST_YEAR, LAST_YEAR)
    balance = extract_indicator(weo, FISCAL_BALANCE_CODE, FIRST_YEAR, LAST_YEAR)
    fiscal = debt.rename(columns={"value": "imf_debt_to_gdp", "is_estimate": "imf_is_estimate"})
    fiscal = fiscal.merge(
        balance[["iso3", "year", "value"]].rename(columns={"value": "imf_fiscal_balance"}),
        on=["iso3", "year"],
        how="outer",
    )
    fiscal["weo_edition"] = args.weo_file.stem
    fiscal = fiscal.sort_values(["country", "year"])
    fiscal.to_csv(OUTPUT, index=False)

    print(f"Saved {len(fiscal)} rows from {args.weo_file.name} -> {OUTPUT}")
    for column in ["imf_debt_to_gdp", "imf_fiscal_balance"]:
        share = fiscal[column].notna().mean()
        print(f"  {column}: {share:.0%} of country-years filled")
    print(f"  marked as IMF estimates: {fiscal['imf_is_estimate'].mean():.0%}")


if __name__ == "__main__":
    main()
