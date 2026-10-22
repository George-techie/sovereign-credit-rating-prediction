"""
weo.py — Read an IMF World Economic Outlook "entire dataset" file.

The IMF's ".xls" WEO downloads are tab-separated text in UTF-16 little-endian,
without a byte-order mark: read as UTF-8 they "load" as garbage columns, which
is why the first debt parser found nothing. Rows are country x indicator,
columns are years, and "Estimates Start After" gives the last year of actual
data for each row: later years are IMF estimates or projections.
"""

import pandas as pd

from .countries import COUNTRY_BY_ISO3

DEBT_CODE = "GGXWDG_NGDP"  # general government gross debt, % of GDP
_MISSING = {"", "n/a", "--", "nan", "NaN"}


def read_weo(path):
    """The WEO table with string cells; the trailing footnote rows are dropped."""
    df = pd.read_csv(path, sep="\t", encoding="utf-16-le", dtype=str, on_bad_lines="skip")
    df.columns = [str(c).strip().lstrip("﻿") for c in df.columns]
    return df[df["ISO"].notna() & df["WEO Subject Code"].notna()]


def extract_indicator(weo, code, first_year, last_year):
    """Long table country, iso3, year, value, is_estimate for our 80 countries."""
    rows = weo[(weo["WEO Subject Code"] == code) & weo["ISO"].isin(COUNTRY_BY_ISO3)]
    records = []
    for _, row in rows.iterrows():
        actual_until = _year(row.get("Estimates Start After"))
        for year in range(first_year, last_year + 1):
            value = _number(row.get(str(year)))
            records.append({
                "country": COUNTRY_BY_ISO3[row["ISO"]],
                "iso3": row["ISO"],
                "year": year,
                "value": value,
                "is_estimate": actual_until is not None and year > actual_until,
            })
    return pd.DataFrame(records, columns=["country", "iso3", "year", "value", "is_estimate"])


def _number(text):
    if text is None or str(text).strip() in _MISSING:
        return None
    try:
        return float(str(text).replace(",", ""))
    except ValueError:
        return None


def _year(text):
    value = _number(text)
    return int(value) if value is not None else None
