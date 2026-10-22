"""
weo.py — Read an IMF World Economic Outlook "entire dataset" file.

Two download formats are supported, both with one row per country x indicator
and one column per year:

- Classic ".xls" files from imf.org (up to the October 2024 edition): tab-
  separated text in UTF-16 little-endian without a byte-order mark. Read as
  UTF-8 they "load" as garbage columns, which is why the first debt parser
  found nothing. "Estimates Start After" gives the last year of actual data.
- CSV files from the IMF Data portal (from 2025): the country and indicator
  share one SERIES_CODE such as "KEN.GGXWDG_NGDP.A", and the last actual year
  is LATEST_ACTUAL_ANNUAL_DATA.

Both are returned with the classic column names, so later years are flagged as
IMF estimates or projections the same way.
"""

import pandas as pd

from .countries import COUNTRY_BY_ISO3

DEBT_CODE = "GGXWDG_NGDP"  # general government gross debt, % of GDP
_MISSING = {"", "n/a", "--", "nan", "NaN"}


def read_weo(path):
    """The WEO table with string cells and the classic column names, either format."""
    if _is_portal_csv(path):
        return _read_portal_csv(path)
    df = pd.read_csv(path, sep="\t", encoding="utf-16-le", dtype=str, on_bad_lines="skip")
    df.columns = [str(c).strip().lstrip("﻿") for c in df.columns]
    return df[df["ISO"].notna() & df["WEO Subject Code"].notna()]


def _is_portal_csv(path):
    with open(path, "rb") as head:
        start = head.read(200).decode("utf-8-sig", errors="ignore")
    return "SERIES_CODE" in start


def _read_portal_csv(path):
    df = pd.read_csv(path, dtype=str, encoding="utf-8-sig")
    parts = df["SERIES_CODE"].str.split(".", expand=True)
    return df.assign(**{
        "ISO": parts[0],
        "WEO Subject Code": parts[1],
        "Country": df["COUNTRY"],
        "Estimates Start After": df["LATEST_ACTUAL_ANNUAL_DATA"],
    })


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
