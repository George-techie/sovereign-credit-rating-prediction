"""
ratings.py — S&P sovereign rating histories: parsing and year-end labels.

Source: the S&P tables on countryeconomy.com (one page per country), which list
every change to the long-term foreign-currency rating, the standard sovereign
rating, with its date and outlook. Rows that change only the outlook, such as
"(Negative)", leave the rating unchanged, so the rating is carried forward.

All functions are pure: the network fetch lives in scripts/collect_sp_ratings.py.
"""

import re
from html.parser import HTMLParser

import numpy as np
import pandas as pd

from .features import rating_to_class

# S&P long-term scale, best first. A notch is one step; default (SD/D) is 0.
SP_SCALE = [
    "AAA", "AA+", "AA", "AA-", "A+", "A", "A-",
    "BBB+", "BBB", "BBB-", "BB+", "BB", "BB-", "B+", "B", "B-",
    "CCC+", "CCC", "CCC-", "CC", "C",
]
SP_DEFAULT = ["SD", "D"]
SP_NOTCH = {rating: len(SP_SCALE) - i for i, rating in enumerate(SP_SCALE)}
SP_NOTCH.update({rating: 0 for rating in SP_DEFAULT})

_RATING = re.compile(r"^\s*([A-Z]{1,3}[+-]?)?\s*(?:\(([^)]*)\))?\s*$")


class _SectionTables(HTMLParser):
    """Collects the rows of the first table inside the element with id `section_id`."""

    def __init__(self, section_id):
        super().__init__()
        self.section_id = section_id
        self.depth = 0          # div depth inside the section, 0 = outside
        self.in_table = False
        self.done = False
        self.rows, self.row, self.cell = [], None, None

    def handle_starttag(self, tag, attrs):
        if self.done:
            return
        if self.depth == 0:
            if tag == "div" and dict(attrs).get("id") == self.section_id:
                self.depth = 1
            return
        if tag == "div":
            self.depth += 1
        elif tag == "table":
            self.in_table = True
        elif self.in_table and tag == "tr":
            self.row = []
        elif self.in_table and tag == "td" and self.row is not None:
            self.cell = ""

    def handle_endtag(self, tag):
        if self.done or self.depth == 0:
            return
        if tag == "div":
            self.depth -= 1
        elif tag == "table" and self.in_table:
            self.done = True
        elif tag == "td" and self.cell is not None:
            self.row.append(self.cell.strip())
            self.cell = None
        elif tag == "tr" and self.row is not None:
            if self.row:
                self.rows.append(self.row)
            self.row = None

    def handle_data(self, data):
        if self.cell is not None:
            self.cell += data


def parse_rating(text):
    """'B+ (Stable)' -> ('B+', 'Stable'); '(Negative)' -> (None, 'Negative').

    Returns (None, None) for anything that is not a known S&P rating, so a
    typo or a new symbol is dropped rather than mislabelled.
    """
    match = _RATING.match(text or "")
    if not match:
        return None, None
    rating, outlook = match.group(1), match.group(2)
    if rating is not None and rating not in SP_NOTCH:
        return None, None
    return rating, (outlook.strip() if outlook else None)


def parse_sp_history(html):
    """S&P long-term foreign-currency history from one countryeconomy page.

    Returns columns date, rating, outlook, oldest first, with outlook-only rows
    filled with the rating in force. Rows before the first rating are dropped.
    """
    parser = _SectionTables("sp")
    parser.feed(html)
    records = []
    for row in parser.rows:
        if len(row) < 2 or not row[0]:
            continue
        date = pd.to_datetime(row[0], errors="coerce")
        rating, outlook = parse_rating(row[1])
        if pd.isna(date) or (rating is None and outlook is None):
            continue
        records.append({"date": date, "rating": rating, "outlook": outlook})
    history = pd.DataFrame(records, columns=["date", "rating", "outlook"])
    history = history.sort_values("date", kind="stable").reset_index(drop=True)
    history["rating"] = history["rating"].ffill()
    return history[history["rating"].notna()].reset_index(drop=True)


def year_end_ratings(history, years):
    """The rating in force on 31 December of each year (NaN before the first rating)."""
    rows = []
    for year in years:
        in_force = history[history["date"] <= pd.Timestamp(year=year, month=12, day=31)]
        if in_force.empty:
            rows.append({"year": year, "rating": None, "outlook": None})
        else:
            last = in_force.iloc[-1]
            rows.append({"year": year, "rating": last["rating"], "outlook": last["outlook"]})
    annual = pd.DataFrame(rows)
    annual["notch"] = annual["rating"].map(SP_NOTCH)
    annual["rating_class"] = annual["rating"].map(sp_rating_class)
    return annual


def sp_rating_class(rating):
    """Default / Junk / Investment Grade as 0 / 1 / 2, the project's classes."""
    if rating is None or (isinstance(rating, float) and np.isnan(rating)):
        return np.nan
    return rating_to_class(rating)
