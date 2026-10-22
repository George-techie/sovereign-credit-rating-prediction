"""S&P history parsing and year-end labels."""

import numpy as np
import pandas as pd

from src.ratings import SP_NOTCH, parse_rating, parse_sp_history, sp_rating_class, year_end_ratings

ROW = "<tr><td>{}</td><td>{}</td><td></td><td></td><td></td><td></td><td></td><td></td></tr>"


def page(*rows, fitch_rows=()):
    """A minimal countryeconomy page: a Fitch table before the S&P one, as on the site."""
    table = "<table><thead><tr><th>Date</th></tr></thead><tbody>{}</tbody></table>"
    return (
        '<div id="fitch"><div>' + table.format("".join(ROW.format(*r) for r in fitch_rows))
        + '</div></div><div class="tab-pane" id="sp"><div class="tabletit">Rating S&amp;P</div>'
        '<div class="table-responsive">' + table.format("".join(ROW.format(*r) for r in rows))
        + "</div></div>"
    )


def test_rating_and_outlook_are_split():
    assert parse_rating("BB- (Positive)") == ("BB-", "Positive")


def test_outlook_only_cell_has_no_rating():
    assert parse_rating("(Negative)") == (None, "Negative")


def test_unknown_symbol_is_dropped_not_mislabelled():
    assert parse_rating("Baa2 (Stable)") == (None, None)


def test_only_the_sp_table_is_read():
    html = page(("2020-01-01", "BB (Stable)"), fitch_rows=[("2019-01-01", "AAA")])

    history = parse_sp_history(html)

    assert history["rating"].tolist() == ["BB"]


def test_outlook_change_carries_the_rating_forward():
    html = page(("2021-03-05", "B (Stable)"), ("2020-07-14", "(Negative)"),
                ("2010-11-19", "B+ (Stable)"))

    history = parse_sp_history(html)

    assert history["rating"].tolist() == ["B+", "B+", "B"]  # oldest first
    assert history["outlook"].tolist() == ["Stable", "Negative", "Stable"]


def test_year_end_rating_is_the_one_in_force_on_31_december():
    history = pd.DataFrame({
        "date": pd.to_datetime(["2019-05-01", "2019-12-31", "2020-06-01"]),
        "rating": ["BB", "BB-", "SD"],
        "outlook": ["Stable", "Negative", None],
    })

    annual = year_end_ratings(history, [2018, 2019, 2020])

    assert annual["rating"].tolist()[1:] == ["BB-", "SD"]
    assert np.isnan(annual["notch"].iloc[0])  # not yet rated
    assert annual["rating_class"].tolist()[1:] == [1, 0]


def test_notches_are_ordered_with_default_at_zero():
    assert SP_NOTCH["AAA"] > SP_NOTCH["BBB-"] > SP_NOTCH["BB+"] > SP_NOTCH["C"] > SP_NOTCH["SD"]
    assert SP_NOTCH["SD"] == SP_NOTCH["D"] == 0


def test_investment_grade_boundary_is_bbb_minus():
    assert sp_rating_class("BBB-") == 2
    assert sp_rating_class("BB+") == 1
    assert np.isnan(sp_rating_class(None))
