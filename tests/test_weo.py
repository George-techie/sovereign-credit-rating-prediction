"""Reading IMF WEO files: UTF-16 tab-separated text, estimates flagged."""

from src.weo import DEBT_CODE, extract_indicator, read_weo

HEADER = ["WEO Country Code", "ISO", "WEO Subject Code", "Country", "2022", "2023", "2024",
          "Estimates Start After"]


def write_weo(path, *rows):
    lines = ["\t".join(HEADER)] + ["\t".join(row) for row in rows]
    lines.append("International Monetary Fund, World Economic Outlook Database")  # footnote
    path.write_bytes("\n".join(lines).encode("utf-16-le"))  # no byte-order mark, as IMF ships it


def test_utf16_file_without_bom_is_read(tmp_path):
    path = tmp_path / "WEO.xls"
    write_weo(path, ["664", "KEN", DEBT_CODE, "Kenya", "67.8", "73.107", "69.873", "2023"])

    weo = read_weo(path)

    assert weo["ISO"].tolist() == ["KEN"]  # footnote row dropped


def test_years_after_the_last_actual_are_estimates(tmp_path):
    path = tmp_path / "WEO.xls"
    write_weo(path, ["664", "KEN", DEBT_CODE, "Kenya", "67.8", "73.107", "69.873", "2023"])

    debt = extract_indicator(read_weo(path), DEBT_CODE, 2022, 2024)

    assert debt["value"].tolist() == [67.8, 73.107, 69.873]
    assert debt["is_estimate"].tolist() == [False, False, True]
    assert debt["country"].unique().tolist() == ["Kenya"]


def test_missing_cells_and_other_countries_are_skipped(tmp_path):
    path = tmp_path / "WEO.xls"
    write_weo(path,
              ["664", "KEN", DEBT_CODE, "Kenya", "n/a", "--", "1,234.5", "2023"],
              ["999", "XYZ", DEBT_CODE, "Elsewhere", "1", "2", "3", "2023"])

    debt = extract_indicator(read_weo(path), DEBT_CODE, 2022, 2024)

    assert debt["value"].isna().tolist() == [True, True, False]
    assert debt["value"].iloc[2] == 1234.5
    assert set(debt["iso3"]) == {"KEN"}


def test_imf_portal_csv_is_read_with_the_classic_columns(tmp_path):
    path = tmp_path / "dataset_IMF.RES_WEO.csv"
    header = '"DATASET","SERIES_CODE","COUNTRY","LATEST_ACTUAL_ANNUAL_DATA","2024","2025"'
    row = '"IMF.RES:WEO(9.0.0)","KEN.GGXWDG_NGDP.A","Kenya","2024","67.306","71.269"'
    path.write_text(header + "\n" + row + "\n", encoding="utf-8-sig")  # portal adds a BOM

    debt = extract_indicator(read_weo(path), DEBT_CODE, 2024, 2025)

    assert debt["value"].tolist() == [67.306, 71.269]
    assert debt["is_estimate"].tolist() == [False, True]
