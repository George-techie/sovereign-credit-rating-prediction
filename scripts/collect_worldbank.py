"""
collect_worldbank.py  (v3)
--------------------------
Fetches macroeconomic indicators from World Bank for 80 countries,
years 2000-2025. Uses direct REST API calls (not wbgapi) to avoid
URL-length issues.

Run:
    python collect_worldbank.py

Requirements:
    pip install pandas requests
"""

import time
import warnings
import requests
import pandas as pd
from pathlib import Path

warnings.filterwarnings("ignore")

OUTPUT_PATH = Path("data/raw/macro/worldbank_macro.csv")
START_YEAR  = 2000
END_YEAR    = 2025
WB_BASE     = "https://api.worldbank.org/v2"

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/120.0.0.0 Safari/537.36"
    )
}

# ─── 80 Countries ─────────────────────────────────────────────────────────────
COUNTRIES = {
    "ZAF":"South Africa","KEN":"Kenya","GHA":"Ghana","EGY":"Egypt",
    "NGA":"Nigeria","ETH":"Ethiopia","BWA":"Botswana","MAR":"Morocco",
    "ZMB":"Zambia","TZA":"Tanzania","UGA":"Uganda","RWA":"Rwanda",
    "SEN":"Senegal","CIV":"Ivory Coast","TUN":"Tunisia","AGO":"Angola",
    "MOZ":"Mozambique","CMR":"Cameroon","ZWE":"Zimbabwe","NAM":"Namibia",
    "MUS":"Mauritius","CPV":"Cape Verde","BEN":"Benin","MLI":"Mali",
    "BFA":"Burkina Faso","COD":"Democratic Republic of Congo",
    "SDN":"Sudan","LBY":"Libya","DZA":"Algeria",
    "BRA":"Brazil","MEX":"Mexico","ARG":"Argentina","COL":"Colombia",
    "CHL":"Chile","PER":"Peru","URY":"Uruguay","ECU":"Ecuador",
    "BOL":"Bolivia","PRY":"Paraguay","VEN":"Venezuela","CRI":"Costa Rica",
    "PAN":"Panama",
    "CHN":"China","IND":"India","JPN":"Japan","KOR":"South Korea",
    "IDN":"Indonesia","MYS":"Malaysia","THA":"Thailand","PHL":"Philippines",
    "VNM":"Vietnam","PAK":"Pakistan","BGD":"Bangladesh","LKA":"Sri Lanka",
    "KAZ":"Kazakhstan","AZE":"Azerbaijan","GEO":"Georgia","MNG":"Mongolia",
    "DEU":"Germany","GBR":"United Kingdom","FRA":"France","ITA":"Italy",
    "TUR":"Turkey","POL":"Poland","UKR":"Ukraine","ROU":"Romania",
    "HUN":"Hungary","CZE":"Czech Republic","SRB":"Serbia","BGR":"Bulgaria",
    "SAU":"Saudi Arabia","ARE":"United Arab Emirates","ISR":"Israel",
    "JOR":"Jordan","LBN":"Lebanon","BHR":"Bahrain",
    "USA":"United States","CAN":"Canada","AUS":"Australia","NZL":"New Zealand",
}

# All indicators via World Bank REST API
# Using SL.UEM.TOTL.ZS (ILO modelled) for better unemployment coverage
INDICATORS = {
    "gdp_growth":              "NY.GDP.MKTP.KD.ZG",
    "inflation":               "FP.CPI.TOTL.ZG",
    "debt_to_gdp":             "GC.DOD.TOTL.GD.ZS",
    "current_account_pct_gdp": "BN.CAB.XOKA.GD.ZS",
    "reserves_months_imports": "FI.RES.TOTL.MO",
    "unemployment":            "SL.UEM.TOTL.ZS",
    "gdp_per_capita":          "NY.GDP.PCAP.KD",
    "population":              "SP.POP.TOTL",
    "trade_pct_gdp":           "NE.TRD.GNFS.ZS",
    "fdi_pct_gdp":             "BX.KLT.DINV.WD.GD.ZS",
}


def build_grid():
    rows = []
    for iso, name in COUNTRIES.items():
        for year in range(START_YEAR, END_YEAR + 1):
            rows.append({"iso3": iso, "country": name, "year": year})
    return pd.DataFrame(rows)


def fetch_indicator_rest(col_name, wb_code, iso_codes, batch_size=20):
    """Fetch via direct REST API in batches of batch_size countries."""
    print(f"  [{wb_code}]  batch_size={batch_size}")

    batches   = [iso_codes[i:i+batch_size] for i in range(0, len(iso_codes), batch_size)]
    all_rows  = []

    for b_idx, batch in enumerate(batches):
        country_str = ";".join(batch)
        url = (
            f"{WB_BASE}/country/{country_str}/indicator/{wb_code}"
            f"?date={START_YEAR}:{END_YEAR}&format=json&per_page=2000"
        )

        success = False
        for attempt in range(3):
            try:
                resp = requests.get(url, headers=HEADERS, timeout=30)
                resp.raise_for_status()
                data = resp.json()

                if len(data) < 2 or data[1] is None:
                    break

                for rec in data[1]:
                    iso3  = rec.get("countryiso3code", "")
                    yr    = rec.get("date", "")
                    val   = rec.get("value")
                    if iso3 and yr:
                        all_rows.append({
                            "iso3":    iso3,
                            "year":    int(yr),
                            col_name:  float(val) if val is not None else float("nan"),
                        })

                success = True
                break

            except Exception as e:
                print(f"    Batch {b_idx+1} attempt {attempt+1} failed: {e}")
                time.sleep(2 ** attempt)

        if not success:
            print(f"    Batch {b_idx+1} failed — filling with NaN")
            for iso in batch:
                for yr in range(START_YEAR, END_YEAR + 1):
                    all_rows.append({"iso3": iso, "year": yr, col_name: float("nan")})

        time.sleep(0.4)

    if not all_rows:
        return pd.DataFrame(columns=["iso3", "year", col_name])

    result = pd.DataFrame(all_rows)
    result["year"] = result["year"].astype(int)
    # Deduplicate (keep first occurrence)
    result = result.drop_duplicates(subset=["iso3", "year"])
    return result[["iso3", "year", col_name]]


def main():
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    iso_codes = list(COUNTRIES.keys())
    total     = len(INDICATORS)

    print(f"\n{'='*60}")
    print(f"World Bank Macro Data Collector  (v3 — direct REST API)")
    print(f"Countries  : {len(COUNTRIES)}")
    print(f"Years      : {START_YEAR}–{END_YEAR}")
    print(f"Indicators : {total}")
    print(f"{'='*60}\n")

    df = build_grid()

    for i, (col_name, wb_code) in enumerate(INDICATORS.items(), 1):
        print(f"[{i}/{total}] {col_name}")
        fetched = fetch_indicator_rest(col_name, wb_code, iso_codes)

        if fetched.empty or col_name not in fetched.columns:
            print(f"  WARNING: no data returned for {col_name}")
            df[col_name] = float("nan")
        else:
            n_non_null = fetched[col_name].notna().sum()
            print(f"  Got {len(fetched)} rows, {n_non_null} non-null")
            df = df.merge(fetched, on=["iso3", "year"], how="left")

    # Finalize
    all_cols = list(INDICATORS.keys())
    df = (
        df.drop(columns=["iso3"])
        .sort_values(["country", "year"])
        .reset_index(drop=True)
    )
    col_order = ["country", "year"] + [c for c in all_cols if c in df.columns]
    df = df[col_order]

    # Coverage report
    total_rows    = len(df)
    complete_rows = df[all_cols].notna().all(axis=1).sum()

    print(f"\n{'='*60}")
    print(f"Coverage Report")
    print(f"{'='*60}")
    print(f"Total rows    : {total_rows}")
    print(f"Complete rows : {complete_rows} ({100*complete_rows/total_rows:.1f}%)")
    print(f"\nPer-indicator coverage:")
    for col in all_cols:
        if col in df.columns:
            n   = df[col].notna().sum()
            pct = 100 * n / total_rows
            bar = "✅" if pct >= 80 else ("⚠️" if pct >= 50 else "❌")
            print(f"  {bar} {col:<35} {n:>5} / {total_rows}  ({pct:.1f}%)")

    africa_names = [COUNTRIES[k] for k in [
        "ZAF","KEN","GHA","EGY","NGA","ETH","BWA","MAR","ZMB","TZA","UGA",
        "RWA","SEN","CIV","TUN","AGO","MOZ","CMR","ZWE","NAM","MUS","CPV",
        "BEN","MLI","BFA","COD","SDN","LBY","DZA"
    ]]

    regions = [
        ("Africa",        africa_names),
        ("Latin America", ["Brazil","Mexico","Argentina","Colombia","Chile","Peru",
                           "Uruguay","Ecuador","Bolivia","Paraguay","Venezuela",
                           "Costa Rica","Panama"]),
        ("Asia-Pacific",  ["China","India","Japan","South Korea","Indonesia",
                           "Malaysia","Thailand","Philippines","Vietnam","Pakistan",
                           "Bangladesh","Sri Lanka","Kazakhstan","Azerbaijan",
                           "Georgia","Mongolia"]),
        ("Europe",        ["Germany","United Kingdom","France","Italy","Turkey",
                           "Poland","Ukraine","Romania","Hungary","Czech Republic",
                           "Serbia","Bulgaria"]),
        ("Middle East",   ["Saudi Arabia","United Arab Emirates","Israel",
                           "Jordan","Lebanon","Bahrain"]),
        ("North America", ["United States","Canada","Australia","New Zealand"]),
    ]

    print(f"\nPer-region coverage (% complete rows):")
    for region, clist in regions:
        sub = df[df["country"].isin(clist)]
        if len(sub) == 0:
            continue
        pct = 100 * sub[all_cols].notna().all(axis=1).sum() / len(sub)
        n_c = len(sub["country"].unique())
        print(f"  {region:<20} {pct:.1f}%  ({n_c} countries × 26 years)")

    df.to_csv(OUTPUT_PATH, index=False)
    print(f"\n✅ Saved {len(df)} rows → {OUTPUT_PATH}")
    print(f"   Columns: {list(df.columns)}\n")


if __name__ == "__main__":
    main()
