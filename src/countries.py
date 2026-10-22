"""
countries.py — The 80-country panel: ISO3 codes and regions.

ISO3 codes join sources that spell names differently (IMF "Türkiye, Republic
of", World Bank "Turkiye", our "Turkey"). Regions follow the World Bank's
grouping, with sub-Saharan and North Africa kept together as "Africa" because
the bias analysis compares Africa with the rest of the world.
"""

COUNTRIES = {
    # Africa
    "Algeria": ("DZA", "Africa"), "Angola": ("AGO", "Africa"), "Benin": ("BEN", "Africa"),
    "Botswana": ("BWA", "Africa"), "Burkina Faso": ("BFA", "Africa"),
    "Cameroon": ("CMR", "Africa"), "Cape Verde": ("CPV", "Africa"),
    "Democratic Republic of Congo": ("COD", "Africa"), "Egypt": ("EGY", "Africa"),
    "Ethiopia": ("ETH", "Africa"), "Ghana": ("GHA", "Africa"), "Ivory Coast": ("CIV", "Africa"),
    "Kenya": ("KEN", "Africa"), "Libya": ("LBY", "Africa"), "Mali": ("MLI", "Africa"),
    "Mauritius": ("MUS", "Africa"), "Morocco": ("MAR", "Africa"),
    "Mozambique": ("MOZ", "Africa"), "Namibia": ("NAM", "Africa"), "Nigeria": ("NGA", "Africa"),
    "Rwanda": ("RWA", "Africa"), "Senegal": ("SEN", "Africa"),
    "South Africa": ("ZAF", "Africa"), "Sudan": ("SDN", "Africa"), "Tanzania": ("TZA", "Africa"),
    "Tunisia": ("TUN", "Africa"), "Uganda": ("UGA", "Africa"), "Zambia": ("ZMB", "Africa"),
    "Zimbabwe": ("ZWE", "Africa"),
    # Latin America and the Caribbean
    "Argentina": ("ARG", "Latin America"), "Bolivia": ("BOL", "Latin America"),
    "Brazil": ("BRA", "Latin America"), "Chile": ("CHL", "Latin America"),
    "Colombia": ("COL", "Latin America"), "Costa Rica": ("CRI", "Latin America"),
    "Ecuador": ("ECU", "Latin America"), "Mexico": ("MEX", "Latin America"),
    "Panama": ("PAN", "Latin America"), "Paraguay": ("PRY", "Latin America"),
    "Peru": ("PER", "Latin America"), "Uruguay": ("URY", "Latin America"),
    "Venezuela": ("VEN", "Latin America"),
    # Asia and the Pacific
    "Australia": ("AUS", "Asia-Pacific"), "Bangladesh": ("BGD", "Asia-Pacific"),
    "China": ("CHN", "Asia-Pacific"), "India": ("IND", "Asia-Pacific"),
    "Indonesia": ("IDN", "Asia-Pacific"), "Japan": ("JPN", "Asia-Pacific"),
    "Malaysia": ("MYS", "Asia-Pacific"), "Mongolia": ("MNG", "Asia-Pacific"),
    "New Zealand": ("NZL", "Asia-Pacific"), "Pakistan": ("PAK", "Asia-Pacific"),
    "Philippines": ("PHL", "Asia-Pacific"), "South Korea": ("KOR", "Asia-Pacific"),
    "Sri Lanka": ("LKA", "Asia-Pacific"), "Thailand": ("THA", "Asia-Pacific"),
    "Vietnam": ("VNM", "Asia-Pacific"),
    # Europe and Central Asia
    "Azerbaijan": ("AZE", "Europe & Central Asia"), "Bulgaria": ("BGR", "Europe & Central Asia"),
    "Czech Republic": ("CZE", "Europe & Central Asia"), "France": ("FRA", "Europe & Central Asia"),
    "Georgia": ("GEO", "Europe & Central Asia"), "Germany": ("DEU", "Europe & Central Asia"),
    "Hungary": ("HUN", "Europe & Central Asia"), "Italy": ("ITA", "Europe & Central Asia"),
    "Kazakhstan": ("KAZ", "Europe & Central Asia"), "Poland": ("POL", "Europe & Central Asia"),
    "Romania": ("ROU", "Europe & Central Asia"), "Serbia": ("SRB", "Europe & Central Asia"),
    "Turkey": ("TUR", "Europe & Central Asia"), "Ukraine": ("UKR", "Europe & Central Asia"),
    "United Kingdom": ("GBR", "Europe & Central Asia"),
    # Middle East
    "Bahrain": ("BHR", "Middle East"), "Israel": ("ISR", "Middle East"),
    "Jordan": ("JOR", "Middle East"), "Lebanon": ("LBN", "Middle East"),
    "Saudi Arabia": ("SAU", "Middle East"), "United Arab Emirates": ("ARE", "Middle East"),
    # North America
    "Canada": ("CAN", "North America"), "United States": ("USA", "North America"),
}

ISO3 = {country: code for country, (code, _) in COUNTRIES.items()}
COUNTRY_BY_ISO3 = {code: country for country, code in ISO3.items()}
REGION = {country: region for country, (_, region) in COUNTRIES.items()}
