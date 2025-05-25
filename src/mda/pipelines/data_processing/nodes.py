"""
Data preprocessing nodes for the MDA pipeline.
"""
import pandas as pd
import pycountry
from typing import Dict


def load__preprocess(project_csv_path: str, gdp_csv_path: str) -> Dict[str, pd.DataFrame]:
    projects_raw = project_csv_path
    gdp_raw = gdp_csv_path
    
    # Convert GDP column to numeric
    gdp_raw["2023"] = pd.to_numeric(gdp_raw["2023"], errors="coerce")
    
    # ISO‑3 → ISO‑2 mapping via pycountry
    iso3_to2 = {c.alpha_3: c.alpha_2 for c in pycountry.countries if hasattr(c, "alpha_3")}
    gdp_raw["iso2"] = gdp_raw["Country Code"].map(iso3_to2)

    # Define EU ISO2 codes
    EU_ISO2 = [
        "AT", "BE", "BG", "HR", "CY", "CZ", "DK", "EE", "FI", "FR", "DE", "GR", "HU", "IE",
        "IT", "LV", "LT", "LU", "MT", "NL", "PL", "PT", "RO", "SK", "SI", "ES", "SE"
    ]

    # Extract project statuses
    project_statuses = projects_raw['status'].unique().tolist()
    
    return {
        "projects_preprocessed": projects_raw,
        "gdp_preprocessed": gdp_raw,
        "eu_iso2": EU_ISO2,
        "project_statuses": project_statuses
    }
    