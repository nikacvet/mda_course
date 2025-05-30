import pandas as pd
import re
import pycountry

def preprocess_and_merge_projects(project_raw: pd.DataFrame, organization_raw: pd.DataFrame) -> pd.DataFrame:
    df = project_raw

    # Step 1: Filter projects where 'topics' start with 'HORIZON-CL5-' or 'HORIZON-CL6-'
    mask_cl5_cl6 = df['topics'].astype(str).str.startswith(('HORIZON-CL5-', 'HORIZON-CL6-'))
    cl5_cl6_projects = df[mask_cl5_cl6]

    # Step 2: Exclude those to get the remaining projects
    remaining_projects = df[~mask_cl5_cl6]

    # Step 3: Define regex pattern for climate-related keywords
    pattern = re.compile(
        r"\b(climate change|zero emission|net[- ]zero|carbon[- ]neutral|decarboni[sz]ation|greenhouse|CO2|GHG|"
        r"carbon neutrality|carbon capture|carbon sink|carbon sequestration|zero[- ]carbon|"
        r"low[- ]carbon|climate mitigation|climate adaptation|climate-neutral|"
        r"renewable energy|clean energy|green hydrogen|"
        r"sustainable energy|climate-smart|climate crisis|climate risk|"
        r"sea level rise|extreme weather|biodiversity loss|circular economy|geoengineering|green technology|"
        r"global warming|environmental impact|environmental sustainability|energy transition)\b",
        flags=re.IGNORECASE
    )

    # Step 4: Apply regex to the 'objective' column of the remaining projects
    remaining_climate_projects = remaining_projects[
        remaining_projects['objective'].astype(str).str.contains(pattern)
    ]

    # Step 5: Combine both sets of climate-related projects
    all_climate_related_projects = pd.concat([cl5_cl6_projects, remaining_climate_projects])

    # Merge with organization data
    organization_df = organization_raw
    merged_df = pd.merge(organization_df, all_climate_related_projects, left_on="projectID", right_on="id", how="inner")

    # Drop unwanted columns
    columns_to_drop = [
        'projectAcronym', 'street', 'postCode', 'city', 'nutsCode', 'geolocation', 'organizationURL',
        'contactForm','rcn_x','order','active', 'id', 'acronym', 'legalBasis','ecSignatureDate',
        'frameworkProgramme','masterCall','nature','rcn_y','grantDoi','subCall','shortName','frameworkProgramme','vatNumber'
    ]
    merged_df.drop(columns=columns_to_drop, inplace=True, errors='ignore')

    # Rename columns
    rename_map = {
        'SME': 'organisationSME',
        'activityType': 'organisationActivityType',
        'contentUpdateDate_x': 'organisationContentUpdateDate',
        'role': 'organisationRole',
        'totalCost_x': 'organisationTotalCost',
        'endOfParticipation': 'organisationEndOfParticipation',
        'totalCost_y': 'projectTotalCost',
        'contentUpdateDate_y': 'projectContentUpdateDate',
        'title': 'projectTitle',
        'name': 'organisationName'
    }
    merged_df.rename(columns=rename_map, inplace=True)

    return merged_df


def load__preprocess(project_csv_path: str, gdp_csv_path: str):
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
    