import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from typing import List
import plotly.express as px
from pytorch_tabnet.tab_model import TabNetRegressor
from sklearn.model_selection import train_test_split
from sklearn.manifold import TSNE
import torch



def to_float_eu(x):
    if pd.isna(x):
        return np.nan
    s = str(x).replace("€", "").replace(" ", "").replace(" ", "")
    if "," in s and "." in s:
        if s.find(",") > s.find("."):
            s = s.replace(".", "").replace(",", ".")
        else:
            s = s.replace(",", "")
    elif "," in s:
        s = s.replace(".", "").replace(",", ".")
    else:
        s = s.replace(",", "")
    try:
        return float(s)
    except ValueError:
        return np.nan


def country_features(projects, gdp, eu_iso2):
    # Convert monetary columns to float
    money_cols = [
        "ecContribution", "netEcContribution", "organisationTotalCost",
        "projectTotalCost", "ecMaxContribution"
    ]
    for col in money_cols:
        projects[col] = projects[col].apply(to_float_eu)

    # Collaboration flag
    collab_map = projects.groupby("projectID")["country"].nunique()
    projects["is_collab"] = projects["projectID"].isin(collab_map[collab_map > 1].index)

    # Group by country and calculate metrics
    grp = projects.groupby("country")
    country_features = pd.DataFrame({
        "total_ec_contribution": grp["ecContribution"].sum(),
        "total_projects":        grp["projectID"].nunique(),
        "unique_orgs":           grp["organisationID"].nunique(),
        "collab_project_ratio":  grp["is_collab"].mean(),
        "signed_pct":            grp.apply(lambda g: g["status"].str.upper().eq("SIGNED").mean(), include_groups=False),
        "closed_pct":            grp.apply(lambda g: g["status"].str.upper().eq("CLOSED").mean(), include_groups=False),
        "terminated_pct":        grp.apply(lambda g: g["status"].str.upper().eq("TERMINATED").mean(), include_groups=False),
    }).reset_index()

    # Calculate success ratio
    country_features["success_ratio"] = (country_features["signed_pct"] + country_features["closed_pct"]) - country_features["terminated_pct"]

    # Merge with GDP data
    gdp_eu = gdp[["iso2", "2023"]].rename(columns={"iso2": "country", "2023": "gdp_2023"})
    country_features = country_features.merge(gdp_eu, on="country", how="left")
    country_features = country_features[country_features["country"].isin(eu_iso2)]

    # Calculate EC contribution per GDP
    country_features["ec_per_gdp"] = country_features["total_ec_contribution"] / country_features["gdp_2023"]
    country_features.fillna(0, inplace=True)

    # New enriched score components
    country_features["funding_share"] = country_features["total_ec_contribution"] / country_features["total_ec_contribution"].sum()
    country_features["project_share"] = country_features["total_projects"] / country_features["total_projects"].max()
    country_features["org_share"]     = country_features["unique_orgs"] / country_features["unique_orgs"].max()

    # Use all factors for scoring
    metric_cols = [
        "funding_share", "ec_per_gdp", "project_share", "org_share",
        "collab_project_ratio", "success_ratio"
    ]

    # Scale metrics and calculate final score
    scaler = MinMaxScaler()
    country_features[[m + "_scaled" for m in metric_cols]] = scaler.fit_transform(country_features[metric_cols])
    scaled_cols = [m + "_scaled" for m in metric_cols]
    country_features["Score"] = country_features[scaled_cols].mean(axis=1)

    return country_features


def compute_weighted_scores_and_tabnet_insights(country_features):
    weights = {    
        "funding_share_scaled": 0.20,
        "ec_per_gdp_scaled": 0.20,
        "project_share_scaled": 0.15,
        "org_share_scaled": 0.15,
        "collab_project_ratio_scaled": 0.15,
        "success_ratio_scaled": 0.15
    }

    if "Score" not in country_features.columns:
        scaled_cols = list(weights.keys())
        country_features["Score"] = country_features[scaled_cols].mean(axis=1)

    country_features["Weighted_Score"] = sum(country_features[col] * w for col, w in weights.items())

    features_list = [
        "funding_share", "ec_per_gdp", "project_share", "org_share",
        "collab_project_ratio", "success_ratio"
    ]
    tabnet_X = country_features[features_list].values.astype("float32")

    tabnet_y_w = country_features["Weighted_Score"].values.astype("float32").reshape(-1, 1)
    X_train, X_test, y_train, y_test = train_test_split(tabnet_X, tabnet_y_w, test_size=0.2, random_state=42)
    clf_weighted = TabNetRegressor(seed=42, verbose=0)
    clf_weighted.fit(X_train=X_train, y_train=y_train, eval_set=[(X_test, y_test)], max_epochs=500, patience=50)

    imp_w = clf_weighted.feature_importances_

    display_names = [f if f != "success_ratio" else "completion_percentage" for f in features_list]

    importance_json = [
        {"feature": name, "importance": float(score)}
        for name, score in zip(display_names, imp_w)
    ]

    tabnet_y_e = country_features["Score"].values.astype("float32").reshape(-1, 1)
    X_train_e, X_test_e, y_train_e, y_test_e = train_test_split(tabnet_X, tabnet_y_e, test_size=0.2, random_state=42)
    clf_equal = TabNetRegressor(seed=0, verbose=0)
    clf_equal.fit(X_train=X_train_e, y_train=y_train_e, eval_set=[(X_test_e, y_test_e)], max_epochs=500, patience=50)

    with torch.no_grad():
        clf_weighted.network.eval()
        embeddings = clf_weighted.network.embedder.forward(torch.from_numpy(tabnet_X)).detach().cpu().numpy()

    perplexity = min(5, len(embeddings) - 1)
    tsne = TSNE(n_components=2, perplexity=perplexity, random_state=42)
    coords = tsne.fit_transform(embeddings)
    country_features["x_tsne"] = coords[:, 0]
    country_features["y_tsne"] = coords[:, 1]

    quantiles = pd.qcut(country_features["Weighted_Score"], q=5, labels=False)
    color_map = px.colors.sequential.Viridis
    country_features["color"] = quantiles.map(lambda q: color_map[q])
    country_features_weighted = country_features
    return country_features_weighted, country_features.sort_values("Weighted_Score", ascending=False)[["country", "Weighted_Score", "color"]], importance_json


