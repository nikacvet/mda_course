from kedro.pipeline import Pipeline, node
from .nodes import country_features, compute_weighted_scores_and_tabnet_insights

def create_pipeline(**kwargs):
    return Pipeline([
        node(
            func=country_features,
            inputs=["projects_preprocessed", "gdp_preprocessed", "eu_iso2"],
            outputs="country_features",
            name="country_features_node"
        ),
        node(
            func=compute_weighted_scores_and_tabnet_insights,
            inputs="country_features",
            outputs="country_scores_json",
            name="export_country_scores_json_node"
        )
    ])