from kedro.pipeline import Pipeline, node
from .nodes import country_features, compute_weighted_scores_and_tabnet_insights

def create_pipeline(**kwargs):
    return Pipeline([
        node(
            func=country_features,
            inputs=["projects_preprocessed", "gdp_preprocessed", "eu_iso2"],
            outputs="country_features",
            name="dataset_normalization"
        ),
        node(
            func=compute_weighted_scores_and_tabnet_insights,
            inputs="country_features",
            outputs=[
                "country_features_weighted",
                "country_scores_json",
                "feature_importance"
            ],
            name="feature_selection"
        )
    ])