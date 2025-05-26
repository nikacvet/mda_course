from kedro.pipeline import Pipeline, node
from .nodes import project_with_realistic_dynamics


def create_pipeline(**kwargs):
    return Pipeline([
        node(
            func=project_with_realistic_dynamics,
            inputs=dict(
                df="country_features_weighted",
                targets="params:projection_targets",
                shocks="params:projection_shocks"
            ),
            outputs="country_projections",
            name="future_projections"
        ),
    ])