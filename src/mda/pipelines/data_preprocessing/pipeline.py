from kedro.pipeline import Pipeline, node
from .nodes import preprocess_and_merge_projects, load__preprocess

def create_pipeline(**kwargs):
    return Pipeline([
        node(
            func=preprocess_and_merge_projects,
            inputs=["project_raw", "organization_raw"],
            outputs="projects_csv",
            name="data_merge"
        ),
        node(
            func=load__preprocess,
            inputs=["projects_csv", "gdp_csv"],
            outputs=dict(
                projects_preprocessed="projects_preprocessed",
                gdp_preprocessed="gdp_preprocessed",
                eu_iso2="eu_iso2",
                project_statuses="project_statuses"
            ),
            name="data_preprocessing"
        )
    ])