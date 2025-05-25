from kedro.pipeline import Pipeline, node
from .nodes import preprocess_and_merge_projects

def create_pipeline(**kwargs):
    return Pipeline([
        node(
            func=preprocess_and_merge_projects,
            inputs=["project_raw", "organization_raw"],
            outputs="projects_csv",
            name="preprocess_and_merge_projects_node"
        )
    ])