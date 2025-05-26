from kedro.pipeline import Pipeline, node
from .nodes import extract_keyphrases

def create_pipeline(**kwargs):
    return Pipeline([
        # node(
        #     func=cluster_objectives_with_bert,
        #     inputs="projects_preprocessed",
        #     outputs=None,
        #     name="cluster_objectives_with_bert_node"
        # ),
        node(
            func=extract_keyphrases,
            inputs="projects_preprocessed",
            outputs="project_keyphrases",
            name="extract_project_keyphrases"
        )
    ])