"""
This module contains the data processing pipeline.
"""
from kedro.pipeline import Pipeline, node

from .nodes import load__preprocess


def create_pipeline(**kwargs) -> Pipeline:
    """
    Create the data processing pipeline.
    
    Returns:
        A pipeline containing data loading and preprocessing nodes.
    """
    return Pipeline(
        [
            node(
                func=load__preprocess,
                inputs=["projects_csv", "gdp_csv"],
                outputs=dict(
                    projects_preprocessed="projects_preprocessed",
                    gdp_preprocessed="gdp_preprocessed",
                    eu_iso2="eu_iso2",
                    project_statuses="project_statuses"
                ),
                name="load_preprocess_node"
            )
        ]
    )