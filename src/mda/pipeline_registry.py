from typing import Dict
from kedro.pipeline import Pipeline

from mda.pipelines import data_preprocessing
from mda.pipelines import feature_selection
from mda.pipelines import text_analysis
from mda.pipelines import future_projections

def register_pipelines() -> Dict[str, Pipeline]:
    """Register the project's pipelines."""
    data_preprocessing_pipeline = data_preprocessing.create_pipeline()
    feature_selection_pipeline = feature_selection.create_pipeline()
    text_analysis_pipeline = text_analysis.create_pipeline()
    future_projections_pipeline = future_projections.create_pipeline()

    return {
        "data_preprocessing": data_preprocessing_pipeline,
        "feature_selection": feature_selection_pipeline,
        "text_analysis": text_analysis_pipeline,
        "future_projections": future_projections_pipeline,
        "__default__": (
            data_preprocessing_pipeline
            + feature_selection_pipeline
            + text_analysis_pipeline
            + future_projections_pipeline
        )
    }