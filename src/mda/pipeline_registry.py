from typing import Dict
from kedro.pipeline import Pipeline

from mda.pipelines import data_preprocessing
from mda.pipelines import data_processing
from mda.pipelines import feature_selection
from mda.pipelines import text_analysis

def register_pipelines() -> Dict[str, Pipeline]:
    """Register the project's pipelines."""
    data_preprocessing_pipeline = data_preprocessing.create_pipeline()
    data_processing_pipeline = data_processing.create_pipeline()
    feature_selection_pipeline = feature_selection.create_pipeline()
    text_analysis_pipeline = text_analysis.create_pipeline()

    return {
        "data_preprocessing": data_preprocessing_pipeline,
        "data_processing": data_processing_pipeline,
        "feature_selection": feature_selection_pipeline,
        "text_analysis": text_analysis_pipeline,
        "__default__": (
            data_preprocessing_pipeline
            + data_processing_pipeline
            + feature_selection_pipeline
            + text_analysis_pipeline
        )
    }