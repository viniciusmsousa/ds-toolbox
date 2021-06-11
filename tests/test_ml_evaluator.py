import numpy as np
import pandas as pd
from ds_toolbox.ml.evaluator import binary_classifier_metrics


# Binary Classifier Metrics
def test_binary_classifier_metrics_pandas(df_binary_classification_prediction):
    out = binary_classifier_metrics(
        df_prediction=df_binary_classification_prediction,
        col_target='target',
        col_prediction='predicted'
    )

    for metric in ['confusion_matrix', 'accuracy', 'f1', 'precision', 'recall', 'aucroc', 'aucpr']:
        assert metric in out.keys()

    assert type(out['confusion_matrix']) == pd.DataFrame
    assert type(out['accuracy']) in [float, np.nan]
    assert type(out['f1']) in [float, np.nan]
    assert type(out['precision']) in [float, np.nan]
    assert type(out['recall']) in [float, np.nan]
    assert type(out['aucroc']) in [float, np.nan]
    assert type(out['aucpr']) in [float, np.nan]

def test_binary_classifier_metrics_spark(dfs_binary_classification_prediction, spark):
    out = binary_classifier_metrics(
        df_prediction=dfs_binary_classification_prediction, col_target='target',
        col_prediction='predicted', spark=spark
    )

    for metric in ['confusion_matrix', 'accuracy', 'f1', 'precision', 'recall', 'aucroc', 'aucpr']:
        assert metric in out.keys()

    assert type(out['confusion_matrix']) == pd.DataFrame
    assert type(out['accuracy']) in [float, np.nan]
    assert type(out['f1']) in [float, np.nan]
    assert type(out['precision']) in [float, np.nan]
    assert type(out['recall']) in [float, np.nan]
    assert type(out['aucroc']) in [float, np.nan]
    assert type(out['aucpr']) in [float, np.nan]
