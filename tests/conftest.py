import pandas as pd
from pytest import fixture
from ds_toolbox.spark_utils import start_local_spark

@fixture
def spark():
    return start_local_spark(max_mem=1, n_cores=4)

@fixture
def df_binary_classification_prediction():
    return pd.DataFrame({
        'target': [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0],
        'predicted': [1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0]
    })

@fixture
def dfs_binary_classification_prediction(spark, df_binary_classification_prediction):
    return spark.createDataFrame(df_binary_classification_prediction)
