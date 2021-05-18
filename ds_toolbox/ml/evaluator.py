from typing import Union
from typeguard import typechecked

import pyspark
from pyspark.ml.evaluation import BinaryClassificationEvaluator
import pandas as pd

from ds_toolbox.spark_utils import start_local_spark

@typechecked
def binary_classifier_metrics(
    df_prediction: Union[pyspark.sql.dataframe.DataFrame, pd.DataFrame], col_target: str, col_prediction: str,
    spark: Union[pyspark.sql.session.SparkSession, None] = None, max_mem: int = 3, n_cores: int = 2
) -> dict:
    """Computes Evaluation metrics of a binary classification result on pandas and spark df.

    Args:
        df_prediction (Union[pyspark.sql.dataframe.DataFrame, pd.DataFrame]): DataFrame with observed and predicted values.
        col_target (str): Column name of ground truth class.
        col_prediction (str): Column name with predicted class.
        spark (Union[pyspark.sql.session.SparkSession, None], optional): Spark session where computation will take place. 
            If none, then a local is created. Defaults to None.
        max_mem (int, optional): Max memory to be allocated to spark. Defaults to 3.
        n_cores (int, optional): Number os cores to be allocated to spark. Defaults to 2.

    Raises:
        Exception: Errors.

    Returns:
        dict: Dict with: confusion matrix, accuracy, f1 score, precision, recall, auroc, aupr.
    """
    try:
        if spark is None and type(df_prediction) == pyspark.sql.dataframe.DataFrame:
            raise ValueError('type(spark) is None. Please pass a valid spark session to spark argument.')

        if type(df_prediction) == pd.DataFrame:
            input_pandas = True
            spark = start_local_spark(max_mem=max_mem, n_cores=n_cores)
            df_prediction = spark.createDataFrame(df_prediction)
        else:
            input_pandas = False

        # Confusion Matrix
        confusion_matrix = df_prediction.groupBy(col_target, col_prediction).count() 
        TN = df_prediction.filter(f'{col_prediction} = 0 AND {col_target} = 0').count()
        TP = df_prediction.filter(f'{col_prediction} = 1 AND {col_target} = 1').count()
        FN = df_prediction.filter(f'{col_prediction} = 0 AND {col_target} = 1').count()
        FP = df_prediction.filter(f'{col_prediction} = 1 AND {col_target} = 0').count()

        # Computing Metrics from Confusion Matrix
        accuracy = (TN + TP) / (TN + TP + FN + FP)
        precision = TP/(TP+FP) if (TP+FP) > 0 else 0.0
        recall = TP/(TP+FN)
        f1 = 2*(precision*recall/(precision+recall)) if (precision+recall) > 0 else 0.0 

        evaluator = BinaryClassificationEvaluator(labelCol=col_target, rawPredictionCol=col_prediction, metricName='areaUnderROC')
        aucroc = evaluator.evaluate(df_prediction)
        evaluator = BinaryClassificationEvaluator(labelCol=col_target, rawPredictionCol=col_prediction, metricName='areaUnderPR')
        aucpr = evaluator.evaluate(df_prediction)

        # Results Dict
        out_dict = {
            'confusion_matrix': confusion_matrix.toPandas(),
            'accuracy': accuracy,
            'f1': f1,
            'precision': precision,
            'recall': recall,
            'aucroc': aucroc,
            'aucpr': aucpr
        }

        if input_pandas:
            spark.stop()
        
        return out_dict
    except Exception as e:
        raise Exception(e)
