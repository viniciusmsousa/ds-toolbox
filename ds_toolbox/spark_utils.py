from typeguard import typechecked
import pyspark
from pyspark.sql import SparkSession


@typechecked
def start_local_spark(max_mem: int = 3, n_cores: int = 2) -> pyspark.sql.session.SparkSession:
    """Starts a local spark session.
    Used to convert pandas DF into spark df for computing a few tests and metrics.

    Args:
        max_mem (int, optional): Max memory to be allocated. Defaults to 3.
        n_cores (int, optional): Number os cores to be allocated. Defaults to 2.

    Raises:
        Exception: Every error.

    Returns:
        pyspark.sql.session.SparkSession: spark session object.
    """
    try:
        spark = SparkSession.builder\
                .appName('Ml-Pipes') \
                .master(f'local[{n_cores}]') \
                .config('spark.executor.memory', f'{max_mem}G') \
                .config('spark.driver.memory', f'{max_mem}G') \
                .config('spark.memory.offHeap.enabled', 'true') \
                .config('spark.memory.offHeap.size', f'{max_mem}G') \
                .getOrCreate()
        return spark
    except Exception as e:
        raise Exception(e)
