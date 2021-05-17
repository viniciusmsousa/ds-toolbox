import pandas as pd
from numpy import random
from pyspark.sql import SparkSession

from ds_toolbox.statistics import mannwhitney_pairwise, contigency_chi2_test, ks_test

# Mann Whitney
def test_compute_pairwise_mannwhitney():
    s1 = random.normal(loc=1, scale=2, size=(1, 50))
    s2 = random.normal(loc=20, scale=7.6, size=(1, 50))
    dict_df = {
        'group': ['A'] * 50 + ['B'] * 50,
        'value': s1.tolist()[0] + s2.tolist()[0]
    }
    df_test = pd.DataFrame.from_dict(dict_df)
    df_test_result = mannwhitney_pairwise(df = df_test, col_group = 'group', col_variable = 'value')

    assert type(df_test_result) == pd.DataFrame
    assert df_test_result.columns.tolist() == ['group1', 'group2','variable', 'mean_variable_group1',
                                                'mean_variable_group2', 'mw_pvalue', 'conclusion']
    assert df_test_result['mw_pvalue'][0]==0


# Contigency Chi2 Test
def test_contigency_chi2_test():
    dict_df = {
        'group_interest': ['A'] * 800 + ['B'] * 200 + ['C'] * 300,
        'group_compare': ['group1'] * 200 + ['group2'] * 800 + ['group3']*300
    }
    df_test = pd.DataFrame.from_dict(dict_df)
    stats_chi2_result, df_results = contigency_chi2_test(df=df_test, col_interest='group_interest', col_groups='group_compare')

    assert type(stats_chi2_result) == tuple
    assert len(stats_chi2_result) == 4
    assert type(df_results) == pd.DataFrame
    assert df_results.shape == (3, 16)

# KS test
s1 = random.normal(loc=0.2, scale=0.05, size=(1, 50))
s2 = random.normal(loc=0.6, scale=0.05, size=(1, 50))
dict_df = {
    'group': [1] * 50 + [0] * 50,
    'value': s1.tolist()[0] + s2.tolist()[0]
}
df_test = pd.DataFrame.from_dict(dict_df)

def test_ks_test():
    ks_out = ks_test(df=df_test, col_target='group', col_probability='value')
    assert type(ks_out) == dict
    assert type(ks_out['ks_table']) == pd.DataFrame
    assert type(ks_out['max_ks']) == float

def test_ks_test_spark_df():
    spark = SparkSession.builder\
            .master('local[1]') \
            .config('spark.executor.memory', '1G') \
            .config('spark.driver.memory', '1G') \
            .config('spark.memory.offHeap.enabled', 'true') \
            .config('spark.memory.offHeap.size', '1G') \
            .getOrCreate()

    dfs_test = spark.createDataFrame(df_test.copy())
    
    ks_out = ks_test(df=dfs_test, spark=spark, col_target='group', col_probability='value')
    assert type(ks_out) == dict
    assert type(ks_out['ks_table']) == pd.DataFrame
    assert type(ks_out['max_ks']) == float
