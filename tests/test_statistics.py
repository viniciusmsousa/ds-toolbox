import pandas as pd

from ds_toolbox.statistics import mannwhitney_pairwise, contigency_chi2_test, ks_test


# Mann-Whitney
def test_compute_pairwise_mannwhitney(df_mannwhitney):
    df_test_result = mannwhitney_pairwise(df = df_mannwhitney, col_group = 'group', col_variable = 'value')

    assert type(df_test_result) == pd.DataFrame
    assert df_test_result.columns.tolist() == ['group1', 'group2','variable', 'mean_variable_group1',
                                                'mean_variable_group2', 'mw_pvalue', 'conclusion']
    assert df_test_result['mw_pvalue'][0]==0

# Chi2
def test_contigency_chi2_test(df_chi2):
    stats_chi2_result, df_results = contigency_chi2_test(
        df=df_chi2,
        col_interest='group_interest',
        col_groups='group_compare'
    )

    assert type(stats_chi2_result) == tuple
    assert len(stats_chi2_result) == 4
    assert type(df_results) == pd.DataFrame
    assert df_results.shape == (3, 16)

# KS Test
def test_ks_test(df_ks):
    ks_out = ks_test(df=df_ks, col_target='group', col_probability='value')
    assert type(ks_out) == dict
    assert type(ks_out['ks_table']) == pd.DataFrame
    assert type(ks_out['max_ks']) == float

def test_ks_test_spark_df(spark, df_ks):
    dfs_test = spark.createDataFrame(df_ks.copy())
    
    ks_out = ks_test(df=dfs_test, spark=spark, col_target='group', col_probability='value')
    assert type(ks_out) == dict
    assert type(ks_out['ks_table']) == pd.DataFrame
    assert type(ks_out['max_ks']) == float
