import pandas as pd

from ds_toolbox.statistics import mannwhitney_pairwise, contigency_chi2_test, ks_test, ab_test, ab_test_pairwise


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

# ab_test
def test_ab_test():
    test_result = ab_test(
        g1 = "A", g2 = 'B',
        confidence=0.95, h0=0,
        g1_mean = 0.722721,
        g1_std = 2.460937,
        g1_var = 6.056210,
        g1_count = 50,
        g2_mean = 19.774589,
        g2_std = 8.096614,
        g2_var = 65.555154,
        g2_count = 50
    )

    assert type(test_result) == pd.DataFrame 
    assert test_result.shape == (1, 8)
    assert list(test_result.columns) == ['Group1', 'Group2', 'Group1_95_Percent_CI', 'Group2_95_Percent_CI', 'Group1_Minus_Group2_95_Percent_CI', 'Z_statistic', 'P_Value', 'Conclusion']
    assert test_result['Group1_95_Percent_CI'].tolist()[0] == '0.722721 +- 0.6821243999567245'
    assert test_result['P_Value'].tolist()[0] == 2.3174051092612236e-57

def test_ab_test_pairwise_pandas(df_ab_test_pairwise):
    tests_results = ab_test_pairwise(
        df=df_ab_test_pairwise,
        col_group='group',
        col_variable='value',
        confidence=0.95,
        h0=0
    )

    assert type(tests_results) == pd.DataFrame
    assert tests_results.shape == (3, 8)
    assert list(tests_results.columns) == ['Group1', 'Group2', 'Group1_95_Percent_CI', 'Group2_95_Percent_CI', 'Group1_Minus_Group2_95_Percent_CI', 'Z_statistic', 'P_Value', 'Conclusion']

def test_ab_test_pairwise_spark(dfs_ab_test_pairwise):
    tests_results = ab_test_pairwise(
        df=dfs_ab_test_pairwise,
        col_group='group',
        col_variable='value',
        confidence=0.95,
        h0=0
    )

    assert type(tests_results) == pd.DataFrame
    assert tests_results.shape == (3, 8)
    assert list(tests_results.columns) == ['Group1', 'Group2', 'Group1_95_Percent_CI', 'Group2_95_Percent_CI', 'Group1_Minus_Group2_95_Percent_CI', 'Z_statistic', 'P_Value', 'Conclusion']
