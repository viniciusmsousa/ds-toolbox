import pytest
import pandas as pd
from numpy import random

from ds_toolbox.statistics import mannwhitney_pairwise, contigency_chi2_test

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
                                                'mean_variable_group2', 'mw_pvalue', 'conclusion',
                                                'group1_more_profitable']
    assert df_test_result['mw_pvalue'][0]==0


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
