import pytest
import pandas as pd
from numpy import random

from ds_toolbox.statistics import compute_pairwise_mannwhitney

def test_compute_pairwise_mannwhitney():
    s1 = random.normal(loc=1, scale=2, size=(1, 50))
    s2 = random.normal(loc=20, scale=7.6, size=(1, 50))
    dict_df = {
        'group': ['A'] * 50 + ['B'] * 50,
        'value': s1.tolist()[0] + s2.tolist()[0]
    }
    df_test = pd.DataFrame.from_dict(dict_df)
    df_test_result = compute_pairwise_mannwhitney(df = df_test, col_group = 'group', col_variable = 'value')

    assert type(df_test_result) == pd.DataFrame
    assert df_test_result.columns.tolist() == ['group1', 'group2','variable', 'mean_variable_group1',
                                                'mean_variable_group2', 'mw_pvalue', 'conclusion',
                                                'group1_more_profitable']
    assert df_test_result['mw_pvalue'][0]==0