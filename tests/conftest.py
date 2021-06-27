import pickle
from numpy import random
import pandas as pd
from pytest import fixture
from ds_toolbox.utils import start_local_spark


@fixture
def spark():
    return start_local_spark(max_mem=1, n_cores=4)

@fixture
def df_mannwhitney():
    s1 = random.normal(loc=1, scale=2, size=(1, 50))
    s2 = random.normal(loc=20, scale=7.6, size=(1, 50))
    dict_df = {
        'group': ['A'] * 50 + ['B'] * 50,
        'value': s1.tolist()[0] + s2.tolist()[0]
    }
    return pd.DataFrame.from_dict(dict_df)

@fixture
def df_chi2():
    dict_df = {
        'group_interest': ['A'] * 800 + ['B'] * 200 + ['C'] * 300,
        'group_compare': ['group1'] * 200 + ['group2'] * 800 + ['group3']*300
    }
    return pd.DataFrame.from_dict(dict_df)

@fixture
def df_ks():
    s1 = random.normal(loc=0.2, scale=0.05, size=(1, 50))
    s2 = random.normal(loc=0.6, scale=0.05, size=(1, 50))
    dict_df = {
        'group': [1] * 50 + [0] * 50,
        'value': s1.tolist()[0] + s2.tolist()[0]
    }
    return pd.DataFrame.from_dict(dict_df)

@fixture
def df_binary_classification_prediction():
    return pd.DataFrame({
        'target': [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0],
        'predicted': [1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0]
    })

@fixture
def dfs_binary_classification_prediction(spark, df_binary_classification_prediction):
    return spark.createDataFrame(df_binary_classification_prediction)

@fixture
def df_ab_test_pairwise():
    s1 = random.normal(loc=1, scale=2, size=(1, 50))
    s2 = random.normal(loc=20, scale=7.6, size=(1, 50))
    s3 = random.normal(loc=-30, scale=3.1, size=(1, 50))
    dict_df = {
        'group': ['A'] * 50 + ['B'] * 50 + ['C'] * 50,
        'value': s1.tolist()[0] + s2.tolist()[0] + s3.tolist()[0]
    }
    return pd.DataFrame.from_dict(dict_df)

@fixture
def dfs_ab_test_pairwise(spark, df_ab_test_pairwise):
    return spark.createDataFrame(df_ab_test_pairwise)

@fixture
def df_ice_cream():
    return pd.read_csv('tests/data/ice_creams.csv')

@fixture
def df_test_compute_cum_elasticity():
    return pd.read_csv('tests/data/df_test_compute_cum_elasticity.csv')

@fixture
def df_test_compute_cum_elasticity_output():
    return pd.read_csv('tests/data/df_test_compute_cum_elasticity_output.csv')

@fixture
def loaded_CausalRegression_instance():
    """Ground truth to test CausalRegression Class.
    Code that generated the instance
        import pickle  
        import pandas as pd
        from ds_toolbox.econometrics.causal_regression import CausalRegression

        prices = pd.read_csv("tests/data/ice_creams.csv")
        t = CausalRegression(
            df_train=prices.head(5000),
            df_test=prices.tail(5000),
            y='sales',
            t='price',
            numeric_regressors = ['temp', 'cost'],
            categorical_regressors = ['weekday'],
            test_size = 0.5,
            h = 0.01
        )

        test_file_causal_regression = open('tests/data/test_CausalRegression.pk', 'wb') 
        pickle.dump(t, test_file_causal_regression)
    """
    return pickle.load(open('tests/data/test_CausalRegression.pk', 'rb'))