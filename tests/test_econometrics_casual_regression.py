from numpy import tracemalloc_domain
from ds_toolbox.econometrics.casual_regression import create_sm_formula

def test_create_sm_formula():
    formula = create_sm_formula(
        y='y', treatment_col='t',
        numeric_regressors=['n1', 'n2'],
        categorical_regressors=['c1', 'c2']
    )
    assert formula == 'y ~ t*n1 + t*n2 + t*C(c1) + t*C(c2)'

    formula = create_sm_formula(
        y='y',
        numeric_regressors=['n1', 'n2'],
        categorical_regressors=['c1', 'c2']
    )
    assert formula == 'y ~ n1 + n2 + C(c1) + C(c2)'

    formula = create_sm_formula(
        y='y', treatment_col='t',
        numeric_regressors=['n1'],
        categorical_regressors=['c2']
    )
    assert formula == 'y ~ t*n1 + t*C(c2)'

    formula = create_sm_formula(
        y='y',
        numeric_regressors=['n1'],
        categorical_regressors=['c2']
    )
    assert formula == 'y ~ n1 + C(c2)'

    formula = create_sm_formula(
        y='y', treatment_col='t',
        categorical_regressors=['c1', 'c2']
    )
    assert formula == 'y ~ t*C(c1) + t*C(c2)'

    formula = create_sm_formula(
        y='y',
        numeric_regressors=['n1', 'n2']
    )
    assert formula == 'y ~ n1 + n2'