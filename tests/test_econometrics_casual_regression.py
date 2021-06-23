from ds_toolbox.econometrics.casual_regression import create_sm_formula, linear_coefficient, elasticity_ci

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

def test_linear_coefficient(df_ice_cream):
    assert linear_coefficient(df=df_ice_cream, y='sales', x='price') == 1.2293746061779545

def test_elasticity_ci(df_ice_cream):
    d = {
        'elasticity': 1.2293746061779545,
        'lower_ci': 0.9112897490212906,
        'upper_ci': 1.5474594633346184
    }
    assert elasticity_ci(df=df_ice_cream, y='sales', t='price', z=1.96) == d