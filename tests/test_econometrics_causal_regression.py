from ds_toolbox.econometrics.causal_regression import create_sm_formula, linear_coefficient, elasticity_ci,\
    compute_cum_elasticity, CausalRegression

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

def test_compute_cum_elasticity(df_test_compute_cum_elasticity, df_test_compute_cum_elasticity_output):
    df_test = compute_cum_elasticity(
        df=df_test_compute_cum_elasticity, predicted_elasticity='pred_elasticity', y='sales(X)', t='price(X)', min_units=30, steps=40, z=1.96
    )
    assert all(round(df_test, 3) == round(df_test_compute_cum_elasticity_output, 3))

def test_CausalRegression_class(df_ice_cream, loaded_CausalRegression_instance):
    computed_CausalRegression_instance = CausalRegression(
        df_train=df_ice_cream.head(5000),
        df_test=df_ice_cream.tail(5000),
        y='sales',
        t='price',
        numeric_regressors = ['temp', 'cost'],
        categorical_regressors = ['weekday'],
        h = 0.01
    )

    # Testing the atributes computed in the Class
    assert computed_CausalRegression_instance.categorical_regressors == loaded_CausalRegression_instance.categorical_regressors
    assert computed_CausalRegression_instance.numeric_regressors == loaded_CausalRegression_instance.numeric_regressors
    assert all(computed_CausalRegression_instance.df_elasticity_ci == loaded_CausalRegression_instance.df_elasticity_ci)
    assert all(computed_CausalRegression_instance.df_test == loaded_CausalRegression_instance.df_test)
    assert all(computed_CausalRegression_instance.df_train == loaded_CausalRegression_instance.df_train)
    assert computed_CausalRegression_instance.formula_multiplicative_treatment_term == loaded_CausalRegression_instance.formula_multiplicative_treatment_term
    assert computed_CausalRegression_instance.formula_t_x == loaded_CausalRegression_instance.formula_t_x
    assert computed_CausalRegression_instance.formula_y_x == loaded_CausalRegression_instance.formula_y_x
    assert computed_CausalRegression_instance.h == loaded_CausalRegression_instance.h
    assert type(computed_CausalRegression_instance.m_elasticity) == type(loaded_CausalRegression_instance.m_elasticity)
    assert dir(computed_CausalRegression_instance.m_elasticity) == dir(loaded_CausalRegression_instance.m_elasticity)
    assert type(computed_CausalRegression_instance.my) == type(loaded_CausalRegression_instance.my)
    assert dir(computed_CausalRegression_instance.my) == dir(loaded_CausalRegression_instance.my)
    assert type(computed_CausalRegression_instance.mt) == type(loaded_CausalRegression_instance.mt)
    assert dir(computed_CausalRegression_instance.mt) == dir(loaded_CausalRegression_instance.mt)
    assert all(computed_CausalRegression_instance.df_test_unbiased == loaded_CausalRegression_instance.df_test_unbiased)
    assert computed_CausalRegression_instance.y == loaded_CausalRegression_instance.y
    assert computed_CausalRegression_instance.t == loaded_CausalRegression_instance.t