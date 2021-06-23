from typing import Union, List

from typeguard import typechecked
import pandas as pd
import numpy as np

@typechecked
def create_sm_formula(y: str, numeric_regressors: Union[None, List] = None, categorical_regressors: Union[None, List] = None, treatment_col: Union[None, str] = None):
    """Creates a formula to be passed to a import statsmodels.formula.api.

    Args:
        y (str): Name of the y variable.
        numeric_regressors (Union[None, List], optional): List with name of the numeric regressors. Defaults to None.
        categorical_regressors (Union[None, List], optional): List os strings with the names of categorical regressors. Defaults to None.
        treatment_col (Union[None, str], optional): Name with the name of the treatment variable. Defaults to None.

    Raises:
        ValueError: At least one of numeric_regressors or categorical_regressors must be not None. If both are None ValueError will be raised

    Returns:
        str: str with the formula to be passed to a statsmodels.formula.api function.
    """
    if (numeric_regressors is None) & (categorical_regressors is None):
        raise ValueError('At least of one "numeric_regressors" or "categorical_regressors" must str list.')

    if treatment_col is None:
        if categorical_regressors is None:
            fit_formula = f'{y} ~ ' + ' + '.join(numeric_regressors)
        elif numeric_regressors is None:
            fit_formula = f'{y} ~ ' + ' + '.join([f'C({f})' for f in categorical_regressors])
        else:
            fit_formula = f'{y} ~ ' + ' + '.join(numeric_regressors) + ' + ' + ' + '.join([f'C({f})' for f in categorical_regressors])
    else:
        if categorical_regressors is None:
            fit_formula = f'{y} ~ {treatment_col}*' + f' + {treatment_col}*'.join(numeric_regressors)
        elif numeric_regressors is None:
            fit_formula = f'{y} ~ {treatment_col}*' + f' + {treatment_col}*'.join([f'C({f})' for f in categorical_regressors])
        else:
            fit_formula = f'{y} ~ {treatment_col}*' + f' + {treatment_col}*'.join(numeric_regressors) + f' + {treatment_col}*' + f' + {treatment_col}*'.join([f'C({f})' for f in categorical_regressors])
    return fit_formula

@typechecked
def linear_coefficient(df: pd.DataFrame, y: str, x: str):
    """Computes the linear regression coefficient (OLS).

    Args:
        data (pd.DataFrame): pd.DataFrame.
        y (str): column name of the y variable.
        x (str): columns name of the regressor variable.

    Returns:
        np.float64: The linear coefficient.
    """
    return (np.sum((df[x] - df[x].mean())*(df[y] - df[y].mean())) / np.sum((df[x] - df[x].mean())**2))

