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
def linear_coefficient(df: pd.DataFrame, y: str, x: str) -> np.float64:
    """Computes the linear regression coefficient (OLS).

    Args:
        data (pd.DataFrame): pd.DataFrame.
        y (str): column name of the y variable.
        x (str): columns name of the regressor variable.

    Returns:
        np.float64: The linear coefficient.
    """
    return (np.sum((df[x] - df[x].mean())*(df[y] - df[y].mean())) / np.sum((df[x] - df[x].mean())**2))

@typechecked
def elasticity_ci(df: pd.DataFrame, y: str, t: str, z: float = 1.96) -> dict:
    """Computes the confidence interval of a linear coefficient of a regression.
    Used to compute the elasticity confidence interval.

    Args:
        df (pd.DataFrame): Data Frame with y and t columns.
        y (str): Column name of the y variable.
        t (str): Column name of the treatment variable.
        z (float, optional): z value for the normal distribution. Defaults to 1.96.

    Returns:
        dict: dict in the form {'elasticity': float, 'lower_ci': float, 'upper_ci': float}.
    """
    n = df.shape[0]
    t_bar = df[t].mean()
    beta1 = linear_coefficient(df=df, y=y, x=t)
    beta0 = df[y].mean() - beta1 * t_bar
    e = df[y] - (beta0 + beta1*df[t])
    se = np.sqrt(((1/(n-2))*np.sum(e**2))/np.sum((df[t]-t_bar)**2))

    out_dict = {
        'elasticity': beta1,
        'lower_ci': beta1 - z*se,
        'upper_ci': beta1 + z*se
    }
    return out_dict

@typechecked
def compute_cum_elasticity(df: pd.DataFrame, predicted_elasticity: str, y: str, t: str, min_units: int = 30, steps: int = 100, z: float = 1.96) -> pd.DataFrame:
    """Computes the cumulative elasticity from a data frame with the predicted elasticity.
    The result of this function is used to evaluate if a casual model is detecting the heterogeneity in treatment response.

    Args:
        df (pd.DataFrame): DataFrame resulted with a elasticity prediction.
        predicted_elasticity (str): Column name of the predicted elasticity variable.
        y (str): Column name of the response variable (y).
        t (str): Column name of the treatment variable (t).
        min_units (int, optional): Number of units to add in each step. Defaults to 30.
        steps (int, optional): [description]. Defaults to 100.
        z (float, optional): [description]. Defaults to 1.96.

    Raises:
        ValueError: If 'min_units' value is greater then the number of units in the dataset (df.shape[0]) this error is raised. 

    Returns:
        pd.DataFrame: [description]
    """
    # 1) Computing help values
    size = df.shape[0]
    if min_units >= size:
        raise ValueError('Choose a min_units value that is smaller then the number os units in the dataset.')
    df_ordered = df.sort_values(predicted_elasticity, ascending=False).reset_index(drop=True)
    nrows = list(range(min_units, size, size //steps)) + [size]

    df_elasticity_ci = {
        'units_count': list(),
        'units_proportion': list(),
        'cum_elasticity': list(),
        'lower_ci': list(),
        'upper_ci': list() 
    }
    for rows in nrows:
        df_elasticity_ci['units_count'].append(rows)
        df_elasticity_ci['units_proportion'].append(round(rows/max(nrows), 4))

        elasticity = elasticity_ci(df=df_ordered.head(rows), y=y, t=t, z=z)
        df_elasticity_ci['cum_elasticity'].append(elasticity['elasticity'])
        df_elasticity_ci['lower_ci'].append(elasticity['lower_ci'])
        df_elasticity_ci['upper_ci'].append(elasticity['upper_ci'])

    df_elasticity_ci = pd.DataFrame.from_dict(df_elasticity_ci)

    return df_elasticity_ci
