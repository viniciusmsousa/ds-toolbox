from typing import Union, List
from statsmodels import formula

from typeguard import typechecked
import pandas as pd
import numpy as np
import statsmodels as sm
import statsmodels.formula.api as smf
import seaborn as sns

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

@typechecked
def predict_elast(
    model: sm.regression.linear_model.RegressionResultsWrapper,
    df_test: pd.DataFrame,
    t: str,
    h: float = 0.01
) -> pd.core.series.Series:
    """Please refer to the CausalRegression class for a documentation.
    This function is used internally.

    Args:
        model (sm.regression.linear_model.RegressionResultsWrapper): A trained elasticity statsmodel model.
        df_test (pd.DataFrame): test DataFrame.
        t (str): Treatment column name.
        h (float, optional): Value to add to treatment column to compute the elasticity. Defaults to 0.01.

    Returns:
        float: Elasticity of the treatment column in the response variable from model.
    """
    return (model.predict(df_test.assign(t = df_test[t] + h).drop(columns = {t}).rename(columns={'t':t})) - model.predict(df_test)) / h

class CausalRegression:
    """Class that provides the fit on train data and evaluate on test data the elasticity of a treatment on a response variable.
    The objective is to separate the units from the dataset (customer, stores, etc.) according to the sensitivity os their response.
    The steps taken are based on the chapters 19-21 from the book Causal Inference for The Brave and True that can be found at
    https://github.com/matheusfacure/python-causality-handbook/tree/master.

    Attributes:
        formula_multiplicative_treatment_term (str): The formula of the multiplicative model, e.g., 'y ~ t*categorical_variables + t*numeric_variables + e'.
        m_elasticity (sm.regression.linear_model.RegressionResultsWrapper): Fitted model of the formula_multiplicative_treatment_term in the df_train.
        formula_y_x (str): The formula of the y variable dependent on the categorical and numeric features, e.g., 'y ~ categorical_variables + numeric_variables + e'.
        my (sm.regression.linear_model.RegressionResultsWrapper): Fitted model of the formula_y_x in the df_train.
        formula_t_x (str): The formula of the t variable dependent on the categorical and numeric features, e.g., 't ~ categorical_variables + numeric_variables + e'.
        mt (sm.regression.linear_model.RegressionResultsWrapper): Fitted model of the formula_t_x in the df_train.
        test_unbiased (pd.DataFrame): DataFrame with the original coluns, unbiased columns and predicted elasticity of the df_test.
        df_elasticity_ci (pd.DataFrame): DataFrame with cumulative elasticity (see the method plot_cumulative_elasticity_curve).

    Methods:
        fit_causal_regression: Fits the causal regression This method is called when the class is initiated.
        plot_cumulative_elasticity_curve: plots the cumulative elasticity curve. See chapter 20 os the book referenced in the class description.
    """
    @typechecked
    def __init__(
        self, df_train: pd.DataFrame, df_test: pd.DataFrame, y: str, t: str,
        numeric_regressors: Union[None, List], categorical_regressors: Union[None, List],
        h:float = 0.01 
    ):
        """Initiates the Class CausalRegression. This will compute the fit_causal_regression()
        method.

        Args:
            df_train (pd.DataFrame): Train DataFrame with the columns y, y, numeric_regressors and categorical_regressors.
            df_test (pd.DataFrame): Test DataFrame with the columns y, y, numeric_regressors and categorical_regressors.
            y (str): Column name of the response variable.
            t (str): Column name of the treatment variable.
            numeric_regressors (Union[None, List]): Column names of the numeric regressors.
            categorical_regressors (Union[None, List]): Column names of the categorical regressors.
            h (float, optional): Value to be added to each treatment in order to estimate the elasticity. Defaults to 0.01.
        """
        # Param Values
        self.df_train = df_train
        self.df_test = df_test
        self.y = y
        self.t = t
        self.numeric_regressors = numeric_regressors
        self.categorical_regressors = categorical_regressors
        self.h = h

        # Fitting the Causal Regression
        self.fit_causal_regression()
        self.df_elasticity_ci = compute_cum_elasticity(df=self.df_test_unbiased, predicted_elasticity='pred_elasticity', y = f'{self.y}(X)', t = f'{self.t}(X)', min_units = 30, steps = 100, z = 1.96)
    
    def fit_causal_regression(self):
        """This function computes the causal regression. The steps are [to-do]
        """

        # 1) Model Fit with Trainning Set (Regression with Multiplicative terms)
        formula_multiplicative_treatment_term = create_sm_formula(
            y=self.y, treatment_col=self.t,
            numeric_regressors=self.numeric_regressors,
            categorical_regressors=self.categorical_regressors
        )
        m_elasticity = smf.ols(formula_multiplicative_treatment_term, data=self.df_train).fit()

        self.formula_multiplicative_treatment_term = formula_multiplicative_treatment_term
        self.m_elasticity = m_elasticity

        # 2) Evaluating the Model (Cumulative Elasticity Curve)
        ## 2i) Removing the Cofounding Bias (Frisch-Waugh-Lovell [1933] Teorem)
        ### 2ia) Estimating the treatment from the features
        formula_t_x = create_sm_formula(
            y=self.t, numeric_regressors=self.numeric_regressors,
            categorical_regressors=self.categorical_regressors
        )
        mt = smf.ols(formula_t_x, data=self.df_train).fit()        

        self.formula_t_x = formula_t_x
        self.mt = mt

        ### 2ib) Estimating the response (y) variable from the features
        formula_y_x = create_sm_formula(
            y=self.y, numeric_regressors=self.numeric_regressors,
            categorical_regressors=self.categorical_regressors
        )
        my = smf.ols(formula_y_x, data=self.df_train).fit()

        self.formula_y_x = formula_y_x
        self.my = my

        ## 2ii) Adding the unbiased response and treatment values to the test set
        df_test_unbiased = self.df_test.assign(**{
            f'{self.y}(X)': self.df_test[self.y] - my.predict(self.df_test),
            f'{self.t}(X)': self.df_test[self.t] - mt.predict(self.df_test)
        })

        ## 2iii) Predicting the Elasticity in the test dataset
        df_test_unbiased = df_test_unbiased.assign(**{
            'pred_elasticity': predict_elast(
                model=self.m_elasticity,
                df_test=self.df_test,
                t=self.t,
                h=self.h
            )
        })
        self.df_test_unbiased = df_test_unbiased

    @typechecked
    def plot_cumulative_elasticity_curve(self, title: str = 'Cumulative Elasticity', min_units: int = 30, steps: int = 100, z: float = 1.96):
        """Plots the cumulative elastocity curve. See chapter 20 of the book indicated in the class init.

        Args:
            title (str, optional): Plot Title. Defaults to 'Cumulative Elasticity'.
            min_units (int, optional): Number of units in the first bin. Defaults to 30.
            steps (int, optional): Number of total buckets. Defaults to 100.
            z (float, optional): z-value for the normal distribution. Default value sets a 95% confidence interval. Defaults to 1.96.

        Returns:
            seabornplot
        """
        plot = sns.lineplot(
            data=pd.melt(self.df_elasticity_ci, id_vars=['units_count', 'units_proportion']),
            x='units_proportion',
            y='value',
            hue='variable',
            palette = ['black', 'gray', 'gray']
        )
        plot.set(title=title)
        plot.axhline(0, **{'c': 'lightgray'})
        plot.axhline(linear_coefficient(self.test_unbiased, y=f'{self.y}(X)', x=f'{self.t}(X)'), **{'ls':'--', 'c':'gray', 'label':'Average Elasticity'})
        return plot
