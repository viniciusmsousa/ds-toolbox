from itertools import combinations
from typing import List, Set, Dict, Tuple
from typeguard import typechecked
import pandas as pd
import numpy as np
from scipy.stats import chi2_contingency
import scikit_posthocs as sp


@typechecked
def contigency_chi2_test(
    df: pd.DataFrame, col_interest: str, col_groups: str
) -> Tuple:
    """Compute the Chi-Squared Contigency Table test, returning a formated
    DataFrame.
    Args:
        df (pd.DataFrame): DataFrame that contains the columns to create a
        contigency table.
        col_interest (str): Column name of the column of interest (the values
        from these colum will spreaded into columns to be compared).
        col_groups (str): Colum with the groups to be compared.
    Returns:
        tuple: First element is the scipy object with the test result.
        Second element is an analytical DataFrame.
    """
    try:
        # 1) Computing contigency table
        df_contigency = df\
            .value_counts([col_interest, col_groups])\
            .reset_index()\
            .pivot_table(index=col_groups, columns=col_interest, values=0)\
            .fillna(value=0)

        # 2) Computing Chi-Square Test
        stats_chi2_result = chi2_contingency(df_contigency)

        # 3) Creating Results table
        list_groups_interest = df[col_interest].unique().tolist()
        df_results = df_contigency.reset_index()
        for i in [*range(0, len(list_groups_interest))]:
            group_interest = list_groups_interest[i]

            # Adding expected value column
            str_col_expected = f'expected_{group_interest}'
            df_results[str_col_expected] = stats_chi2_result[3].T[i]

            # Adding comparison between Observed and Expected values columns
            str_col_dif = f'diff_{group_interest}'
            df_results[str_col_dif] = round(
                (df_results[group_interest] - df_results[str_col_expected]) /
                df_results[str_col_expected],
                4
            )

            # Adding percentual od observation within the group and population
            str_col_percent_group = f'percent_group_{group_interest}'
            str_col_percent_pop = f'percent_pop_{group_interest}'
            int_n_group = df.loc[df[col_interest] == group_interest].shape[0]
            int_n_pop = df.shape[0]
            df_results[str_col_percent_group] = round(
                (df_results[group_interest]/int_n_group)*100, 4
            )
            df_results[str_col_percent_pop] = round(
                (df_results[group_interest]/int_n_pop)*100, 4
            )

            # Renaming group of interest column
            str_obs_values = 'obs_{s}'
            df_results = df_results\
                .rename(
                    columns={
                        group_interest: str_obs_values
                        .format(s=group_interest)
                    }
                )

        # 4) Filtering expected values smaller then 5
        for g in list_groups_interest:
            df_results = df_results\
                .loc[df_results[str_col_expected.format(s=g)] >= 5]

        return (stats_chi2_result, df_results)
    except Exception as e:
        raise Exception(f'Error in compute_contigency_chi2_test(): {e}')





@typechecked
def mannwhitney_pairwise(
    df: pd.DataFrame, col_group: str,
    col_variable: str, p_value_threshold: float = 0.05
) -> pd.DataFrame:
    """Function to Compute a pairwise Mann Whitney Test.
    Args:
        df (pd.DataFrame): DataFrame with value column and group column
            to compute the test.
        col_group (str): Column name of the groups to be compared.
        col_variable (str): Columns name of the numeric variable
            to be compared.
        p_value_threshold (float, optional): Threshold to compare
            the p-value with. Defaults to 0.05.
    Returns:
        pd.DataFrame: DataFrame with the columns: group1, group2, variable,
            mean_variable_group1, mean_variable_group2, mw_pvalue,
            conclusion, group1_more_profitable.
    """
    try:
        # 1) Computing MannWhitney Test Pairwise
        df_mannwhitney = sp.posthoc_mannwhitney(
            df, val_col=col_variable, group_col=col_group
        )

        # 2) Getting possible group combinations
        uniques = [i for i in combinations(list(df[col_group].unique()), 2)]

        # 3) Empty dict to fill
        dict_test_moa = {
            'group1': list(),
            'group2': list(),
            'variable': list(),
            'mean_variable_group1': list(),
            'mean_variable_group2': list(),
            'mw_pvalue': list(),
            'conclusion': list()
        }

        # 4) Populating the Dict
        for tup in uniques:
            group1 = tup[0]
            group2 = tup[1]

            dict_test_moa['group1'].append(group1)
            dict_test_moa['group2'].append(group2)
            dict_test_moa['variable'].append(col_variable)
            dict_test_moa['mean_variable_group1'].append(float(round(
                df.loc[df[col_group] == group1][[col_variable]].mean(), 2
            )))
            dict_test_moa['mean_variable_group2'].append(float(round(
                df.loc[df[col_group] == group2][[col_variable]].mean(), 2
            )))
            dict_test_moa['mw_pvalue'].append(round(
                df_mannwhitney.loc[tup], 6
            ))
            dict_test_moa['conclusion'].append(
                str(np.where(
                    df_mannwhitney.loc[tup] <= p_value_threshold,
                    'Evidência estatística de que há diferença.',
                    'Evidência estatística de que NÃO há diferença.'
                ))
            )

        # 5) Computing the difference
        out_df = pd.DataFrame.from_dict(dict_test_moa)\
            .assign(group1_more_profitable=lambda df: np.where(
                df['mean_variable_group1'] > df['mean_variable_group2'],
                1, 0
            ))

        return out_df
    except Exception as e:
        raise Exception(f'Error in compute_pairwise_mannwhitney(): {e}')
