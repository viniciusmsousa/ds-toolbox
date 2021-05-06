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
        raise Exception(e)





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
            dict_test_moa['mean_variable_group1'].append(float(
                df.loc[df[col_group] == group1][[col_variable]].mean()
            ))
            dict_test_moa['mean_variable_group2'].append(float(
                df.loc[df[col_group] == group2][[col_variable]].mean()
            ))
            dict_test_moa['mw_pvalue'].append(round(
                df_mannwhitney.loc[tup], 6
            ))
            dict_test_moa['conclusion'].append(
                str(np.where(
                    df_mannwhitney.loc[tup] <= p_value_threshold,
                    'Statistical evidence that the groups are different.',
                    'Statistical evidence that there is no difference between groups..'
                ))
            )

        # 5) Computing the difference
        out_df = pd.DataFrame.from_dict(dict_test_moa)

        return out_df
    except Exception as e:
        raise Exception(e)

@typechecked
def ks_test(df: pd.DataFrame, col_target: str, col_probability: str) -> Dict:
    """Function to compute a Ks Test and depicts a detailed result table.
    https://en.wikipedia.org/wiki/Kolmogorov%E2%80%93Smirnov_test

    Args:
        df (pd.DataFrame): DataFrame with Probability and Classification value.
        col_target (str): Column name with classification value.
        col_probability (int): Column name with probability values.

    Returns:
        Dict: Dict with 'ks_table': table with the results and 'max_ks': Max KS Value. 
    """
    try:
        # Computing the KS
        df['target0'] = 1 - df[col_target]
        df['prob_qcut'] = pd.qcut(df[col_probability], 10, duplicates='drop')
        grouped = df.groupby('prob_qcut', as_index = False)
        ks_table = pd.DataFrame()
        ks_table['min_prob'] = grouped.min()[col_probability]
        ks_table['max_prob'] = grouped.max()[col_probability]
        ks_table['events']   = grouped.sum()[col_target]
        ks_table['nonevents'] = grouped.sum()['target0']
        ks_table = ks_table.sort_values(by="min_prob", ascending=False).reset_index(drop = True)
        ks_table['event_rate'] = (ks_table.events / df[col_target].sum()).apply('{0:.2%}'.format)
        ks_table['nonevent_rate'] = (ks_table.nonevents / df['target0'].sum()).apply('{0:.2%}'.format)
        ks_table['cum_eventrate']=(ks_table.events / df[col_target].sum()).cumsum()
        ks_table['cum_noneventrate']=(ks_table.nonevents / df['target0'].sum()).cumsum()
        ks_table['KS'] = np.round(ks_table['cum_eventrate']-ks_table['cum_noneventrate'], 3) * 100

        # Formating the Detailed Table
        ks_table['cum_eventrate']= ks_table['cum_eventrate'].apply('{0:.2%}'.format)
        ks_table['cum_noneventrate']= ks_table['cum_noneventrate'].apply('{0:.2%}'.format)
        ks_table.index = range(1, (ks_table.shape[0]+1))
        ks_table.index.rename('percentile', inplace=True)

        # Output Dictionary
        out_dict = {
            'ks_table': ks_table,
            'max_ks': max(ks_table['KS'])
        }

        return out_dict
    except Exception as e:
        raise Exception(e)
