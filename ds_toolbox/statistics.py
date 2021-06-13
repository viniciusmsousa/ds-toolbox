import sys
from itertools import combinations
from typing import Union, Dict, Tuple
from typeguard import typechecked

import pandas as pd
import numpy as np
from scipy import stats
from scipy.stats import chi2_contingency
import scikit_posthocs as sp

import pyspark
from pyspark.sql import SparkSession
import pyspark.ml.feature as FF
from pyspark.sql import functions as F
from pyspark.sql.window import Window

from ds_toolbox.spark_utils import start_local_spark


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
        df (pd.DataFrame): DataFrame with value column and group column to compute the test.
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
def ks_test(
    df: Union[pd.DataFrame, pyspark.sql.dataframe.DataFrame],
    col_target: str, col_probability: str, spark: Union[pyspark.sql.session.SparkSession, None] = None,
    max_mem: int = 2, n_cores: int =4
) -> Dict:
    """Function to compute a Ks Test and depicts a detailed result table.
    https://en.wikipedia.org/wiki/Kolmogorov%E2%80%93Smirnov_test

    Args:
        df (Union[pd.DataFrame, pyspark.sql.dataframe.DataFrame]): DataFrame with Probability and Classification value.
        col_target (str): Column name with classification value.
        col_probability (int): Column name with probability values.
        max_mem (int, optional): Max memory to be allocated to local spark if df is pandasDF. 
        n_cores (int, optional): Number of cores to be allocated to local spark if df is pandas.

    Raises:
        ValueError: If df is sparkDF an spark is None.

    Returns:
        Dict: Dict with 'ks_table': table with the results and 'max_ks': Max KS Value. 
    """
    try:
        if spark is None and type(df) == pyspark.sql.dataframe.DataFrame:
            raise ValueError('type(spark) is None. Please pass a valid spark session to spark argument.')

        if type(df) == pd.DataFrame:
            input_pandas = True
            spark = start_local_spark(max_mem=max_mem, n_cores=n_cores)
            dfs = spark.createDataFrame(df)
        else:
            input_pandas = False
            dfs = df
        
        dfs = dfs.withColumn('target0',1 - dfs[col_target])
        qds = FF.QuantileDiscretizer(numBuckets=10,inputCol=col_probability, outputCol="prob_qcut", relativeError=0, handleInvalid="error")
        dfs = qds.setHandleInvalid("skip").fit(dfs).transform(dfs)

        ks_table = dfs\
                .groupBy('prob_qcut')\
                .agg(
                    F.min(col_probability),
                    F.max(col_probability),
                    F.sum(col_target),
                    F.sum('target0')
                )\
                .orderBy(F.col(f'min({col_probability})').desc())\
                .withColumnRenamed(f'min({col_probability})', 'min_prob')\
                .withColumnRenamed(f'max({col_probability})', 'max_prob')\
                .withColumnRenamed(f'sum({col_target})', 'events')\
                .withColumnRenamed('sum(target0)','non_events')
        ks_table = ks_table.withColumn('event_rate', ks_table.events/dfs.filter(f'{col_target} == 1').count())
        ks_table = ks_table.withColumn('nonevent_rate', ks_table.non_events/dfs.filter(f'{col_target} == 0').count())

        win = Window.partitionBy().orderBy().rowsBetween(-sys.maxsize, 0)
        ks_table = ks_table.withColumn('cum_eventrate', F.sum(ks_table.event_rate).over(win))
        ks_table = ks_table.withColumn('cum_noneventrate', F.sum(ks_table.nonevent_rate).over(win))

        ks_table = ks_table.withColumn('ks', F.expr('cum_eventrate - cum_noneventrate'))

        ks_table = ks_table.withColumn('percentile', F.row_number().over(Window.orderBy(F.lit(1))))

        ks_table = ks_table.select(
            'percentile', 'min_prob', 'max_prob', 'events', 'non_events',
            'event_rate', 'nonevent_rate', 'cum_eventrate', 'cum_noneventrate',
            'ks'
        )

        out_dict = {
            'ks_table': ks_table.toPandas(),
            'max_ks': ks_table.agg({"ks": "max"}).collect()[0][0]
        }

        if type(input_pandas):
            spark.stop()

        return out_dict
    except Exception as e:
        raise Exception(e)

@typechecked
def ab_test(
    g1: str, g2: str, 
    g1_mean: float, g1_std: float, g1_var: float, g1_count: int,
    g2_mean: float, g2_std: float, g2_var: float, g2_count: int,
    confidence=0.95, h0=0
) -> pd.DataFrame:
    """Internal Function. Please refer to ab_test_pairwise.

    Args:
        g1 (str): Group 1 Identifier.
        g2 (str): Group 2 Identifier.
        g1_mean (float): Group 1 Mean of Variable of interest.
        g1_std (float): Group 1 Standard Deviation os Variable of interest.
        g1_var (float): Group 1 Variance of variable of interest
        g1_count (int): Group 1 number of observations.
        g2_mean (float): Same as Group 1 but for Group 2.
        g2_std (float): Same as Group 1 but for Group 2.
        g2_var (float): Same as Group 1 but for Group 2.
        g2_count (int): Same as Group 1 but for Group 2.
        confidence (float, optional): Desired Confidence Level. Defaults to 0.95.
        h0 (int, optional): Null hypothesis difference between Group 1 and Group 2 in variable of interest. Defaults to 0.

    Raises:
        Exception: Any error will be raised as an exception.

    Returns:
        pd.DataFrame: DataFrame with the columns
            - Group1
            - Group2
            - Group1_{confidence*100}_Percent_CI
            - Group2_{confidence*100}_Percent_CI
            - Group1_Minus_Group2_{confidence*100}_Percent_CI
            - Z_statistic
            - P_Value
    """
    try:
        se1, se2 = g1_std / np.sqrt(g1_count), g2_std / np.sqrt(g2_count)
        
        diff = g1_mean - g2_mean
        se_diff = np.sqrt(g1_var/g1_count + g2_var/g2_count)
        
        z_stats = (diff-h0)/se_diff
        p_value = stats.norm.cdf(z_stats)
        
        def critial(se): return -se*stats.norm.ppf((1 - confidence)/2)
        
        ci_percent = int(confidence*100)
        out_dict = {
            'Group1': [g1],
            'Group2': [g2],
            f'Group1_{ci_percent}_Percent_CI': [f'{g1_mean} +- {critial(se1)}'],
            f'Group2_{ci_percent}_Percent_CI': [f'{g2_mean} +- {critial(se2)}'],
            f'Group1_Minus_Group2_{ci_percent}_Percent_CI': [f'{diff} +- {critial(se_diff)}'],
            'Z_statistic': [z_stats],
            'P_Value': [p_value],
            'Conclusion': [str(np.where(
                p_value <= (1-confidence),
                'Statistical evidence that the groups are different.',
                'Statistical evidence that there is no difference between groups.'
            ))]
        }

        return pd.DataFrame(out_dict)
    except Exception as e:
        raise Exception(e)

@typechecked
def ab_test_pairwise(
    df: Union[pd.DataFrame, pyspark.sql.dataframe.DataFrame], col_group: str, col_variable: str, confidence: float = 0.95, h0: float = 0
) -> pd.DataFrame:
    """Function that computes a simple AB test (based on mean, std and var) for each pair of a categorical column. Works with Both PandasDF and SparkDF.

    Args:
        df (Union[pd.DataFrame, pyspark.sql.dataframe.DataFrame]): Data Frame with a group column and a numeric variable column.
        col_group (str): Column name with group column. Distinct values of this column will the used as comparison in pairwise.
        col_variable (str): Variable column. Numeric column with the output to be compared between the group column.
        confidence (float, optional): Desired Confidence Level. Defaults to 0.95.
        h0 (float, optional): Null hypothesis value. Defaults to 0.

    Raises:
        Exception: Erros.

        pd.DataFrame: DataFrame one row per possible pair between values from col_group and the following columns
            - Group1
            - Group2
            - Group1_{confidence*100}_Percent_CI
            - Group2_{confidence*100}_Percent_CI
            - Group1_Minus_Group2_{confidence*100}_Percent_CI
            - Z_statistic
            - P_Value
    """
    try:
        # 1) Getting possible group combinations and Summary statistics on each group
        if type(df) == pd.DataFrame:
            uniques = [i for i in combinations(list(df[col_group].unique()), 2)]
            df_group_summary = df.groupby([col_group]).agg({col_variable:['mean', 'std', 'var', 'count']}).reset_index().droplevel(0, axis=1)
        else:
            group_units = list(df.select(col_group).drop_duplicates().toPandas()[col_group])
            uniques = [i for i in combinations(group_units, 2)]
            df_group_summary = df.groupBy([col_group]).agg(F.mean(col_variable), F.stddev_pop(col_variable), F.var_pop(col_variable), F.count(col_variable)).toPandas()
            
        df_group_summary.columns = [col_group, f'{col_variable}_mean', f'{col_variable}_std', f'{col_variable}_var', f'{col_variable}_count']
        
        # 2) Results DataFrame to be filled
        tests_result = pd.DataFrame()
 
        # 4) Populating the Dict
        for tup in uniques:
            test_result = ab_test(
                g1=tup[0], g2 = tup[1],
                confidence=confidence, h0=h0,
                g1_mean = df_group_summary.loc[df_group_summary[col_group] ==  tup[0]][f'{col_variable}_mean'].tolist()[0],
                g1_std = df_group_summary.loc[df_group_summary[col_group] ==  tup[0]][f'{col_variable}_std'].tolist()[0],
                g1_var = df_group_summary.loc[df_group_summary[col_group] ==  tup[0]][f'{col_variable}_var'].tolist()[0],
                g1_count = df_group_summary.loc[df_group_summary[col_group] ==  tup[0]][f'{col_variable}_count'].tolist()[0],
                g2_mean = df_group_summary.loc[df_group_summary[col_group] ==  tup[1]][f'{col_variable}_mean'].tolist()[0],
                g2_std = df_group_summary.loc[df_group_summary[col_group] ==  tup[1]][f'{col_variable}_std'].tolist()[0],
                g2_var = df_group_summary.loc[df_group_summary[col_group] ==  tup[1]][f'{col_variable}_var'].tolist()[0],
                g2_count = df_group_summary.loc[df_group_summary[col_group] ==  tup[1]][f'{col_variable}_count'].tolist()[0]
            )

            tests_result = tests_result.append(test_result)

        return tests_result
    except Exception as e:
        raise Exception(e)