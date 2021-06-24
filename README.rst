
DS-ToolBox
----------

A set of functions to help the analytical work of a data scientist. Full documentation can be found in `Package Homepage <https://viniciusmsousa.github.io/ds-toolbox/>`_. The main motivation of the package is to facilitate the taks by using a common input and output structure, SparkDF and PandasDF.

Instalation
^^^^^^^^^^^

The package can be installed either using PyPi:

.. code-block::

   pip install ds-toolbox

Or directly form github:

.. code-block::

   pip install git+https://github.com/viniciusmsousa/ds-toolbox.git#main

Current availiable modules and functions are listed bellow:
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^


* statistics:

  * `contigency_chi2_test`: Wrapper for [Scipy](https://github.com/scipy/scipy) function;
  * `mannwhitney_pairwise`: Wrapper for [scikit-posthocs](https://github.com/maximtrp/scikit-posthocs) function;
  * `ks_test`: Compute the [KS-Test](https://en.wikipedia.org/wiki/Kolmogorov%E2%80%93Smirnov_test), for Pandas and Spark DF;
  * ``ab_test_pairwise``\ : A Simple ab test based on mean, std and var, PandasDF and Spark DF.

* ml:

  * evaluator:

    * ``binary_classifier_metrics``\ : Computes classification metrics (confusion_matrix, accuracy, f1, precision, recall, aucroc, aucpr) based on a dataframe (SparkDF or PandasDF) with ground truth and prediction.

* econometrics:

  * causal_regression:

    * `Causalregression`: A class built on top of what is presented in the chapters 19-21 from the book [Causal Inference for The Brave and True](https://github.com/matheusfacure/python-causality-handbook/tree/master).
