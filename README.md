## DS-ToolBox

<!-- badges: start -->
[![PyPI Latest Release](https://img.shields.io/pypi/v/ds-toolbox.svg)](https://pypi.org/project/ds-toolbox/)
[![Codecov test coverage](https://codecov.io/gh/viniciusmsousa/ds-toolbox/branch/main/graph/badge.svg)](https://codecov.io/gh/viniciusmsousa/ds-toolbox?branch=main)
<!-- badges: end -->

A set of functions and classes that helps the analytical work of a data scientist. Full documentation can be found [here](https://viniciusmsousa.github.io/ds-toolbox/) 

### Instalation

The package can be installed either using PyPi:
```
pip install ds-toolbox
```

Or directly form github:
```
pip install git+https://github.com/viniciusmsousa/ds-toolbox.git#main
```

### Current availiable modules and functions are listed bellow:
- statistics:
    - `contigency_chi2_test`: Wrapper for [Scipy](https://github.com/scipy/scipy) function;
    - `mannwhitney_pairwise`: Wrapper for [scikit-posthocs](https://github.com/maximtrp/scikit-posthocs) function;
    - `ks_test`: Compute the [KS-Test](https://en.wikipedia.org/wiki/Kolmogorov%E2%80%93Smirnov_test), for Pandas and Spark DF.
- ml:
    - evaluator:
        - `binary_classifier_metrics`: Computes classification metrics (confusion_matrix, accuracy, f1, precision, recall, aucroc, aucpr) based on a dataframe (SparkDF or PandasDF) with ground truth and prediction.
