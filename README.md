## DS-ToolBox

A set of functions and classes that helps the analytical work of a data scientist.  

### Instalation

```
pip install git+https://github.com/viniciusmsousa/ds-toolbox.git#main
```

### Current availiable modules

- statistics:
    - `contigency_chi2_test`: Wrapper for [Scipy](https://github.com/scipy/scipy) function;
    - `mannwhitney_pairwise`: Wrapper for [scikit-posthocs](https://github.com/maximtrp/scikit-posthocs) function;
    - `ks_test`: Compute the [KS-Test](https://en.wikipedia.org/wiki/Kolmogorov%E2%80%93Smirnov_test), for Pandas and Spark DF.
- ml:
    - evaluator:
        - `binary_classifier_metrics`: Computes classification metrics (confusion_matrix, accuracy, f1, precision, recall, aucroc, aucpr) based on a dataframe (SparkDF and PandasDF) with ground truth and prediction.
