=========
shapicant
=========

**shapicant** is a feature selection package based on `SHAP <https://github.com/slundberg/shap>`_ [LUN]_ and target permutation, for pandas and Spark.

It is inspired by PIMP [ALT]_, with some differences:

- PIMP fits a probability distribution to the population of null importances or, alternatively, uses a non-parametric estimation of the PIMP p-values. Instead, shapicant only implements the non-parametric estimation.
- For the non-parametric estimation, PIMP computes the fraction of null importances that are more extreme than the true importance (i.e. :code:`r/n`). Instead, shapicant computes it as :code:`(r+1)/(n+1)` [NOR]_.
- PIMP uses the Gini importance of Random Forest models or the Mutual Information criterion. Instead, shapicant uses SHAP values.
- While feature importance measures such as the Gini importance show an absolute feature importance, SHAP provides both positive and negative impacts. Instead of taking the mean absolute value of the SHAP values for each feature as feature importance, shapicant takes the mean value for positive and negative SHAP values separately. The true importance needs to be consistently higher than null importances for both positive and negative impacts. For multi-class classification, the true importance needs to be higher for at least one of the classes.
- While feature importance measures such as the Gini importance of Random Forest models are computed on the training set, SHAP values can be computed out-of-sample. Therefore, shapicant allows to compute them on a distinct validation set. To decide whether to compute them on the training set or on a validation set, you can refer to this discussion for "`Training vs. Test Data <https://compstat-lmu.github.io/iml_methods_limitations/pfi-data.html>`_" (it talks about PFI [BRE]_, which is a different algorithm, but the general idea is still applicable).

Permuting the response vector instead of permuting features has some advantages:

- The dependence between predictor variables remains unchanged.
- The number of permutations can be much smaller than the number of predictor variables for high dimensional datasets (unlike PFI [BRE]_) and there is no need to add shadow features (unlike Boruta [KUR]_).
- Since the features set does not change during iterations, the distributed implementation is more straightforward.

------------
Installation
------------

^^^^^^^^^^^^
Dependencies
^^^^^^^^^^^^

shapicant requires:

- Python (>= 3.6)
- shap (>= 0.36.0)
- numpy
- pandas
- scikit-learn
- tqdm

For Spark, we also need:

- pyspark (>= 2.4)
- pyarrow

^^^^^^^^^^^^^^^^^
User installation
^^^^^^^^^^^^^^^^^

The easiest way to install shapicant is using :code:`pip`

.. code:: bash

    pip install shapicant

or :code:`conda`

.. code:: bash

    conda install -c conda-forge shapicant

--------
Examples
--------

^^^^^^^^^^^^^^
PandasSelector
^^^^^^^^^^^^^^

If our data fit into the memory of a single machine, :code:`PandasSelector` is a sensible choice. This selector works on Pandas DataFrames and supports estimators that have a sklearn-like API.

First we’ll need to import a bunch of useful packages and generate some data to work with.

.. code:: python

    import pandas as pd
    from sklearn.datasets import make_classification
    from sklearn.model_selection import train_test_split

    # Generate a random classification problem
    X, y = make_classification(
        n_samples=1000,
        n_features=25,
        n_informative=3,
        n_redundant=2,
        n_repeated=2,
        n_classes=3,
        n_clusters_per_class=1,
        shuffle=False,
        random_state=42,
    )

    # PandasSelector works with pandas DataFrames, so convert X to a DataFrame
    X = pd.DataFrame(X)

    # Split training and validation sets
    # Note: in a real world setting, you probably want a test set as well
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.20, stratify=y, random_state=42)

We will use :code:`PandasSelector` with a LightGBM classifier in Random Forest mode and SHAP's TreeExplainer.

.. code:: python

    from shapicant import PandasSelector
    import lightgbm as lgb
    import shap

    # LightGBM in RandomForest-like mode (with rows subsampling), without columns subsampling
    model = lgb.LGBMClassifier(
        boosting_type="rf",
        subsample_freq=1,
        subsample=0.632,
        n_estimators=100,
        n_jobs=-1,
        random_state=42,
    )

    # This is the class (not its instance) of SHAP's TreeExplainer
    explainer_type = shap.TreeExplainer

    # Use PandasSelector with 100 iterations
    selector = PandasSelector(model, explainer_type, n_iter=100, random_state=42)

    # Run the feature selection
    # If we provide a validation set, SHAP values are computed on it, otherwise they are computed on the training set
    # We can also provide additional parameters to the underlying estimator's fit method through estimator_params
    selector.fit(X_train, y_train, X_validation=X_val, estimator_params={"categorical_feature": None})

    # Get the DataFrame with the selected features (with a p-value <= 0.05)
    X_train_selected = selector.transform(X_train, alpha=0.05)
    X_val_selected = selector.transform(X_val, alpha=0.05)

    # Just get the features list
    selected_features = selector.get_features(alpha=0.05)

    # We can also get the p-values as pandas Series
    p_values = selector.p_values_

^^^^^^^^^^^^^^
SparkSelector
^^^^^^^^^^^^^^

If our data does not fit into the memory of a single machine, :code:`SparkSelector` can be an alternative. This selector works on Spark DataFrames and supports PySpark estimators.

Please keep in mind the following caveats:

- Spark can add a lot of overhead, so if our data fit into the memory of a single machine, other selectors will be much faster.
- SHAP does not support categorical features with Spark estimators (see https://github.com/slundberg/shap/pull/721).
- Data provided to :code:`SparkSelector` is assumed to have already been preprocessed and each feature must correspond to a separate column. For example, if we want to one-hot encode a categorical feature, we must do so before providing the dataset to :code:`SparkSelector` and each binary variable must have its own column (Vector type columns are not supported).

Let's generate some data to work with.

.. code:: python

    import pandas as pd
    from sklearn.datasets import make_classification
    from pyspark.sql import SparkSession

    # Generate a random classification problem
    X, y = make_classification(
        n_samples=10000,
        n_features=25,
        n_informative=3,
        n_redundant=2,
        n_repeated=2,
        n_classes=3,
        n_clusters_per_class=1,
        shuffle=False,
        random_state=42,
    )

    # SparkSelector works with Spark DataFrames, so convert data to a DataFrame
    # Note: in a real world setting, you probably load data from parquet files or other sources
    spark = SparkSession.builder.getOrCreate()
    sdf = spark.createDataFrame(pd.DataFrame(X).assign(label=y))

    # Split training and validation sets (to keep the example simple, we don't split in a stratified fashion)
    # Note: in a real world setting, you probably want a test set as well
    sdf_train, sdf_val = sdf.randomSplit([0.80, 0.20], seed=42)

We will use :code:`SparkSelector` with a Random Forest classifier and SHAP's TreeExplainer.

.. code:: python

    from shapicant import SparkSelector
    from pyspark.ml.classification import RandomForestClassifier
    import shap

    # Spark's Random Forest (with bootstrap), without columns subsampling
    # Note: the "featuresCol" and "labelCol" parameters are ignored here, since they are set by SparkSelector
    model = RandomForestClassifier(
        featureSubsetStrategy="all",
        numTrees=20,
        seed=42,
    )

    # This is the class (not its instance) of SHAP's TreeExplainer
    explainer_type = shap.TreeExplainer

    # Use SparkSelector with 50 iterations
    selector = SparkSelector(model, explainer_type, n_iter=50, random_state=42)

    # Run the feature selection
    # If we provide a validation set, SHAP values are computed on it, otherwise they are computed on the training set
    selector.fit(sdf_train, label_col="label", sdf_validation=sdf_val, broadcast=True)

    # Get the DataFrame with the selected features (with a p-value <= 0.10)
    sdf_train_selected = selector.transform(sdf_train, label_col="label", alpha=0.10)
    sdf_val_selected = selector.transform(sdf_val, label_col="label", alpha=0.10)

    # Just get the features list
    selected_features = selector.get_features(alpha=0.10)

    # We can also get the p-values as pandas Series
    p_values = selector.p_values_

^^^^^^^^^^^^^^^^
SparkUdfSelector
^^^^^^^^^^^^^^^^

If we have a Spark cluster and our data fit into the memory of Spark executors, :code:`SparkUdfSelector` can be used to parallelize iterations. This selector works on Spark DataFrames and supports estimators that have a sklearn-like API.

Let's generate some data to work with.

.. code:: python

    import pandas as pd
    from sklearn.datasets import make_classification
    from pyspark.sql import SparkSession

    # Generate a random classification problem
    X, y = make_classification(
        n_samples=1000,
        n_features=25,
        n_informative=3,
        n_redundant=2,
        n_repeated=2,
        n_classes=3,
        n_clusters_per_class=1,
        shuffle=False,
        random_state=42,
    )

    # SparkUdfSelector works with Spark DataFrames, so convert data to a DataFrame
    # Note: in a real world setting, you probably load data from parquet files or other sources
    spark = SparkSession.builder.getOrCreate()
    sdf = spark.createDataFrame(pd.DataFrame(X).assign(label=y))

    # Split training and validation sets (to keep the example simple, we don't split in a stratified fashion)
    # Note: in a real world setting, you probably want a test set as well
    sdf_train, sdf_val = sdf.randomSplit([0.80, 0.20], seed=42)

We will use :code:`SparkUdfSelector` with a LightGBM classifier in Random Forest mode and SHAP's TreeExplainer.

.. code:: python

    from shapicant import SparkUdfSelector
    import lightgbm as lgb
    import shap

    # LightGBM in RandomForest-like mode (with rows subsampling), without columns subsampling
    model = lgb.LGBMClassifier(
        boosting_type="rf",
        subsample_freq=1,
        subsample=0.632,
        n_estimators=100,
        n_jobs=2,
        random_state=42,
    )

    # This is the class (not its instance) of SHAP's TreeExplainer
    explainer_type = shap.TreeExplainer

    # Use SparkUdfSelector with 100 iterations
    selector = SparkUdfSelector(model, explainer_type, n_iter=100, random_state=42)

    # Run the feature selection
    # If we provide a validation set, SHAP values are computed on it, otherwise they are computed on the training set
    # We can also provide additional parameters to the underlying estimator's fit method through estimator_params
    selector.fit(sdf_train, label_col="label", sdf_validation=sdf_val, estimator_params={"categorical_feature": None})

    # Get the DataFrame with the selected features (with a p-value <= 0.05)
    sdf_train_selected = selector.transform(sdf_train, label_col="label", alpha=0.05)
    sdf_val_selected = selector.transform(sdf_val, label_col="label", alpha=0.05)

    # Just get the features list
    selected_features = selector.get_features(alpha=0.05)

    # We can also get the p-values as pandas Series
    p_values = selector.p_values_

----------
References
----------

.. [LUN] Lundberg, S., & Lee, S.I. (2017). A unified approach to interpreting model predictions. In *Advances in Neural Information Processing Systems* (pp. 4765–4774).
.. [ALT] Altmann, A., Toloşi, L., Sander, O., & Lengauer, T. (2010). Permutation importance: a corrected feature importance measure *Bioinformatics, 26* (10), 1340-1347.
.. [NOR] North, B. V., Curtis, D., & Sham, P. C. (2002). A note on the calculation of empirical P values from Monte Carlo procedures. *American journal of human genetics, 71* (2), 439–441.
.. [BRE] Breiman, L. (2001). Random Forests *Machine Learning, 45* (1), 5–32.
.. [KUR] Kursa, M., & Rudnicki, W. (2010). Feature Selection with Boruta Package *Journal of Statistical Software, 36*, 1-13.
