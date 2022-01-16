Examples
========

PandasSelector
--------------

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

SparkSelector
-------------

If our data does not fit into the memory of a single machine, :code:`SparkSelector` can be an alternative. This selector works on Spark DataFrames and supports Spark ML estimators.

Please keep in mind the following caveats:

- Spark can add a lot of overhead, so if our data fit into the memory of a single machine, other selectors will be much faster.
- SHAP does not support categorical features with Spark ML estimators (see https://github.com/slundberg/shap/pull/721).
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

SparkUdfSelector
----------------

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
