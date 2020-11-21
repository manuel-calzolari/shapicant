import lightgbm as lgb
import pandas as pd
import pytest
import shap
from pyspark.ml.classification import RandomForestClassifier
from pyspark.sql import SparkSession
from sklearn.datasets import make_classification
from shapicant import PandasSelector, SparkSelector


@pytest.fixture
def data():
    return make_classification(
        n_samples=1000,
        n_features=25,
        n_informative=3,
        n_redundant=2,
        n_repeated=0,
        n_classes=3,
        n_clusters_per_class=1,
        shuffle=False,
        random_state=42,
    )


def test_pandas_selector(data):
    X = pd.DataFrame(data[0])
    y = data[1]
    model = lgb.LGBMClassifier(
        boosting_type="rf", subsample_freq=1, subsample=0.632, n_estimators=100, n_jobs=-1, random_state=42
    )
    explainer_type = shap.TreeExplainer
    selector = PandasSelector(model, explainer_type, n_iter=50, random_state=42)
    selector.fit(X, y)
    X_selected = selector.transform(X, alpha=0.05)
    assert X_selected.columns.tolist() == [0, 1, 2, 3, 4]


def test_spark_selector(data):
    spark = SparkSession.builder.config("spark.sql.shuffle.partitions", "10").getOrCreate()
    sdf = spark.createDataFrame(pd.DataFrame(data[0]).assign(label=data[1]))
    model = RandomForestClassifier(featureSubsetStrategy="all", numTrees=20, seed=42)
    explainer_type = shap.TreeExplainer
    selector = SparkSelector(model, explainer_type, n_iter=10, random_state=42)
    selector.fit(sdf, label_col="label")
    sdf_selected = selector.transform(sdf, label_col="label", alpha=0.10)
    assert sdf_selected.columns == ["0", "1", "2", "3", "4", "label"]
