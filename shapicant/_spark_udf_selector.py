"""
Class for the Spark UDF selector.

"""

import importlib
from typing import Callable, Dict, Optional, Type, Union

import numpy as np
import pandas as pd
from numpy.random import RandomState
from shap import Explainer
from sklearn.base import BaseEstimator
from tqdm import tqdm

try:
    from pyspark.sql import DataFrame, SparkSession
    from pyspark.sql import functions as F
except ImportError:
    DataFrame = None

from ._base import BaseSelector

SPARK_VALIDATION_NAME = "__shapicant_validation__"
SPARK_REPLICATION_NAME = "__shapicant_replication__"
SPARK_SIGN_NAME = "__shapicant_sign__"
SPARK_CLS_NAME = "__shapicant_cls__"


class SparkUdfSelector(BaseSelector):
    """Class for the Spark UDF selector in shapicant.

    Args:
        estimator: A supervised learning estimator with a 'fit' method.
        explainer_type: A SHAP explainer type.
        n_iter: The number of iterations to perform.
        verbose: Controls verbosity of output.
        random_state: Parameter to control the random number generator used.

    """

    def __init__(
        self,
        estimator: Union[BaseEstimator, Callable],
        explainer_type: Type[Explainer],
        n_iter: int = 100,
        verbose: Union[int, bool] = 1,
        random_state: Optional[Union[int, RandomState]] = None,
    ) -> None:
        super().__init__(
            estimator=estimator,
            explainer_type=explainer_type,
            n_iter=n_iter,
            verbose=verbose,
            random_state=random_state.randint(np.iinfo(np.int32).max)
            if isinstance(random_state, RandomState)
            else random_state,
        )

    def fit(
        self,
        sdf: DataFrame,
        label_col: str = "label",
        sdf_validation: Optional[DataFrame] = None,
        estimator_params: Optional[Dict[str, object]] = None,
        explainer_type_params: Optional[Dict[str, object]] = None,
        explainer_params: Optional[Dict[str, object]] = None,
    ) -> "SparkUdfSelector":
        """Fit the Spark UDF selector with the provided estimator.

        Args:
            sdf: The training input samples.
            label_col: The target column name.
            sdf_validation: The validation input samples.
            estimator_params: Additional parameters for the underlying estimator's fit method.
            explainer_type_params: Additional parameters for the explainer's init.
            explainer_params: Additional parameters for the explainer's shap_values method.

        """

        # Check if pyspark and pyarrow are installed
        if DataFrame is None or importlib.util.find_spec("pyarrow") is None:
            raise ImportError("SparkUdfSelector requires both pyspark and pyarrow.")

        # Get Spark session
        spark = SparkSession.builder.getOrCreate()

        # Validate parameters
        self._validate_params()

        # Merge the validation set into a single Spark DataFrame with an indicator column
        sdf = sdf.withColumn(SPARK_VALIDATION_NAME, F.lit(False))
        if sdf_validation is not None:
            sdf_validation = sdf_validation.withColumn(SPARK_VALIDATION_NAME, F.lit(True))
            if label_col not in sdf_validation.columns:
                sdf_validation = sdf_validation.withColumn(label_col, F.lit(None))
            sdf = sdf.unionByName(sdf_validation)

        # Get the shap values in a parallel way
        tqdm.write("Computing SHAP values...")
        n_replicas = 1 + self.n_iter
        sdf_replication = spark.range(n_replicas).withColumnRenamed("id", SPARK_REPLICATION_NAME)
        sdf_replicated = sdf.crossJoin(F.broadcast(sdf_replication)).repartition(n_replicas, SPARK_REPLICATION_NAME)
        sdf_shap_values = self._get_shap_values(
            estimator=self.estimator,
            explainer_type=self.explainer_type,
            sdf=sdf_replicated,
            label_col=label_col,
            estimator_params=estimator_params,
            explainer_type_params=explainer_type_params,
            explainer_params=explainer_params,
            random_state=self.random_state,
        )

        # Back to non-distributed regime with pandas
        shap_values = sdf_shap_values.toPandas()
        self._n_outputs = shap_values[SPARK_CLS_NAME].nunique()

        # Split positive and negative shap values
        pos_shap_values = shap_values[shap_values[SPARK_SIGN_NAME] == 1].drop(columns=SPARK_SIGN_NAME)
        neg_shap_values = shap_values[shap_values[SPARK_SIGN_NAME] == -1].drop(columns=SPARK_SIGN_NAME)

        # Extract the true shap values and the null shap values
        true_pos_shap_values = []
        true_neg_shap_values = []
        null_pos_shap_values = []
        null_neg_shap_values = []
        for i in range(self._n_outputs):
            cls_pos_shap_values = (
                pos_shap_values[pos_shap_values[SPARK_CLS_NAME] == i]
                .drop(columns=SPARK_CLS_NAME)
                .set_index(SPARK_REPLICATION_NAME)
                .T
            )
            cls_neg_shap_values = (
                neg_shap_values[neg_shap_values[SPARK_CLS_NAME] == i]
                .drop(columns=SPARK_CLS_NAME)
                .set_index(SPARK_REPLICATION_NAME)
                .T
            )
            true_pos_shap_values.append(cls_pos_shap_values.pop(0))
            true_neg_shap_values.append(cls_neg_shap_values.pop(0))
            null_pos_shap_values.append(cls_pos_shap_values)
            null_neg_shap_values.append(cls_neg_shap_values)

        # Compute p-values
        self.p_values_ = self._compute_p_values(
            true_pos_shap_values, null_pos_shap_values, true_neg_shap_values, null_neg_shap_values
        )

        # Cleanup
        self._n_outputs = None

        return self

    def transform(self, sdf: DataFrame, label_col: str = "label", alpha: float = 0.05) -> DataFrame:
        """Reduce data to the selected features.

        Args:
            sdf: The input samples.
            label_col: The target column name.
            alpha: Level at which the empirical p-values will get rejected.

        Returns:
            The input DataFrame reduced to the selected features and target.

        """

        selected_features = self.get_features(alpha=alpha)
        return sdf.select(selected_features + [label_col])

    def fit_transform(
        self,
        sdf: DataFrame,
        label_col: str = "label",
        sdf_validation: Optional[DataFrame] = None,
        estimator_params: Optional[Dict[str, object]] = None,
        explainer_type_params: Optional[Dict[str, object]] = None,
        explainer_params: Optional[Dict[str, object]] = None,
        alpha: float = 0.05,
    ) -> DataFrame:
        """Fit the Spark UDF selector and reduce data to the selected features.

        Args:
            sdf: The training input samples.
            label_col: The target column name.
            sdf_validation: The validation input samples.
            estimator_params: Additional parameters for the underlying estimator's fit method.
            explainer_type_params: Additional parameters for the explainer's init.
            explainer_params: Additional parameters for the explainer's shap_values method.
            alpha: Level at which the empirical p-values will get rejected.

        Returns:
            The input DataFrame reduced to the selected features and target.

        """

        self.fit(sdf, label_col, sdf_validation, estimator_params, explainer_type_params, explainer_params)
        return self.transform(sdf, label_col, alpha)

    @staticmethod
    def _get_shap_values(
        estimator: Union[BaseEstimator, Callable],
        explainer_type: Type[Explainer],
        sdf: DataFrame,
        label_col: str = "label",
        estimator_params: Optional[Dict[str, object]] = None,
        explainer_type_params: Optional[Dict[str, object]] = None,
        explainer_params: Optional[Dict[str, object]] = None,
        random_state: Optional[int] = None,
    ) -> DataFrame:
        def predict_contrib_udf(pdf):
            # Get the current replica
            current_replica = pdf.pop(SPARK_REPLICATION_NAME).iloc[0]

            # Split validation set and label
            X = pdf[~pdf[SPARK_VALIDATION_NAME]].drop(columns=SPARK_VALIDATION_NAME)
            y = X.pop(label_col)
            X_validation = pdf[pdf[SPARK_VALIDATION_NAME]].drop(columns=[SPARK_VALIDATION_NAME, label_col])

            # Don't shuffle to get true shap values, shuffle to get null shap values
            if current_replica != 0:
                sampling_seed = random_state + current_replica if random_state is not None else None
                y = y.sample(frac=1.0, random_state=sampling_seed)

            # Train the model
            fit = estimator.__self__.fit if callable(estimator) else estimator.fit
            fit(X, y.values, **estimator_params or {})

            # Explain the model
            explainer = explainer_type(estimator, **explainer_type_params or {})

            # If we have a validation set, compute shap values on it instead of the training set
            if not X_validation.empty:
                X = X_validation

            # Compute shap values
            shap_values = explainer.shap_values(X, **explainer_params or {})
            if not isinstance(shap_values, list):
                shap_values = [shap_values]

            # Average positive and negative shap values for each class
            df_shap_values = pd.DataFrame()
            for i, cls_shap_values in enumerate(shap_values):
                df_pos = pd.DataFrame(
                    data=np.mean(np.where(cls_shap_values >= 0, cls_shap_values, 0), axis=0).reshape(1, -1),
                    columns=X.columns,
                )
                df_neg = pd.DataFrame(
                    data=np.mean(np.where(cls_shap_values < 0, cls_shap_values, 0), axis=0).reshape(1, -1),
                    columns=X.columns,
                )
                df_pos[SPARK_SIGN_NAME] = 1
                df_neg[SPARK_SIGN_NAME] = -1
                df_pos[SPARK_CLS_NAME] = i
                df_neg[SPARK_CLS_NAME] = i
                df_shap_values = df_shap_values.append(df_pos).append(df_neg)

            # Add an indicator column for the current replica
            df_shap_values[SPARK_REPLICATION_NAME] = current_replica

            return df_shap_values

        # Build the Pandas UDF schema
        schema = ", ".join(
            [
                f"`{col}` DOUBLE"
                for col in sdf.columns
                if col not in (label_col, SPARK_VALIDATION_NAME, SPARK_REPLICATION_NAME)
            ]
            + [f"`{SPARK_SIGN_NAME}` BYTE", f"`{SPARK_CLS_NAME}` INT", f"`{SPARK_REPLICATION_NAME}` LONG"]
        )

        return sdf.groupby(SPARK_REPLICATION_NAME).applyInPandas(predict_contrib_udf, schema=schema)
