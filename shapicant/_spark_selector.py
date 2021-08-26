"""
Class for the Spark selector.

"""

import importlib
import logging
import warnings
from typing import Dict, List, Optional, Tuple, Type, Union

import numpy as np
import pandas as pd
from numpy.random import RandomState
from pandas import Series
from shap import Explainer
from tqdm import tqdm

try:
    from pyspark.ml.feature import VectorAssembler
    from pyspark.ml.wrapper import JavaEstimator
    from pyspark.sql import Column, DataFrame
    from pyspark.sql import functions as F
    from pyspark.sql import types as T
except ImportError:
    JavaEstimator = None
    Column = None
    DataFrame = None

from ._base import BaseSelector

SPARK_FEATURES_NAME = "__shapicant_features__"
SPARK_INDEX_NAME = "__shapicant_index__"

logger = logging.getLogger(__name__)


class SparkSelector(BaseSelector):
    """Class for the Spark selector in shapicant.

    Args:
        estimator: A supervised learning estimator with a 'fit' method.
        explainer_type: A SHAP explainer type.
        n_iter: The number of iterations to perform.
        verbose: Controls verbosity of output.
        random_state: Parameter to control the random number generator used.

    """

    def __init__(
        self,
        estimator: JavaEstimator,
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
        self._current_iter = None
        self._X_with_index = None  # Cache

    def fit(
        self,
        sdf: DataFrame,
        label_col: str = "label",
        sdf_validation: Optional[DataFrame] = None,
        estimator_params: Optional[Dict[str, object]] = None,
        explainer_type_params: Optional[Dict[str, object]] = None,
        explainer_params: Optional[Dict[str, object]] = None,
        broadcast: bool = True,
    ) -> "SparkSelector":
        """Fit the Spark selector with the provided estimator.

        Args:
            sdf: The training input samples.
            label_col: The target column name.
            sdf_validation: The validation input samples.
            estimator_params: Additional parameters for the underlying estimator's fit method.
            explainer_type_params: Additional parameters for the explainer's init.
            explainer_params: Additional parameters for the explainer's shap_values method.
            broadcast: Whether to broadcast the target column when joining.

        """

        # Check if pyspark and pyarrow are installed
        if DataFrame is None or importlib.util.find_spec("pyarrow") is None:
            raise ImportError("SparkSelector requires both pyspark and pyarrow.")

        # Validate parameters
        self._validate_params()

        # Set estimator parameters
        self.estimator.setFeaturesCol(SPARK_FEATURES_NAME)
        self.estimator.setLabelCol(label_col)

        # Make sure that check_additivity is disabled (it's not supported for Spark estimators)
        explainer_params = self._set_additivity_false(explainer_params)

        # Assembly the features vector
        features = [col for col in sdf.columns if col != label_col]
        assembler = VectorAssembler(inputCols=features, outputCol=SPARK_FEATURES_NAME)
        sdf = assembler.transform(sdf)

        # With the progress bar
        with tqdm(total=self.n_iter, disable=not self.verbose) as pbar:
            # Get the true shap values (i.e. without shuffling)
            pbar.set_description("Computing true SHAP values")
            true_pos_shap_values, true_neg_shap_values = self._get_shap_values(
                sdf,
                label_col=label_col,
                shuffle=False,
                sdf_validation=sdf_validation,
                estimator_params=estimator_params,
                explainer_type_params=explainer_type_params,
                explainer_params=explainer_params,
            )

            # Get the null shap values (i.e. with shuffling)
            pbar.set_description("Computing null SHAP values")
            null_pos_shap_values = [None] * self._n_outputs
            null_neg_shap_values = [None] * self._n_outputs
            for i in range(self.n_iter):
                self._current_iter = i + 1
                if self.verbose:
                    logger.info(f"Iteration {self._current_iter}/{self.n_iter}")
                pos_shap_values, neg_shap_values = self._get_shap_values(
                    sdf,
                    label_col=label_col,
                    shuffle=True,
                    sdf_validation=sdf_validation,
                    estimator_params=estimator_params,
                    explainer_type_params=explainer_type_params,
                    explainer_params=explainer_params,
                    broadcast=broadcast,
                )
                for j in range(self._n_outputs):
                    if i == 0:
                        null_pos_shap_values[j] = pos_shap_values[j].to_frame()
                        null_neg_shap_values[j] = neg_shap_values[j].to_frame()
                    else:
                        null_pos_shap_values[j] = null_pos_shap_values[j].join(
                            pos_shap_values[j], rsuffix=f"_{self._current_iter}"
                        )
                        null_neg_shap_values[j] = null_neg_shap_values[j].join(
                            neg_shap_values[j], rsuffix=f"_{self._current_iter}"
                        )
                pbar.update(1)

        # Compute p-values
        self.p_values_ = self._compute_p_values(
            true_pos_shap_values, null_pos_shap_values, true_neg_shap_values, null_neg_shap_values
        )

        # Cleanup
        self._n_outputs = None
        self._X_with_index = None

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
        broadcast: bool = True,
        alpha: float = 0.05,
    ) -> DataFrame:
        """Fit the Spark selector and reduce data to the selected features.

        Args:
            sdf: The training input samples.
            label_col: The target column name.
            sdf_validation: The validation input samples.
            estimator_params: Additional parameters for the underlying estimator's fit method.
            explainer_type_params: Additional parameters for the explainer's init.
            explainer_params: Additional parameters for the explainer's shap_values method.
            broadcast: Whether to broadcast the target column when joining.
            alpha: Level at which the empirical p-values will get rejected.

        Returns:
            The input DataFrame reduced to the selected features and target.

        """

        self.fit(sdf, label_col, sdf_validation, estimator_params, explainer_type_params, explainer_params, broadcast)
        return self.transform(sdf, label_col, alpha)

    def _get_shap_values(
        self,
        sdf: DataFrame,
        label_col: str = "label",
        shuffle: bool = False,
        sdf_validation: DataFrame = None,
        estimator_params: Optional[Dict[str, object]] = None,
        explainer_type_params: Optional[Dict[str, object]] = None,
        explainer_params: Optional[Dict[str, object]] = None,
        broadcast: bool = True,
    ) -> Tuple[List[Series], List[Series]]:
        # Don't shuffle to get true shap values, shuffle to get null shap values
        if shuffle:
            shuffling_seed = self.random_state + self._current_iter if self.random_state is not None else None
            sdf = self._shuffle(sdf, label_col=label_col, broadcast=broadcast, seed=shuffling_seed)

        # Train the model
        model = self.estimator.fit(sdf, **estimator_params or {})

        # Explain the model
        explainer = self.explainer_type(model, **explainer_type_params or {})

        # If we have a validation set, compute shap values on it instead of the training set
        if sdf_validation is not None:
            sdf = sdf_validation

        # Compute shap values
        features = [col for col in sdf.columns if col not in (label_col, SPARK_FEATURES_NAME)]
        sdf = sdf.select(
            self._compute_shap_values(F.array(features), features, explainer, explainer_params).alias("shap_values")
        )
        if self._n_outputs is None:
            self._n_outputs = len(sdf.head()[0])
        sdf = sdf.select(*[F.col("shap_values")[i].alias(f"{i}") for i in range(self._n_outputs)])
        shap_values = [
            sdf.select(*[sdf[f"{i}"][j].alias(feature) for j, feature in enumerate(features)])
            for i in range(self._n_outputs)
        ]

        # Average positive and negative shap values for each class
        pos_shap_values = []
        neg_shap_values = []
        for cls_shap_values in shap_values:
            sdf_pos = cls_shap_values.select(
                *[
                    F.mean(F.when(F.col(col_name) >= 0, F.col(col_name)).otherwise(0)).name(col_name)
                    for col_name in cls_shap_values.columns
                ]
            )
            sdf_neg = cls_shap_values.select(
                *[
                    F.mean(F.when(F.col(col_name) < 0, F.col(col_name)).otherwise(0)).name(col_name)
                    for col_name in cls_shap_values.columns
                ]
            )
            # Back to "little data" regime with pandas
            s_pos = pd.Series(data=sdf_pos.head(), index=features, name="shap_values")
            s_neg = pd.Series(data=sdf_neg.head(), index=features, name="shap_values")
            pos_shap_values.append(s_pos)
            neg_shap_values.append(s_neg)

        return (pos_shap_values, neg_shap_values)

    def _shuffle(
        self,
        sdf: DataFrame,
        label_col: str = "label",
        broadcast: bool = True,
        seed: Optional[int] = None,
    ) -> DataFrame:
        # Take the target column
        y = sdf.select(label_col)

        # Take the feature columns
        X = sdf.drop(label_col)

        # Shuffle the target column
        y_shuffled = y.orderBy(F.rand(seed=seed))

        # Attach a sequential index to the shuffled target dataframe so we can join it back
        y_shuffled_with_index = self._attach_index(y_shuffled)

        # Attach a sequential index to the features dataframe so we can join it back
        # The features dataframe never changes, so we can compute it only the first time
        if self._X_with_index is None:
            self._X_with_index = self._attach_index(X).cache()

        # Join back the features and the shuffled target
        if broadcast:
            y_shuffled_with_index = F.broadcast(y_shuffled_with_index)
        sdf_shuffled = self._X_with_index.join(y_shuffled_with_index, on=SPARK_INDEX_NAME).drop(SPARK_INDEX_NAME)

        return sdf_shuffled

    @staticmethod
    def _set_additivity_false(explainer_params: Optional[Dict[str, object]]) -> Dict[str, object]:
        explainer_params = explainer_params or {}
        check_additivity = explainer_params.get("check_additivity", None)
        if check_additivity:
            warnings.warn("check_additivity is not supported for Spark estimators.")
        explainer_params["check_additivity"] = False
        return explainer_params

    @staticmethod
    def _attach_index(sdf: DataFrame) -> DataFrame:
        return sdf.rdd.zipWithIndex().map(lambda p: (p[1],) + tuple(p[0])).toDF([SPARK_INDEX_NAME] + sdf.columns)

    @staticmethod
    def _compute_shap_values(
        cols: Column, column_names: List[str], explainer: Explainer, explainer_params: Dict[str, object]
    ):
        @F.pandas_udf(returnType=T.ArrayType(T.ArrayType(T.DoubleType())))
        def predict_contrib_udf(cols):
            X = pd.DataFrame(data=np.vstack(cols.tolist()), columns=column_names)
            shap_values = explainer.shap_values(X, **explainer_params or {})
            if not isinstance(shap_values, list):
                shap_values = [shap_values]
            return pd.Series(zip(*shap_values))

        return predict_contrib_udf(cols)
