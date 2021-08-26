"""
Class for the Pandas selector.

"""

import logging
from typing import Callable, Dict, List, Optional, Tuple, Type, Union

import numpy as np
import pandas as pd
from numpy import array
from numpy.random import RandomState
from pandas import DataFrame, Series
from shap import Explainer
from sklearn.base import BaseEstimator
from tqdm import tqdm

from ._base import BaseSelector

logger = logging.getLogger(__name__)


class PandasSelector(BaseSelector):
    """Class for the Pandas selector in shapicant.

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
            random_state=random_state,
        )
        self._current_iter = None

    def fit(
        self,
        X: DataFrame,
        y: Union[array, Series, DataFrame],
        X_validation: Optional[DataFrame] = None,
        estimator_params: Optional[Dict[str, object]] = None,
        explainer_type_params: Optional[Dict[str, object]] = None,
        explainer_params: Optional[Dict[str, object]] = None,
    ) -> "PandasSelector":
        """Fit the Pandas selector with the provided estimator.

        Args:
            X: The training input samples.
            y: The target values.
            X_validation: The validation input samples.
            estimator_params: Additional parameters for the underlying estimator's fit method.
            explainer_type_params: Additional parameters for the explainer's init.
            explainer_params: Additional parameters for the explainer's shap_values method.

        """

        # Validate parameters
        self._validate_params()

        # Normalize target type to DataFrame/Series
        y = pd.DataFrame(y).squeeze()

        # With the progress bar
        with tqdm(total=self.n_iter, disable=not self.verbose) as pbar:
            # Get the true shap values (i.e. without shuffling)
            pbar.set_description("Computing true SHAP values")
            true_pos_shap_values, true_neg_shap_values = self._get_shap_values(
                X,
                y,
                shuffle=False,
                X_validation=X_validation,
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
                    X,
                    y,
                    shuffle=True,
                    X_validation=X_validation,
                    estimator_params=estimator_params,
                    explainer_type_params=explainer_type_params,
                    explainer_params=explainer_params,
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

        return self

    def transform(self, X: DataFrame, alpha: float = 0.05) -> DataFrame:
        """Reduce data to the selected features.

        Args:
            X: The input samples.
            alpha: Level at which the empirical p-values will get rejected.

        Returns:
            The input DataFrame reduced to the selected features.

        """

        selected_features = self.get_features(alpha=alpha)
        return X[selected_features]

    def fit_transform(
        self,
        X: DataFrame,
        y: Union[array, Series, DataFrame],
        X_validation: Optional[DataFrame] = None,
        estimator_params: Optional[Dict[str, object]] = None,
        explainer_type_params: Optional[Dict[str, object]] = None,
        explainer_params: Optional[Dict[str, object]] = None,
        alpha: float = 0.05,
    ) -> DataFrame:
        """Fit the Pandas selector and reduce data to the selected features.

        Args:
            X: The training input samples.
            y: The target values.
            X_validation: The validation input samples.
            estimator_params: Additional parameters for the underlying estimator's fit method.
            explainer_type_params: Additional parameters for the explainer's init.
            explainer_params: Additional parameters for the explainer's shap_values method.
            alpha: Level at which the empirical p-values will get rejected.

        Returns:
            The input DataFrame reduced to the selected features.

        """

        self.fit(X, y, X_validation, estimator_params, explainer_type_params, explainer_params)
        return self.transform(X, alpha)

    def _get_shap_values(
        self,
        X: DataFrame,
        y: Union[Series, DataFrame],
        shuffle: bool,
        X_validation: DataFrame = None,
        estimator_params: Optional[Dict[str, object]] = None,
        explainer_type_params: Optional[Dict[str, object]] = None,
        explainer_params: Optional[Dict[str, object]] = None,
    ) -> Tuple[List[Series], List[Series]]:
        # Don't shuffle to get true shap values, shuffle to get null shap values
        if shuffle:
            sampling_seed = self.random_state + self._current_iter if self.random_state is not None else None
            y = y.sample(frac=1.0, random_state=sampling_seed)

        # Train the model
        fit = self.estimator.__self__.fit if callable(self.estimator) else self.estimator.fit
        fit(X, y.values, **estimator_params or {})

        # Explain the model
        explainer = self.explainer_type(self.estimator, **explainer_type_params or {})

        # If we have a validation set, compute shap values on it instead of the training set
        if X_validation is not None:
            X = X_validation

        # Compute shap values
        shap_values = explainer.shap_values(X, **explainer_params or {})
        if not isinstance(shap_values, list):
            shap_values = [shap_values]
        if self._n_outputs is None:
            self._n_outputs = len(shap_values)

        # Average positive and negative shap values for each class
        pos_shap_values = []
        neg_shap_values = []
        for cls_shap_values in shap_values:
            s_pos = pd.Series(
                data=np.mean(np.where(cls_shap_values >= 0, cls_shap_values, 0), axis=0),
                index=X.columns,
                name="shap_values",
            )
            s_neg = pd.Series(
                data=np.mean(np.where(cls_shap_values < 0, cls_shap_values, 0), axis=0),
                index=X.columns,
                name="shap_values",
            )
            pos_shap_values.append(s_pos)
            neg_shap_values.append(s_neg)

        return (pos_shap_values, neg_shap_values)
