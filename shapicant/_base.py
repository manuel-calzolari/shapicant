"""
Base class for all selectors.

"""

from abc import ABCMeta, abstractmethod
from functools import reduce
from typing import List, Optional, Type, Union

from numpy.random import RandomState
from pandas import Series
from shap import Explainer


class BaseSelector(metaclass=ABCMeta):
    """Abstract base class for all selectors in shapicant.

    Args:
        estimator: A supervised learning estimator with a 'fit' method.
        explainer_type: A SHAP explainer type.
        n_iter: The number of iterations to perform.
        verbose: Controls verbosity of output.
        random_state: Parameter to control the random number generator used.

    Attributes:
        p_values_ (Series): Series containing the empirical p-values ​​of the features.

    """

    def __init__(
        self,
        estimator: object,
        explainer_type: Type[Explainer],
        n_iter: int = 100,
        verbose: Union[int, bool] = 1,
        random_state: Optional[Union[int, RandomState]] = None,
    ) -> None:
        self.estimator = estimator
        self.explainer_type = explainer_type
        self.n_iter = n_iter
        self.verbose = verbose
        self.random_state = random_state
        self.p_values_ = None
        self._current_iter = None
        self._n_outputs = None

    @abstractmethod
    def fit(self, *args, **kwargs):
        """
        Abstract 'fit' method.

        """

    @abstractmethod
    def transform(self, *args, **kwargs):
        """
        Abstract 'transform' method.

        """

    @abstractmethod
    def fit_transform(self, *args, **kwargs):
        """
        Abstract 'fit_transform' method.

        """

    def _check_is_fitted(self):
        if self.p_values_ is None:
            raise AttributeError(
                "This instance is not fitted yet. Call 'fit' with appropriate arguments before using this method."
            )

    def _validate_params(self):
        if self.n_iter < 10:
            raise ValueError("n_iter must be greater than or equal to 10.")

    def _compute_p_values(
        self,
        true_pos_shap_values: List[Series],
        null_pos_shap_values: List[Series],
        true_neg_shap_values: List[Series],
        null_neg_shap_values: List[Series],
    ) -> Series:
        pos_results = [None] * self._n_outputs
        neg_results = [None] * self._n_outputs
        results = [None] * self._n_outputs
        for i in range(self._n_outputs):
            pos_results[i] = null_pos_shap_values[i].ge(true_pos_shap_values[i], axis=0)
            neg_results[i] = null_neg_shap_values[i].le(true_neg_shap_values[i], axis=0)
            results[i] = pos_results[i] | neg_results[i]
        results = reduce(lambda df_0, df_1: df_0 & df_1, results).sum(axis=1)
        p_values = (results + 1) / (self.n_iter + 1)
        return p_values
