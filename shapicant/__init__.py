"""
The shapicant module implements a feature selection algorithm based on SHAP and target permutation.

"""
from ._base import BaseSelector
from ._pandas_selector import PandasSelector
from ._spark_selector import SparkSelector

__version__ = "0.2.0"

__all__ = ["BaseSelector", "PandasSelector", "SparkSelector"]
