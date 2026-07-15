"""A from-scratch NumPy implementation of an MLP digit classifier.

Public API:

* :class:`~digit_recognizer.mlp.MLP` -- the multi-layer perceptron model.
* :class:`~digit_recognizer.preprocess.Preprocessor` -- data cleaning helpers.
"""

from digit_recognizer.mlp import MLP
from digit_recognizer.preprocess import Preprocessor

__all__ = ["MLP", "Preprocessor"]
__version__ = "1.0.0"
