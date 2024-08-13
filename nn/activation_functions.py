import dataclasses
from typing import Callable

import numpy as np


def sigmoid(inputs: np.ndarray) -> np.ndarray:
    return 1 / (1 + np.exp(-inputs))


def relu(inputs: np.ndarray) -> np.ndarray:
    return np.maximum(0, inputs)


@dataclasses.dataclass
class ActivationFunction:
    SIGMOID: Callable = sigmoid
    RELU: Callable = relu
